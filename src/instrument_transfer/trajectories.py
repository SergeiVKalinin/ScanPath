import numpy as np


# ============================================================
# Utilities
# ============================================================

def resample_polyline(points, n_points):
    """
    Resample a piecewise-linear trajectory to approximately uniform arc length.

    Parameters
    ----------
    points : ndarray, shape (M, 2)
        Polyline vertices.

    n_points : int
        Number of output samples.

    Returns
    -------
    path : ndarray, shape (n_points, 2)
        Resampled path.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M, 2).")

    if len(points) < 2:
        raise ValueError("Need at least two points to resample a polyline.")

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length == 0:
        return np.repeat(points[:1], n_points, axis=0)

    target = np.linspace(0.0, total_length, int(n_points))
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])

    return np.column_stack([x, y])


def _as_int(x, minimum=1):
    return max(int(x), minimum)


def _normalize_to_box(points, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    points = np.asarray(points, dtype=float)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    span = pmax - pmin
    span[span == 0.0] = 1.0
    q = (points - pmin) / span
    out = np.empty_like(q)
    out[:, 0] = x0 + (x1 - x0) * q[:, 0]
    out[:, 1] = y0 + (y1 - y0) * q[:, 1]
    return out


def _clip_to_box(path, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    out = np.asarray(path, dtype=float).copy()
    out[:, 0] = np.clip(out[:, 0], min(x0, x1), max(x0, x1))
    out[:, 1] = np.clip(out[:, 1], min(y0, y1), max(y0, y1))
    return out


def _path_from_segments(points, n_points, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    path = resample_polyline(points, n_points)
    return _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)


def _path_tangent_normal(path):
    path = np.asarray(path, dtype=float)
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    speed = np.sqrt(dx**2 + dy**2) + 1e-12
    tx = dx / speed
    ty = dy / speed
    nx = -ty
    ny = tx
    return tx, ty, nx, ny


def _apply_moving_frame_modulation(
    backbone,
    radius=0.02,
    n_turns=40,
    radial_cycles=6,
    mode="circle",
    radial_strength=0.6,
    tangent_weight=1.0,
    normal_weight=1.0,
    clip_box=(0.0, 1.0, 0.0, 1.0),
):
    """
    Attach a Class-0 local ornament in the moving frame of a backbone path.

    Parameters
    ----------
    backbone : ndarray, shape (N, 2)
        Carrier trajectory.
    radius : float
        Base modulation radius in normalized coordinates.
    n_turns : int
        Number of local wobble turns over the whole backbone.
    radial_cycles : int
        Slow radial breathing cycles.
    mode : str
        One of: 'circle', 'spiral', 'ellipse', 'lissajous'.
    radial_strength : float
        Strength of slow in/out modulation.
    tangent_weight, normal_weight : float
        Anisotropy of local ornament.
    clip_box : tuple
        (x0, x1, y0, y1) clipping box.
    """
    backbone = np.asarray(backbone, dtype=float)
    n = len(backbone)
    t = np.linspace(0.0, 1.0, n)
    tx, ty, nx, ny = _path_tangent_normal(backbone)

    phi = 2.0 * np.pi * float(n_turns) * t
    breath = 1.0 + float(radial_strength) * np.sin(2.0 * np.pi * float(radial_cycles) * t)
    rr = float(radius) * np.clip(breath, 0.05, None)

    mode = str(mode).lower()
    if mode in ["circle", "circular"]:
        local_t = tangent_weight * rr * np.sin(phi)
        local_n = normal_weight * rr * np.cos(phi)
    elif mode in ["spiral", "inout_spiral", "wobble_spiral"]:
        local_t = tangent_weight * rr * np.sin(phi)
        local_n = normal_weight * rr * np.cos(phi)
    elif mode in ["ellipse", "elliptic"]:
        local_t = 0.6 * tangent_weight * rr * np.sin(phi)
        local_n = 1.2 * normal_weight * rr * np.cos(phi)
    elif mode in ["lissajous", "local_lissajous"]:
        local_t = tangent_weight * rr * np.sin(2.0 * phi)
        local_n = normal_weight * rr * np.sin(3.0 * phi + np.pi / 2.0)
    else:
        raise ValueError("Unknown modulation mode.")

    out = np.empty_like(backbone)
    out[:, 0] = backbone[:, 0] + local_t * tx + local_n * nx
    out[:, 1] = backbone[:, 1] + local_t * ty + local_n * ny
    return _clip_to_box(out, *clip_box)


# ============================================================
# Class 0: atomic single-chart / non-recursive families
# ============================================================

def make_rectangular_raster(
    x0=0.0,
    x1=1.0,
    y0=0.0,
    y1=1.0,
    n_lines=100,
    points_per_line=120,
    flyback_points=20,
    bidirectional=False,
    include_flyback=True,
):
    """Generate a rectangular raster trajectory."""
    path_parts = []
    line_ids = []

    y_values = np.linspace(y0, y1, _as_int(n_lines))

    for line_index, y in enumerate(y_values):
        if bidirectional and line_index % 2 == 1:
            x_line = np.linspace(x1, x0, _as_int(points_per_line))
        else:
            x_line = np.linspace(x0, x1, _as_int(points_per_line))

        y_line = np.full_like(x_line, y)
        line = np.column_stack([x_line, y_line])

        path_parts.append(line)
        line_ids.extend([line_index] * len(line))

        if include_flyback and line_index < len(y_values) - 1:
            y_next = y_values[line_index + 1]
            if bidirectional:
                x_fb = np.full(_as_int(flyback_points), x_line[-1])
                y_fb = np.linspace(y, y_next, _as_int(flyback_points))
            else:
                x_fb = np.linspace(x1, x0, _as_int(flyback_points))
                y_fb = np.linspace(y, y_next, _as_int(flyback_points))
            flyback = np.column_stack([x_fb, y_fb])
            path_parts.append(flyback)
            line_ids.extend([line_index] * len(flyback))

    path = np.vstack(path_parts)
    return path, np.asarray(line_ids, dtype=int)


def make_sinusoidal_raster(
    x0=0.0,
    x1=1.0,
    y0=0.0,
    y1=1.0,
    n_lines=80,
    points_per_line=180,
    phase=0.0,
):
    """Smooth raster using sinusoidal x-motion and monotone y-progression."""
    n_lines = _as_int(n_lines)
    points_per_line = _as_int(points_per_line)
    total = n_lines * points_per_line
    t = np.linspace(0.0, 1.0, total)
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    ax = 0.5 * abs(x1 - x0)
    ay = 0.5 * abs(y1 - y0)
    x = x_mid + ax * np.sin(2.0 * np.pi * n_lines * t + phase)
    y = y0 + (y1 - y0) * t
    path = np.column_stack([x, y])
    line_ids = np.floor(t * n_lines).astype(int)
    line_ids = np.clip(line_ids, 0, n_lines - 1)
    return path, line_ids


def make_triangle_raster(
    x0=0.0,
    x1=1.0,
    y0=0.0,
    y1=1.0,
    n_lines=80,
    points_per_line=180,
):
    """Raster-like path with triangle-wave x-motion and smooth y progression."""
    n_lines = _as_int(n_lines)
    points_per_line = _as_int(points_per_line)
    total = n_lines * points_per_line
    t = np.linspace(0.0, 1.0, total)
    phase = (n_lines * t) % 1.0
    tri = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    x_mid = 0.5 * (x0 + x1)
    y = y0 + (y1 - y0) * t
    x = x_mid + 0.5 * (x1 - x0) * tri
    path = np.column_stack([x, y])
    line_ids = np.floor(t * n_lines).astype(int)
    line_ids = np.clip(line_ids, 0, n_lines - 1)
    return path, line_ids


def make_spiral(
    center=(0.5, 0.5),
    radius=0.48,
    n_turns=25,
    n_points=12000,
    inward=False,
    phase=0.0,
    aspect=(1.0, 1.0),
):
    """Generate a smooth Archimedean spiral trajectory."""
    t = np.linspace(0.0, 1.0, _as_int(n_points))
    r = radius * (1.0 - t) if inward else radius * t
    theta = 2.0 * np.pi * float(n_turns) * t + phase
    x = center[0] + aspect[0] * r * np.cos(theta)
    y = center[1] + aspect[1] * r * np.sin(theta)
    path = np.column_stack([x, y])
    line_ids = np.floor((theta - phase) / (2.0 * np.pi)).astype(int)
    line_ids = np.clip(line_ids, 0, _as_int(n_turns) - 1)
    return path, line_ids


def make_log_spiral(
    center=(0.5, 0.5),
    radius=0.48,
    growth=3.0,
    n_turns=18,
    n_points=12000,
    inward=False,
):
    """Generate a logarithmic spiral trajectory."""
    t = np.linspace(0.0, 1.0, _as_int(n_points))
    s = 1.0 - t if inward else t
    r = radius * (np.exp(growth * s) - 1.0) / (np.exp(growth) - 1.0 + 1e-12)
    theta = 2.0 * np.pi * float(n_turns) * t
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    path = np.column_stack([x, y])
    line_ids = np.floor(float(n_turns) * t).astype(int)
    line_ids = np.clip(line_ids, 0, _as_int(n_turns) - 1)
    return path, line_ids


def make_fermat_spiral(
    center=(0.5, 0.5),
    radius=0.48,
    n_turns=20,
    n_points=12000,
    phase=0.0,
):
    """Generate a Fermat-like spiral trajectory."""
    t = np.linspace(0.0, 1.0, _as_int(n_points))
    r = radius * np.sqrt(t)
    theta = 2.0 * np.pi * float(n_turns) * t + phase
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    path = np.column_stack([x, y])
    line_ids = np.floor(float(n_turns) * t).astype(int)
    line_ids = np.clip(line_ids, 0, _as_int(n_turns) - 1)
    return path, line_ids


def make_rectangular_spiral(
    center=(0.5, 0.5),
    width=0.96,
    height=0.96,
    n_segments=160,
    n_points=12000,
    outward=True,
):
    """Generate a rectangular/square spiral trajectory made from axis-aligned segments."""
    n_segments = _as_int(n_segments)
    n_points = _as_int(n_points)

    vertices = [np.array([0.0, 0.0])]
    pos = np.array([0.0, 0.0])
    directions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]
    direction_index = 0

    for segment_count in range(n_segments):
        current_length = np.ceil((segment_count + 1) / 2.0)
        pos = pos + directions[direction_index] * current_length
        vertices.append(pos.copy())
        direction_index = (direction_index + 1) % 4

    path = _path_from_segments(np.asarray(vertices), n_points)
    path[:, 0] = center[0] - width / 2.0 + width * path[:, 0]
    path[:, 1] = center[1] - height / 2.0 + height * path[:, 1]

    if not outward:
        path = path[::-1]

    line_ids = np.linspace(0, n_segments - 1, n_points).astype(int)
    return path, line_ids


def make_lissajous(
    center=(0.5, 0.5),
    amplitude=(0.48, 0.48),
    ax=23,
    ay=31,
    phase=np.pi / 2,
    n_periods=8,
    n_points=20000,
):
    """Generate a Lissajous trajectory."""
    t = np.linspace(0.0, 2.0 * np.pi * float(n_periods), _as_int(n_points))
    x = center[0] + amplitude[0] * np.sin(float(ax) * t + phase)
    y = center[1] + amplitude[1] * np.sin(float(ay) * t)
    path = np.column_stack([x, y])
    line_ids = np.floor(t / (2.0 * np.pi)).astype(int)
    line_ids = np.clip(line_ids, 0, _as_int(n_periods) - 1)
    return path, line_ids


def make_fourier_scan(
    center=(0.5, 0.5),
    amplitude=(0.48, 0.48),
    coeffs_x=((1.0, 3.0, 0.0), (0.35, 7.0, 0.3)),
    coeffs_y=((1.0, 4.0, 1.1), (0.25, 9.0, -0.4)),
    n_periods=6,
    n_points=18000,
):
    """Low-order Fourier trajectory: still Class 0 when the number of modes is small."""
    t = np.linspace(0.0, 2.0 * np.pi * float(n_periods), _as_int(n_points))
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    for amp, freq, ph in coeffs_x:
        x += float(amp) * np.sin(float(freq) * t + float(ph))
    for amp, freq, ph in coeffs_y:
        y += float(amp) * np.sin(float(freq) * t + float(ph))
    x = center[0] + amplitude[0] * x / (np.max(np.abs(x)) + 1e-12)
    y = center[1] + amplitude[1] * y / (np.max(np.abs(y)) + 1e-12)
    path = np.column_stack([x, y])
    line_ids = np.floor(t / (2.0 * np.pi)).astype(int)
    line_ids = np.clip(line_ids, 0, _as_int(n_periods) - 1)
    return path, line_ids


def make_rose_curve(
    center=(0.5, 0.5),
    radius=0.48,
    k=5,
    n_turns=4,
    n_points=14000,
):
    """Rose-curve scan in polar form."""
    theta = np.linspace(0.0, 2.0 * np.pi * float(n_turns), _as_int(n_points))
    r = radius * np.cos(float(k) * theta)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    path = np.column_stack([x, y])
    line_ids = np.floor((theta - theta.min()) / (2.0 * np.pi)).astype(int)
    line_ids = np.clip(line_ids, 0, _as_int(n_turns) - 1)
    return path, line_ids


def make_epitrochoid(
    center=(0.5, 0.5),
    a=5.0,
    b=3.0,
    h=2.0,
    radius=0.45,
    n_periods=6,
    n_points=18000,
):
    """Epitrochoid / spirograph-like single-chart path."""
    t = np.linspace(0.0, 2.0 * np.pi * float(n_periods), _as_int(n_points))
    x = (a + b) * np.cos(t) - h * np.cos((a + b) * t / b)
    y = (a + b) * np.sin(t) - h * np.sin((a + b) * t / b)
    path = _normalize_to_box(np.column_stack([x, y]), center[0] - radius, center[0] + radius, center[1] - radius, center[1] + radius)
    line_ids = np.floor((t - t.min()) / (2.0 * np.pi)).astype(int)
    line_ids = np.clip(line_ids, 0, _as_int(n_periods) - 1)
    return path, line_ids


def make_hypotrochoid(
    center=(0.5, 0.5),
    a=5.0,
    b=3.0,
    h=5.0,
    radius=0.45,
    n_periods=8,
    n_points=18000,
):
    """Hypotrochoid / spirograph-like single-chart path."""
    t = np.linspace(0.0, 2.0 * np.pi * float(n_periods), _as_int(n_points))
    x = (a - b) * np.cos(t) + h * np.cos((a - b) * t / b)
    y = (a - b) * np.sin(t) - h * np.sin((a - b) * t / b)
    path = _normalize_to_box(np.column_stack([x, y]), center[0] - radius, center[0] + radius, center[1] - radius, center[1] + radius)
    line_ids = np.floor((t - t.min()) / (2.0 * np.pi)).astype(int)
    line_ids = np.clip(line_ids, 0, _as_int(n_periods) - 1)
    return path, line_ids


# ============================================================
# Class 0+0: non-recursive composite analytic families
# ============================================================

def make_raster_circular_wobble(
    x0=0.0,
    x1=1.0,
    y0=0.0,
    y1=1.0,
    n_lines=40,
    points_per_line=200,
    bidirectional=True,
    radius=0.02,
    n_turns=50,
):
    backbone, line_ids = make_rectangular_raster(
        x0=x0, x1=x1, y0=y0, y1=y1,
        n_lines=n_lines,
        points_per_line=points_per_line,
        flyback_points=max(8, int(points_per_line * 0.12)),
        bidirectional=bidirectional,
        include_flyback=True,
    )
    path = _apply_moving_frame_modulation(backbone, radius=radius, n_turns=n_turns, mode="circle", clip_box=(x0, x1, y0, y1))
    return path, line_ids


def make_raster_spiral_wobble(
    x0=0.0,
    x1=1.0,
    y0=0.0,
    y1=1.0,
    n_lines=40,
    points_per_line=200,
    bidirectional=True,
    radius=0.02,
    n_turns=55,
    radial_cycles=7,
):
    backbone, line_ids = make_rectangular_raster(
        x0=x0, x1=x1, y0=y0, y1=y1,
        n_lines=n_lines,
        points_per_line=points_per_line,
        flyback_points=max(8, int(points_per_line * 0.12)),
        bidirectional=bidirectional,
        include_flyback=True,
    )
    path = _apply_moving_frame_modulation(
        backbone,
        radius=radius,
        n_turns=n_turns,
        radial_cycles=radial_cycles,
        mode="spiral",
        clip_box=(x0, x1, y0, y1),
    )
    return path, line_ids


def make_raster_lissajous_wobble(
    x0=0.0,
    x1=1.0,
    y0=0.0,
    y1=1.0,
    n_lines=40,
    points_per_line=200,
    bidirectional=True,
    radius=0.02,
    n_turns=45,
):
    backbone, line_ids = make_rectangular_raster(
        x0=x0, x1=x1, y0=y0, y1=y1,
        n_lines=n_lines,
        points_per_line=points_per_line,
        flyback_points=max(8, int(points_per_line * 0.12)),
        bidirectional=bidirectional,
        include_flyback=True,
    )
    path = _apply_moving_frame_modulation(backbone, radius=radius, n_turns=n_turns, mode="lissajous", clip_box=(x0, x1, y0, y1))
    return path, line_ids


def make_spiral_circular_wobble(
    center=(0.5, 0.5),
    radius=0.46,
    n_turns=24,
    n_points=16000,
    wobble_radius=0.015,
    wobble_turns=75,
):
    backbone, line_ids = make_spiral(center=center, radius=radius, n_turns=n_turns, n_points=n_points)
    path = _apply_moving_frame_modulation(backbone, radius=wobble_radius, n_turns=wobble_turns, mode="circle", clip_box=(0.0, 1.0, 0.0, 1.0))
    return path, line_ids


def make_lissajous_circular_wobble(
    center=(0.5, 0.5),
    amplitude=(0.46, 0.46),
    ax=17,
    ay=23,
    phase=np.pi/2,
    n_periods=8,
    n_points=18000,
    wobble_radius=0.012,
    wobble_turns=70,
):
    backbone, line_ids = make_lissajous(center=center, amplitude=amplitude, ax=ax, ay=ay, phase=phase, n_periods=n_periods, n_points=n_points)
    path = _apply_moving_frame_modulation(backbone, radius=wobble_radius, n_turns=wobble_turns, mode="circle", clip_box=(0.0, 1.0, 0.0, 1.0))
    return path, line_ids


def make_spiral_radial_wobble(
    center=(0.5, 0.5),
    radius=0.46,
    n_turns=18,
    n_points=16000,
    wobble_amplitude=0.08,
    wobble_frequency=9.0,
):
    """Class 0+0: polar backbone + radial modulation in the same chart."""
    t = np.linspace(0.0, 1.0, _as_int(n_points))
    base_r = radius * t
    theta = 2.0 * np.pi * float(n_turns) * t
    mod = 1.0 + wobble_amplitude * np.sin(2.0 * np.pi * wobble_frequency * t)
    r = base_r * mod
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    path = np.column_stack([x, y])
    line_ids = np.floor(float(n_turns) * t).astype(int)
    line_ids = np.clip(line_ids, 0, _as_int(n_turns) - 1)
    return path, line_ids


# ============================================================
# Recursive helpers for Class 1 and 2
# ============================================================

def _transform_unit_path(path, transform="I"):
    path = np.asarray(path, dtype=float)
    x = path[:, 0]
    y = path[:, 1]
    if transform == "I":
        xx, yy = x, y
    elif transform == "R90":
        xx, yy = 1.0 - y, x
    elif transform == "R180":
        xx, yy = 1.0 - x, 1.0 - y
    elif transform == "R270":
        xx, yy = y, 1.0 - x
    elif transform == "FX":
        xx, yy = 1.0 - x, y
    elif transform == "FY":
        xx, yy = x, 1.0 - y
    elif transform == "FD":  # diagonal y=x
        xx, yy = y, x
    elif transform == "FA":  # anti-diagonal
        xx, yy = 1.0 - y, 1.0 - x
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return np.column_stack([xx, yy])


def _cell_bounds(index, nx, ny):
    col = int(index) % int(nx)
    row = int(index) // int(nx)
    x0 = col / float(nx)
    x1 = (col + 1) / float(nx)
    y0 = row / float(ny)
    y1 = (row + 1) / float(ny)
    return x0, x1, y0, y1


def _recursive_uniform_path(levels, nx, ny, order, transform="I", reverse=False):
    """
    Generic recursive path built by placing the same local rule into each child cell.
    This is intentionally simple: good for toy recursive families.
    """
    if levels <= 0:
        return np.array([[0.5, 0.5]])

    base = _recursive_uniform_path(levels - 1, nx, ny, order, transform=transform, reverse=False)
    base = _transform_unit_path(base, transform)
    if reverse:
        base = base[::-1]

    parts = []
    for child in order:
        x0, x1, y0, y1 = _cell_bounds(child, nx, ny)
        part = np.empty_like(base)
        part[:, 0] = x0 + (x1 - x0) * base[:, 0]
        part[:, 1] = y0 + (y1 - y0) * base[:, 1]
        parts.append(part)
    return np.vstack(parts)


def _recursive_serpentine_path(levels, nx, ny):
    """Recursive snake / boustrophedon traversal over an nx-by-ny subdivision."""
    order = []
    for row in range(ny):
        cols = list(range(nx)) if row % 2 == 0 else list(range(nx - 1, -1, -1))
        for col in cols:
            order.append(row * nx + col)
    return _recursive_uniform_path(levels, nx, ny, order)


def _recursive_spiral_path(levels, nx, ny):
    """Recursive spiral ordering over child cells."""
    grid = np.arange(nx * ny).reshape(ny, nx)
    top, bottom = ny - 1, 0
    left, right = 0, nx - 1
    order = []
    while left <= right and bottom <= top:
        for c in range(left, right + 1):
            order.append(int(grid[bottom, c]))
        bottom += 1
        for r in range(bottom, top + 1):
            order.append(int(grid[r, right]))
        right -= 1
        if bottom <= top:
            for c in range(right, left - 1, -1):
                order.append(int(grid[top, c]))
            top -= 1
        if left <= right:
            for r in range(top, bottom - 1, -1):
                order.append(int(grid[r, left]))
            left += 1
    return _recursive_uniform_path(levels, nx, ny, order)


def _recursive_diagonal_path(levels, nx, ny):
    """Recursive diagonal / anti-diagonal child ordering."""
    order = []
    for s in range(nx + ny - 1):
        band = []
        for row in range(ny):
            col = s - row
            if 0 <= col < nx:
                band.append(row * nx + col)
        if s % 2 == 1:
            band = band[::-1]
        order.extend(band)
    return _recursive_uniform_path(levels, nx, ny, order)


def _recursive_strip_path(levels, arity=2, orientation="horizontal"):
    """
    Recursive strip family. 1x2 / 2x1 are Class 1, 1x3 / 3x1 are Class 2 here.
    """
    levels = _as_int(levels)

    def recurse(level, reverse=False):
        if level <= 0:
            return np.array([[0.5, 0.5]])
        base = recurse(level - 1, reverse=False)
        parts = []
        indices = list(range(arity)) if not reverse else list(range(arity - 1, -1, -1))
        for i, idx in enumerate(indices):
            part = base.copy()
            if orientation == "horizontal":
                part[:, 1] = (idx + part[:, 1]) / float(arity)
            else:
                part[:, 0] = (idx + part[:, 0]) / float(arity)
            if i % 2 == 1:
                part = part[::-1]
            parts.append(part)
        return np.vstack(parts)

    return recurse(levels)


def _interleave_bits(x):
    x = int(x)
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x


def _morton_indices(order):
    n = 2 ** _as_int(order)
    coords = []
    for y in range(n):
        for x in range(n):
            code = _interleave_bits(x) | (_interleave_bits(y) << 1)
            coords.append((code, x, y))
    coords.sort(key=lambda z: z[0])
    pts = np.array([(x, y) for _, x, y in coords], dtype=float)
    pts[:, 0] += 0.5
    pts[:, 1] += 0.5
    pts /= float(n)
    return pts


def _rot_hilbert(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _hilbert_d2xy(n, d):
    x = y = 0
    t = int(d)
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot_hilbert(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def _hilbert_points(order):
    n = 2 ** _as_int(order)
    pts = [_hilbert_d2xy(n, d) for d in range(n * n)]
    pts = np.asarray(pts, dtype=float)
    pts[:, 0] = (pts[:, 0] + 0.5) / float(n)
    pts[:, 1] = (pts[:, 1] + 0.5) / float(n)
    return pts


# ============================================================
# Class 1: simple recursive families
# ============================================================

def make_hilbert_curve(order=5, n_points=16000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _hilbert_points(order)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, 2 ** (2 * _as_int(order)) - 1, len(path))).astype(int)
    return path, line_ids


def make_morton_curve(order=5, n_points=16000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _morton_indices(order)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, 2 ** (2 * _as_int(order)) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_2x2_snake(order=5, n_points=16000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _recursive_serpentine_path(order, 2, 2)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_2x2_spiral(order=5, n_points=16000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _recursive_spiral_path(order, 2, 2)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_2x2_diagonal(order=5, n_points=16000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _recursive_diagonal_path(order, 2, 2)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_strip_1x2(order=8, n_points=14000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _recursive_strip_path(order, arity=2, orientation="horizontal")
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_strip_2x1(order=8, n_points=14000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _recursive_strip_path(order, arity=2, orientation="vertical")
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


# ============================================================
# Class 1+0: recursive backbone + local Class-0 ornament
# ============================================================

def make_hilbert_circular_wobble(order=5, n_points=18000, wobble_radius=0.015, wobble_turns=70):
    backbone, line_ids = make_hilbert_curve(order=order, n_points=n_points)
    path = _apply_moving_frame_modulation(backbone, radius=wobble_radius, n_turns=wobble_turns, mode="circle")
    return path, line_ids


def make_hilbert_spiral_wobble(order=5, n_points=18000, wobble_radius=0.015, wobble_turns=75, radial_cycles=8):
    backbone, line_ids = make_hilbert_curve(order=order, n_points=n_points)
    path = _apply_moving_frame_modulation(backbone, radius=wobble_radius, n_turns=wobble_turns, radial_cycles=radial_cycles, mode="spiral")
    return path, line_ids


def make_morton_circular_wobble(order=5, n_points=18000, wobble_radius=0.015, wobble_turns=70):
    backbone, line_ids = make_morton_curve(order=order, n_points=n_points)
    path = _apply_moving_frame_modulation(backbone, radius=wobble_radius, n_turns=wobble_turns, mode="circle")
    return path, line_ids


def make_recursive_2x2_snake_wobble(order=5, n_points=18000, wobble_radius=0.015, wobble_turns=70):
    backbone, line_ids = make_recursive_2x2_snake(order=order, n_points=n_points)
    path = _apply_moving_frame_modulation(backbone, radius=wobble_radius, n_turns=wobble_turns, mode="circle")
    return path, line_ids


def make_recursive_strip_1x2_wobble(order=8, n_points=18000, wobble_radius=0.015, wobble_turns=65):
    backbone, line_ids = make_recursive_strip_1x2(order=order, n_points=n_points)
    path = _apply_moving_frame_modulation(backbone, radius=wobble_radius, n_turns=wobble_turns, mode="lissajous")
    return path, line_ids


# ============================================================
# Class 2: richer recursive families
# ============================================================

def make_peano_curve(order=4, n_points=18000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    """Peano-like recursive snake over a 3x3 subdivision."""
    pts = _recursive_serpentine_path(order, 3, 3)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_3x3_spiral(order=4, n_points=18000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _recursive_spiral_path(order, 3, 3)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_3x3_diagonal(order=4, n_points=18000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _recursive_diagonal_path(order, 3, 3)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_strip_1x3(order=6, n_points=18000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _recursive_strip_path(order, arity=3, orientation="horizontal")
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_strip_3x1(order=6, n_points=18000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    pts = _recursive_strip_path(order, arity=3, orientation="vertical")
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_3x3_checker(order=4, n_points=18000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    base_order = [0, 2, 1, 5, 3, 4, 8, 6, 7]
    pts = _recursive_uniform_path(order, 3, 3, base_order)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


def make_recursive_3x3_coil(order=4, n_points=18000, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    base_order = [0, 1, 2, 5, 8, 7, 6, 3, 4]
    pts = _recursive_uniform_path(order, 3, 3, base_order)
    path = resample_polyline(pts, _as_int(n_points))
    path = _normalize_to_box(path, x0=x0, x1=x1, y0=y0, y1=y1)
    line_ids = np.floor(np.linspace(0, len(pts) - 1, len(path))).astype(int)
    return path, line_ids


# ============================================================
# Catalog and unified entry point
# ============================================================

def get_scan_path_catalog():
    """
    Return metadata for available trajectory families.

    The classes used here are the nonadaptive classes discussed in chat:
    - Class 0   : atomic single-chart families
    - Class 0+0 : composite non-recursive families
    - Class 1   : simple recursive families
    - Class 1+0 : recursive backbone with local Class-0 ornament
    - Class 2   : richer recursive families
    """
    return {
        # Class 0
        "raster": {"class": "0", "generator": make_rectangular_raster},
        "rectangular": {"class": "0", "generator": make_rectangular_raster},
        "rectangular_raster": {"class": "0", "generator": make_rectangular_raster},
        "sinusoidal_raster": {"class": "0", "generator": make_sinusoidal_raster},
        "triangle_raster": {"class": "0", "generator": make_triangle_raster},
        "spiral": {"class": "0", "generator": make_spiral},
        "archimedean_spiral": {"class": "0", "generator": make_spiral},
        "smooth_spiral": {"class": "0", "generator": make_spiral},
        "log_spiral": {"class": "0", "generator": make_log_spiral},
        "fermat_spiral": {"class": "0", "generator": make_fermat_spiral},
        "rectangular_spiral": {"class": "0", "generator": make_rectangular_spiral},
        "square_spiral": {"class": "0", "generator": make_rectangular_spiral},
        "lissajous": {"class": "0", "generator": make_lissajous},
        "lisajous": {"class": "0", "generator": make_lissajous},
        "fourier": {"class": "0", "generator": make_fourier_scan},
        "fourier_scan": {"class": "0", "generator": make_fourier_scan},
        "rose": {"class": "0", "generator": make_rose_curve},
        "rose_curve": {"class": "0", "generator": make_rose_curve},
        "epitrochoid": {"class": "0", "generator": make_epitrochoid},
        "hypotrochoid": {"class": "0", "generator": make_hypotrochoid},

        # Class 0+0
        "raster_circular_wobble": {"class": "0+0", "generator": make_raster_circular_wobble},
        "raster_spiral_wobble": {"class": "0+0", "generator": make_raster_spiral_wobble},
        "raster_lissajous_wobble": {"class": "0+0", "generator": make_raster_lissajous_wobble},
        "spiral_circular_wobble": {"class": "0+0", "generator": make_spiral_circular_wobble},
        "lissajous_circular_wobble": {"class": "0+0", "generator": make_lissajous_circular_wobble},
        "spiral_radial_wobble": {"class": "0+0", "generator": make_spiral_radial_wobble},

        # Class 1
        "hilbert": {"class": "1", "generator": make_hilbert_curve},
        "hilbert_curve": {"class": "1", "generator": make_hilbert_curve},
        "morton": {"class": "1", "generator": make_morton_curve},
        "morton_curve": {"class": "1", "generator": make_morton_curve},
        "recursive_2x2_snake": {"class": "1", "generator": make_recursive_2x2_snake},
        "recursive_2x2_spiral": {"class": "1", "generator": make_recursive_2x2_spiral},
        "recursive_2x2_diagonal": {"class": "1", "generator": make_recursive_2x2_diagonal},
        "recursive_strip_1x2": {"class": "1", "generator": make_recursive_strip_1x2},
        "recursive_strip_2x1": {"class": "1", "generator": make_recursive_strip_2x1},

        # Class 1+0
        "hilbert_circular_wobble": {"class": "1+0", "generator": make_hilbert_circular_wobble},
        "hilbert_spiral_wobble": {"class": "1+0", "generator": make_hilbert_spiral_wobble},
        "morton_circular_wobble": {"class": "1+0", "generator": make_morton_circular_wobble},
        "recursive_2x2_snake_wobble": {"class": "1+0", "generator": make_recursive_2x2_snake_wobble},
        "recursive_strip_1x2_wobble": {"class": "1+0", "generator": make_recursive_strip_1x2_wobble},

        # Class 2
        "peano": {"class": "2", "generator": make_peano_curve},
        "peano_curve": {"class": "2", "generator": make_peano_curve},
        "recursive_3x3_spiral": {"class": "2", "generator": make_recursive_3x3_spiral},
        "recursive_3x3_diagonal": {"class": "2", "generator": make_recursive_3x3_diagonal},
        "recursive_strip_1x3": {"class": "2", "generator": make_recursive_strip_1x3},
        "recursive_strip_3x1": {"class": "2", "generator": make_recursive_strip_3x1},
        "recursive_3x3_checker": {"class": "2", "generator": make_recursive_3x3_checker},
        "recursive_3x3_coil": {"class": "2", "generator": make_recursive_3x3_coil},
    }


def list_scan_paths(include_aliases=False):
    """Return available path names."""
    catalog = get_scan_path_catalog()
    if include_aliases:
        return sorted(catalog.keys())

    preferred = []
    seen_generators = set()
    for name, meta in catalog.items():
        key = id(meta["generator"])
        if key not in seen_generators:
            preferred.append(name)
            seen_generators.add(key)
    return sorted(preferred)


def generate_scan_path(path_type="raster", **kwargs):
    """
    Unified trajectory generator.

    Parameters
    ----------
    path_type : str
        Any key returned by list_scan_paths(include_aliases=True).

    Returns
    -------
    path : ndarray, shape (N, 2)
    line_ids : ndarray, shape (N,)
    """
    key = str(path_type).lower()
    catalog = get_scan_path_catalog()
    if key not in catalog:
        raise ValueError(
            f"Unknown path_type '{path_type}'. Available options: {', '.join(list_scan_paths())}"
        )
    return catalog[key]["generator"](**kwargs)
