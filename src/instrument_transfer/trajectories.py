import numpy as np


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

    target = np.linspace(0.0, total_length, n_points)
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])

    return np.column_stack([x, y])


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
    """
    Generate a rectangular raster trajectory.

    Parameters
    ----------
    x0, x1, y0, y1 : float
        Scan bounds.

    n_lines : int
        Number of raster lines.

    points_per_line : int
        Number of samples on each imaging line.

    flyback_points : int
        Number of samples used for flyback between lines.

    bidirectional : bool
        If False, every line scans left-to-right and then flies back.
        If True, alternate lines scan in opposite directions.

    include_flyback : bool
        If True, include explicit flyback segments.

    Returns
    -------
    path : ndarray, shape (N, 2)
        Ideal trajectory.

    line_ids : ndarray, shape (N,)
        Line index for each point.
    """
    path_parts = []
    line_ids = []

    y_values = np.linspace(y0, y1, int(n_lines))

    for line_index, y in enumerate(y_values):
        if bidirectional and line_index % 2 == 1:
            x_line = np.linspace(x1, x0, int(points_per_line))
        else:
            x_line = np.linspace(x0, x1, int(points_per_line))

        y_line = np.full_like(x_line, y)
        line = np.column_stack([x_line, y_line])

        path_parts.append(line)
        line_ids.extend([line_index] * len(line))

        if include_flyback and line_index < len(y_values) - 1:
            y_next = y_values[line_index + 1]

            if bidirectional:
                x_fb = np.linspace(x_line[-1], x_line[-1], int(flyback_points))
                y_fb = np.linspace(y, y_next, int(flyback_points))
            else:
                x_fb = np.linspace(x1, x0, int(flyback_points))
                y_fb = np.linspace(y, y_next, int(flyback_points))

            flyback = np.column_stack([x_fb, y_fb])
            path_parts.append(flyback)
            line_ids.extend([line_index] * len(flyback))

    path = np.vstack(path_parts)
    line_ids = np.asarray(line_ids, dtype=int)

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
    """
    Generate a smooth Archimedean spiral trajectory.

    Returns
    -------
    path : ndarray, shape (N, 2)

    line_ids : ndarray, shape (N,)
        Turn index for each point.
    """
    t = np.linspace(0.0, 1.0, int(n_points))

    r = radius * (1.0 - t) if inward else radius * t
    theta = 2.0 * np.pi * n_turns * t + phase

    x = center[0] + aspect[0] * r * np.cos(theta)
    y = center[1] + aspect[1] * r * np.sin(theta)

    path = np.column_stack([x, y])

    line_ids = np.floor((theta - phase) / (2.0 * np.pi)).astype(int)
    line_ids = np.clip(line_ids, 0, int(n_turns) - 1)

    return path, line_ids


def make_rectangular_spiral(
    center=(0.5, 0.5),
    width=0.96,
    height=0.96,
    n_segments=160,
    n_points=12000,
    outward=True,
):
    """
    Generate a rectangular/square spiral trajectory made from axis-aligned segments.

    Returns
    -------
    path : ndarray, shape (N, 2)

    line_ids : ndarray, shape (N,)
        Segment index for each point.
    """
    n_segments = int(n_segments)
    n_points = int(n_points)

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

    vertices = np.asarray(vertices)

    min_xy = vertices.min(axis=0)
    max_xy = vertices.max(axis=0)
    span = max_xy - min_xy
    span[span == 0] = 1.0

    vertices_norm = (vertices - min_xy) / span

    vertices_scaled = np.empty_like(vertices_norm)
    vertices_scaled[:, 0] = center[0] - width / 2.0 + width * vertices_norm[:, 0]
    vertices_scaled[:, 1] = center[1] - height / 2.0 + height * vertices_norm[:, 1]

    path = resample_polyline(vertices_scaled, n_points)

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
    """
    Generate a Lissajous trajectory.

    x(t) = cx + Ax sin(ax t + phase)
    y(t) = cy + Ay sin(ay t)

    Returns
    -------
    path : ndarray, shape (N, 2)

    line_ids : ndarray, shape (N,)
        Period index for each point.
    """
    t = np.linspace(0.0, 2.0 * np.pi * n_periods, int(n_points))

    x = center[0] + amplitude[0] * np.sin(ax * t + phase)
    y = center[1] + amplitude[1] * np.sin(ay * t)

    path = np.column_stack([x, y])

    line_ids = np.floor(t / (2.0 * np.pi)).astype(int)
    line_ids = np.clip(line_ids, 0, int(n_periods) - 1)

    return path, line_ids


def generate_scan_path(path_type="raster", **kwargs):
    """
    Unified trajectory generator.

    Parameters
    ----------
    path_type : str
        Options:
        - "raster"
        - "spiral"
        - "rectangular_spiral"
        - "lissajous"

    Returns
    -------
    path : ndarray, shape (N, 2)

    line_ids : ndarray, shape (N,)
    """
    path_type = path_type.lower()

    if path_type in ["raster", "rectangular", "rectangular_raster"]:
        return make_rectangular_raster(**kwargs)

    if path_type in ["spiral", "archimedean_spiral", "smooth_spiral"]:
        return make_spiral(**kwargs)

    if path_type in ["rectangular_spiral", "square_spiral"]:
        return make_rectangular_spiral(**kwargs)

    if path_type in ["lissajous", "lisajous"]:
        return make_lissajous(**kwargs)

    raise ValueError(
        "Unknown path_type. Use raster, spiral, rectangular_spiral, or lissajous."
    )
