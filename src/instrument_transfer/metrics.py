import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def _check_paths(ideal_path, real_path):
    ideal_path = np.asarray(ideal_path, dtype=float)
    real_path = np.asarray(real_path, dtype=float)

    if ideal_path.ndim != 2 or ideal_path.shape[1] != 2:
        raise ValueError("ideal_path must have shape (N, 2).")

    if real_path.ndim != 2 or real_path.shape[1] != 2:
        raise ValueError("real_path must have shape (N, 2).")

    if ideal_path.shape != real_path.shape:
        raise ValueError("ideal_path and real_path must have the same shape.")

    return ideal_path, real_path


def _domain_bbox(ideal_path, real_path, pad_fraction=0.05):
    all_points = np.vstack([ideal_path, real_path])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    span = maxs - mins
    span[span == 0] = 1.0

    mins = mins - pad_fraction * span
    maxs = maxs + pad_fraction * span

    return mins, maxs


def _normalized_coverage_map(path, bbox, grid_size=128, sigma=1.5):
    mins, maxs = bbox

    H, _, _ = np.histogram2d(
        path[:, 0],
        path[:, 1],
        bins=grid_size,
        range=[[mins[0], maxs[0]], [mins[1], maxs[1]]],
    )

    H = gaussian_filter(H, sigma=sigma)

    total = H.sum()
    if total > 0:
        H = H / total

    return H


def evaluate_smooth_coverage(
    ideal_path,
    real_path,
    grid_size=128,
    sigma=1.5,
    mask_threshold_fraction=0.01,
    pad_fraction=0.05,
):
    """
    Evaluate smooth spatial coverage of the realized path relative to the ideal path.

    Returns
    -------
    metrics : dict
        coverage_mismatch : lower is better
        coverage_roughness : lower is better
        coverage_score : higher is better
    """
    ideal_path, real_path = _check_paths(ideal_path, real_path)

    bbox = _domain_bbox(ideal_path, real_path, pad_fraction=pad_fraction)

    H_ideal = _normalized_coverage_map(
        ideal_path,
        bbox=bbox,
        grid_size=grid_size,
        sigma=sigma,
    )

    H_real = _normalized_coverage_map(
        real_path,
        bbox=bbox,
        grid_size=grid_size,
        sigma=sigma,
    )

    threshold = mask_threshold_fraction * H_ideal.max()
    mask = H_ideal > threshold

    if np.sum(mask) < 10:
        mask = H_ideal > 0

    eps = 1e-12

    diff = H_real - H_ideal

    rmse = np.sqrt(np.mean(diff[mask] ** 2))
    reference = np.mean(H_ideal[mask]) + eps
    coverage_mismatch = rmse / reference

    gx, gy = np.gradient(H_real)
    grad_energy = np.sqrt(np.mean((gx[mask] ** 2 + gy[mask] ** 2)))
    coverage_roughness = grad_energy / (np.mean(H_real[mask]) + eps)

    coverage_score = 1.0 / (1.0 + coverage_mismatch + coverage_roughness)

    return {
        "coverage_mismatch": float(coverage_mismatch),
        "coverage_roughness": float(coverage_roughness),
        "coverage_score": float(coverage_score),
    }


def evaluate_smooth_distortion(
    ideal_path,
    real_path,
    k_neighbors=12,
    max_points=5000,
    seed=0,
):
    """
    Evaluate smoothness of the distortion field

        d_i = real_path_i - ideal_path_i

    using a nearest-neighbor graph in ideal-coordinate space.

    Returns
    -------
    metrics : dict
        distortion_magnitude : RMS magnitude of displacement
        distortion_gradient_rms : local first derivative scale
        distortion_roughness_rms : non-affine local roughness
        distortion_score : higher is better
    """
    ideal_path, real_path = _check_paths(ideal_path, real_path)

    N = len(ideal_path)
    rng = np.random.default_rng(seed)

    if N > max_points:
        idx = rng.choice(N, size=max_points, replace=False)
        idx = np.sort(idx)
        u = ideal_path[idx]
        r = real_path[idx]
    else:
        u = ideal_path
        r = real_path

    d = r - u

    domain_span = np.ptp(u, axis=0)
    domain_diameter = np.linalg.norm(domain_span)
    if domain_diameter == 0:
        domain_diameter = 1.0

    tree = cKDTree(u)
    distances, indices = tree.query(u, k=k_neighbors + 1)

    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    eps = 1e-12

    local_gradients = []

    for i in range(len(u)):
        nbrs = neighbor_indices[i]
        dist = neighbor_distances[i]

        dd = d[nbrs] - d[i]
        dd_norm = np.linalg.norm(dd, axis=1)

        grad = dd_norm / (dist + eps)
        local_gradients.append(grad)

    local_gradients = np.asarray(local_gradients)
    distortion_gradient_rms = np.sqrt(np.mean(local_gradients**2))

    local_laplacians = []

    for i in range(len(u)):
        nbrs = neighbor_indices[i]
        dist = neighbor_distances[i]

        local_spacing = np.mean(dist) + eps
        neighbor_mean = np.mean(d[nbrs], axis=0)

        residual = d[i] - neighbor_mean
        laplacian_like = domain_diameter * residual / (local_spacing**2)

        local_laplacians.append(np.linalg.norm(laplacian_like))

    local_laplacians = np.asarray(local_laplacians)

    distortion_magnitude = np.sqrt(np.mean(np.sum(d**2, axis=1)))
    distortion_roughness_rms = np.sqrt(np.mean(local_laplacians**2))
    distortion_score = 1.0 / (1.0 + distortion_roughness_rms)

    return {
        "distortion_magnitude": float(distortion_magnitude),
        "distortion_gradient_rms": float(distortion_gradient_rms),
        "distortion_roughness_rms": float(distortion_roughness_rms),
        "distortion_score": float(distortion_score),
    }


def evaluate_smooth_work(path, dt=1.0):
    """
    Evaluate mechanical/electronic effort associated with a trajectory.

    This is not a distortion metric. It measures how hard a trajectory is
    to execute, based on velocity, acceleration, and jerk.

    Parameters
    ----------
    path : ndarray, shape (N, 2)
        Trajectory to evaluate.

    dt : float
        Time step between samples.

    Returns
    -------
    metrics : dict
        path_length, rms_velocity, rms_acceleration, rms_jerk,
        max_velocity, max_acceleration, smooth_work_penalty,
        smooth_work_score.
    """
    path = np.asarray(path, dtype=float)

    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (N, 2).")

    if len(path) < 4:
        raise ValueError("path must contain at least four points.")

    velocity = np.gradient(path, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)

    speed = np.linalg.norm(velocity, axis=1)
    accel_mag = np.linalg.norm(acceleration, axis=1)
    jerk_mag = np.linalg.norm(jerk, axis=1)

    step_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path_length = np.sum(step_lengths)

    rms_velocity = np.sqrt(np.mean(speed**2))
    rms_acceleration = np.sqrt(np.mean(accel_mag**2))
    rms_jerk = np.sqrt(np.mean(jerk_mag**2))

    max_velocity = np.max(speed)
    max_acceleration = np.max(accel_mag)

    smooth_work_penalty = rms_acceleration + 0.1 * rms_jerk
    smooth_work_score = 1.0 / (1.0 + smooth_work_penalty)

    return {
        "path_length": float(path_length),
        "rms_velocity": float(rms_velocity),
        "rms_acceleration": float(rms_acceleration),
        "rms_jerk": float(rms_jerk),
        "max_velocity": float(max_velocity),
        "max_acceleration": float(max_acceleration),
        "smooth_work_penalty": float(smooth_work_penalty),
        "smooth_work_score": float(smooth_work_score),
    }


def evaluate_path_quality(
    ideal_path,
    real_path,
    dt=1.0,
    grid_size=128,
    sigma=1.5,
    k_neighbors=12,
):
    """
    Evaluate coverage, distortion, and smooth work.

    Returns
    -------
    metrics : dict
        Combined metric dictionary.
    """
    coverage = evaluate_smooth_coverage(
        ideal_path,
        real_path,
        grid_size=grid_size,
        sigma=sigma,
    )

    distortion = evaluate_smooth_distortion(
        ideal_path,
        real_path,
        k_neighbors=k_neighbors,
    )

    work = evaluate_smooth_work(real_path, dt=dt)

    metrics = {}
    metrics.update(coverage)
    metrics.update(distortion)
    metrics.update(work)

    return metrics
