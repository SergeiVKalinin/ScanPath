import numpy as np
from scipy.interpolate import griddata


def make_checkerboard(
    size=512,
    n_checks_x=10,
    n_checks_y=10,
    low=0.0,
    high=1.0,
):
    """
    Create a checkerboard image on the domain [0, 1] x [0, 1].

    Returns
    -------
    image : ndarray, shape (size, size)
    """
    y = np.linspace(0.0, 1.0, int(size))
    x = np.linspace(0.0, 1.0, int(size))

    X, Y = np.meshgrid(x, y, indexing="xy")

    ix = np.floor(n_checks_x * X).astype(int)
    iy = np.floor(n_checks_y * Y).astype(int)

    board = (ix + iy) % 2

    image = np.where(board == 0, low, high).astype(float)

    return image


def sample_image_along_path(
    image,
    path,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    clip=True,
):
    """
    Sample a 2D image along a continuous 2D path using bilinear interpolation.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Image defined over xlim x ylim.

    path : ndarray, shape (N, 2)
        Continuous path coordinates.

    Returns
    -------
    values : ndarray, shape (N,)
        Sampled intensities.
    """
    image = np.asarray(image, dtype=float)
    path = np.asarray(path, dtype=float)

    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (N, 2).")

    H, W = image.shape

    xmin, xmax = xlim
    ymin, ymax = ylim

    x = (path[:, 0] - xmin) / (xmax - xmin) * (W - 1)
    y = (path[:, 1] - ymin) / (ymax - ymin) * (H - 1)

    if clip:
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)

    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    dx = x - x0
    dy = y - y0

    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    values = (
        Ia * (1.0 - dx) * (1.0 - dy)
        + Ib * dx * (1.0 - dy)
        + Ic * (1.0 - dx) * dy
        + Id * dx * dy
    )

    return values


def reconstruct_from_assigned_positions(
    assigned_positions,
    sample_values,
    grid_size=256,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    method="linear",
    fill_with_nearest=True,
):
    """
    Reconstruct an image on a regular grid from irregular samples.

    assigned_positions are the coordinates where the measurements are assumed
    to have occurred.

    Returns
    -------
    recon : ndarray, shape (grid_size, grid_size)
    """
    assigned_positions = np.asarray(assigned_positions, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float)

    if assigned_positions.ndim != 2 or assigned_positions.shape[1] != 2:
        raise ValueError("assigned_positions must have shape (N, 2).")

    if len(sample_values) != len(assigned_positions):
        raise ValueError("sample_values and assigned_positions must have same length.")

    xmin, xmax = xlim
    ymin, ymax = ylim

    xs = np.linspace(xmin, xmax, int(grid_size))
    ys = np.linspace(ymin, ymax, int(grid_size))

    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

    recon = griddata(
        assigned_positions,
        sample_values,
        (Xg, Yg),
        method=method,
    )

    if fill_with_nearest and np.any(np.isnan(recon)):
        recon_nearest = griddata(
            assigned_positions,
            sample_values,
            (Xg, Yg),
            method="nearest",
        )
        recon[np.isnan(recon)] = recon_nearest[np.isnan(recon)]

    return recon


def simulate_unknown_trajectory_image(
    sample_image,
    ideal_path,
    real_path,
    grid_size=256,
    interpolation_method="linear",
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
):
    """
    Simulate image acquisition when the real trajectory is unknown.

    Ideal case:
        sample at ideal_path and reconstruct at ideal_path.

    Distorted case:
        sample at real_path but reconstruct/display at ideal_path.

    This models the common experimental situation where the instrument believes
    it scanned the ideal path, while the physical trajectory was distorted.

    Returns
    -------
    results : dict
    """
    ideal_path = np.asarray(ideal_path, dtype=float)
    real_path = np.asarray(real_path, dtype=float)

    if ideal_path.shape != real_path.shape:
        raise ValueError("ideal_path and real_path must have the same shape.")

    ideal_samples = sample_image_along_path(
        sample_image,
        ideal_path,
        xlim=xlim,
        ylim=ylim,
    )

    distorted_samples = sample_image_along_path(
        sample_image,
        real_path,
        xlim=xlim,
        ylim=ylim,
    )

    ideal_reconstruction = reconstruct_from_assigned_positions(
        assigned_positions=ideal_path,
        sample_values=ideal_samples,
        grid_size=grid_size,
        xlim=xlim,
        ylim=ylim,
        method=interpolation_method,
    )

    distorted_reconstruction = reconstruct_from_assigned_positions(
        assigned_positions=ideal_path,
        sample_values=distorted_samples,
        grid_size=grid_size,
        xlim=xlim,
        ylim=ylim,
        method=interpolation_method,
    )

    return {
        "ideal_samples": ideal_samples,
        "distorted_samples": distorted_samples,
        "ideal_reconstruction": ideal_reconstruction,
        "distorted_reconstruction": distorted_reconstruction,
    }
