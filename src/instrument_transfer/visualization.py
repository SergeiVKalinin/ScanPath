import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(path, title="Trajectory", ax=None, linewidth=1.0):
    """
    Plot a 2D trajectory.
    """
    path = np.asarray(path, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(path[:, 0], path[:, 1], linewidth=linewidth)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    return ax


def plot_trajectory_comparison(
    ideal_path,
    real_path,
    title="Ideal vs real trajectory",
    linewidth=1.0,
):
    """
    Plot ideal and real trajectories on the same axes.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        ideal_path[:, 0],
        ideal_path[:, 1],
        linewidth=linewidth,
        label="Ideal",
    )

    ax.plot(
        real_path[:, 0],
        real_path[:, 1],
        linewidth=linewidth,
        label="Real",
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend()

    return fig, ax


def plot_metric_comparison(
    all_results,
    metric_name,
    title=None,
):
    """
    Plot one metric vs hysteresis for multiple trajectory families.

    Parameters
    ----------
    all_results : dict
        Output of run_all_trajectory_comparisons.

    metric_name : str
        Name of metric to plot.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for trajectory_name, sweep in all_results.items():
        h = np.array([r["hysteresis_strength"] for r in sweep["results"]])
        y = np.array([r["metrics"][metric_name] for r in sweep["results"]])

        ax.plot(h, y, marker="o", label=trajectory_name)

    ax.set_xlabel("Hysteresis strength")
    ax.set_ylabel(metric_name)

    if title is None:
        title = f"{metric_name} vs hysteresis"

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_image_gallery(
    sweep,
    n_cols=3,
    cmap="gray",
):
    """
    Plot original checkerboard, ideal reconstruction, and distorted reconstructions.

    Parameters
    ----------
    sweep : dict
        Output of run_hysteresis_sweep_for_one_trajectory.
    """
    results = sweep["results"]
    n_panels = len(results) + 2
    n_rows = int(np.ceil(n_panels / n_cols))

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    idx = 1

    ax = plt.subplot(n_rows, n_cols, idx)
    ax.imshow(
        sweep["checkerboard"],
        cmap=cmap,
        origin="lower",
        extent=[0, 1, 0, 1],
    )
    ax.set_title("Original checkerboard")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    idx += 1

    ax = plt.subplot(n_rows, n_cols, idx)
    ax.imshow(
        results[0]["ideal_reconstruction"],
        cmap=cmap,
        origin="lower",
        extent=[0, 1, 0, 1],
    )
    ax.set_title("Ideal reconstruction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    idx += 1

    for result in results:
        ax = plt.subplot(n_rows, n_cols, idx)
        ax.imshow(
            result["distorted_reconstruction"],
            cmap=cmap,
            origin="lower",
            extent=[0, 1, 0, 1],
        )
        ax.set_title(f"h = {result['hysteresis_strength']:.3g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        idx += 1

    plt.tight_layout()

    return fig


def plot_trajectory_gallery(
    sweep,
    n_cols=3,
    linewidth=0.8,
):
    """
    Plot ideal trajectory and realized trajectories for a hysteresis sweep.
    """
    results = sweep["results"]
    n_panels = len(results) + 1
    n_rows = int(np.ceil(n_panels / n_cols))

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    idx = 1

    ax = plt.subplot(n_rows, n_cols, idx)
    ax.plot(
        sweep["ideal_path"][:, 0],
        sweep["ideal_path"][:, 1],
        linewidth=linewidth,
    )
    ax.set_title("Ideal trajectory")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    idx += 1

    for result in results:
        ax = plt.subplot(n_rows, n_cols, idx)
        ax.plot(
            result["real_path"][:, 0],
            result["real_path"][:, 1],
            linewidth=linewidth,
        )
        ax.set_title(f"h = {result['hysteresis_strength']:.3g}")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        idx += 1

    plt.tight_layout()

    return fig
