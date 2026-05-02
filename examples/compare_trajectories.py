import numpy as np
import matplotlib.pyplot as plt

from instrument_transfer.sweeps import run_all_trajectory_comparisons
from instrument_transfer.visualization import (
    plot_metric_comparison,
    plot_trajectory_gallery,
    plot_image_gallery,
)


def main():
    hysteresis_values = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0])

    all_results = run_all_trajectory_comparisons(
        hysteresis_values=hysteresis_values,
        other_distortion_scale=0.0,
        dt=1.0,
        image_grid_size=256,
        seed=0,
    )

    metric_names = [
        "coverage_score",
        "coverage_mismatch",
        "distortion_score",
        "distortion_roughness_rms",
        "distortion_magnitude",
        "smooth_work_score",
    ]

    for metric_name in metric_names:
        plot_metric_comparison(all_results, metric_name)

    for sweep in all_results.values():
        plot_trajectory_gallery(sweep, n_cols=3)
        plot_image_gallery(sweep, n_cols=3)

    plt.show()


if __name__ == "__main__":
    main()
