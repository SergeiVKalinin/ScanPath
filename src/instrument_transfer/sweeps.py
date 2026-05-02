import numpy as np

from .trajectories import generate_scan_path
from .instruments import (
    spm_transfer_function,
    build_spm_params_hysteresis_dominated,
)
from .metrics import evaluate_path_quality
from .imaging import make_checkerboard, simulate_unknown_trajectory_image


def run_hysteresis_sweep_for_one_trajectory(
    trajectory_type,
    trajectory_kwargs,
    hysteresis_values,
    other_distortion_scale=1e-4,
    dt=1.0,
    image_grid_size=256,
    checkerboard=None,
    seed=0,
):
    """
    Run a hysteresis sweep for one trajectory family.

    Returns
    -------
    sweep : dict
    """
    ideal_path, line_ids = generate_scan_path(
        trajectory_type,
        **trajectory_kwargs,
    )

    if checkerboard is None:
        checkerboard = make_checkerboard(
            size=512,
            n_checks_x=10,
            n_checks_y=10,
        )

    results = []

    for i, hval in enumerate(hysteresis_values):
        params = build_spm_params_hysteresis_dominated(
            hysteresis_strength=hval,
            other_distortion_scale=other_distortion_scale,
        )

        real_path = spm_transfer_function(
            ideal_path,
            dt=dt,
            params=params,
            line_ids=line_ids,
            seed=seed + i,
        )

        metrics = evaluate_path_quality(
            ideal_path,
            real_path,
            dt=dt,
        )

        image_simulation = simulate_unknown_trajectory_image(
            sample_image=checkerboard,
            ideal_path=ideal_path,
            real_path=real_path,
            grid_size=image_grid_size,
            interpolation_method="linear",
        )

        results.append({
            "hysteresis_strength": float(hval),
            "ideal_path": ideal_path,
            "real_path": real_path,
            "params": params,
            "metrics": metrics,
            "ideal_reconstruction": image_simulation["ideal_reconstruction"],
            "distorted_reconstruction": image_simulation["distorted_reconstruction"],
            "ideal_samples": image_simulation["ideal_samples"],
            "distorted_samples": image_simulation["distorted_samples"],
        })

    sweep = {
        "trajectory_type": trajectory_type,
        "trajectory_kwargs": trajectory_kwargs,
        "hysteresis_values": np.asarray(hysteresis_values, dtype=float),
        "ideal_path": ideal_path,
        "line_ids": line_ids,
        "checkerboard": checkerboard,
        "results": results,
    }

    return sweep


def run_all_trajectory_comparisons(
    hysteresis_values=None,
    other_distortion_scale=1e-4,
    dt=1.0,
    image_grid_size=256,
    seed=0,
):
    """
    Compare rectangular raster, spiral, rectangular spiral, and Lissajous scans.

    Returns
    -------
    all_results : dict
        Keys are trajectory family names.
    """
    if hysteresis_values is None:
        hysteresis_values = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0])

    checkerboard = make_checkerboard(
        size=512,
        n_checks_x=10,
        n_checks_y=10,
    )

    trajectory_configs = {
        "raster": {
            "n_lines": 100,
            "points_per_line": 120,
            "flyback_points": 20,
            "bidirectional": False,
            "include_flyback": True,
        },
        "spiral": {
            "center": (0.5, 0.5),
            "radius": 0.48,
            "n_turns": 25,
            "n_points": 12000,
            "inward": False,
        },
        "rectangular_spiral": {
            "center": (0.5, 0.5),
            "width": 0.96,
            "height": 0.96,
            "n_segments": 160,
            "n_points": 12000,
            "outward": True,
        },
        "lissajous": {
            "center": (0.5, 0.5),
            "amplitude": (0.48, 0.48),
            "ax": 23,
            "ay": 31,
            "phase": np.pi / 2,
            "n_periods": 8,
            "n_points": 20000,
        },
    }

    all_results = {}

    for trajectory_name, trajectory_kwargs in trajectory_configs.items():
        all_results[trajectory_name] = run_hysteresis_sweep_for_one_trajectory(
            trajectory_type=trajectory_name,
            trajectory_kwargs=trajectory_kwargs,
            hysteresis_values=hysteresis_values,
            other_distortion_scale=other_distortion_scale,
            dt=dt,
            image_grid_size=image_grid_size,
            checkerboard=checkerboard,
            seed=seed,
        )

    return all_results


def metric_table_from_sweep(sweep):
    """
    Convert one sweep dictionary into a list of metric rows.

    Returns
    -------
    rows : list of dict
    """
    rows = []

    for result in sweep["results"]:
        row = {
            "trajectory_type": sweep["trajectory_type"],
            "hysteresis_strength": result["hysteresis_strength"],
        }
        row.update(result["metrics"])
        rows.append(row)

    return rows


def metric_table_from_all_results(all_results):
    """
    Convert all sweep results into a list of metric rows.

    This avoids adding pandas as a dependency.
    """
    rows = []

    for sweep in all_results.values():
        rows.extend(metric_table_from_sweep(sweep))

    return rows
