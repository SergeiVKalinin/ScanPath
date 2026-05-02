import numpy as np

from instrument_transfer.trajectories import generate_scan_path
from instrument_transfer.instruments import (
    spm_transfer_function,
    build_spm_params_hysteresis_dominated,
)


def test_zero_distortion_spm_identity():
    ideal_path, line_ids = generate_scan_path(
        "raster",
        n_lines=10,
        points_per_line=20,
        flyback_points=5,
    )

    params = build_spm_params_hysteresis_dominated(
        hysteresis_strength=0.0,
        other_distortion_scale=0.0,
    )

    real_path = spm_transfer_function(
        ideal_path,
        params=params,
        line_ids=line_ids,
        seed=0,
    )

    assert np.allclose(ideal_path, real_path)


def test_nonzero_hysteresis_changes_path():
    ideal_path, line_ids = generate_scan_path(
        "raster",
        n_lines=10,
        points_per_line=20,
        flyback_points=5,
    )

    params = build_spm_params_hysteresis_dominated(
        hysteresis_strength=2.0,
        other_distortion_scale=0.0,
    )

    real_path = spm_transfer_function(
        ideal_path,
        params=params,
        line_ids=line_ids,
        seed=0,
    )

    assert not np.allclose(ideal_path, real_path)
