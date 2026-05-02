import numpy as np

from instrument_transfer.trajectories import generate_scan_path
from instrument_transfer.metrics import evaluate_path_quality


def test_metrics_are_finite_for_identity_path():
    path, _ = generate_scan_path(
        "spiral",
        n_points=1000,
        n_turns=5,
    )

    metrics = evaluate_path_quality(path, path)

    for value in metrics.values():
        assert np.isfinite(value)
