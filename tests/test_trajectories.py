from instrument_transfer.trajectories import generate_scan_path


def test_lissajous_shape():
    path, line_ids = generate_scan_path(
        "lissajous",
        n_points=1000,
        ax=7,
        ay=9,
    )

    assert path.shape == (1000, 2)
    assert len(line_ids) == 1000


def test_spiral_shape():
    path, line_ids = generate_scan_path(
        "spiral",
        n_points=1000,
        n_turns=10,
    )

    assert path.shape == (1000, 2)
    assert len(line_ids) == 1000


def test_raster_shape():
    path, line_ids = generate_scan_path(
        "raster",
        n_lines=10,
        points_per_line=20,
        flyback_points=5,
    )

    assert path.ndim == 2
    assert path.shape[1] == 2
    assert len(path) == len(line_ids)
