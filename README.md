# ScanPath
# Instrument Transfer Functions

Toy models for microscope scan-path transfer functions.

The package models the mapping

```math
u(t) \rightarrow r(t)
```

where `u(t)` is the ideal commanded trajectory and `r(t)` is the realized trajectory.

## Components

- trajectory synthesis
- instrument transfer-function models
- path-quality metrics
- checkerboard image simulation
- parameter sweeps
- visualization utilities

## Installation from GitHub

```bash
pip install git+https://github.com/YOUR_USERNAME/instrument-transfer-functions.git
```

Replace `YOUR_USERNAME` with your GitHub username or organization.

## Basic example

```python
from instrument_transfer import (
    generate_scan_path,
    spm_transfer_function,
    build_spm_params_hysteresis_dominated,
    evaluate_path_quality,
)

ideal_path, line_ids = generate_scan_path("lissajous")

params = build_spm_params_hysteresis_dominated(
    hysteresis_strength=2.0,
    other_distortion_scale=0.0,
)

real_path = spm_transfer_function(
    ideal_path,
    params=params,
    line_ids=line_ids,
)

metrics = evaluate_path_quality(ideal_path, real_path)

print(metrics)
```

## Full comparison example

```python
from instrument_transfer.sweeps import run_all_trajectory_comparisons
from instrument_transfer.visualization import plot_metric_comparison

all_results = run_all_trajectory_comparisons()

plot_metric_comparison(all_results, "distortion_score")
```

## Current trajectory families

- rectangular raster
- smooth Archimedean spiral
- rectangular spiral
- Lissajous trajectory

## Current instrument model

The first implemented model is an SPM-like scanner transfer function including:

- affine distortion
- weak static nonlinear distortion
- first-order scanner response
- hysteresis
- creep
- drift
- line jitter

The default parameter builder can generate a clean hysteresis-dominated model where all other distortions are essentially zero.
