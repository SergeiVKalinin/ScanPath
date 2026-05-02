import numpy as np


def build_spm_params_hysteresis_dominated(
    hysteresis_strength=1.0,
    other_distortion_scale=1e-4,
):
    """
    Build an SPM parameter dictionary where hysteresis is the dominant
    swept mechanism.

    At hysteresis_strength = 0 and other_distortion_scale = 0,
    the transfer function is identity up to numerical precision.

    Parameters
    ----------
    hysteresis_strength : float
        Multiplier for the hysteresis output amplitude.

    other_distortion_scale : float
        Controls all non-hysteretic imperfections.

    Returns
    -------
    params : dict
        Parameter dictionary for spm_transfer_function.
    """
    eps = float(other_distortion_scale)

    A = np.array([
        [1.0 + 0.001 * eps, 0.001 * eps],
        [-0.001 * eps, 1.0 - 0.001 * eps],
    ])

    nonlinear = {
        "x20": 0.001 * eps,
        "x11": 0.001 * eps,
        "x02": -0.001 * eps,
        "x30": 0.001 * eps,
        "y20": -0.001 * eps,
        "y11": 0.001 * eps,
        "y02": 0.001 * eps,
        "y03": -0.001 * eps,
    }

    # For dt = 1, tau = 1 makes the first-order scanner lag vanish
    # in the explicit Euler update used here.
    tau = np.array([1.0, 1.0])

    alpha_h = np.array([0.18, 0.14])
    beta_h = np.array([1.8, 1.4])

    base_B_h = np.array([
        [0.030, 0.000],
        [0.000, 0.025],
    ])

    B_h = float(hysteresis_strength) * base_B_h

    params = {
        "offset": np.array([0.0, 0.0]),
        "A": A,
        "nonlinear": nonlinear,
        "tau": tau,
        "alpha_h": alpha_h,
        "beta_h": beta_h,
        "B_h": B_h,
        "creep_taus": np.array([80.0, 300.0]),
        "creep_weights": eps * np.array([1e-4, 5e-5]),
        "drift_velocity": eps * np.array([1e-5, -1e-5]),
        "drift_random_walk_sigma": eps * 1e-6,
        "jitter_rho": 0.7,
        "jitter_sigma": eps * 1e-5,
    }

    return params


def spm_transfer_function(
    ideal_path,
    dt=1.0,
    params=None,
    line_ids=None,
    seed=0,
    return_components=False,
):
    """
    Simulate an SPM scanner transfer function.

    The input is an ideal commanded path. The output is the realized path.

    Parameters
    ----------
    ideal_path : ndarray, shape (N, 2)
        Ideal commanded trajectory.

    dt : float
        Time step between path samples.

    params : dict
        Scanner parameter dictionary.

    line_ids : ndarray, shape (N,), optional
        Line/segment IDs used for line jitter.

    seed : int
        Random seed for drift/jitter components.

    return_components : bool
        If True, return internal state arrays.

    Returns
    -------
    real_path : ndarray, shape (N, 2)

    components : dict, optional
    """
    rng = np.random.default_rng(seed)

    ideal_path = np.asarray(ideal_path, dtype=float)

    if ideal_path.ndim != 2 or ideal_path.shape[1] != 2:
        raise ValueError("ideal_path must have shape (N, 2).")

    N = ideal_path.shape[0]

    if params is None:
        params = build_spm_params_hysteresis_dominated(
            hysteresis_strength=0.0,
            other_distortion_scale=0.0,
        )

    offset = np.asarray(params.get("offset", [0.0, 0.0]), dtype=float)
    A = np.asarray(params.get("A", [[1.0, 0.0], [0.0, 1.0]]), dtype=float)

    nonlinear = params.get(
        "nonlinear",
        {
            "x20": 0.0,
            "x11": 0.0,
            "x02": 0.0,
            "x30": 0.0,
            "y20": 0.0,
            "y11": 0.0,
            "y02": 0.0,
            "y03": 0.0,
        },
    )

    tau = np.asarray(params.get("tau", [1.0, 1.0]), dtype=float)
    alpha_h = np.asarray(params.get("alpha_h", [0.18, 0.14]), dtype=float)
    beta_h = np.asarray(params.get("beta_h", [1.8, 1.4]), dtype=float)
    B_h = np.asarray(params.get("B_h", [[0.0, 0.0], [0.0, 0.0]]), dtype=float)

    creep_taus = np.asarray(params.get("creep_taus", [80.0, 300.0]), dtype=float)
    creep_weights = np.asarray(params.get("creep_weights", [0.0, 0.0]), dtype=float)

    drift_velocity = np.asarray(params.get("drift_velocity", [0.0, 0.0]), dtype=float)
    drift_random_walk_sigma = float(params.get("drift_random_walk_sigma", 0.0))

    jitter_rho = float(params.get("jitter_rho", 0.0))
    jitter_sigma = float(params.get("jitter_sigma", 0.0))

    if len(creep_taus) != len(creep_weights):
        raise ValueError("creep_taus and creep_weights must have the same length.")

    real_path = np.zeros_like(ideal_path)
    scanner_state = np.zeros_like(ideal_path)
    quasi_static_target = np.zeros_like(ideal_path)
    hysteresis_state = np.zeros_like(ideal_path)
    drift_state = np.zeros_like(ideal_path)

    n_creep = len(creep_taus)
    creep_state = np.zeros((N, n_creep, 2))

    scanner_state[0] = ideal_path[0]
    real_path[0] = ideal_path[0]

    for m in range(n_creep):
        creep_state[0, m] = ideal_path[0]

    line_jitter = np.zeros_like(ideal_path)

    if line_ids is not None:
        line_ids = np.asarray(line_ids, dtype=int)

        if len(line_ids) != N:
            raise ValueError("line_ids must have the same length as ideal_path.")

        previous_jitter = np.zeros(2)

        for line in np.unique(line_ids):
            current_jitter = (
                jitter_rho * previous_jitter
                + rng.normal(scale=jitter_sigma, size=2)
            )
            line_jitter[line_ids == line] = current_jitter
            previous_jitter = current_jitter

    for i in range(1, N):
        u = ideal_path[i]
        u_prev = ideal_path[i - 1]
        du = (u - u_prev) / dt

        x, y = u

        nonlinear_distortion = np.array([
            nonlinear["x20"] * x**2
            + nonlinear["x11"] * x * y
            + nonlinear["x02"] * y**2
            + nonlinear["x30"] * x**3,

            nonlinear["y20"] * x**2
            + nonlinear["y11"] * x * y
            + nonlinear["y02"] * y**2
            + nonlinear["y03"] * y**3,
        ])

        hysteresis_state[i] = (
            hysteresis_state[i - 1]
            + dt * (
                alpha_h * du
                - beta_h * np.abs(du) * hysteresis_state[i - 1]
            )
        )

        creep_error = np.zeros(2)

        for m in range(n_creep):
            creep_state[i, m] = (
                creep_state[i - 1, m]
                + dt * (u - creep_state[i - 1, m]) / creep_taus[m]
            )
            creep_error += creep_weights[m] * (u - creep_state[i, m])

        quasi_static_target[i] = (
            offset
            + A @ u
            + nonlinear_distortion
            + B_h @ hysteresis_state[i]
            - creep_error
        )

        scanner_state[i] = (
            scanner_state[i - 1]
            + dt * (quasi_static_target[i] - scanner_state[i - 1]) / tau
        )

        drift_state[i] = (
            drift_state[i - 1]
            + drift_velocity * dt
            + rng.normal(scale=drift_random_walk_sigma, size=2)
        )

        real_path[i] = scanner_state[i] + drift_state[i] + line_jitter[i]

    if return_components:
        components = {
            "scanner_state": scanner_state,
            "quasi_static_target": quasi_static_target,
            "hysteresis_state": hysteresis_state,
            "creep_state": creep_state,
            "drift_state": drift_state,
            "line_jitter": line_jitter,
        }
        return real_path, components

    return real_path
