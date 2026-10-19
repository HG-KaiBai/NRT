"""Train-only IPCW and common survival-curve risk scores, with strict support checks."""
import numpy as np
from scipy.integrate import trapezoid
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import CensoringDistributionEstimator


def evaluation_grid(y_train, size=256):
    # Endpoints depend only on observed training follow-up, before any model fit.
    observed = np.asarray(y_train["time"], dtype=float)
    if not len(observed) or not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("Training follow-up must be nonempty, finite, and nonnegative")
    lower, upper = 0., float(observed.max())
    if upper <= 0 or size < 2:
        raise ValueError("Need positive maximum training follow-up and at least two grid points")
    grid = np.linspace(lower, upper, size)
    landmarks = [t for t in (30., 90., 180., 365., 730.) if lower <= t <= upper]
    return np.unique(np.r_[grid, landmarks]), lower, upper


def validate_survival(survival, n, grid):
    survival = np.asarray(survival, dtype=float)
    if survival.shape != (n, len(grid)) or not np.isfinite(survival).all():
        raise ValueError("Invalid survival prediction shape or nonfinite values")
    if (survival < -1e-6).any() or (survival > 1 + 1e-6).any() or (np.diff(survival, axis=1) > 1e-5).any():
        raise ValueError("Survival curves must be probabilities and nonincreasing")
    return survival


def ipcw_brier(y_train, y_test, survival, times):
    """Graf IPCW using scikit-survival's reverse-KM censoring convention.

    Events at/before t contribute S(t)^2/G(T); observations after t contribute
    (1-S(t))^2/G(t). Earlier censored observations contribute zero. Divide by
    the entire test sample size. Only evaluate G(T) for events within the grid;
    later test follow-up need not lie within the training censoring support.
    Require positive G only for nonzero indicator terms. At the maximum training
    time G may be zero: the control sum is empty if no test observation extends
    past that time. An event requiring zero G still raises. No epsilon clipping,
    endpoint truncation, or renormalization of weights is used.
    """
    times = np.asarray(times, dtype=float)
    s = validate_survival(survival, len(y_test), times)
    estimator = CensoringDistributionEstimator().fit(y_train)
    g_grid = estimator.predict_proba(times)
    event, observed = y_test["event"], y_test["time"]
    needed = event & (observed <= times[-1])
    g_event = np.ones(len(y_test))
    g_event[needed] = estimator.predict_proba(observed[needed])
    cases = event[:, None] & (observed[:, None] <= times[None, :])
    controls = observed[:, None] > times[None, :]
    if np.any(g_grid[controls.any(axis=0)] <= 0) or np.any(g_event[needed] <= 0):
        raise ValueError("Censoring survival is zero for a required IPCW contribution")
    # Mask division itself: multiplying zero by 1/G first would produce NaN.
    contributions = np.divide(s ** 2, g_event[:, None], out=np.zeros_like(s), where=cases)
    contributions += np.divide((1 - s) ** 2, g_grid[None, :], out=np.zeros_like(s), where=controls)
    return contributions.mean(0), g_grid


def evaluate(y_train, y_test, survival, grid, lower, native_risk=None):
    survival = validate_survival(survival, len(y_test), grid)
    risk = -trapezoid(survival, grid, axis=1)
    c_index = float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])
    selected = grid >= lower
    times = grid[selected]
    brier, g = ipcw_brier(y_train, y_test, survival[:, selected], times)
    integral = float(trapezoid(brier, times))
    metrics = dict(c_index=c_index,
                   integrated_brier_score=integral / (times[-1] - times[0]),
                   ipcw_brier_integral=integral,
                   evaluation_start_days=float(times[0]), evaluation_end_days=float(times[-1]),
                   min_censoring_survival=float(g.min()), risk_definition="negative_RMST_0_to_tau")
    if native_risk is not None:
        metrics["c_index_native_risk"] = float(concordance_index_censored(
            y_test["event"], y_test["time"], np.asarray(native_risk))[0])
    for landmark in (30, 90, 180, 365, 730):
        match = np.flatnonzero(times == landmark)
        metrics[f"brier_{landmark}d"] = float(brier[match[0]]) if len(match) else None
    rows = [dict(time_days=float(t), brier_ipcw=float(b), censoring_survival=float(w))
            for t, b, w in zip(times, brier, g)]
    return metrics, rows, risk
