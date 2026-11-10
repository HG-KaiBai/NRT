"""Harrell/Brier for exact outcomes; Uno/Yang IPCW for right censoring.

The marginal censoring KM assumes censoring independent of event time and risk.
Predictions always come from the evaluated model's own survival distribution.
See METRIC_REFERENCES and README.md for sources and discrete-time conventions.
"""
import numpy as np
import torch


METRIC_REFERENCES = {
    'harrell': dict(authors='Harrell FE Jr, Califf RM, Pryor DB, Lee KL, Rosati RA',
                    year=1982, title='Evaluating the Yield of Medical Tests',
                    doi='10.1001/jama.1982.03320430047030'),
    'brier': dict(authors='Brier GW', year=1950,
                  title='Verification of Forecasts Expressed in Terms of Probability',
                  doi='10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2'),
    'uno': dict(authors="Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ", year=2011,
                title='On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data',
                doi='10.1002/sim.4154'),
    'yang': dict(authors='Yang Z, Rizopoulos D, Newcomb LF, Erler NS', year=2026,
                 title='Time-Dependent Predictive Accuracy Metrics in the Context of Interval Censoring and Competing Risks',
                 doi='10.1002/bimj.70108'),
    'graf': dict(authors='Graf E, Schmoor C, Sauerbrei W, Schumacher M', year=1999,
                 title='Assessment and comparison of prognostic classification schemes for survival data',
                 doi='10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5'),
    'km': dict(authors='Kaplan EL, Meier P', year=1958,
               title='Nonparametric Estimation from Incomplete Observations',
               doi='10.1080/01621459.1958.10501452'),
}


def evaluation_grid(train_time, train_event, *, censored, size=256):
    """Match simulation: torch.linspace(0, max finite training bound, 256).

    For exact/right-censored outcomes the finite lower bound is the observed time;
    finite upper bounds repeat event times. Censored observations also define tau.
    The float32 conversion matches RealData.bounds and simulation Dataset.bounds.
    """
    times = torch.as_tensor(np.asarray(train_time), dtype=torch.float32)
    finite = times[torch.isfinite(times)]
    if not len(finite) or float(finite.max()) <= 0 or size != 256:
        raise ValueError('Invalid evaluation grid')
    return torch.linspace(0., float(finite.max()), size).numpy()


def censoring_survival(time, event, query, *, counts=None, reverse=False, left_limit=False):
    """KM on the evaluation sample with the referenced author's tie convention.

    Default: ordinary KM, survival::survfit(Surv(T, 1-event)~1), as in Yang's code.
    reverse=True: remove tied events before censorings, matching sksurv's
    CensoringDistributionEstimator. left_limit selects G(t-) instead of G(t).
    Counts implement an exactly equivalent repeated-subject bootstrap sample.
    """
    time, event = np.asarray(time), np.asarray(event, bool)
    counts = np.ones(len(time)) if counts is None else np.asarray(counts, float)
    unique, inverse = np.unique(time, return_inverse=True)
    total = np.bincount(inverse, weights=counts, minlength=len(unique))
    censored = np.bincount(inverse, weights=counts * ~event, minlength=len(unique))
    at_risk = np.cumsum(total[::-1])[::-1]
    if reverse:
        at_risk -= total - censored
    hazards = np.divide(censored, at_risk, out=np.zeros_like(total), where=at_risk > 0)
    survival = np.r_[1., np.cumprod(1. - hazards)]
    return survival[np.searchsorted(unique, query, side='left' if left_limit else 'right')]


def brier_contributions(time, event, survival, grid, *, censored, counts=None):
    """Yang IPCW Brier summands for exact/right censoring and landmark t=0.

    Cases have event time <= u, controls are observed past u; already-censored
    subjects contribute zero. The denominator is n_test, not the sum of weights.
    An empty case/control summand stays zero even when its unused G is zero.
    No horizon truncation or positive floor is applied to the censoring survival.
    """
    time, event = np.asarray(time), np.asarray(event, bool)
    survival, grid = np.asarray(survival, float), np.asarray(grid, float)
    alive = time[:, None] > grid[None, :]
    failed = (~alive) & event[:, None]
    if not censored:
        if not event.all():
            raise ValueError('Ordinary Brier requires uncensored outcomes')
        return (alive.astype(float) - survival)**2
    present = np.ones(len(time), bool) if counts is None else np.asarray(counts) > 0
    failed &= present[:, None]
    alive &= present[:, None]
    event_g = censoring_survival(time, event, time, counts=counts)[:, None]
    grid_g = censoring_survival(time, event, grid, counts=counts)[None, :]
    if np.any(failed & (event_g <= 0)) or np.any(alive & (grid_g <= 0)):
        raise ValueError('Undefined IPCW for a contributing subject')
    cases, controls = np.zeros_like(survival), np.zeros_like(survival)
    np.divide(survival**2, event_g, out=cases, where=failed)
    np.divide((1-survival)**2, grid_g, out=controls, where=alive)
    return cases + controls


def concordance_pairs(time, event, risk, *, horizon=None):
    """Observed comparable pairs and concordance credit, before IPCW weighting.

    Event/event time ties are excluded; event/censor time ties are comparable.
    Prediction ties within 1e-8 get half credit, matching sksurv. A finite horizon
    restricts event anchors to T_i < horizon, as in concordance_index_ipcw(tau=...).
    Diagonals and pairs between duplicated copies of one subject are zero.
    """
    time, event, risk = np.asarray(time), np.asarray(event, bool), np.asarray(risk)
    anchors = event if horizon is None else event & (time < horizon)
    denominator = (anchors[:, None] & ((time[:, None] < time[None, :]) |
                   ((time[:, None] == time[None, :]) & ~event[None, :]))).astype(float)
    difference = risk[:, None] - risk[None, :]
    credit = (difference > 1e-8) + .5 * (np.abs(difference) <= 1e-8)
    return denominator * credit, denominator


def uno_ipcw_weights(time, event, horizon, *, counts=None):
    """Uno pair weights I(event, T_i < horizon) / G(T_i)^2, fitted on test KM.

    Match sksurv.metrics.concordance_index_ipcw with the evaluation sample as
    both KM reference and evaluation sample. Counts refit KM on each bootstrap.
    A zero G at an included event makes C unavailable; it is not clipped or
    silently dropped. Unselected copies and events at/after horizon get zero.
    """
    time = np.asarray(time)
    event = np.asarray(event, bool)
    present = np.ones(len(time), bool) if counts is None else np.asarray(counts) > 0
    active = event & (time < horizon) & present
    g = censoring_survival(time, event, time, counts=counts, reverse=True)
    weights = np.divide(1., g**2, out=np.zeros_like(g), where=active & (g > 0))
    weights[active & (g <= 0)] = np.nan
    return weights


def concordance_ratio(numerator, denominator, *, counts=None, weights=None):
    """Ratio of pair sums; optional multiplicities exactly implement resampling."""
    n = len(denominator)
    counts = np.ones(n) if counts is None else np.asarray(counts)
    anchors = counts if weights is None else counts * weights
    den = float(anchors @ (denominator @ counts))
    return float(anchors @ (numerator @ counts) / den) if den > 0 else float('nan')


def uno_concordance_curve(time, event, survival, grid):
    """Uno C with varying truncation t; fixed marker 1-S(tau) on the whole grid."""
    survival = np.asarray(survival, float)
    risk = 1-survival[:, -1]
    values = []
    for t in grid:
        weights = uno_ipcw_weights(time, event, t)
        num, den = concordance_pairs(time, event, risk, horizon=t)
        values.append(concordance_ratio(num, den, weights=weights))
    return np.asarray(values)


def evaluation_components(train_time, train_event, time, event, survival, grid, *, censored):
    """Validated Brier contributions and the protocol's distribution-derived risk."""
    time, event = np.asarray(time), np.asarray(event, bool)
    survival, grid = np.asarray(survival, float), np.asarray(grid, float)
    if (time.ndim != 1 or event.shape != time.shape or not len(time)
            or not np.isfinite(time).all() or grid.ndim != 1 or len(grid) < 2
            or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0)):
        raise ValueError('Invalid outcomes or evaluation grid')
    if survival.shape != (len(time), len(grid)) or not np.isfinite(survival).all():
        raise ValueError('Invalid survival prediction shape or nonfinite values')
    if (survival.min() < -1e-6 or survival.max() > 1 + 1e-6 or
            np.any(np.diff(survival, axis=1) > 1e-5)):
        raise ValueError('Predictions must be nonincreasing survival probabilities')
    # Each statistic uses one scalar marker derived from the fitted distribution.
    risk = 1-survival[:, -1] if censored else -np.trapezoid(survival, grid, axis=1)
    contributions = brier_contributions(time, event, survival, grid, censored=censored)
    return contributions, risk


def evaluate(train_time, train_event, time, event, survival, grid, *, censored):
    time, event, grid = np.asarray(time), np.asarray(event, bool), np.asarray(grid, float)
    contributions, risk = evaluation_components(
        train_time, train_event, time, event, survival, grid, censored=censored)
    curve = contributions.mean(axis=0)
    num, den = concordance_pairs(time, event, risk, horizon=grid[-1] if censored else None)
    weights = uno_ipcw_weights(time, event, grid[-1]) if censored else None
    if censored and not np.isfinite(weights).all():
        raise ValueError('Undefined Uno IPCW: zero censoring survival at an event before tau')
    cindex = concordance_ratio(num, den, weights=weights)
    if not np.isfinite(cindex):
        raise ValueError('No usable comparable event pairs')
    ibs = float(np.trapezoid(curve, grid) / (grid[-1]-grid[0]))
    references = {name: METRIC_REFERENCES[key] for name, key in (
        ('c_index', 'uno' if censored else 'harrell'),
        ('brier_score', 'yang' if censored else 'brier'),
        ('integrated_brier_score', 'graf'))}
    if censored:
        references['censoring_distribution'] = METRIC_REFERENCES['km']
    return dict(c_index=cindex, integrated_brier_score=ibs, brier_at_horizon=float(curve[-1]),
                horizon=float(grid[-1]), metric_variant='Uno_IPCW_C + Yang_IPCW_Brier' if censored else 'Harrell_C + ordinary_Brier',
                brier_definition='Yang_2026_IPCW_single_event_landmark_zero' if censored else 'ordinary',
                brier_censoring_distribution='test_KM' if censored else 'none',
                c_index_definition='Uno_2011_IPCW_C_at_tau' if censored else 'Harrell_1982_C',
                c_index_censoring_distribution='test_reverse_KM' if censored else 'none',
                c_index_pair_selection='observed event before comparator; event/censor time ties included; event/event ties excluded',
                c_index_event_horizon='T_i < tau' if censored else 'all observed event times',
                c_index_risk_ties='half credit within 1e-8',
                c_index_zero_g_policy='unavailable if G(T_i)=0 at any event before tau' if censored else 'not applicable',
                c_index_pair_weight='1 / G(T_i)^2' if censored else '1',
                c_index_censoring_positive_at_horizon=bool(censoring_survival(time, event, [grid[-1]], reverse=True)[0] > 0) if censored else True,
                risk_definition='event_probability_at_horizon' if censored else 'negative_RMST_to_common_training_horizon',
                metric_references=references), curve, risk
