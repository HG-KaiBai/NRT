"""Six independent uniform covariates and four conditional survival models.

See ``testdata2.md`` for the distributions and observation process.
"""

from __future__ import annotations

from functools import lru_cache
import math

import torch


SCENARIOS = ("ph", "lognormal", "po", "multimodal")
PH_RATE = -math.log(0.1)
LOGNORMAL_SIGMA = 0.5
PO_SHAPE = 2.0
PO_MAX_TIME = 20.0
# Narrower peaks, similar to the legacy shape-5 mixture. Distinct shapes
# retain time-varying hazard ratios after marginalizing over components.
MULTIMODAL_SHAPES = (6.0, 8.0)
MULTIMODAL_SCALE_MULTIPLIER_CASES = {
    "separate": (1.0, 6.0),
    "closer": (3.0, 6.0),
}
MULTIMODAL_SCALE_MULTIPLIERS = MULTIMODAL_SCALE_MULTIPLIER_CASES["separate"]
MULTIMODAL_WEIGHTS = (0.5, 0.5)


def _radius(scenario: str, r: float | None) -> float:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    radius = (0.5 if scenario == "po" else 0.7) if r is None else float(r)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("r must be finite and positive")
    return radius


def _parameters(
    x: torch.Tensor, scenario: str, mu: float, r: float,
    multimodal_case: str = "separate",
) -> tuple[torch.Tensor, torch.Tensor]:
    eta = -x[:, :2].square().sum(dim=1) / (2.0 * r**2)
    if scenario == "multimodal":
        shapes = x.new_tensor(MULTIMODAL_SHAPES).expand(len(x), -1)
        scales = x.new_tensor(MULTIMODAL_SCALE_MULTIPLIER_CASES[multimodal_case]) * torch.exp(
            mu - eta[:, None] / shapes
        )
    elif scenario == "lognormal":
        shapes = torch.full_like(eta, LOGNORMAL_SIGMA)
        scales = torch.exp(eta + mu)
    elif scenario == "po":
        # For bounded PO, return the endpoint and dimensionless odds scale.
        shapes = torch.full_like(eta, PO_MAX_TIME * math.exp(mu))
        scales = torch.exp(-eta / PO_SHAPE) / PO_MAX_TIME
    else:
        shapes = torch.ones_like(eta)
        scales = torch.exp(mu - eta)
        if scenario == "ph":
            scales = scales / PH_RATE
    return shapes, scales


def survival_curves(
    shape: torch.Tensor,
    scale: torch.Tensor,
    grid: torch.Tensor,
    *,
    scenario: str,
) -> torch.Tensor:
    """Conditional survival for PH, PO, or lognormal parameters returned below."""
    if scenario not in {"ph", "po", "lognormal"}:
        raise ValueError("use multimodal_survival for the mixture scenario")
    if shape.ndim != 1 or scale.shape != shape.shape:
        raise ValueError("shape and scale must have the same one-dimensional shape")
    if grid.ndim != 1 or bool((grid < 0).any()):
        raise ValueError("grid must be one-dimensional and nonnegative")
    if bool((shape <= 0).any()) or bool((scale <= 0).any()):
        raise ValueError("shape and scale must be positive")
    grid = grid.to(device=scale.device, dtype=scale.dtype)
    shape = shape.to(device=scale.device, dtype=scale.dtype)
    if scenario == "lognormal":
        z = (torch.log(grid[None, :]) - torch.log(scale[:, None])) / shape[:, None]
        return 0.5 * torch.erfc(z / math.sqrt(2.0))
    scaled_time = grid[None, :] / scale[:, None]
    if scenario == "po":
        return _po_survival(shape, scale, grid)
    return torch.exp(-scaled_time)


def _po_survival(endpoint: torch.Tensor, scale: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bounded PO with baseline odds (t / (endpoint - t)) ** PO_SHAPE."""
    remaining = (endpoint[:, None] - grid[None, :]).clamp_min(torch.finfo(scale.dtype).tiny)
    odds_root = grid[None, :] / remaining / scale[:, None]
    survival = 1.0 / (1.0 + odds_root.pow(PO_SHAPE))
    return torch.where(grid[None, :] < endpoint[:, None], survival, 0.0)


def multimodal_survival(
    mixture_weights: torch.Tensor,
    component_scales: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Marginalize over both latent Weibull components, each with its own shape."""
    if component_scales.ndim != 2 or component_scales.shape[1] != 2:
        raise ValueError("component_scales must have shape (rows, 2)")
    if mixture_weights.shape != component_scales.shape:
        raise ValueError("mixture_weights must match component_scales")
    if bool((component_scales <= 0).any()):
        raise ValueError("component_scales must be positive")
    if bool((mixture_weights < 0).any()) or not torch.allclose(
        mixture_weights.sum(dim=1), torch.ones_like(mixture_weights[:, 0])
    ):
        raise ValueError("mixture_weights must be nonnegative and sum to one")
    if grid.ndim != 1 or bool((grid < 0).any()):
        raise ValueError("grid must be one-dimensional and nonnegative")
    grid = grid.to(device=component_scales.device, dtype=component_scales.dtype)
    weights = mixture_weights.to(component_scales)
    shapes = component_scales.new_tensor(MULTIMODAL_SHAPES)
    survival = torch.exp(
        -(grid[None, None, :] / component_scales[:, :, None]).pow(shapes[None, :, None])
    )
    return (weights[:, :, None] * survival).sum(dim=1)


def _uniform_censor_probability(
    tau: float, shape: torch.Tensor, scale: torch.Tensor, scenario: str
) -> float:
    """E[min(T, tau)] / tau = P(T > C), for independent C ~ U[0, tau]."""
    ratio = tau / scale
    if scenario == "ph":
        probability = -torch.expm1(-ratio) / ratio
    elif scenario == "po":
        # Simpson quadrature over the finite support. Calibration uses mu=0;
        # scaling the resulting horizon by exp(mu) preserves censoring rates.
        limit = min(tau, float(shape[0]))
        grid = torch.linspace(0.0, limit, 513, device=scale.device, dtype=scale.dtype)
        survival = _po_survival(shape, scale, grid)
        integral = (survival[:, 0] + survival[:, -1]
                    + 4.0 * survival[:, 1:-1:2].sum(dim=1)
                    + 2.0 * survival[:, 2:-1:2].sum(dim=1)) * (limit / 1536.0)
        probability = integral / tau
    elif scenario == "lognormal":
        z = torch.log(ratio) / shape
        survival = 0.5 * torch.erfc(z / math.sqrt(2.0))
        truncated_mean = scale * torch.exp(shape.square() / 2.0) * (
            0.5 * torch.erfc((shape - z) / math.sqrt(2.0))
        )
        probability = survival + truncated_mean / tau
    else:
        # Integral of Weibull survival uses the regularized incomplete gamma.
        probability = (
            scale / tau * torch.exp(torch.lgamma(1.0 + 1.0 / shape))
            * torch.special.gammainc(1.0 / shape, ratio.pow(shape))
        )
        probability = (probability * scale.new_tensor(MULTIMODAL_WEIGHTS)).sum(dim=1)
    return float(probability.mean())


@lru_cache(maxsize=128)
def _calibrated_horizon(scenario: str, q: float, r: float, multimodal_case: str) -> float:
    # Deterministic population quadrature, independent of the realized events,
    # covariates, latent components, and the caller's random-number stream.
    points = (torch.arange(64, dtype=torch.float64) + 0.5) / 32.0 - 1.0
    x = torch.cartesian_prod(points, points)
    shape, scale = _parameters(x, scenario, 0.0, r, multimodal_case)
    lower, upper = 0.0, 1.0
    for _ in range(1024):
        if _uniform_censor_probability(upper, shape, scale, scenario) <= q:
            break
        upper *= 2.0
    else:
        raise ValueError("q is too small to calibrate a finite censoring horizon")
    for _ in range(50):
        midpoint = (lower + upper) / 2.0
        if _uniform_censor_probability(midpoint, shape, scale, scenario) > q:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def censoring_horizon(
    scenario: str, q: float = 0.2, *, mu: float = 0.0, r: float | None = None,
    multimodal_case: str = "separate",
) -> float:
    """Population-calibrated uniform follow-up limit; defaults target 20% censoring."""
    radius = _radius(scenario, r)
    if scenario == "multimodal" and multimodal_case not in MULTIMODAL_SCALE_MULTIPLIER_CASES:
        raise ValueError(f"multimodal_case must be one of {tuple(MULTIMODAL_SCALE_MULTIPLIER_CASES)}")
    if not math.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1) for censored data")
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    horizon = _calibrated_horizon(scenario, float(q), radius, multimodal_case) * math.exp(mu)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("mu must yield a finite positive censoring horizon")
    return horizon


def _observe(
    event_time: torch.Tensor, censor_type: str, tau: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if censor_type == "exact":
        return event_time.clone(), event_time.clone()
    censor_time = torch.rand_like(event_time) * tau
    # Exclude the zero endpoint: right-censoring at zero makes AFT log(0).
    zero = censor_time == 0
    while bool(zero.any()):
        censor_time[zero] = torch.rand_like(censor_time[zero]) * tau
        zero = censor_time == 0
    censored = event_time >= censor_time
    if censor_type == "right":
        L = torch.minimum(event_time, censor_time)
        return L, L.masked_fill(censored, torch.inf)

    L = torch.where(censored, censor_time, torch.zeros_like(event_time))
    R = censor_time.masked_fill(censored, torch.inf)
    if censor_type == "current":
        return L, R

    # Gamma-Poisson visits with interval bracketing between inspections.
    indices = (~censored).nonzero(as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        return L, R
    events, censors = event_time[indices], censor_time[indices]
    counts = torch.poisson(torch.distributions.Gamma(
        event_time.new_tensor(3.0), event_time.new_tensor(1.0 / 3.0)
    ).sample((indices.numel(),))).long()
    maximum_count = int(counts.max())
    if maximum_count:
        valid = torch.arange(maximum_count, device=event_time.device)[None, :] < counts[:, None]
        visits = torch.rand(
            indices.numel(), maximum_count, device=event_time.device, dtype=event_time.dtype
        ) * censors[:, None]
        before = valid & (visits < events[:, None])
        after = valid & (visits > events[:, None])
        lower = visits.masked_fill(~before, -torch.inf).max(dim=1).values
        upper = visits.masked_fill(~after, torch.inf).min(dim=1).values
        L[indices] = torch.where(torch.isfinite(lower), lower, torch.zeros_like(lower))
        R[indices] = torch.where(torch.isfinite(upper), upper, censors)
    return L, R


def _validate(
    n: int, x_dim: int, mu: float, q: float, censor_type: str, scenario: str, r: float | None,
    multimodal_case: str = "separate", tau: float | None = None,
) -> tuple[float, float | None]:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if x_dim != 6:
        raise ValueError("x_dim must be 6 for this data-generating process")
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if censor_type not in {"exact", "right", "current", "interval"}:
        raise ValueError("censor_type must be 'exact', 'right', 'current', or 'interval'")
    radius = _radius(scenario, r)
    if scenario == "multimodal" and multimodal_case not in MULTIMODAL_SCALE_MULTIPLIER_CASES:
        raise ValueError(f"multimodal_case must be one of {tuple(MULTIMODAL_SCALE_MULTIPLIER_CASES)}")
    if tau is not None and (not math.isfinite(tau) or tau <= 0):
        raise ValueError("tau must be finite and positive")
    if censor_type != "exact" and (not math.isfinite(q) or not 0 < q < 1):
        raise ValueError("q must be in (0, 1) for censored data")
    tau = None if censor_type == "exact" else tau if tau is not None else censoring_horizon(
        scenario, q, mu=mu, r=radius, multimodal_case=multimodal_case
    )
    return radius, tau


def data_reg(
    n: int = 2000,
    x_dim: int = 6,
    mu: float = 0.0,
    q: float = 0.2,
    censor_type: str = "right",
    device: str | torch.device = "cpu",
    scenario: str = "ph",
    *,
    r: float | None = None,
    tau: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X, L, R, T, shape, scale) for PH, lognormal, or PO.

    ``shape`` is one for PH, sigma for lognormal, and the upper endpoint for
    bounded PO. ``scale`` is the exponential mean, lognormal median, or
    dimensionless PO odds scale, respectively. PO times are below
    ``PO_MAX_TIME * exp(mu)``; use ``survival_curves`` for their truth.
    ``mu`` multiplies all event times and the calibrated censoring horizon by exp(mu).
    Supply ``tau`` to reuse the actual censoring horizon saved during tuning.
    """
    if scenario == "multimodal":
        raise ValueError("use data_reg_multimodal for the mixture scenario")
    radius, tau = _validate(n, x_dim, mu, q, censor_type, scenario, r, tau=tau)
    x = 2.0 * torch.rand(n, x_dim, device=device) - 1.0
    shape, scale = _parameters(x, scenario, mu, radius)
    if scenario == "lognormal":
        event_time = torch.distributions.LogNormal(torch.log(scale), shape).sample()
    elif scenario == "po":
        # Map log-logistic odds onto finite support, without a clipping atom.
        u = torch.rand_like(scale).clamp_min(torch.finfo(scale.dtype).eps)
        odds_root = scale * ((1.0 - u) / u).pow(1.0 / PO_SHAPE)
        event_time = shape * (odds_root / (1.0 + odds_root))
    else:
        event_time = torch.distributions.Exponential(1.0 / scale).sample()
    L, R = _observe(event_time, censor_type, tau)
    return x, L, R, event_time, shape, scale


def data_reg_multimodal(
    n: int = 2000,
    x_dim: int = 6,
    mu: float = 0.0,
    q: float = 0.2,
    censor_type: str = "right",
    device: str | torch.device = "cpu",
    *,
    r: float = 0.7,
    multimodal_case: str = "separate",
    tau: float | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Return (X, L, R, T, weights, component_scales, latent_component).

    Each component is PH with predictor eta and a distinct baseline Weibull
    shape. The equal-weight mixture violates PH after marginalizing over Z.
    Censoring is independent of both X and Z.
    ``multimodal_case`` selects separate peaks (scales 1.0, 6.0) or closer
    peaks (scales 3.0, 6.0), with separate peaks as the default. Shapes 6 and
    8 give a density valley even after pooling covariates. Supply ``tau``
    to reuse the actual censoring horizon saved during tuning.
    """
    radius, tau = _validate(n, x_dim, mu, q, censor_type, "multimodal", r, multimodal_case, tau)
    x = 2.0 * torch.rand(n, x_dim, device=device) - 1.0
    shapes, scales = _parameters(x, "multimodal", mu, radius, multimodal_case)
    weights = x.new_tensor(MULTIMODAL_WEIGHTS).expand(n, -1)
    component = torch.distributions.Categorical(probs=weights).sample()
    selected_scale = scales.gather(1, component[:, None]).squeeze(1)
    selected_shape = shapes.gather(1, component[:, None]).squeeze(1)
    event_time = torch.distributions.Weibull(selected_scale, selected_shape).sample()
    L, R = _observe(event_time, censor_type, tau)
    return x, L, R, event_time, weights, scales, component
