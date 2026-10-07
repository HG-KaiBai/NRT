"""Shared data generation and evaluation for paper benchmark methods."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

import torch

from ..testdata2 import (
    MULTIMODAL_SCALE_MULTIPLIER_CASES,
    MULTIMODAL_SHAPES,
    censoring_horizon,
    survival_curves,
    data_reg,
    data_reg_multimodal,
)


class SurvivalEstimator(Protocol):
    supported_censor_types: frozenset[str]

    def fit(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        *,
        censor_type: str,
    ) -> "SurvivalEstimator": ...

    def predict_survival(
        self, x: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    scenario: str
    censor_type: str
    n_samples: int
    n_train: int
    n_test: int
    censor_q: float
    seed: int
    grid_size: int
    max_bound: float
    average_grid_sse: float
    tau: float | None = None
    right_censoring_rate: float = 0.0

    def save_json(
        self,
        path: str | Path,
        *,
        modeling_parameters: Mapping[str, object] | None = None,
        tuning_parameters: Mapping[str, object] | None = None,
    ) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        if modeling_parameters is not None:
            payload["modeling_parameters"] = dict(modeling_parameters)
        if tuning_parameters is not None:
            payload["tuning_parameters"] = dict(tuning_parameters)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def average_grid_sse(
    estimated_survival: torch.Tensor,
    shape: torch.Tensor,
    scale: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Return the notebook's mean per-subject survival-grid SSE.

    This deliberately uses a sum over grid points (not a numerical integral),
    followed by a mean over test observations.
    """
    if estimated_survival.shape != (shape.numel(), grid.numel()):
        raise ValueError(
            "estimated_survival must have shape (number of rows, number of grid points)"
        )
    shape = shape.to(device=estimated_survival.device, dtype=estimated_survival.dtype)
    scale = scale.to(device=estimated_survival.device, dtype=estimated_survival.dtype)
    grid = grid.to(device=estimated_survival.device, dtype=estimated_survival.dtype)
    true_survival = torch.exp(
        -((grid.unsqueeze(0) / scale.unsqueeze(1)) ** shape.unsqueeze(1))
    )
    return (estimated_survival - true_survival).square().sum(dim=1).mean()


def lognormal_survival(
    shape: torch.Tensor,
    scale: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Return lognormal survival curves parameterized by sigma and median."""
    if shape.ndim != 1 or scale.shape != shape.shape:
        raise ValueError("shape and scale must be one-dimensional tensors of equal size")
    if bool((shape <= 0.0).any()) or bool((scale <= 0.0).any()):
        raise ValueError("shape and scale must be positive")
    if bool((grid < 0.0).any()):
        raise ValueError("grid times must be nonnegative")
    grid = grid.to(device=scale.device, dtype=scale.dtype)
    standardized_log_time = (
        torch.log(grid.unsqueeze(0)) - torch.log(scale.unsqueeze(1))
    ) / (shape.unsqueeze(1) * (2.0**0.5))
    return 0.5 * torch.erfc(standardized_log_time)


def multimodal_survival(
    mixture_weights: torch.Tensor,
    component_scales: torch.Tensor,
    grid: torch.Tensor,
    *,
    shape: float | None = None,
) -> torch.Tensor:
    """Conditional survival, marginalizing over the unobserved component."""
    if mixture_weights.shape != component_scales.shape or mixture_weights.ndim != 2:
        raise ValueError("weights and scales must have the same (rows, components) shape")
    grid = grid.to(device=component_scales.device, dtype=component_scales.dtype)
    shapes = component_scales.new_tensor(MULTIMODAL_SHAPES)[None, :, None] if shape is None else shape
    component_survival = torch.exp(
        -((grid[None, None, :] / component_scales[:, :, None]) ** shapes)
    )
    return (mixture_weights[:, :, None] * component_survival).sum(dim=1)


def benchmark_model(
    model: SurvivalEstimator,
    *,
    method: str,
    scenario: str = "ph",
    censor_type: str,
    n_samples: int = 512,
    x_dim: int = 6,
    mu: float = 0.0,
    censor_q: float = 0.2,
    seed: int = 123,
    grid_size: int = 256,
    device: str | torch.device = "cpu",
    multimodal_case: str = "separate",
    tau: float | None = None,
) -> BenchmarkResult:
    """Fit one method and score it exactly as ``model_mlp.ipynb`` does."""
    if censor_type not in model.supported_censor_types:
        supported = ", ".join(sorted(model.supported_censor_types))
        raise ValueError(
            f"{method} does not support censor_type={censor_type!r}; supported: {supported}"
        )
    if n_samples < 2:
        raise ValueError("n_samples must provide at least one training and test row")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    set_seed(seed)
    if censor_type != "exact" and tau is None:
        tau = censoring_horizon(scenario, censor_q, mu=mu, multimodal_case=multimodal_case)
    data_kwargs = dict(
        n=n_samples, x_dim=x_dim, mu=mu, q=censor_q,
        censor_type=censor_type, device="cpu", tau=tau,
    )
    if scenario == "multimodal":
        x, L, R, _event_time, weights, scales, _component = data_reg_multimodal(
            **data_kwargs, multimodal_case=multimodal_case
        )
    else:
        x, L, R, _event_time, shape, scale = data_reg(
            **data_kwargs, scenario=scenario
        )
    n_train = int(0.8 * n_samples)
    if n_train < 1 or n_train == n_samples:
        raise ValueError("n_samples must provide at least one training and test row")

    x_train, x_test = x[:n_train], x[n_train:]
    L_train, R_train = L[:n_train], R[:n_train]
    finite_bounds = torch.cat(
        (L_train[torch.isfinite(L_train)], R_train[torch.isfinite(R_train)])
    )
    max_obs = float(finite_bounds.max().item())
    max_bound = 1.1 * max_obs
    grid = torch.linspace(0.0, max_obs, grid_size, device=x.device, dtype=x.dtype)

    model.fit(x_train, L_train, R_train, censor_type=censor_type)
    estimated = model.predict_survival(x_test, grid)
    if scenario == "multimodal":
        truth = multimodal_survival(weights[n_train:], scales[n_train:], grid)
        if estimated.shape != truth.shape:
            raise ValueError("estimated_survival must match the multimodal truth grid")
        score = (estimated - truth.to(device=estimated.device)).square().sum(dim=1).mean()
    elif scenario == "lognormal":
        truth = lognormal_survival(shape[n_train:], scale[n_train:], grid)
        if estimated.shape != truth.shape:
            raise ValueError("estimated_survival must match the lognormal truth grid")
        score = (estimated - truth.to(device=estimated.device)).square().sum(dim=1).mean()
    else:
        truth = survival_curves(shape[n_train:], scale[n_train:], grid, scenario=scenario)
        score = (estimated - truth.to(estimated)).square().sum(dim=1).mean()
    return BenchmarkResult(
        method=method,
        scenario=scenario,
        censor_type=censor_type,
        n_samples=n_samples,
        n_train=n_train,
        n_test=n_samples - n_train,
        censor_q=censor_q,
        seed=seed,
        grid_size=grid_size,
        max_bound=max_bound,
        average_grid_sse=float(score.item()),
        tau=tau, right_censoring_rate=float(torch.isinf(R).float().mean()),
    )
