"""Neural mass and split heads; no imports from the neural_only package."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import nn

from .tree import Tree, TreeState


@dataclass(frozen=True)
class ModelConfig:
    depth: int = 3
    x_dim: int = 6
    h_dim: int = 32
    mlp_hidden: int = 32
    mlp_layers: int = 3
    split_min: float = 0.1
    min_leaf_width: float = 1e-4
    learn_splits: bool = True


class Observations(NamedTuple):
    x: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    exact: torch.Tensor
    log_jacobian: torch.Tensor


def _encoder(config: ModelConfig) -> nn.Sequential:
    if min(config.x_dim, config.h_dim, config.mlp_hidden, config.mlp_layers) < 1:
        raise ValueError("network dimensions and layer count must be positive")
    layers: list[nn.Module] = []
    width = config.x_dim
    for _ in range(config.mlp_layers - 1):
        layers.extend((nn.Linear(width, config.mlp_hidden), nn.SiLU()))
        width = config.mlp_hidden
    layers.append(nn.Linear(width, config.h_dim))
    return nn.Sequential(*layers)


class ConditionalTree(nn.Module):
    """A conditional tree on u=t/(t+s), covering the whole nonnegative time axis.

    The time scale s is estimated once from observable training bounds.
    Exact-time likelihoods include du/dt; censored likelihoods are interval
    probabilities and are invariant to this monotone transformation.
    """

    def __init__(self, config: ModelConfig, time_scale: float) -> None:
        super().__init__()
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if not 0 <= config.split_min < 0.5:
            raise ValueError("split_min must be in [0, 0.5)")
        self.config = config
        self.tree = Tree(config.depth, config.min_leaf_width)
        self.register_buffer("time_scale", torch.tensor(float(time_scale)))
        self.mass_encoder = _encoder(config)
        self.split_encoder = deepcopy(self.mass_encoder)
        self.mass_head = nn.Linear(config.h_dim, self.tree.n_internal)
        self.split_head = nn.Linear(config.h_dim, self.tree.n_internal)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # All methods start with the same uniform distribution on transformed
        # time. Seeds affect hidden features, not arbitrary initial partitions.
        nn.init.zeros_(self.mass_head.weight)
        nn.init.zeros_(self.split_head.weight)
        if not config.learn_splits:
            self.split_encoder.requires_grad_(False)
            self.split_head.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> TreeState:
        logits = self.mass_head(self.mass_encoder(x))
        if self.config.learn_splits:
            splits = self.config.split_min + (1 - 2 * self.config.split_min) * torch.sigmoid(
                self.split_head(self.split_encoder(x))
            )
        else:
            splits = torch.full_like(logits, 0.5)
        return self.tree(logits, splits)

    def prepare(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> Observations:
        """Validate original bounds, then transform them once outside the training loop."""
        device = self.time_scale.device
        dtype = self.time_scale.dtype
        x, left, right = (v.to(device=device, dtype=dtype) for v in (x, left, right))
        if x.ndim != 2 or x.shape[1] != self.config.x_dim or len(x) == 0:
            raise ValueError("x must be a nonempty matrix with config.x_dim columns")
        if left.shape != (len(x),) or right.shape != left.shape:
            raise ValueError("observation bounds must be vectors matching x")
        valid = (torch.isfinite(x).all() & torch.isfinite(left).all()
                 & ~torch.isnan(right).any() & (left >= 0).all()
                 & (right >= left).all() & ~torch.isneginf(right).any())
        if not bool(valid):
            raise ValueError("bounds must satisfy 0 <= L <= R <= infinity; x must be finite")
        exact = left == right
        scale = self.time_scale
        safe_right = torch.where(torch.isfinite(right), right, 0)
        transformed_left = left / (left + scale)
        transformed_right = torch.where(
            torch.isfinite(right), safe_right / (safe_right + scale), 1.0,
        )
        log_jacobian = scale.log() - 2 * (left + scale).log()
        return Observations(x, transformed_left, transformed_right, exact, log_jacobian)

    def terms(self, observations: Observations) -> tuple[torch.Tensor, torch.Tensor]:
        state = self(observations.x)
        nll = -self.tree.loglik(state, *observations[1:]).mean()
        return nll, self.tree.regularizer(state).mean()

    def nll(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.terms(self.prepare(x, left, right))[0]

    @torch.no_grad()
    def predict_survival(self, x: torch.Tensor, times: torch.Tensor,
                         batch_size: int = 256) -> torch.Tensor:
        if times.ndim != 1 or bool(torch.isnan(times).any() | (times < 0).any()):
            raise ValueError("times must be a nonnegative vector")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.eval()
        x = x.to(self.time_scale)
        times = times.to(self.time_scale)
        safe_times = torch.where(torch.isfinite(times), times, 0)
        transformed = torch.where(torch.isfinite(times),
                                  safe_times / (safe_times + self.time_scale), 1.0)
        return torch.cat([self.tree.survival(self(batch), transformed)
                          for batch in x.split(batch_size)])
