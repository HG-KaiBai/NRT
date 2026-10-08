"""Deterministic neural conditional tree model."""

from __future__ import annotations

from copy import deepcopy
from typing import NamedTuple

import torch
from torch import nn

try:  # Support package imports and direct script execution.
    from .networks import MLP, init_linear
    from .tree import Tree
except ImportError:  # pragma: no cover
    from networks import MLP, init_linear
    from tree import Tree

EPS = 1e-6


class TreeParameters(NamedTuple):
    split_props: torch.Tensor
    left_probs: torch.Tensor


class ConditionalTree(nn.Module):
    """An ordinary neural network paired with a tree.

    The topology is fixed. The network directly estimates split locations
    and left-branch probabilities, producing point survival estimates.
    """

    def __init__(
        self,
        depth: int,
        x_dim: int,
        h_dim: int = 128,
        mlp_hidden: int = 256,
        mlp_layers: int = 2,
        dropout: float = 0.0,
        split_min: float = 0.03,
        max_bound: float = 1.0,
        max_observation: float | None = None,
        shared_layers: int = 0,
    ) -> None:
        super().__init__()
        if not 0.0 <= split_min < 0.5:
            raise ValueError("split_min must be in [0, 0.5)")
        if h_dim < 1:
            raise ValueError("h_dim must be at least 1")
        if isinstance(shared_layers, bool) or not isinstance(shared_layers, int) or shared_layers < 0:
            raise ValueError("shared_layers must be a nonnegative integer")

        self.depth = int(depth)
        self.x_dim = int(x_dim)
        self.h_dim = int(h_dim)
        self.shared_layers = shared_layers
        self.split_min = float(split_min)
        self.tree = Tree(depth=depth, max_bound=max_bound)
        self.max_observation = (
            float(max_bound) if max_observation is None else float(max_observation)
        )
        if not 0.0 < self.max_observation <= max_bound:
            raise ValueError("max_observation must be positive and at most max_bound")
        shared: list[nn.Module] = []
        encoder_in_dim = x_dim
        for _ in range(shared_layers):
            shared.extend((nn.Linear(encoder_in_dim, mlp_hidden), nn.SiLU()))
            if dropout > 0.0:
                shared.append(nn.Dropout(dropout))
            encoder_in_dim = mlp_hidden
        # if shared_layers > 0:
        #     shared.append(nn.LayerNorm(mlp_hidden))
        self.shared_encoder = nn.Sequential(*shared)
        # self.shared_encoder.apply(init_linear)
        self.encoder = MLP(
            encoder_in_dim,
            h_dim,
            hidden_dim=mlp_hidden,
            n_layers=mlp_layers,
            dropout=dropout,
        )
        # Private branches start from identical features. Optional shared
        # layers receive gradients from either head during alternating updates.
        self.encoder_split = deepcopy(self.encoder)
        self.head_m = nn.Linear(h_dim, self.tree.n_internal)
        self.head_split = nn.Linear(h_dim, self.tree.n_internal)
        # for head in (self.head_m, self.head_split):
        #     init_linear(head)

    def forward(self, x: torch.Tensor) -> TreeParameters:
        if x.ndim != 2 or x.size(1) != self.x_dim:
            raise ValueError(f"x must have shape (batch, {self.x_dim}); got {tuple(x.shape)}")
        shared_hidden = self.shared_encoder(x)
        hidden = self.encoder(shared_hidden)
        left_probs = torch.sigmoid(self.head_m(hidden)).clamp(EPS, 1.0 - EPS)
        split_props = self.split_min + (1.0 - 2.0 * self.split_min) * torch.sigmoid(
            self.head_split(self.encoder_split(shared_hidden))
        )
        return TreeParameters(split_props, left_probs)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # Older checkpoints used one encoder for both heads.
        if not any(key.startswith(prefix + "encoder_split.") for key in state_dict):
            for key, value in list(state_dict.items()):
                if key.startswith(prefix + "encoder."):
                    suffix = key[len(prefix + "encoder.") :]
                    state_dict[prefix + "encoder_split." + suffix] = value.clone()
        super()._load_from_state_dict(state_dict, prefix, *args)

    def loglik(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        *,
        include_point_density: bool = True,
    ) -> torch.Tensor:
        params = self(x)
        return self.tree.loglik(
            L,
            R,
            params.split_props,
            params.left_probs,
            include_point_density=include_point_density,
        )

    def mass_nll(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        """NLL estimating loss for ``m`` with the structure ``rho`` fixed.

        Exact observations contribute their terminal-node mass, without the
        terminal-node width.  The omitted width is constant in this step
        because the cut proportions are detached.
        """
        params = self(x)
        log_likelihood = self.tree.loglik(
            L,
            R,
            params.split_props.detach(),
            params.left_probs,
            include_point_density=False,
        )
        return -log_likelihood.reshape(x.size(0), -1).mean(dim=1).mean()

    def structure_brier_loss(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        grid_size: int = 256,
    ) -> torch.Tensor:
        """Expected integrated Brier estimating loss for ``rho`` with ``m`` fixed.

        For censored observations, the unavailable binary-survival target is
        replaced by its conditional expectation under the current induced
        distribution and detached before differentiation.
        """
        return self._brier_loss(
            x,
            L,
            R,
            grid_size=grid_size,
            detach_target=True,
            detach_left_probs=True,
        )

    def nll(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        update: str = "both",
    ) -> torch.Tensor:
        """Censoring-aware NLL, including leaf density for exact observations."""
        if update not in ("both", "m", "rho"):
            raise ValueError("update must be 'both', 'm', or 'rho'")
        params = self(x)
        log_likelihood = self.tree.loglik(
            L, R,
            params.split_props.detach() if update == "m" else params.split_props,
            params.left_probs.detach() if update == "rho" else params.left_probs,
            include_point_density=True,
        )
        return -log_likelihood.reshape(x.size(0), -1).mean(dim=1).mean()

    def brier_loss(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        grid_size: int = 256,
        update: str = "both",
    ) -> torch.Tensor:
        """Expected integrated Brier loss; detach censored conditional targets."""
        if update not in ("both", "m", "rho"):
            raise ValueError("update must be 'both', 'm', or 'rho'")
        return self._brier_loss(
            x, L, R, grid_size=grid_size, detach_target=True,
            detach_left_probs=update == "rho", detach_split_props=update == "m",
        )

    def objective(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        loss: str = "brier",
        grid_size: int = 256,
        update: str = "both",
    ) -> torch.Tensor:
        """Update both parameters jointly, or just the selected component."""
        if update not in ("both", "m", "rho"):
            raise ValueError("update must be 'both', 'm', or 'rho'")
        if loss == "nll":
            if update == "m":
                return self.mass_nll(x, L, R)
            return self.nll(x, L, R, update=update)
        if loss != "brier":
            raise ValueError("loss must be 'nll' or 'brier'")
        return self.brier_loss(x, L, R, grid_size=grid_size, update=update)

    @torch.no_grad()
    def predict_survival(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        prediction_batch_size: int | None = 32,
    ) -> torch.Tensor:
        """Return one estimated survival curve per observation."""
        if x.ndim != 2 or x.size(1) != self.x_dim:
            raise ValueError(f"x must have shape (batch, {self.x_dim}); got {tuple(x.shape)}")
        if times.ndim != 1:
            raise ValueError("times must be a one-dimensional common grid")
        if prediction_batch_size is not None and prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be at least 1 or None")
        if x.size(0) < 1:
            raise ValueError("x must contain at least one observation")

        times = times.to(device=x.device, dtype=x.dtype)
        batch_size = prediction_batch_size or x.size(0)
        predictions = []
        for batch_start in range(0, x.size(0), batch_size):
            x_batch = x[batch_start : batch_start + batch_size]
            params = self(x_batch)
            masses = self.tree.leaf_masses(params.left_probs)
            predictions.append(
                (1.0 - self.tree.cdf(times, params.split_props, masses)).clamp(0.0, 1.0)
            )
        return torch.cat(predictions, dim=0)

    @torch.no_grad()
    def brier_score(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        grid_size: int = 256,
        prediction_batch_size: int | None = 32,
    ) -> torch.Tensor:
        """Integrated Brier score from the estimated survival curves."""
        if prediction_batch_size is not None and prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be at least 1 or None")
        if x.size(0) < 1:
            raise ValueError("x must contain at least one observation")

        batch_size = prediction_batch_size or x.size(0)
        total = x.new_zeros(())
        for batch_start in range(0, x.size(0), batch_size):
            batch_end = min(batch_start + batch_size, x.size(0))
            total = total + self._brier_loss(
                x[batch_start:batch_end],
                L[batch_start:batch_end],
                R[batch_start:batch_end],
                grid_size=grid_size,
                detach_target=False,
                detach_left_probs=False,
            ) * (batch_end - batch_start)
        return total / x.size(0)

    def _brier_loss(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        grid_size: int,
        detach_target: bool,
        detach_left_probs: bool,
        detach_split_props: bool = False,
    ) -> torch.Tensor:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if L.shape != R.shape or L.ndim < 1 or L.size(0) != x.size(0):
            raise ValueError("L and R must have matching shapes and batch dimension")

        grid = torch.linspace(
            0.0,
            self.max_observation,
            steps=grid_size,
            device=x.device,
            dtype=x.dtype,
        )
        flat_L = L.reshape(x.size(0), -1)
        flat_R = R.reshape(x.size(0), -1)
        bounded_L = torch.nan_to_num(
            flat_L, nan=0.0, posinf=self.tree.max_bound, neginf=0.0
        ).clamp(0.0, self.tree.max_bound)
        bounded_R = torch.nan_to_num(
            flat_R,
            nan=self.tree.max_bound,
            posinf=self.tree.max_bound,
            neginf=0.0,
        ).clamp(0.0, self.tree.max_bound)
        params = self(x)
        split_props = (
            params.split_props.detach() if detach_split_props else params.split_props
        )
        left_probs = (
            params.left_probs.detach() if detach_left_probs else params.left_probs
        )
        masses = self.tree.leaf_masses(left_probs)
        leaf_lower, leaf_upper = self.tree.leaf_intervals(split_props)
        survival_grid = (
            1.0 - self.tree._cdf_from_intervals(
                grid.unsqueeze(0).expand(x.size(0), -1),
                leaf_lower, leaf_upper, masses,
            )
        ).clamp(0.0, 1.0)
        # Censored targets are detached during training, so their observation
        # CDFs do not need an autograd graph. Compute both bounds together.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not detach_target):
            survival_bounds = (
                1.0 - self.tree._cdf_from_intervals(
                    torch.cat((bounded_L, bounded_R), dim=1),
                    leaf_lower, leaf_upper, masses,
                )
            ).clamp(0.0, 1.0)
        survival_L, survival_R = (
            values.reshape_as(L) for values in survival_bounds.chunk(2, dim=1)
        )

        survival_grid = survival_grid.view(
            x.size(0), *([1] * (L.ndim - 1)), grid_size
        ).expand(*L.shape, grid_size)
        time_grid = grid.view(*([1] * L.ndim), grid_size)
        L_grid = L.unsqueeze(-1)
        R_grid = R.unsqueeze(-1)
        interval_mass = (survival_L - survival_R).clamp(min=EPS)
        target_probability = (
            survival_grid - survival_R.unsqueeze(-1)
        ) / interval_mass.unsqueeze(-1)
        target_probability = torch.where(
            time_grid <= L_grid, torch.ones_like(target_probability), target_probability
        )
        target_probability = torch.where(
            torch.isfinite(R_grid) & (time_grid >= R_grid),
            torch.zeros_like(target_probability),
            target_probability,
        )
        is_point = torch.isfinite(L) & torch.isfinite(R) & torch.isclose(L, R)
        target_probability = torch.where(
            is_point.unsqueeze(-1),
            (L_grid > time_grid).to(L.dtype),
            target_probability,
        ).clamp(0.0, 1.0)
        if detach_target:
            target_probability = target_probability.detach()
        brier_grid = (
            target_probability
            - 2.0 * target_probability * survival_grid
            + survival_grid.square()
        )
        return torch.trapz(brier_grid, grid, dim=-1).mean()
