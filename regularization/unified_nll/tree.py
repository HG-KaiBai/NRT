"""Normalized tree geometry and a common likelihood for every observation type."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class TreeState(NamedTuple):
    lower: torch.Tensor
    upper: torch.Tensor
    log_mass: torch.Tensor

    @property
    def width(self) -> torch.Tensor:
        return self.upper - self.lower


class Tree(nn.Module):
    """Piecewise-uniform distribution on [0, 1] with genuinely bounded leaf widths.

    Each split reserves enough space for *all* descendant leaves. This is
    stronger than just bounding the fraction at each individual node.
    """

    def __init__(self, depth: int, min_leaf_width: float = 1e-4) -> None:
        super().__init__()
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if not 0 <= min_leaf_width < 1 / 2**depth:
            raise ValueError("min_leaf_width must be in [0, 1 / 2**depth)")
        self.depth = depth
        self.n_internal = 2**depth - 1
        self.n_leaves = 2**depth
        self.min_leaf_width = float(min_leaf_width)

    def forward(self, mass_logits: torch.Tensor, split_props: torch.Tensor) -> TreeState:
        if mass_logits.shape != split_props.shape or mass_logits.ndim != 2:
            raise ValueError("mass_logits and split_props must be matching matrices")
        if mass_logits.shape[1] != self.n_internal:
            raise ValueError("the node count does not match the tree depth")
        lower = mass_logits.new_zeros((len(mass_logits), 1))
        upper = mass_logits.new_ones((len(mass_logits), 1))
        log_mass = mass_logits.new_zeros((len(mass_logits), 1))
        for level in range(self.depth):
            start = 2**level - 1
            stop = start + 2**level
            logits = mass_logits[:, start:stop]
            reserved = 2 ** (self.depth - level - 1) * self.min_leaf_width
            cut = lower + reserved + (upper - lower - 2 * reserved) * split_props[:, start:stop]
            lower = torch.stack((lower, cut), dim=-1).flatten(1)
            upper = torch.stack((cut, upper), dim=-1).flatten(1)
            log_mass = torch.stack(
                (log_mass + F.logsigmoid(logits), log_mass + F.logsigmoid(-logits)),
                dim=-1,
            ).flatten(1)
        return TreeState(lower, upper, log_mass)

    @staticmethod
    def loglik(state: TreeState, left: torch.Tensor, right: torch.Tensor,
               exact: torch.Tensor, log_jacobian: torch.Tensor) -> torch.Tensor:
        """Density for exact times, interval mass for every censored observation.

        ``left`` and ``right`` are transformed bounds in [0, 1]. An exact
        observation is identified by equality in the *original* time scale.
        CDF subtraction and probability clipping are deliberately avoided.
        """
        width = state.width
        log_width = width.log()
        point_in_leaf = (left[:, None] >= state.lower) & (left[:, None] < state.upper)
        # Finite very large times can round to 1 in the transformed scale.
        point_in_leaf = point_in_leaf | ((left[:, None] == 1) & (state.upper == 1))
        overlap = (torch.minimum(state.upper, right[:, None])
                   - torch.maximum(state.lower, left[:, None])).clamp_min(0)
        tiny = torch.finfo(width.dtype).tiny
        interval_log_fraction = torch.where(
            overlap > 0, overlap.clamp_min(tiny).log() - log_width, -torch.inf,
        )
        point_log_density = torch.where(point_in_leaf, -log_width, -torch.inf)
        log_factor = torch.where(exact[:, None], point_log_density, interval_log_fraction)
        log_likelihood = torch.logsumexp(state.log_mass + log_factor, dim=-1)
        return log_likelihood + torch.where(exact, log_jacobian, 0.0)

    @staticmethod
    def regularizer(state: TreeState) -> torch.Tensor:
        """KL(f_U || Uniform[0,1]), equivalent to mass-weighted node KLs.

        The geometric branch proportions here are the effective proportions
        *after* reserving the minimum leaf widths.
        """
        return (state.log_mass.exp() * (state.log_mass - state.width.log())).sum(dim=-1)

    @staticmethod
    def survival(state: TreeState, times: torch.Tensor) -> torch.Tensor:
        """Direct upper-tail integration avoids subtracting a CDF close to one."""
        fractions = ((state.upper[:, None, :] - times[None, :, None])
                     / state.width[:, None, :]).clamp(0, 1)
        return (fractions * state.log_mass.exp()[:, None, :]).sum(dim=-1)
