"""Binary-tree interval geometry and likelihood calculations."""

from __future__ import annotations

import torch
from torch import nn

EPS = 1e-8


class _PositiveProduct(torch.autograd.Function):
    """Product of strictly positive factors without a backward host sync.

    PyTorch 2.10's general prod backward checks for zeros on the CPU. Our
    factors are clamped above zero, so the derivative is simply product/x.
    """

    @staticmethod
    def forward(ctx, factors):
        product = factors.prod(dim=-1)
        ctx.save_for_backward(factors, product)
        return product

    @staticmethod
    def backward(ctx, grad):
        factors, product = ctx.saved_tensors
        return grad.unsqueeze(-1) * (product.unsqueeze(-1) / factors)


class Tree(nn.Module):
    """Complete binary partition of ``[0, max_bound]`` with leaf masses."""

    def __init__(self, depth: int, max_bound: float = 1.0) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if max_bound <= 0.0:
            raise ValueError("max_bound must be positive")

        self.depth = int(depth)
        self.n_internal = 2**depth - 1
        self.n_leaves = 2**depth
        self.max_bound = float(max_bound)
        paths, directions = self._leaf_paths(depth)
        self.register_buffer("leaf_paths", paths)
        self.register_buffer("leaf_directions", directions)

    @staticmethod
    def _leaf_paths(depth: int) -> tuple[torch.Tensor, torch.Tensor]:
        paths = torch.zeros((2**depth, depth), dtype=torch.long)
        directions = torch.zeros((2**depth, depth), dtype=torch.float32)
        for leaf_index in range(2**depth):
            node_index = 0
            for level in range(depth):
                goes_right = (leaf_index >> (depth - level - 1)) & 1
                paths[leaf_index, level] = node_index
                directions[leaf_index, level] = float(not goes_right)
                node_index = 2 * node_index + 1 + goes_right
        return paths, directions

    def _check_node_values(self, values: torch.Tensor, name: str) -> None:
        if values.ndim != 2 or values.size(1) != self.n_internal:
            raise ValueError(
                f"{name} must have shape (batch, {self.n_internal}); "
                f"got {tuple(values.shape)}"
            )

    def leaf_intervals(self, split_props: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower and upper bounds for every leaf."""
        self._check_node_values(split_props, "split_props")
        batch_size = split_props.size(0)
        lower = split_props.new_zeros((batch_size, 1))
        upper = split_props.new_full((batch_size, 1), self.max_bound)
        # Process an entire level at once, rather than launching operations
        # separately for every node. Child columns stay in left-to-right order.
        for level in range(self.depth):
            start = 2**level - 1
            splits = split_props[:, start : start + 2**level]
            cut = lower + splits * (upper - lower)
            lower = torch.stack((lower, cut), dim=-1).flatten(1)
            upper = torch.stack((cut, upper), dim=-1).flatten(1)
        return lower, upper

    def leaf_masses(self, left_probs: torch.Tensor) -> torch.Tensor:
        """Map internal-node left-branch probabilities to terminal leaf masses."""
        self._check_node_values(left_probs, "left_probs")
        batch_size = left_probs.size(0)
        path_index = self.leaf_paths.unsqueeze(0).expand(batch_size, -1, -1)
        probabilities = torch.gather(
            left_probs.unsqueeze(1).expand(-1, self.n_leaves, -1),
            2,
            path_index,
        )
        goes_left = self.leaf_directions.to(left_probs).unsqueeze(0)
        factors = probabilities * goes_left + (1.0 - probabilities) * (1.0 - goes_left)
        masses = _PositiveProduct.apply(factors.clamp(min=EPS))
        return masses / masses.sum(dim=1, keepdim=True).clamp(min=EPS)

    def _observation_fractions(
        self,
        L: torch.Tensor,
        R: torch.Tensor,
        leaf_lower: torch.Tensor,
        leaf_upper: torch.Tensor,
        include_point_density: bool = True,
    ) -> torch.Tensor:
        if L.shape != R.shape:
            raise ValueError("L and R must have the same shape")
        if L.ndim < 1 or L.size(0) != leaf_lower.size(0):
            raise ValueError("the first dimension of L and R must be the batch dimension")

        batch_size = leaf_lower.size(0)
        extra_dims = L.ndim - 1
        expanded_shape = (batch_size, *([1] * extra_dims), self.n_leaves)
        lower = leaf_lower.view(expanded_shape)
        upper = leaf_upper.view(expanded_shape)
        left = L.unsqueeze(-1)
        right = R.unsqueeze(-1)
        leaf_width = (upper - lower).clamp(min=EPS)
        overlap = (torch.minimum(upper, right) - torch.maximum(lower, left)).clamp(min=0.0)
        interval_fraction = overlap / leaf_width

        is_point = torch.isclose(left, right)
        in_leaf = (left >= lower) & (left < upper)
        in_last_leaf = (
            torch.isclose(left, left.new_full((), self.max_bound))
            & torch.isclose(upper, upper.new_full((), self.max_bound))
            & (left >= lower)
        )
        point_contribution = (in_leaf | in_last_leaf).to(leaf_width.dtype)
        if include_point_density:
            point_contribution = point_contribution / leaf_width
        return torch.where(
            is_point.expand_as(interval_fraction),
            point_contribution,
            interval_fraction,
        )

    def loglik(
        self,
        L: torch.Tensor,
        R: torch.Tensor,
        split_props: torch.Tensor,
        left_probs: torch.Tensor,
        include_point_density: bool = True,
    ) -> torch.Tensor:
        leaf_lower, leaf_upper = self.leaf_intervals(split_props)
        fractions = self._observation_fractions(
            L,
            R,
            leaf_lower,
            leaf_upper,
            include_point_density=include_point_density,
        )
        masses = self.leaf_masses(left_probs)
        masses = masses.view(masses.size(0), *([1] * (L.ndim - 1)), self.n_leaves)
        return (masses * fractions).sum(dim=-1).clamp(min=EPS).log()

    def cdf(
        self,
        times: torch.Tensor,
        split_props: torch.Tensor,
        leaf_masses: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the piecewise-uniform leaf distribution on a common time grid."""
        if times.ndim != 1:
            raise ValueError("times must be a one-dimensional common grid")
        leaf_lower, leaf_upper = self.leaf_intervals(split_props)
        return self._cdf_from_intervals(
            times.unsqueeze(0).expand(split_props.size(0), -1),
            leaf_lower, leaf_upper, leaf_masses,
        )

    @staticmethod
    def _cdf_from_intervals(
        times: torch.Tensor,
        leaf_lower: torch.Tensor,
        leaf_upper: torch.Tensor,
        leaf_masses: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate paired times using already-computed leaf geometry."""
        leaf_width = (leaf_upper - leaf_lower).clamp(min=EPS)
        flat_times = times.reshape(times.size(0), -1)
        fraction = (
            (flat_times.unsqueeze(-1) - leaf_lower.unsqueeze(1))
            / leaf_width.unsqueeze(1)
        ).clamp(0.0, 1.0)
        values = (leaf_masses.unsqueeze(1) * fraction).sum(dim=-1)
        return values.reshape_as(times)

    def cdf_at_observation_times(
        self,
        times: torch.Tensor,
        split_props: torch.Tensor,
        leaf_masses: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate each observation's CDF at its own time values.

        Unlike :meth:`cdf`, this method does not construct the Cartesian
        product of every observation and every supplied time.  The first
        dimension of ``times`` is paired with the first dimension of the
        tree parameters, which keeps differentiable Brier training tractable.
        """
        if times.ndim < 1 or times.size(0) != split_props.size(0):
            raise ValueError(
                "times must have the same first (batch) dimension as split_props"
            )
        if leaf_masses.shape != (split_props.size(0), self.n_leaves):
            raise ValueError(
                f"leaf_masses must have shape (batch, {self.n_leaves}); "
                f"got {tuple(leaf_masses.shape)}"
            )

        leaf_lower, leaf_upper = self.leaf_intervals(split_props)
        return self._cdf_from_intervals(times, leaf_lower, leaf_upper, leaf_masses)
