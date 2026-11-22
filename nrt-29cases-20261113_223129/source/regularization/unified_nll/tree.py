"""Body-tree size constraints and direct survival integration."""
import torch
from torch import nn


class Tree(nn.Module):
    """Shared body geometry; raw-time splitting is defined by RawGeometry."""

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

    @staticmethod
    def survival(state, times: torch.Tensor) -> torch.Tensor:
        """Direct upper-tail integration avoids subtracting a CDF close to one."""
        fractions = ((state.upper[:, None, :] - times[None, :, None])
                     / state.width[:, None, :]).clamp(0, 1)
        return (fractions * state.log_mass.exp()[:, None, :]).sum(dim=-1)
