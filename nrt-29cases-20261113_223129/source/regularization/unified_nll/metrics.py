"""The same pointwise grid SSE used by neural_only/model_mlp.ipynb."""

import torch


def average_grid_sse(estimated: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Sum squared differences over grid points, then average over subjects.

    Survival and CDF differences give the same score. There is no division by
    grid size, multiplication by spacing, integration, or horizon normalization.
    """
    if estimated.ndim != 2 or estimated.shape != truth.shape or 0 in estimated.shape:
        raise ValueError("estimated and truth must be matching nonempty subject-by-grid matrices")
    if not bool(torch.isfinite(estimated).all() & torch.isfinite(truth).all()):
        raise ValueError("curve values must be finite")
    return (estimated-truth).square().sum(dim=1).mean()
