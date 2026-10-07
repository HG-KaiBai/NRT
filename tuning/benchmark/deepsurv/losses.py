"""Cox partial-likelihood loss used by the PyTorch DeepSurv port.

The calculation follows ``runopti/stg/python/stg/losses.py``.  DeepSurv
orders observations by decreasing observed time before calling it, so the
cumulative sum at row ``i`` is the Cox risk set for that observation.
"""

from __future__ import annotations

import torch


def cox_partial_log_likelihood(
    logits: torch.Tensor,
    event: torch.Tensor,
    *,
    ties: str = "noties",
) -> torch.Tensor:
    """Return the average negative Cox partial log-likelihood.

    Args:
        logits: One-dimensional log-risk scores in decreasing-time order.
        event: One-dimensional event indicators aligned with ``logits``.
        ties: Tie method.  The cited DeepSurv and STG implementations only
            define the no-ties calculation, so other values are rejected.

    ``logcumsumexp`` is the numerically stable PyTorch equivalent of STG's
    ``log(cumsum(exp(logits)))`` calculation.
    """
    if logits.ndim != 1 or event.ndim != 1 or logits.shape != event.shape:
        raise ValueError("logits and event must be aligned one-dimensional tensors")
    if ties != "noties":
        raise NotImplementedError(
            "the DeepSurv/STG Cox loss supports only ties='noties'"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")

    event = event.to(device=logits.device, dtype=logits.dtype)
    observations = event.sum()
    if not bool(observations > 0):
        raise ValueError("the Cox loss requires at least one observed event")

    log_risk = torch.logcumsumexp(logits, dim=0)
    return -((logits - log_risk) * event).sum() / observations
