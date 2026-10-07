"""Official-architecture DeepSurv implemented with PyTorch/CUDA."""

from .cuda_model import (
    OFFICIAL_DEEPSURV_COMMIT,
    OFFICIAL_MODEL_PARAMETERS,
    OFFICIAL_OPTIMIZATION_PARAMETERS,
    PAPER_GAUSSIAN_PARAMETERS,
    DeepSurvCUDA,
)
from .losses import cox_partial_log_likelihood

# Preserve the public benchmark name without maintaining a second backend.
DeepSurv = DeepSurvCUDA


def backend_available() -> bool:
    """Return whether the retained PyTorch backend can be imported."""
    return True

__all__ = [
    "DeepSurv",
    "DeepSurvCUDA",
    "OFFICIAL_DEEPSURV_COMMIT",
    "OFFICIAL_MODEL_PARAMETERS",
    "OFFICIAL_OPTIMIZATION_PARAMETERS",
    "PAPER_GAUSSIAN_PARAMETERS",
    "backend_available",
    "cox_partial_log_likelihood",
]
