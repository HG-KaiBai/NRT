"""Cox proportional-hazards benchmark."""

from .model import CoxPH, r_backend_available

__all__ = ["CoxPH", "r_backend_available"]
