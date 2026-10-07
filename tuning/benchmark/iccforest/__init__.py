"""Python adapter for the authors' ICcforest R implementation."""

from .model import ICCForest, r_backend_available

__all__ = ["ICCForest", "r_backend_available"]
