"""Parametric accelerated failure-time benchmark."""

from .model import ParametricAFT, distribution_for_scenario, r_backend_available

__all__ = ["ParametricAFT", "distribution_for_scenario", "r_backend_available"]
