"""Prepared raw-time observations shared by CPU and CUDA training."""
from typing import NamedTuple

import torch


class Observations(NamedTuple):
    x: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    exact: torch.Tensor
