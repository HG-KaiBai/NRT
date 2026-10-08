"""Ordinary feed-forward neural-network building blocks."""

from __future__ import annotations

import torch
from torch import nn


def init_linear(module: nn.Module) -> None:
    """Initialize linear layers with Glorot uniform weights and zero biases."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



class MLP(nn.Module):
    """A deterministic feed-forward network."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(n_layers - 1):
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.SiLU()))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)
        # self.net.apply(init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
