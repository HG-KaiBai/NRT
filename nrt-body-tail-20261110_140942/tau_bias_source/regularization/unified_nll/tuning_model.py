"""Shared trunk and separate mass/split branches for the six-parameter search."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .model import ConditionalTree
from .regularization import partition_penalty
from .tree import Tree


@dataclass(frozen=True)
class Architecture:
    depth: int = 3
    shared_width: int = 32
    shared_layers: int = 0
    private_width: int = 32
    private_layers: int = 3
    regularization_strength: float = 500.0
    x_dim: int = 6
    split_min: float = .1
    total_width_floor: float = 8e-4


def _branch(in_dim, width, layers):
    modules = []
    for index in range(layers):
        modules.append(nn.Linear(in_dim, width))
        if index < layers - 1:
            modules.append(nn.SiLU())
        in_dim = width
    return nn.Sequential(*modules)


class TuningTree(ConditionalTree):
    """Private layer count includes its final feature projection, excludes head.

    Shared layers all have SiLU activations. A zero-layer shared trunk is the
    identity, in which case shared_width has no effect. Both branches have the
    same private architecture but independent weights. No unused encoders.
    """
    def __init__(self, config: Architecture, time_scale: float):
        nn.Module.__init__(self)
        if (config.shared_layers < 0 or min(config.shared_width, config.private_width,
                config.private_layers, config.x_dim) < 1):
            raise ValueError("invalid network dimensions")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be positive and finite")
        if not 0 <= config.split_min < .5 or not 0 < config.total_width_floor < 1:
            raise ValueError("invalid split/leaf constraints")
        if not math.isfinite(config.regularization_strength) or config.regularization_strength < 0:
            raise ValueError("regularization strength must be finite and nonnegative")
        self.config = config
        self.tree = Tree(config.depth, config.total_width_floor / 2**config.depth)
        self.register_buffer("time_scale", torch.tensor(float(time_scale)))
        layers, width = [], config.x_dim
        for _ in range(config.shared_layers):
            layers.extend((nn.Linear(width, config.shared_width), nn.SiLU()))
            width = config.shared_width
        self.shared = nn.Sequential(*layers)
        self.mass_encoder = _branch(width, config.private_width, config.private_layers)
        self.split_encoder = _branch(width, config.private_width, config.private_layers)
        self.mass_head = nn.Linear(config.private_width, self.tree.n_internal)
        self.split_head = nn.Linear(config.private_width, self.tree.n_internal)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.mass_head.weight)
        nn.init.zeros_(self.split_head.weight)

    def forward(self, x):
        features = self.shared(x)
        logits = self.mass_head(self.mass_encoder(features))
        splits = self.config.split_min + (1 - 2*self.config.split_min) * torch.sigmoid(
            self.split_head(self.split_encoder(features)))
        return self.tree(logits, splits)

    def terms(self, observations):
        state = self(observations.x)
        return (-self.tree.loglik(state, *observations[1:]).mean(),
                partition_penalty(state, self.config.depth).mean())
