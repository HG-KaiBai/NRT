"""Shared trunk and separate mass/split branches for the six-parameter search."""
from dataclasses import dataclass
import math
from torch import nn
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


class TuningTree(nn.Module):
    """Private layer count includes its final feature projection, excludes head.

    Shared layers all have SiLU activations. A zero-layer shared trunk is the
    identity, in which case shared_width has no effect. Both branches have the
    same private architecture but independent weights. No unused encoders.
    """
    def __init__(self, config: Architecture):
        super().__init__()
        if (config.shared_layers < 0 or min(config.shared_width, config.private_width,
                config.private_layers, config.x_dim) < 1):
            raise ValueError("invalid network dimensions")
        if not 0 <= config.split_min < .5 or not 0 < config.total_width_floor < 1:
            raise ValueError("invalid split/leaf constraints")
        if not math.isfinite(config.regularization_strength) or config.regularization_strength < 0:
            raise ValueError("regularization strength must be finite and nonnegative")
        self.config = config
        self.tree = Tree(config.depth, config.total_width_floor / 2**config.depth)
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
