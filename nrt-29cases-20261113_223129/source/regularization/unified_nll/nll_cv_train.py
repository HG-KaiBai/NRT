"""One observed-data loss, NLL checkpoints, and a genuine fixed-rho control."""
from dataclasses import dataclass

import torch

from .tuning_model import TuningTree


@dataclass(frozen=True)
class NLLCVOptimization:
    split_lr_ratio: float = .1
    warmup_epochs: int = 10
    learn_splits: bool = True


class NLLCVTree(TuningTree):
    def __init__(self, architecture, optimization):
        super().__init__(architecture)
        self.register_buffer("split_gate", torch.tensor(float(optimization.warmup_epochs == 0)))
        self.learn_splits = optimization.learn_splits
        if not self.learn_splits:
            self.split_encoder.requires_grad_(False)
            self.split_head.requires_grad_(False)

    def rho(self, x):
        if self.learn_splits:
            raw = self.split_head(self.split_encoder(self.shared(x)))
            return self.config.split_min + (1-2*self.config.split_min)*torch.sigmoid(self.split_gate*raw)
        return x.new_full((len(x), self.tree.n_internal), .5)

    def forward(self, x):
        features = self.shared(x)
        logits = self.mass_head(self.mass_encoder(features))
        if self.learn_splits:
            raw = self.split_head(self.split_encoder(features))
            rho = self.config.split_min+(1-2*self.config.split_min)*torch.sigmoid(self.split_gate*raw)
        else:
            rho = logits.new_full((len(x), self.tree.n_internal), .5)
        return self.tree(logits, rho)
