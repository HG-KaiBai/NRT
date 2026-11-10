"""Standalone raw-time adapter for the preserved regularized NLL network."""
import math

import torch
from torch.nn import functional as F

from .unified_nll.model import Observations
from .unified_nll.nll_cv_train import NLLCVOptimization, NLLCVTree
from .unified_nll.tree import Tree, TreeState
from .unified_nll.tuning_model import Architecture


PENALTIES = ("partition_kl",)


def observed_upper_bound(observed):
    x, left, right = observed
    if len(x) == 0 or left.shape != (len(x),) or right.shape != left.shape:
        raise ValueError("need nonempty fit data with matching endpoint vectors")
    endpoints = torch.cat((left.detach().double(), right.detach().double()))
    finite = endpoints[torch.isfinite(endpoints)]
    if not len(finite) or not bool((finite >= 0).all()) or not float(finite.max()) > 0:
        raise ValueError("need positive finite observed training endpoints")
    upper = 1.1 * float(finite.max())
    if not math.isfinite(upper):
        raise ValueError("support overflow")
    return upper


class RawGeometry(Tree):
    def __init__(self, architecture, upper_bound):
        super().__init__(architecture.depth, architecture.total_width_floor / 2**architecture.depth)
        self.register_buffer("upper_bound", torch.tensor(upper_bound, dtype=torch.float64))

    def forward(self, logits, rho):
        # Geometry is computed in float64; network and log masses stay float32.
        lower = torch.zeros((len(logits), 1), device=logits.device, dtype=torch.float64)
        upper = torch.ones_like(lower) * self.upper_bound
        log_mass = torch.zeros_like(logits[:, :1])
        for level in range(self.depth):
            start, stop = 2**level - 1, 2**(level + 1) - 1
            reserved = 2**(self.depth-level-1) * self.min_leaf_width * self.upper_bound
            cut = lower + reserved + (upper-lower-2*reserved) * rho[:, start:stop].double()
            lower = torch.stack((lower, cut), dim=-1).flatten(1)
            upper = torch.stack((cut, upper), dim=-1).flatten(1)
            branch = logits[:, start:stop]
            log_mass = torch.stack((log_mass+F.logsigmoid(branch),
                                    log_mass+F.logsigmoid(-branch)), dim=-1).flatten(1)
        return TreeState(lower, upper, log_mass)

    def loglik(self, state, left, right, exact, log_jacobian):
        width = state.width
        inside = (left[:, None] >= state.lower) & (left[:, None] < state.upper)
        inside |= (left[:, None] == self.upper_bound) & (state.upper == self.upper_bound)
        overlap = (torch.minimum(state.upper, right[:, None])
                   - torch.maximum(state.lower, left[:, None])).clamp_min(0)
        fraction = torch.where(overlap > 0,
                               overlap.clamp_min(torch.finfo(width.dtype).tiny).log()-width.log(),
                               -torch.inf)
        density = torch.where(inside, -width.log(), -torch.inf)
        return torch.logsumexp(state.log_mass + torch.where(exact[:, None], density, fraction), -1)


class RawTimeKLTree(NLLCVTree):
    def __init__(self, architecture=Architecture(), upper_bound=1., optimization=NLLCVOptimization(),
                 penalty="partition_kl"):
        if not math.isfinite(upper_bound) or upper_bound <= 0:
            raise ValueError("support must be positive and finite")
        if penalty not in PENALTIES:
            raise ValueError(f"penalty must be one of {PENALTIES}")
        super().__init__(architecture, 1., optimization)
        self.tree = RawGeometry(architecture, upper_bound)
        self.penalty = penalty

    @property
    def upper_bound(self):
        return self.tree.upper_bound

    def prepare(self, x, left, right):
        x = x.to(device=self.upper_bound.device, dtype=torch.float32)
        left, right = (v.to(device=self.upper_bound.device, dtype=torch.float64) for v in (left, right))
        if (x.ndim != 2 or x.shape[1] != self.config.x_dim or not len(x)
                or left.shape != (len(x),) or right.shape != left.shape):
            raise ValueError("invalid observation shapes")
        if not bool(torch.isfinite(x).all() & torch.isfinite(left).all()
                    & ~torch.isnan(right).any() & (left >= 0).all() & (right >= left).all()):
            raise ValueError("need finite X and 0 <= L <= R <= infinity")
        return Observations(x, left, right, left == right, torch.zeros_like(left))

    def terms(self, observations):
        state = self(observations.x)
        nll = -self.tree.loglik(state, *observations[1:]).mean()
        penalty = (-2/self.config.depth) * (state.width/(self.upper_bound/self.tree.n_leaves)).log().mean()
        return nll, penalty

    @torch.no_grad()
    def predict_survival(self, x, times, batch_size=128):
        if times.ndim != 1 or bool(torch.isnan(times).any()) or batch_size < 1:
            raise ValueError("need a time vector without NaN and positive batch_size")
        self.eval()
        x = x.to(device=self.upper_bound.device, dtype=torch.float32)
        times = times.to(self.upper_bound)
        return torch.cat([self.tree.survival(self(batch), times) for batch in x.split(batch_size)])
