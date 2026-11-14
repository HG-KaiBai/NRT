"""Raw-time NRT with a root body/tail split and a finite body subtree."""
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .unified_nll.model import Observations
from .unified_nll.nll_cv_train import NLLCVOptimization, NLLCVTree
from .unified_nll.tree import Tree
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
    return float(finite.max())


class RawTreeState(NamedTuple):
    """Body intervals/masses and the separate, width-free mass for T > tau."""

    lower: torch.Tensor
    upper: torch.Tensor
    log_mass: torch.Tensor
    log_tail_mass: torch.Tensor

    @property
    def width(self):
        return self.upper - self.lower


class RawGeometry(Tree):
    """Root logit first, then breadth-first body logits; cuts are body-only.

    ``depth`` counts body levels, excluding the fixed split at tau.
    """

    def __init__(self, architecture, upper_bound):
        super().__init__(architecture.depth, architecture.total_width_floor / 2**architecture.depth)
        self.register_buffer("upper_bound", torch.tensor(upper_bound, dtype=torch.float64))

    def forward(self, logits, rho):
        if (logits.ndim != 2 or rho.ndim != 2
                or logits.shape != (len(rho), self.n_internal + 1)
                or rho.shape[1] != self.n_internal):
            raise ValueError("need 2**depth probability logits and 2**depth-1 cut proportions")
        # Geometry is computed in float64; network and log masses stay float32.
        lower = torch.zeros((len(logits), 1), device=logits.device, dtype=torch.float64)
        upper = torch.ones_like(lower) * self.upper_bound
        # Split at tau before partitioning (0, tau]. The right child is the
        # terminal tail; the left child's mass is shared by every body path.
        log_mass = F.logsigmoid(logits[:, :1])
        tail_mass = F.logsigmoid(-logits[:, 0])
        for level in range(self.depth):
            start, stop = 2**level - 1, 2**(level + 1) - 1
            reserved = 2**(self.depth-level-1) * self.min_leaf_width * self.upper_bound
            cut = lower + reserved + (upper-lower-2*reserved) * rho[:, start:stop].double()
            lower = torch.stack((lower, cut), dim=-1).flatten(1)
            upper = torch.stack((cut, upper), dim=-1).flatten(1)
            branch = logits[:, start+1:stop+1]
            log_mass = torch.stack((log_mass+F.logsigmoid(branch),
                                    log_mass+F.logsigmoid(-branch)), dim=-1).flatten(1)
        return RawTreeState(lower, upper, log_mass, tail_mass)

    def loglik(self, state, left, right, exact, log_jacobian):
        """Root-weighted body density/overlap plus the root's tail probability.

        Body masses already contain P(T <= tau | x); the tail contributes
        P(T > tau | x) only when the observation includes the entire tail.
        """
        width = state.width
        inside = (left[:, None] > state.lower) & (left[:, None] <= state.upper)
        overlap = (torch.minimum(state.upper, right[:, None])
                   - torch.maximum(state.lower, left[:, None])).clamp_min(0)
        fraction = torch.where(overlap > 0,
                               overlap.clamp_min(torch.finfo(width.dtype).tiny).log()-width.log(),
                               -torch.inf)
        density = torch.where(inside, -width.log(), -torch.inf)
        body = state.log_mass + torch.where(exact[:, None], density, fraction)
        tail = torch.where(~exact & torch.isposinf(right), state.log_tail_mass, -torch.inf)
        # One reduction also keeps gradients finite for tail-only observations.
        return torch.logsumexp(torch.cat((body, tail[:, None]), dim=-1), dim=-1)

    def survival(self, state, times):
        if bool(((times < 0) | (torch.isfinite(times) & (times > self.upper_bound))).any()):
            raise ValueError("survival is defined only on [0, tau] and at +infinity")
        body = super().survival(state, times)
        tail = torch.where(torch.isposinf(times)[None, :], 0., state.log_tail_mass.exp()[:, None])
        return body + tail


class RawTimeKLTree(NLLCVTree):
    def __init__(self, architecture=Architecture(), upper_bound=1., optimization=NLLCVOptimization(),
                 penalty="partition_kl"):
        if not math.isfinite(upper_bound) or upper_bound <= 0:
            raise ValueError("support must be positive and finite")
        if penalty not in PENALTIES:
            raise ValueError(f"penalty must be one of {PENALTIES}")
        super().__init__(architecture, 1., optimization)
        self.tree = RawGeometry(architecture, upper_bound)
        self.mass_head = nn.Linear(architecture.private_width, self.tree.n_internal + 1)
        nn.init.zeros_(self.mass_head.weight)
        nn.init.zeros_(self.mass_head.bias)
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
        # Collapse all information beyond tau to the single tail category.
        # In particular, an exact t > tau becomes the censored set (tau, inf).
        left = torch.minimum(left, self.upper_bound)
        right = torch.where(right > self.upper_bound, torch.inf, right)
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
