"""Raw-time KL on [0, 1.1 * maximum finite fit endpoint].

The completed normalization comparison supplies the verified geometry and
optimizer. No nonlinear time transform is used by this entry point.
"""
import math
import torch
from pycode.finite_support_kl.model import FiniteKLTree
from pycode.finite_support_kl.train import fit_finite_kl
from ..nll_cv_train import NLLCVOptimization
from ..train import TrainingConfig
from ..tuning_model import Architecture

SUPPORT = "raw_time_finite"


def observed_upper_bound(observed):
    """Use only finite L/R endpoints of the actual fit subset, never latent T."""
    x, left, right = observed
    if len(x) == 0 or left.shape != (len(x),) or right.shape != left.shape:
        raise ValueError("need nonempty fit data with matching endpoint vectors")
    endpoints = torch.cat((left.detach().double(), right.detach().double()))
    finite = endpoints[torch.isfinite(endpoints)]
    if not len(finite) or not bool((finite >= 0).all()) or not float(finite.max()) > 0:
        raise ValueError("need a positive finite observed fit endpoint")
    upper_bound = 1.1 * float(finite.max())
    if not math.isfinite(upper_bound):
        raise ValueError("finite fit endpoints overflow the support upper bound")
    return upper_bound


class RawTimeKLTree(FiniteKLTree):
    def __init__(self, architecture, upper_bound, optimization=NLLCVOptimization()):
        super().__init__(architecture, 1., optimization, upper_bound, "raw_time_finite_kl")

    @property
    def upper_bound(self):
        return self.upperbound


@torch.no_grad()
def predict_density(model, x, times, batch_size=128):
    """Original-time piecewise-constant density, zero outside [0,U]."""
    if times.ndim != 1 or bool(torch.isnan(times).any()) or batch_size < 1:
        raise ValueError("need a time vector without NaN and positive batch_size")
    if model.arm != "raw_time_finite_kl":
        raise ValueError("raw-time density requires a raw-time KL model")
    model.eval()
    x = x.to(device=model.upperbound.device, dtype=torch.float32)
    times = times.to(device=model.upperbound.device, dtype=torch.float64)
    values = []
    for batch in x.split(batch_size):
        state = model(batch)
        leaf = torch.searchsorted(state.upper.contiguous(),
            times.expand(len(batch), -1).contiguous(), right=True).clamp_max(model.tree.n_leaves-1)
        density = (state.log_mass.gather(1, leaf)-state.width.log().gather(1, leaf)).exp()
        values.append(torch.where((times >= 0) & (times <= model.upperbound), density, 0.))
    return torch.cat(values)


def fit_raw_time_kl(train, valid=None, *, architecture=Architecture(),
                    training=TrainingConfig(patience=50), optimization=NLLCVOptimization(),
                    seed=101, device="cuda", fixed_epochs=None):
    """Estimate U from train only; keep KL 1/L, 2/depth and a/sqrt(n_fit)."""
    upper_bound = observed_upper_bound(train)
    # An impossible validation observation must not enlarge training support.
    if valid is not None:
        left, right = valid[1:]
        exact = left == right
        impossible = (exact & ((left < 0) | (left > upper_bound))) | (
            ~exact & (torch.minimum(right.double(), right.new_tensor(upper_bound, dtype=torch.float64))
                      <= left.double().clamp_min(0)))
        if bool(impossible.any()):
            raise ValueError(f"{int(impossible.sum())} validation observations have zero likelihood "
                             f"under train-only raw support [0,{upper_bound}]")
    model, history = fit_finite_kl(train, valid, architecture=architecture,
        training=training, optimization=optimization, time_scale=1., upperbound=upper_bound,
        arm="raw_time_finite_kl", seed=seed, device=device, fixed_epochs=fixed_epochs)
    model.fit_info.update(support=SUPPORT, time_coordinate="raw", upper_bound=upper_bound,
                          max_observation=upper_bound/1.1)
    return model, history
