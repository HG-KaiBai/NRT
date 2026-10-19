"""Real-world NRT search space and sampler, independent of regularization tuning.

Edit SearchConfig here or pass --nrt-search-space with search/baseline JSON.
Only the NRT architecture/training implementation is shared with regularization.
"""
from dataclasses import dataclass, replace
import math
import sys
from scipy.stats import qmc

from . import NRT_ROOT
from workflow import fingerprint

# Explicit local anchors, independent of upstream Architecture defaults.
BASELINE = dict(depth=8, shared_width=32, shared_layers=2, private_width=32,
                private_layers=3, regularization_strength=500., split_min=.1,
                total_width_floor=8e-4)


@dataclass(frozen=True)
class SearchConfig:
    trials: int = 500
    initial_trials: int = 20
    folds: int = 4
    fold_seed: int = 321
    search_seed: int = 871
    search_fit_seed: int = 101
    depth_min: int = 6
    depth_max: int = 12
    widths: tuple = (16, 32, 64, 128)
    shared_layers_max: int = 3
    private_layers_max: int = 4
    strength_max: float = 10000.


def optuna_module():
    try:
        import optuna
    except ImportError:
        sys.path.append(str(NRT_ROOT / "regularization/.deps"))
        import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def coordinates(architecture, adaptive):
    result = {key: getattr(architecture, key) for key in
              ("depth", "shared_layers", "shared_width", "private_width", "private_layers")}
    if adaptive:
        result["log10_one_plus_strength"] = math.log10(1+architecture.regularization_strength)
    return result


def startup(search, baseline, adaptive, seed):
    # Keep every startup anchor inside the user-selected local search space.
    baseline = replace(baseline,
        depth=min(max(baseline.depth, search.depth_min), search.depth_max),
        shared_layers=min(baseline.shared_layers, search.shared_layers_max),
        shared_width=baseline.shared_width if baseline.shared_width in search.widths else search.widths[0],
        private_width=baseline.private_width if baseline.private_width in search.widths else search.widths[0],
        private_layers=min(baseline.private_layers, search.private_layers_max),
        regularization_strength=min(baseline.regularization_strength, search.strength_max))
    first = baseline if adaptive else replace(baseline, regularization_strength=0.)
    result = [coordinates(first, adaptive)]
    if adaptive:
        result.append(coordinates(replace(baseline, regularization_strength=0.), True))
        for strength in (1000., 10000.):
            if strength <= search.strength_max:
                result.append(coordinates(replace(baseline, regularization_strength=strength), True))
    result = list({fingerprint(row): row for row in result}.values())
    if len(result) >= search.initial_trials:
        return result[:search.initial_trials]
    sampler = qmc.Sobol(6, scramble=True, seed=seed)
    points = sampler.random_base2(math.ceil(math.log2(max(32, search.initial_trials*8))))
    seen = {fingerprint(row) for row in result}
    for u in points:
        architecture = replace(first,
            depth=search.depth_min+int(u[0]*(search.depth_max-search.depth_min+1)),
            shared_layers=int(u[1]*(search.shared_layers_max+1)),
            shared_width=search.widths[int(u[2]*len(search.widths))],
            private_width=search.widths[int(u[3]*len(search.widths))],
            private_layers=1+int(u[4]*search.private_layers_max),
            regularization_strength=10**(u[5]*math.log10(1+search.strength_max))-1 if adaptive else 0.)
        if architecture.shared_layers == 0:
            architecture = replace(architecture, shared_width=baseline.shared_width)
        row = coordinates(architecture, adaptive)
        if fingerprint(row) not in seen:
            result.append(row)
            seen.add(fingerprint(row))
        if len(result) >= search.initial_trials:
            return result[:search.initial_trials]
    # A singleton architecture space is valid; TPE can repeat configurations.
    return result


def suggest(trial, search, baseline, adaptive):
    shared = trial.suggest_int("shared_layers", 0, search.shared_layers_max)
    inactive_width = baseline.shared_width if baseline.shared_width in search.widths else search.widths[0]
    return replace(baseline,
        depth=trial.suggest_int("depth", search.depth_min, search.depth_max), shared_layers=shared,
        shared_width=trial.suggest_categorical("shared_width", search.widths) if shared else inactive_width,
        private_width=trial.suggest_categorical("private_width", search.widths),
        private_layers=trial.suggest_int("private_layers", 1, search.private_layers_max),
        regularization_strength=10**trial.suggest_float("log10_one_plus_strength", 0.,
            math.log10(1+search.strength_max))-1 if adaptive else 0.)
