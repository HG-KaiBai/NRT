"""Shared case definitions, deterministic data, and simulation evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
SETTINGS = ("ph", "lognormal", "po", "multimodal_separate", "multimodal_closer")
CENSOR_TYPES = ("exact", "right", "current", "interval")
NRT_LOSSES = {"NRT_NLL": "joint_nll", "NRT_Brier": "joint_brier", "NRT_2Loss": "alternating"}
DEFAULT_EPOCHS = {"NRT": 200, **{method: 200 for method in NRT_LOSSES}, "NNCDE": 200, "DeepSurv": 1000}
CPU_METHODS = ("deep_distribution_regression", "ctree", "iccforest", "cox", "aft")
METHODS = (*NRT_LOSSES, "NNCDE", "DeepSurv", *CPU_METHODS)
SUPPORTED = {
    **{method: CENSOR_TYPES for method in (*NRT_LOSSES, "ctree", "cox", "aft")},
    "NNCDE": ("exact", "right"), "DeepSurv": ("exact", "right"),
    "deep_distribution_regression": ("exact",), "iccforest": ("right", "current", "interval"),
}
PREFLIGHT_EXIT_CODE = 75
BENCHMARK_BACKENDS = {
    "NNCDE": ("nncde.model", "NNCDE"), "DeepSurv": ("deepsurv.cuda_model", "DeepSurvCUDA"),
    "deep_distribution_regression": ("deep_distribution_regression.model", "DeepDistributionRegression"),
    "ctree": ("ctree.model", "CTree"), "iccforest": ("iccforest.model", "ICCForest"),
    "cox": ("cox.model", "CoxPH"), "aft": ("aft.model", "ParametricAFT"),
}
R_REQUIREMENTS = {
    "ctree": "partykit, icenReg, survival",
    "iccforest": "ICcforest, icenReg, survival",
    "cox": "survival, icenReg",
    "aft": "survival",
}


@dataclass(frozen=True)
class Case:
    setting: str
    censor_type: str
    q: float

    @property
    def scenario(self):
        return "multimodal" if self.setting.startswith("multimodal_") else self.setting

    @property
    def multimodal_case(self):
        return self.setting.split("_", 1)[1] if self.scenario == "multimodal" else "separate"

    @property
    def id(self):
        return f"{self.setting}_{self.censor_type}_q{self.q:g}"

    @property
    def index(self):
        # Keep existing case seeds stable when redundant exact cases are removed.
        return (SETTINGS.index(self.setting) * 8
                + CENSOR_TYPES.index(self.censor_type) * 2
                + (0 if self.q == 0.2 else 1))


ALL_CASES = tuple(Case(s, c, q) for s in SETTINGS for c in CENSOR_TYPES
                  for q in ((0.2,) if c == "exact" else (0.2, 0.5)))


def select_cases(case_ids="all"):
    if case_ids == "all":
        return list(ALL_CASES)
    requested = case_ids.split(",")
    by_id = {case.id: case for case in ALL_CASES}
    unknown = set(requested) - by_id.keys()
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    return [by_id[key] for key in dict.fromkeys(requested)]


def select_methods(names):
    requested = names.split(",")
    unknown = set(requested) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    return list(dict.fromkeys(requested))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = stream.name
    os.replace(temporary, path)


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_device(device):
    resolved = torch.device(device)
    host = socket.gethostname().split(".", maxsplit=1)[0]
    print(f"Node preflight on {host}: device={resolved}; "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Use a CPU or CUDA device")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for a smoke test")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(f"GPU {resolved.index} is not visible")
        torch.cuda.set_device(resolved.index if resolved.index is not None else 0)
    # Availability alone does not establish that the allocated node can execute.
    values = torch.arange(1, 4097, device=resolved, dtype=torch.float32)
    result = torch.sum(values * values)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    if not torch.isfinite(result).item():
        raise RuntimeError(f"{resolved} calculation returned a non-finite value")
    if resolved.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved)
        print(f"GPU preflight passed on {host}: {properties.name}; "
              f"capability={properties.major}.{properties.minor}", flush=True)
    else:
        print(f"CPU execution preflight passed on {host}", flush=True)
    return resolved


def node_preflight(device, methods, package):
    """Check execution and selected dependencies before creating fits or workers.

    Only startup failures use exit 75, which the launchers may requeue.
    """
    try:
        check_device(device)
        for method in sorted(set(methods)):
            if method in NRT_LOSSES:
                importlib.import_module(f"{package}.train_mlp")
            else:
                module_name, class_name = BENCHMARK_BACKENDS[method]
                module = importlib.import_module(f"{package}.benchmark.{module_name}")
                getattr(module, class_name)
                if method in R_REQUIREMENTS and not module.r_backend_available():
                    raise RuntimeError(f"{method} requires Rscript and R packages {R_REQUIREMENTS[method]}")
                if package == "tuning" and method == "DeepSurv":
                    importlib.import_module("optunity")
            print(f"Backend preflight passed: {method}", flush=True)
    except Exception as error:
        print(f"Node preflight failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(PREFLIGHT_EXIT_CODE) from error


def seed_fit(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def horizon(case, *, mu=0.0, r=None):
    if case.censor_type == "exact":
        return None
    module = importlib.import_module("tuning.testdata2")
    return module.censoring_horizon(case.scenario, case.q, mu=mu, r=r, multimodal_case=case.multimodal_case)


@dataclass
class Dataset:
    x: torch.Tensor
    L: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    train: torch.Tensor
    test: torch.Tensor
    case: Case
    tau: float | None

    def bounds(self, indices):
        return self.x[indices], self.L[indices], self.R[indices]

    def max_observation(self, indices):
        _, left, right = self.bounds(indices)
        return float(torch.cat((left[torch.isfinite(left)], right[torch.isfinite(right)])).max())

    def metadata(self):
        return {
            "n_samples": len(self.x), "n_train": len(self.train), "n_test": len(self.test),
            "tau": self.tau, "right_censoring_rate": float(torch.isinf(self.R).float().mean()),
            "train_right_censoring_rate": float(torch.isinf(self.R[self.train]).float().mean()),
            "test_right_censoring_rate": float(torch.isinf(self.R[self.test]).float().mean()),
        }

    def truth(self, indices, grid):
        module = importlib.import_module("tuning.testdata2")
        if self.case.scenario == "multimodal":
            return module.multimodal_survival(self.first[indices], self.second[indices], grid)
        return module.survival_curves(self.first[indices], self.second[indices], grid, scenario=self.case.scenario)


def generate(case, n_samples, seed, tau, *, mu=0.0, r=None):
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10 for the train/test and CV splits")
    if case.censor_type != "exact" and tau is None:
        raise ValueError("Censored simulation cases require tau from tuning")
    module = importlib.import_module("tuning.testdata2")
    # Never advance a caller's training RNG; every method receives identical data.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        kwargs = dict(n=n_samples, mu=mu, q=case.q, censor_type=case.censor_type, device="cpu", tau=tau)
        if r is not None:
            kwargs["r"] = r
        if case.scenario == "multimodal":
            values = module.data_reg_multimodal(**kwargs, multimodal_case=case.multimodal_case)
        else:
            values = module.data_reg(**kwargs, scenario=case.scenario)
    order = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed + 1))
    split = int(0.8 * n_samples)
    return Dataset(*values[:6], order[:split], order[split:], case, tau)


def cv_splits(indices, seed, folds=4):
    if len(indices) < folds:
        raise ValueError("Too few training observations for four-fold CV")
    order = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))]
    chunks = torch.tensor_split(order, folds)
    return [(torch.cat([chunk for j, chunk in enumerate(chunks) if j != i]), chunk)
            for i, chunk in enumerate(chunks)]


def fit_model(method, data, indices, parameters, training, device, seed, package):
    seed_fit(seed)
    x, left, right = data.bounds(indices)
    if method in NRT_LOSSES:
        module = importlib.import_module(f"{package}.train_mlp")
        model, history = module.fit(
            x, left, right, 1.1 * data.max_observation(indices),
            module.ModelConfig(**parameters),
            module.TrainingConfig(**training, training_method=NRT_LOSSES[method]), device, verbose=False,
        )
        return model
    name, class_name = BENCHMARK_BACKENDS[method]
    model_class = getattr(importlib.import_module(f"{package}.benchmark.{name}"), class_name)
    kwargs = {**parameters, **training}
    if method in ("NNCDE", "DeepSurv", "deep_distribution_regression"):
        kwargs.update(device=device, seed=seed)
    elif method in ("ctree", "iccforest"):
        kwargs["seed"] = seed
    model = model_class(**kwargs)
    try:
        return model.fit(x, left, right, censor_type=data.case.censor_type)
    except BaseException:
        close_model(model)
        raise


def close_model(model):
    if hasattr(model, "close"):
        model.close()


@torch.no_grad()
def nncde_nll(model, data, indices):
    x, left, right = (value.to(model.device) for value in data.bounds(indices))
    observed, event = model._outcomes(left, right, data.case.censor_type)
    starts, stops = model._time_grid(observed)
    losses = []
    for fitted in model._models:
        total = torch.zeros((), device=model.device)
        for start in range(0, len(x), model.batch_size):
            stop = min(start + model.batch_size, len(x))
            total += model._loss(fitted, x[start:stop], observed[start:stop], event[start:stop], starts, stops) * (stop - start)
        losses.append(total / len(x))
    return float(torch.stack(losses).mean())


@torch.no_grad()
def evaluate(model, data, grid_size, *, method, device, package):
    grid = torch.linspace(0.0, data.max_observation(data.train), grid_size)
    # Bound prediction memory for the NNCDE expansion and deep trees.
    total = torch.zeros(())
    for indices in data.test.split(256):
        prediction_x = data.x[indices].to(device) if method in NRT_LOSSES else data.x[indices]
        estimated = model.predict_survival(prediction_x, grid.to(prediction_x.device)).detach().cpu()
        truth = data.truth(indices, grid)
        if estimated.shape != truth.shape or not bool(torch.isfinite(estimated).all()):
            raise ValueError("Invalid survival predictions")
        total += (estimated - truth).square().sum()
    metrics = {"average_grid_sse": float(total / len(data.test))}
    if method in NRT_LOSSES:
        module = importlib.import_module(f"{package}.train_mlp")
        x, left, right = data.bounds(data.test)
        metrics.update(test_nll=module.evaluate_nll(model, x, left, right, device),
                       test_brier=module.evaluate_brier(model, x, left, right, grid_size, device))
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite test metrics")
    return metrics
