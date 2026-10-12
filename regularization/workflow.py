"""Share the existing NRT datasets, seeds, splits and atomic output helpers."""
from contextlib import contextmanager
import csv
import fcntl
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from workflow import (ROOT as NRT_ROOT, SCHEMA_VERSION, atomic_json, check_device,
                      cv_splits, fingerprint, generate, horizon, select_cases, write_csv)
from .unified_nll.metrics import average_grid_sse
from .unified_nll.train import evaluate_nll

ROOT = Path(__file__).resolve().parent
METHODS = ("fixed_nll_cv", "adaptive_nll_cv", "fixed_at_adaptive_architecture")
SEARCH_METHODS = METHODS[:2]
MATCHED = METHODS[2]


def select_methods(names="all"):
    methods = list(METHODS) if names == "all" else list(dict.fromkeys(names.split(",")))
    if not methods or set(methods)-set(METHODS):
        raise ValueError(f"methods must be selected from {METHODS}")
    return methods


@contextmanager
def writer_lock(directory, name=".writer.lock"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/name).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another worker is writing {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@lru_cache(maxsize=1)
def source_hash():
    files = [ROOT/"model.py", ROOT/"train.py", ROOT/"workflow.py", NRT_ROOT/"workflow.py",
             NRT_ROOT/"tuning/testdata2.py", ROOT/"tuning/main.py"]
    files += [ROOT/"unified_nll"/f"{name}.py" for name in
              ("model", "tree", "train", "regularization", "tuning_model", "cv_train", "nll_cv_train")]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(NRT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dataset_hash(data):
    """Hash all six generated tensors and both random split index vectors."""
    digest = hashlib.sha256()
    for tensor in (data.x, data.L, data.R, data.T, data.first, data.second, data.train, data.test):
        value = tensor.detach().cpu().contiguous()
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tuning_data(case, *, n_samples=5000, base_seed=123, mu=0., r=None):
    seed = base_seed + case.index * 10_000
    tau = horizon(case, mu=mu, r=r)
    return generate(case, n_samples, seed, tau, mu=mu, r=r), seed


def simulation_data(case, spec, replication, *, n_samples=5000, base_seed=123):
    seed = base_seed + 1_000_000 + case.index * 10_000 + replication
    return generate(case, n_samples, seed, spec["tau"], mu=spec["mu"], r=spec["r"]), seed


def preflight(device, threads):
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    try:
        check_device(device)
    except Exception as error:
        print(f"Preflight failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(75) from error


def evaluate(model, data, grid_size=256):
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    grid = torch.linspace(0, data.max_observation(data.train), grid_size)
    estimate = model.predict_survival(data.x[data.test], grid).cpu()
    sse = float(average_grid_sse(estimate, data.truth(data.test, grid)))
    nll = evaluate_nll(model, *data.bounds(data.test))
    if math.isnan(nll) or nll == -math.inf:
        raise RuntimeError("invalid test NLL")
    # JSON is strict: genuinely impossible test observations have NLL=+infinity.
    return dict(average_grid_sse=sse, test_nll=nll if math.isfinite(nll) else None,
                test_nll_infinite=not math.isfinite(nll), grid_size=grid_size,
                grid_max=data.max_observation(data.train))


def export_tuning(output, cases, methods):
    """Export each method only when every selected case is complete."""
    output = Path(output)
    # Atomic replacements tolerate multiple case workers finishing concurrently.
    for method in methods:
        results = []
        for case in cases:
            path = output/method/case.id/"result.json"
            if not path.exists():
                break
            result = json.loads(path.read_text())
            if result.get("status") != "ok":
                break
            results.append(result)
        else:
            rows = [dict(model_name=method, data_types=r["case_id"],
                         parameters=json.dumps(r["parameters"], sort_keys=True),
                         refit_epochs=r["refit_epochs"],
                         training_config=json.dumps(r["training_config"], sort_keys=True),
                         optimization=json.dumps(r["optimization"], sort_keys=True),
                         tau=r["tau"], mu=r["mu"], r=r["r"], n_samples=r["n_samples"],
                         base_seed=r["base_seed"], source_sha256=r["source_sha256"],
                         tuning_hash=r["request_hash"], test_average_grid_sse=r["average_grid_sse"])
                    for r in results]
            write_csv(output/"hyper_para"/f"{method}.csv", rows)
            write_csv(output/f"{method}_summary.csv", [
                {key: r[key] for key in ("method", "case_id", "tau", "right_censoring_rate",
                 "cv_score", "refit_epochs", "average_grid_sse", "test_nll", "test_nll_infinite")}
                for r in results])


def load_hyperparameters(directory, cases, methods):
    tables, shared = {}, {}
    for method in methods:
        path = Path(directory)/f"{method}.csv"
        with path.open(newline="") as stream:
            table = {}
            for row in csv.DictReader(stream):
                case_id = row["data_types"]
                if row["model_name"] != method or case_id in table:
                    raise ValueError(f"invalid or duplicate row in {path}")
                spec = dict(method=method, case_id=case_id,
                            parameters=json.loads(row["parameters"]),
                            training_config=json.loads(row["training_config"]),
                            optimization=json.loads(row["optimization"]),
                            refit_epochs=int(row["refit_epochs"]),
                            tau=float(row["tau"]) if row["tau"] else None,
                            mu=float(row["mu"]), r=float(row["r"]) if row["r"] else None,
                            n_samples=int(row["n_samples"]), base_seed=int(row["base_seed"]),
                            source_sha256=row["source_sha256"], tuning_hash=row["tuning_hash"])
                fingerprint(spec)
                if (spec["refit_epochs"] <= spec["optimization"]["warmup_epochs"]
                        or spec["optimization"]["learn_splits"] != (method == METHODS[1])
                        or (method != METHODS[1] and spec["parameters"]["regularization_strength"] != 0)):
                    raise ValueError(f"invalid frozen fitting configuration in {path}")
                table[case_id] = spec
            for case in cases:
                if case.id not in table:
                    raise ValueError(f"missing {case.id} in {path}")
                spec = table[case.id]
                if ((case.censor_type == "exact" and spec["tau"] is not None)
                        or (case.censor_type != "exact" and
                            (spec["tau"] is None or not math.isfinite(spec["tau"]) or spec["tau"] <= 0))):
                    raise ValueError(f"invalid tau for {case.id}")
                contract = {k: spec[k] for k in ("tau", "mu", "r", "n_samples", "base_seed")}
                if case.id in shared and shared[case.id] != contract:
                    raise ValueError(f"methods disagree on dataset settings for {case.id}")
                shared[case.id] = contract
        tables[method] = table
    return tables
