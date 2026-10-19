"""SUPPORT: python -m realworld.run {prepare,preflight,run,summarize}.

From the NRT directory, submit the full-budget server analysis with:
    bash realworld/submit_support.sh
Uses the existing environment: torch, numpy>=1.26, pandas, scipy, optuna==4.5.0,
lifelines>=0.30,<0.31 and scikit-survival>=0.25,<0.29. No new environment is made.
Eight-method results are exported to realworld/output/support/comparison.csv
and brier_curves.csv only when all methods complete. --smoke is validation only.
"""
import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, replace
import gc
import hashlib
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from . import ROOT, NRT_ROOT
from .data import DEFAULT_DATA, FEATURES, load_support
from .metrics import evaluate, evaluation_grid
from .models import PaperCox, SupportDeepSurv, MemoryBoundNNCDE, support_parameters
from .tuning import NRT_METHODS, default_config, nrt_settings, select_nrt, select_nncde
from regularization.model import observed_upper_bound
from regularization.train import fit_raw_time_kl
from regularization.workflow import writer_lock
from workflow import atomic_json, cv_splits, fingerprint, write_csv

METHODS = (*NRT_METHODS, "DeepSurv", "NNCDE", "Cox", "Ctree", "ICcforest")


@contextmanager
def manifest_lock(output):
    """Allow independent Slurm workers to initialize the same manifest safely."""
    with ExitStack() as stack:
        deadline = time.monotonic() + 30
        while True:
            try:
                stack.enter_context(writer_lock(output, ".manifest.lock"))
                break
            except RuntimeError as error:
                if "another worker is writing" not in str(error) or time.monotonic() >= deadline:
                    raise
                time.sleep(.1)
        yield


def source_hash():
    paths = list(ROOT.glob("*.py"))
    paths += list((NRT_ROOT / "regularization").rglob("*.py"))
    paths += list((NRT_ROOT / "testing/benchmark").rglob("*.py"))
    paths += list((NRT_ROOT / "testing/benchmark").rglob("*.R"))
    paths += [NRT_ROOT / "workflow.py", NRT_ROOT / "tuning/main.py"]
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(NRT_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    result = dict(python=platform.python_version(), platform=platform.platform(),
                  cuda=torch.version.cuda)
    for name in ("torch", "numpy", "scipy", "pandas", "optuna", "lifelines", "sksurv"):
        module = importlib.import_module(name)
        result[name] = module.__version__
    return result


def preflight(methods, device):
    versions = environment()
    if device.startswith("cuda"):
        torch.ones(1, device=device).add_(1)
        versions["gpu"] = torch.cuda.get_device_name(torch.device(device))
    if set(methods) & {"Ctree", "ICcforest"}:
        from testing.benchmark.ctree.model import _find_rscript
        executable = _find_rscript()
        if executable is None:
            raise RuntimeError("Rscript missing: install R and packages in realworld/install_r.R")
        packages = ["survival", "partykit", "icenReg"]
        if "ICcforest" in methods:
            packages.append("ICcforest")
        script = "p <- c(" + ",".join(repr(p) for p in packages) + "); "
        script += ("ok <- sapply(p, requireNamespace, quietly=TRUE); if(!all(ok)) stop(paste('Missing', paste(p[!ok],collapse=', '))); "
                   "cat(R.version.string, '\\n'); for(k in p) cat(k, as.character(packageVersion(k)), '\\n')")
        result = subprocess.run([executable, "--vanilla", "-e", script], text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        versions["R"] = result.stdout.strip()
    return versions


def initialize(output, data, config):
    grid, lower, upper = evaluation_grid(data.outcomes(data.train), config["grid_size"])
    splits = cv_splits(data.train, config["fold_seed"], config["folds"])
    request = dict(schema_version=2, dataset="SUPPORT", dataset_sha256=data.sha256,
                   source_sha256=source_hash(), config=config,
                   train_indices=data.train.tolist(), test_indices=data.test.tolist(),
                   features=FEATURES, evaluation_grid=grid.tolist(),
                   evaluation_start_days=lower, evaluation_end_days=upper,
                   evaluation_support="zero_to_max_observed_training_time",
                   nrt_upper_bound=observed_upper_bound(data.bounds(data.train)),
                   cv_validation_indices=[v.tolist() for _, v in splits],
                   python_versions={k: v for k, v in environment().items() if k != "platform"})
    # Device is recorded per result, not a cross-method contract (CPU R vs GPU NN).
    request_hash = fingerprint(request)
    with manifest_lock(output):
        path = output / "request.json"
        if path.exists() and json.loads(path.read_text())["request_hash"] != request_hash:
            raise ValueError("Data/source/config/environment changed; choose a new --output-dir")
        atomic_json(path, dict(request_hash=request_hash, **request))
        write_csv(output / "dataset_summary.csv", data.metadata())
        rows = []
        fold_for = {int(ix): k for k, (_, valid) in enumerate(splits, 1) for ix in valid}
        for label, indices in (("train", data.train), ("test", data.test)):
            for ix in indices.tolist():
                rows.append(dict(csv_row=ix, split=label, row_in_split=int(data.frame.iloc[ix].row_in_split),
                                 cv_validation_fold=fold_for.get(ix)))
        write_csv(output / "split_assignments.csv", rows)
    return request_hash, splits, grid, lower


def save_model(model, directory, method):
    if method in NRT_METHODS:
        torch.save(model.state_dict(), directory / "model.pt")
    elif method == "DeepSurv":
        torch.save(dict(state=model._model.state_dict(), input_mean=model._input_mean,
                        input_std=model._input_std, baseline_times=model._baseline_times,
                        baseline_cumulative_hazard=model._baseline_cumulative_hazard,
                        risk_shift=model._risk_shift), directory / "model.pt")
        atomic_json(directory / "history.json", model.training_history_)
    elif method == "NNCDE":
        torch.save([dict(state=f.network.state_dict(), input_mean=f.input_mean, input_std=f.input_std)
                    for f in model._models], directory / "model.pt")
        atomic_json(directory / "crossfit_details.json", model.fit_details_)
    elif method == "Cox":
        model.model.summary.to_csv(directory / "coefficients.csv")
        model.model.baseline_cumulative_hazard_.to_csv(directory / "baseline_cumulative_hazard.csv")
    elif method in ("Ctree", "ICcforest"):
        shutil.copy2(model._model_path, directory / "model.rds")


def run_method(method, data, config, splits, grid, lower, output, request_hash, device, versions):
    directory = output / method
    with writer_lock(directory):
        path = directory / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("request_hash") != request_hash:
                raise ValueError("Result request mismatch")
            if result.get("status") == "ok":
                print(f"{method}: reused complete result", flush=True)
                return
        started = time.monotonic()
        model, scaler = None, None
        try:
            if method in NRT_METHODS:
                # Server launcher keeps adaptive and matched control in one worker.
                selection = select_nrt(method, data, splits, config, output, device)
                _, _, training, optimization = nrt_settings(config)
                from regularization.unified_nll.tuning_model import Architecture
                architecture = Architecture(**selection["parameters"])
                optimization = replace(optimization, learn_splits=method == NRT_METHODS[1])
                scaler = data.scaler(data.train)
                model, history = fit_raw_time_kl(data.bounds(data.train, scaler), architecture=architecture,
                    training=training, optimization=optimization, seed=config["seed"], device=device,
                    fixed_epochs=selection["refit_epochs"], upper_bound=observed_upper_bound(data.bounds(data.train)))
                atomic_json(directory / "history.json", history)
                atomic_json(directory / "scaler.json", dict(mean=scaler[0].tolist(), std=scaler[1].tolist()))
            elif method == "NNCDE":
                selection = select_nncde(data, splits, config, output, device)
                model = MemoryBoundNNCDE(**selection["parameters"], epochs=config["nncde_epochs"],
                    patience=config["nncde_patience"], seed=config["seed"], device=device)
                model.fit(*data.bounds(data.train), censor_type="right")
            elif method == "DeepSurv":
                parameters = {**support_parameters(), "epochs": config["deepsurv_epochs"]}
                selection = dict(parameters=parameters, cv_metric="fixed_paper_parameters",
                    training_semantics="paper_inverse_time_decay_breslow_ties_author_output_L2")
                model = SupportDeepSurv(**parameters, seed=config["seed"], device=device)
                model.fit(*data.bounds(data.train), censor_type="right")
            elif method == "Cox":
                selection = dict(parameters=dict(penalizer=0., ties="efron", baseline="breslow"),
                                 cv_metric="author_default_unpenalized")
                model = PaperCox().fit(*data.bounds(data.train))
            else:
                if method == "Ctree":
                    from testing.benchmark.ctree.model import CTree
                    model = CTree(**config["ctree"], seed=config["seed"])
                    parameters = config["ctree"]
                else:
                    from testing.benchmark.iccforest.model import ICCForest
                    model = ICCForest(**config["iccforest"], seed=config["seed"])
                    parameters = config["iccforest"]
                selection = dict(parameters=parameters, cv_metric="existing_backend_defaults_or_train_only_mtry")
                model.fit(*data.bounds(data.train), censor_type="right")
            atomic_json(directory / "selection.json", selection)
            x_test = data.bounds(data.test, scaler)[0]
            times = torch.tensor(grid, dtype=torch.float64)
            with torch.no_grad():
                # R subprocesses predict in one call; bound NN prediction memory by subjects.
                batches = [x_test] if method in ("Ctree", "ICcforest", "Cox") else x_test.split(64)
                survival = torch.cat([model.predict_survival(batch, times).cpu() for batch in batches]).double().numpy()
                native_risk = model.predict_risk(x_test).cpu().numpy() if method in ("DeepSurv", "Cox") else None
            metrics, brier_rows, risk = evaluate(data.outcomes(data.train), data.outcomes(data.test),
                                                survival, grid, lower, native_risk)
            save_model(model, directory, method)
            np.savez_compressed(directory / "predictions.npz", survival=survival, times=grid,
                                risk=risk, csv_rows=data.test.numpy(),
                                native_risk=np.array([]) if native_risk is None else native_risk)
            write_csv(directory / "brier_curve.csv", [dict(method=method, **r) for r in brier_rows])
            result = dict(status="ok", method=method, dataset="SUPPORT", profile=config["profile"],
                          request_hash=request_hash, device=device, environment=versions,
                          torch_threads=torch.get_num_threads(),
                          n_train=len(data.train), n_test=len(data.test), seed=config["seed"],
                          elapsed_seconds=time.monotonic() - started, selection=selection, **metrics)
            atomic_json(path, result)
            print(f"{method}: C-index={metrics['c_index']:.5f}, IBS={metrics['integrated_brier_score']:.5f}", flush=True)
        except Exception as error:
            atomic_json(path, dict(status="failed", method=method, request_hash=request_hash,
                                  error=str(error), traceback=traceback.format_exc()))
            raise
        finally:
            if hasattr(model, "close"):
                model.close()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def summarize(output, *, require_complete=True):
    manifest = json.loads((output / "request.json").read_text())
    with writer_lock(output, ".summary.lock"):
        results, status = [], []
        for method in METHODS:
            path = output / method / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            if result and result.get("request_hash") != manifest["request_hash"]:
                raise ValueError(f"Mismatched result: {method}")
            status.append(dict(method=method, status=result.get("status", "pending"), error=result.get("error", "")))
            if result.get("status") == "ok":
                results.append(result)
        write_csv(output / "status.csv", status)
        if len(results) != len(METHODS):
            if require_complete:
                raise RuntimeError(f"Only {len(results)}/{len(METHODS)} methods complete; see status.csv")
            return
        columns = ["dataset", "profile", "method", "n_train", "n_test", "seed", "c_index",
                   "c_index_native_risk", "integrated_brier_score", "ipcw_brier_integral",
                   "brier_30d", "brier_90d", "brier_180d",
                   "brier_365d", "brier_730d", "evaluation_start_days", "evaluation_end_days",
                   "min_censoring_survival", "risk_definition", "elapsed_seconds"]
        write_csv(output / "comparison.csv", [{k: r.get(k) for k in columns} for r in results])
        write_csv(output / "selected_parameters.csv", [dict(method=r["method"],
            selection=json.dumps(r["selection"], sort_keys=True), request_hash=r["request_hash"]) for r in results])
        import csv
        curves = []
        for method in METHODS:
            with (output / method / "brier_curve.csv").open(newline="") as stream:
                curves.extend(csv.DictReader(stream))
        write_csv(output / "brier_curves.csv", curves)
        print(f"Saved {output / 'comparison.csv'} ({manifest['config']['profile']})", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "summarize"))
    parser.add_argument("--methods", default="all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="160/80 rows and tiny budgets; never scientific results")
    args = parser.parse_args(argv)
    methods = list(METHODS) if args.methods == "all" else args.methods.split(",")
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        parser.error(f"Choose methods from {METHODS}")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    output = (args.output_dir or ROOT / "output" / ("support_smoke" if args.smoke else "support")).resolve()
    if args.command == "summarize":
        summarize(output)
        return
    if args.command == "preflight":
        print(json.dumps(preflight(methods, args.device), indent=2))
        return
    config = default_config(args.smoke)
    data = load_support(args.data, smoke=args.smoke, seed=config["seed"])
    request_hash, splits, grid, lower = initialize(output, data, config)
    if args.command == "prepare":
        print(json.dumps(dict(output=str(output), request_hash=request_hash, dataset=data.metadata(),
                              nrt_cv_fits=2*config["nrt_search"]["trials"]*config["folds"]+config["folds"],
                              nncde_candidates=np.prod([len(v) for v in config["nncde_grid"].values()]).item()), indent=2))
        return
    versions = preflight(methods, args.device)
    scratch = output / ".tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    for method in methods:
        print(f"Starting {method} [{config['profile']}]", flush=True)
        run_method(method, data, config, splits, grid, lower, output, request_hash, args.device, versions)
    # One final explicit summarize job publishes the complete server comparison.
    # A single-process run can publish immediately without racing worker writers.
    if set(methods) == set(METHODS):
        summarize(output)


if __name__ == "__main__":
    main()
