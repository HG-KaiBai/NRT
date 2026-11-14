"""Sobol/anchor + TPE NLL CV; export fixed configurations for simulation."""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import shutil
import sys
import time
import warnings

import numpy as np
from scipy.stats import qmc
import torch

from ..model import PENALTIES, observed_upper_bound
from ..train import fit_raw_time_kl
from ..unified_nll.nll_cv_train import NLLCVOptimization
from ..unified_nll.train import TrainingConfig
from ..unified_nll.tuning_model import Architecture
from ..workflow import (ROOT, METHODS, SEARCH_METHODS, MATCHED, SCHEMA_VERSION,
                        atomic_json, cv_splits, dataset_hash, evaluate, export_tuning,
                        fingerprint, preflight, select_cases, select_methods,
                        source_hash, tuning_data, write_csv, writer_lock)


@dataclass(frozen=True)
class SearchConfig:
    trials: int = 200
    initial_trials: int = 20
    folds: int = 4
    fold_seed: int = 321
    search_seed: int = 871
    search_fit_seed: int = 101
    depth_min: int = 3
    depth_max: int = 8
    widths: tuple = (16, 32, 64, 128)
    shared_layers_max: int = 3
    private_layers_max: int = 4
    strength_max: float = 10000.


def optuna_module():
    try:
        import optuna
    except ImportError:
        sys.path.append(str(ROOT/".deps"))
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
    first = baseline if adaptive else replace(baseline, regularization_strength=0.)
    result = [coordinates(first, adaptive)]
    if adaptive:
        result.append(coordinates(replace(baseline, regularization_strength=0.), True))
        for strength in (1000., 10000.):
            if strength <= search.strength_max:
                result.append(coordinates(replace(baseline, regularization_strength=strength), True))
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
    raise ValueError("startup budget exceeds the available architecture space")


def suggest(trial, search, baseline, adaptive):
    shared = trial.suggest_int("shared_layers", 0, search.shared_layers_max)
    return replace(baseline,
        depth=trial.suggest_int("depth", search.depth_min, search.depth_max), shared_layers=shared,
        shared_width=trial.suggest_categorical("shared_width", search.widths) if shared else baseline.shared_width,
        private_width=trial.suggest_categorical("private_width", search.widths),
        private_layers=trial.suggest_int("private_layers", 1, search.private_layers_max),
        regularization_strength=10**trial.suggest_float("log10_one_plus_strength", 0.,
            math.log10(1+search.strength_max))-1 if adaptive else 0.)


def score_candidate(data, splits, architecture, training, optimization, search, device,
                    *, upper_bound, penalty="partition_kl"):
    """Keep fold diagnostics in memory; only the aggregate enters CV results."""
    records = []
    for fold, (train, valid) in enumerate(splits, 1):
        model, _ = fit_raw_time_kl(data.bounds(train), data.bounds(valid),
            architecture=architecture, training=training, optimization=optimization,
            seed=search.search_fit_seed, device=device, upper_bound=upper_bound, penalty=penalty)
        record = dict(fold=fold, n_valid=len(valid), **model.fit_info)
        del model
        records.append(record)
    epochs = [r["best_epoch"] for r in records]
    return dict(cv_score=float(np.average([r["validation_nll"] for r in records],
                                          weights=[r["n_valid"] for r in records])),
                cv_best_epochs=epochs, refit_epochs=math.ceil(float(np.median(epochs)))), records


def search_method(folder, request_hash, data, splits, method, search, baseline, training, optimization, device,
                  *, upper_bound, penalty="partition_kl", cv_path=None):
    optuna = optuna_module()
    adaptive = method == SEARCH_METHODS[1]
    optimization = replace(optimization, learn_splits=adaptive)
    folder.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{(folder/'study.sqlite3').resolve()}")
    study = optuna.create_study(study_name=method, direction="minimize", storage=storage, load_if_exists=True)
    cv_path = folder/"cv_results.csv" if cv_path is None else cv_path
    if not study.trials:
        for row in startup(search, baseline, adaptive, search.search_seed+data.case.index):
            study.enqueue_trial(row)
    while True:
        completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if len(completed) >= search.trials:
            break
        running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
        if len(running) > 1:
            raise RuntimeError("study has multiple running trials")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            study.sampler = optuna.samplers.TPESampler(seed=search.search_seed+1009*data.case.index+len(completed),
                n_startup_trials=search.initial_trials, multivariate=True, group=True, n_ei_candidates=48)
        trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
        architecture = suggest(trial, search, baseline, adaptive)
        scores, _ = score_candidate(data, splits, architecture, training, optimization, search, device,
            upper_bound=upper_bound, penalty=penalty)
        trial.set_user_attr("parameters", asdict(architecture))
        trial.set_user_attr("scores", scores)
        study.tell(trial, scores["cv_score"])
        write_csv(cv_path, [dict(trial=t.number, **t.user_attrs["parameters"], **t.user_attrs["scores"])
                           for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))])
        print(f"{method} {data.case.id}: trial {len(completed)+1}/{search.trials}, NLL={scores['cv_score']:.7g}", flush=True)
    rows = [dict(trial=t.number, **t.user_attrs["parameters"], **t.user_attrs["scores"])
            for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))]
    write_csv(cv_path, rows)
    winner = min(rows, key=lambda row: (row["cv_score"], row["trial"]))
    storage.remove_session()
    storage.engine.dispose()
    return Architecture(**{key: winner[key] for key in asdict(baseline)}), {
        key: winner[key] for key in ("cv_score", "cv_best_epochs", "refit_epochs", "trial")}


def compact_case_outputs(output, case_id, methods):
    """Discard temporary search state only after final outputs are complete.

    Also accepts the previous fold-cache layout, preserving its CV tables and
    the three final JSON files per method. Incomplete cases are left untouched.
    """
    output = Path(output)
    folder = output/"studies"/case_id
    for method in methods:
        final = output/method/case_id
        if not all((final/name).is_file() for name in ("selection.json", "result.json", "history.json")):
            return False
        result = json.loads((final/"result.json").read_text())
        selection = json.loads((final/"selection.json").read_text())
        if result.get("status") != "ok" or result["request_hash"] != selection["request_hash"]:
            raise ValueError(f"inconsistent final outputs: {final}")
        if not (final/"cv_results.csv").exists() and not (folder/method/"cv_results.csv").exists():
            if method != MATCHED:
                raise ValueError(f"missing CV results: {final}")
    for method in methods:
        final, temporary = output/method/case_id, folder/method
        old_csv = temporary/"cv_results.csv"
        if old_csv.exists():
            if (final/"cv_results.csv").exists() and old_csv.read_bytes() != (final/"cv_results.csv").read_bytes():
                raise ValueError(f"conflicting CV results: {final}")
            old_csv.replace(final/"cv_results.csv")
        if method == MATCHED and not (final/"cv_results.csv").exists():
            selection = json.loads((final/"selection.json").read_text())
            write_csv(final/"cv_results.csv", [dict(**selection["parameters"], **selection["scores"])])
        # Only these known generated artifacts are disposable, never final results.
        if (temporary/"fold_cache").exists():
            shutil.rmtree(temporary/"fold_cache")
        for pattern in ("fold_*.json", "history_*.json", "study.sqlite3*"):
            for path in temporary.glob(pattern):
                path.unlink()
        if temporary.exists() and not any(temporary.iterdir()):
            temporary.rmdir()
    # Shared metadata is removed only when all methods present have final fits.
    remaining = [p for p in folder.iterdir() if p.is_dir()] if folder.exists() else []
    if not remaining:
        for name in ("request.json", "folds.json"):
            (folder/name).unlink(missing_ok=True)
        (output/"cases"/f"{case_id}.json").unlink(missing_ok=True)
    return True


def tune_case(args, case, methods, search, baseline, training, optimization):
    data, seed = tuning_data(case, n_samples=args.n_samples, base_seed=args.seed, mu=args.mu, r=args.r)
    metadata = dict(schema_version=SCHEMA_VERSION, case=asdict(case), case_id=case.id,
                    seed=seed, mu=args.mu, r=args.r, **data.metadata())
    # This manifest intentionally has exactly the same fields as tuning/main.py.
    manifest_path = args.output_dir/"cases"/f"{case.id}.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != metadata:
        raise ValueError("dataset settings changed; use a fresh output directory")
    atomic_json(manifest_path, metadata)
    # Freeze one domain before CV/search, using observed bounds from the 80% pool.
    # Fold-local maxima can exclude validation tails for every architecture.
    upper_bound = observed_upper_bound(data.bounds(data.train))
    request = dict(case=asdict(case), data=metadata, search=asdict(search), parameters=asdict(baseline),
                   training_config=asdict(training), optimization=asdict(optimization), penalty=args.penalty,
                   support=dict(policy="tuning_train_pool", upper_bound=upper_bound),
                   device=args.device, grid_size=args.grid_size, source_sha256=source_hash(), torch=torch.__version__,
                   optuna=optuna_module().__version__)
    request_hash = fingerprint(request)
    folder = args.output_dir/"studies"/case.id
    with writer_lock(folder):
        path = folder/"request.json"
        if path.exists() and json.loads(path.read_text())["request_hash"] != request_hash:
            raise ValueError("source/configuration changed; use a fresh output directory")
        atomic_json(path, dict(request=request, request_hash=request_hash))
        splits = cv_splits(data.train, search.fold_seed, search.folds)
        selected = {}
        # The matched control depends on adaptive tuning, but adds no search.
        needed = [m for m in SEARCH_METHODS if m in methods or (m == METHODS[1] and MATCHED in methods)]
        for method in needed:
            path = args.output_dir/method/case.id/"selection.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["request_hash"] != request_hash:
                    raise ValueError("selection cache mismatch")
                selected[method] = (Architecture(**saved["parameters"]), saved["scores"])
            else:
                selected[method] = search_method(folder/method, request_hash, data, splits, method,
                    search, baseline, training, optimization, args.device, upper_bound=upper_bound,
                    penalty=args.penalty if method == METHODS[1] else "partition_kl",
                    cv_path=args.output_dir/method/case.id/"cv_results.csv")
                architecture, scores = selected[method]
                atomic_json(path, dict(parameters=asdict(architecture), scores=scores, request_hash=request_hash))
        if MATCHED in methods:
            architecture = replace(selected[METHODS[1]][0], regularization_strength=0.)
            path = args.output_dir/MATCHED/case.id/"selection.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["request_hash"] != request_hash or saved["parameters"] != asdict(architecture):
                    raise ValueError("matched selection cache mismatch")
                scores = saved["scores"]
            else:
                scores, _ = score_candidate(data, splits, architecture,
                    training, replace(optimization, learn_splits=False), search, args.device,
                    upper_bound=upper_bound, penalty=args.penalty)
                scores = dict(scores, trial=-1)
                atomic_json(path, dict(parameters=asdict(architecture), scores=scores, request_hash=request_hash))
            selected[MATCHED] = architecture, scores
            write_csv(path.with_name("cv_results.csv"), [dict(**asdict(architecture), **scores)])
        # Freeze all requested selections before any held-out test evaluation.
        for method in methods:
            result_path = args.output_dir/method/case.id/"result.json"
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if old["request_hash"] != request_hash:
                    raise ValueError("result cache mismatch")
                continue
            architecture, scores = selected[method]
            actual_optimization = replace(optimization, learn_splits=method == METHODS[1])
            actual_penalty = args.penalty if method in (METHODS[1], MATCHED) else "partition_kl"
            start = time.monotonic()
            model, history = fit_raw_time_kl(data.bounds(data.train), architecture=architecture,
                training=training, optimization=actual_optimization, fixed_epochs=scores["refit_epochs"],
                seed=search.search_fit_seed, device=args.device, penalty=actual_penalty)
            result = dict(**metadata, method=method, status="ok", parameters=asdict(architecture),
                training_config=asdict(training), optimization=asdict(actual_optimization), penalty=actual_penalty,
                **scores, cv_metric="nll", **evaluate(model, data, args.grid_size),
                fit_info=model.fit_info, elapsed_seconds=time.monotonic()-start,
                base_seed=args.seed, dataset_sha256=dataset_hash(data),
                source_sha256=request["source_sha256"], request_hash=request_hash)
            atomic_json(result_path, result)
            atomic_json(result_path.with_name("history.json"), history)
            del model
        compact_case_outputs(args.output_dir, case.id, methods)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default="all")
    p.add_argument("--penalty", choices=PENALTIES, default="partition_kl",
                   help="adaptive penalty; simulation restores this choice from the exported CSV")
    p.add_argument("--cases", default="all")
    p.add_argument("--task-id", type=int, help="case index in --cases; one GPU owns all selected methods")
    p.add_argument("--output-dir", type=Path, default=ROOT/"tuning/output")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mu", type=float, default=0.)
    p.add_argument("--r", type=float)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--initial-trials", type=int, default=20)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--fold-seed", type=int, default=321)
    p.add_argument("--search-space", type=Path, help="JSON overrides for SearchConfig and baseline Architecture")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--split-lr-ratio", type=float, default=.1)
    p.add_argument("--validation-every", type=int, default=5)
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cases, methods = select_cases(args.cases), select_methods(args.methods)
    if args.summarize:
        export_tuning(args.output_dir, cases, methods)
        for case in cases:
            with writer_lock(args.output_dir/"studies"/case.id):
                compact_case_outputs(args.output_dir, case.id, methods)
        return
    selected_cases = cases
    if args.task_id is not None:
        if not 0 <= args.task_id < len(cases):
            raise ValueError("task-id is outside the selected cases")
        selected_cases = [cases[args.task_id]]
    overrides = json.loads(args.search_space.read_text()) if args.search_space else {}
    search = replace(SearchConfig(trials=args.trials, initial_trials=args.initial_trials,
                                  folds=args.folds, fold_seed=args.fold_seed), **overrides.get("search", {}))
    baseline = Architecture(**overrides.get("baseline", {}))
    if (not 2 <= search.initial_trials <= search.trials or search.folds < 2
            or not 1 <= search.depth_min <= baseline.depth <= search.depth_max
            or not search.widths or len(set(search.widths)) != len(search.widths)
            or min(search.widths) < 1 or baseline.shared_width not in search.widths
            or baseline.private_width not in search.widths
            or not 0 <= baseline.shared_layers <= search.shared_layers_max
            or not 1 <= baseline.private_layers <= search.private_layers_max
            or not baseline.regularization_strength <= search.strength_max < float("inf")
            or args.epochs <= args.warmup_epochs or args.n_samples < 10 or args.grid_size < 3):
        raise ValueError("invalid search/training configuration")
    training = TrainingConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, validation_every=args.validation_every, cuda_graphs=args.cuda_graphs)
    optimization = NLLCVOptimization(warmup_epochs=args.warmup_epochs, split_lr_ratio=args.split_lr_ratio)
    if args.dry_run:
        print(f"{len(selected_cases)} cases; methods={','.join(methods)}; penalty={args.penalty}; "
              f"{search.trials} trials; {search.folds} folds")
        for case in selected_cases:
            print(f"{case.id} data_seed={args.seed+case.index*10_000}")
        return
    preflight(args.device, args.threads)
    for case in selected_cases:
        tune_case(args, case, methods, search, baseline, training, optimization)
    export_tuning(args.output_dir, cases, methods)


if __name__ == "__main__":
    main()
