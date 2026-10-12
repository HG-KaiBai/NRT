"""Current KL search: fixed/adaptive raw-time trees with train-only finite support."""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc
import torch

from ..cv_search import (CVSearchConfig, SOURCE_FILES as CV_SOURCES, _startup,
                        fold_indices, observed_pool, refit_epoch_count)
from ..depth_experiment import _evaluation_data
from ..diagnostic_experiment import architecture_from, expanded_settings
from ..experiment import ExperimentConfig
from ..metrics import average_grid_sse
from ..model import ModelConfig
from ..nll_cv_train import NLLCVOptimization
from .model import fit_raw_time_kl as fit_nll_cv, SUPPORT
from ..nll_cv_io import write_json as _json
from pycode.finite_support_kl.io import WriterLock
from functools import wraps
from ..parameter_search import _history, _optuna, _suggest
from ..search_io import write_csv
from ..train import TrainingConfig, evaluate_nll
from ..tuning_model import Architecture


SEARCH_METHODS = ("fixed_nll_cv", "adaptive_nll_cv")
MATCHED = "fixed_at_adaptive_architecture"
METHODS = SEARCH_METHODS+(MATCHED,)
SOURCE_FILES = tuple("../"+name for name in CV_SOURCES+("nll_cv_train.py",)) + (
    "../../finite_support_kl/model.py", "../../finite_support_kl/train.py",
    "model.py", "search.py", "report.py")


def _single_writer(function):
    @wraps(function)
    def guarded(output_dir, *args, **kwargs):
        with WriterLock(Path(output_dir).resolve()):
            return function(output_dir, *args, **kwargs)
    return guarded


def _manifest(output, config, search, training, optimization, baseline, device):
    entries = expanded_settings(config)
    if (config.resample_data or config.data_seed in config.seeds or search.search_fit_seed in config.seeds
            or not optimization.learn_splits or not 2 <= search.initial_trials <= search.trials
            or not 2 <= search.folds or not 1 <= search.depth_min <= baseline.depth <= search.depth_max
            or baseline.shared_width not in search.widths or baseline.private_width not in search.widths
            or min(search.widths) < 1 or search.shared_layers_max < baseline.shared_layers
            or search.private_layers_max < baseline.private_layers
            or not baseline.regularization_strength <= search.strength_max < float("inf")
            or training.epochs <= optimization.warmup_epochs):
        raise ValueError("invalid NLL CV configuration")
    n = int(config.n_samples*config.train_fraction)+int(config.n_samples*config.validation_fraction)
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update((Path(__file__).parent/name).read_bytes())
    spec = dict(experiment=asdict(config), search=asdict(search), training=asdict(training),
        optimization=asdict(optimization), baseline=asdict(baseline), device=str(device),
        source_sha256=digest.hexdigest(), methods=METHODS, search_methods=SEARCH_METHODS,
        support=SUPPORT, time_coordinate="raw",
        upper_bound_rule="1.1 * maximum finite observed L/R endpoint in each fit subset",
        width_kl="-2/(depth*L) * sum log(width/(upper_bound/L)); lambda=a/sqrt(n_fit)",
        settings=len(entries), n_fit_pool=n,
        selector="mean held-out observed-data NLL; TPE objective and final selector both NLL",
        fixed_search="five effective architecture parameters, penalty a=0 because width KL is zero",
        adaptive_search="six parameters including finite penalty a in 0..strength_max",
        matched_control="fixed rho=0.5 at adaptive-selected architecture; own four-fold NLL refit epoch",
        checkpoint="same window for both methods: validation NLL after warmup_epochs",
        final_replicates="new train/test pair per data seed; same data and initialization seed for all methods",
        refit="all first 80%; ceil(median of four best CV epochs); no test stopping",
        metric="sum squared CDF errors on equally spaced 0..training max_obs grid; mean test subjects")
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output/"config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("NLL CV source/config changed; use a new result directory")
    else:
        _json(path, dict(spec, fingerprint=fingerprint, torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None,
            optuna=_optuna().__version__))
    splits = fold_indices(n, search.folds, search.fold_seed)
    expected = dict(n_pool=n, seed=search.fold_seed, validation_indices=[v.tolist() for _, v in splits])
    path = output/"folds.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("cached fold partition changed")
    else:
        _json(path, expected)
    return fingerprint, splits


def _fixed_startup(search, baseline, seed):
    first = dict(depth=baseline.depth, shared_layers=baseline.shared_layers,
        shared_width=baseline.shared_width, private_width=baseline.private_width,
        private_layers=baseline.private_layers)
    candidates, seen = [first], {tuple(first.values())}
    sampler = qmc.Sobol(5, scramble=True, seed=seed)
    points = sampler.random_base2(math.ceil(math.log2(max(32, 8*search.initial_trials))))
    for u in points:
        candidate = dict(depth=search.depth_min+int(u[0]*(1+search.depth_max-search.depth_min)),
            shared_layers=int(u[1]*(search.shared_layers_max+1)),
            shared_width=search.widths[int(u[2]*len(search.widths))],
            private_width=search.widths[int(u[3]*len(search.widths))],
            private_layers=1+int(u[4]*search.private_layers_max))
        if not candidate["shared_layers"]:
            candidate["shared_width"] = baseline.shared_width
        key = tuple(candidate.values())
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
        if len(candidates) == search.initial_trials:
            return candidates
    raise ValueError("fixed-rho startup space too small")


def _suggest_fixed(trial, search, baseline):
    shared_layers = trial.suggest_int("shared_layers", 0, search.shared_layers_max)
    return replace(baseline, depth=trial.suggest_int("depth", search.depth_min, search.depth_max),
        shared_layers=shared_layers, shared_width=trial.suggest_categorical("shared_width", search.widths)
            if shared_layers else baseline.shared_width,
        private_width=trial.suggest_categorical("private_width", search.widths),
        private_layers=trial.suggest_int("private_layers", 1, search.private_layers_max),
        regularization_strength=0.)


def _candidate(output, fingerprint, name, family, setting, method, trial_number,
               architecture, folds, training, optimization, search, device):
    prefix = f"{name}_{method}_trial{trial_number}"
    cache_path = output/"fold_cache"/f"{prefix}.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache["fingerprint"] != fingerprint or cache["architecture"] != asdict(architecture):
            raise ValueError("NLL fold cache mismatch")
    else:
        cache = dict(fingerprint=fingerprint, architecture=asdict(architecture), folds=[])
    for fold, (train, valid) in enumerate(folds):
        if any(r["fold"] == fold for r in cache["folds"]):
            continue
        started = time.perf_counter()
        model, history = fit_nll_cv(train, valid, architecture=architecture,
            training=training, optimization=replace(optimization, learn_splits=method == SEARCH_METHODS[1]),
            seed=search.search_fit_seed, device=device)
        record = dict(setting=name, family=family, censor_type=setting.censor_type, q=setting.q,
            method=method, trial=trial_number, fold=fold, seed=search.search_fit_seed,
            n_valid=len(valid[0]), fit_seconds=time.perf_counter()-started,
            **asdict(architecture), **model.fit_info)
        _history(output, f"{prefix}_fold{fold}", history)
        cache["folds"].append(record)
        _json(cache_path, cache)
        del model
    records = sorted(cache["folds"], key=lambda r: r["fold"])
    if [r["fold"] for r in records] != list(range(search.folds)):
        raise RuntimeError("incomplete or duplicate fold scores")
    scores = dict(cv_nll=float(np.average([r["validation_nll"] for r in records],
        weights=[r["n_valid"] for r in records])),
        cv_nll_fold_std=float(np.std([r["validation_nll"] for r in records], ddof=1)),
        refit_epochs=refit_epoch_count([r["best_epoch"] for r in records], optimization.warmup_epochs),
        cv_best_epochs=[r["best_epoch"] for r in records])
    return scores, records


@_single_writer
def tune_nll_cv(output_dir, *, config=ExperimentConfig(), search=CVSearchConfig(),
                training=TrainingConfig(patience=50), optimization=NLLCVOptimization(),
                baseline=Architecture(), device="cuda", stop_after_settings=None):
    output = Path(output_dir).resolve()
    fingerprint, splits = _manifest(output, config, search, training, optimization, baseline, device)
    (output/"fold_cache").mkdir(exist_ok=True)
    optuna = _optuna()
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{output/'nll_cv.sqlite3'}")
    trial_rows, fold_rows, selections = [], [], []
    entries = expanded_settings(config)
    if stop_after_settings is not None:
        entries = entries[:stop_after_settings]
    try:
        for index, (name, family, setting, actual) in enumerate(entries):
            data = observed_pool(setting, actual, baseline)
            folds = []
            for train_index, valid_index in splits:
                train = tuple(v[train_index] for v in data["pool"])
                valid = tuple(v[valid_index] for v in data["pool"])
                folds.append((train, valid))
            for method in SEARCH_METHODS:
                study = optuna.create_study(study_name=f"{name}_{method}", storage=storage,
                    direction="minimize", load_if_exists=True)
                if not study.trials:
                    startup = _startup if method == SEARCH_METHODS[1] else _fixed_startup
                    for candidate in startup(search, baseline, search.search_seed+index):
                        study.enqueue_trial(candidate)
                while True:
                    completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
                    if len(completed) >= search.trials:
                        break
                    running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
                    if len(running) > 1:
                        raise RuntimeError("do not share this study across processes")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                        study.sampler = optuna.samplers.TPESampler(
                            seed=search.search_seed+1009*index+len(completed),
                            n_startup_trials=search.initial_trials, multivariate=True, group=True,
                            n_ei_candidates=48)
                    trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
                    architecture = (_suggest if method == SEARCH_METHODS[1] else _suggest_fixed)(trial, search, baseline)
                    trial.set_user_attr("architecture", asdict(architecture))
                    scores, _ = _candidate(output, fingerprint, name, family, setting, method,
                        trial.number, architecture, folds, training, optimization, search, device)
                    trial.set_user_attr("scores", scores)
                    study.tell(trial, scores["cv_nll"])
                    if (len(completed)+1) % 8 == 0:
                        print(f"NLL CV [{index+1}/{len(expanded_settings(config))}] {name} {method}: "
                            f"{len(completed)+1}/{search.trials}, best NLL={study.best_value:.7g}", flush=True)
                rows = []
                for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
                    row = dict(setting=name, family=family, censor_type=setting.censor_type,
                        q=setting.q, method=method, trial=t.number, objective_value=t.value,
                        **t.user_attrs["architecture"], **t.user_attrs["scores"])
                    rows.append(row)
                    trial_rows.append(row)
                    prefix = f"{name}_{method}_trial{t.number}"
                    fold_rows.extend(json.loads((output/"fold_cache"/f"{prefix}.json").read_text(encoding="utf-8"))["folds"])
                winner = pd.DataFrame(rows).sort_values(["cv_nll", "trial"]).iloc[0].to_dict()
                selections.append(winner)
            adaptive = selections[-1]
            architecture = replace(architecture_from(adaptive, baseline), regularization_strength=0.)
            scores, records = _candidate(output, fingerprint, name, family, setting, MATCHED,
                -1, architecture, folds, training, optimization, search, device)
            fold_rows.extend(records)
            selections.append(dict(setting=name, family=family, censor_type=setting.censor_type,
                q=setting.q, method=MATCHED, trial=-1, source_adaptive_trial=int(adaptive["trial"]),
                **asdict(architecture), **scores))
            write_csv(pd.DataFrame(trial_rows), output/"trials.csv")
            write_csv(pd.DataFrame(fold_rows), output/"cv_folds.csv")
            write_csv(pd.DataFrame(selections), output/"selected_parameters.csv")
            print(f"SELECTED {name}: "+"; ".join(f"{r['method']} D{r['depth']} "
                f"a={r['regularization_strength']:.5g}, epochs={r['refit_epochs']}, NLL={r['cv_nll']:.7g}"
                for r in selections[-len(METHODS):]), flush=True)
        if len(entries) == len(expanded_settings(config)):
            _json(output/"selection_complete.json", dict(fingerprint=fingerprint,
                settings=len(entries), methods=METHODS,
                selected_sha256=hashlib.sha256((output/"selected_parameters.csv").read_bytes()).hexdigest()))
    finally:
        storage.remove_session()
        storage.engine.dispose()
    return pd.DataFrame(selections), pd.DataFrame(trial_rows), pd.DataFrame(fold_rows)


@_single_writer
def evaluate_nll_cv(output_dir, *, config=ExperimentConfig(), search=CVSearchConfig(),
                    training=TrainingConfig(patience=50), optimization=NLLCVOptimization(),
                    baseline=Architecture(), device="cuda"):
    output = Path(output_dir).resolve()
    fingerprint, _ = _manifest(output, config, search, training, optimization, baseline, device)
    frozen = json.loads((output/"selection_complete.json").read_text(encoding="utf-8"))
    if (frozen["fingerprint"] != fingerprint or frozen["settings"] != len(expanded_settings(config))
            or frozen["selected_sha256"] != hashlib.sha256((output/"selected_parameters.csv").read_bytes()).hexdigest()):
        raise ValueError("freeze all parameters before accessing test truth")
    selected = pd.read_csv(output/"selected_parameters.csv", float_precision="round_trip").set_index(["setting", "method"])
    path = output/"runs.csv"
    rows = pd.read_csv(path, float_precision="round_trip").to_dict("records") if path.exists() else []
    done = {(r["setting"], r["method"], int(r["seed"])) for r in rows}
    (output/"curves").mkdir(exist_ok=True)
    total = len(expanded_settings(config))*len(METHODS)*len(config.seeds)
    for name, family, setting, actual in expanded_settings(config):
        replicate_data = {}
        for method in METHODS:
            chosen = selected.loc[(name, method)]
            architecture = architecture_from(chosen, baseline)
            epochs = int(chosen.refit_epochs)
            for seed in config.seeds:
                if (name, method, seed) in done:
                    continue
                if seed not in replicate_data:
                    replicate_data[seed] = _evaluation_data(setting, actual, ModelConfig(x_dim=baseline.x_dim), seed)
                data = replicate_data[seed]
                pool = tuple(torch.cat((a, b)) for a, b in zip(data["train"], data["valid"]))
                started = time.perf_counter()
                model, history = fit_nll_cv(pool, architecture=architecture, training=training,
                    optimization=replace(optimization, learn_splits=method == SEARCH_METHODS[1]),
                    seed=seed, device=device, fixed_epochs=epochs)
                prediction = model.predict_survival(data["test"][0], data["grid"]).cpu()
                prefix = f"{name}_{method}_seed{seed}"
                curves_path = output/"curves"/f"{prefix}.npz"
                np.savez_compressed(curves_path, grid=data["grid"].numpy(), truth=data["truth"][:3].numpy(),
                    survival=prediction[:3].numpy(),
                    subject_grid_sse=(prediction-data["truth"]).square().sum(dim=1).numpy())
                _history(output, prefix, history)
                rows.append(dict(setting=name, family=family, censor_type=setting.censor_type, q=setting.q,
                    method=method, seed=seed, data_seed=seed, initialization_seed=seed,
                    n_test=len(data["test"][0]), trial=int(chosen.trial), **asdict(architecture),
                    **model.fit_info, fit_seconds=time.perf_counter()-started,
                    average_grid_sse=float(average_grid_sse(prediction, data["truth"])),
                    test_nll=evaluate_nll(model, *data["test"]), max_obs=data["max_obs"],
                    curves_file=str(curves_path), histories_file=str(output/"histories"/f"{prefix}.json.gz")))
                done.add((name, method, seed))
                write_csv(pd.DataFrame(rows), path)
                del model
            values = [r["average_grid_sse"] for r in rows if r["setting"] == name and r["method"] == method]
            print(f"TEST [{len(rows)}/{total}] {name} {method}: "
                f"{np.mean(values):.6f} +/- {np.std(values, ddof=1):.6f}", flush=True)
    runs = pd.DataFrame(rows)
    if len(runs) != total or runs.duplicated(["setting", "method", "seed"]).any():
        raise RuntimeError("incomplete or duplicate final evaluations")
    summary = runs.groupby(["setting", "family", "censor_type", "q", "method"], sort=False).agg(
        n_runs=("seed", "size"), average_grid_sse_mean=("average_grid_sse", "mean"),
        average_grid_sse_std=("average_grid_sse", "std"),
        rho_mean_absolute_shift=("rho_mean_absolute_shift", "mean"),
        rho_root_mean_absolute_shift=("rho_root_mean_absolute_shift", "mean")).reset_index()
    write_csv(summary, output/"summary.csv")
    return runs, summary
