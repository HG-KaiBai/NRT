"""Four-fold, single-dataset Brier TPE search; every candidate learns rho."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import torch

from .cv_train import CVOptimization, fit_cv
from .depth_experiment import _evaluation_data, _generate
from .diagnostic_experiment import architecture_from, expanded_settings
from .experiment import ExperimentConfig
from .metrics import average_grid_sse
from .model import ModelConfig
from .parameter_search import _coordinates, _history, _initial_candidates, _optuna, _suggest
from .search_io import write_csv
from .train import TrainingConfig, evaluate_nll
from .tuning_model import Architecture


METHODS = ("adaptive_baseline", "brier_cv", "nll_cv_control")
SOURCE_FILES = ("tree.py", "model.py", "train.py", "simulation.py", "experiment.py",
    "regularization.py", "depth_experiment.py", "metrics.py", "tuning_model.py",
    "tuning_train.py", "brier.py", "parameter_search.py", "search_io.py",
    "stable_train.py", "diagnostic_experiment.py", "cv_train.py", "cv_search.py")


@dataclass(frozen=True)
class CVSearchConfig:
    trials: int = 64
    initial_trials: int = 16
    folds: int = 4
    fold_seed: int = 20261010
    search_seed: int = 871
    search_fit_seed: int = 101
    depth_min: int = 3
    depth_max: int = 8
    widths: tuple[int, ...] = (16, 32, 64, 128)
    shared_layers_max: int = 3
    private_layers_max: int = 4
    strength_max: float = 100000.


def fold_indices(n, folds=4, seed=20261010):
    if not 2 <= folds <= n:
        raise ValueError("invalid fold count")
    order = np.random.default_rng(seed).permutation(n)
    held_out = np.array_split(order, folds)
    return [(torch.as_tensor(np.concatenate([v for j, v in enumerate(held_out) if j != i])),
             torch.as_tensor(valid)) for i, valid in enumerate(held_out)]


def observed_scale(observed):
    bounds = torch.cat((observed[1], observed[2]))
    positive = bounds[torch.isfinite(bounds) & (bounds > 0)]
    return float(positive.median()) if len(positive) else 1.


def observed_pool(setting, config, baseline):
    """First 80% only. Does not return test subjects or latent truth."""
    n = int(config.n_samples*config.train_fraction)+int(config.n_samples*config.validation_fraction)
    observed = _generate(setting, config, ModelConfig(x_dim=baseline.x_dim), config.data_seed)[:3]
    pool = tuple(v[:n] for v in observed)
    bounds = torch.cat((pool[1], pool[2]))
    max_obs = float(bounds[torch.isfinite(bounds)].max())
    if not max_obs > 0:
        raise ValueError("need positive max_obs")
    return dict(pool=pool, grid=torch.linspace(0, max_obs, config.grid_size),
                max_obs=max_obs, n_pool=n, time_scale=observed_scale(pool))


def refit_epoch_count(epochs, warmup_epochs):
    if not epochs or min(epochs) <= warmup_epochs:
        raise ValueError("all CV checkpoints must have learned rho")
    return int(math.ceil(float(np.median(epochs))))


def _json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _manifest(output, config, search, training, optimization, baseline, device):
    if config.resample_data or search.search_fit_seed in config.seeds or config.data_seed in config.seeds:
        raise ValueError("CV uses one independent dataset; final data seeds must exclude the CV data seed")
    expanded_settings(config)
    if (not 2 <= search.initial_trials <= search.trials or search.folds < 2
            or not 1 <= search.depth_min <= baseline.depth <= search.depth_max
            or baseline.shared_width not in search.widths or baseline.private_width not in search.widths
            or min(search.widths) < 1 or search.shared_layers_max < baseline.shared_layers
            or search.private_layers_max < baseline.private_layers
            or not baseline.regularization_strength <= search.strength_max < float("inf")):
        raise ValueError("invalid search space")
    n = int(config.n_samples*config.train_fraction)+int(config.n_samples*config.validation_fraction)
    if n < search.folds or training.epochs <= optimization.warmup_epochs:
        raise ValueError("insufficient CV data or learned-rho epochs")
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update((Path(__file__).parent/name).read_bytes())
    spec = dict(experiment=asdict(config), search=asdict(search), training=asdict(training),
        optimization=asdict(optimization), baseline=asdict(baseline), device=str(device),
        source_sha256=digest.hexdigest(), methods=METHODS,
        selector="mean original neural_only Brier over four held-out folds",
        nll_control="rerank the same Brier-driven TPE candidate pool by mean CV NLL",
        checkpoint="validation NLL, only after warmup", n_fit_pool=n,
        grid=f"{config.grid_size} equally spaced points 0..max finite observed bound in first 80% pool",
        refit="all first 80%; ceil(median of four best CV epochs); no test stopping",
        final_replicates="seed is BOTH data-generation seed and initialization seed; new train/test pair per seed",
        regularizer="same width KL as earlier, endpoints subtracted in float64",
        all_candidates_learn_rho=True)
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output/"config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("CV source/config changed; use a new result directory")
    else:
        _json(path, dict(spec, fingerprint=fingerprint, torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None,
            optuna=_optuna().__version__))
    splits = fold_indices(n, search.folds, search.fold_seed)
    folds_path = output/"folds.json"
    expected = dict(n_pool=n, seed=search.fold_seed,
        validation_indices=[v.tolist() for _, v in splits])
    if folds_path.exists():
        if json.loads(folds_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("cached folds differ")
    else:
        _json(folds_path, expected)
    return fingerprint, splits


def _startup(search, baseline, seed):
    candidates = _initial_candidates(search, baseline, seed)
    # Deterministic same-architecture anchors test strong shrinkage without
    # removing any trainable split weights. Remaining startup points are Sobol.
    strengths = [a for a in (1000., 10000., 100000.) if a <= search.strength_max]
    for index, strength in enumerate(strengths[:max(0, len(candidates)-2)]):
        candidates[-index-1] = _coordinates(replace(baseline, regularization_strength=strength))
    return candidates


def tune_cv(output_dir, *, config=ExperimentConfig(), search=CVSearchConfig(),
            training=TrainingConfig(patience=50), optimization=CVOptimization(),
            baseline=Architecture(), device="cuda", stop_after_settings=None):
    """All four folds score every candidate; there is no seed confirmation stage."""
    output = Path(output_dir).resolve()
    fingerprint, splits = _manifest(output, config, search, training, optimization, baseline, device)
    optuna = _optuna()
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{output/'cv.sqlite3'}")
    trial_rows, fold_rows, selections = [], [], []
    cache_dir = output/"fold_cache"
    cache_dir.mkdir(exist_ok=True)
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
                folds.append((train, valid, observed_scale(train)))
            study = optuna.create_study(study_name=name, storage=storage,
                direction="minimize", load_if_exists=True)
            if not study.trials:
                for candidate in _startup(search, baseline, search.search_seed+index):
                    study.enqueue_trial(candidate)
            while True:
                completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
                if len(completed) >= search.trials:
                    break
                running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
                if len(running) > 1:
                    raise RuntimeError("multiple running trials: do not share this study across processes")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                    study.sampler = optuna.samplers.TPESampler(
                        seed=search.search_seed+1009*index+len(completed),
                        n_startup_trials=search.initial_trials, multivariate=True, group=True,
                        n_ei_candidates=48)
                trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
                architecture = _suggest(trial, search, baseline)
                trial.set_user_attr("architecture", asdict(architecture))
                cache_path = cache_dir/f"{name}_trial{trial.number}.json"
                if cache_path.exists():
                    cache = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cache["fingerprint"] != fingerprint or cache["architecture"] != asdict(architecture):
                        raise ValueError("fold cache mismatch")
                else:
                    cache = dict(fingerprint=fingerprint, architecture=asdict(architecture), folds=[])
                for fold, (train, valid, scale) in enumerate(folds):
                    if any(r["fold"] == fold for r in cache["folds"]):
                        continue
                    started = time.perf_counter()
                    model, history = fit_cv(train, valid, data["grid"], architecture=architecture,
                        training=training, optimization=optimization, time_scale=scale,
                        seed=search.search_fit_seed, device=device)
                    record = dict(setting=name, family=family, censor_type=setting.censor_type,
                        q=setting.q, trial=trial.number, fold=fold, seed=search.search_fit_seed,
                        n_valid=len(valid[0]), time_scale=scale, max_obs=data["max_obs"],
                        fit_seconds=time.perf_counter()-started, **asdict(architecture), **model.fit_info)
                    _history(output, f"{name}_trial{trial.number}_fold{fold}", history)
                    cache["folds"].append(record)
                    _json(cache_path, cache)
                    del model
                records = cache["folds"]
                if sorted(r["fold"] for r in records) != list(range(search.folds)):
                    raise RuntimeError("incomplete or duplicated fold scores")
                weights = [r["n_valid"] for r in records]
                scores = dict(cv_brier=float(np.average([r["validation_brier"] for r in records], weights=weights)),
                    cv_nll=float(np.average([r["validation_nll"] for r in records], weights=weights)),
                    cv_brier_fold_std=float(np.std([r["validation_brier"] for r in records], ddof=1)),
                    cv_nll_fold_std=float(np.std([r["validation_nll"] for r in records], ddof=1)),
                    refit_epochs=refit_epoch_count([r["best_epoch"] for r in records], optimization.warmup_epochs),
                    cv_best_epochs=[r["best_epoch"] for r in sorted(records, key=lambda r: r["fold"])])
                trial.set_user_attr("scores", scores)
                study.tell(trial, scores["cv_brier"])
                if (len(completed)+1) % 4 == 0:
                    print(f"CV [{index+1}/{len(expanded_settings(config))}] {name}: "
                          f"{len(completed)+1}/{search.trials} candidates x {search.folds} folds; "
                          f"best Brier={study.best_value:.6g}", flush=True)
            rows = []
            for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
                row = dict(setting=name, family=family, censor_type=setting.censor_type, q=setting.q,
                    trial=t.number, **t.user_attrs["architecture"], **t.user_attrs["scores"])
                rows.append(row)
                trial_rows.append(row)
                fold_rows.extend(json.loads((cache_dir/f"{name}_trial{t.number}.json").read_text(encoding="utf-8"))["folds"])
            frame = pd.DataFrame(rows)
            baseline_trial = next(r for r in rows if all(np.isclose(r[k], v, rtol=1e-10, atol=1e-10)
                for k, v in asdict(baseline).items()))
            for method, row in ((METHODS[0], baseline_trial),
                    (METHODS[1], frame.sort_values(["cv_brier", "cv_nll", "trial"]).iloc[0].to_dict()),
                    (METHODS[2], frame.sort_values(["cv_nll", "cv_brier", "trial"]).iloc[0].to_dict())):
                selections.append(dict(method=method, **row))
            write_csv(pd.DataFrame(trial_rows), output/"trials.csv")
            write_csv(pd.DataFrame(fold_rows), output/"cv_folds.csv")
            write_csv(pd.DataFrame(selections), output/"selected_parameters.csv")
            print(f"SELECTED {name}: "+"; ".join(f"{r['method']} trial{r['trial']} "
                f"a={r['regularization_strength']:.5g}, epochs={r['refit_epochs']}"
                for r in selections[-len(METHODS):]), flush=True)
        if len(entries) == len(expanded_settings(config)):
            _json(output/"selection_complete.json", dict(fingerprint=fingerprint,
                settings=len(expanded_settings(config)), methods=METHODS,
                selected_sha256=hashlib.sha256((output/"selected_parameters.csv").read_bytes()).hexdigest()))
    finally:
        storage.remove_session()
        storage.engine.dispose()
    return pd.DataFrame(selections), pd.DataFrame(trial_rows), pd.DataFrame(fold_rows)


def evaluate_cv(output_dir, *, config=ExperimentConfig(), search=CVSearchConfig(),
                training=TrainingConfig(patience=50), optimization=CVOptimization(),
                baseline=Architecture(), device="cuda"):
    output = Path(output_dir).resolve()
    fingerprint, _ = _manifest(output, config, search, training, optimization, baseline, device)
    frozen = json.loads((output/"selection_complete.json").read_text(encoding="utf-8"))
    if (frozen["fingerprint"] != fingerprint or frozen["settings"] != len(expanded_settings(config))
            or frozen["selected_sha256"] != hashlib.sha256((output/"selected_parameters.csv").read_bytes()).hexdigest()):
        raise ValueError("freeze all settings before accessing test truth")
    selected = pd.read_csv(output/"selected_parameters.csv").set_index(["setting", "method"])
    path = output/"runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["setting"], r["method"], int(r["seed"])) for r in rows}
    (output/"curves").mkdir(exist_ok=True)
    total = len(expanded_settings(config))*len(METHODS)*len(config.seeds)
    for name, family, setting, actual in expanded_settings(config):
        # Each final seed generates a NEW train/test combination. The tuning
        # dataset (config.data_seed) is independent of every final replicate.
        replicate_data = {}
        for method in METHODS:
            chosen = selected.loc[(name, method)]
            architecture = architecture_from(chosen, baseline)
            epochs = int(chosen.refit_epochs)
            for seed in config.seeds:
                if (name, method, seed) in done:
                    continue
                if seed not in replicate_data:
                    replicate_data[seed] = _evaluation_data(setting, actual,
                        ModelConfig(x_dim=baseline.x_dim), seed)
                data = replicate_data[seed]
                pool = tuple(torch.cat((a, b)) for a, b in zip(data["train"], data["valid"]))
                scale = observed_scale(pool)
                started = time.perf_counter()
                model, history = fit_cv(pool, architecture=architecture, training=training,
                    optimization=optimization, time_scale=scale, seed=seed, device=device, fixed_epochs=epochs)
                prediction = model.predict_survival(data["test"][0], data["grid"]).cpu()
                prefix = f"{name}_{method}_seed{seed}"
                curves_path = output/"curves"/f"{prefix}.npz"
                np.savez_compressed(curves_path, grid=data["grid"].numpy(),
                    truth=data["truth"][:3].numpy(), survival=prediction[:3].numpy(),
                    subject_grid_sse=(prediction-data["truth"]).square().sum(dim=1).numpy())
                _history(output, prefix, history)
                rows.append(dict(setting=name, family=family, censor_type=setting.censor_type,
                    q=setting.q, method=method, seed=seed, data_seed=seed, initialization_seed=seed,
                    n_test=len(data["test"][0]), trial=int(chosen.trial),
                    **asdict(architecture), **model.fit_info, time_scale=scale,
                    fit_seconds=time.perf_counter()-started,
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
