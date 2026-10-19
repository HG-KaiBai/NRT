"""Reuse current NRT Sobol/anchor+TPE search and the complete NNCDE grid."""
from dataclasses import asdict, replace
import gc
import itertools
import json
import math
import time
import warnings

import numpy as np
import torch

from regularization.model import observed_upper_bound
from regularization.train import fit_raw_time_kl
from regularization.tuning.main import SearchConfig, startup, suggest, optuna_module
from regularization.unified_nll.nll_cv_train import NLLCVOptimization
from regularization.unified_nll.train import TrainingConfig
from regularization.unified_nll.tuning_model import Architecture
from tuning.main import NNCDE_GRID
from workflow import atomic_json, write_csv
from regularization.workflow import writer_lock
from .models import MemoryBoundNNCDE

NRT_METHODS = ("fixed_nll_cv", "adaptive_nll_cv", "fixed_at_adaptive_architecture")


def default_config(smoke=False):
    config = dict(profile="full", seed=123, fold_seed=321, folds=4,
                  nrt_search=asdict(SearchConfig()),
                  nrt_training=asdict(TrainingConfig(patience=50)),
                  nrt_optimization=asdict(NLLCVOptimization()),
                  nncde_grid=NNCDE_GRID, nncde_epochs=200, nncde_patience=20,
                  deepsurv_epochs=800, grid_size=256,
                  ctree=dict(mincriterion=.95, minsplit=20, minbucket=7),
                  iccforest=dict(ntree=100, tune_mtry=True, tune_ntree=100, step_factor=1.5))
    if smoke:
        config.update(profile="smoke", folds=2, nncde_epochs=3, nncde_patience=2,
                      deepsurv_epochs=5, grid_size=32,
                      nncde_grid=dict(hidden_nodes=[8], learning_rate=[.001], batch_size=[32]),
                      iccforest=dict(ntree=5, tune_mtry=False, tune_ntree=5, step_factor=1.5))
        config["nrt_search"].update(trials=2, initial_trials=2, depth_min=3, depth_max=3,
                                    widths=[32], shared_layers_max=0, private_layers_max=3)
        config["nrt_training"].update(epochs=3, patience=2, validation_every=1, cuda_graphs=False)
        config["nrt_optimization"]["warmup_epochs"] = 1
    config["nrt_search"].update(folds=config["folds"], fold_seed=config["fold_seed"])
    return config


def nrt_settings(config):
    return (SearchConfig(**config["nrt_search"]), Architecture(x_dim=14),
            TrainingConfig(**config["nrt_training"]), NLLCVOptimization(**config["nrt_optimization"]))


def score_nrt(data, splits, architecture, config, directory, adaptive, device):
    _, _, training, optimization = nrt_settings(config)
    optimization = replace(optimization, learn_splits=adaptive)
    upper = observed_upper_bound(data.bounds(data.train))
    records = []
    for fold, (train, valid) in enumerate(splits, 1):
        path = directory / f"fold_{fold}.json"
        if path.exists():
            record = json.loads(path.read_text())
        else:
            scaler = data.scaler(train)
            model, history = fit_raw_time_kl(data.bounds(train, scaler), data.bounds(valid, scaler),
                architecture=architecture, training=training, optimization=optimization,
                seed=config["nrt_search"]["search_fit_seed"], device=device, upper_bound=upper)
            record = dict(fold=fold, n_valid=len(valid), **model.fit_info)
            atomic_json(path, record)
            atomic_json(directory / f"history_{fold}.json", history)
            del model
        records.append(record)
    epochs = [r["best_epoch"] for r in records]
    return dict(cv_score=float(np.average([r["validation_nll"] for r in records],
                                         weights=[r["n_valid"] for r in records])),
                cv_best_epochs=epochs, refit_epochs=math.ceil(float(np.median(epochs))))


def select_nrt(method, data, splits, config, output, device):
    directory = output / method
    path = directory / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    search, baseline, _, _ = nrt_settings(config)
    if method == NRT_METHODS[2]:
        with writer_lock(output / NRT_METHODS[1]):
            adaptive_selection = select_nrt(NRT_METHODS[1], data, splits, config, output, device)
        architecture = replace(Architecture(**adaptive_selection["parameters"]), regularization_strength=0.)
        scores = score_nrt(data, splits, architecture, config, directory / "cv", False, device)
        selection = dict(parameters=asdict(architecture), **scores, cv_metric="validation_nll",
                         architecture_source=NRT_METHODS[1])
    else:
        adaptive = method == NRT_METHODS[1]
        optuna = optuna_module()
        directory.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(study_name=method, direction="minimize",
                    storage=f"sqlite:///{(directory / 'study.sqlite3').resolve()}", load_if_exists=True)
        if not study.trials:
            for row in startup(search, baseline, adaptive, search.search_seed):
                study.enqueue_trial(row)
        while True:
            completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            if len(completed) >= search.trials:
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                study.sampler = optuna.samplers.TPESampler(seed=search.search_seed + len(completed),
                    n_startup_trials=search.initial_trials, multivariate=True, group=True, n_ei_candidates=48)
            running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
            if len(running) > 1:
                raise RuntimeError("Multiple running trials in one method; use one worker per method")
            trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
            architecture = suggest(trial, search, baseline, adaptive)
            started = time.monotonic()
            scores = score_nrt(data, splits, architecture, config,
                               directory / "fold_cache" / f"trial_{trial.number}", adaptive, device)
            trial.set_user_attr("parameters", asdict(architecture))
            trial.set_user_attr("scores", scores)
            study.tell(trial, scores["cv_score"])
            print(f"{method}: {len(completed)+1}/{search.trials}, NLL={scores['cv_score']:.6f}, "
                  f"{time.monotonic()-started:.1f}s", flush=True)
        rows = [dict(trial=t.number, **t.user_attrs["parameters"], **t.user_attrs["scores"])
                for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))]
        write_csv(directory / "cv_results.csv", rows)
        winner = min(rows, key=lambda r: (r["cv_score"], r["trial"]))
        selection = dict(parameters={k: winner[k] for k in asdict(baseline)},
                         **{k: winner[k] for k in ("cv_score", "refit_epochs", "cv_best_epochs", "trial")},
                         cv_metric="validation_nll")
    atomic_json(path, selection)
    return selection


def select_nncde(data, splits, config, output, device):
    directory = output / "NNCDE"
    path = directory / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    grid = config["nncde_grid"]
    candidates = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    records, candidates_rows = [], []
    for trial, candidate in enumerate(candidates, 1):
        trial_records = []
        for fold, (train, valid) in enumerate(splits, 1):
            cache = directory / "fold_cache" / f"trial_{trial}_fold_{fold}.json"
            if cache.exists():
                record = json.loads(cache.read_text())
            else:
                model = MemoryBoundNNCDE(**candidate, epochs=config["nncde_epochs"],
                    patience=config["nncde_patience"], device=device,
                    seed=config["fold_seed"] + (trial - 1) * config["folds"] + fold)
                model.fit(*data.bounds(train), censor_type="right")
                score = model.validation_nll(data.bounds(valid))
                if not math.isfinite(score):
                    raise RuntimeError("Nonfinite NNCDE CV score")
                record = dict(trial=trial, fold=fold, n_valid=len(valid), nll=score,
                              parameters=candidate, crossfit_details=model.fit_details_)
                atomic_json(cache, record)
                del model
                gc.collect()
            records.append(record)
            trial_records.append(record)
        # Match tuning/main.py: unweighted mean across the four CV folds.
        score = float(np.mean([r["nll"] for r in trial_records]))
        candidates_rows.append(dict(trial=trial, **candidate, cv_score=score))
        write_csv(directory / "cv_results.csv", candidates_rows)
        print(f"NNCDE: {trial}/{len(candidates)}, NLL={score:.6f}", flush=True)
    winner = min(candidates_rows, key=lambda r: (r["cv_score"], r["trial"]))
    selection = dict(parameters={k: winner[k] for k in grid}, cv_metric="validation_nll",
                     cv_score=winner["cv_score"], trial=winner["trial"],
                     epochs=config["nncde_epochs"], patience=config["nncde_patience"],
                     refit_protocol="existing_two_half_crossfit_with_internal_validation")
    atomic_json(path, selection)
    return selection
