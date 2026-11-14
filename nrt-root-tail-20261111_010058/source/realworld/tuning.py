"""Simulation search budgets and samplers, with real-data fold preprocessing."""
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
from testing.benchmark.deepsurv.cuda_model import PAPER_GAUSSIAN_FIXED_PARAMETERS, OFFICIAL_TUNING_PARAMETERS
from tuning.main import NNCDE_GRID
from workflow import atomic_json, write_csv, cv_splits
from .models import MemoryBoundNNCDE, RealDeepSurv

NRT_METHODS = ('fixed_nll_cv', 'adaptive_nll_cv', 'fixed_at_adaptive_architecture')


def default_config(smoke=False):
    config = dict(profile='smoke' if smoke else 'full', seed=123, fold_seed=321, folds=4,
        nrt_search=asdict(SearchConfig()), nrt_baseline=asdict(Architecture()),
        nrt_training=asdict(TrainingConfig(patience=50)), nrt_optimization=asdict(NLLCVOptimization()),
        nncde_grid=NNCDE_GRID, nncde_epochs=200, nncde_patience=20,
        deepsurv_epochs=1000, deepsurv_evals=50,
        deepsurv_fixed={**PAPER_GAUSSIAN_FIXED_PARAMETERS, 'standardize': True},
        deepsurv_ranges=OFFICIAL_TUNING_PARAMETERS['reference_search']['ranges'],
        grid_size=256, forest_trees=100, ddr_epochs=100, ddr_patience=20,
        aft_distribution='weibull', aft_max_iter=250)
    if smoke:
        config.update(folds=2, nncde_epochs=3, nncde_patience=2,
                      deepsurv_epochs=3, deepsurv_evals=2, forest_trees=3,
                      ddr_epochs=3, ddr_patience=2,
                      nncde_grid=dict(hidden_nodes=[8], learning_rate=[.001], batch_size=[32]))
        config['nrt_search'].update(trials=2, initial_trials=2, folds=2, depth_min=3, depth_max=3,
                                    shared_layers_max=0, private_layers_max=3)
        config['nrt_training'].update(epochs=3, patience=2, validation_every=1, cuda_graphs=False)
        config['nrt_optimization']['warmup_epochs'] = 1
    return config


def nrt_settings(config, dimension):
    baseline = {**config['nrt_baseline'], 'x_dim': dimension}
    return (SearchConfig(**config['nrt_search']), Architecture(**baseline),
            TrainingConfig(**config['nrt_training']), NLLCVOptimization(**config['nrt_optimization']))


def score_nrt(data, architecture, config, directory, adaptive, device, penalty):
    search, _, training, optimization = nrt_settings(config, data.frame.shape[1])
    optimization = replace(optimization, learn_splits=adaptive)
    upper = observed_upper_bound(data.bounds(data.train))
    records = []
    for fold, (train, valid) in enumerate(cv_splits(data.train, search.fold_seed, search.folds), 1):
        path = directory / f'fold_{fold}.json'
        if path.exists():
            record = json.loads(path.read_text())
        else:
            scaler = data.scaler(train)
            model, history = fit_raw_time_kl(data.bounds(train, scaler), data.bounds(valid, scaler),
                architecture=architecture, training=training, optimization=optimization,
                seed=search.search_fit_seed, device=device, upper_bound=upper, penalty=penalty)
            record = dict(fold=fold, n_valid=len(valid), **model.fit_info)
            atomic_json(path, record)
            atomic_json(directory / f'history_{fold}.json', history)
            del model
        records.append(record)
    epochs = [r['best_epoch'] for r in records]
    return dict(cv_score=float(np.average([r['validation_nll'] for r in records],
                                         weights=[r['n_valid'] for r in records])),
                cv_best_epochs=epochs, refit_epochs=math.ceil(float(np.median(epochs))))


def select_nrt(method, data, config, output, device):
    if method not in NRT_METHODS:
        raise ValueError(f'Unsupported NRT method: {method}')
    directory = output / method
    path = directory / 'selection.json'
    if path.exists():
        return json.loads(path.read_text())
    search, baseline, _, _ = nrt_settings(config, data.frame.shape[1])
    penalty = 'partition_kl'
    adaptive = method.endswith('adaptive_nll_cv')
    if method.endswith('fixed_at_adaptive_architecture'):
        source = 'adaptive_nll_cv'
        source_path = output / source / 'selection.json'
        if not source_path.exists():
            raise RuntimeError(f'Run {source} before its matched control')
        architecture = replace(Architecture(**json.loads(source_path.read_text())['parameters']),
                               regularization_strength=0.)
        scores = score_nrt(data, architecture, config, directory / 'cv', False, device, penalty)
        selection = dict(parameters=asdict(architecture), **scores, architecture_source=source)
    else:
        optuna = optuna_module()
        directory.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(study_name=method, direction='minimize',
            storage=f"sqlite:///{(directory / 'study.sqlite3').resolve()}", load_if_exists=True)
        if not study.trials:
            # A narrow discrete search may contain fewer unique startup points.
            initial = search.initial_trials
            if not adaptive or search.strength_max == 0:
                capacity = ((search.depth_max-search.depth_min+1) * len(search.widths)
                            * search.private_layers_max * (1+search.shared_layers_max*len(search.widths)))
                initial = min(initial, capacity)
            startup_search = replace(search, initial_trials=initial)
            for row in startup(startup_search, baseline, adaptive, search.search_seed):
                study.enqueue_trial(row)
        while True:
            completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            if len(completed) >= search.trials:
                break
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
                study.sampler = optuna.samplers.TPESampler(seed=search.search_seed+len(completed),
                    n_startup_trials=search.initial_trials, multivariate=True, group=True, n_ei_candidates=48)
            running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
            if len(running) > 1:
                raise RuntimeError('Multiple running trials; use one worker per method')
            trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
            architecture = suggest(trial, search, baseline, adaptive)
            started = time.monotonic()
            scores = score_nrt(data, architecture, config, directory / 'fold_cache' / f'trial_{trial.number}',
                               adaptive, device, penalty)
            trial.set_user_attr('parameters', asdict(architecture))
            trial.set_user_attr('scores', scores)
            study.tell(trial, scores['cv_score'])
            print(f"{method}: {len(completed)+1}/{search.trials} NLL={scores['cv_score']:.6g} "
                  f"({time.monotonic()-started:.1f}s)", flush=True)
        rows = [dict(trial=t.number, **t.user_attrs['parameters'], **t.user_attrs['scores'])
                for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))]
        write_csv(directory / 'cv_results.csv', rows)
        winner = min(rows, key=lambda r: (r['cv_score'], r['trial']))
        selection = dict(parameters={k: winner[k] for k in asdict(baseline)},
            **{k: winner[k] for k in ('cv_score', 'refit_epochs', 'cv_best_epochs', 'trial')})
    selection.update(cv_metric='validation_nll', penalty=penalty,
                     search_config=asdict(search), baseline=asdict(baseline))
    atomic_json(path, selection)
    return selection


def select_nncde(data, config, directory, device):
    path = directory / 'selection.json'
    if path.exists():
        return json.loads(path.read_text())
    grid = config['nncde_grid']
    candidates = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    rows = []
    for trial, candidate in enumerate(candidates, 1):
        scores = []
        for fold, (train, valid) in enumerate(cv_splits(data.train, config['fold_seed'], config['folds']), 1):
            cache = directory / 'fold_cache' / f'trial_{trial}_fold_{fold}.json'
            if cache.exists():
                record = json.loads(cache.read_text())
            else:
                model = MemoryBoundNNCDE(**candidate, epochs=config['nncde_epochs'],
                    patience=config['nncde_patience'], device=device,
                    seed=config['fold_seed']+(trial-1)*config['folds']+fold)
                # NNCDE estimates its expanded-input moments inside each half-fit.
                try:
                    model.fit(*data.bounds(train), censor_type=data.censor_type)
                    score = model.validation_nll(data.bounds(valid))
                    if not math.isfinite(score):
                        raise ValueError('Nonfinite NNCDE CV score')
                    record = dict(trial=trial, fold=fold, n_valid=len(valid), nll=score,
                                  parameters=candidate, crossfit_details=model.fit_details_)
                except (ValueError, RuntimeError) as error:
                    # Match simulation: numerical candidates are recorded and excluded.
                    # Resource/device failures must remain eligible for job-level retry.
                    message = str(error).lower()
                    if any(s in message for s in ('cuda', 'out of memory', 'device')):
                        raise
                    record = dict(trial=trial, fold=fold, n_valid=len(valid), nll=None,
                                  parameters=candidate, error=str(error))
                atomic_json(cache, record)
                del model
                gc.collect()
            scores.append(record['nll'])
        mean = float(np.mean(scores)) if all(s is not None for s in scores) else None
        rows.append(dict(trial=trial, **candidate, cv_score=mean))
        write_csv(directory / 'cv_results.csv', rows)
        print(f"NNCDE: {trial}/{len(candidates)} NLL={mean}", flush=True)
    valid = [r for r in rows if r['cv_score'] is not None]
    if not valid:
        raise RuntimeError('All NNCDE candidates failed; inspect fold_cache and cv_results.csv')
    winner = min(valid, key=lambda r: (r['cv_score'], r['trial']))
    selection = dict(parameters={k: winner[k] for k in grid}, cv_metric='validation_nll',
                     cv_score=winner['cv_score'], epochs=config['nncde_epochs'], patience=config['nncde_patience'])
    atomic_json(path, selection)
    return selection


def select_deepsurv(data, config, directory, device):
    path = directory / 'selection.json'
    if path.exists():
        return json.loads(path.read_text())
    parameters = {**config['deepsurv_fixed'], 'epochs': config['deepsurv_epochs']}
    model = RealDeepSurv(**parameters, tune=True, cv_folds=config['folds'],
        tuning_evals=config['deepsurv_evals'], tuning_epochs=config['deepsurv_epochs'],
        device=device, seed=config['fold_seed'], verbose=True)
    model.fit(*data.bounds(data.train), censor_type=data.censor_type)
    score = model.best_validation_c_index_
    if score is None or not math.isfinite(score):
        raise RuntimeError('DeepSurv search failed')
    atomic_json(directory / 'cv_results.json', model.tuning_parameters)
    selection = dict(parameters={**parameters, **model.best_hyperparameters_},
                     cv_metric='validation_Harrell_c_index', cv_score=score)
    atomic_json(path, selection)
    return selection
