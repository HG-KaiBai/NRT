"""Reproducible real-data tuning, fitting, predictions and evaluation."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from . import ROOT, PROJECT_ROOT
from .data import load_dataset
from .bootstrap import (bootstrap_method, summary_columns, validate_options,
                        DEFAULT_RESAMPLES, DEFAULT_CONFIDENCE, DEFAULT_SEED)
from .metrics import evaluate, evaluation_grid, uno_concordance_curve
from .models import MemoryBoundNNCDE, RealDeepSurv
from .nrt_config import DATASETS, resolve_search_spaces, dataset_configuration
from .tuning import (NRT_METHODS, default_config, nrt_settings,
                     select_nrt, select_nncde, select_deepsurv)
from regularization.model import observed_upper_bound
from regularization.train import fit_raw_time_kl
from regularization.unified_nll.tuning_model import Architecture
from regularization.workflow import writer_lock
from workflow import atomic_json, write_csv, fingerprint, node_preflight, close_model

BENCHMARKS = ('NNCDE', 'DeepSurv', 'ctree', 'cox', 'aft')


def methods_for(dataset):
    return NRT_METHODS + BENCHMARKS + (('deep_distribution_regression',)
        if dataset == 'gt_2011' else ('iccforest',))


def tasks():
    result = []
    for dataset in DATASETS:
        groups = [(NRT_METHODS[0],), NRT_METHODS[1:3],
                  ('NNCDE',), ('DeepSurv',), ('ctree',), ('cox',), ('aft',),
                  (('deep_distribution_regression',) if dataset == 'gt_2011' else ('iccforest',))]
        for group in groups:
            result.append(dict(task_id=len(result), dataset=dataset, methods=list(group),
                               device='cpu' if group[0] in ('ctree','cox','aft','iccforest') else 'cuda:0'))
    return result


def source_manifest():
    paths = []
    for folder in ('realworld', 'regularization', 'tuning', 'testing'):
        for parent, directories, files in os.walk(PROJECT_ROOT/folder):
            directories[:] = [name for name in directories if not name.startswith(('.', 'output'))
                              and name not in ('runs', 'log', 'logs', '__pycache__')]
            paths.extend(Path(parent)/name for name in files
                         if Path(name).suffix in ('.py', '.R', '.sh') and not name.startswith('test_'))
    paths += [PROJECT_ROOT/name for name in ('workflow.py', 'cuda_retry.py', 'node_runner.py',
                                             'runtime.sh', 'environment/source-compatibility.json')]
    return {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def environment():
    return dict(python=sys.version, executable=sys.executable,
                packages={n: importlib.metadata.version(n) for n in
                          ('torch','numpy','pandas','scipy','optuna','optunity','lifelines','scikit-survival')})


def load(args, dataset):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    return load_dataset(dataset,
                        gt_target='CO', seed=args.seed,
                        limit=80 if args.profile == 'smoke' else None)


def configuration(args):
    if hasattr(args, '_configuration'):
        return args._configuration
    config = default_config(args.profile == 'smoke')
    config['seed'] = args.seed
    path = getattr(args, 'nrt_search_space', None)
    if path is None and (args.output/'nrt_search_spaces.json').exists():
        path = args.output/'nrt_search_spaces.json'
    overrides = json.loads(Path(path).read_text()) if path is not None else {}
    if not isinstance(overrides, dict):
        raise ValueError('NRT search spaces must be a JSON object')
    config['nrt_by_dataset'] = resolve_search_spaces(config, overrides)
    args._configuration = config
    return config


def prepare(args):
    if (args.output/'reevaluation.json').exists():
        raise ValueError('This is a reevaluation directory; use a new directory for training')
    config = configuration(args)
    request = dict(schema_version=2, config=config, datasets=list(DATASETS),
                   endpoints=dict(gt_2011='CO', rotterdam_recur='rtime/recur', rotterdam_death='dtime/death'),
                   source=source_manifest(), environment=environment(),
                   data={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.csv')})
    digest = fingerprint(request)
    existing = args.output/'request.json'
    if existing.exists():
        if json.loads(existing.read_text())['request_hash'] != digest:
            raise ValueError('Source, data or configuration changed; choose a new output directory')
        if all((args.output/d/'data.json').exists() for d in DATASETS):
            return digest
    with writer_lock(args.output):
        path = args.output/'request.json'
        if path.exists() and json.loads(path.read_text())['request_hash'] != digest:
            raise ValueError('Source, data or configuration changed; choose a new output directory')
        atomic_json(args.output/'nrt_search_spaces.json', config['nrt_by_dataset'])
        atomic_json(path, dict(request_hash=digest, request=request))
        atomic_json(args.output/'tasks.json', tasks())
        for dataset in DATASETS:
            data = load(args, dataset)
            folder = args.output/dataset
            assignment = np.full(len(data.time), 'test', dtype=object)
            assignment[data.train.numpy()] = 'train'
            write_csv(folder/'split_assignments.csv', [dict(row_id=int(i), partition=str(s))
                for i,s in zip(data.row_ids, assignment)])
            grid = evaluation_grid(data.time[data.train], data.event[data.train],
                censored=data.censor_type == 'right', size=config['grid_size'])
            atomic_json(folder/'data.json', dict(dataset=dataset, target=data.target, n=len(data.time),
                n_train=len(data.train), n_test=len(data.test), covariates=list(data.frame),
                n_events=int(data.event.sum()), censoring_rate=float(1-data.event.mean()),
                train_censoring_rate=float(1-data.event[data.train].mean()),
                test_censoring_rate=float(1-data.event[data.test].mean()),
                source_sha256=data.source_sha256, grid=grid.tolist(),
                horizon_policy='simulation: 256 points from zero to largest finite training bound',
                methods=list(methods_for(dataset)), nrt=config['nrt_by_dataset'][dataset]))
    return digest


def fit(method, data, config, selection, device):
    scaler = data.scaler(data.train)
    x,l,r = data.bounds(data.train, scaler)
    seed = config['seed']
    if method in NRT_METHODS:
        _, _, training, optimization = nrt_settings(config, x.shape[1])
        optimization = replace(optimization, learn_splits=method.endswith('adaptive_nll_cv'))
        model, history = fit_raw_time_kl((x,l,r), architecture=Architecture(**selection['parameters']),
            training=training, optimization=optimization, seed=seed, device=device,
            fixed_epochs=selection['refit_epochs'], upper_bound=observed_upper_bound((x,l,r)),
            penalty=selection['penalty'])
        return model, scaler, history
    if method == 'NNCDE':
        model = MemoryBoundNNCDE(**selection['parameters'], epochs=config['nncde_epochs'],
            patience=config['nncde_patience'], seed=seed, device=device)
        scaler = None
    elif method == 'DeepSurv':
        model = RealDeepSurv(**selection['parameters'], device=device, seed=seed)
        scaler = None
    elif method == 'ctree':
        from testing.benchmark.ctree import CTree
        model = CTree(seed=seed)
    elif method == 'cox':
        from testing.benchmark.cox import CoxPH
        model = CoxPH()
    elif method == 'aft':
        from testing.benchmark.aft import ParametricAFT
        model = ParametricAFT(distribution=config['aft_distribution'], max_iter=config['aft_max_iter'])
    elif method == 'iccforest':
        from testing.benchmark.iccforest import ICCForest
        model = ICCForest(ntree=config['forest_trees'], tune_ntree=config['forest_trees'],
                          tune_mtry=config['profile'] != 'smoke', seed=seed)
    elif method == 'deep_distribution_regression':
        from testing.benchmark.deep_distribution_regression import DeepDistributionRegression
        model = DeepDistributionRegression(epochs=config['ddr_epochs'], patience=config['ddr_patience'],
                                           device=device, seed=seed)
    else:
        raise ValueError(method)
    try:
        model.fit(*data.bounds(data.train, scaler), censor_type=data.censor_type)
    except BaseException:
        close_model(model)
        raise
    return model, scaler, getattr(model, 'training_history_', [])


def run_method(args, dataset, method, digest):
    config = dataset_configuration(configuration(args), dataset)
    data = load(args, dataset)
    output = args.output/dataset
    folder = output/method
    with writer_lock(folder):
        result_path = folder/'result.json'
        if result_path.exists():
            old = json.loads(result_path.read_text())
            if old.get('request_hash') == digest and old.get('status') == 'complete':
                print(f'Reusing {dataset}/{method}', flush=True)
                return
        started, model = time.monotonic(), None
        try:
            if method in NRT_METHODS:
                selection = select_nrt(method, data, config, output, args.device)
            elif method == 'NNCDE':
                selection = select_nncde(data, config, folder, args.device)
            elif method == 'DeepSurv':
                selection = select_deepsurv(data, config, folder, args.device)
            else:
                selection = dict(parameters=dict(distribution=config['aft_distribution']) if method=='aft' else {},
                                 cv_metric=None, cv_score=None)
                atomic_json(folder/'selection.json', selection)
            model, scaler, history = fit(method, data, config, selection, args.device)
            grid = np.array(json.loads((output/'data.json').read_text())['grid'])
            x,_,_ = data.bounds(data.test, scaler)
            with torch.no_grad():
                survival = torch.cat([model.predict_survival(batch, torch.tensor(grid, dtype=torch.float32)).cpu()
                                      for batch in x.split(64)]).numpy()
            scores, curve, risk = evaluate(data.time[data.train], data.event[data.train],
                data.time[data.test], data.event[data.test], survival, grid, censored=data.censor_type=='right')
            np.savez_compressed(folder/'predictions.npz', row_ids=data.row_ids[data.test],
                time=data.time[data.test], event=data.event[data.test], grid=grid, survival=survival, risk=risk)
            write_csv(folder/'brier_curve.csv', [dict(time=float(t), brier_score=float(s)) for t,s in zip(grid,curve)])
            if data.censor_type == 'right':
                c_curve = uno_concordance_curve(data.time[data.test], data.event[data.test], survival, grid)
                write_csv(folder/'c_index_curve.csv', [dict(time=float(t), c_index=float(c) if np.isfinite(c) else None)
                                                      for t,c in zip(grid,c_curve)])
            atomic_json(folder/'preprocessing.json', dict(scaler=scaler,
                internal_scaling=method in ('NNCDE','DeepSurv'), covariates=list(data.frame)))
            atomic_json(folder/'training_history.json', history)
            if getattr(model, '_model_path', None) is not None:
                shutil.copy2(model._model_path, folder/'model.rds')
            else:
                torch.save(model, folder/'model.pt')
            result = dict(dataset=dataset, method=method, status='complete', request_hash=digest,
                          profile=args.profile, device=args.device, **scores, selection=selection,
                          elapsed_seconds=time.monotonic()-started)
            atomic_json(result_path, result)
            print(json.dumps({k:v for k,v in result.items() if k!='selection'}), flush=True)
        except BaseException as error:
            atomic_json(folder/'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
            raise
        finally:
            if model is not None:
                close_model(model)


def summarize(args):
    rows, statuses = [], []
    for dataset in DATASETS:
        for method in methods_for(dataset):
            path = args.output/dataset/method/'result.json'
            status, bootstrap_status = 'pending', 'pending'
            if path.exists():
                result = json.loads(path.read_text())
                status = result['status']
                row = {k:result[k] for k in ('dataset','method','profile','metric_variant',
                    'c_index','integrated_brier_score','brier_at_horizon','horizon','elapsed_seconds')}
                intervals = summary_columns(args.output, dataset, method)
                row.update(intervals)
                if intervals:
                    bootstrap_status = ('complete' if intervals['c_index_ci_lower'] is not None
                                        else 'insufficient_comparable_pairs')
                rows.append(row)
            elif (path.parent/'failure.json').exists():
                status = 'failed'
            statuses.append(dict(dataset=dataset, method=method, status=status,
                                 bootstrap_status=bootstrap_status))
    # Multiple array workers may finish together; this short publication lock waits.
    import fcntl
    (args.output/'summary').mkdir(parents=True, exist_ok=True)
    with (args.output/'summary'/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_csv(args.output/'status.csv', statuses)
        if rows:
            write_csv(args.output/'comparison.csv', rows)
            for dataset in DATASETS:
                subset = [r for r in rows if r['dataset']==dataset]
                if subset:
                    write_csv(args.output/dataset/'comparison.csv', subset)
    n_bootstrap = sum(row['bootstrap_status']=='complete' for row in statuses)
    print(f'{len(rows)}/{len(statuses)} methods complete; '
          f'{n_bootstrap}/{len(statuses)} bootstrap intervals complete; {args.output}', flush=True)
    if args.require_complete and (any(row['status']!='complete' for row in statuses)
                                  or n_bootstrap != len(statuses)):
        raise RuntimeError('Not all methods and bootstrap intervals completed; see status.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run','reevaluate','bootstrap','summarize','plan','preflight'])
    parser.add_argument('--output', type=Path, default=ROOT/'runs'/'full_co_recur_death')
    parser.add_argument('--profile', choices=['full','smoke'], default='full')
    parser.add_argument('--source-output', type=Path, help='Completed training run for reevaluate')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nrt-search-space', type=Path,
                        help='JSON with defaults and per-dataset NRT search/baseline overrides')
    parser.add_argument('--dataset', choices=['all',*DATASETS], default='all')
    parser.add_argument('--methods', default='all')
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--bootstrap-resamples', type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument('--bootstrap-confidence', type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument('--bootstrap-seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if ROOT not in args.output.parents:
        parser.error('Keep all experiment outputs inside realworld/')
    try:
        validate_options(args.bootstrap_resamples, args.bootstrap_confidence, args.bootstrap_seed)
    except ValueError as error:
        parser.error(str(error))
    if args.command == 'plan':
        try:
            config = configuration(args)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(dict(tasks=tasks(), nrt_search_spaces=config['nrt_by_dataset']), indent=2)); return
    if args.command == 'summarize':
        summarize(args); return
    torch.set_num_threads(args.threads)
    if args.command == 'preflight':
        print(json.dumps(environment(), indent=2))
        node_preflight(args.device, ['ctree','iccforest','cox','aft','deep_distribution_regression','NNCDE','DeepSurv'], 'testing')
        return
    # Saved-prediction evaluation uses its own provenance; it never calls prepare,
    # which deliberately rejects training after any source/configuration change.
    if args.command == 'reevaluate' and args.source_output is None:
        parser.error('reevaluate requires --source-output')
    if args.command not in ('bootstrap', 'reevaluate'):
        try:
            digest = prepare(args)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.command == 'prepare':
        print(f'Prepared {args.output}'); return
    if args.task_id is not None:
        if not 0 <= args.task_id < len(tasks()):
            parser.error('Invalid task ID')
        task = tasks()[args.task_id]
        jobs = [(task['dataset'], method) for method in task['methods']]
    else:
        datasets = DATASETS if args.dataset == 'all' else (args.dataset,)
        jobs = []
        for dataset in datasets:
            methods = methods_for(dataset) if args.methods == 'all' else args.methods.split(',')
            if set(methods)-set(methods_for(dataset)):
                parser.error(f'Unsupported methods for {dataset}')
            jobs.extend((dataset,m) for m in methods)
    for dataset, method in jobs:
        if args.command == 'run':
            run_method(args, dataset, method, digest)
        elif args.command == 'reevaluate':
            from .reevaluate import reevaluate_method
            reevaluate_method(args.source_output, args.output, dataset, method)
        bootstrap_method(args.output, dataset, method,
                         n_resamples=args.bootstrap_resamples,
                         confidence_level=args.bootstrap_confidence, seed=args.bootstrap_seed)
    summarize(args)


if __name__ == '__main__':
    main()
