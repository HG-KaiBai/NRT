"""GPU tuning with four-fold CV inside an 80/20 train/test split."""
from __future__ import annotations

import argparse
from copy import copy
from dataclasses import asdict
import gc
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tuning.benchmark.deepsurv.cuda_model import (
    OFFICIAL_TUNING_PARAMETERS, PAPER_GAUSSIAN_FIXED_PARAMETERS,
)
from workflow import (
    ROOT, SCHEMA_VERSION, ALL_CASES, DEFAULT_EPOCHS, NRT_LOSSES, SUPPORTED, atomic_json, node_preflight, close_model,
    cv_splits, evaluate, fingerprint, fit_model, generate, horizon, nncde_nll, select_cases,
    write_csv,
)

# Explicit grid choices. Edit here or supply --search-space with a JSON file.
NRT_SEARCH_SPACE = {
    'depth': [7, 8], 'h_dim': [8, 16, 32], 'mlp_hidden': [8, 16, 32],
    'mlp_layers': [1, 2, 3], 'shared_layers': [1, 2, 3],
}
NNCDE_GRID = {'hidden_nodes': [64, 128, 256],
              'learning_rate': [0.01, 0.001, 0.0001], 'batch_size': [64, 128, 256]}
CV_METRICS = {'NRT_NLL': 'brier', 'NRT_Brier': 'brier', 'NRT_2Loss': 'brier',
              'NNCDE': 'nll', 'DeepSurv': 'c_index'}


def validate_nrt_search_space(space):
    if not isinstance(space, dict) or set(space) != set(NRT_SEARCH_SPACE):
        raise ValueError('NRT search space must define depth, h_dim, mlp_hidden, mlp_layers, shared_layers')
    for key, choices in space.items():
        minimum = 0 if key == 'shared_layers' else 1
        if (not isinstance(choices, (list, tuple)) or not choices
                or any(type(v) is not int or v < minimum for v in choices)):
            raise ValueError(f'{key} requires a nonempty list of integer choices >= {minimum}')
        if len(set(choices)) != len(choices):
            raise ValueError(f'{key} choices must be unique')


def grid_candidates(space):
    validate_nrt_search_space(space)
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]


def configurations(args):
    if args.method == 'NRT':
        space = json.loads(Path(args.search_space).read_text()) if args.search_space else NRT_SEARCH_SPACE
        return grid_candidates(space)
    grid = json.loads(Path(args.search_space).read_text()) if args.search_space else NNCDE_GRID
    if set(grid) != set(NNCDE_GRID) or any(not values for values in grid.values()):
        raise ValueError('NNCDE grid needs nonempty hidden_nodes, learning_rate, batch_size lists')
    if any(type(v) is not int or v < 1 for key in ('hidden_nodes', 'batch_size') for v in grid[key]):
        raise ValueError('NNCDE hidden_nodes and batch_size must be positive integers')
    if any(not math.isfinite(v) or v <= 0 for v in grid['learning_rate']):
        raise ValueError('NNCDE learning rates must be finite and positive')
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def training_config(args, method):
    if method in NRT_LOSSES:
        return dict(epochs=args.epochs, training_batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, patience=args.patience,
                    brier_grid_size=args.grid_size, cuda_graphs=args.cuda_graphs, max_grad_norm=1.0)
    if method == 'NNCDE':
        return dict(epochs=args.epochs, patience=args.patience)
    return {**PAPER_GAUSSIAN_FIXED_PARAMETERS,
            'epochs': args.epochs if args.epochs is not None else DEFAULT_EPOCHS['DeepSurv']}


def tuning_jobs(args, cases, methods):
    if args.task_id is None:
        return [(method, case) for method in methods for case in cases
                if case.censor_type in SUPPORTED[method]]
    # Array indices always use the complete applicable matrix, even when --cases or
    # --nrt-methods filters the requested work. Case.index is a seed index,
    # so it must not be used for this contiguous 35-case task mapping.
    array_methods = list(NRT_LOSSES) if args.method == 'NRT' else [args.method]
    array_cases = [case for case in ALL_CASES
                   if any(case.censor_type in SUPPORTED[method] for method in array_methods)]
    task_count = len(array_methods) * len(array_cases)
    if not 0 <= args.task_id < task_count:
        raise ValueError(f'{args.method} task ID must be between 0 and {task_count - 1}')
    method_index, case_index = divmod(args.task_id, len(array_cases))
    method, case = array_methods[method_index], array_cases[case_index]
    if method not in methods or case not in cases or case.censor_type not in SUPPORTED[method]:
        return []
    return [(method, case)]


def check_tuning_result(result, method, path):
    expected = CV_METRICS[method]
    if result.get('cv_metric') != expected:
        raise ValueError(f'{path} uses CV metric {result.get("cv_metric")!r}; expected {expected!r}. '
                         'Retune using a new output directory or --force')
    if method == 'DeepSurv':
        ranges = OFFICIAL_TUNING_PARAMETERS['reference_search']['ranges']
        if result.get('request', {}).get('search_ranges') != ranges:
            raise ValueError(f'{path} uses different or unrecorded DeepSurv search ranges. '
                             'Retune using a new output directory or --force')


def write_summary(args, jobs):
    missing = [args.output_dir / method / case.id / 'result.json'
               for method, case in jobs
               if not (args.output_dir / method / case.id / 'result.json').exists()]
    if missing:
        raise RuntimeError(f'{len(missing)} tuning results missing; complete tuning before summarizing: {missing[0]}')
    rows = []
    hyper_rows = {}
    for method, case in jobs:
        path = args.output_dir / method / case.id / 'result.json'
        result = json.loads(path.read_text())
        check_tuning_result(result, method, path)
        rows.append({key: result[key] for key in ('method', 'case_id', 'tau', 'right_censoring_rate',
                                                 'cv_metric', 'cv_score', 'average_grid_sse')})
        if args.task_id is None:
            hyper_rows.setdefault(method, []).append(dict(
                model_name=method, data_types=case.id,
                parameters=json.dumps(result['parameters'], separators=(',', ':'), allow_nan=False),
                test_average_grid_sse=result['average_grid_sse'], tau=result['tau']))
    if args.task_id is not None:
        # Workers write only in their own method/case directory.
        method, case = jobs[0]
        path = args.output_dir / method / case.id / 'summary.csv'
    else:
        methods = list(dict.fromkeys(method for method, _ in jobs))
        path = args.output_dir / f'{args.method}_{"_".join(methods)}_summary.csv'
    write_csv(path, rows)
    print(f'Saved {path}', flush=True)
    for method, method_rows in hyper_rows.items():
        path = args.output_dir / 'hyper_para' / f'{method}.csv'
        write_csv(path, method_rows)
        print(f'Saved {path}', flush=True)


def auto_summary(args, cases, methods, worker_jobs):
    """Publish complete method tables, then the family table, after each worker.

    Every worker checks the entire requested case selection, independent of its
    array index. Incomplete selections leave existing tables untouched. Writers
    use atomic replacement, so simultaneous finishers publish complete tables.
    """
    summary_args = copy(args)
    summary_args.task_id = None
    completed_methods = list(dict.fromkeys(method for method, _ in worker_jobs))
    for method in completed_methods:
        jobs = tuning_jobs(summary_args, cases, [method])
        missing = sum(not (args.output_dir / method / case.id / 'result.json').is_file()
                      for method, case in jobs)
        if missing:
            print(f'{method} summary pending: {missing}/{len(jobs)} selected results missing', flush=True)
        else:
            write_summary(summary_args, jobs)
    if len(methods) > 1:
        jobs = tuning_jobs(summary_args, cases, methods)
        if all((args.output_dir / method / case.id / 'result.json').is_file()
               for method, case in jobs):
            write_summary(summary_args, jobs)


def tune_cv(args, method, data, candidates, output):
    from tuning.train_mlp import evaluate_brier
    training = training_config(args, method)
    metric = CV_METRICS[method]
    folds = cv_splits(data.train, args.tune_seed)
    records = []
    best_score, best_parameters = math.inf, None
    for trial, candidate in enumerate(candidates, 1):
        parameters = {**candidate}
        if method in NRT_LOSSES:
            parameters.update(x_dim=6, dropout=0.0, split_min=args.split_min)
        scores = []
        for fold, (train, valid) in enumerate(folds, 1):
            model = None
            error = ''
            try:
                model = fit_model(method, data, train, parameters, training, args.device,
                                  args.tune_seed + (trial - 1) * 4 + fold, 'tuning')
                if method in NRT_LOSSES:
                    score = evaluate_brier(model, *data.bounds(valid), args.grid_size, args.device)
                else:
                    score = nncde_nll(model, data, valid)
                if not math.isfinite(score):
                    raise ValueError('Non-finite validation score')
            except (RuntimeError, ValueError) as exc:
                score, error = None, str(exc)
            finally:
                if model is not None:
                    close_model(model)
                    del model
                gc.collect()
                if str(args.device).startswith('cuda'):
                    torch.cuda.empty_cache()
            scores.append(score)
            records.append(dict(trial=trial, fold=fold, **candidate, validation_score=score, error=error))
        mean = statistics.fmean(scores) if all(s is not None for s in scores) else None
        write_csv(output / 'cv_results.csv', records)
        print(f'{method} {data.case.id} trial={trial}/{len(candidates)} cv={mean}', flush=True)
        if mean is not None and mean < best_score:
            best_score, best_parameters = mean, parameters
    if best_parameters is None:
        raise RuntimeError(f'All candidates failed for {method} {data.case.id}; inspect {output / "cv_results.csv"}')
    return best_parameters, training, best_score, metric


def tune_case(args, method, case, candidates):
    output = args.output_dir / method / case.id
    data_seed = args.seed + case.index * 10_000
    tau = horizon(case, mu=args.mu, r=args.r)
    request = dict(schema_version=SCHEMA_VERSION, case=asdict(case), method=method,
                   seed=data_seed, tune_seed=args.tune_seed, n_samples=args.n_samples,
                   mu=args.mu, r=args.r, tau=tau, cv_folds=4, grid_size=args.grid_size,
                   candidates=candidates,
                   tuning_evals=len(candidates) if method in NRT_LOSSES else args.tuning_evals,
                   training_config=training_config(args, method), split_min=args.split_min)
    if method == 'DeepSurv':
        request['search_ranges'] = OFFICIAL_TUNING_PARAMETERS['reference_search']['ranges']
    request_hash = fingerprint(request)
    result_path = output / 'result.json'
    if result_path.exists() and not args.force:
        old = json.loads(result_path.read_text())
        check_tuning_result(old, method, result_path)
        if old.get('request_hash') != request_hash:
            raise ValueError(f'{result_path} belongs to different tuning settings; choose a new output directory or --force')
        print(f'Reusing {result_path}', flush=True)
        return
    data = generate(case, args.n_samples, data_seed, tau, mu=args.mu, r=args.r)
    manifest = dict(schema_version=SCHEMA_VERSION, case=asdict(case), case_id=case.id,
                    seed=data_seed, mu=args.mu, r=args.r, **data.metadata())
    manifest_path = args.output_dir / 'cases' / f'{case.id}.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f'{manifest_path} contains different data settings; use a new output directory')
    atomic_json(manifest_path, manifest)
    print(f'{method} {case.id}: tau={tau}, right_censoring_rate={manifest["right_censoring_rate"]:.6f}', flush=True)
    start = time.monotonic()
    model = None
    try:
        if method == 'DeepSurv':
            training = training_config(args, method)
            model = fit_model(method, data, data.train, {},
                              dict(**training, tune=True, cv_folds=4,
                                   tuning_evals=args.tuning_evals, tuning_epochs=args.epochs),
                              args.device, args.tune_seed, 'tuning')
            parameters = model.best_hyperparameters_
            score, metric = model.best_validation_c_index_, 'c_index'
            if parameters is None or score is None or not math.isfinite(score):
                raise ValueError('DeepSurv tuning did not produce a finite optimum')
            atomic_json(output / 'cv_results.json', model.tuning_parameters)
        else:
            parameters, training, score, metric = tune_cv(args, method, data, candidates, output)
            model = fit_model(method, data, data.train, parameters, training, args.device, args.seed, 'tuning')
        metrics = evaluate(model, data, args.grid_size, method=method, device=args.device, package='tuning')
        atomic_json(result_path, dict(**manifest, method=method, request_hash=request_hash,
                    request=request, parameters=parameters, training_config=training,
                    cv_folds=4, cv_metric=metric, cv_score=score, **metrics,
                    elapsed_seconds=time.monotonic() - start))
    finally:
        if model is not None:
            close_model(model)
    print(f'Saved {result_path}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('NRT', 'NNCDE', 'DeepSurv'), required=True)
    p.add_argument('--nrt-methods', default=','.join(NRT_LOSSES))
    p.add_argument('--cases', default='all', help='all or comma-separated case IDs; see --list-cases')
    p.add_argument('--list-cases', action='store_true', help='List supported case IDs in array order')
    p.add_argument('--task-id', type=int,
                   help='Zero-based array task: NRT uses three blocks of 35; NNCDE/DeepSurv use 15 cases')
    p.add_argument('--summarize', action='store_true', help='Combine existing tuning results without fitting; run after the array completes')
    output_dir = ROOT / 'tuning' / 'output'
    p.add_argument('--output-dir', type=Path, default=output_dir, help=f'Default: {output_dir}')
    p.add_argument('--n-samples', type=int, default=5000)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--tune-seed', type=int, default=321)
    p.add_argument('--mu', type=float, default=0.0)
    p.add_argument('--r', type=float)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--tuning-evals', type=int, default=50,
                   help='DeepSurv candidate count (default 50); grid searches use their full grids')
    p.add_argument('--search-space', help='JSON file containing explicit grid choice lists')
    p.add_argument('--epochs', type=int, help='Epoch limit: defaults to 200 for NRT/NNCDE, 1000 for DeepSurv')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--grid-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--split-min', type=float, default=0.1)
    p.add_argument('--cuda-graphs', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.epochs is None:
        args.epochs = DEFAULT_EPOCHS[args.method]
    cases = select_cases(args.cases)
    methods = args.nrt_methods.split(',') if args.method == 'NRT' else [args.method]
    if args.method == 'NRT' and (not methods or set(methods) - NRT_LOSSES.keys()):
        raise ValueError('--nrt-methods must contain NRT_NLL, NRT_Brier or NRT_2Loss')
    if args.list_cases:
        print('\n'.join(case.id for case in cases
                        if any(case.censor_type in SUPPORTED[method] for method in methods)))
        return
    if min(args.epochs, args.patience, args.threads, args.tuning_evals, args.batch_size) < 1 or args.grid_size < 2:
        raise ValueError('Epochs, patience, threads, evaluations and batch size must be positive; grid size >= 2')
    if args.summarize and args.task_id is not None:
        raise ValueError('--summarize must run once after the array, without --task-id')
    jobs = tuning_jobs(args, cases, methods)
    if not jobs:
        if args.task_id is not None:
            print(f'Skipping {args.method} task {args.task_id}: case is excluded by the requested selection', flush=True)
            return
        raise ValueError('No applicable cases for the requested methods')
    if args.summarize:
        write_summary(args, jobs)
        return
    candidates = None if args.method == 'DeepSurv' else configurations(args)
    if args.dry_run:
        for method, case in jobs:
            search = (f'grid configurations={len(candidates)}' if candidates is not None
                      else f'Sobol evaluations={args.tuning_evals}')
            print(f'{method} {case.id} 80/20 split; 4-fold CV; {search}; '
                  f'cv_metric={CV_METRICS[method]}; device={args.device}')
        return
    node_preflight(args.device, {method for method, case in jobs}, 'tuning')
    torch.set_num_threads(args.threads)
    for method, case in jobs:
        tune_case(args, method, case, candidates)
    write_summary(args, jobs)
    if args.task_id is not None:
        auto_summary(args, cases, methods, jobs)


if __name__ == '__main__':
    main()
