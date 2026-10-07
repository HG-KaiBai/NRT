"""Run reproducible replications using saved tuning parameters and tau."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gc
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from testing.benchmark.deepsurv.cuda_model import PAPER_GAUSSIAN_FIXED_PARAMETERS
from workflow import (
    ROOT, SCHEMA_VERSION, DEFAULT_EPOCHS, ALL_CASES, CPU_METHODS, METHODS, NRT_LOSSES, SUPPORTED, atomic_json,
    node_preflight, close_model, evaluate, fingerprint, fit_model, generate,
    select_cases, select_methods, write_csv,
)


def result_path(output, method, case, replication):
    return output / method / case.id / f'replication_{replication:04d}.json'


def hyperparameter_paths(args):
    """Allow a single neural run to use an explicitly named CSV in isolation."""
    neural_methods = (*NRT_LOSSES, 'NNCDE', 'DeepSurv')
    if args.hyper_file is not None:
        methods = select_methods(args.methods)
        if len(methods) != 1 or methods[0] not in neural_methods:
            raise ValueError('--hyper-file requires exactly one neural method in --methods')
        return {methods[0]: args.hyper_file}
    return {method: args.hyper_dir / f'{method}.csv' for method in neural_methods}


def hyperparameter_tables(args):
    """Read the neural CSV exports once and check their shared tau."""
    if hasattr(args, '_hyper_tables'):
        return args._hyper_tables
    tables, taus = {}, {}
    cases = {case.id: case for case in ALL_CASES}
    for method, path in hyperparameter_paths(args).items():
        if not path.exists():
            if args.hyper_file is not None:
                raise FileNotFoundError(f'Missing hyperparameter CSV: {path}')
            continue
        table = {}
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not {'model_name', 'data_types', 'parameters', 'tau'} <= set(reader.fieldnames or []):
                raise ValueError(f'Missing hyperparameter CSV columns: {path}')
            for row in reader:
                case_id = row['data_types']
                if row['model_name'] != method or case_id not in cases:
                    raise ValueError(f'Invalid model/case in {path}: {case_id}')
                case = cases[case_id]
                if case.censor_type not in SUPPORTED[method] or case_id in table:
                    raise ValueError(f'Unsupported or duplicate case in {path}: {case_id}')
                parameters = json.loads(row['parameters'])
                if not isinstance(parameters, dict) or not parameters:
                    raise ValueError(f'Invalid parameters in {path}: {case_id}')
                # Also reject NaN/Infinity in JSON parameter values.
                fingerprint(parameters)
                tau = float(row['tau']) if row['tau'].strip() else None
                if case.censor_type == 'exact':
                    if tau is not None:
                        raise ValueError(f'Exact cases must have empty tau in {path}: {case_id}')
                elif tau is None or not math.isfinite(tau) or tau <= 0:
                    raise ValueError(f'Missing or invalid tau in {path}: {case_id}')
                if case_id in taus and taus[case_id] != tau:
                    raise ValueError(f'Hyperparameter CSVs disagree on tau for {case_id}')
                taus[case_id] = tau
                table[case_id] = dict(method=method, case_id=case_id, parameters=parameters, tau=tau)
        tables[method] = table
    args._hyper_tables = tables, taus
    return args._hyper_tables


def cpu_case_manifest(args, case):
    """Reuse the common generator settings saved by tuning, without neural CSVs."""
    path = args.tuning_dir / 'cases' / f'{case.id}.json'
    saved = json.loads(path.read_text())
    if (saved.get('schema_version') != SCHEMA_VERSION or saved.get('case_id') != case.id
            or saved.get('case') != asdict(case)):
        raise ValueError(f'Invalid tuning case manifest: {path}')
    tau = saved.get('tau')
    if case.censor_type == 'exact':
        if tau is not None:
            raise ValueError(f'Exact cases must have null tau in {path}')
    elif (isinstance(tau, bool) or not isinstance(tau, (int, float))
          or not math.isfinite(tau) or tau <= 0):
        raise ValueError(f'Missing or invalid tau in {path}')
    return {key: saved[key] for key in ('schema_version', 'case', 'case_id', 'tau', 'mu', 'r')}


def load_spec(args, method, case):
    if method in (*NRT_LOSSES, 'NNCDE', 'DeepSurv'):
        tables, taus = hyperparameter_tables(args)
        path = hyperparameter_paths(args)[method]
        if case.censor_type != 'exact' and case.id not in taus:
            raise ValueError(f'No saved tau for {case.id} in {path}')
        manifest = dict(schema_version=SCHEMA_VERSION, case=asdict(case), case_id=case.id,
                        tau=taus.get(case.id), mu=args.mu, r=args.r)
        if method not in tables:
            raise FileNotFoundError(f'Missing hyperparameter CSV: {path}')
        if case.id not in tables[method]:
            raise ValueError(f'No hyperparameters for {case.id} in {path}')
        tuned = tables[method][case.id]
        parameters = dict(tuned['parameters'])
        if method in NRT_LOSSES:
            training = dict(epochs=DEFAULT_EPOCHS[method], training_batch_size=256,
                            lr=1e-3, weight_decay=1e-4, patience=20,
                            brier_grid_size=args.grid_size, cuda_graphs=True, max_grad_norm=1.0)
        elif method == 'NNCDE':
            training = dict(epochs=DEFAULT_EPOCHS[method], patience=20)
        else:
            training = {name: value for name, value in PAPER_GAUSSIAN_FIXED_PARAMETERS.items()
                        if name not in parameters}
        source_hash = fingerprint(tuned)
    else:
        manifest = cpu_case_manifest(args, case)
        from testing.benchmark.aft.model import distribution_for_scenario
        parameters, training = {}, {}
        if method == 'aft':
            parameters = dict(distribution=distribution_for_scenario(case.scenario), max_iter=args.aft_max_iter)
        elif method == 'deep_distribution_regression':
            training = dict(epochs=args.ddr_epochs, patience=args.ddr_patience)
        elif method == 'iccforest':
            parameters = dict(ntree=args.forest_trees, tune_ntree=args.forest_trees)
        source_hash = fingerprint(manifest)
    if args.epochs is not None and method in (*NRT_LOSSES, 'NNCDE', 'DeepSurv'):
        training['epochs'] = args.epochs
    return manifest, parameters, training, source_hash


def make_job(args, method, case, replication, spec):
    manifest, parameters, training, source_hash = spec
    # Disjoint from tuning seeds, shared across all methods and execution orders.
    seed = args.base_seed + 1_000_000 + case.index * 10_000 + replication
    request = dict(schema_version=SCHEMA_VERSION, method=method, case=asdict(case),
                   replication=replication, data_seed=seed, n_samples=args.n_samples,
                   grid_size=args.grid_size, tau=manifest['tau'], mu=manifest['mu'], r=manifest['r'],
                   parameters=parameters, training_config=training, tuning_hash=source_hash)
    return dict(request=request, request_hash=fingerprint(request), case=case,
                output=result_path(args.output_dir, method, case, replication),
                device=args.device, threads=args.threads_per_fit, force=args.force)


def fit_job(job):
    request, output = job['request'], job['output']
    if output.exists() and not job['force']:
        existing = json.loads(output.read_text())
        if existing.get('request_hash') != job['request_hash']:
            raise ValueError(f'{output} contains a different simulation; use a new output directory or --force')
        if existing.get('status') == 'ok':
            return existing
    torch.set_num_threads(job['threads'])
    model = None
    start = time.monotonic()
    payload = {**request, 'case_id': job['case'].id, 'request_hash': job['request_hash'], 'status': 'error'}
    try:
        data = generate(job['case'], request['n_samples'], request['data_seed'], request['tau'],
                        mu=request['mu'], r=request['r'])
        payload.update(data.metadata())
        model = fit_model(request['method'], data, data.train, request['parameters'],
                          request['training_config'], job['device'], request['data_seed'] + 1, 'testing')
        payload.update(evaluate(model, data, request['grid_size'], method=request['method'],
                                device=job['device'], package='testing'))
        payload['status'] = 'ok'
    except Exception as error:
        payload['error'] = f'{type(error).__name__}: {error}'
    finally:
        if model is not None:
            close_model(model)
    payload['elapsed_seconds'] = time.monotonic() - start
    atomic_json(output, payload)
    print(f'{request["method"]} {job["case"].id} replication={request["replication"]} '
          f'tau={request["tau"]} right_censoring_rate={payload.get("right_censoring_rate")} '
          f'status={payload["status"]}', flush=True)
    return payload


def run(args, cases, methods):
    if args.parallel_fits > 1 and any(method not in CPU_METHODS for method in methods):
        raise ValueError('Neural GPU replications must run sequentially; --parallel-fits is for CPU benchmarks')
    if any(method in CPU_METHODS for method in methods) and torch.device(args.device).type != 'cpu':
        raise ValueError('CPU benchmark methods require --device cpu')
    applicable = [(method, case) for case in cases for method in methods if case.censor_type in SUPPORTED[method]]
    if not applicable:
        raise ValueError('No applicable cases for the requested methods')
    if args.replication is not None:
        replications = [args.replication]
    else:
        last = args.end_replication if args.end_replication is not None else args.replications
        replications = range(args.start_replication, last + 1)
    if any(rep < 1 or rep > args.replications for rep in replications) or len(replications) == 0:
        raise ValueError('Replication range must be within 1..replications')
    if args.dry_run:
        print(f'{len(replications)} replications; {len(applicable)} applicable fits per replication; '
              f'parallel_fits={args.parallel_fits}; '
              f'iccforest_parallel_fits={args.iccforest_parallel_fits}; device={args.device}')
        for method, case in applicable:
            print(f'{method} {case.id}')
        return
    node_preflight(args.device, {method for method, case in applicable}, 'testing')
    specs = {(method, case): load_spec(args, method, case) for method, case in applicable}
    failures = 0
    rates = {(method, case): [] for method, case in applicable}

    def record(result):
        if result['status'] == 'ok':
            case = next(case for case in cases if case.id == result['case_id'])
            rates[result['method'], case].append(result['right_censoring_rate'])
            return 0
        return 1

    # Finish and close the regular pool before the memory-intensive R forests.
    stages = (
        ('other methods', [(method, case) for method, case in applicable if method != 'iccforest'],
         args.parallel_fits),
        ('ICcforest', [(method, case) for method, case in applicable if method == 'iccforest'],
         args.iccforest_parallel_fits),
    )
    for stage_name, stage_cases, workers in stages:
        if not stage_cases:
            continue
        print(f'Running {stage_name} with {workers} fit(s) at a time', flush=True)
        if workers > 1:
            pool_options = dict(max_workers=workers, mp_context=mp.get_context('spawn'))
            if sys.version_info >= (3, 11):
                pool_options['max_tasks_per_child'] = 1
            with ProcessPoolExecutor(**pool_options) as pool:
                for rep in replications:
                    jobs = [make_job(args, method, case, rep, specs[method, case])
                            for method, case in stage_cases]
                    failures += sum(record(result) for result in pool.map(fit_job, jobs, chunksize=1))
        else:
            if stage_name == 'ICcforest':
                gc.collect()
            for rep in replications:
                for method, case in stage_cases:
                    result = fit_job(make_job(args, method, case, rep, specs[method, case]))
                    failures += record(result)
    for (method, case), values in rates.items():
        if not values:
            continue
        mean = statistics.fmean(values)
        tau = specs[method, case][0]['tau']
        print(f'{method} {case.id}: tau={tau} average_right_censoring_rate={mean:.6f} '
              f'q={case.q:g} n_replications={len(values)}', flush=True)
    # Array workers share the saved results. Publish one report only after the
    # entire requested replication set succeeds, independently of GPU methods.
    summarize(args, cases, methods, complete_only=True, censoring_only=True)
    auto_summary(args, cases, methods)
    if failures:
        raise RuntimeError(f'{failures} fits failed; errors are recorded in the replication JSON files. Rerun to retry them.')


def auto_summary(args, cases, methods):
    """Check complete replications across workers before publishing case tables."""
    if args.summary_methods == 'all':
        methods = list(METHODS)
    elif args.summary_methods is not None:
        methods = select_methods(args.summary_methods)
    summarize(args, cases, methods, complete_only=True)


def summarize(args, cases, methods, *, complete_only=False, censoring_only=False):
    # The combined report needs every case. Exit before loading any JSON while
    # other array tasks are still running, even if individual cases are complete.
    if censoring_only and complete_only and any(
            not result_path(args.output_dir, method, case, rep).is_file()
            for rep in range(args.replications, 0, -1) for case in cases
            for method in methods if case.censor_type in SUPPORTED[method]):
        print('Censoring report pending: waiting for all requested successful replications', flush=True)
        return
    tables, censoring_rows, method_censoring_rows = {}, [], []
    completed = 0
    missing, failed = 0, 0
    for case in cases:
        data_by_replication = {}
        case_methods = [method for method in methods if case.censor_type in SUPPORTED[method]]
        if not case_methods:
            continue
        # Most array workers finish before the full result set exists. Check
        # backwards and stop at the first absent file to avoid reading partial
        # results or repeatedly scanning every replication on shared storage.
        if complete_only and not censoring_only and any(
                not result_path(args.output_dir, method, case, rep).is_file()
                for rep in range(args.replications, 0, -1) for method in case_methods):
            missing += 1
            continue
        table_rows = [dict(replication=rep, **{method: None for method in case_methods})
                      for rep in range(1, args.replications + 1)]
        scores = {method: [] for method in case_methods}
        case_incomplete = False
        for method in case_methods:
            results = []
            group_missing = group_failed = 0
            for rep in range(1, args.replications + 1):
                path = result_path(args.output_dir, method, case, rep)
                if not path.exists():
                    group_missing += 1
                    continue
                result = json.loads(path.read_text())
                if result.get('case') != asdict(case) or result.get('method') != method or result.get('replication') != rep:
                    raise ValueError(f'Result identifiers disagree with path: {path}')
                if result['status'] != 'ok':
                    group_failed += 1
                    continue
                metadata = {key: result[key] for key in ('data_seed', 'n_samples', 'tau', 'mu', 'r', 'right_censoring_rate')}
                if rep in data_by_replication and data_by_replication[rep] != metadata:
                    raise ValueError(f'Methods used different simulation data for {case.id}, replication {rep}')
                data_by_replication[rep] = metadata
                value = float(result['average_grid_sse'])
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite average_grid_sse: {path}')
                table_rows[rep - 1][method] = value
                scores[method].append(value)
                results.append(result)
            missing += group_missing
            failed += group_failed
            case_incomplete |= bool(group_missing or group_failed)
            if not results:
                continue
            taus = {result['tau'] for result in results}
            configs = {fingerprint({key: result[key] for key in ('parameters', 'training_config', 'tuning_hash', 'n_samples', 'grid_size', 'mu', 'r')}) for result in results}
            if len(taus) != 1 or len(configs) != 1:
                raise ValueError(f'Mixed tuning parameters or tau for {method} {case.id}')
            if censoring_only:
                rates = [result['right_censoring_rate'] for result in results]
                mean = statistics.fmean(rates)
                method_censoring_rows.append(dict(method=method, case_id=case.id, **asdict(case),
                    tau=results[0]['tau'], n_replications=len(rates), average_right_censoring_rate=mean,
                    sd_right_censoring_rate=statistics.stdev(rates) if len(rates) > 1 else 0.0,
                    difference_from_q=None if case.censor_type == 'exact' else mean - case.q))
            completed += len(results)
            print(f'{method} {case.id} n={len(results)}/{args.replications} tau={results[0]["tau"]} '
                  f'average_right_censoring_rate={statistics.fmean(result["right_censoring_rate"] for result in results):.6f}', flush=True)
        if complete_only and case_incomplete:
            continue
        table_rows.append(dict(replication='average', **{
            method: statistics.fmean(values) if values else None
            for method, values in scores.items()}))
        table_rows.append(dict(replication='standard_deviation', **{
            method: (statistics.stdev(values) if len(values) > 1 else 0.0) if values else None
            for method, values in scores.items()}))
        tables[case.id] = table_rows
        if data_by_replication:
            rates = [data['right_censoring_rate'] for data in data_by_replication.values()]
            taus = {data['tau'] for data in data_by_replication.values()}
            if len(taus) != 1:
                raise ValueError(f'Mixed tau across methods for {case.id}')
            censoring_rows.append(dict(case_id=case.id, **asdict(case), tau=next(iter(taus)),
                n_replications=len(rates), average_right_censoring_rate=statistics.fmean(rates),
                sd_right_censoring_rate=statistics.stdev(rates) if len(rates) > 1 else 0.0,
                difference_from_q=None if case.censor_type == 'exact' else statistics.fmean(rates) - case.q))
    if complete_only and (missing or failed):
        if censoring_only:
            print('Censoring report pending: waiting for all requested successful replications', flush=True)
        else:
            print(f'Simulation summary pending: {len(tables)} case table(s) complete; '
                  'waiting for all requested successful replications', flush=True)
    if not complete_only and (missing or failed) and not args.allow_incomplete:
        raise RuntimeError(f'{missing} results missing, {failed} failed; use --allow-incomplete for a partial summary')
    if complete_only and not tables:
        return
    if not completed:
        raise RuntimeError('No successful simulation results to summarize')
    if censoring_only:
        if complete_only and (missing or failed):
            return
        path = args.output_dir / f'censoring_{"_".join(methods)}_rep1_{args.replications}.csv'
        write_csv(path, method_censoring_rows)
        print(f'Wrote {path}', flush=True)
        return
    for case_id, table_rows in tables.items():
        path = args.output_dir / 'tables' / f'{case_id}.csv'
        write_csv(path, table_rows)
        print(f'Wrote {path}', flush=True)
    if not complete_only or not (missing or failed):
        write_csv(args.output_dir / 'censoring_summary.csv', censoring_rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'summarize'))
    p.add_argument('--methods', default=','.join(METHODS))
    p.add_argument('--summary-methods',
                   help='Automatic table columns: all or comma-separated methods; defaults to --methods for direct runs')
    p.add_argument('--cases', default='all')
    p.add_argument('--hyper-dir', type=Path, default=ROOT / 'testing' / 'hyper_para',
                   help='Directory containing neural <method>.csv parameter and tau tables')
    p.add_argument('--hyper-file', type=Path,
                   help='Explicit CSV for a single neural method; ignores other files in --hyper-dir')
    p.add_argument('--tuning-dir', type=Path, default=ROOT / 'tuning' / 'output',
                   help='Tuning output directory containing cases/<case_id>.json for CPU methods')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'testing' / 'output')
    p.add_argument('--replications', type=int, default=500)
    p.add_argument('--replication', type=int)
    p.add_argument('--start-replication', type=int, default=1)
    p.add_argument('--end-replication', type=int)
    p.add_argument('--n-samples', type=int, default=5000)
    p.add_argument('--base-seed', type=int, default=123)
    p.add_argument('--mu', type=float, default=0.0, help='Generator mu used when tuning the CSV parameters')
    p.add_argument('--r', type=float, help='Generator r used when tuning the CSV parameters')
    p.add_argument('--grid-size', type=int, default=256)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--parallel-fits', type=int, default=1)
    p.add_argument('--iccforest-parallel-fits', type=int, default=1,
                   help='ICcforest fits in a separate phase; default 1 limits R memory use')
    p.add_argument('--threads-per-fit', type=int, default=1)
    p.add_argument('--epochs', type=int, help='Override epoch limit: NRT/NNCDE default to 200; DeepSurv defaults to 1000')
    p.add_argument('--ddr-epochs', type=int, default=100)
    p.add_argument('--ddr-patience', type=int, default=20)
    p.add_argument('--forest-trees', type=int, default=100)
    p.add_argument('--aft-max-iter', type=int, default=250)
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--censoring-only', action='store_true',
                   help='For summarize, write only one method/case censoring report across replications')
    return p


def main():
    args = parser().parse_args()
    if min(args.replications, args.parallel_fits, args.iccforest_parallel_fits, args.threads_per_fit, args.ddr_epochs,
           args.ddr_patience, args.forest_trees, args.aft_max_iter) < 1 or args.grid_size < 2 or args.n_samples < 10:
        raise ValueError('Counts must be positive; n_samples >= 10 and grid_size >= 2')
    if args.epochs is not None and args.epochs < 1:
        raise ValueError('epochs must be positive')
    cases, methods = select_cases(args.cases), select_methods(args.methods)
    if args.hyper_file is not None:
        hyperparameter_paths(args)
    if args.summary_methods is not None and args.summary_methods != 'all':
        select_methods(args.summary_methods)
    if args.command == 'run':
        run(args, cases, methods)
    else:
        summarize(args, cases, methods, censoring_only=args.censoring_only)


if __name__ == '__main__':
    main()
