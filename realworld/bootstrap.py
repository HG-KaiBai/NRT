"""Test-subject bootstrap conditional on the fitted model and training sample."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT
from .data import load_dataset
from .metrics import (evaluate, evaluation_components, brier_contributions,
                      concordance_pairs, uno_ipcw_weights)
from regularization.workflow import writer_lock
from workflow import atomic_json, fingerprint, write_csv

METRICS = ('c_index', 'integrated_brier_score', 'brier_at_horizon')
DEFAULT_RESAMPLES = 500
DEFAULT_SEED = 20261102
DEFAULT_CONFIDENCE = .95


def validate_options(n_resamples, confidence_level, seed):
    if type(n_resamples) is not int or n_resamples < 2:
        raise ValueError('bootstrap resamples must be an integer >= 2')
    if not 0 < confidence_level < 1:
        raise ValueError('bootstrap confidence level must be between 0 and 1')
    if type(seed) is not int or seed < 0:
        raise ValueError('bootstrap seed must be a nonnegative integer')


def bootstrap_metrics(train_time, train_event, time, event, survival, grid, *, censored,
                      n_resamples=DEFAULT_RESAMPLES, confidence_level=DEFAULT_CONFIDENCE,
                      seed=DEFAULT_SEED):
    """Resample subjects with fixed predictions; refit both IPCW censoring KMs.

    Multiplicity counts exactly reproduce explicit resampling. Duplicated copies
    of the same subject have tied outcomes and cannot form comparable pairs.
    """
    validate_options(n_resamples, confidence_level, seed)
    time, event, grid = np.asarray(time), np.asarray(event, bool), np.asarray(grid)
    estimates, curve, _ = evaluate(
        train_time, train_event, time, event, survival, grid, censored=censored)
    losses, risk = evaluation_components(
        train_time, train_event, time, event, survival, grid, censored=censored)
    numerator, denominator = concordance_pairs(
        time, event, risk, horizon=grid[-1] if censored else None)
    rng = np.random.default_rng(seed)
    n = len(time)
    cindex = np.full(n_resamples, np.nan)
    curves = np.empty((n_resamples, len(grid)))
    # Generate indices in a fixed order so all methods use identical draws.
    for start in range(0, n_resamples, 64):
        stop = min(start + 64, n_resamples)
        indices = rng.integers(0, n, size=(stop-start, n))
        counts = np.array([np.bincount(row, minlength=n) for row in indices], dtype=float)
        anchors = counts.copy()
        if censored:
            for j, count in enumerate(counts):
                anchors[j] *= uno_ipcw_weights(time, event, grid[-1], counts=count)
                resampled_losses = brier_contributions(time, event, survival, grid,
                                                       censored=True, counts=count)
                curves[start+j] = count @ resampled_losses / n
        else:
            curves[start:stop] = counts @ losses / n
        den = np.sum((counts @ denominator.T) * anchors, axis=1)
        num = np.sum((counts @ numerator.T) * anchors, axis=1)
        np.divide(num, den, out=cindex[start:stop], where=den > 0)
    replicates = dict(c_index=cindex,
                      integrated_brier_score=np.trapezoid(curves, grid, axis=1)/(grid[-1]-grid[0]),
                      brier_at_horizon=curves[:, -1])
    quantiles = [(1-confidence_level)/2, (1+confidence_level)/2]
    intervals = {}
    for name, values in replicates.items():
        valid = values[np.isfinite(values)]
        bounds = np.quantile(valid, quantiles) if len(valid) >= 2 else [None, None]
        intervals[name] = dict(estimate=estimates[name],
                               lower=None if bounds[0] is None else float(bounds[0]),
                               upper=None if bounds[1] is None else float(bounds[1]),
                               n_valid=len(valid), n_invalid=n_resamples-len(valid))
    lower, upper = np.quantile(curves, quantiles, axis=0)
    return dict(metrics=intervals, replicates=replicates, curve=curve,
                curve_lower=lower, curve_upper=upper, estimates=estimates)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_method(output, dataset, method, *, n_resamples=DEFAULT_RESAMPLES,
                     confidence_level=DEFAULT_CONFIDENCE, seed=DEFAULT_SEED):
    """Evaluate saved predictions using their saved split, without calling training."""
    validate_options(n_resamples, confidence_level, seed)
    output = Path(output)
    folder = output/dataset/method
    with writer_lock(folder):
        result = json.loads((folder/'result.json').read_text())
        if result['status'] != 'complete' or result['dataset'] != dataset or result['method'] != method:
            raise ValueError(f'No matching completed result for {dataset}/{method}')
        manifest = json.loads((output/dataset/'data.json').read_text())
        data = load_dataset(dataset)
        if data.source_sha256 != manifest['source_sha256']:
            raise ValueError('Source data changed since the saved predictions were generated')
        splits = pd.read_csv(output/dataset/'split_assignments.csv')
        if (not pd.api.types.is_integer_dtype(splits['row_id'])
                or splits['row_id'].duplicated().any()
                or not splits['row_id'].between(0, len(data.time)-1).all()
                or not splits['partition'].isin(['train', 'test']).all()):
            raise ValueError('Invalid saved train/test assignments')
        train_ids = splits.loc[splits['partition']=='train', 'row_id'].to_numpy()
        test_ids = np.sort(splits.loc[splits['partition']=='test', 'row_id'].to_numpy())
        if len(train_ids) != manifest['n_train'] or len(test_ids) != manifest['n_test']:
            raise ValueError('Saved split size does not match the dataset manifest')
        with np.load(folder/'predictions.npz', allow_pickle=False) as saved:
            order = np.argsort(saved['row_ids'])
            if not np.array_equal(saved['row_ids'][order], test_ids):
                raise ValueError('Prediction row IDs do not match the saved test set')
            time, event = saved['time'][order], saved['event'][order]
            survival, grid = saved['survival'][order], saved['grid']
        if (not np.array_equal(time, data.time[test_ids])
                or not np.array_equal(event, data.event[test_ids])
                or not np.array_equal(grid, manifest['grid'])):
            raise ValueError('Saved prediction outcomes or grid do not match the dataset')
        inputs = {str(path.relative_to(output)): _sha256(path) for path in (
            folder/'predictions.npz', folder/'result.json', output/dataset/'data.json',
            output/dataset/'split_assignments.csv')}
        request = dict(schema_version=4, dataset=dataset, method=method,
                       training_request_hash=result['request_hash'], input_sha256=inputs,
                       n_resamples=n_resamples, confidence_level=confidence_level, seed=seed,
                       source_sha256={name: _sha256(ROOT/name) for name in
                                      ('bootstrap.py', 'metrics.py', 'data.py')},
                       versions={name: importlib.metadata.version(name)
                                 for name in ('numpy', 'scikit-survival')})
        digest = fingerprint(request)
        destination = folder/'bootstrap.json'
        if destination.exists() and all((folder/name).exists() for name in
                                        ('bootstrap_replicates.csv', 'brier_curve_ci.csv')):
            previous = json.loads(destination.read_text())
            if previous.get('bootstrap_hash') == digest:
                print(f'Reusing bootstrap {dataset}/{method}', flush=True)
                return previous
        values = bootstrap_metrics(data.time[train_ids], data.event[train_ids],
                                   time, event, survival, grid, censored=data.censor_type=='right',
                                   n_resamples=n_resamples, confidence_level=confidence_level, seed=seed)
        for definition in ('c_index_definition', 'brier_definition', 'risk_definition'):
            if values['estimates'][definition] != result.get(definition):
                raise ValueError(f'Saved {definition} differs from current evaluation; reevaluate first')
        for metric in METRICS:
            if not np.isclose(values['estimates'][metric], result[metric], rtol=1e-10, atol=1e-12):
                raise ValueError(f'Saved {metric} does not match recomputed evaluation')
        write_csv(folder/'bootstrap_replicates.csv', [dict(replicate=i+1, **{
            name: float(samples[i]) if np.isfinite(samples[i]) else None
            for name, samples in values['replicates'].items()}) for i in range(n_resamples)])
        write_csv(folder/'brier_curve_ci.csv', [dict(time=float(t), brier_score=float(value),
            ci_lower=float(low), ci_upper=float(high), confidence_level=confidence_level,
            n_resamples=n_resamples) for t,value,low,high in
            zip(grid, values['curve'], values['curve_lower'], values['curve_upper'])])
        report = dict(status='complete', bootstrap_hash=digest, request=request,
                      interval_method='percentile', quantile_method='linear', n_test=len(test_ids),
                      sampling='test subjects with replacement; sample size equals original test size',
                      row_order='ascending original row_id; same seed gives paired draws across methods',
                      conditional_on='fixed training sample, fitted model and evaluation grid',
                      brier_censoring_distribution=('test KM refitted in every bootstrap replicate'
                                                    if data.censor_type=='right' else 'none'),
                      c_index_censoring_distribution=('test reverse KM refitted in every bootstrap replicate'
                                                      if data.censor_type=='right' else 'none'),
                      c_index_definition=values['estimates']['c_index_definition'],
                      metric_references=values['estimates']['metric_references'],
                      curve_intervals='pointwise, not simultaneous',
                      invalid_c_index_policy='no comparable pairs or zero event-time G: omit only C-index; counts reported',
                      metrics=values['metrics'])
        atomic_json(destination, report)
        valid = values['metrics']['c_index']['n_valid']
        print(f'Bootstrap {dataset}/{method}: {n_resamples} draws, {valid} valid C-index values', flush=True)
        return report


def summary_columns(output, dataset, method):
    """Attach intervals only when they belong to the current saved predictions."""
    path = Path(output)/dataset/method/'bootstrap.json'
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    request = report['request']
    if report['status'] != 'complete' or any(
            not (Path(output)/name).is_file() or _sha256(Path(output)/name) != digest
            for name,digest in request['input_sha256'].items()):
        return {}
    columns = {f'bootstrap_{key}': request[key] for key in ('n_resamples', 'confidence_level', 'seed')}
    for name, interval in report['metrics'].items():
        columns.update({f'{name}_ci_lower': interval['lower'], f'{name}_ci_upper': interval['upper'],
                        f'{name}_bootstrap_valid': interval['n_valid']})
    return columns
