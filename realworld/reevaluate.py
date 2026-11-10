"""Re-predict saved fitted models on a revised grid, without tuning or fitting."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from . import ROOT
from .bootstrap import _sha256
from .data import load_dataset
from .metrics import evaluate, evaluation_grid, uno_concordance_curve
from regularization.workflow import writer_lock
from workflow import atomic_json, write_csv, fingerprint, close_model


def load_saved_model(folder, method, feature_count):
    """Load trusted local experiment artifacts on CPU, including R wrappers."""
    folder = Path(folder)
    if (folder/'model.pt').exists():
        model = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
        if hasattr(model, 'device'):
            model.device = torch.device('cpu')
        if isinstance(model, torch.nn.Module):
            model.eval()
        return model
    from testing.benchmark.cox import CoxPH
    from testing.benchmark.ctree import CTree
    from testing.benchmark.aft import ParametricAFT
    from testing.benchmark.iccforest import ICCForest
    model = dict(cox=CoxPH, ctree=CTree, aft=ParametricAFT, iccforest=ICCForest)[method]()
    model._temporary_directory = tempfile.TemporaryDirectory(prefix='realworld_reevaluate_')
    model._model_path = Path(model._temporary_directory.name)/'model.rds'
    shutil.copy2(folder/'model.rds', model._model_path)
    model._feature_count = feature_count
    return model


def reevaluate_method(source, output, dataset, method):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('Reevaluation requires a separate output directory to preserve training provenance')
    original, folder = source/dataset/method, output/dataset/method
    result = json.loads((original/'result.json').read_text())
    if result.get('status') != 'complete' or result['dataset'] != dataset or result['method'] != method:
        raise ValueError('A matching completed fitted model is required')
    manifest = json.loads((source/dataset/'data.json').read_text())
    data = load_dataset(dataset)
    if data.source_sha256 != manifest['source_sha256']:
        raise ValueError('Source data changed since training')
    splits = pd.read_csv(source/dataset/'split_assignments.csv')
    if (not pd.api.types.is_integer_dtype(splits['row_id']) or splits['row_id'].duplicated().any()
            or not splits['row_id'].between(0, len(data.time)-1).all()
            or not splits['partition'].isin(['train', 'test']).all()):
        raise ValueError('Invalid saved train/test assignments')
    train = np.sort(splits.loc[splits['partition']=='train', 'row_id'].to_numpy())
    test = np.sort(splits.loc[splits['partition']=='test', 'row_id'].to_numpy())
    if len(train) != manifest['n_train'] or len(test) != manifest['n_test']:
        raise ValueError('Saved split sizes do not match the manifest')
    preprocessing = json.loads((original/'preprocessing.json').read_text())
    if preprocessing['covariates'] != list(data.frame):
        raise ValueError('Saved covariate order does not match the data')
    grid = evaluation_grid(data.time[train], data.event[train], censored=data.censor_type=='right')
    model_path = original/('model.pt' if (original/'model.pt').exists() else 'model.rds')
    # This is evaluation provenance; the original training request remains intact.
    request = dict(schema_version=1, source_run=str(source), dataset=dataset, method=method,
                   training_request_hash=result['request_hash'],
                   inputs={str(p.relative_to(source)): _sha256(p) for p in
                           (model_path, original/'preprocessing.json', original/'result.json',
                            source/dataset/'split_assignments.csv', source/dataset/'data.json')},
                   source_sha256={name: _sha256(ROOT/name) for name in
                                  ('reevaluate.py', 'metrics.py', 'data.py', 'models.py')},
                   grid=grid.tolist())
    digest = fingerprint(request)
    root_provenance = dict(training_source=str(source), source_sha256=request['source_sha256'])
    with writer_lock(output):
        path = output/'reevaluation.json'
        if path.exists() and json.loads(path.read_text()) != root_provenance:
            raise ValueError('Reevaluation source changed; choose a new output directory')
        if (output/'request.json').exists():
            raise ValueError('Do not overwrite a training run with reevaluation outputs')
        atomic_json(path, root_provenance)
    new_manifest = dict(manifest, grid=grid.tolist(),
                        horizon_policy='simulation: 256 points from zero to largest finite training bound',
                        training_source=str(source/dataset),
                        original_training_manifest_sha256=_sha256(source/dataset/'data.json'))
    with writer_lock(output/dataset):
        destination = output/dataset/'data.json'
        if destination.exists() and json.loads(destination.read_text()) != new_manifest:
            raise ValueError('Reevaluation dataset changed; choose a new output directory')
        atomic_json(destination, new_manifest)
        shutil.copy2(source/dataset/'split_assignments.csv', output/dataset/'split_assignments.csv')
    with writer_lock(folder):
        if (folder/'result.json').exists():
            previous = json.loads((folder/'result.json').read_text())
            if previous.get('evaluation_hash') == digest:
                print(f'Reusing reevaluation {dataset}/{method}', flush=True)
                return
            raise ValueError('Reevaluation inputs changed; choose a new output directory')
        print(f'Re-predicting {dataset}/{method} on 256 points through {grid[-1]:g}', flush=True)
        model = load_saved_model(original, method, data.frame.shape[1])
        try:
            x, _, _ = data.bounds(test, preprocessing['scaler'])
            with torch.no_grad():
                # R backends handle the whole test set; neural adapters stream it.
                batches = (x,) if model_path.suffix == '.rds' else x.split(64)
                survival = torch.cat([model.predict_survival(batch, torch.as_tensor(grid)).cpu()
                                      for batch in batches]).numpy()
        finally:
            close_model(model)
        scores, curve, risk = evaluate(data.time[train], data.event[train], data.time[test],
            data.event[test], survival, grid, censored=data.censor_type=='right')
        np.savez_compressed(folder/'predictions.npz', row_ids=test, time=data.time[test],
                            event=data.event[test], survival=survival, grid=grid, risk=risk)
        write_csv(folder/'brier_curve.csv', [dict(time=float(t), brier_score=float(b))
                                            for t, b in zip(grid, curve)])
        if data.censor_type == 'right':
            c_curve = uno_concordance_curve(data.time[test], data.event[test], survival, grid)
            write_csv(folder/'c_index_curve.csv', [dict(time=float(t), c_index=float(c) if np.isfinite(c) else None)
                                                  for t, c in zip(grid, c_curve)])
        atomic_json(folder/'evaluation.json', dict(evaluation_hash=digest, request=request))
        atomic_json(folder/'result.json', dict(result, **scores, evaluation_hash=digest,
                    training_source=str(original), evaluation_device='cpu'))
        shutil.copy2(original/'selection.json', folder/'selection.json')
        shutil.copy2(original/'preprocessing.json', folder/'preprocessing.json')
        print(f'Reevaluated {dataset}/{method}: C={scores["c_index"]:.6f}, '
              f'IBS={scores["integrated_brier_score"]:.6f}', flush=True)
