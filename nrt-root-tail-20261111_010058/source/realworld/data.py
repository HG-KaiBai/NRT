"""Dataset definitions, common 80/20 splits, and training-only preprocessing."""
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
import torch

from . import ROOT


@dataclass
class RealData:
    name: str
    target: str
    censor_type: str
    frame: pd.DataFrame
    time: np.ndarray
    event: np.ndarray
    train: torch.Tensor
    test: torch.Tensor
    row_ids: np.ndarray
    source_sha256: str

    def scaler(self, indices):
        values = self.frame.iloc[np.asarray(indices)].to_numpy(dtype=float)
        return dict(columns=list(self.frame), mean=values.mean(0).tolist(),
                    scale=np.maximum(values.std(0), 1e-6).tolist())

    def bounds(self, indices, scaler=None):
        ix = np.asarray(indices)
        x = self.frame.iloc[ix].to_numpy(dtype=float)
        if scaler is not None:
            x = (x - np.asarray(scaler['mean'])) / np.asarray(scaler['scale'])
        t = self.time[ix]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32),
                torch.tensor(np.where(self.event[ix], t, np.inf), dtype=torch.float32))


def load_dataset(name, *, gt_target='CO', seed=123, limit=None):
    source = 'rotterdam' if name in ('rotterdam', 'rotterdam_recur', 'rotterdam_death') else name
    path = ROOT / 'data' / f'{source}.csv'
    raw = pd.read_csv(path)
    row_ids = np.arange(len(raw))
    if name == 'gt_2011':
        if gt_target not in ('NOX', 'CO', 'TEY'):
            raise ValueError('gt target must be NOX, CO or TEY')
        target, censor = gt_target, 'exact'
        time = raw[gt_target].to_numpy(float)
        event = np.ones(len(raw), dtype=bool)
        frame = raw.drop(columns=gt_target)
    elif source == 'rotterdam':
        target, time_column = ('death', 'dtime') if name == 'rotterdam_death' else ('recur', 'rtime')
        censor = 'right'
        time = raw[time_column].to_numpy(float)
        event = raw[target].to_numpy()
        if not np.isin(event, [0, 1]).all():
            raise ValueError('Event must be binary')
        event = event.astype(bool)
        # Explicit schema, independent of test-set category frequencies.
        frame = raw.drop(columns=['pid', 'rtime', 'recur', 'dtime', 'death', 'size'])
        if not raw['size'].isin(['<=20', '20-50', '>50']).all():
            raise ValueError('Unknown tumour-size category')
        frame['size_20_50'] = (raw['size'] == '20-50').astype(float)
        frame['size_gt50'] = (raw['size'] == '>50').astype(float)
    else:
        raise ValueError(f'Unknown dataset: {name}')
    if (not np.isfinite(frame.to_numpy(float)).all() or
            not np.isfinite(time).all() or not (time > 0).all()):
        raise ValueError('This protocol requires finite covariates and positive outcomes')
    if limit is not None:
        # Smoke subset only; full runs always retain every row.
        ix = np.random.default_rng(seed).permutation(len(raw))[:limit]
        frame, time, event, row_ids = frame.iloc[ix].reset_index(drop=True), time[ix], event[ix], row_ids[ix]
    order = torch.randperm(len(frame), generator=torch.Generator().manual_seed(seed + 1))
    split = int(.8 * len(frame))
    return RealData(name, target, censor, frame, time, event, order[:split], order[split:],
                    row_ids, hashlib.sha256(path.read_bytes()).hexdigest())
