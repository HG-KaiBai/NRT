"""Read the self-contained SUPPORT CSV; never use split metadata as features.

Exact export of the author-distributed HDF5 from jaredleekatzman/DeepSurv,
commit f0803f57bff9395f2bfbb99da2f1031adcf01331,
experiments/data/support/support_train_test.h5. DOI: 10.1186/s12874-018-0482-1.
8,873 rows, 14 anonymous x columns, original 7,098/1,775 split; time is in days,
event=1 means observed death. No imputation, filtering, relabeling or scaling
was applied to the CSV. The feature order is the original HDF5 matrix order.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import ROOT

DEFAULT_DATA = ROOT / "data/support/support_deepsurv.csv"
FEATURES = [f"x{i:02d}" for i in range(14)]
SUPPORT_SHA256 = "76916ee819f1e7610d1c4376ead3ef238877058607f5ca291fb3e6a1ba9bf734"
# The current CSV serialization was also verified against every x/t/e value
# and source row in the original HDF5 at the commit documented above. Accept
# only these two verified byte representations; do not bypass the checksum.
SUPPORT_VERIFIED_SHA256 = frozenset({
    SUPPORT_SHA256,
    "aac5a35ec6de18a1b39bb584f5e1aa22a7c6e3348a1115d8867229ca8a0b31e8",
})


@dataclass
class SupportData:
    frame: pd.DataFrame
    x: torch.Tensor
    time: torch.Tensor
    event: torch.Tensor
    train: torch.Tensor
    test: torch.Tensor
    sha256: str

    def bounds(self, indices, scaler=None):
        x = self.x[indices]
        if scaler is not None:
            x = (x - scaler[0]) / scaler[1]
        left = self.time[indices]
        right = torch.where(self.event[indices], left, torch.inf)
        return x, left, right

    def scaler(self, indices):
        x = self.x[indices]
        return x.mean(0), x.std(0, correction=0).clamp_min(1e-6)

    def outcomes(self, indices):
        from sksurv.util import Surv
        return Surv.from_arrays(self.event[indices].numpy(), self.time[indices].numpy())

    def metadata(self):
        return [dict(split=name, n=len(ix), events=int(self.event[ix].sum()),
                     censored=int((~self.event[ix]).sum()),
                     censoring_rate=float((~self.event[ix]).double().mean()),
                     min_days=float(self.time[ix].min()), max_days=float(self.time[ix].max()))
                for name, ix in (("train", self.train), ("test", self.test))]


def load_support(path=DEFAULT_DATA, *, smoke=False, seed=123):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest not in SUPPORT_VERIFIED_SHA256:
        raise ValueError("SUPPORT CSV checksum differs from the verified author-data export")
    frame = pd.read_csv(path, float_precision="round_trip")
    if set(frame.columns) != set(FEATURES + ["split", "row_in_split", "time", "event"]):
        raise ValueError("Unexpected SUPPORT schema")
    if frame.duplicated(["split", "row_in_split"]).any():
        raise ValueError("Duplicate source row identifiers")
    numeric = frame[FEATURES + ["time", "event"]].to_numpy()
    if not np.isfinite(numeric).all() or (frame.time <= 0).any() or not frame.event.isin([0, 1]).all():
        raise ValueError("Invalid/missing covariates or survival outcomes")
    train = torch.tensor(np.flatnonzero(frame.split == "train"))
    test = torch.tensor(np.flatnonzero(frame.split == "test"))
    if (len(train), len(test)) != (7098, 1775):
        raise ValueError("Expected the original 7098/1775 train/test partition")
    if smoke:
        generator = torch.Generator().manual_seed(seed)
        train = train[torch.randperm(len(train), generator=generator)[:160]]
        test = test[torch.randperm(len(test), generator=generator)[:80]]
    return SupportData(frame, torch.tensor(frame[FEATURES].to_numpy(), dtype=torch.float32),
                       torch.tensor(frame.time.to_numpy(), dtype=torch.float64),
                       torch.tensor(frame.event.to_numpy(), dtype=torch.bool), train, test, digest)
