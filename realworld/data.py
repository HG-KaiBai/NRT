"""Read pinned SUPPORT/METABRIC CSVs; never use split metadata as features.

Exact export of the author-distributed HDF5 from jaredleekatzman/DeepSurv,
commit f0803f57bff9395f2bfbb99da2f1031adcf01331,
experiments/data/support/support_train_test.h5. DOI: 10.1186/s12874-018-0482-1.
8,873 rows, 14 anonymous x columns, original 7,098/1,775 split; time is in days,
event=1 means observed death. No imputation, filtering, relabeling or scaling
was applied to the SUPPORT CSV. Feature order is the original HDF5 matrix order.
METABRIC uses the supplied 1523/381 split and nine features; its raw follow-up
is in months. Its checksum pins the supplied file, not an independently verified
HDF5 export. Preserve the test row censored at time zero.
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
class SurvivalData:
    frame: pd.DataFrame
    x: torch.Tensor
    time: torch.Tensor
    event: torch.Tensor
    train: torch.Tensor
    test: torch.Tensor
    sha256: str
    name: str = "support"
    time_unit: str = "days"

    @property
    def features(self):
        return [f"x{i:02d}" for i in range(self.x.shape[1])]

    @property
    def landmarks(self):
        return (30, 90, 180, 365, 730) if self.name == "support" else (12, 36, 60, 120, 240)

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
                     min_time=float(self.time[ix].min()), max_time=float(self.time[ix].max()),
                     time_unit=self.time_unit)
                for name, ix in (("train", self.train), ("test", self.test))]


METABRIC_DATA = ROOT / "data/metabric/metabric_deepsurv.csv"
METABRIC_SHA256 = "ba502e364efab52916d4316e97d6020e924545bef0f88802f4d77f973cf0eeb1"
DATASETS = {
    "support": dict(path=DEFAULT_DATA, features=FEATURES, sizes=(7098, 1775),
                    checksums=SUPPORT_VERIFIED_SHA256, time_unit="days"),
    "metabric": dict(path=METABRIC_DATA, features=[f"x{i:02d}" for i in range(9)],
                     sizes=(1523, 381), checksums={METABRIC_SHA256}, time_unit="months"),
}


def load_data(dataset="support", path=None, *, smoke=False, seed=123):
    spec = DATASETS[dataset]
    features = spec["features"]
    path = Path(path or spec["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest not in spec["checksums"]:
        raise ValueError(f"{dataset.upper()} CSV checksum differs from the pinned export")
    frame = pd.read_csv(path, float_precision="round_trip")
    if set(frame.columns) != set(features + ["split", "row_in_split", "time", "event"]):
        raise ValueError("Unexpected survival-data schema")
    if frame.duplicated(["split", "row_in_split"]).any():
        raise ValueError("Duplicate source row identifiers")
    numeric = frame[features + ["time", "event"]].to_numpy()
    if not np.isfinite(numeric).all() or (frame.time < 0).any() or not frame.event.isin([0, 1]).all():
        raise ValueError("Invalid/missing covariates or survival outcomes")
    train = torch.tensor(np.flatnonzero(frame.split == "train"))
    test = torch.tensor(np.flatnonzero(frame.split == "test"))
    if (len(train), len(test)) != spec["sizes"] or len(train) + len(test) != len(frame):
        raise ValueError(f"Expected the original {spec['sizes']} train/test partition")
    if smoke:
        generator = torch.Generator().manual_seed(seed)
        train = train[torch.randperm(len(train), generator=generator)[:160]]
        test = test[torch.randperm(len(test), generator=generator)[:80]]
    return SurvivalData(frame, torch.tensor(frame[features].to_numpy(), dtype=torch.float32),
                       torch.tensor(frame.time.to_numpy(), dtype=torch.float64),
                       torch.tensor(frame.event.to_numpy(), dtype=torch.bool), train, test, digest, dataset, spec["time_unit"])


# Keep the existing SUPPORT Python entry points available.
SupportData = SurvivalData


def load_support(path=DEFAULT_DATA, *, smoke=False, seed=123):
    return load_data("support", path, smoke=smoke, seed=seed)


def load_metabric(path=METABRIC_DATA, *, smoke=False, seed=123):
    return load_data("metabric", path, smoke=smoke, seed=seed)
