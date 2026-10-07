"""Python adapter for the interval-censored ``partykit::ctree`` benchmark."""

from __future__ import annotations

import csv
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch


def _find_rscript() -> str | None:
    executable = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if executable:
        return executable
    candidates = sorted(
        glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"), reverse=True
    )
    return candidates[0] if candidates else None


def r_backend_available() -> bool:
    """Return whether Rscript and the required R packages are available."""
    executable = _find_rscript()
    if executable is None:
        return False
    probe = subprocess.run(
        [
            executable,
            "--vanilla",
            "-e",
            "packages <- c('partykit','icenReg','survival'); "
            "quit(status=ifelse(all(sapply(packages, requireNamespace, quietly=TRUE)),0,1))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


class CTree:
    """Fu--Simonoff interval-censored conditional inference survival tree."""

    supported_censor_types = frozenset({"exact", "right", "interval", "current"})

    def __init__(
        self,
        *,
        mincriterion: float = 0.95,
        minsplit: int = 20,
        minbucket: int = 7,
        maxdepth: int | None = None,
        seed: int = 123,
        rscript: str | None = None,
    ) -> None:
        if not 0.0 < mincriterion < 1.0:
            raise ValueError("mincriterion must be in (0, 1)")
        if minsplit < 2 or minbucket < 1:
            raise ValueError("minsplit must be at least 2 and minbucket must be positive")
        if maxdepth is not None and maxdepth < 0:
            raise ValueError("maxdepth must be nonnegative or None")
        self.mincriterion = mincriterion
        self.minsplit = minsplit
        self.minbucket = minbucket
        self.maxdepth = maxdepth
        self.seed = seed
        self.rscript = rscript or _find_rscript()
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._model_path: Path | None = None
        self._feature_count: int | None = None

    @staticmethod
    def _write_training_csv(
        path: Path, x: torch.Tensor, L: torch.Tensor, R: torch.Tensor
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([f"x_{index}" for index in range(x.shape[1])] + ["L", "R"])
            for features, left, right in zip(
                x.detach().cpu().tolist(),
                L.detach().cpu().tolist(),
                R.detach().cpu().tolist(),
            ):
                writer.writerow([*features, left, right])

    @staticmethod
    def _write_features_csv(path: Path, x: torch.Tensor) -> None:
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([f"x_{index}" for index in range(x.shape[1])])
            writer.writerows(x.detach().cpu().tolist())

    def _run_backend(self, *arguments: str) -> None:
        if self.rscript is None:
            raise RuntimeError(
                "Rscript was not found. Install R plus partykit, icenReg, and survival."
            )
        process = subprocess.run(
            [self.rscript, "--vanilla", str(Path(__file__).with_name("backend.R")), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip()
            raise RuntimeError(f"ctree R backend failed: {detail}")

    def fit(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        *,
        censor_type: str,
    ) -> "CTree":
        if censor_type not in self.supported_censor_types:
            raise ValueError(f"unsupported censor type: {censor_type}")
        if (
            x.ndim != 2
            or L.ndim != 1
            or R.ndim != 1
            or x.shape[0] != L.numel()
            or L.shape != R.shape
        ):
            raise ValueError("x must be 2-D and bounds must have one value per row")
        if not bool((L >= 0).all()) or not bool((R >= L).all()):
            raise ValueError("ctree requires 0 <= L <= R")

        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="ctree_")
        directory = Path(self._temporary_directory.name)
        training_path = directory / "training.csv"
        self._model_path = directory / "model.rds"
        self._feature_count = x.shape[1]
        self._write_training_csv(training_path, x, L, R)
        self._run_backend(
            "fit",
            str(training_path),
            str(self._model_path),
            str(self.mincriterion),
            str(self.minsplit),
            str(self.minbucket),
            "Inf" if self.maxdepth is None else str(self.maxdepth),
            str(self.seed),
        )
        return self

    def predict_survival(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if self._model_path is None or self._temporary_directory is None:
            raise RuntimeError("fit must be called before predict_survival")
        if x.ndim != 2 or x.shape[1] != self._feature_count:
            raise ValueError("x has the wrong number of features")
        if times.ndim != 1 or not bool((times[1:] >= times[:-1]).all()):
            raise ValueError("times must be a sorted one-dimensional tensor")

        directory = Path(self._temporary_directory.name)
        features_path = directory / "prediction.csv"
        grid_path = directory / "grid.csv"
        output_path = directory / "survival.csv"
        self._write_features_csv(features_path, x)
        with grid_path.open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(times.detach().cpu().tolist())
        self._run_backend(
            "predict",
            str(self._model_path),
            str(features_path),
            str(grid_path),
            str(output_path),
        )
        with output_path.open(newline="", encoding="utf-8") as source:
            rows = [[float(value) for value in row] for row in csv.reader(source)]
        return torch.tensor(rows, device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
            self._model_path = None

    def __enter__(self) -> "CTree":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
