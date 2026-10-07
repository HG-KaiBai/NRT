"""Subprocess adapter around the ICcforest package published with the paper."""

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
    """Return whether Rscript and the three required R packages are available."""
    executable = _find_rscript()
    if executable is None:
        return False
    probe = subprocess.run(
        [
            executable,
            "--vanilla",
            "-e",
            "packages <- c('ICcforest','icenReg','survival'); "
            "quit(status=ifelse(all(sapply(packages, requireNamespace, quietly=TRUE)),0,1))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


class ICCForest:
    """Conditional inference forest for interval-censored survival outcomes.

    Fitting and prediction use the paper authors' CRAN ``ICcforest`` package.
    The R model is kept in a temporary RDS file for the life of this object.
    """

    supported_censor_types = frozenset({"right", "interval", "current"})

    def __init__(
        self,
        *,
        ntree: int = 100,
        tune_mtry: bool = True,
        tune_ntree: int = 100,
        step_factor: float = 1.5,
        inf_replacement: float = 9_999_999.0,
        seed: int = 123,
        rscript: str | None = None,
    ) -> None:
        if ntree < 1 or tune_ntree < 1:
            raise ValueError("ntree and tune_ntree must be positive")
        if step_factor <= 1.0:
            raise ValueError("step_factor must be greater than one")
        self.ntree = ntree
        self.tune_mtry = tune_mtry
        self.tune_ntree = tune_ntree
        self.step_factor = step_factor
        self.inf_replacement = inf_replacement
        self.seed = seed
        self.rscript = rscript or _find_rscript()
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._model_path: Path | None = None
        self._feature_count: int | None = None

    @staticmethod
    def _write_training_csv(
        path: Path, x: torch.Tensor, L: torch.Tensor, R: torch.Tensor
    ) -> None:
        x_values = x.detach().cpu().tolist()
        left_values = L.detach().cpu().tolist()
        right_values = R.detach().cpu().tolist()
        fieldnames = [f"x_{index}" for index in range(x.shape[1])] + ["L", "R"]
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(fieldnames)
            for features, left, right in zip(x_values, left_values, right_values):
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
                "Rscript was not found. Install R plus ICcforest, icenReg, and survival."
            )
        backend = Path(__file__).with_name("backend.R")
        process = subprocess.run(
            [self.rscript, "--vanilla", str(backend), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip()
            raise RuntimeError(f"ICcforest R backend failed: {detail}")

    def fit(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        *,
        censor_type: str,
    ) -> "ICCForest":
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
            raise ValueError("ICcforest requires 0 <= L <= R")

        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="iccforest_")
        directory = Path(self._temporary_directory.name)
        training_path = directory / "training.csv"
        self._model_path = directory / "model.rds"
        self._feature_count = x.shape[1]
        self._write_training_csv(training_path, x, L, R)
        self._run_backend(
            "fit",
            str(training_path),
            str(self._model_path),
            str(self.ntree),
            "TRUE" if self.tune_mtry else "FALSE",
            str(self.tune_ntree),
            str(self.step_factor),
            str(self.inf_replacement),
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
        rows: list[list[float]] = []
        with output_path.open(newline="", encoding="utf-8") as source:
            rows.extend([float(value) for value in row] for row in csv.reader(source))
        return torch.tensor(rows, device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
            self._model_path = None

    def __enter__(self) -> "ICCForest":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
