"""Python adapter for the R Cox proportional-hazards benchmarks."""

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
    """Return whether Rscript, survival, and icenReg are available."""
    executable = _find_rscript()
    if executable is None:
        return False
    probe = subprocess.run(
        [
            executable,
            "--vanilla",
            "-e",
            "packages <- c('survival','icenReg'); "
            "quit(status=ifelse(all(sapply(packages, requireNamespace, quietly=TRUE)),0,1))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


class CoxPH:
    """Linear Cox PH model fitted by R for every observation scheme.

    ``survival::coxph`` handles exact and right-censored observations.
    ``icenReg::ic_sp(model='ph')`` handles interval- and current-status
    observations. Both paths expose survival curves through the same adapter.
    """

    supported_censor_types = frozenset({"exact", "right", "interval", "current"})

    def __init__(self, *, rscript: str | None = None) -> None:
        self.rscript = rscript or _find_rscript()
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._model_path: Path | None = None
        self._feature_count: int | None = None

    @staticmethod
    def _validate_training_data(
        x: torch.Tensor, L: torch.Tensor, R: torch.Tensor
    ) -> None:
        if (
            x.ndim != 2
            or L.ndim != 1
            or R.ndim != 1
            or x.shape[0] != L.numel()
            or L.shape != R.shape
        ):
            raise ValueError("x must be 2-D and bounds must have one value per row")
        if not bool((L >= 0).all()) or not bool((R >= L).all()):
            raise ValueError("CoxPH requires 0 <= L <= R")

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
                "Rscript was not found. Install R plus survival and icenReg."
            )
        process = subprocess.run(
            [
                self.rscript,
                "--vanilla",
                str(Path(__file__).with_name("backend.R")),
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip()
            raise RuntimeError(f"Cox R backend failed: {detail}")

    def fit(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        *,
        censor_type: str,
    ) -> "CoxPH":
        if censor_type not in self.supported_censor_types:
            raise ValueError(f"unsupported censor type: {censor_type}")
        self._validate_training_data(x, L, R)

        self.close()
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="cox_r_")
        directory = Path(self._temporary_directory.name)
        training_path = directory / "training.csv"
        self._model_path = directory / "model.rds"
        self._feature_count = x.shape[1]
        self._write_training_csv(training_path, x, L, R)
        self._run_backend(
            "fit", str(training_path), str(self._model_path), censor_type
        )
        return self

    def _predict(
        self, mode: str, x: torch.Tensor, times: torch.Tensor | None
    ) -> torch.Tensor:
        if self._model_path is None or self._temporary_directory is None:
            raise RuntimeError("fit must be called before prediction")
        if x.ndim != 2 or x.shape[1] != self._feature_count:
            raise ValueError("x has the wrong number of features")

        directory = Path(self._temporary_directory.name)
        features_path = directory / "prediction.csv"
        output_path = directory / f"{mode}.csv"
        self._write_features_csv(features_path, x)
        arguments = [mode, str(self._model_path), str(features_path)]
        if times is not None:
            grid_path = directory / "grid.csv"
            with grid_path.open("w", newline="", encoding="utf-8") as output:
                csv.writer(output).writerow(times.detach().cpu().tolist())
            arguments.append(str(grid_path))
        arguments.append(str(output_path))
        self._run_backend(*arguments)
        with output_path.open(newline="", encoding="utf-8") as source:
            rows = [[float(value) for value in row] for row in csv.reader(source)]
        return torch.tensor(rows, device=x.device, dtype=x.dtype)

    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:
        """Return the R model's fitted linear log-risk score."""
        return self._predict("risk", x, None).reshape(-1)

    def predict_survival(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if (
            times.ndim != 1
            or bool((times < 0).any())
            or not bool((times[1:] >= times[:-1]).all())
        ):
            raise ValueError("times must be a sorted nonnegative one-dimensional tensor")
        return self._predict("predict", x, times).clamp(0.0, 1.0)

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self._temporary_directory = None
        self._model_path = None
        self._feature_count = None

    def __enter__(self) -> "CoxPH":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
