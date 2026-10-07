"""Python adapter for the R parametric AFT benchmarks."""

from __future__ import annotations

import csv
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch


def distribution_for_scenario(scenario: str) -> str:
    """Select the event-time family corresponding to the simulation's AFT error."""
    families = {
        "ph": "exponential",  # Weibull shape one; minimum extreme-value log error.
        "exponential": "exponential",
        "weibull": "weibull",
        "non_ph": "weibull",
        "lognormal": "lognormal",  # Gaussian log error.
        "po": "loglogistic",  # Working model for bounded PO; no exact family match.
        "multimodal": "weibull",  # Deliberate single-family working model.
    }
    try:
        return families[scenario]
    except KeyError:
        raise ValueError(f"unknown simulation scenario: {scenario!r}") from None


def _find_rscript() -> str | None:
    executable = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if executable:
        return executable
    candidates = sorted(
        glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"), reverse=True
    )
    return candidates[0] if candidates else None


def r_backend_available() -> bool:
    """Return whether Rscript and the survival package are available."""
    executable = _find_rscript()
    if executable is None:
        return False
    probe = subprocess.run(
        [
            executable,
            "--vanilla",
            "-e",
            "quit(status=ifelse(requireNamespace('survival', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


class ParametricAFT:
    """Single-distribution AFT model fitted in R.

    Exponential, Weibull, lognormal, and log-logistic models use
    ``survival::survreg`` with an
    ``interval2`` response, so the same fit accepts exact, right-censored,
    interval-censored, and current-status observations. A Weibull fit remains
    single-component even when the benchmark data come from a mixture.
    """

    supported_censor_types = frozenset({"exact", "right", "interval", "current"})

    def __init__(
        self,
        *,
        distribution: str = "weibull",
        max_iter: int = 250,
        rscript: str | None = None,
    ) -> None:
        if distribution not in {"exponential", "weibull", "lognormal", "loglogistic"}:
            raise ValueError(
                "distribution must be 'exponential', 'weibull', 'lognormal', or 'loglogistic'"
            )
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.distribution = distribution
        self.max_iter = max_iter
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
            raise ValueError("ParametricAFT requires 0 <= L <= R")

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
            raise RuntimeError("Rscript was not found. Install R plus survival.")
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
            raise RuntimeError(f"AFT R backend failed: {detail}")

    def fit(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        *,
        censor_type: str,
    ) -> "ParametricAFT":
        if censor_type not in self.supported_censor_types:
            raise ValueError(f"unsupported censor type: {censor_type}")
        self._validate_training_data(x, L, R)

        self.close()
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="aft_r_")
        directory = Path(self._temporary_directory.name)
        training_path = directory / "training.csv"
        self._model_path = directory / "model.rds"
        self._feature_count = x.shape[1]
        self._write_training_csv(training_path, x, L, R)
        self._run_backend(
            "fit",
            str(training_path),
            str(self._model_path),
            self.distribution,
            str(self.max_iter),
        )
        return self

    def predict_survival(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if self._model_path is None or self._temporary_directory is None:
            raise RuntimeError("fit must be called before predict_survival")
        if x.ndim != 2 or x.shape[1] != self._feature_count:
            raise ValueError("x has the wrong number of features")
        if (
            times.ndim != 1
            or bool((times < 0).any())
            or not bool((times[1:] >= times[:-1]).all())
        ):
            raise ValueError("times must be a sorted nonnegative one-dimensional tensor")

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
        temporary_directory = getattr(self, "_temporary_directory", None)
        if temporary_directory is not None:
            temporary_directory.cleanup()
        self._temporary_directory = None
        self._model_path = None
        self._feature_count = None

    def __enter__(self) -> "ParametricAFT":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
