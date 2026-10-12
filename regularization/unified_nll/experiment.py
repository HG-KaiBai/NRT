"""Paired repeated-fit experiments with resumable, independently scored results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from .model import ModelConfig
from .simulation import (SCENARIOS, censoring_horizon, data_reg, data_reg_multimodal,
                         multimodal_survival, survival_curves)
from .train import TrainingConfig, evaluate_nll, fit


METHODS = ("fixed_nll", "learned_nll", "regularized_nll")


@dataclass(frozen=True)
class ExperimentConfig:
    scenarios: tuple[str, ...] = SCENARIOS
    censor_types: tuple[str, ...] = ("exact", "right", "interval", "current")
    censor_qs: tuple[float, ...] = (0.2, 0.5)
    methods: tuple[str, ...] = METHODS
    seeds: tuple[int, ...] = tuple(range(10))
    data_seed: int = 123
    resample_data: bool = False
    n_samples: int = 5000
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    grid_size: int = 256
    followup_fraction: float = 0.9
    multimodal_case: str = "separate"


@dataclass(frozen=True)
class Setting:
    scenario: str
    censor_type: str
    q: float

    @property
    def name(self) -> str:
        return f"{self.scenario}_{self.censor_type}_q{self.q:g}"


def settings(config: ExperimentConfig) -> list[Setting]:
    if not config.scenarios or any(s not in SCENARIOS for s in config.scenarios):
        raise ValueError(f"scenarios must be a nonempty subset of {SCENARIOS}")
    if not config.censor_types or any(c not in ("exact", "right", "interval", "current")
                                      for c in config.censor_types):
        raise ValueError("unsupported or empty censor_types")
    if not config.censor_qs or any(not 0 < q < 1 for q in config.censor_qs):
        raise ValueError("censor_qs must be nonempty and strictly between zero and one")
    if not config.methods or any(method not in METHODS for method in config.methods):
        raise ValueError(f"methods must be a nonempty subset of {METHODS}")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be a nonempty sequence of distinct seeds")
    if any(len(set(values)) != len(values) for values in
           (config.scenarios, config.censor_types, config.censor_qs, config.methods)):
        raise ValueError("duplicate experiment settings are not allowed")
    if config.n_samples < 30 or config.grid_size < 3:
        raise ValueError("at least 30 samples and 3 grid points are required")
    if not (0 < config.train_fraction < 1 and 0 < config.validation_fraction < 1
            and config.train_fraction + config.validation_fraction < 1):
        raise ValueError("train/validation fractions must leave a nonempty test split")
    if not 0 < config.followup_fraction < 1:
        raise ValueError("followup_fraction must be in (0, 1)")
    return [Setting(scenario, censor_type, q)
            for scenario in config.scenarios for censor_type in config.censor_types
            for q in ((0.0,) if censor_type == "exact" else config.censor_qs)]


def _data(setting: Setting, config: ExperimentConfig, model_config: ModelConfig, seed: int):
    # Using the same data seed also pairs latent events/covariates across
    # censoring schemes. Each method sees identical training/validation/test data.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        kwargs = dict(n=config.n_samples, x_dim=model_config.x_dim,
                      q=setting.q or 0.2, censor_type=setting.censor_type, device="cpu")
        if setting.scenario == "multimodal":
            raw = data_reg_multimodal(**kwargs, multimodal_case=config.multimodal_case)
        else:
            raw = data_reg(**kwargs, scenario=setting.scenario)
    x, left, right = raw[:3]
    n_train = int(len(x) * config.train_fraction)
    n_validation = int(len(x) * config.validation_fraction)
    train = (x[:n_train], left[:n_train], right[:n_train])
    valid = tuple(v[n_train:n_train + n_validation] for v in (x, left, right))
    test = tuple(v[n_train + n_validation:] for v in (x, left, right))
    positive = torch.cat((train[1], train[2]))
    positive = positive[torch.isfinite(positive) & (positive > 0)]
    time_scale = float(positive.median()) if len(positive) else 1.0
    # Evaluation horizons are set by the simulation design, not test outcomes.
    # The full horizon is common across censoring rates within a scenario.
    full_horizon = censoring_horizon(setting.scenario, 0.2,
                                     multimodal_case=config.multimodal_case)
    followup = (float("inf") if setting.censor_type == "exact" else censoring_horizon(
        setting.scenario, setting.q, multimodal_case=config.multimodal_case))
    identified_horizon = min(full_horizon, config.followup_fraction * followup)
    grid = torch.linspace(0, identified_horizon, config.grid_size)
    full_grid = torch.linspace(0, full_horizon, config.grid_size)
    offset = n_train + n_validation

    def truth(times):
        if setting.scenario == "multimodal":
            return multimodal_survival(raw[4][offset:], raw[5][offset:], times)
        return survival_curves(raw[4][offset:], raw[5][offset:], times,
                               scenario=setting.scenario)

    return dict(train=train, valid=valid, test=test, time_scale=time_scale,
                grid=grid, full_grid=full_grid, truth=truth(grid), full_truth=truth(full_grid),
                identified_horizon=identified_horizon, full_horizon=full_horizon,
                followup_horizon=followup,
                observed_right_censor_fraction=float(torch.isinf(right).float().mean()))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def summarize(runs: pd.DataFrame, expected_repeats: int | None = None) -> pd.DataFrame:
    """Average errors of separately fitted models; never score the averaged CDF."""
    group = ["scenario", "censor_type", "q", "setting", "method"]
    result = runs.groupby(group, sort=False).agg(
        n_runs=("seed", "size"), cdf_mse_mean=("cdf_mse", "mean"),
        cdf_mse_std=("cdf_mse", "std"), cdf_mse_min=("cdf_mse", "min"),
        cdf_mse_max=("cdf_mse", "max"), full_cdf_mse_mean=("full_cdf_mse", "mean"),
        full_cdf_mse_std=("full_cdf_mse", "std"), test_nll_mean=("test_nll", "mean"),
        test_nll_std=("test_nll", "std"), fit_seconds_mean=("fit_seconds", "mean"),
        epochs_mean=("epochs_run", "mean"), best_epoch_mean=("best_epoch", "mean"),
        identified_horizon=("identified_horizon", "first"),
        full_horizon=("full_horizon", "first"),
    ).reset_index()
    result["cdf_mse_se"] = result.cdf_mse_std / np.sqrt(result.n_runs)
    if expected_repeats is not None and not result.n_runs.eq(expected_repeats).all():
        raise ValueError("incomplete repeats; inspect runs.csv before comparing methods")
    return result


def paired_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    """Paired seed differences: a negative difference favors the candidate."""
    index = ["scenario", "censor_type", "q", "setting", "seed", "data_seed"]
    pivot = runs.pivot(index=index, columns="method", values="cdf_mse")
    rows = []
    for reference, candidate in (("fixed_nll", "learned_nll"),
                                 ("learned_nll", "regularized_nll"),
                                 ("fixed_nll", "regularized_nll")):
        if reference not in pivot or candidate not in pivot:
            continue
        delta = (pivot[candidate] - pivot[reference]).rename("delta").reset_index()
        for key, group in delta.groupby(index[:4], sort=False):
            rows.append(dict(zip(index[:4], key), reference=reference, candidate=candidate,
                             n_pairs=len(group), mean_difference=group.delta.mean(),
                             std_difference=group.delta.std(ddof=1),
                             candidate_win_fraction=float((group.delta < 0).mean())))
    return pd.DataFrame(rows)


def run_experiments(output_dir: str | Path, *,
                    config: ExperimentConfig = ExperimentConfig(),
                    model_config: ModelConfig = ModelConfig(),
                    training_config: TrainingConfig = TrainingConfig(),
                    device: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Execute (or resume) all settings, methods, and paired initialization seeds."""
    all_settings = settings(config)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "curves").mkdir(exist_ok=True)
    (output / "histories").mkdir(exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    source_hash = hashlib.sha256()
    for name in ("tree.py", "model.py", "train.py", "simulation.py", "experiment.py"):
        source_hash.update((Path(__file__).parent / name).read_bytes())
    spec = dict(experiment=asdict(config), model=asdict(model_config),
                training=asdict(training_config), device=device, source_sha256=source_hash.hexdigest())
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "config.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("cached results use different code/configuration; choose a new output_dir")
    else:
        manifest = dict(spec, fingerprint=fingerprint, python=platform.python_version(),
                        torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                        gpu=torch.cuda.get_device_name(0) if device.startswith("cuda") else None)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    csv_path = output / "runs.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    completed = {(row["setting"], row["method"], int(row["seed"])) for row in rows}
    total = len(all_settings) * len(config.methods) * len(config.seeds)
    for setting in all_settings:
        data_cache = {}
        for repeat_index, seed in enumerate(config.seeds):
            data_seed = config.data_seed + (repeat_index if config.resample_data else 0)
            needed = [method for method in config.methods
                      if (setting.name, method, seed) not in completed]
            if not needed:
                continue
            if data_seed not in data_cache:
                data_cache[data_seed] = _data(setting, config, model_config, data_seed)
            data = data_cache[data_seed]
            for method in needed:
                actual_model = replace(model_config, learn_splits=method != "fixed_nll")
                actual_training = replace(training_config,
                    regularization_strength=(training_config.regularization_strength
                                             if method == "regularized_nll" else 0.0))
                started = time.perf_counter()
                model, history = fit(*data["train"], validation=data["valid"],
                    model_config=actual_model, training_config=actual_training,
                    time_scale=data["time_scale"], device=device, seed=seed)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                fit_seconds = time.perf_counter() - started
                test_x, test_left, test_right = data["test"]
                predicted = model.predict_survival(test_x, data["grid"]).cpu()
                full_predicted = model.predict_survival(test_x, data["full_grid"]).cpu()
                per_subject = torch.trapezoid((predicted - data["truth"]).square(),
                    data["grid"], dim=-1) / data["identified_horizon"]
                full_per_subject = torch.trapezoid((full_predicted - data["full_truth"]).square(),
                    data["full_grid"], dim=-1) / data["full_horizon"]
                with torch.no_grad():
                    state = model(test_x.to(model.time_scale))
                    min_width = float(state.width.min())
                    regularizer = float(model.tree.regularizer(state).mean())
                run_name = f"{setting.name}_{method}_seed{seed}"
                curves_path = output / "curves" / f"{run_name}.npz"
                sample_indices = np.arange(min(3, len(test_x)))
                np.savez_compressed(curves_path, grid=data["full_grid"].numpy(),
                    truth_cdf=(1 - data["full_truth"][sample_indices]).numpy(),
                    predicted_cdf=(1 - full_predicted[sample_indices]).numpy(),
                    subject_indices=sample_indices, x=test_x[sample_indices].numpy())
                with gzip.open(output / "histories" / f"{run_name}.json.gz", "wt",
                               encoding="utf-8") as stream:
                    json.dump(history, stream)
                row = dict(scenario=setting.scenario, censor_type=setting.censor_type,
                    q=setting.q, setting=setting.name, method=method, seed=seed,
                    data_seed=data_seed, cdf_mse=float(per_subject.mean()),
                    cdf_ise=float(per_subject.mean()) * data["identified_horizon"],
                    full_cdf_mse=float(full_per_subject.mean()),
                    test_nll=evaluate_nll(model, test_x, test_left, test_right),
                    fit_seconds=fit_seconds, min_leaf_width=min_width,
                    regularizer=regularizer, **model.fit_info,
                    identified_horizon=data["identified_horizon"],
                    full_horizon=data["full_horizon"], followup_horizon=data["followup_horizon"],
                    observed_right_censor_fraction=data["observed_right_censor_fraction"],
                    curves_file=str(curves_path.relative_to(output)))
                metrics = (row["cdf_mse"], row["full_cdf_mse"], row["test_nll"])
                if not np.isfinite(metrics).all():
                    raise RuntimeError(f"nonfinite evaluation metrics: {run_name}")
                rows.append(row)
                completed.add((setting.name, method, seed))
                _write_csv(pd.DataFrame(rows), csv_path)
                if verbose:
                    print(f"[{len(rows):3d}/{total}] {run_name}: "
                          f"CDF MSE={row['cdf_mse']:.6f}, "
                          f"val NLL={row['validation_nll']:.4f}, "
                          f"best epoch={row['best_epoch']}, {fit_seconds:.1f}s", flush=True)
                del model
        if verbose:
            print(f"Completed setting: {setting.name}", flush=True)
    runs = pd.DataFrame(rows)
    if len(runs) != total:
        raise RuntimeError(f"expected {total} fits but found {len(runs)}")
    _write_csv(summarize(runs, len(config.seeds)), output / "summary.csv")
    _write_csv(paired_comparisons(runs), output / "paired_comparisons.csv")
    return runs
