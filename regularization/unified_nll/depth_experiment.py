"""One global regularizer across observation schemes; depths 3, 6 and 8.

Global strength selection uses an independent pilot dataset and validation NLL
only. Test CDFs are not constructed or scored in that phase. The final ten-seed
comparison reports individual-model errors, including depth selection by NLL.
"""

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

from .experiment import ExperimentConfig, _write_csv, settings
from .metrics import average_grid_sse
from .model import ModelConfig
from .regularization import depth_model_config, fit_regularized, partition_penalty
from .train import TrainingConfig, evaluate_nll


@dataclass(frozen=True)
class DepthConfig:
    depths: tuple[int, ...] = (3, 6, 8)
    strengths: tuple[float, ...] = (0.0, 0.5, 5.0, 50.0, 500.0)
    total_width_floor: float = 8e-4
    pilot_data_seed: int = 2026
    pilot_fit_seed: int = 101


def _validate(depth_config: DepthConfig):
    if (not depth_config.depths or len(set(depth_config.depths)) != len(depth_config.depths)
            or any(isinstance(d, bool) or not isinstance(d, int) or d < 1
                   for d in depth_config.depths)):
        raise ValueError("depths must be distinct positive integers")
    if (not depth_config.strengths
            or len(set(depth_config.strengths)) != len(depth_config.strengths)
            or any(not np.isfinite(a) or a < 0 for a in depth_config.strengths)):
        raise ValueError("strengths must be distinct, finite and nonnegative")
    if not 0 < depth_config.total_width_floor < 1:
        raise ValueError("total_width_floor must be in (0,1)")


def _manifest(output, experiment, depth_config, model_config, training_config, device):
    output.mkdir(parents=True, exist_ok=True)
    source = hashlib.sha256()
    for name in ("tree.py", "model.py", "train.py", "simulation.py", "experiment.py",
                 "regularization.py", "depth_experiment.py", "metrics.py"):
        source.update((Path(__file__).parent / name).read_bytes())
    data_config = {k:v for k,v in asdict(experiment).items() if k not in ("methods","followup_fraction")}
    spec = dict(experiment=data_config, comparison_methods=["fixed_nll","learned_nll","partition_nll"],
                depth=asdict(depth_config),
                model=asdict(model_config), training=asdict(training_config),
                device=device, source_sha256=source.hexdigest())
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = output / "config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("cached results use different code/configuration; choose a new output_dir")
    else:
        path.write_text(json.dumps(dict(spec, fingerprint=fingerprint,
            python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            pandas=pd.__version__, gpu=torch.cuda.get_device_name(0)
            if device.startswith("cuda") else None), indent=2), encoding="utf-8")
    return fingerprint


def _generate(setting, config, model_config, seed):
    from .simulation import data_reg, data_reg_multimodal
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        kwargs = dict(n=config.n_samples, x_dim=model_config.x_dim,
                      q=setting.q or .2, censor_type=setting.censor_type, device="cpu")
        raw = (data_reg_multimodal(**kwargs, multimodal_case=config.multimodal_case)
               if setting.scenario == "multimodal" else data_reg(**kwargs, scenario=setting.scenario))
    return raw


def _observed_data(setting, config, model_config, seed):
    """Only observed x,L,R enter selection; no truth CDF is computed here."""
    observed = _generate(setting, config, model_config, seed)[:3]
    n_train = int(config.n_samples * config.train_fraction)
    n_valid = int(config.n_samples * config.validation_fraction)
    train = tuple(v[:n_train] for v in observed)
    valid = tuple(v[n_train:n_train + n_valid] for v in observed)
    bounds = torch.cat((train[1], train[2]))
    positive = bounds[torch.isfinite(bounds) & (bounds > 0)]
    scale = float(positive.median()) if len(positive) else 1.0
    return train, valid, scale


def _evaluation_data(setting, config, model_config, seed):
    """Match the original 256-point grid, 0 to maximum finite first-80% bound.

    The first 80% fitting pool is split internally into 60% fitting and 20%
    validation. The remaining 20% is the original test split. All methods and
    depths for a setting use the same observed-bound grid and test subjects.
    """
    from .simulation import censoring_horizon, multimodal_survival, survival_curves
    raw = _generate(setting, config, model_config, seed)
    observed = raw[:3]
    n_train = int(config.n_samples*config.train_fraction)
    offset = n_train + int(config.n_samples*config.validation_fraction)
    train = tuple(v[:n_train] for v in observed)
    valid = tuple(v[n_train:offset] for v in observed)
    test = tuple(v[offset:] for v in observed)
    train_bounds = torch.cat((train[1],train[2]))
    positive = train_bounds[torch.isfinite(train_bounds) & (train_bounds>0)]
    time_scale = float(positive.median()) if len(positive) else 1.0
    pool_bounds = torch.cat((observed[1][:offset], observed[2][:offset]))
    max_obs = float(pool_bounds[torch.isfinite(pool_bounds)].max())
    if not max_obs > 0:
        raise ValueError("evaluation needs at least one positive finite fitting-pool bound")
    grid = torch.linspace(0,max_obs,config.grid_size)
    if setting.scenario == "multimodal":
        truth = multimodal_survival(raw[4][offset:],raw[5][offset:],grid)
    else:
        truth = survival_curves(raw[4][offset:],raw[5][offset:],grid,scenario=setting.scenario)
    followup = (float("inf") if setting.censor_type=="exact" else
                censoring_horizon(setting.scenario,setting.q,multimodal_case=config.multimodal_case))
    return dict(train=train,valid=valid,test=test,time_scale=time_scale,grid=grid,truth=truth,
                max_obs=max_obs,followup_horizon=followup,
                observed_right_censor_fraction=float(torch.isinf(observed[2]).float().mean()))


def tune_global_strength(output_dir, *, config=ExperimentConfig(), depth_config=DepthConfig(),
                         model_config=ModelConfig(), training_config=TrainingConfig(),
                         device=None, verbose=True):
    """Choose ONE a for ALL depths, censoring schemes, rates and distributions.

    Equal weight per setting and depth, equal validation-set sizes. The global
    score averages validation observed NLL across the independent pilot fits.
    Zero is a legitimate candidate; no improvement is forced by the protocol.
    """
    _validate(depth_config)
    all_settings = settings(config)
    if depth_config.pilot_data_seed == config.data_seed:
        raise ValueError("pilot_data_seed must differ from the final comparison data seed")
    if depth_config.pilot_fit_seed in config.seeds:
        raise ValueError("pilot_fit_seed must differ from final initialization seeds")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output_dir).resolve()
    fingerprint = _manifest(output, config, depth_config, model_config, training_config, device)
    path = output / "tuning_runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["setting"], int(r["depth"]), float(r["strength"])) for r in rows}
    total = len(all_settings) * len(depth_config.depths) * len(depth_config.strengths)
    for setting in all_settings:
        train, valid, scale = _observed_data(setting, config, model_config, depth_config.pilot_data_seed)
        for depth in depth_config.depths:
            actual_model = depth_model_config(model_config, depth, depth_config.total_width_floor)
            actual_model = replace(actual_model, learn_splits=True)
            for strength in depth_config.strengths:
                if (setting.name, depth, strength) in done:
                    continue
                started = time.perf_counter()
                model, _ = fit_regularized(*train, validation=valid, model_config=actual_model,
                    training_config=replace(training_config, regularization_strength=strength),
                    penalty="partition_kl", time_scale=scale, device=device,
                    seed=depth_config.pilot_fit_seed)
                row = dict(setting=setting.name, scenario=setting.scenario,
                    censor_type=setting.censor_type, q=setting.q, depth=depth, strength=strength,
                    fit_seconds=time.perf_counter()-started, **model.fit_info)
                rows.append(row)
                done.add((setting.name, depth, strength))
                _write_csv(pd.DataFrame(rows), path)
                if verbose:
                    print(f"Pilot [{len(rows)}/{total}] {setting.name}, D={depth}, a={strength:g}, "
                          f"validation NLL={row['validation_nll']:.6f}", flush=True)
                del model
    runs = pd.DataFrame(rows)
    if len(runs) != total or not np.isfinite(runs.validation_nll).all():
        raise RuntimeError("incomplete or nonfinite global strength search")
    scores = runs.groupby("strength").agg(
        validation_nll_mean=("validation_nll", "mean"), n_fits=("setting", "size")
    ).reset_index().sort_values(["validation_nll_mean", "strength"])
    selected = float(scores.iloc[0].strength)
    scores["selected"] = scores.strength.eq(selected)
    _write_csv(scores, output / "tuning_summary.csv")
    (output / "selected_strength.json").write_text(json.dumps(dict(strength=selected,
        criterion="mean pilot validation observed-data NLL; one common strength for all settings/depths",
        fingerprint=fingerprint), indent=2), encoding="utf-8")
    return selected, scores


def depth_summary(runs, repeats):
    group = ["scenario", "censor_type", "q", "setting", "depth", "method"]
    frame = runs.groupby(group, sort=False).agg(n_runs=("seed", "size"),
        average_grid_sse_mean=("average_grid_sse", "mean"),
        average_grid_sse_std=("average_grid_sse", "std"),
        test_nll_mean=("test_nll", "mean"), validation_nll_mean=("validation_nll", "mean"),
        min_leaf_width_mean=("min_leaf_width", "mean"),
        partition_penalty_mean=("partition_penalty", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"), best_epoch_mean=("best_epoch", "mean"),
        epochs_mean=("epochs_run", "mean")).reset_index()
    if not frame.n_runs.eq(repeats).all():
        raise RuntimeError("incomplete repeated fits")
    return frame


def run_depth_comparison(output_dir, *, config=ExperimentConfig(), depth_config=DepthConfig(),
                         model_config=ModelConfig(), training_config=TrainingConfig(),
                         device=None, verbose=True):
    _validate(depth_config)
    all_settings = settings(config)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output_dir).resolve()
    fingerprint = _manifest(output, config, depth_config, model_config, training_config, device)
    selected_path = output / "selected_strength.json"
    if not selected_path.exists():
        raise ValueError("run tune_global_strength before the ten-seed comparison")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["fingerprint"] != fingerprint:
        raise ValueError("global strength selection used different code/configuration")
    strength = float(selected["strength"])
    methods = ("fixed_nll", "learned_nll", "partition_nll")
    for name in ("curves", "histories"):
        (output / name).mkdir(exist_ok=True)
    path = output / "runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["setting"], int(r["depth"]), r["method"], int(r["seed"])) for r in rows}
    total = len(all_settings) * len(depth_config.depths) * len(methods) * len(config.seeds)
    for setting in all_settings:
        cache = {}
        for repeat_index, seed in enumerate(config.seeds):
            data_seed = config.data_seed + (repeat_index if config.resample_data else 0)
            if data_seed not in cache:
                cache[data_seed] = _evaluation_data(setting, config, model_config, data_seed)
            data = cache[data_seed]
            for depth in depth_config.depths:
                for method in methods:
                    key = (setting.name, depth, method, seed)
                    if key in done:
                        continue
                    actual_model = depth_model_config(model_config, depth, depth_config.total_width_floor)
                    actual_model = replace(actual_model, learn_splits=method != "fixed_nll")
                    actual_training = replace(training_config,
                        regularization_strength=strength if method == "partition_nll" else 0.0)
                    started = time.perf_counter()
                    model, history = fit_regularized(*data["train"], validation=data["valid"],
                        model_config=actual_model, training_config=actual_training,
                        penalty="partition_kl" if method == "partition_nll" else "none",
                        time_scale=data["time_scale"], device=device, seed=seed)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    test_x, test_left, test_right = data["test"]
                    predicted = model.predict_survival(test_x, data["grid"]).cpu()
                    error = float(average_grid_sse(predicted, data["truth"]))
                    with torch.no_grad():
                        state = model(test_x.to(model.time_scale))
                        width = float(state.width.min())
                        geometry = float(partition_penalty(state, depth).mean())
                        density_kl = float(model.tree.regularizer(state).mean())
                    run_name = f"{setting.name}_depth{depth}_{method}_seed{seed}"
                    curves_path = output / "curves" / f"{run_name}.npz"
                    indices = np.arange(min(3, len(test_x)))
                    np.savez_compressed(curves_path, grid=data["grid"].numpy(),
                        truth_cdf=(1-data["truth"][indices]).numpy(),
                        predicted_cdf=(1-predicted[indices]).numpy(),
                        subject_indices=indices, x=test_x[indices].numpy())
                    with gzip.open(output / "histories" / f"{run_name}.json.gz", "wt",
                                   encoding="utf-8") as stream:
                        json.dump(history, stream)
                    row = dict(scenario=setting.scenario, censor_type=setting.censor_type,
                        q=setting.q, setting=setting.name, depth=depth, n_leaves=2**depth,
                        method=method, seed=seed, data_seed=data_seed, average_grid_sse=error,
                        test_nll=evaluate_nll(model,test_x,test_left,test_right),
                        fit_seconds=seconds, min_leaf_width=width,
                        leaf_width_floor=actual_model.min_leaf_width,
                        partition_penalty=geometry, density_kl=density_kl,
                        strength=actual_training.regularization_strength, **model.fit_info,
                        grid_size=config.grid_size,grid_horizon=data["max_obs"],
                        followup_horizon=data["followup_horizon"],
                        observed_right_censor_fraction=data["observed_right_censor_fraction"],
                        curves_file=str(curves_path.relative_to(output)))
                    if not np.isfinite([error, row["test_nll"]]).all():
                        raise RuntimeError(f"nonfinite evaluation for {run_name}")
                    rows.append(row)
                    done.add(key)
                    _write_csv(pd.DataFrame(rows), path)
                    if verbose:
                        print(f"Final [{len(rows)}/{total}] {run_name}: grid SSE={error:.6f}, "
                              f"validation NLL={row['validation_nll']:.5f}, {seconds:.1f}s", flush=True)
                    del model
        if verbose:
            print(f"Completed setting: {setting.name}", flush=True)
    runs = pd.DataFrame(rows)
    if len(runs) != total:
        raise RuntimeError(f"expected {total} fits, found {len(runs)}")
    _write_csv(depth_summary(runs, len(config.seeds)), output / "summary.csv")
    # Each setting/seed/method chooses its depth using validation NLL only.
    # Sort depth first so exact ties prefer the shallower candidate.
    indices = (runs.sort_values("depth").groupby(["setting", "seed", "method"])
               .validation_nll.idxmin())
    chosen = runs.loc[indices].copy()
    _write_csv(chosen, output / "validation_selected_depth_runs.csv")
    selected_summary = chosen.groupby(["scenario","censor_type","q","setting","method"]).agg(
        n_runs=("seed","size"), average_grid_sse_mean=("average_grid_sse","mean"),
        average_grid_sse_std=("average_grid_sse","std"), test_nll_mean=("test_nll","mean"),
        mean_depth=("depth","mean")).reset_index()
    if not selected_summary.n_runs.eq(len(config.seeds)).all():
        raise RuntimeError("incomplete validation-selected depth fits")
    _write_csv(selected_summary, output / "validation_selected_depth_summary.csv")
    pairs = []
    for depth in depth_config.depths:
        wide = runs[runs.depth.eq(depth)].pivot(index=["setting","seed"],columns="method",values="average_grid_sse")
        for reference,candidate in (("fixed_nll","learned_nll"),("learned_nll","partition_nll"),
                                    ("fixed_nll","partition_nll")):
            delta = (wide[candidate]-wide[reference]).rename("delta").reset_index()
            for name,group in delta.groupby("setting",sort=False):
                pairs.append(dict(setting=name,depth=depth,reference=reference,candidate=candidate,
                    n_pairs=len(group),mean_difference=group.delta.mean(),
                    std_difference=group.delta.std(ddof=1),
                    candidate_win_fraction=float((group.delta<0).mean())))
    _write_csv(pd.DataFrame(pairs), output / "paired_comparisons.csv")
    return runs
