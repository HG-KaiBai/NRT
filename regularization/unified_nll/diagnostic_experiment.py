"""Factor-by-factor diagnostics on 35 settings, including both mixture cases.

The Brier-selected architecture remains the primary selection. Validation NLL
selection is an explicit control on the SAME Brier-proposed candidate pool.
All choices are frozen before the new ten-seed test evaluations.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .brier import score_brier
from .depth_experiment import _evaluation_data
from .experiment import ExperimentConfig, settings
from .metrics import average_grid_sse
from .model import ModelConfig
from .parameter_search import SearchConfig, _fit, _history, _observed_split, tune_per_setting
from .search_io import write_csv
from .stable_train import SplitOptimization, fit_stable
from .train import TrainingConfig, evaluate_nll
from .tuning_model import Architecture


METHODS = ("common_baseline", "brier_original", "nll_original", "brier_slowrho",
           "brier_warmup", "brier_stable", "brier_fixed", "nll_stable")


def groups(config):
    """Public close label maps to the original simulator's `closer` case."""
    for scenario in config.scenarios:
        if scenario == "multimodal":
            yield "multimodal_separate", replace(config, scenarios=(scenario,), multimodal_case="separate")
            yield "multimodal_close", replace(config, scenarios=(scenario,), multimodal_case="closer")
        else:
            yield scenario, replace(config, scenarios=(scenario,))


def expanded_settings(config):
    return [(f"{label}_{s.censor_type}_q{s.q:g}", label, s, actual)
            for label, actual in groups(config) for s in settings(actual)]


def architecture_from(row, baseline=Architecture()):
    return Architecture(**{key: int(row[key]) if isinstance(value, int) else float(row[key])
                          for key, value in asdict(baseline).items()})


def tune_close(output_dir, *, config=ExperimentConfig(), search=SearchConfig(),
               training=TrainingConfig(), baseline=Architecture(), device="cuda"):
    return tune_per_setting(Path(output_dir)/"close_search", config=replace(config,
        scenarios=("multimodal",), multimodal_case="closer"), search=search,
        training=training, baseline=baseline, device=device)


def _source(label, historical, output):
    return output/"close_search" if label == "multimodal_close" else historical


def _check_source(source, config, search, training, baseline, device):
    manifest = json.loads((source/"config.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for name in ("tree.py", "model.py", "train.py", "simulation.py", "experiment.py",
                 "regularization.py", "depth_experiment.py", "metrics.py", "tuning_model.py",
                 "tuning_train.py", "brier.py", "parameter_search.py", "search_io.py"):
        digest.update((Path(__file__).parent/name).read_bytes())
    if digest.hexdigest() != manifest["source_sha256"]:
        raise ValueError("source study no longer matches its original numerical code")
    actual = manifest["experiment"]
    for key, value in asdict(config).items():
        if key == "scenarios":
            if not set(value).issubset(actual[key]):
                raise ValueError("source does not contain requested scenarios")
        elif actual[key] != (list(value) if isinstance(value, tuple) else value):
            raise ValueError(f"source data configuration differs: {key}")
    for name, expected in (("search", search), ("training", training), ("baseline", baseline)):
        if manifest[name] != json.loads(json.dumps(asdict(expected))):
            raise ValueError(f"source {name} configuration differs")
    if manifest["device"] != device:
        raise ValueError("source device differs")


def _manifest(output, historical, config, search, training, baseline, device):
    digest = hashlib.sha256()
    for name in ("tree.py", "model.py", "train.py", "simulation.py", "experiment.py",
                 "regularization.py", "depth_experiment.py", "metrics.py", "tuning_model.py",
                 "tuning_train.py", "brier.py", "parameter_search.py", "search_io.py",
                 "stable_train.py", "diagnostic_experiment.py"):
        digest.update((Path(__file__).parent/name).read_bytes())
    inputs = {}
    for source in (historical, output/"close_search"):
        for name in ("config.json", "selection_complete.json", "trials.csv",
                     "confirmation_runs.csv", "best_parameters.csv"):
            inputs[str(source/name)] = hashlib.sha256((source/name).read_bytes()).hexdigest()
        if (source/"final_runs.csv").exists():
            inputs[str(source/"final_runs.csv")] = hashlib.sha256((source/"final_runs.csv").read_bytes()).hexdigest()
    spec = dict(experiment=asdict(config), search=asdict(search), training=asdict(training),
        baseline=asdict(baseline), device=device, methods=METHODS,
        stable_split_lr_ratio=.1, warmup_epochs=10, stable_patience=50,
        inputs=inputs, source_sha256=digest.hexdigest(),
        nll_control="reranking the same Brier-proposed pool; not independent NLL TPE",
        scope="exploratory on previously evaluated settings; close is newly added")
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = output/"config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("different code/configuration; use a new output directory")
    else:
        path.write_text(json.dumps(dict(spec, fingerprint=fingerprint), indent=2), encoding="utf-8")
    return fingerprint


def select_diagnostic_parameters(output_dir, historical_dir, *, config=ExperimentConfig(),
        search=SearchConfig(), training=TrainingConfig(), baseline=Architecture(), device="cuda"):
    """Confirm baseline + top-three NLL candidates with new seeds, reuse overlaps.

    Brier selection is kept exactly as in the source study. NLL uses the same
    confirmation seeds and shortlist size. No test CDF is constructed here.
    """
    output, historical = Path(output_dir).resolve(), Path(historical_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for label, actual in groups(config):
        _check_source(_source(label, historical, output), actual, search, training, baseline, device)
    fingerprint = _manifest(output, historical, config, search, training, baseline, device)
    path = output/"extra_confirmation_runs.csv"
    extras = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["setting"], int(r["trial"]), int(r["seed"])) for r in extras}
    selections = []
    for name, label, setting, actual in expanded_settings(config):
        source = _source(label, historical, output)
        trials = pd.read_csv(source/"trials.csv")
        trials = trials[trials.setting == setting.name]
        confirmed = pd.read_csv(source/"confirmation_runs.csv")
        confirmed = confirmed[confirmed.setting == setting.name].copy()
        confirmed["setting"] = name
        brier = pd.read_csv(source/"best_parameters.csv").set_index("setting").loc[setting.name]
        baseline_trial = next(int(r.trial) for _, r in trials.iterrows() if all(
            np.isclose(r[key], value, rtol=1e-10, atol=1e-10)
            for key, value in asdict(baseline).items()))
        ordered = trials.sort_values(["validation_nll", "validation_brier", "trial"])
        shortlist = [baseline_trial]+[int(t) for t in ordered.trial if int(t) != baseline_trial][:search.shortlist-1]
        data = _observed_split(setting, actual, baseline)
        for number in shortlist:
            architecture = architecture_from(trials.set_index("trial").loc[number], baseline)
            for seed in search.confirmation_seeds:
                key = name, number, seed
                if key in done or ((confirmed.trial == number)&(confirmed.seed == seed)).any():
                    continue
                model, history, info = _fit(data, architecture, training, seed, device)
                extras.append(dict(setting=name, trial=number, seed=seed, **asdict(architecture), **info))
                done.add(key)
                _history(output, f"{name}_nllconfirm{number}_seed{seed}", history)
                write_csv(pd.DataFrame(extras), path)
                del model
        available = pd.concat((confirmed, pd.DataFrame([r for r in extras if r["setting"] == name])), ignore_index=True)
        available = available[available.trial.isin(shortlist)]
        scores = available.groupby("trial").agg(selection_nll=("validation_nll", "mean"),
            selection_brier=("validation_brier", "mean"), n=("seed", "size"))
        if not scores.n.eq(len(search.confirmation_seeds)).all() or len(scores) != len(shortlist):
            raise RuntimeError("incomplete/duplicated NLL confirmation")
        number = int(scores.sort_values(["selection_nll", "selection_brier"]).index[0])
        nll = trials[trials.trial == number].iloc[0]
        for selector, row in (("brier", brier), ("nll", nll)):
            selections.append(dict(setting=name, family=label, censor_type=setting.censor_type,
                q=setting.q, selector=selector, trial=int(row.trial), **asdict(architecture_from(row, baseline)),
                confirmation_brier=float(brier.selection_brier) if selector == "brier" else float(scores.loc[number, "selection_brier"]),
                confirmation_nll=float(brier.confirmation_nll) if selector == "brier" else float(scores.loc[number, "selection_nll"]),
                source=str(source), max_obs=data["max_obs"]))
        write_csv(pd.DataFrame(selections), output/"selected_parameters.csv")
        print(f"FROZEN {name}: Brier trial {int(brier.trial)}, NLL trial {number}", flush=True)
    (output/"selection_complete.json").write_text(json.dumps(dict(fingerprint=fingerprint,
        settings=len(expanded_settings(config)), methods=METHODS,
        selected_sha256=hashlib.sha256((output/"selected_parameters.csv").read_bytes()).hexdigest()), indent=2), encoding="utf-8")
    return pd.DataFrame(selections)


def _method_configuration(method, brier, nll, baseline, training):
    architecture = baseline if method == "common_baseline" else nll if method.startswith("nll") else brier
    if method in ("common_baseline", "brier_original", "nll_original"):
        return architecture, training, None
    optimization = SplitOptimization(learn_splits=method != "brier_fixed", split_lr_ratio=.1,
        warmup_epochs=0 if method in ("brier_slowrho", "brier_fixed") else 10)
    actual_training = replace(training, patience=50) if method in ("brier_stable", "brier_fixed", "nll_stable") else training
    return architecture, actual_training, optimization


def evaluate_diagnostics(output_dir, historical_dir, *, config=ExperimentConfig(),
        search=SearchConfig(), training=TrainingConfig(), baseline=Architecture(), device="cuda"):
    output, historical = Path(output_dir).resolve(), Path(historical_dir).resolve()
    fingerprint = _manifest(output, historical, config, search, training, baseline, device)
    frozen = json.loads((output/"selection_complete.json").read_text(encoding="utf-8"))
    if frozen["fingerprint"] != fingerprint or frozen["settings"] != len(expanded_settings(config)):
        raise ValueError("freeze every setting before evaluating")
    if frozen["selected_sha256"] != hashlib.sha256((output/"selected_parameters.csv").read_bytes()).hexdigest():
        raise ValueError("selected parameters changed after freezing")
    selected = pd.read_csv(output/"selected_parameters.csv").set_index(["setting", "selector"])
    path = output/"runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["setting"], r["method"], int(r["seed"])) for r in rows}
    (output/"curves").mkdir(exist_ok=True)
    total = len(expanded_settings(config))*len(METHODS)*len(config.seeds)
    for name, label, setting, actual in expanded_settings(config):
        data = None  # Test truth is created only after frozen selections.
        brier = architecture_from(selected.loc[(name, "brier")], baseline)
        nll = architecture_from(selected.loc[(name, "nll")], baseline)
        source = _source(label, historical, output)
        old_path = source/"final_runs.csv"
        old = pd.read_csv(old_path) if old_path.exists() else pd.DataFrame()
        for method in METHODS:
            architecture, actual_training, optimization = _method_configuration(method, brier, nll, baseline, training)
            for seed in config.seeds:
                key = name, method, seed
                if key in done:
                    continue
                old_method = "common_baseline" if method == "common_baseline" else "per_setting_tuned" if method == "brier_original" else None
                imported = (old[(old.setting == setting.name)&(old.method == old_method)&(old.seed == seed)]
                            if old_method and len(old) else pd.DataFrame())
                prefix = f"{name}_{method}_seed{seed}"
                if len(imported):
                    if len(imported) != 1:
                        raise RuntimeError("duplicate cached fit")
                    row = imported.iloc[0].to_dict()
                    if architecture_from(row, baseline) != architecture:
                        raise ValueError("cached final architecture differs")
                    old_prefix = f"{setting.name}_{old_method}_seed{seed}"
                    curves_file = source/"curves"/f"{old_prefix}.npz"
                    histories_file = source/"histories"/f"{old_prefix}.json.gz"
                    row.update(imported_from=str(source))
                else:
                    if data is None:
                        data = _evaluation_data(setting, actual, ModelConfig(x_dim=baseline.x_dim), config.data_seed)
                    started = time.perf_counter()
                    if optimization is None:
                        model, history, info = _fit(data, architecture, actual_training, seed, device)
                    else:
                        model, history = fit_stable(data["train"], data["valid"], data["grid"],
                            architecture=architecture, training=actual_training, optimization=optimization,
                            time_scale=data["time_scale"], seed=seed, device=device)
                        info = dict(**model.fit_info, fit_seconds=time.perf_counter()-started)
                    prediction = model.predict_survival(data["test"][0], data["grid"]).cpu()
                    row = dict(**asdict(architecture), **info,
                        average_grid_sse=float(average_grid_sse(prediction, data["truth"])),
                        test_nll=evaluate_nll(model, *data["test"]),
                        test_brier=score_brier(model, data["test"], data["grid"]), max_obs=data["max_obs"], imported_from="")
                    curves_file = output/"curves"/f"{prefix}.npz"
                    histories_file = output/"histories"/f"{prefix}.json.gz"
                    np.savez_compressed(curves_file, grid=data["grid"].numpy(),
                        truth=data["truth"][:3].numpy(), survival=prediction[:3].numpy())
                    _history(output, prefix, history)
                    del model
                row.update(setting=name, family=label, scenario=setting.scenario, censor_type=setting.censor_type,
                    q=setting.q, method=method, seed=seed, curves_file=str(curves_file), histories_file=str(histories_file),
                    split_lr_ratio=1. if optimization is None else optimization.split_lr_ratio,
                    warmup_epochs=0 if optimization is None else optimization.warmup_epochs,
                    patience=actual_training.patience, learn_splits=optimization is None or optimization.learn_splits)
                rows.append(row)
                done.add(key)
                write_csv(pd.DataFrame(rows), path)
            values = [r["average_grid_sse"] for r in rows if r["setting"] == name and r["method"] == method]
            print(f"TEST [{len(rows)}/{total}] {name} {method}: {np.mean(values):.6f} +/- {np.std(values, ddof=1):.6f}", flush=True)
    runs = pd.DataFrame(rows)
    if len(runs) != total or runs.duplicated(["setting", "method", "seed"]).any():
        raise RuntimeError("incomplete or duplicate repeated evaluations")
    summary = runs.groupby(["setting", "family", "censor_type", "q", "method"], sort=False).agg(
        n_runs=("seed", "size"), average_grid_sse_mean=("average_grid_sse", "mean"),
        average_grid_sse_std=("average_grid_sse", "std"), validation_nll_mean=("validation_nll", "mean"),
        validation_brier_mean=("validation_brier", "mean"), best_epoch_mean=("best_epoch", "mean"),
        early_checkpoint_fraction=("best_epoch", lambda s: float((s <= 5).mean()))).reset_index()
    if not summary.n_runs.eq(len(config.seeds)).all():
        raise RuntimeError("incomplete seed group")
    write_csv(summary, output/"summary.csv")
    return runs, summary
