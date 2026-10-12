"""Per-setting Sobol-initialized TPE search; observed validation Brier only.

Training and checkpoint selection use NLL. Test curves are not constructed
until all settings' hyperparameters have been frozen. Each setting uses one
fixed dataset, common initialization seeds, and a completely held-out test set.
"""
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc
import torch

from .depth_experiment import _evaluation_data, _generate
from .experiment import ExperimentConfig, settings
from .search_io import write_csv as _write_csv
from .metrics import average_grid_sse
from .model import ModelConfig
from .train import TrainingConfig, evaluate_nll
from .tuning_model import Architecture
from .tuning_train import fit_tuning
from .brier import score_brier


@dataclass(frozen=True)
class SearchConfig:
    trials: int = 64
    initial_trials: int = 16
    shortlist: int = 4  # includes the common baseline
    search_seed: int = 871
    search_fit_seed: int = 101
    confirmation_seeds: tuple[int, ...] = (102,103,104)
    depth_min: int = 3
    depth_max: int = 8
    widths: tuple[int, ...] = (16,32,64,128)
    shared_layers_max: int = 3
    private_layers_max: int = 4
    strength_max: float = 1000.


def _optuna():
    try:
        import optuna
    except ImportError:
        # Append, rather than prepend, to preserve the user's installed numpy.
        sys.path.append(str(Path(__file__).parent/".deps"))
        import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _validate(config, search, baseline):
    settings(config)
    if config.resample_data:
        raise ValueError("per-setting tuning requires one fixed dataset")
    if not 2 <= search.initial_trials <= search.trials or not 2 <= search.shortlist <= search.trials:
        raise ValueError("invalid search budget")
    if not 1 <= search.depth_min <= baseline.depth <= search.depth_max:
        raise ValueError("baseline depth must be in the search space")
    if (not search.widths or len(set(search.widths)) != len(search.widths)
            or min(search.widths) < 1 or baseline.private_width not in search.widths
            or baseline.shared_width not in search.widths):
        raise ValueError("invalid search widths")
    if (search.shared_layers_max < baseline.shared_layers
            or search.private_layers_max < baseline.private_layers
            or search.strength_max < baseline.regularization_strength):
        raise ValueError("baseline must be in the search space")
    tune_seeds = (search.search_fit_seed,*search.confirmation_seeds)
    if (not search.confirmation_seeds or len(set(tune_seeds)) != len(tune_seeds)
            or set(tune_seeds) & set(config.seeds)):
        raise ValueError("search/confirmation/final seeds must be distinct")


def _manifest(output, config, search, training, baseline, device):
    output.mkdir(parents=True,exist_ok=True)
    digest = hashlib.sha256()
    for name in ("tree.py","model.py","train.py","simulation.py","experiment.py",
            "regularization.py","depth_experiment.py","metrics.py","tuning_model.py",
            "tuning_train.py","brier.py","parameter_search.py","search_io.py"):
        digest.update((Path(__file__).parent/name).read_bytes())
    specification = dict(experiment=asdict(config),search=asdict(search),
        training=asdict(training),baseline=asdict(baseline),device=str(device),
        criterion="neural_only self-imputed integrated validation Brier, 0..max_obs",
        checkpoint="unpenalized validation observed-data NLL",
        source_sha256=digest.hexdigest())
    fingerprint = hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()
    path = output/"config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("cached search uses different code/configuration; choose a new output_dir")
    else:
        path.write_text(json.dumps(dict(specification,fingerprint=fingerprint,
            python=platform.python_version(),torch=torch.__version__,optuna=_optuna().__version__,
            gpu=torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None),indent=2),
            encoding="utf-8")
    return fingerprint


def _observed_split(setting, config, architecture):
    """No latent outcomes/true distribution parameters enter this return value."""
    observed = _generate(setting,config,ModelConfig(x_dim=architecture.x_dim),config.data_seed)[:3]
    ntrain = int(config.n_samples*config.train_fraction)
    offset = ntrain+int(config.n_samples*config.validation_fraction)
    train = tuple(v[:ntrain] for v in observed)
    valid = tuple(v[ntrain:offset] for v in observed)
    bounds = torch.cat((train[1],train[2]))
    positive = bounds[torch.isfinite(bounds)&(bounds>0)]
    scale = float(positive.median()) if len(positive) else 1.
    pool = torch.cat((observed[1][:offset],observed[2][:offset]))
    max_obs = float(pool[torch.isfinite(pool)].max())
    if max_obs <= 0:
        raise ValueError("need positive max_obs")
    return dict(train=train,valid=valid,time_scale=scale,
                grid=torch.linspace(0,max_obs,config.grid_size),max_obs=max_obs)


def _coordinates(architecture):
    return dict(depth=architecture.depth,shared_width=architecture.shared_width,
        shared_layers=architecture.shared_layers,private_width=architecture.private_width,
        private_layers=architecture.private_layers,
        log10_one_plus_strength=math.log10(1+architecture.regularization_strength))


def _initial_candidates(search,baseline,seed):
    candidates = [_coordinates(baseline),_coordinates(replace(baseline,regularization_strength=0.))]
    exponent = math.ceil(math.log2(max(1,search.initial_trials-2)))
    points = qmc.Sobol(6,scramble=True,seed=seed).random_base2(exponent)
    for u in points[:search.initial_trials-2]:
        candidates.append(dict(depth=search.depth_min+int(u[0]*(1+search.depth_max-search.depth_min)),
            shared_width=search.widths[int(u[1]*len(search.widths))],
            shared_layers=int(u[2]*(search.shared_layers_max+1)),
            private_width=search.widths[int(u[3]*len(search.widths))],
            private_layers=1+int(u[4]*search.private_layers_max),
            log10_one_plus_strength=float(u[5]*math.log10(1+search.strength_max))))
    return candidates


def _suggest(trial,search,baseline):
    depth = trial.suggest_int("depth",search.depth_min,search.depth_max)
    shared_layers = trial.suggest_int("shared_layers",0,search.shared_layers_max)
    shared_width = (trial.suggest_categorical("shared_width",search.widths)
                    if shared_layers else baseline.shared_width)
    return replace(baseline,depth=depth,shared_layers=shared_layers,shared_width=shared_width,
        private_width=trial.suggest_categorical("private_width",search.widths),
        private_layers=trial.suggest_int("private_layers",1,search.private_layers_max),
        regularization_strength=10**trial.suggest_float("log10_one_plus_strength",0,
            math.log10(1+search.strength_max))-1)


def _fit(data,architecture,training,seed,device):
    started = time.perf_counter()
    model,history = fit_tuning(data["train"],data["valid"],data["grid"],
        architecture=architecture,training=training,time_scale=data["time_scale"],seed=seed,device=device)
    info = dict(**model.fit_info,fit_seconds=time.perf_counter()-started)
    return model,history,info


def _history(output,name,history):
    folder = output/"histories"
    folder.mkdir(exist_ok=True)
    with gzip.open(folder/f"{name}.json.gz","wt",encoding="utf-8") as handle:
        json.dump(history,handle)


def _tune_per_setting(output_dir, *, config=ExperimentConfig(), search=SearchConfig(),
        training=TrainingConfig(), baseline=Architecture(),device=None,verbose=True,_storage=None):
    """64 distinct configurations per setting, followed by seed confirmation.

    TPE proposes configurations adaptively. Sobol startup includes the common
    baseline and its unregularized counterpart. Shortlist includes the baseline
    and the best remaining search candidates. The final winner minimizes mean
    confirmation Brier across *new* seeds; the initial seed is excluded.
    """
    _validate(config,search,baseline)
    optuna = _optuna()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output_dir).resolve()
    fingerprint = _manifest(output,config,search,training,baseline,device)
    storage = _storage
    confirmed_path = output/"confirmation_runs.csv"
    confirmed = pd.read_csv(confirmed_path).to_dict("records") if confirmed_path.exists() else []
    done = {(r["setting"],int(r["trial"]),int(r["seed"])) for r in confirmed}
    all_rows,best_rows = [],[]
    for index,setting in enumerate(settings(config)):
        data = _observed_split(setting,config,baseline)
        study = optuna.create_study(study_name=setting.name,storage=storage,
            direction="minimize",load_if_exists=True)
        if not study.trials:
            for candidate in _initial_candidates(search,baseline,search.search_seed+index):
                study.enqueue_trial(candidate)
        # Interrupted trials have no score and must not consume the budget.
        for trial in study.get_trials(deepcopy=False,states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
            study.enqueue_trial(trial.params)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        while len(completed) < search.trials:
            # A deterministic sampler per trial makes interruption/resume reproducible.
            number = max((t.number for t in study.trials if t.state != optuna.trial.TrialState.WAITING),default=-1)+1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",optuna.exceptions.ExperimentalWarning)
                study.sampler = optuna.samplers.TPESampler(seed=search.search_seed+1009*index+number,
                    n_startup_trials=search.initial_trials,multivariate=True,group=True,n_ei_candidates=48)
            trial = study.ask()
            architecture = _suggest(trial,search,baseline)
            trial.set_user_attr("architecture",asdict(architecture))
            model,history,info = _fit(data,architecture,training,search.search_fit_seed,device)
            trial.set_user_attr("fit_info",info)
            _history(output,f"{setting.name}_trial{trial.number}_seed{search.search_fit_seed}",history)
            study.tell(trial,info["validation_brier"])
            if verbose and (len(completed)+1) % 8 == 0:
                print(f"[{index+1}/{len(settings(config))}] {setting.name}: "
                    f"{len(completed)+1}/{search.trials} trials, best Brier={study.best_value:.6g}",flush=True)
            del model
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ordered = sorted(completed,key=lambda t:(t.value,t.number))
        # Include baseline, so its seed variability is assessed in the same way.
        baseline_trial = next(t for t in completed if all(
            math.isclose(t.user_attrs["architecture"][k],v,rel_tol=1e-10,abs_tol=1e-10)
            for k,v in asdict(baseline).items()))
        shortlist = [baseline_trial]+[t for t in ordered if t.number != baseline_trial.number][:search.shortlist-1]
        for trial in completed:
            all_rows.append(dict(setting=setting.name,scenario=setting.scenario,censor_type=setting.censor_type,
                q=setting.q,trial=trial.number,stage="sobol/anchors" if trial.number<search.initial_trials else "TPE",
                seed=search.search_fit_seed,**trial.user_attrs["architecture"],**trial.user_attrs["fit_info"]))
        _write_csv(pd.DataFrame(all_rows),output/"trials.csv")
        for trial in shortlist:
            architecture = Architecture(**trial.user_attrs["architecture"])
            for seed in search.confirmation_seeds:
                key = (setting.name,trial.number,seed)
                if key in done:
                    continue
                model,history,info = _fit(data,architecture,training,seed,device)
                confirmed.append(dict(setting=setting.name,trial=trial.number,seed=seed,
                    **asdict(architecture),**info))
                done.add(key)
                _history(output,f"{setting.name}_confirm{trial.number}_seed{seed}",history)
                _write_csv(pd.DataFrame(confirmed),confirmed_path)
                del model
        candidate_scores = pd.DataFrame([r for r in confirmed if r["setting"]==setting.name]).groupby("trial").agg(
            selection_brier=("validation_brier","mean"),selection_brier_std=("validation_brier","std"),
            confirmation_nll=("validation_nll","mean"),n=("seed","size"))
        if not candidate_scores.n.eq(len(search.confirmation_seeds)).all():
            raise RuntimeError("incomplete shortlist confirmation")
        best_number = int(candidate_scores.sort_values(["selection_brier","confirmation_nll"]).index[0])
        winner = next(t for t in shortlist if t.number == best_number)
        best_rows.append(dict(setting=setting.name,scenario=setting.scenario,censor_type=setting.censor_type,
            q=setting.q,trial=best_number,**winner.user_attrs["architecture"],
            **candidate_scores.loc[best_number].to_dict(),max_obs=data["max_obs"]))
        _write_csv(pd.DataFrame(best_rows),output/"best_parameters.csv")
        if verbose:
            print(f"SELECTED {setting.name}: {winner.user_attrs['architecture']}; "
                  f"confirmation Brier={candidate_scores.loc[best_number,'selection_brier']:.6g}",flush=True)
    (output/"selection_complete.json").write_text(json.dumps(dict(fingerprint=fingerprint,
        settings=len(best_rows),criterion="mean validation Brier over confirmation seeds"),indent=2),encoding="utf-8")
    return pd.DataFrame(best_rows),pd.DataFrame(all_rows),pd.DataFrame(confirmed)


def tune_per_setting(output_dir, **kwargs):
    """Run/reload per-setting TPE search; release SQLite handles on Windows."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    storage = _optuna().storages.RDBStorage(url=f"sqlite:///{(output/'studies.sqlite3').as_posix()}")
    try:
        return _tune_per_setting(output_dir,**kwargs,_storage=storage)
    finally:
        storage.remove_session()
        storage.engine.dispose()


def evaluate_selected(output_dir, *, config=ExperimentConfig(), search=SearchConfig(),
        training=TrainingConfig(),baseline=Architecture(),device=None,verbose=True):
    """Freeze the selection first, then ten fits each for tuned/common baseline.

    Error is grid SSE then mean over test subjects, and finally mean over fits.
    The score is never the error of a seed-averaged survival curve.
    """
    _validate(config,search,baseline)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output_dir).resolve()
    fingerprint = _manifest(output,config,search,training,baseline,device)
    selection = json.loads((output/"selection_complete.json").read_text(encoding="utf-8"))
    if selection["fingerprint"] != fingerprint or selection["settings"] != len(settings(config)):
        raise ValueError("hyperparameter selection must finish before evaluating test curves")
    best = pd.read_csv(output/"best_parameters.csv").set_index("setting")
    path = output/"final_runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["setting"],r["method"],int(r["seed"])) for r in rows}
    (output/"curves").mkdir(exist_ok=True)
    for setting in settings(config):
        data = _evaluation_data(setting,config,ModelConfig(x_dim=baseline.x_dim),config.data_seed)
        fields = asdict(baseline)
        architecture = Architecture(**{k:(int(best.loc[setting.name,k]) if isinstance(v,int)
            else float(best.loc[setting.name,k])) for k,v in fields.items()})
        for method,actual in (("common_baseline",baseline),("per_setting_tuned",architecture)):
            for seed in config.seeds:
                if (setting.name,method,seed) in done:
                    continue
                model,history,info = _fit(data,actual,training,seed,device)
                prediction = model.predict_survival(data["test"][0],data["grid"]).cpu()
                row = dict(setting=setting.name,scenario=setting.scenario,censor_type=setting.censor_type,
                    q=setting.q,method=method,seed=seed,**asdict(actual),**info,
                    average_grid_sse=float(average_grid_sse(prediction,data["truth"])),
                    test_nll=evaluate_nll(model,*data["test"]),
                    test_brier=score_brier(model,data["test"],data["grid"]),max_obs=data["max_obs"])
                rows.append(row)
                done.add((setting.name,method,seed))
                _history(output,f"{setting.name}_{method}_seed{seed}",history)
                np.savez_compressed(output/"curves"/f"{setting.name}_{method}_seed{seed}.npz",
                    grid=data["grid"].numpy(),truth=data["truth"][:3].numpy(),survival=prediction[:3].numpy())
                _write_csv(pd.DataFrame(rows),path)
                del model
            if verbose:
                selected = [r["average_grid_sse"] for r in rows if r["setting"]==setting.name and r["method"]==method]
                print(f"TEST {setting.name} {method}: SSE={np.mean(selected):.6g} +/- {np.std(selected,ddof=1):.6g}",flush=True)
    runs = pd.DataFrame(rows)
    group = ["setting","scenario","censor_type","q","method"]
    summary = runs.groupby(group,sort=False).agg(n_runs=("seed","size"),
        average_grid_sse_mean=("average_grid_sse","mean"),average_grid_sse_std=("average_grid_sse","std"),
        validation_brier_mean=("validation_brier","mean"),validation_nll_mean=("validation_nll","mean"),
        test_brier_mean=("test_brier","mean"),test_nll_mean=("test_nll","mean"),
        parameters=("parameters","first"),epochs_mean=("epochs_run","mean")).reset_index()
    if not summary.n_runs.eq(len(config.seeds)).all():
        raise RuntimeError("incomplete repeated evaluation")
    _write_csv(summary,output/"final_summary.csv")
    paired = runs.pivot(index=["setting","seed"],columns="method",values="average_grid_sse")
    paired["difference"] = paired.per_setting_tuned-paired.common_baseline
    _write_csv(paired.reset_index(),output/"paired_errors.csv")
    return runs,summary
