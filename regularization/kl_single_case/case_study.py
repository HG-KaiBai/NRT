"""One KL fit on exactly neural_only/testdata2.py's data and 80/20 split."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from pycode.neural_only import testdata2 as simulation
from pycode.unified_nll.raw_time.model import (observed_upper_bound, predict_density,
                                              fit_raw_time_kl, SUPPORT)
from pycode.unified_nll.metrics import average_grid_sse
from pycode.unified_nll.nll_cv_train import NLLCVOptimization
from pycode.unified_nll.train import TrainingConfig, evaluate_nll
from pycode.unified_nll.tuning_model import Architecture


@dataclass(frozen=True)
class CaseConfig:
    seed: int = 123
    n_samples: int = 5000
    x_dim: int = 6
    scenario: str = "ph"
    censor_type: str = "right"
    censor_q: float = 0.5
    multimodal_case: str = "separate"
    grid_size: int = 256

    @property
    def setting(self):
        family = ("multimodal_close" if self.multimodal_case == "closer"
                  else "multimodal_separate") if self.scenario == "multimodal" else self.scenario
        return f"{family}_{self.censor_type}_q{0 if self.censor_type == 'exact' else self.censor_q:g}"


def generate_case(config=CaseConfig()):
    """Reset the CPU seed immediately before the very same generator call.

    There is no additional covariate scaling, event clipping, reshuffling,
    censoring calibration, or regenerated test sample in this wrapper.
    """
    if config.n_samples < 5 or config.x_dim < 2 or config.grid_size < 3:
        raise ValueError("need n_samples>=5, x_dim>=2 and grid_size>=3")
    torch.manual_seed(config.seed)
    kwargs = dict(n=config.n_samples, x_dim=config.x_dim, q=config.censor_q,
                  censor_type=config.censor_type, device=torch.device("cpu"))
    if config.scenario == "multimodal":
        raw = simulation.data_reg_multimodal(**kwargs, multimodal_case=config.multimodal_case)
    else:
        raw = simulation.data_reg(**kwargs, scenario=config.scenario)
    n_train = int(0.8 * len(raw[0]))
    train = tuple(value[:n_train] for value in raw[:3])
    test = tuple(value[n_train:] for value in raw[:3])
    bounds = torch.cat((train[1][torch.isfinite(train[1])], train[2][torch.isfinite(train[2])]))
    max_obs = bounds.max().item()
    if not max_obs > 0:
        raise ValueError("need a positive finite training endpoint")
    grid = torch.linspace(0.0, max_obs, config.grid_size)
    if config.scenario == "multimodal":
        truth = simulation.multimodal_survival(raw[4][n_train:], raw[5][n_train:], grid)
    else:
        truth = simulation.survival_curves(raw[4][n_train:], raw[5][n_train:], grid,
                                          scenario=config.scenario)
    return dict(raw=raw, train=train, test=test, n_train=n_train, max_obs=max_obs,
                grid=grid, truth=truth, upper_bound=observed_upper_bound(train))


def load_selected(case=CaseConfig(), path=None):
    """Load only a frozen raw-time finite-support KL selection."""
    if path is None:
        path = (Path(__file__).resolve().parents[1] / "finite_support_kl" / "results"
                / "finite_support_normalization_comparison" / "selected_parameters.csv")
    frame = pd.read_csv(path, float_precision="round_trip")
    if "arm" in frame:
        rows = frame[(frame.setting == case.setting) & (frame.arm == "raw_time_finite_kl")]
    elif "method" in frame:
        manifest = json.loads(Path(path).with_name("config.json").read_text(encoding="utf-8"))
        if manifest.get("support") != SUPPORT or manifest.get("time_coordinate") != "raw":
            raise ValueError("saved selection must use raw-time finite support")
        rows = frame[(frame.setting == case.setting) & (frame.method == "adaptive_nll_cv")]
    else:
        raise ValueError("saved selection must identify the raw-time finite arm")
    if len(rows) != 1:
        raise ValueError(f"expected one adaptive KL selection for {case.setting}")
    row = rows.iloc[0]
    if int(row.x_dim) != case.x_dim:
        raise ValueError("saved selection has a different X_DIM; set LOAD_SAVED_SELECTION=False")
    defaults = Architecture()
    values = {field.name: (int(row[field.name]) if isinstance(getattr(defaults, field.name), int)
                           else float(row[field.name])) for field in fields(defaults)}
    return Architecture(**values), int(row.refit_epochs)


def prepare_output(output, config, architecture, training, optimization, epochs, device):
    """Reject incompatible reruns before fitting or overwriting artifacts."""
    output = Path(output)
    generator = Path(simulation.__file__)
    manifest = dict(case=asdict(config), architecture=asdict(architecture),
                    training=asdict(training), optimization=asdict(optimization),
                    refit_epochs=int(epochs), device=str(device),
                    generator_sha256=hashlib.sha256(generator.read_bytes()).hexdigest(),
                    generator="pycode.neural_only.testdata2",
                    split="first 80% fit, last 20% test; no shuffle",
                    selection="frozen raw-time finite KL CV parameters or explicit manual parameters; no test stopping",
                    support=SUPPORT, time_coordinate="raw",
                    upper_bound_rule="1.1 * maximum finite observed training L/R endpoint")
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("case configuration changed; use a new OUTPUT directory")
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_case(data, config, architecture, epochs, *, training=TrainingConfig(patience=50),
               optimization=NLLCVOptimization(), device="cpu"):
    if architecture.x_dim != config.x_dim or not optimization.learn_splits:
        raise ValueError("single case requires matching X_DIM and adaptive KL")
    started = time.perf_counter()
    model, history = fit_raw_time_kl(data["train"], architecture=architecture,
        training=training, optimization=optimization,
        seed=config.seed, device=str(device), fixed_epochs=epochs)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return model, history, time.perf_counter() - started


@torch.no_grad()
def evaluate_case(model, data, fit_seconds):
    estimated = model.predict_survival(data["test"][0], data["grid"]).cpu()
    errors = (estimated - data["truth"]).square().sum(dim=1)
    result = dict(method="nll_plus_width_kl", test_nll=evaluate_nll(model, *data["test"]),
                  average_grid_sse=float(average_grid_sse(estimated, data["truth"])),
                  n_fit=len(data["train"][0]), n_test=len(data["test"][0]),
                  training_epochs=model.fit_info["epochs_run"], fit_seconds=fit_seconds,
                  support=SUPPORT, upper_bound=float(model.upperbound), max_obs=data["max_obs"])
    return pd.DataFrame([result]), estimated, errors


def plot_case(model, data, estimated, history, *, index=4, output=None):
    """Survival, training loss, and raw-time piecewise-constant density."""
    import matplotlib.pyplot as plt
    if not 0 <= index < len(estimated):
        raise ValueError("INDEX must refer to a test observation")
    figures = {}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data["grid"], estimated[index], label="NLL + KL")
    ax.plot(data["grid"], data["truth"][index], "k--", label="true")
    ax.set(xlabel="time", ylabel="survival probability", ylim=(0, 1),
           title=f"Conditional survival: test observation {index}")
    ax.legend(); figures["survival"] = fig
    fig, ax = plt.subplots(figsize=(8, 4))
    frame = pd.DataFrame(history)
    ax.plot(frame.epoch, frame.train_nll, label="training NLL")
    ax.plot(frame.epoch, frame.objective, label="NLL + KL")
    ax.set(xlabel="epoch", ylabel="loss"); ax.legend(); figures["training"] = fig
    if len(data["raw"]) == 7:
        grid = torch.linspace(0, data["upper_bound"], 1024, device=model.upperbound.device, dtype=torch.float64)
        raw = data["raw"]; offset = data["n_train"] + index
        scales, weights = raw[5][offset].to(grid), raw[4][offset].to(grid)
        shapes = scales.new_tensor(simulation.MULTIMODAL_SHAPES)
        z = grid[:, None] / scales[None, :]
        true_pdf = ((shapes / scales) * z.pow(shapes - 1) * torch.exp(-z.pow(shapes)) * weights).sum(1)
        density = predict_density(model, data["test"][0][index:index + 1], grid)[0]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(grid.cpu(), density.cpu(), label="NLL + KL")
        ax.plot(grid.cpu(), true_pdf.cpu(), "k--", label="true PDF")
        ax.set(xlabel="time", ylabel="density"); ax.legend(); figures["density"] = fig
    for name, fig in figures.items():
        fig.tight_layout()
        if output is not None:
            fig.savefig(Path(output) / f"{name}.png", dpi=160)
    return figures


def save_case(output, config, data, model, history, results, estimated, errors):
    output = Path(output)
    results.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame({"test_observation": np.arange(len(errors)), "grid_sse": errors.numpy()}).to_csv(
        output / "per_observation_sse.csv", index=False)
    np.savez_compressed(output / "generated_data.npz",
                        **{f"raw_{i}": value.numpy() for i, value in enumerate(data["raw"])})
    np.savez_compressed(output / "curves.npz", grid=data["grid"].numpy(),
                        estimated=estimated.numpy(), truth=data["truth"].numpy())
    torch.save(dict(state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                    architecture=asdict(model.config), fit_info=model.fit_info,
                    case=asdict(config), support=SUPPORT, time_coordinate="raw",
                    optimization=dict(learn_splits=model.learn_splits,
                        split_lr_ratio=model.fit_info["split_lr_ratio"],
                        warmup_epochs=model.fit_info["warmup_epochs"]),
                    upper_bound=float(model.upperbound)), output / "model.pt")
