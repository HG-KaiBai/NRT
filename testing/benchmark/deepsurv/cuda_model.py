"""CUDA execution of the pinned DeepSurv network and training algorithm.

The module follows upstream ``DeepSurv/deepsurv/deep_surv.py`` at the pinned
benchmark commit.  PyTorch replaces Theano/Lasagne as the tensor backend; the
layer order, Cox objective, parameter regularization, and experiment-runner
optimizer semantics are kept the same.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .losses import cox_partial_log_likelihood
from ...validation import training_validation_split


OFFICIAL_DEEPSURV_COMMIT = "41eed003e5b892c81e7855e400861fa7a2d9da4f"

# Authors' first nonlinear-Gaussian configuration, retained as a reference.
OFFICIAL_MODEL_PARAMETERS: dict[str, object] = {
    "hidden_layers": (17, 17, 17),
    "activation": "rectify",
    "dropout": 0.4013037109375,
    "batch_norm": True,
    "standardize": False,
}
OFFICIAL_OPTIMIZATION_PARAMETERS: dict[str, object] = {
    "learning_rate": 0.00031940075434571123,
    "lr_decay": 0.000317275390625,
    "momentum": 0.9363432617187499,
    "l2_reg": 4.4249755859375,
    "l1_reg": 0.0,
    "epochs": 500,
    "optimizer": "sgd",
}

# Exact model and runner used by experiments/deepsurv/Dockerfile.gaussian.
PAPER_GAUSSIAN_PARAMETERS: dict[str, object] = {
    "hidden_layers": (39, 39, 39),
    "activation": "selu",
    "dropout": 0.0735595703125,
    "batch_norm": False,
    "standardize": False,
    "learning_rate": 0.001485675043584323,
    "lr_decay": 0.00024959960937500004,
    "momentum": 0.93199267578125,
    "l2_reg": 2.78821044921875,
    "l1_reg": 0.0,
    "epochs": 1000,
    "optimizer": "adam",
}

# Hidden-layer widths/depth and the other search variables may vary; all
# remaining model/training defaults come from the paper Gaussian preset.
TUNABLE_PARAMETER_NAMES = frozenset({
    "hidden_layers", "learning_rate", "lr_decay", "momentum", "l2_reg", "dropout",
})
PAPER_GAUSSIAN_FIXED_PARAMETERS = {
    name: value for name, value in PAPER_GAUSSIAN_PARAMETERS.items()
    if name not in TUNABLE_PARAMETER_NAMES
}

OFFICIAL_TUNING_PARAMETERS: dict[str, object] = {
    "reference_search": {
        "ranges": {
            "log10_learning_rate": [-5.0, -3.0],
            "nodes_per_layer": [30, 50],
            "hidden_layers": [1, 4],
            "lr_decay": [0.0, 0.001],
            "momentum": [0.8, 0.95],
            "l2_reg": [0.05, 5.0],
            "dropout": [0.0, 0.2],
        }
    }
}


@dataclass(frozen=True)
class _Configuration:
    hidden_layers: tuple[int, ...]
    learning_rate: float
    lr_decay: float
    momentum: float
    l2_reg: float
    dropout: float | None
    activation: str
    batch_norm: bool
    standardize: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "hidden_layers": list(self.hidden_layers),
            "learning_rate": self.learning_rate,
            "lr_decay": self.lr_decay,
            "momentum": self.momentum,
            "l2_reg": self.l2_reg,
            "dropout": self.dropout,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "standardize": self.standardize,
        }


class _LasagneBatchNorm1d(nn.Module):
    """Lasagne BatchNormLayer semantics implemented with CUDA tensors."""

    def __init__(self, features: int, *, alpha: float = 0.1, epsilon: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_inv_std", torch.ones(features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = values.mean(dim=0)
            inv_std = torch.rsqrt(values.var(dim=0, correction=0) + self.epsilon)
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.alpha)
                self.running_inv_std.lerp_(inv_std, self.alpha)
        else:
            mean = self.running_mean
            inv_std = self.running_inv_std
        return (values - mean) * (self.weight * inv_std) + self.bias


class _RiskNetwork(nn.Module):
    """Lasagne's DeepSurv layer graph expressed with PyTorch modules."""

    def __init__(
        self,
        input_features: int,
        hidden_layers: tuple[int, ...],
        *,
        activation: str,
        dropout: float | None,
        batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_features
        for width in hidden_layers:
            linear = nn.Linear(previous, width, bias=not batch_norm)
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            if batch_norm:
                layers.append(_LasagneBatchNorm1d(width))
            layers.append(nn.ReLU() if activation == "rectify" else nn.SELU())
            if dropout is not None and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous = width

        output = nn.Linear(previous, 1)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class DeepSurvCUDA:
    """DeepSurv's neural Cox model implemented with PyTorch CUDA.

    This reproduces the official network, full-risk-set Cox loss, Xavier
    initialization, dropout, batch normalization, output-layer
    regularization, and full-batch optimizer used by the pinned experiment
    runner.
    It is intentionally labelled as a CUDA reimplementation rather than the
    authors' original Theano source.
    """

    supported_censor_types = frozenset({"exact", "right"})

    def __init__(
        self,
        *,
        hidden_layers: tuple[int, ...] = PAPER_GAUSSIAN_PARAMETERS["hidden_layers"],
        learning_rate: float = PAPER_GAUSSIAN_PARAMETERS["learning_rate"],
        lr_decay: float = PAPER_GAUSSIAN_PARAMETERS["lr_decay"],
        momentum: float = PAPER_GAUSSIAN_PARAMETERS["momentum"],
        l2_reg: float = PAPER_GAUSSIAN_PARAMETERS["l2_reg"],
        l1_reg: float = PAPER_GAUSSIAN_PARAMETERS["l1_reg"],
        activation: str = PAPER_GAUSSIAN_PARAMETERS["activation"],
        dropout: float | None = PAPER_GAUSSIAN_PARAMETERS["dropout"],
        batch_norm: bool = PAPER_GAUSSIAN_PARAMETERS["batch_norm"],
        standardize: bool = PAPER_GAUSSIAN_PARAMETERS["standardize"],
        epochs: int = PAPER_GAUSSIAN_PARAMETERS["epochs"],
        patience: int = 20,
        tune: bool = False,
        tuning_evals: int = 50,
        cv_folds: int = 3,
        tuning_epochs: int = PAPER_GAUSSIAN_PARAMETERS["epochs"],
        optimizer: str = PAPER_GAUSSIAN_PARAMETERS["optimizer"],
        seed: int = 123,
        device: str | torch.device = "cuda",
        verbose: bool = False,
    ) -> None:
        hidden_layers = tuple(int(width) for width in hidden_layers)
        if not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")
        if learning_rate <= 0.0 or lr_decay < 0.0:
            raise ValueError("learning_rate must be positive and lr_decay nonnegative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if l1_reg < 0.0 or l2_reg < 0.0:
            raise ValueError("regularization coefficients must be nonnegative")
        if activation not in {"rectify", "selu"}:
            raise ValueError("activation must be 'rectify' or 'selu'")
        if dropout is not None and not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1) or None")
        if epochs < 1 or tuning_epochs < 1 or tuning_evals < 1:
            raise ValueError("epoch and tuning counts must be positive")
        if patience < 1:
            raise ValueError("patience must be positive")
        if cv_folds < 2:
            raise ValueError("cv_folds must be at least two")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'")

        self.hidden_layers = hidden_layers
        self.learning_rate = float(learning_rate)
        self.lr_decay = float(lr_decay)
        self.momentum = float(momentum)
        self.l2_reg = float(l2_reg)
        self.l1_reg = float(l1_reg)
        self.activation = activation
        self.dropout = None if dropout is None else float(dropout)
        self.batch_norm = bool(batch_norm)
        self.standardize = bool(standardize)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.tune = bool(tune)
        self.tuning_evals = int(tuning_evals)
        self.cv_folds = int(cv_folds)
        self.tuning_epochs = int(tuning_epochs)
        self.optimizer = optimizer
        self.seed = int(seed)
        self.device = torch.device(device)
        self.verbose = bool(verbose)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        self._model: _RiskNetwork | None = None
        self._feature_count: int | None = None
        self._input_mean: torch.Tensor | None = None
        self._input_std: torch.Tensor | None = None
        self._baseline_times: torch.Tensor | None = None
        self._baseline_cumulative_hazard: torch.Tensor | None = None
        self._risk_shift: torch.Tensor | None = None
        self._best_cv_c_index: float | None = None
        self._tuning_history: list[dict[str, object]] = []
        self.best_hyperparameters_: dict[str, object] | None = None
        self.best_validation_c_index_: float | None = None
        self.training_history_: list[dict[str, float]] = []

    @property
    def modeling_parameters(self) -> dict[str, object]:
        return {
            "implementation": (
                "PyTorch CUDA reproduction of jaredleekatzman/DeepSurv@"
                + OFFICIAL_DEEPSURV_COMMIT
            ),
            "device": str(self.device),
            "hidden_layers": list(self.hidden_layers),
            "learning_rate": self.learning_rate,
            "lr_decay": self.lr_decay,
            "momentum": self.momentum,
            "l2_reg": self.l2_reg,
            "l1_reg": self.l1_reg,
            "activation": self.activation,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "standardize": self.standardize,
            "epochs": self.epochs,
            "patience": self.patience,
            "validation_fraction": 0.2,
            "early_stopping_metric": "validation negative Cox partial log-likelihood",
            "optimizer": f"full-batch {self.optimizer.upper()}",
            "optimizer_semantics": (
                "constant learning rate; the pinned runner compiles its "
                "optimizer once and does not apply the later Python "
                "learning-rate and momentum rebindings"
            ),
            "regularization_scope": "final linear risk-layer weight",
            "loss": "negative Cox partial log-likelihood",
        }

    @property
    def tuning_parameters(self) -> dict[str, object]:
        reference = OFFICIAL_TUNING_PARAMETERS["reference_search"]
        return {
            "requested": self.tune,
            "performed_in_benchmark": self._best_cv_c_index is not None,
            "device": str(self.device),
            "sampler": "Optunity Sobol",
            "evaluations": self.tuning_evals,
            "cross_validation_folds": self.cv_folds,
            "epochs_per_fold": self.tuning_epochs,
            "selection_metric": "validation Harrell C-index",
            "ranges": dict(reference["ranges"]),
            "fixed_candidate_parameters": {
                "activation": self.activation,
                "batch_norm": self.batch_norm,
                "standardize": self.standardize,
                "l1_reg": self.l1_reg,
                "epochs": self.tuning_epochs,
                "optimizer": self.optimizer,
            },
            "best_cv_c_index": self._best_cv_c_index,
            "selected_parameters": (
                None
                if self._best_cv_c_index is None
                else dict(self.best_hyperparameters_ or {})
            ),
            "cross_validation_fits": list(self._tuning_history),
        }

    @staticmethod
    def _validate_training_data(
        x: torch.Tensor,
        lower: torch.Tensor,
        right: torch.Tensor,
        censor_type: str,
    ) -> None:
        if censor_type not in DeepSurvCUDA.supported_censor_types:
            raise ValueError("DeepSurv supports only exact and right-censored data")
        if (
            x.ndim != 2
            or lower.ndim != 1
            or right.ndim != 1
            or x.shape[0] != lower.numel()
            or lower.shape != right.shape
        ):
            raise ValueError("x must be 2-D and bounds must have one value per row")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(lower).all()):
            raise ValueError("DeepSurv requires finite covariates and observed times")
        if not bool((lower >= 0).all()) or not bool((right >= lower).all()):
            raise ValueError("DeepSurv requires 0 <= L <= R")
        if censor_type == "exact" and not bool(
            (torch.isfinite(right) & (right == lower)).all()
        ):
            raise ValueError("exact observations require finite R equal to L")
        if censor_type == "right" and not bool(
            (torch.isinf(right) | (right == lower)).all()
        ):
            raise ValueError(
                "right-censored observations require R=L for events and "
                "R=inf for censored rows"
            )
        if not bool(torch.isfinite(right).any()):
            raise ValueError("DeepSurv requires at least one observed event")

    def _seed(self, offset: int = 0) -> None:
        value = self.seed + offset
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)

    def _make_model(
        self, input_features: int, configuration: _Configuration, seed_offset: int
    ) -> _RiskNetwork:
        self._seed(seed_offset)
        return _RiskNetwork(
            input_features,
            configuration.hidden_layers,
            activation=configuration.activation,
            dropout=configuration.dropout,
            batch_norm=configuration.batch_norm,
        ).to(self.device)

    def _regularization(
        self, model: _RiskNetwork, configuration: _Configuration
    ) -> torch.Tensor:
        """Match ``regularize_layer_params(self.network, ...)`` exactly.

        In the GitHub source ``self.network`` is the final DenseLayer, not the
        complete layer graph.  Lasagne therefore regularizes only that layer's
        regularizable weight; biases and all hidden/BatchNorm parameters are
        excluded.  Using the supplied configuration is also important during
        tuning, where every candidate has its own L2 coefficient.
        """
        penalty = torch.zeros((), device=self.device)
        if self.l1_reg == 0.0 and configuration.l2_reg == 0.0:
            return penalty
        output_weight = model.network[-1].weight
        if self.l1_reg:
            penalty = penalty + self.l1_reg * output_weight.abs().sum()
        if configuration.l2_reg:
            penalty = penalty + configuration.l2_reg * output_weight.square().sum()
        return penalty

    @staticmethod
    def _normalization(
        values: torch.Tensor, *, standardize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not standardize:
            return values.new_zeros(values.shape[1]), values.new_ones(values.shape[1])
        return values.mean(dim=0), values.std(dim=0, correction=0).clamp_min(1e-6)

    def _train_network(
        self,
        x: torch.Tensor,
        observed: torch.Tensor,
        event: torch.Tensor,
        configuration: _Configuration,
        *,
        epochs: int,
        seed_offset: int,
        validation_data=None,
    ) -> tuple[_RiskNetwork, torch.Tensor, torch.Tensor]:
        if validation_data is None:
            training, validation = training_validation_split(
                len(x), seed=self.seed + seed_offset, device=x.device,
            )
            if int(event.sum()) < 2:
                raise ValueError("DeepSurv needs at least two observed events for training and validation")
            # Keep the split sizes fixed while ensuring both Cox losses have an event.
            for target, source in ((training, validation), (validation, training)):
                if not bool(event[target].any()):
                    source_position = torch.nonzero(event[source], as_tuple=False)[0, 0]
                    replacement = target[0].clone()
                    target[0] = source[source_position]
                    source[source_position] = replacement
            self._training_indices = training
            self._validation_indices = validation
            validation_x = x[validation]
            validation_times, validation_event = observed[validation], event[validation]
            x, observed, event = x[training], observed[training], event[training]
        else:
            self._training_indices = torch.arange(len(x), device=x.device)
            validation_x, validation_times, validation_event = validation_data
            if not bool(event.any()) or not bool(validation_event.any()):
                raise ValueError("DeepSurv training and validation each require an observed event")
        mean, std = self._normalization(
            x, standardize=configuration.standardize
        )
        normalized = (x - mean) / std
        order = torch.argsort(observed, descending=True, stable=True)
        normalized = normalized[order]
        sorted_event = event[order]
        validation_order = torch.argsort(validation_times, descending=True, stable=True)
        normalized_validation = ((validation_x - mean) / std)[validation_order]
        sorted_validation_event = validation_event[validation_order]
        model = self._make_model(x.shape[1], configuration, seed_offset)
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=configuration.learning_rate
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=configuration.learning_rate
            )

        best_loss, best_state, stale = float("inf"), None, 0
        self.training_history_ = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            risk = model(normalized)
            loss = cox_partial_log_likelihood(
                risk, sorted_event, ties="noties"
            ) + self._regularization(
                model, configuration
            )
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = cox_partial_log_likelihood(
                    model(normalized_validation), sorted_validation_event, ties="noties",
                ).item()
            if not np.isfinite(validation_loss):
                raise RuntimeError("Non-finite DeepSurv validation loss")
            self.training_history_.append(dict(epoch=epoch + 1, validation_nll=validation_loss))
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if self.verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                print(f"DeepSurv CUDA epoch {epoch + 1}/{epochs}: validation_nll={validation_loss:.6f}")
            if stale >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model, mean, std

    def _tune(
        self, x: torch.Tensor, observed: torch.Tensor, event: torch.Tensor
    ) -> _Configuration:
        """Run the GitHub hyperparameter-search construction on CUDA fits."""
        import optunity
        from lifelines.utils import concordance_index

        x_values = x.detach().cpu().numpy().astype(np.float32, copy=False)
        times = observed.detach().cpu().numpy().astype(np.float32, copy=False)
        events = event.detach().cpu().numpy().astype(np.int32, copy=False)
        outcomes = np.column_stack((events, times))

        random.seed(self.seed)
        np.random.seed(self.seed)
        censored = np.flatnonzero(events == 0).tolist()
        strata = [censored] if censored else None
        folds = optunity.generate_folds(
            x_values.shape[0], num_folds=self.cv_folds, strata=strata
        )
        self._tuning_history = []

        @optunity.cross_validated(
            x=x_values,
            y=outcomes,
            num_folds=self.cv_folds,
            folds=[folds],
        )
        def objective(
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            learning_rate: float,
            num_nodes: float,
            num_layers: float,
            lr_decay: float,
            momentum: float,
            L2_reg: float,
            dropout: float,
        ) -> float:
            configuration = _Configuration(
                hidden_layers=(int(num_nodes),) * int(num_layers),
                learning_rate=10.0 ** float(learning_rate),
                lr_decay=float(lr_decay),
                momentum=float(momentum),
                l2_reg=float(L2_reg),
                dropout=float(dropout),
                activation=self.activation,
                batch_norm=self.batch_norm,
                standardize=self.standardize,
            )
            train_x = torch.as_tensor(x_train, device=self.device)
            train_times = torch.as_tensor(y_train[:, 1], device=self.device)
            train_event = torch.as_tensor(
                y_train[:, 0], device=self.device, dtype=torch.float32
            )
            model, mean, std = self._train_network(
                train_x,
                train_times,
                train_event,
                configuration,
                epochs=self.tuning_epochs,
                seed_offset=len(self._tuning_history),
                validation_data=(
                    torch.as_tensor(x_test, device=self.device),
                    torch.as_tensor(y_test[:, 1], device=self.device),
                    torch.as_tensor(y_test[:, 0], device=self.device, dtype=torch.float32),
                ),
            )
            with torch.no_grad():
                validation_x = torch.as_tensor(x_test, device=self.device)
                validation_risk = (
                    model((validation_x - mean) / std).detach().cpu().numpy()
                )
            partial_hazard = np.exp(np.clip(validation_risk, -50.0, 50.0))
            score = float(
                concordance_index(
                    y_test[:, 1], -partial_hazard, y_test[:, 0].astype(bool)
                )
            )
            record = configuration.as_dict()
            record["device"] = str(self.device)
            record["validation_c_index"] = score
            self._tuning_history.append(record)
            if self.verbose:
                print(
                    f"DeepSurv CUDA tuning fit {len(self._tuning_history)}: "
                    f"C-index={score:.6f}"
                )
            return score

        optimum, details, _ = optunity.maximize(
            objective,
            num_evals=self.tuning_evals,
            solver_name="sobol",
            **{
                argument: list(OFFICIAL_TUNING_PARAMETERS["reference_search"]["ranges"][name])
                for argument, name in {
                    "learning_rate": "log10_learning_rate",
                    "num_nodes": "nodes_per_layer",
                    "num_layers": "hidden_layers",
                    "lr_decay": "lr_decay",
                    "momentum": "momentum",
                    "L2_reg": "l2_reg",
                    "dropout": "dropout",
                }.items()
            },
        )
        self._best_cv_c_index = float(details.optimum)
        self.best_validation_c_index_ = self._best_cv_c_index
        configuration = _Configuration(
            hidden_layers=(int(optimum["num_nodes"]),)
            * int(optimum["num_layers"]),
            learning_rate=10.0 ** float(optimum["learning_rate"]),
            lr_decay=float(optimum["lr_decay"]),
            momentum=float(optimum["momentum"]),
            l2_reg=float(optimum["L2_reg"]),
            dropout=float(optimum["dropout"]),
            activation=self.activation,
            batch_norm=self.batch_norm,
            standardize=self.standardize,
        )
        self.best_hyperparameters_ = configuration.as_dict()
        return configuration

    def _current_configuration(self) -> _Configuration:
        return _Configuration(
            hidden_layers=self.hidden_layers,
            learning_rate=self.learning_rate,
            lr_decay=self.lr_decay,
            momentum=self.momentum,
            l2_reg=self.l2_reg,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
            standardize=self.standardize,
        )

    def _apply_configuration(self, configuration: _Configuration) -> None:
        self.hidden_layers = configuration.hidden_layers
        self.learning_rate = configuration.learning_rate
        self.lr_decay = configuration.lr_decay
        self.momentum = configuration.momentum
        self.l2_reg = configuration.l2_reg
        self.dropout = configuration.dropout
        self.activation = configuration.activation
        self.batch_norm = configuration.batch_norm
        self.standardize = configuration.standardize

    def fit(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        *,
        censor_type: str,
        validation_data=None,
    ) -> "DeepSurvCUDA":
        self._validate_training_data(x, L, R, censor_type)
        x_device = x.detach().to(self.device, dtype=torch.float32)
        observed = L.detach().to(self.device, dtype=torch.float32)
        event = torch.isfinite(R.detach()).to(self.device, dtype=torch.float32)
        if validation_data is not None:
            valid_x, valid_L, valid_R = validation_data
            self._validate_training_data(valid_x, valid_L, valid_R, censor_type)
            validation_data = (
                valid_x.detach().to(self.device, dtype=torch.float32),
                valid_L.detach().to(self.device, dtype=torch.float32),
                torch.isfinite(valid_R.detach()).to(self.device, dtype=torch.float32),
            )
        if x_device.shape[0] < self.cv_folds and self.tune:
            raise ValueError("training rows must be at least cv_folds")

        configuration = self._current_configuration()
        self._best_cv_c_index = None
        self._tuning_history = []
        self.best_hyperparameters_ = None
        self.best_validation_c_index_ = None
        if self.tune:
            configuration = self._tune(x_device, observed, event)
            self._apply_configuration(configuration)

        model, mean, std = self._train_network(
            x_device,
            observed,
            event,
            configuration,
            epochs=self.epochs,
            seed_offset=0,
            validation_data=validation_data,
        )
        with torch.no_grad():
            x_device, observed, event = (
                values[self._training_indices] for values in (x_device, observed, event)
            )
            fitted_risk = model((x_device - mean) / std)
            event_mask = event.bool()
            event_times = torch.unique(observed[event_mask], sorted=True)
            risk_shift = fitted_risk.max()
            relative_risk = torch.exp((fitted_risk - risk_shift).clamp(-80.0, 80.0))
            increments = torch.stack(
                [
                    ((observed == time) & event_mask).sum() /
                    relative_risk[observed >= time].sum()
                    for time in event_times
                ]
            )

        self._model = model
        self._feature_count = x_device.shape[1]
        self._input_mean = mean
        self._input_std = std
        self._baseline_times = event_times
        self._baseline_cumulative_hazard = torch.cumsum(increments, dim=0)
        self._risk_shift = risk_shift
        return self

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.device, torch.dtype]:
        if self._model is None:
            raise RuntimeError("fit must be called before prediction")
        if x.ndim != 2 or x.shape[1] != self._feature_count:
            raise ValueError("x has the wrong number of features")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("DeepSurv requires finite covariates")
        dtype = x.dtype if x.is_floating_point() else torch.float32
        return x.detach().to(self.device, dtype=torch.float32), x.device, dtype

    @torch.no_grad()
    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:
        values, output_device, output_dtype = self._features(x)
        assert self._model is not None
        assert self._input_mean is not None
        assert self._input_std is not None
        self._model.eval()
        risk = self._model((values - self._input_mean) / self._input_std)
        return risk.to(device=output_device, dtype=output_dtype)

    @torch.no_grad()
    def predict_survival(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        values, output_device, output_dtype = self._features(x)
        times_device = times.detach().to(self.device, dtype=torch.float32)
        if (
            times_device.ndim != 1
            or not bool(torch.isfinite(times_device).all())
            or bool((times_device < 0).any())
            or not bool((times_device[1:] >= times_device[:-1]).all())
        ):
            raise ValueError("times must be a sorted finite nonnegative vector")
        assert self._model is not None
        assert self._input_mean is not None
        assert self._input_std is not None
        assert self._baseline_times is not None
        assert self._baseline_cumulative_hazard is not None
        assert self._risk_shift is not None
        self._model.eval()
        risk = self._model((values - self._input_mean) / self._input_std)
        positions = torch.searchsorted(self._baseline_times, times_device, right=True) - 1
        baseline = torch.zeros_like(times_device)
        available = positions >= 0
        baseline[available] = self._baseline_cumulative_hazard[positions[available]]
        relative_risk = torch.exp((risk - self._risk_shift).clamp(-80.0, 80.0))
        survival = torch.exp(-relative_risk[:, None] * baseline[None, :])
        return survival.clamp(0.0, 1.0).to(output_device, output_dtype)

    def close(self) -> None:
        self._model = None

    def __enter__(self) -> "DeepSurvCUDA":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
