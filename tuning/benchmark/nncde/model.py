"""PyTorch implementation of Hu and Nan's likelihood-based NNCDE."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from ...validation import training_validation_split


class _LogHazardNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_nodes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


@dataclass
class _FittedNetwork:
    network: _LogHazardNetwork
    input_mean: torch.Tensor
    input_std: torch.Tensor


class NNCDE:
    """Conditional survival estimator from Hu and Nan (2023).

    The paper's full-likelihood construction observes either an exact event or
    a right-censoring time. Interval- and current-status bounds are therefore
    rejected instead of being silently converted to point observations.
    """

    supported_censor_types = frozenset({"exact", "right"})

    def __init__(
        self,
        *,
        hidden_nodes: int = 64,
        learning_rate: float = 1e-3,
        batch_size: int = 100,
        epochs: int = 200,
        patience: int = 10,
        seed: int = 123,
        device: str | torch.device | None = None,
    ) -> None:
        if hidden_nodes < 1 or batch_size < 1 or epochs < 1 or patience < 1:
            raise ValueError("hidden_nodes, batch_size, epochs, and patience must be positive")
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._models: list[_FittedNetwork] = []

    @staticmethod
    def _outcomes(
        L: torch.Tensor, R: torch.Tensor, censor_type: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if censor_type not in NNCDE.supported_censor_types:
            raise ValueError("NNCDE supports only exact and right-censored data")
        point = torch.isfinite(R) & torch.isclose(L, R)
        if censor_type == "exact" and not bool(point.all()):
            raise ValueError("exact data require finite equal L and R bounds")
        if censor_type == "right" and not bool((point | torch.isinf(R)).all()):
            raise ValueError("right-censored data require L=R events or R=inf censoring")
        return L, point.to(L.dtype)

    @staticmethod
    def _time_grid(observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stops = torch.unique(observed, sorted=True)
        stops = stops[stops > 0]
        if stops.numel() == 0:
            raise ValueError("at least one positive observed time is required")
        starts = torch.cat((stops.new_zeros(1), stops[:-1]))
        return starts, stops

    @staticmethod
    def _expanded_inputs(
        x: torch.Tensor, observed: torch.Tensor
    ) -> torch.Tensor:
        starts, stops = NNCDE._time_grid(observed)
        mask = stops.unsqueeze(0) <= observed.unsqueeze(1)
        n, m = mask.shape
        covariates = x.unsqueeze(1).expand(n, m, x.shape[1])
        start_values = starts.view(1, m, 1).expand(n, m, 1)
        stop_values = stops.view(1, m, 1).expand(n, m, 1)
        return torch.cat((covariates, start_values, stop_values), dim=-1)[mask]

    def _loss(
        self,
        fitted: _FittedNetwork,
        x: torch.Tensor,
        observed: torch.Tensor,
        event: torch.Tensor,
        starts: torch.Tensor | None = None,
        stops: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if starts is None or stops is None:
            starts, stops = self._time_grid(observed)
        n, m = x.shape[0], stops.numel()
        values = torch.cat(
            (
                x.unsqueeze(1).expand(n, m, x.shape[1]),
                starts.view(1, m, 1).expand(n, m, 1),
                stops.view(1, m, 1).expand(n, m, 1),
            ),
            dim=-1,
        )
        values = (values - fitted.input_mean) / fitted.input_std
        log_hazard = fitted.network(values)
        active = stops.unsqueeze(0) <= observed.unsqueeze(1)
        widths = stops - starts
        cumulative_hazard = (
            torch.exp(log_hazard.clamp(max=30.0))
            * widths.unsqueeze(0)
            * active
        ).sum(dim=1)
        final_index = active.sum(dim=1).clamp_min(1) - 1
        event_log_hazard = log_hazard.gather(1, final_index.unsqueeze(1)).squeeze(1)
        return (cumulative_hazard - event * event_log_hazard).mean()

    def _train_one(
        self,
        x_train: torch.Tensor,
        observed_train: torch.Tensor,
        event_train: torch.Tensor,
        seed: int,
        validation_data=None,
    ) -> _FittedNetwork:
        if validation_data is None:
            training, validation = training_validation_split(len(x_train), seed=seed, device=self.device)
            x_valid, observed_valid, event_valid = (
                values[validation] for values in (x_train, observed_train, event_train)
            )
            x_train, observed_train, event_train = (
                values[training] for values in (x_train, observed_train, event_train)
            )
        else:
            x_valid, observed_valid, event_valid = validation_data
        if len(x_valid) < 1:
            raise ValueError("the validation dataset cannot be empty")
        torch.manual_seed(seed)
        expanded = self._expanded_inputs(x_train, observed_train)
        input_mean = expanded.mean(dim=0)
        input_std = expanded.std(dim=0, correction=0).clamp_min(1e-6)
        fitted = _FittedNetwork(
            network=_LogHazardNetwork(x_train.shape[1] + 2, self.hidden_nodes).to(self.device),
            input_mean=input_mean,
            input_std=input_std,
        )
        optimizer = torch.optim.Adam(fitted.network.parameters(), lr=self.learning_rate)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        train_starts, train_stops = self._time_grid(observed_train)
        valid_starts, valid_stops = self._time_grid(observed_valid)
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0

        for _epoch in range(self.epochs):
            fitted.network.train()
            order = torch.randperm(x_train.shape[0], generator=generator, device="cpu")
            for start in range(0, order.numel(), self.batch_size):
                indices = order[start : start + self.batch_size].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(
                    fitted,
                    x_train[indices],
                    observed_train[indices],
                    event_train[indices],
                    train_starts,
                    train_stops,
                )
                loss.backward()
                optimizer.step()

            fitted.network.eval()
            with torch.no_grad():
                validation_loss = float(
                    self._loss(
                        fitted,
                        x_valid,
                        observed_valid,
                        event_valid,
                        valid_starts,
                        valid_stops,
                    ).item()
                )
            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_state = copy.deepcopy(fitted.network.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            fitted.network.load_state_dict(best_state)
        fitted.network.eval()
        return fitted

    def fit(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        *,
        censor_type: str,
        validation_data=None,
    ) -> "NNCDE":
        """Cross-fit two networks, using the CV fold or an 80/20 split within each half."""
        x = x.detach().to(self.device, dtype=torch.float32)
        L = L.detach().to(self.device, dtype=torch.float32)
        R = R.detach().to(self.device, dtype=torch.float32)
        observed, event = self._outcomes(L, R, censor_type)
        if validation_data is not None:
            valid_x, valid_L, valid_R = (
                values.detach().to(self.device, dtype=torch.float32) for values in validation_data
            )
            valid_observed, valid_event = self._outcomes(valid_L, valid_R, censor_type)
            validation_data = valid_x, valid_observed, valid_event
        if x.ndim != 2 or observed.ndim != 1 or x.shape[0] != observed.numel():
            raise ValueError("x must be 2-D and bounds must have one value per row")
        if x.shape[0] < 4:
            raise ValueError("NNCDE needs at least four training rows for cross-fitting")

        permutation = torch.randperm(
            x.shape[0], generator=torch.Generator().manual_seed(self.seed)
        ).to(self.device)
        split = x.shape[0] // 2
        first, second = permutation[:split], permutation[split:]
        self._models = [
            self._train_one(
                x[first], observed[first], event[first],
                self.seed,
                validation_data,
            ),
            self._train_one(
                x[second], observed[second], event[second],
                self.seed + 1,
                validation_data,
            ),
        ]
        return self

    @torch.no_grad()
    def predict_survival(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if not self._models:
            raise RuntimeError("fit must be called before predict_survival")
        output_device, output_dtype = x.device, x.dtype
        x = x.detach().to(self.device, dtype=torch.float32)
        times = times.detach().to(self.device, dtype=torch.float32)
        if times.ndim != 1 or not bool((times[1:] >= times[:-1]).all()):
            raise ValueError("times must be a sorted one-dimensional tensor")
        starts = torch.cat((times.new_zeros(1), times[:-1]))
        widths = (times - starts).clamp_min(0.0)
        n, m = x.shape[0], times.numel()
        raw = torch.cat(
            (
                x.unsqueeze(1).expand(n, m, x.shape[1]),
                starts.view(1, m, 1).expand(n, m, 1),
                times.view(1, m, 1).expand(n, m, 1),
            ),
            dim=-1,
        )
        curves = []
        for fitted in self._models:
            values = (raw - fitted.input_mean) / fitted.input_std
            hazard = torch.exp(fitted.network(values).clamp(max=30.0))
            curves.append(torch.exp(-torch.cumsum(hazard * widths.unsqueeze(0), dim=1)))
        survival = torch.stack(curves).mean(dim=0).clamp(0.0, 1.0)
        return survival.to(device=output_device, dtype=output_dtype)
