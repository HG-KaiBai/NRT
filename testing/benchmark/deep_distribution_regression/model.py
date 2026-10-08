"""Deep Distribution Regression with the paper's JBCE objective."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from ...validation import training_validation_split


class _DDRNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_nodes: int, n_bins: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(3):
            layers.extend(
                (
                    nn.Linear(current, hidden_nodes),
                    nn.ELU(),
                    nn.BatchNorm1d(hidden_nodes),
                    nn.Dropout(dropout),
                )
            )
            current = hidden_nodes
        layers.append(nn.Linear(current, n_bins))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(x), dim=-1)


class DeepDistributionRegression:
    """Fixed-partition JBCE neural estimator from Li et al. (2021)."""

    supported_censor_types = frozenset({"exact"})

    def __init__(
        self,
        *,
        cut_fraction: float = 0.10,
        n_cutpoints: int | None = None,
        hidden_nodes: int = 100,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 20,
        validation_fraction: float = 0.20,
        support_margin: float = 0.05,
        seed: int = 123,
        device: str | torch.device | None = None,
    ) -> None:
        if not 0.0 < cut_fraction < 1.0:
            raise ValueError("cut_fraction must be in (0, 1)")
        if n_cutpoints is not None and n_cutpoints < 1:
            raise ValueError("n_cutpoints must be positive")
        if validation_fraction != 0.20:
            raise ValueError("validation_fraction must be 0.20 for the 80/20 training split")
        self.cut_fraction = cut_fraction
        self.n_cutpoints = n_cutpoints
        self.hidden_nodes = hidden_nodes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.support_margin = support_margin
        self.seed = seed
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network: _DDRNetwork | None = None
        self.x_mean: torch.Tensor | None = None
        self.x_std: torch.Tensor | None = None
        self.bin_edges: torch.Tensor | None = None

    @staticmethod
    def _exact_response(L: torch.Tensor, R: torch.Tensor, censor_type: str) -> torch.Tensor:
        if censor_type != "exact":
            raise ValueError("Deep Distribution Regression supports uncensored (exact) data only")
        if not bool((torch.isfinite(L) & torch.isfinite(R) & torch.isclose(L, R)).all()):
            raise ValueError("exact data require finite equal L and R bounds")
        return L

    def _jbce(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert self.network is not None and self.bin_edges is not None
        bin_probabilities = self.network(x)
        cdf_at_cuts = bin_probabilities.cumsum(dim=1)[:, :-1]
        targets = (y.unsqueeze(1) <= self.bin_edges[1:-1].unsqueeze(0)).to(x.dtype)
        return F.binary_cross_entropy(cdf_at_cuts.clamp(1e-7, 1.0 - 1e-7), targets)

    def fit(
        self,
        x: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        *,
        censor_type: str,
        validation_data=None,
    ) -> "DeepDistributionRegression":
        x = x.detach().to(self.device, dtype=torch.float32)
        y = self._exact_response(
            L.detach().to(self.device, dtype=torch.float32),
            R.detach().to(self.device, dtype=torch.float32),
            censor_type,
        )
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.numel():
            raise ValueError("x must be 2-D and bounds must have one value per row")
        if x.shape[0] < 4:
            raise ValueError("at least four training rows are required")

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        if validation_data is None:
            training_indices, validation_indices = training_validation_split(
                len(x), seed=self.seed, device=self.device,
            )
        else:
            valid_x, valid_L, valid_R = (
                values.detach().to(self.device, dtype=torch.float32) for values in validation_data
            )
            valid_y = self._exact_response(valid_L, valid_R, censor_type)
            training_indices = torch.arange(len(x), device=self.device)
            validation_indices = torch.arange(len(x), len(x) + len(valid_x), device=self.device)
            x, y = torch.cat((x, valid_x)), torch.cat((y, valid_y))
        if not validation_indices.numel():
            raise ValueError("the validation dataset cannot be empty")
        if training_indices.numel() < 2:
            raise ValueError("validation split leaves too few training rows")

        self.x_mean = x[training_indices].mean(dim=0)
        self.x_std = x[training_indices].std(dim=0, correction=0).clamp_min(1e-6)
        scaled_x = (x - self.x_mean) / self.x_std

        y_min, y_max = y[training_indices].min(), y[training_indices].max()
        y_range = (y_max - y_min).clamp_min(1e-6)
        lower = torch.minimum(y_min - self.support_margin * y_range, y.new_tensor(0.0))
        upper = y_max + self.support_margin * y_range
        requested = self.n_cutpoints or max(1, round(self.cut_fraction * training_indices.numel()))
        requested = min(requested, max(1, training_indices.numel() - 1))
        candidate_cuts = torch.linspace(lower, upper, requested + 2, device=self.device)[1:-1]

        # Match the reference implementation's merge_empty_bin option.
        occupied = torch.bucketize(y[training_indices], candidate_cuts)
        adjacent = torch.unique(
            torch.cat((occupied - 1, occupied)).clamp(0, candidate_cuts.numel() - 1)
        )
        cuts = candidate_cuts[adjacent]
        self.bin_edges = torch.cat((lower.view(1), cuts, upper.view(1)))

        torch.manual_seed(self.seed)
        self.network = _DDRNetwork(
            x.shape[1], self.hidden_nodes, self.bin_edges.numel() - 1, self.dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=7
        )
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0

        for _epoch in range(self.epochs):
            self.network.train()
            shuffled = training_indices[
                torch.randperm(training_indices.numel(), generator=generator).to(self.device)
            ]
            for start in range(0, shuffled.numel(), self.batch_size):
                indices = shuffled[start : start + self.batch_size]
                if indices.numel() == 1:
                    continue  # BatchNorm needs at least two observations while training.
                optimizer.zero_grad(set_to_none=True)
                loss = self._jbce(scaled_x[indices], y[indices])
                loss.backward()
                optimizer.step()

            self.network.eval()
            with torch.no_grad():
                validation_loss = float(
                    self._jbce(scaled_x[validation_indices], y[validation_indices]).item()
                )
            scheduler.step(validation_loss)
            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.network.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    @torch.no_grad()
    def predict_survival(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if (
            self.network is None
            or self.x_mean is None
            or self.x_std is None
            or self.bin_edges is None
        ):
            raise RuntimeError("fit must be called before predict_survival")
        output_device, output_dtype = x.device, x.dtype
        x = x.detach().to(self.device, dtype=torch.float32)
        times = times.detach().to(self.device, dtype=torch.float32)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        probabilities = self.network((x - self.x_mean) / self.x_std)
        cumulative_edges = torch.cat(
            (
                probabilities.new_zeros(probabilities.shape[0], 1),
                probabilities.cumsum(dim=1),
            ),
            dim=1,
        )
        indices = torch.bucketize(times, self.bin_edges).clamp(1, self.bin_edges.numel() - 1)
        left_index = indices - 1
        left = self.bin_edges[left_index]
        right = self.bin_edges[indices]
        fraction = ((times - left) / (right - left).clamp_min(1e-12)).clamp(0.0, 1.0)
        cdf = cumulative_edges[:, left_index] + fraction.unsqueeze(0) * (
            cumulative_edges[:, indices] - cumulative_edges[:, left_index]
        )
        cdf = torch.where(times.unsqueeze(0) < self.bin_edges[0], torch.zeros_like(cdf), cdf)
        cdf = torch.where(times.unsqueeze(0) >= self.bin_edges[-1], torch.ones_like(cdf), cdf)
        return (1.0 - cdf).clamp(0.0, 1.0).to(output_device, output_dtype)
