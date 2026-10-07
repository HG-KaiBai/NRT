"""Training and evaluation for deterministic neural conditional trees."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:  # Support package imports and direct script execution.
    from .mlp import ConditionalTree
except ImportError:  # pragma: no cover
    from mlp import ConditionalTree


@dataclass
class ModelConfig:
    depth: int = 6
    x_dim: int = 6
    h_dim: int = 8
    mlp_hidden: int = 16
    mlp_layers: int = 2
    dropout: float = 0.0
    split_min: float = 0.03
    shared_layers: int = 0


@dataclass
class TrainingConfig:
    epochs: int = 200
    training_batch_size: int = 256
    lr: float = 5e-3
    weight_decay: float = 1e-4
    patience: int = 20
    brier_grid_size: int = 256
    save_path: str = ""
    training_method: str = "alternating"
    cuda_graphs: bool = True
    max_grad_norm: float = 1.0

    @property
    def monitor(self) -> str:
        if self.training_method == "alternating":
            return "rho_brier"
        return self.training_method.removeprefix("joint_")

    @property
    def steps(self) -> tuple[tuple[str, str], ...]:
        if self.training_method == "alternating":
            return (("m", "nll"), ("rho", "brier"))
        return (("both", self.training_method.removeprefix("joint_")),)


class IntervalDataset(Dataset):
    def __init__(self, x: torch.Tensor, L: torch.Tensor, R: torch.Tensor) -> None:
        if x.size(0) != L.size(0) or L.shape != R.shape:
            raise ValueError("x, L, and R must have matching batch dimensions")
        self.x = x.float()
        self.L = L.float()
        self.R = R.float()

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.L[index], self.R[index]


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


class _CudaBatchGraphs:
    """Replay a complete minibatch, including sequential coordinate updates.

    Each actual batch size has its own graph, so the final partial batch is
    neither padded nor dropped. Parameter and optimizer storage stays fixed.
    """

    def __init__(self, model, optimizer, tensors, config, totals):
        self.graphs = {}
        self.indices = {}
        self.totals = totals
        device = tensors[0].device
        count = tensors[0].size(0)
        sizes = sorted({min(count, config.training_batch_size),
                        count % config.training_batch_size} - {0}, reverse=True)
        initial_state = _clone_state_dict(model)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))

        def step(indices):
            xb, lb, rb = (values.index_select(0, indices) for values in tensors)
            for update, loss_name in config.steps:
                model.encoder.train(update != "rho")
                model.encoder_split.train(update != "m")
                optimizer.zero_grad(set_to_none=True)
                loss = model.objective(
                    xb, lb, rb, loss=loss_name, grid_size=config.brier_grid_size,
                    update=update,
                )
                loss.backward()
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                key = loss_name if update == "both" else f"{update}_{loss_name}"
                totals[key].add_(loss.detach(), alpha=indices.numel())

        # Warmup initializes AdamW state and CUDA libraries on a side stream.
        # Restore weights, moments, counters, and RNG afterward: setup must not
        # introduce extra training updates or consume shuffle/dropout randomness.
        with torch.random.fork_rng(devices=[device.index or 0]):
            with torch.cuda.stream(stream):
                for size in sizes:
                    indices = torch.arange(size, device=device)
                    self.indices[size] = indices
                    for _ in range(3):
                        step(indices)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            for size in sizes:
                optimizer.zero_grad(set_to_none=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    step(self.indices[size])
                self.graphs[size] = graph
            torch.cuda.current_stream(device).wait_stream(stream)
            model.load_state_dict(initial_state)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
            for total in totals.values():
                total.zero_()
        model.train()

    def replay(self, indices):
        self.indices[indices.numel()].copy_(indices)
        self.graphs[indices.numel()].replay()


def _evaluation_batches(dataset: IntervalDataset, device: str | torch.device):
    tensors = tuple(values.to(device) for values in (dataset.x, dataset.L, dataset.R))
    batch_size = 256 if tensors[0].is_cuda else 32
    return zip(*(values.split(batch_size) for values in tensors))


@torch.no_grad()
def evaluate_nll(
    model: ConditionalTree,
    x: torch.Tensor,
    L: torch.Tensor,
    R: torch.Tensor,
    device: str | torch.device,
) -> float:
    """Evaluate full observed-data NLL, including exact-observation density."""
    dataset = IntervalDataset(x, L, R)
    if len(dataset) < 1:
        raise ValueError("the evaluation dataset cannot be empty")
    model.eval()
    total = torch.zeros((), device=device)
    for xb, lb, rb in _evaluation_batches(dataset, device):
        score = model.nll(xb, lb, rb)
        total.add_(score, alpha=xb.size(0))
    return (total / len(dataset)).item()


@torch.no_grad()
def evaluate_brier(
    model: ConditionalTree,
    x: torch.Tensor,
    L: torch.Tensor,
    R: torch.Tensor,
    grid_size: int,
    device: str | torch.device,
) -> float:
    """Evaluate integrated Brier score from point survival estimates."""
    dataset = IntervalDataset(x, L, R)
    if len(dataset) < 1:
        raise ValueError("the evaluation dataset cannot be empty")
    model.eval()
    total_brier = torch.zeros((), device=device)
    for x_batch, L_batch, R_batch in _evaluation_batches(dataset, device):
        score = model.brier_score(
            x_batch,
            L_batch,
            R_batch,
            grid_size=grid_size,
            prediction_batch_size=None,
        )
        total_brier.add_(score, alpha=x_batch.size(0))
    return (total_brier / len(dataset)).item()


def fit(
    x: torch.Tensor,
    L: torch.Tensor,
    R: torch.Tensor,
    max_bound: float,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> tuple[ConditionalTree, list[dict[str, float]]]:
    """Fit with joint NLL, joint Brier, or alternating m-NLL then rho-Brier.

    Joint modes monitor their epoch-average loss. Alternating mode monitors
    rho-Brier, recomputing predictions after each m-NLL update.
    CUDA graphs reduce launch overhead without changing minibatch sizes or
    objectives. Set ``cuda_graphs=False`` to use eager CUDA execution.
    """
    if training_config.training_method not in ("joint_nll", "joint_brier", "alternating"):
        raise ValueError("training_method must be 'joint_nll', 'joint_brier', or 'alternating'")
    if training_config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if training_config.training_batch_size < 1:
        raise ValueError("training_batch_size must be at least 1")
    if training_config.patience < 1:
        raise ValueError("patience must be at least 1")
    if training_config.brier_grid_size < 2:
        raise ValueError("brier_grid_size must be at least 2")
    if not math.isfinite(training_config.max_grad_norm) or training_config.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be finite and non-negative (0 disables clipping)")
    if len(x) < 1:
        raise ValueError("the training dataset cannot be empty")

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = IntervalDataset(x, L, R)
    if resolved_device.type == "cuda":
        # Keep training tensors on the GPU; avoid per-row collation and a
        # host-to-device copy for every minibatch.
        train_x, train_L, train_R = (
            values.to(resolved_device) for values in (dataset.x, dataset.L, dataset.R)
        )
        loader = None
    else:
        loader = DataLoader(
            dataset, batch_size=training_config.training_batch_size,
            shuffle=True, drop_last=False,
        )
    finite_bounds = torch.cat((L[torch.isfinite(L)], R[torch.isfinite(R)]))
    max_observation = finite_bounds.max().item()
    model = ConditionalTree(
        **asdict(model_config), max_bound=max_bound, max_observation=max_observation,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
        fused=resolved_device.type == "cuda",
        capturable=resolved_device.type == "cuda" and training_config.cuda_graphs,
    )

    totals = {
        (loss if update == "both" else f"{update}_{loss}"): torch.zeros((), device=resolved_device)
        for update, loss in training_config.steps
    }
    graph_batches = None
    if resolved_device.type == "cuda" and training_config.cuda_graphs:
        graph_batches = _CudaBatchGraphs(
            model, optimizer, (train_x, train_L, train_R), training_config, totals,
        )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        for total in totals.values():
            total.zero_()
        examples_seen = 0
        if loader is None:
            order = torch.randperm(len(dataset), device=resolved_device)
            if graph_batches is not None:
                for indices in order.split(training_config.training_batch_size):
                    graph_batches.replay(indices)
                    examples_seen += indices.numel()
                batches = ()
            else:
                batches = (
                    (train_x[indices], train_L[indices], train_R[indices])
                    for indices in order.split(training_config.training_batch_size)
                )
        else:
            batches = loader
        for x_batch, L_batch, R_batch in batches:
            x_batch = x_batch.to(resolved_device)
            L_batch = L_batch.to(resolved_device)
            R_batch = R_batch.to(resolved_device)
            batch_size = x_batch.size(0)
            for update, loss_name in training_config.steps:
                # Shared layers train in both steps. Disable dropout on the
                # inactive private branch. Its detached parameters
                # have grad=None, so AdamW skips momentum and weight decay too.
                model.encoder.train(update != "rho")
                model.encoder_split.train(update != "m")
                optimizer.zero_grad(set_to_none=True)
                loss = model.objective(
                    x_batch, L_batch, R_batch, loss=loss_name,
                    grid_size=training_config.brier_grid_size, update=update,
                )
                if resolved_device.type != "cuda" and not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite {loss_name} loss at epoch {epoch}")
                loss.backward()
                if training_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), training_config.max_grad_norm,
                    )
                optimizer.step()
                key = loss_name if update == "both" else f"{update}_{loss_name}"
                totals[key].add_(loss.detach(), alpha=batch_size)
            examples_seen += batch_size

        # One GPU-to-host transfer per epoch, rather than synchronizing after
        # each loss and optimizer step. Non-finite GPU losses fail here.
        epoch_losses = (torch.stack(list(totals.values())) / examples_seen).tolist()
        metrics = {"epoch": float(epoch), **dict(zip(totals, epoch_losses))}
        for key, value in zip(totals, epoch_losses):
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {key} loss at epoch {epoch}")
        history.append(metrics)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == training_config.epochs):
            print(f"Epoch {epoch:03d} | " + " | ".join(
                f"{key}: {metrics[key]:.4f}" for key in totals
            ))

        if metrics[training_config.monitor] < best_loss:
            best_loss = metrics[training_config.monitor]
            best_state = _clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch}: {training_config.monitor} did not "
                        f"improve for {training_config.patience} epochs"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if training_config.save_path:
        save_path = Path(training_config.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model, history
