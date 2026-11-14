"""Joint optimization of one observed-data NLL, with validation-based early stopping."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .model import ConditionalTree, ModelConfig, Observations


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 20  # epochs, not validation checks
    validation_every: int = 5
    min_delta: float = 1e-4
    max_grad_norm: float = 1.0
    regularization_strength: float = 0.5
    cuda_graphs: bool = True


def _subset(data: Observations, indices: torch.Tensor) -> Observations:
    return Observations(*(value.index_select(0, indices) for value in data))


def _snapshot(model: ConditionalTree) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


class _CudaSteps:
    """Capture training batches without introducing extra optimization updates."""

    def __init__(self, model, optimizer, data, config, coefficient, totals):
        self.indices, self.graphs = {}, {}
        device = data.x.device
        sizes = sorted({min(len(data.x), config.batch_size),
                        len(data.x) % config.batch_size} - {0}, reverse=True)
        initial = _snapshot(model)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))

        def step(indices):
            optimizer.zero_grad(set_to_none=True)
            nll, regularizer = model.terms(_subset(data, indices))
            (nll + coefficient * regularizer).backward()
            if config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            totals.add_(torch.stack((nll.detach(), regularizer.detach())), alpha=len(indices))

        with torch.random.fork_rng(devices=[device.index or 0]):
            with torch.cuda.stream(stream):
                for size in sizes:
                    self.indices[size] = torch.arange(size, device=device)
                    for _ in range(3):
                        step(self.indices[size])
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            for size in sizes:
                optimizer.zero_grad(set_to_none=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    step(self.indices[size])
                self.graphs[size] = graph
            torch.cuda.current_stream(device).wait_stream(stream)
            model.load_state_dict(initial)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
            totals.zero_()

    def step(self, indices):
        self.indices[len(indices)].copy_(indices)
        self.graphs[len(indices)].replay()


@torch.no_grad()
def _evaluate(model: ConditionalTree, data: Observations, batch_size: int = 512) -> float:
    model.eval()
    total = data.x.new_zeros(())
    for start in range(0, len(data.x), batch_size):
        batch = Observations(*(value[start:start + batch_size] for value in data))
        total.add_(model.terms(batch)[0], alpha=len(batch.x))
    return float(total / len(data.x))


def evaluate_nll(model: ConditionalTree, x: torch.Tensor,
                 left: torch.Tensor, right: torch.Tensor) -> float:
    return _evaluate(model, model.prepare(x, left, right))


def fit(x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, *,
        validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        model_config: ModelConfig = ModelConfig(),
        training_config: TrainingConfig = TrainingConfig(),
        time_scale: float | None = None,
        device: str | torch.device | None = None,
        seed: int = 0) -> tuple[ConditionalTree, list[dict[str, float]]]:
    """Fit either fixed or learned splits using exactly the same observation loss.

    lambda_n = regularization_strength / sqrt(n_train), fixed throughout a fit.
    There is no parameter-dependent pseudo-target, time-grid training loss,
    oracle CDF supervision, or test-set model selection.
    """
    config = training_config
    if min(config.epochs, config.batch_size, config.patience, config.validation_every) < 1:
        raise ValueError("epochs, batch_size, patience and validation_every must be positive")
    if not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if any(not math.isfinite(v) or v < 0 for v in
           (config.regularization_strength, config.max_grad_norm, config.min_delta)):
        raise ValueError("regularization, clipping and min_delta must be finite and nonnegative")
    if len(x) == 0:
        raise ValueError("training data cannot be empty")
    if time_scale is None:
        observed = torch.cat((left[torch.isfinite(left)], right[torch.isfinite(right)]))
        positive = observed[observed > 0]
        time_scale = float(positive.median()) if len(positive) else 1.0
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = ConditionalTree(model_config, time_scale).to(device)
    train = model.prepare(x, left, right)
    valid = model.prepare(*validation)
    coefficient = config.regularization_strength / math.sqrt(len(x))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config.lr,
        fused=device.type == "cuda", capturable=device.type == "cuda" and config.cuda_graphs,
    )
    totals = train.x.new_zeros(2)
    graphs = (_CudaSteps(model, optimizer, train, config, coefficient, totals)
              if device.type == "cuda" and config.cuda_graphs else None)
    best_score = _evaluate(model, valid)
    if not math.isfinite(best_score):
        raise RuntimeError("initial validation NLL is not finite")
    best_state = _snapshot(model)
    best_epoch = 0
    patience_reference = best_score
    patience_epoch = 0
    history = [{"epoch": 0, "validation_nll": best_score, "lambda_n": coefficient}]
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals.zero_()
        for indices in torch.randperm(len(x), device=device).split(config.batch_size):
            if graphs is not None:
                graphs.step(indices)
            else:
                optimizer.zero_grad(set_to_none=True)
                nll, regularizer = model.terms(_subset(train, indices))
                (nll + coefficient * regularizer).backward()
                if config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                totals.add_(torch.stack((nll.detach(), regularizer.detach())), alpha=len(indices))
        train_nll, train_regularizer = (totals / len(x)).tolist()
        if not all(math.isfinite(v) for v in (train_nll, train_regularizer)):
            raise RuntimeError(f"nonfinite training objective at epoch {epoch}")
        row = {"epoch": epoch, "train_nll": train_nll, "regularizer": train_regularizer,
               "objective": train_nll + coefficient * train_regularizer,
               "lambda_n": coefficient}
        if epoch % config.validation_every == 0 or epoch == config.epochs:
            score = _evaluate(model, valid)
            if not math.isfinite(score):
                raise RuntimeError(f"nonfinite validation NLL at epoch {epoch}")
            row["validation_nll"] = score
            if score < best_score:
                best_score, best_state, best_epoch = score, _snapshot(model), epoch
            if score < patience_reference - config.min_delta:
                patience_reference, patience_epoch = score, epoch
        history.append(row)
        if "validation_nll" in row and epoch - patience_epoch >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    model.fit_info = {"best_epoch": best_epoch, "epochs_run": epoch,
                      "validation_nll": best_score, "lambda_n": coefficient,
                      "time_scale": float(model.time_scale)}
    return model, history
