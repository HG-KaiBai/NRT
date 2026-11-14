"""Training configuration, NLL evaluation, and reusable CPU/CUDA helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import Observations


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 20  # epochs, not validation checks
    validation_every: int = 5
    min_delta: float = 1e-4
    max_grad_norm: float = 1.0
    cuda_graphs: bool = True


def _subset(data: Observations, indices: torch.Tensor) -> Observations:
    return Observations(*(value.index_select(0, indices) for value in data))


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
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
def _evaluate(model: torch.nn.Module, data: Observations, batch_size: int = 512) -> float:
    model.eval()
    total = data.x.new_zeros(())
    for start in range(0, len(data.x), batch_size):
        batch = Observations(*(value[start:start + batch_size] for value in data))
        total.add_(model.terms(batch)[0], alpha=len(batch.x))
    return float(total / len(data.x))


def evaluate_nll(model: torch.nn.Module, x: torch.Tensor,
                 left: torch.Tensor, right: torch.Tensor) -> float:
    return _evaluate(model, model.prepare(x, left, right))
