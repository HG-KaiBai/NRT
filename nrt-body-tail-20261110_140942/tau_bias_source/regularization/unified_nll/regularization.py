"""Observation-independent partition regularization and paired training.

The original three-method experiment remains reproducible: its source files
are unchanged. This extension uses the same training steps and stopping rule.
"""

from __future__ import annotations

from dataclasses import replace
import math

import torch

from .model import ConditionalTree, ModelConfig, Observations
from .train import TrainingConfig, _CudaSteps, _evaluate, _snapshot, _subset
from .tree import TreeState


def partition_penalty(state: TreeState, depth: int) -> torch.Tensor:
    r"""2/D * KL(Uniform over leaves || leaf widths), independently of mass.

    R = -2/(D*K) sum_k log(K * width_k), K=2**D.

    With effective split fractions r_v this is also the average, over levels,
    of the level-wise mean -log(4*r_v*(1-r_v)). It is zero for equal widths,
    nonnegative otherwise, and increases when any interval collapses. Changing
    masses with the partition held fixed has no effect on the penalty.
    """
    if depth < 1 or state.width.shape[-1] != 2**depth:
        raise ValueError("depth must match the number of leaves")
    return -2 / depth * (state.width.log() + depth * math.log(2)).mean(dim=-1)


class RegularizedTree(ConditionalTree):
    def __init__(self, config: ModelConfig, time_scale: float,
                 penalty: str = "partition_kl") -> None:
        super().__init__(config, time_scale)
        if penalty not in ("none", "partition_kl", "density_kl"):
            raise ValueError("unsupported regularizer")
        self.penalty = penalty

    def terms(self, observations: Observations) -> tuple[torch.Tensor, torch.Tensor]:
        state = self(observations.x)
        nll = -self.tree.loglik(state, *observations[1:]).mean()
        if self.penalty == "partition_kl":
            regularizer = partition_penalty(state, self.config.depth).mean()
        elif self.penalty == "density_kl":
            regularizer = self.tree.regularizer(state).mean()
        else:
            regularizer = state.width.new_zeros(())
        return nll, regularizer


def depth_model_config(base: ModelConfig, depth: int,
                       total_width_floor: float = 8e-4) -> ModelConfig:
    """Hold K*minimum_leaf_width constant when changing the tree depth."""
    if not 0 < total_width_floor < 1:
        raise ValueError("total_width_floor must be in (0,1)")
    return replace(base, depth=depth, min_leaf_width=total_width_floor / 2**depth)


def fit_regularized(x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, *,
                    validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    model_config: ModelConfig = ModelConfig(),
                    training_config: TrainingConfig = TrainingConfig(),
                    penalty: str = "partition_kl", time_scale: float | None = None,
                    device: str | torch.device | None = None, seed: int = 0):
    """Same observed NLL and optimizer for fixed, learned and penalized splits.

    lambda_n=a/sqrt(n_train). Checkpoints use unpenalized validation NLL.
    No censoring-type label or distribution-family label enters this function.
    """
    config = training_config
    if min(config.epochs, config.batch_size, config.patience, config.validation_every) < 1:
        raise ValueError("training counts must be positive")
    if not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if any(not math.isfinite(v) or v < 0 for v in
           (config.regularization_strength, config.max_grad_norm, config.min_delta)):
        raise ValueError("regularization, clipping and min_delta must be nonnegative")
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
    model = RegularizedTree(model_config, time_scale, penalty).to(device)
    train, valid = model.prepare(x, left, right), model.prepare(*validation)
    coefficient = config.regularization_strength / math.sqrt(len(x))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
        lr=config.lr, fused=device.type == "cuda",
        capturable=device.type == "cuda" and config.cuda_graphs)
    totals = train.x.new_zeros(2)
    graphs = (_CudaSteps(model, optimizer, train, config, coefficient, totals)
              if device.type == "cuda" and config.cuda_graphs else None)
    best_score = _evaluate(model, valid)
    if not math.isfinite(best_score):
        raise RuntimeError("initial validation NLL is not finite")
    best_state, best_epoch = _snapshot(model), 0
    patience_reference, patience_epoch = best_score, 0
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
        row = dict(epoch=epoch, train_nll=train_nll, regularizer=train_regularizer,
                   objective=train_nll + coefficient * train_regularizer, lambda_n=coefficient)
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
    model.fit_info = dict(best_epoch=best_epoch, epochs_run=epoch,
        validation_nll=best_score, lambda_n=coefficient, time_scale=float(model.time_scale))
    return model, history
