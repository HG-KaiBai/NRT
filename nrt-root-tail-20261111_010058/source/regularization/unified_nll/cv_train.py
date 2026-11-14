"""Adaptive-rho CV fits and full-pool refits, without a test stopping set."""
from dataclasses import dataclass
import math

import torch

from .brier import score_brier
from .train import TrainingConfig, _CudaSteps, _evaluate, _snapshot, _subset
from .tuning_model import Architecture, TuningTree


@dataclass(frozen=True)
class CVOptimization:
    split_lr_ratio: float = .1
    warmup_epochs: int = 10


class CVTree(TuningTree):
    """Every model has trainable split weights; there is no fixed-rho option."""
    def __init__(self, architecture, time_scale, optimization):
        super().__init__(architecture, time_scale)
        self.register_buffer("split_gate", torch.tensor(float(optimization.warmup_epochs == 0)))

    def rho(self, x):
        raw = self.split_head(self.split_encoder(self.shared(x)))
        return self.config.split_min + (1-2*self.config.split_min)*torch.sigmoid(self.split_gate*raw)

    def forward(self, x):
        features = self.shared(x)
        logits = self.mass_head(self.mass_encoder(features))
        raw = self.split_head(self.split_encoder(features))
        rho = self.config.split_min + (1-2*self.config.split_min)*torch.sigmoid(self.split_gate*raw)
        return self.tree(logits, rho)

    def terms(self, observations):
        state = self(observations.x)
        # Same geometric penalty as before. Subtract the endpoints in float64
        # before taking logs: large finite strengths amplify float32 cancellation
        # around the uniform partition. Mass/NN calculations remain float32.
        width = state.upper.double()-state.lower.double()
        reg = (-2/self.config.depth)*torch.log(width*self.tree.n_leaves).mean()
        return -self.tree.loglik(state, *observations[1:]).mean(), reg


def fit_cv(train, valid=None, grid=None, *, architecture=Architecture(),
           training=TrainingConfig(patience=50), optimization=CVOptimization(),
           time_scale=1., seed=101, device="cuda", fixed_epochs=None):
    """NLL-only stopping on folds; final refits take a frozen epoch count.

    Warmup checkpoints cannot win: rho must have received updates before a
    candidate is scored. The final refit never accepts or accesses test data.
    Brier is evaluated once, after restoring a fold's NLL checkpoint.
    """
    if (optimization.split_lr_ratio <= 0 or not math.isfinite(optimization.split_lr_ratio)
            or optimization.warmup_epochs < 0 or min(training.epochs, training.batch_size,
                training.validation_every, training.patience) < 1):
        raise ValueError("invalid training configuration")
    if (fixed_epochs is None) != (valid is not None):
        raise ValueError("supply validation for CV, or fixed_epochs for a full-pool refit")
    epochs = training.epochs if fixed_epochs is None else int(fixed_epochs)
    if epochs <= optimization.warmup_epochs or (valid is not None and grid is None):
        raise ValueError("need a learned-rho phase and a validation grid")
    torch.manual_seed(seed)
    model = CVTree(architecture, time_scale, optimization).to(device)
    prepared = model.prepare(*train)
    validation = model.prepare(*valid) if valid is not None else None
    split = list(model.split_encoder.parameters())+list(model.split_head.parameters())
    split_ids = {id(p) for p in split}
    optimizer = torch.optim.Adam([
        dict(params=[p for p in model.parameters() if id(p) not in split_ids], lr=training.lr),
        dict(params=split, lr=training.lr*optimization.split_lr_ratio)],
        fused=str(device).startswith("cuda"),
        capturable=str(device).startswith("cuda") and training.cuda_graphs)
    coefficient = architecture.regularization_strength/math.sqrt(len(train[0]))
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    graphs = (_CudaSteps(model, optimizer, prepared, training, coefficient, totals)
              if str(device).startswith("cuda") and training.cuda_graphs else None)
    best, reference, best_epoch, patience_epoch = float("inf"), float("inf"), None, 0
    snapshot = None
    history = []
    for epoch in range(1, epochs+1):
        if epoch == optimization.warmup_epochs+1:
            model.split_gate.fill_(1.)
            for p in split:
                for value in optimizer.state.get(p, {}).values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
        model.train()
        totals.zero_()
        for indices in torch.randperm(len(train[0]), device=device).split(training.batch_size):
            if graphs is not None:
                graphs.step(indices)
            else:
                optimizer.zero_grad(set_to_none=True)
                nll, reg = model.terms(_subset(prepared, indices))
                (nll+coefficient*reg).backward()
                if training.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm)
                optimizer.step()
                totals.add_(torch.stack((nll.detach(), reg.detach())), alpha=len(indices))
        nll, reg = (totals/len(train[0])).tolist()
        if not math.isfinite(nll+reg):
            raise RuntimeError("nonfinite training objective")
        row = dict(epoch=epoch, train_nll=nll, regularizer=reg,
                   objective=nll+coefficient*reg, split_active=epoch > optimization.warmup_epochs)
        if validation is not None and (epoch % training.validation_every == 0 or epoch == epochs):
            score = _evaluate(model, validation)
            if not math.isfinite(score):
                raise RuntimeError("nonfinite validation NLL")
            row["validation_nll"] = score
            if row["split_active"]:
                if score < best:
                    best, best_epoch, snapshot = score, epoch, _snapshot(model)
                if score < reference-training.min_delta:
                    reference, patience_epoch = score, epoch
        history.append(row)
        if (validation is not None and "validation_nll" in row and row["split_active"]
                and epoch-patience_epoch >= training.patience):
            break
    if validation is not None:
        if snapshot is None:
            raise RuntimeError("no learned-rho checkpoint evaluated")
        model.load_state_dict(snapshot)
    model.eval()
    info = dict(best_epoch=best_epoch if validation is not None else epochs,
        epochs_run=epoch, lambda_n=coefficient, n_fit=len(train[0]),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        learn_splits=True, split_lr_ratio=optimization.split_lr_ratio,
        warmup_epochs=optimization.warmup_epochs)
    if validation is not None:
        info.update(validation_nll=best, validation_brier=score_brier(model, valid, grid))
    with torch.no_grad():
        diagnostics = validation if validation is not None else prepared
        rho = model.rho(diagnostics.x)
        info.update(rho_mean_absolute_shift=float((rho-.5).abs().mean()),
                    rho_root_mean_absolute_shift=float((rho[:, 0]-.5).abs().mean()))
    model.fit_info = info
    return model, history
