"""One observed-data loss, NLL checkpoints, and a genuine fixed-rho control."""
from dataclasses import dataclass
import math

import torch

from .cv_train import CVTree, CVOptimization
from .train import TrainingConfig, _CudaSteps, _evaluate, _snapshot, _subset
from .tuning_model import Architecture


@dataclass(frozen=True)
class NLLCVOptimization(CVOptimization):
    learn_splits: bool = True


class NLLCVTree(CVTree):
    def __init__(self, architecture, time_scale, optimization):
        super().__init__(architecture, time_scale, optimization)
        self.learn_splits = optimization.learn_splits
        if not self.learn_splits:
            self.split_encoder.requires_grad_(False)
            self.split_head.requires_grad_(False)

    def rho(self, x):
        if self.learn_splits:
            return super().rho(x)
        return x.new_full((len(x), self.tree.n_internal), .5)

    def forward(self, x):
        features = self.shared(x)
        logits = self.mass_head(self.mass_encoder(features))
        if self.learn_splits:
            raw = self.split_head(self.split_encoder(features))
            rho = self.config.split_min+(1-2*self.config.split_min)*torch.sigmoid(self.split_gate*raw)
        else:
            rho = torch.full_like(logits, .5)
        return self.tree(logits, rho)


def fit_nll_cv(train, valid=None, *, architecture=Architecture(),
               training=TrainingConfig(patience=50), optimization=NLLCVOptimization(),
               time_scale=1., seed=101, device="cuda", fixed_epochs=None):
    """No Brier call or test input. Both methods use the same checkpoint window.

    Fixed rho is exactly 0.5, so the inherited width penalty is identically zero.
    Inactive split parameters are excluded from Adam and gradient clipping.
    Both models initialize the same mass/shared weights at a matched architecture.
    """
    if (optimization.split_lr_ratio <= 0 or not math.isfinite(optimization.split_lr_ratio)
            or optimization.warmup_epochs < 0 or min(training.epochs, training.batch_size,
                training.validation_every, training.patience) < 1):
        raise ValueError("invalid training configuration")
    if (fixed_epochs is None) != (valid is not None):
        raise ValueError("supply validation for CV, or fixed_epochs for a full-pool refit")
    epochs = training.epochs if fixed_epochs is None else int(fixed_epochs)
    if epochs <= optimization.warmup_epochs:
        raise ValueError("need epochs after the common warmup/checkpoint window")
    torch.manual_seed(seed)
    model = NLLCVTree(architecture, time_scale, optimization).to(device)
    prepared = model.prepare(*train)
    validation = model.prepare(*valid) if valid is not None else None
    split = list(model.split_encoder.parameters())+list(model.split_head.parameters())
    split_ids = {id(p) for p in split}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in split_ids], lr=training.lr)]
    if optimization.learn_splits:
        groups.append(dict(params=split, lr=training.lr*optimization.split_lr_ratio))
    optimizer = torch.optim.Adam(groups, fused=str(device).startswith("cuda"),
        capturable=str(device).startswith("cuda") and training.cuda_graphs)
    coefficient = architecture.regularization_strength/math.sqrt(len(train[0]))
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    graphs = (_CudaSteps(model, optimizer, prepared, training, coefficient, totals)
              if str(device).startswith("cuda") and training.cuda_graphs else None)
    best, reference, best_epoch, patience_epoch = float("inf"), float("inf"), None, 0
    snapshot, history = None, []
    for epoch in range(1, epochs+1):
        if optimization.learn_splits and epoch == optimization.warmup_epochs+1:
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
        eligible = epoch > optimization.warmup_epochs
        row = dict(epoch=epoch, train_nll=nll, regularizer=reg, objective=nll+coefficient*reg,
            checkpoint_eligible=eligible, split_active=optimization.learn_splits and eligible)
        if validation is not None and (epoch % training.validation_every == 0 or epoch == epochs):
            score = _evaluate(model, validation)
            if not math.isfinite(score):
                raise RuntimeError("nonfinite validation NLL")
            row["validation_nll"] = score
            if eligible:
                if score < best:
                    best, best_epoch, snapshot = score, epoch, _snapshot(model)
                if score < reference-training.min_delta:
                    reference, patience_epoch = score, epoch
        history.append(row)
        if (validation is not None and "validation_nll" in row and eligible
                and epoch-patience_epoch >= training.patience):
            break
    if validation is not None:
        if snapshot is None:
            raise RuntimeError("no eligible NLL checkpoint evaluated")
        model.load_state_dict(snapshot)
    model.eval()
    info = dict(best_epoch=best_epoch if validation is not None else epochs,
        epochs_run=epoch, lambda_n=coefficient, n_fit=len(train[0]),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        learn_splits=optimization.learn_splits, split_lr_ratio=optimization.split_lr_ratio,
        warmup_epochs=optimization.warmup_epochs)
    if validation is not None:
        info["validation_nll"] = best
    with torch.no_grad():
        diagnostics = validation if validation is not None else prepared
        rho = model.rho(diagnostics.x)
        info.update(rho_mean_absolute_shift=float((rho-.5).abs().mean()),
                    rho_root_mean_absolute_shift=float((rho[:, 0]-.5).abs().mean()))
    model.fit_info = info
    return model, history
