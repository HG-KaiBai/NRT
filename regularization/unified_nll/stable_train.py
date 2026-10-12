"""Controlled optimizer changes; the observed likelihood and penalty stay fixed."""
from dataclasses import dataclass
import math
import torch

from .brier import score_brier
from .train import TrainingConfig, _CudaSteps, _evaluate, _snapshot, _subset
from .tuning_model import Architecture, TuningTree


@dataclass(frozen=True)
class SplitOptimization:
    learn_splits: bool = True
    split_lr_ratio: float = .1
    warmup_epochs: int = 10


class StableTree(TuningTree):
    def __init__(self, architecture, time_scale, optimization):
        super().__init__(architecture, time_scale)
        self.optimization = optimization
        self.register_buffer("split_gate", torch.tensor(float(optimization.warmup_epochs == 0)))
        if not optimization.learn_splits:
            self.split_encoder.requires_grad_(False)
            self.split_head.requires_grad_(False)

    def forward(self, x):
        features = self.shared(x)
        logits = self.mass_head(self.mass_encoder(features))
        if self.optimization.learn_splits:
            split_logits = self.split_gate * self.split_head(self.split_encoder(features))
            splits = self.config.split_min + (1-2*self.config.split_min)*torch.sigmoid(split_logits)
        else:
            splits = torch.full_like(logits, .5)
        return self.tree(logits, splits)


def fit_stable(train, valid, grid, *, architecture=Architecture(),
               training=TrainingConfig(patience=50), optimization=SplitOptimization(),
               time_scale=1., seed=101, device="cuda"):
    """NLL checkpoints only; zero split gradients during mass warmup.

    The shared trunk keeps learning during warmup. A tensor gate also keeps
    rho exactly 0.5 despite changing shared features and CUDA graph replay.
    Split Adam moments and step counts are reset when warmup ends.
    """
    if (not math.isfinite(optimization.split_lr_ratio) or optimization.split_lr_ratio <= 0
            or not isinstance(optimization.warmup_epochs, int) or optimization.warmup_epochs < 0):
        raise ValueError("invalid split optimization")
    if min(training.epochs, training.batch_size, training.validation_every, training.patience) < 1:
        raise ValueError("training counts must be positive")
    torch.manual_seed(seed)
    model = StableTree(architecture, time_scale, optimization).to(device)
    prepared_train, prepared_valid = model.prepare(*train), model.prepare(*valid)
    split_params = list(model.split_encoder.parameters())+list(model.split_head.parameters())
    split_ids = {id(p) for p in split_params}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in split_ids], lr=training.lr)]
    if optimization.learn_splits:
        groups.append(dict(params=split_params, lr=training.lr*optimization.split_lr_ratio))
    optimizer = torch.optim.Adam(groups, fused=str(device).startswith("cuda"),
        capturable=str(device).startswith("cuda") and training.cuda_graphs)
    coefficient = architecture.regularization_strength/math.sqrt(len(train[0]))
    totals = prepared_train.x.new_zeros(2)
    graphs = (_CudaSteps(model, optimizer, prepared_train, training, coefficient, totals)
              if str(device).startswith("cuda") and training.cuda_graphs else None)
    best = _evaluate(model, prepared_valid)
    snapshot, best_epoch, reference, patience_epoch = _snapshot(model), 0, best, 0
    history = [dict(epoch=0, validation_nll=best)]
    for epoch in range(1, training.epochs+1):
        if optimization.learn_splits and epoch == optimization.warmup_epochs+1:
            model.split_gate.fill_(1.)
            # Keep the captured storage addresses intact.
            for parameter in split_params:
                for value in optimizer.state.get(parameter, {}).values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
        model.train()
        totals.zero_()
        for indices in torch.randperm(len(train[0]), device=device).split(training.batch_size):
            if graphs is not None:
                graphs.step(indices)
            else:
                optimizer.zero_grad(set_to_none=True)
                nll, reg = model.terms(_subset(prepared_train, indices))
                (nll+coefficient*reg).backward()
                if training.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm)
                optimizer.step()
                totals.add_(torch.stack((nll.detach(), reg.detach())), alpha=len(indices))
        nll, reg = (totals/len(train[0])).tolist()
        if not math.isfinite(nll+reg):
            raise RuntimeError("nonfinite training objective")
        row = dict(epoch=epoch, train_nll=nll, regularizer=reg, objective=nll+coefficient*reg,
            split_active=optimization.learn_splits and epoch > optimization.warmup_epochs,
            mass_lr=training.lr, split_lr=training.lr*optimization.split_lr_ratio
                if optimization.learn_splits and epoch > optimization.warmup_epochs else 0.)
        if epoch % training.validation_every == 0 or epoch == training.epochs:
            score = _evaluate(model, prepared_valid)
            if not math.isfinite(score):
                raise RuntimeError("nonfinite validation NLL")
            row["validation_nll"] = score
            if score < best:
                best, snapshot, best_epoch = score, _snapshot(model), epoch
            if score < reference-training.min_delta:
                reference, patience_epoch = score, epoch
        history.append(row)
        # Give the learned-split phase its full stopping patience after warmup.
        start = optimization.warmup_epochs if optimization.learn_splits else 0
        if "validation_nll" in row and epoch-max(patience_epoch, start) >= training.patience:
            break
    model.load_state_dict(snapshot)
    model.eval()
    with torch.no_grad():
        features = model.shared(prepared_valid.x)
        raw = model.split_head(model.split_encoder(features))
        rho = (architecture.split_min+(1-2*architecture.split_min)*torch.sigmoid(model.split_gate*raw)) if optimization.learn_splits else torch.full_like(raw, .5)
        state = model(prepared_valid.x)
    model.fit_info = dict(best_epoch=best_epoch, epochs_run=epoch, validation_nll=best,
        validation_brier=score_brier(model, valid, grid), lambda_n=coefficient,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        rho_mean_absolute_shift=float((rho-.5).abs().mean()),
        min_leaf_width=float(state.width.min()),
        mass_kl_uniform=float((state.log_mass.exp()*(state.log_mass+math.log(model.tree.n_leaves))).sum(-1).mean()))
    return model, history
