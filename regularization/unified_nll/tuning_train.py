"""One NLL objective and NLL early stopping; Brier is scored after fitting."""
import math
import torch
from .brier import score_brier
from .train import TrainingConfig, _CudaSteps, _evaluate, _snapshot, _subset
from .tuning_model import Architecture, TuningTree


def fit_tuning(train, valid, grid, *, architecture=Architecture(),
               training=TrainingConfig(), time_scale=1., seed=101, device="cuda"):
    """Brier does not participate in gradient updates or checkpoint selection."""
    if min(training.epochs,training.batch_size,training.validation_every,training.patience) < 1:
        raise ValueError("training counts must be positive")
    torch.manual_seed(seed)
    model = TuningTree(architecture,time_scale).to(device)
    prepared_train, prepared_valid = model.prepare(*train), model.prepare(*valid)
    coefficient = architecture.regularization_strength/math.sqrt(len(train[0]))
    optimizer = torch.optim.Adam(model.parameters(),lr=training.lr,fused=str(device).startswith("cuda"),
        capturable=str(device).startswith("cuda") and training.cuda_graphs)
    totals = prepared_train.x.new_zeros(2)
    graphs = (_CudaSteps(model,optimizer,prepared_train,training,coefficient,totals)
              if str(device).startswith("cuda") and training.cuda_graphs else None)
    best = _evaluate(model,prepared_valid)
    snapshot, best_epoch, reference, patience_epoch = _snapshot(model),0,best,0
    history = [dict(epoch=0,validation_nll=best)]
    for epoch in range(1,training.epochs+1):
        model.train()
        totals.zero_()
        for indices in torch.randperm(len(train[0]),device=device).split(training.batch_size):
            if graphs is not None:
                graphs.step(indices)
            else:
                optimizer.zero_grad(set_to_none=True)
                nll, reg = model.terms(_subset(prepared_train,indices))
                (nll+coefficient*reg).backward()
                if training.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),training.max_grad_norm)
                optimizer.step()
                totals.add_(torch.stack((nll.detach(),reg.detach())),alpha=len(indices))
        nll, reg = (totals/len(train[0])).tolist()
        if not math.isfinite(nll+reg):
            raise RuntimeError("nonfinite training loss")
        row = dict(epoch=epoch,train_nll=nll,regularizer=reg,objective=nll+coefficient*reg)
        if epoch % training.validation_every == 0 or epoch == training.epochs:
            score = _evaluate(model,prepared_valid)
            if not math.isfinite(score):
                raise RuntimeError("nonfinite validation NLL")
            row["validation_nll"] = score
            if score < best:
                best,snapshot,best_epoch = score,_snapshot(model),epoch
            if score < reference-training.min_delta:
                reference,patience_epoch = score,epoch
        history.append(row)
        if "validation_nll" in row and epoch-patience_epoch >= training.patience:
            break
    model.load_state_dict(snapshot)
    model.eval()
    model.fit_info = dict(best_epoch=best_epoch,epochs_run=epoch,validation_nll=best,
        validation_brier=score_brier(model,valid,grid),lambda_n=coefficient,
        parameters=sum(p.numel() for p in model.parameters()))
    return model,history
