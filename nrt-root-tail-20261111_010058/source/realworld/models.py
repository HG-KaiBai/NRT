"""Simulation adapters: memory-bounded NNCDE and DeepSurv with Breslow ties."""
import copy
import numpy as np
import torch
from testing.benchmark.deepsurv.cuda_model import DeepSurvCUDA
from testing.benchmark.nncde.model import NNCDE, _FittedNetwork, _LogHazardNetwork

def breslow_loss(logits, observed, event):
    """Stable, order-invariant Cox partial likelihood for tied SUPPORT times."""
    order = torch.argsort(observed, descending=True, stable=True)
    time, risk, death = observed[order], logits[order], event[order]
    _, counts = torch.unique_consecutive(time, return_counts=True)
    ends = counts.cumsum(0) - 1
    denominators = torch.repeat_interleave(torch.logcumsumexp(risk, 0)[ends], counts)
    return -(death * (risk - denominators)).sum() / death.sum()


class RealDeepSurv(DeepSurvCUDA):
    """Simulation architecture and constant Adam learning rate with Breslow ties.

    Reuses the existing Lasagne-compatible architecture, initialization and
    output-weight L2 convention. Momentum is recorded from the preset but Adam
    uses its ordinary beta parameters; no SGD momentum is silently added.
    """
    def _train_network(self, x, observed, event, configuration, *, epochs, seed_offset):
        mean, std = self._normalization(x, standardize=configuration.standardize)
        normalized = (x - mean) / std
        model = self._make_model(x.shape[1], configuration, seed_offset)
        optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
        self.training_history_ = []
        for epoch in range(epochs):
            model.train()
            lr = configuration.learning_rate
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = breslow_loss(model(normalized), observed, event) + self._regularization(model, configuration)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite DeepSurv objective")
            loss.backward()
            optimizer.step()
            if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
                self.training_history_.append(dict(epoch=epoch + 1, loss=float(loss.detach()), learning_rate=lr))
        model.eval()
        return model, mean, std



class MemoryBoundNNCDE(NNCDE):
    """Same NNCDE likelihood, grid, two-half cross-fit and early stopping.

    Only batch the expansion, validation and backward passes to fit real data
    in GPU memory. Gradients from subject microbatches are accumulated before
    each original minibatch optimizer step. No subsampling of time intervals.
    """
    microbatch = 32

    def _expanded_moments(self, x, observed):
        starts, stops = self._time_grid(observed)
        total = torch.zeros(x.shape[1] + 2, device=x.device, dtype=torch.float64)
        squares = torch.zeros_like(total)
        count = 0
        for ix in torch.arange(len(x), device=x.device).split(self.microbatch):
            mask = stops[None, :] <= observed[ix, None]
            values = torch.cat((x[ix, None, :].expand(-1, len(stops), -1),
                                starts[None, :, None].expand(len(ix), -1, -1),
                                stops[None, :, None].expand(len(ix), -1, -1)), dim=-1)[mask].double()
            total += values.sum(0)
            squares += values.square().sum(0)
            count += len(values)
        mean = total / count
        std = (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
        return mean.float(), std.float()

    def _train_one(self, x_train, observed_train, event_train, x_valid, observed_valid, event_valid, seed):
        torch.manual_seed(seed)
        mean, std = self._expanded_moments(x_train, observed_train)
        fitted = _FittedNetwork(_LogHazardNetwork(x_train.shape[1] + 2, self.hidden_nodes).to(self.device), mean, std)
        optimizer = torch.optim.Adam(fitted.network.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(seed)
        train_grid, valid_grid = self._time_grid(observed_train), self._time_grid(observed_valid)
        best, best_state, stale, best_epoch = float("inf"), None, 0, None
        for epoch in range(self.epochs):
            fitted.network.train()
            order = torch.randperm(len(x_train), generator=generator).to(self.device)
            for batch in order.split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                for ix in batch.split(self.microbatch):
                    loss = self._loss(fitted, x_train[ix], observed_train[ix], event_train[ix], *train_grid)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite NNCDE training loss")
                    (loss * (len(ix) / len(batch))).backward()
                optimizer.step()
            fitted.network.eval()
            with torch.no_grad():
                validation = sum(float(self._loss(fitted, x_valid[ix], observed_valid[ix], event_valid[ix], *valid_grid)) * len(ix)
                                 for ix in torch.arange(len(x_valid), device=self.device).split(self.microbatch)) / len(x_valid)
            if not np.isfinite(validation):
                raise RuntimeError("Nonfinite NNCDE validation loss")
            if validation < best - 1e-7:
                best, best_state, stale, best_epoch = validation, copy.deepcopy(fitted.network.state_dict()), 0, epoch + 1
            else:
                stale += 1
                if stale >= self.patience:
                    break
        fitted.network.load_state_dict(best_state)
        fitted.network.eval()
        if not hasattr(self, "fit_details_"):
            self.fit_details_ = []
        self.fit_details_.append(dict(seed=seed, best_epoch=best_epoch, validation_nll=best, epochs_run=epoch + 1))
        return fitted

    @torch.no_grad()
    def validation_nll(self, bounds):
        x, left, right = (v.to(self.device, dtype=torch.float32) for v in bounds)
        observed, event = self._outcomes(left, right, "right")
        grid = self._time_grid(observed)
        scores = []
        for fitted in self._models:
            score = sum(float(self._loss(fitted, x[ix], observed[ix], event[ix], *grid)) * len(ix)
                        for ix in torch.arange(len(x), device=self.device).split(self.microbatch)) / len(x)
            scores.append(score)
        return float(np.mean(scores))
