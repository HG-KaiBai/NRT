"""SUPPORT baselines and a memory-bounded adapter for the existing NNCDE."""
import copy

import numpy as np
import pandas as pd
import torch

from testing.benchmark.deepsurv.cuda_model import DeepSurvCUDA
from testing.benchmark.nncde.model import NNCDE, _FittedNetwork, _LogHazardNetwork


def support_parameters():
    """Exact SUPPORT Table-3 parameters; no external configuration file.

    Source: https://github.com/jaredleekatzman/DeepSurv/blob/
    41eed003e5b892c81e7855e400861fa7a2d9da4f/experiments/deepsurv/models/
    support_selu_model_adam_0.json
    Paper: https://doi.org/10.1186/s12874-018-0482-1
    The 800 epochs come from the SUPPORT Docker runner (Table 3 omits epochs);
    use Table 3's [44] architecture, not that runner's later [33,33,33] revision.
    """
    return dict(hidden_layers=(44,), learning_rate=0.047277902766138385,
                lr_decay=0.00257333984375, momentum=0.8589028320312501,
                l2_reg=8.1196630859375, l1_reg=0., activation="selu",
                dropout=0.2553173828125, batch_norm=False, standardize=True,
                optimizer="adam", epochs=800, tune=False)


# Upstream parameter configuration license (retained with the embedded preset):
# MIT License
# Copyright (c) 2016, Jared Katzman
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def breslow_loss(logits, observed, event):
    """Stable, order-invariant Cox partial likelihood for tied SUPPORT times."""
    order = torch.argsort(observed, descending=True, stable=True)
    time, risk, death = observed[order], logits[order], event[order]
    _, counts = torch.unique_consecutive(time, return_counts=True)
    ends = counts.cumsum(0) - 1
    denominators = torch.repeat_interleave(torch.logcumsumexp(risk, 0)[ends], counts)
    return -(death * (risk - denominators)).sum() / death.sum()


class SupportDeepSurv(DeepSurvCUDA):
    """Table-3 parameters, paper inverse-time LR decay and Breslow ties.

    Reuses the existing Lasagne-compatible architecture, initialization and
    output-weight L2 convention. Momentum is recorded from the paper but Adam
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
            lr = configuration.learning_rate / (1 + epoch * configuration.lr_decay)
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


class PaperCox:
    """Unpenalized lifelines CoxPHFitter, as in the authors' cph_run.py."""
    def fit(self, x, left, right, **_):
        from lifelines import CoxPHFitter
        self.columns = [f"x{i}" for i in range(x.shape[1])]
        frame = pd.DataFrame(x.numpy(), columns=self.columns)
        frame["time"] = left.numpy()
        frame["event"] = torch.isfinite(right).numpy()
        self.model = CoxPHFitter(penalizer=0., baseline_estimation_method="breslow")
        self.model.fit(frame, duration_col="time", event_col="event")
        return self

    def predict_survival(self, x, times):
        frame = pd.DataFrame(x.numpy(), columns=self.columns)
        # Right-continuous step evaluation of the fitted Breslow baseline;
        # avoid lifelines' linear interpolation between event-time jumps.
        baseline = self.model.baseline_cumulative_hazard_
        positions = np.searchsorted(baseline.index.to_numpy(), times.numpy(), side="right") - 1
        cumulative = np.zeros(len(times))
        keep = positions >= 0
        cumulative[keep] = baseline.iloc[positions[keep], 0].to_numpy()
        hazards = self.model.predict_partial_hazard(frame).to_numpy()
        return torch.tensor(np.exp(-hazards[:, None] * cumulative))

    def predict_risk(self, x):
        return torch.tensor(self.model.predict_log_partial_hazard(pd.DataFrame(x.numpy(), columns=self.columns)).to_numpy())


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
