"""Numerical optimizer checks and a complete, small dual-mixture protocol."""
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .diagnostic_experiment import (expanded_settings, tune_close,
    select_diagnostic_parameters, evaluate_diagnostics, METHODS)
from .experiment import ExperimentConfig
from .parameter_search import SearchConfig, tune_per_setting, _observed_split
from .stable_train import StableTree, SplitOptimization, fit_stable
from .train import TrainingConfig
from .tuning_model import Architecture
from .tuning_train import fit_tuning


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        x = torch.randn(23, 6)
        left = torch.linspace(.1, 2., len(x))
        right = left.clone()
        right[::4] = torch.inf
        right[1::4] += .5
        left[2::4] = 0.
        self.train = x, left, right
        self.valid = tuple(v[:7] for v in self.train)
        self.grid = torch.linspace(0, 3, 17)
        self.architecture = Architecture(depth=3, shared_width=5, shared_layers=1,
            private_width=6, private_layers=2, regularization_strength=.5)
        self.training = TrainingConfig(epochs=4, batch_size=8, validation_every=1,
            patience=4, cuda_graphs=False)

    def test_no_schedule_matches_original_and_brier_does_not_train(self):
        kwargs = dict(architecture=self.architecture, training=self.training, seed=9, device="cpu")
        original, _ = fit_tuning(self.train, self.valid, self.grid, **kwargs)
        optimization = SplitOptimization(split_lr_ratio=1., warmup_epochs=0)
        with patch("pycode.unified_nll.stable_train.score_brier", return_value=0.) as first:
            actual, history = fit_stable(self.train, self.valid, self.grid, optimization=optimization, **kwargs)
        with patch("pycode.unified_nll.stable_train.score_brier", return_value=100.) as second:
            other, other_history = fit_stable(self.train, self.valid, self.grid, optimization=optimization, **kwargs)
        self.assertEqual(first.call_count, 1)
        self.assertEqual(second.call_count, 1)
        self.assertEqual(history, other_history)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[key], atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(actual.state_dict()[key], other.state_dict()[key], atol=0, rtol=0)

    def test_warmup_blocks_rho_gradients_and_fixed_rho_is_invariant(self):
        model = StableTree(self.architecture, 1., SplitOptimization(warmup_epochs=2))
        nll, regularizer = model.terms(model.prepare(*self.train))
        (nll+regularizer).backward()
        split_parameters = list(model.split_encoder.parameters())+list(model.split_head.parameters())
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero() == 0 for p in split_parameters))
        model.zero_grad()
        model.split_gate.fill_(1.)
        sum(model.terms(model.prepare(*self.train))).backward()
        self.assertGreater(float(model.split_head.weight.grad.abs().sum()), 0.)
        fixed = StableTree(self.architecture, 1., SplitOptimization(learn_splits=False))
        before = fixed(self.train[0])
        with torch.no_grad():
            for module in (fixed.shared, fixed.split_encoder, fixed.split_head):
                for parameter in module.parameters():
                    parameter.normal_()
        after = fixed(self.train[0])
        torch.testing.assert_close(before.lower, after.lower, atol=0, rtol=0)
        torch.testing.assert_close(before.upper, after.upper, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_replay_matches_eager_across_warmup_transition(self):
        for learn in (True, False):
            kwargs = dict(architecture=self.architecture, seed=9, device="cuda",
                optimization=SplitOptimization(learn_splits=learn, warmup_epochs=2))
            eager, _ = fit_stable(self.train, self.valid, self.grid, training=self.training, **kwargs)
            graph, _ = fit_stable(self.train, self.valid, self.grid,
                training=replace(self.training, cuda_graphs=True), **kwargs)
            for key, value in eager.state_dict().items():
                torch.testing.assert_close(value, graph.state_dict()[key], atol=2e-6, rtol=2e-5)
            self.assertAlmostEqual(eager.fit_info["validation_nll"], graph.fit_info["validation_nll"], places=5)

    def test_dual_mixture_search_freeze_evaluation_and_resume(self):
        self.assertEqual(len(expanded_settings(ExperimentConfig())), 35)
        config = ExperimentConfig(scenarios=("ph", "multimodal"), censor_types=("current",),
            censor_qs=(.5,), seeds=(0, 1), n_samples=60, grid_size=9)
        baseline = Architecture(depth=2, shared_width=4, private_width=4, private_layers=1,
            regularization_strength=.5)
        search = SearchConfig(trials=3, initial_trials=2, shortlist=2, depth_min=2,
            depth_max=3, widths=(4, 8), shared_layers_max=1, private_layers_max=2,
            confirmation_seeds=(102, 103))
        training = TrainingConfig(epochs=2, batch_size=32, validation_every=1, patience=2, cuda_graphs=False)
        kwargs = dict(config=config, search=search, training=training, baseline=baseline, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            historical, output = Path(directory)/"historical", Path(directory)/"diagnostic"
            tune_per_setting(historical, verbose=False, **kwargs)
            tune_close(output, **kwargs)
            selected = select_diagnostic_parameters(output, historical, **kwargs)
            self.assertEqual(len(selected), 6)
            self.assertIn("multimodal_close_current_q0.5", set(selected.setting))
            self.assertNotIn("average_grid_sse", selected.columns)
            repeated = select_diagnostic_parameters(output, historical, **kwargs)
            np.testing.assert_array_equal(selected.trial, repeated.trial)
            runs, summary = evaluate_diagnostics(output, historical, **kwargs)
            self.assertEqual(len(runs), 3*len(METHODS)*2)
            self.assertTrue(summary.n_runs.eq(2).all())
            resumed, _ = evaluate_diagnostics(output, historical, **kwargs)
            np.testing.assert_allclose(runs.average_grid_sse, resumed.average_grid_sse)
            same_x, different_bounds = [], []
            for name, _, setting, actual in expanded_settings(config):
                if name.startswith("multimodal"):
                    data = _observed_split(setting, actual, baseline)
                    same_x.append(data["train"][0])
                    different_bounds.append(data["train"][2])
            torch.testing.assert_close(*same_x)
            self.assertFalse(torch.equal(*different_bounds))


if __name__ == "__main__":
    unittest.main()
