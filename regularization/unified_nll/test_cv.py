"""CV partition, numerical, leakage and independent-replication checks."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .cv_report import audit_cv
from .cv_search import (CVSearchConfig, METHODS, evaluate_cv, fold_indices,
                        observed_pool, refit_epoch_count, tune_cv)
from .cv_train import CVOptimization, CVTree, fit_cv
from .depth_experiment import _evaluation_data
from .experiment import ExperimentConfig
from .regularization import partition_penalty
from .train import TrainingConfig
from .tuning_model import Architecture


class CVTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        x = torch.randn(24, 6)
        left = torch.linspace(.1, 2., len(x))
        right = left.clone()
        right[::3] = torch.inf
        right[1::3] += .5
        self.train = x, left, right
        self.valid = tuple(v[:7] for v in self.train)
        self.grid = torch.linspace(0, 3, 17)
        self.architecture = Architecture(depth=3, shared_width=5, shared_layers=1,
            private_width=6, private_layers=2, regularization_strength=100000.)
        self.training = TrainingConfig(epochs=4, batch_size=8, validation_every=1,
            patience=4, cuda_graphs=False)
        self.optimization = CVOptimization(warmup_epochs=1)

    def test_partition_precision_and_trainable_rho(self):
        splits = fold_indices(4000)
        self.assertEqual(sorted(torch.cat([v for _, v in splits]).tolist()), list(range(4000)))
        for train, valid in splits:
            self.assertEqual((len(train), len(valid)), (3000, 1000))
            self.assertFalse(set(train.tolist()) & set(valid.tolist()))
        model = CVTree(self.architecture, 1., self.optimization)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        state = model(self.train[0])
        self.assertAlmostEqual(float(model.terms(model.prepare(*self.train))[1]), 0., places=12)
        model.split_gate.fill_(1.)
        with torch.no_grad():
            model.split_head.bias.fill_(.0001)
        reg = model.terms(model.prepare(*self.train))[1]
        self.assertGreater(float(reg), 0.)
        reg.backward()
        self.assertTrue(torch.isfinite(model.split_head.bias.grad).all())
        self.assertGreater(float(model.split_head.bias.grad.abs().sum()), 0.)
        with torch.no_grad():
            model.split_head.bias.fill_(.4)
        state = model(self.train[0])
        actual = model.terms(model.prepare(*self.train))[1]
        torch.testing.assert_close(actual.float(), partition_penalty(state, self.architecture.depth).mean(), atol=2e-7, rtol=1e-5)
        self.assertEqual(refit_epoch_count([15, 20, 40, 50], 10), 30)

    def test_brier_is_postfit_and_full_pool_refit_has_no_validation(self):
        kwargs = dict(architecture=self.architecture, training=self.training,
            optimization=self.optimization, seed=5, device="cpu")
        with patch("pycode.unified_nll.cv_train.score_brier", return_value=0.) as score:
            first, history = fit_cv(self.train, self.valid, self.grid, **kwargs)
        self.assertEqual(score.call_count, 1)
        with patch("pycode.unified_nll.cv_train.score_brier", return_value=100.):
            other, other_history = fit_cv(self.train, self.valid, self.grid, **kwargs)
        self.assertEqual(history, other_history)
        self.assertGreater(first.fit_info["best_epoch"], self.optimization.warmup_epochs)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[key], atol=0, rtol=0)
        with patch("pycode.unified_nll.cv_train.score_brier", side_effect=AssertionError("Brier during refit")), \
                patch("pycode.unified_nll.cv_train._evaluate", side_effect=AssertionError("test stopping")):
            refit, history = fit_cv(self.train, fixed_epochs=3, **kwargs)
        self.assertEqual(refit.fit_info["epochs_run"], 3)
        self.assertTrue(all("validation_nll" not in row for row in history))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_and_eager_match_with_large_penalty(self):
        kwargs = dict(architecture=self.architecture, optimization=self.optimization, seed=5, device="cuda")
        eager, _ = fit_cv(self.train, self.valid, self.grid, training=self.training, **kwargs)
        graph, _ = fit_cv(self.train, self.valid, self.grid,
            training=replace(self.training, cuda_graphs=True), **kwargs)
        for key, value in eager.state_dict().items():
            torch.testing.assert_close(value, graph.state_dict()[key], atol=2e-6, rtol=2e-5)

    def test_interrupted_fold_resume_and_independent_final_datasets(self):
        config = ExperimentConfig(scenarios=("ph", "multimodal"), censor_types=("current",),
            censor_qs=(.5,), n_samples=60, seeds=(0, 1), grid_size=9)
        baseline = Architecture(depth=2, shared_width=4, private_width=4, private_layers=1,
            regularization_strength=.5)
        search = CVSearchConfig(trials=2, initial_trials=2, depth_min=2, depth_max=3,
            widths=(4, 8), shared_layers_max=1, private_layers_max=2, strength_max=100000.)
        training = TrainingConfig(epochs=3, batch_size=32, validation_every=1, patience=3, cuda_graphs=False)
        kwargs = dict(config=config, search=search, training=training, baseline=baseline,
            optimization=self.optimization, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            calls = []
            def interrupt_second(*args, **fit_kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("simulated interruption")
                return fit_cv(*args, **fit_kwargs)
            with patch("pycode.unified_nll.cv_search.fit_cv", side_effect=interrupt_second):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    tune_cv(output, **kwargs)
            self.assertFalse((output/"selection_complete.json").exists())
            first_cache = next((output/"fold_cache").glob("*.json"))
            self.assertEqual(len(json.loads(first_cache.read_text())["folds"]), 1)
            with patch("pycode.unified_nll.cv_search._evaluation_data", side_effect=AssertionError("truth during selection")):
                selected, trials, folds = tune_cv(output, **kwargs)
            self.assertEqual(len(selected), 3*len(METHODS))
            self.assertEqual(len(folds), 3*2*4)
            self.assertIn("multimodal_close_current_q0.5", set(selected.setting))
            self.assertNotIn("average_grid_sse", selected.columns)
            data_seeds = []
            def record_generation(setting, actual, model_config, seed):
                data_seeds.append(seed)
                return _evaluation_data(setting, actual, model_config, seed)
            with patch("pycode.unified_nll.cv_search._evaluation_data", side_effect=record_generation):
                runs, summary = evaluate_cv(output, **kwargs)
            self.assertEqual(sorted(data_seeds), [0, 0, 0, 1, 1, 1])
            self.assertTrue(runs.n_fit.eq(48).all())
            self.assertTrue(runs.n_test.eq(12).all())
            self.assertTrue(summary.n_runs.eq(2).all())
            audit = audit_cv(output)
            self.assertTrue(audit["passed"])
            with patch("pycode.unified_nll.cv_search.fit_cv", side_effect=AssertionError("unnecessary refit")):
                tune_cv(output, **kwargs)
                resumed, _ = evaluate_cv(output, **kwargs)
            np.testing.assert_allclose(runs.average_grid_sse, resumed.average_grid_sse)
            # Distinct final seeds must produce distinct observed training data.
            from .diagnostic_experiment import expanded_settings
            _, _, setting, actual = expanded_settings(config)[0]
            first = observed_pool(setting, replace(actual, data_seed=0), baseline)
            second = observed_pool(setting, replace(actual, data_seed=1), baseline)
            self.assertFalse(torch.equal(first["pool"][0], second["pool"][0]))


if __name__ == "__main__":
    unittest.main()
