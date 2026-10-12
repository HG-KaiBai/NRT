"""Analytical likelihood checks and training invariants for the new model."""

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from .experiment import ExperimentConfig, run_experiments, settings, summarize
from .model import ConditionalTree, ModelConfig
from .train import TrainingConfig, fit
from .tree import Tree


class UnifiedLikelihoodTests(unittest.TestCase):
    def test_analytical_density_and_every_censoring_type(self):
        # Uniform U gives F_T(t)=t/(t+s). These are independent analytical
        # answers for exact, right, interval, and both current-status outcomes.
        model = ConditionalTree(ModelConfig(depth=2, x_dim=2), 2.0).double()
        left = torch.tensor([1., 1., 1., 0., 1.], dtype=torch.float64)
        right = torch.tensor([1., math.inf, 3., 1., math.inf], dtype=torch.float64)
        obs = model.prepare(torch.zeros(5, 2, dtype=torch.float64), left, right)
        actual = model.tree.loglik(model(obs.x), *obs[1:])
        expected = torch.tensor([2 / 9, 2 / 3, 3 / 5 - 1 / 3, 1 / 3, 2 / 3],
                                dtype=torch.float64).log()
        torch.testing.assert_close(actual, expected)

    def test_normalization_minimum_width_and_regularizer(self):
        tree = Tree(4, min_leaf_width=0.002).double()
        logits = torch.linspace(-5, 5, 30, dtype=torch.float64).reshape(2, 15)
        splits = torch.linspace(0, 1, 30, dtype=torch.float64).reshape(2, 15)
        state = tree(logits, splits)
        self.assertGreaterEqual(float(state.width.min()), 0.002 - 1e-14)
        torch.testing.assert_close(state.width.sum(1), torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(state.log_mass.exp().sum(1), torch.ones(2, dtype=torch.float64))
        self.assertTrue(bool((tree.regularizer(state) >= -1e-12).all()))
        # Geometric masses match the (nonuniform) leaf widths exactly.
        matched = state._replace(log_mass=state.width.log())
        torch.testing.assert_close(tree.regularizer(matched), torch.zeros(2, dtype=torch.float64))
        grid = torch.linspace(0, 1, 401, dtype=torch.float64)
        survival = tree.survival(state, grid)
        self.assertTrue(bool((survival.diff(dim=1) <= 1e-12).all()))
        torch.testing.assert_close(survival[:, 0], torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(survival[:, -1], torch.zeros(2, dtype=torch.float64))

    def test_likelihood_gradients_away_from_moving_boundaries(self):
        tree = Tree(2, min_leaf_width=0.001).double()
        logits = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64, requires_grad=True)
        splits = torch.tensor([[0.42, 0.36, 0.55]], dtype=torch.float64, requires_grad=True)
        for left, right, exact in ((0.08, 0.08, True), (0.21, 0.83, False),
                                   (0.63, 1.0, False), (0.0, 0.37, False)):
            def objective(mass, geometry):
                return tree.loglik(tree(mass, geometry),
                    torch.tensor([left], dtype=torch.float64),
                    torch.tensor([right], dtype=torch.float64),
                    torch.tensor([exact]), torch.zeros(1, dtype=torch.float64))
            self.assertTrue(torch.autograd.gradcheck(objective, (logits, splits)))

    def test_narrow_interval_is_not_reclassified_as_exact(self):
        model = ConditionalTree(ModelConfig(depth=2, x_dim=2), 1.0).double()
        left = torch.tensor([0.4], dtype=torch.float64)
        right = left + 1e-7
        obs = model.prepare(torch.zeros(1, 2, dtype=torch.float64), left, right)
        self.assertFalse(bool(obs.exact[0]))
        expected = (right / (right + 1) - left / (left + 1)).log()
        torch.testing.assert_close(model.tree.loglik(model(obs.x), *obs[1:]), expected)

    def test_fixed_and_learned_start_with_identical_predictions(self):
        config = ModelConfig(depth=2, x_dim=2, h_dim=4, mlp_hidden=5)
        torch.manual_seed(17)
        learned = ConditionalTree(config, 1.0)
        torch.manual_seed(17)
        fixed = ConditionalTree(replace(config, learn_splits=False), 1.0)
        x = torch.randn(6, 2)
        torch.testing.assert_close(learned(x), fixed(x))
        self.assertFalse(any(p.requires_grad for p in fixed.split_head.parameters()))
        self.assertFalse(any(p.requires_grad for p in fixed.split_encoder.parameters()))

    def test_training_and_cuda_graph_parity(self):
        torch.manual_seed(5)
        x = torch.randn(17, 2)
        left = torch.linspace(.1, 2, len(x))
        right = left.clone()
        right[::3] = torch.inf
        right[1::3] += .3
        config = ModelConfig(depth=2, x_dim=2, h_dim=4, mlp_hidden=5)
        training = TrainingConfig(epochs=3, batch_size=8, validation_every=1, patience=3)
        arguments = dict(validation=(x[:5], left[:5], right[:5]), model_config=config, seed=11)
        _, history = fit(x, left, right, training_config=training, device="cpu", **arguments)
        self.assertTrue(all(math.isfinite(row["validation_nll"]) for row in history))
        if torch.cuda.is_available():
            eager, eager_history = fit(x, left, right,
                training_config=replace(training, cuda_graphs=False), device="cuda", **arguments)
            graph, graph_history = fit(x, left, right,
                training_config=training, device="cuda", **arguments)
            for name, value in eager.state_dict().items():
                torch.testing.assert_close(value, graph.state_dict()[name], atol=2e-6, rtol=2e-5)
            for a, b in zip(eager_history, graph_history):
                self.assertAlmostEqual(a["validation_nll"], b["validation_nll"], places=5)

    def test_repeated_experiment_and_resume(self):
        self.assertEqual(len(settings(ExperimentConfig())), 28)
        config = ExperimentConfig(scenarios=("ph",), censor_types=("current",),
            censor_qs=(.5,), seeds=(0, 1), n_samples=60, grid_size=9)
        model = ModelConfig(depth=2, x_dim=2, h_dim=4, mlp_hidden=5, mlp_layers=2)
        train = TrainingConfig(epochs=2, batch_size=32, validation_every=1, patience=2,
                               cuda_graphs=False)
        with tempfile.TemporaryDirectory() as directory:
            runs = run_experiments(directory, config=config, model_config=model,
                                   training_config=train, device="cpu", verbose=False)
            resumed = run_experiments(directory, config=config, model_config=model,
                                      training_config=train, device="cpu", verbose=False)
            self.assertEqual(len(runs), 6)
            self.assertEqual(len(resumed), 6)
            self.assertTrue(summarize(runs, expected_repeats=2).n_runs.eq(2).all())
            self.assertTrue((Path(directory) / "paired_comparisons.csv").exists())
            with self.assertRaisesRegex(ValueError, "different code/configuration"):
                run_experiments(directory, config=replace(config, n_samples=61),
                    model_config=model, training_config=train, device="cpu", verbose=False)


if __name__ == "__main__":
    unittest.main()
