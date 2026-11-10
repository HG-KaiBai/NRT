"""Analytic body/tail likelihood checks and small CPU training regressions."""
from dataclasses import replace
import math
import unittest

import torch

from .model import RawGeometry, RawTimeKLTree, observed_upper_bound
from .train import fit_raw_time_kl
from .unified_nll.nll_cv_train import NLLCVOptimization
from .unified_nll.train import TrainingConfig
from .unified_nll.tree import Tree
from .unified_nll.tuning_model import Architecture


class BodyTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.arch = Architecture(depth=2, x_dim=2, private_width=8,
                                 private_layers=1, regularization_strength=1.)
        self.tree = RawGeometry(self.arch, 8.)

    def state(self, n=1, *, gradients=False):
        # Four width-2 body intervals. Before the final split their masses
        # are .1, .3, .36, .24; afterwards the last mass is .072, tail .168.
        probs = torch.tensor([[.4, .25, .6, .3]], dtype=torch.float64).repeat(n, 1)
        logits = torch.logit(probs).requires_grad_(gradients)
        rho = torch.full((n, 3), .5, dtype=torch.float64, requires_grad=gradients)
        return self.tree(logits, rho), logits, rho

    def likelihood(self, left, right):
        left, right = (torch.tensor(v, dtype=torch.float64) for v in (left, right))
        state, _, _ = self.state(len(left))
        return self.tree.loglik(state, left, right, left == right, torch.zeros_like(left)).exp()

    def test_observed_horizon_has_no_padding(self):
        x = torch.zeros(3, 2)
        left, right = torch.tensor([0., 3., 5.]), torch.tensor([7., 3., torch.inf])
        self.assertEqual(observed_upper_bound((x, left, right)), 7.)

    def test_only_rightmost_mass_is_split_and_total_mass_is_one(self):
        state, _, _ = self.state()
        torch.testing.assert_close(state.upper, torch.tensor([[2., 4., 6., 8.]], dtype=torch.float64))
        torch.testing.assert_close(state.log_mass.exp(),
                                   torch.tensor([[.1, .3, .36, .072]], dtype=torch.float64))
        torch.testing.assert_close(state.log_tail_mass.exp(), torch.tensor([.168], dtype=torch.float64))
        torch.testing.assert_close(state.log_mass.exp().sum(-1) + state.log_tail_mass.exp(),
                                   torch.ones(1, dtype=torch.float64))

    def test_exact_density_is_left_open_right_closed(self):
        times = [0., 1., 2., 4., 6., 8.]
        expected = torch.tensor([0., .05, .05, .15, .18, .036], dtype=torch.float64)
        torch.testing.assert_close(self.likelihood(times, times), expected)

    def test_shared_tree_uses_same_density_boundaries(self):
        tree = Tree(depth=1)
        state = tree(torch.logit(torch.tensor([[.2]], dtype=torch.float64)),
                     torch.tensor([[.5]], dtype=torch.float64))
        for t, expected in [(0., 0.), (.5, .4), (1., 1.6)]:
            point = torch.tensor([t], dtype=torch.float64)
            actual = tree.loglik(state, point, point, torch.tensor([True]), point * 0).exp()
            torch.testing.assert_close(actual, torch.tensor([expected], dtype=torch.float64))

    def test_mixed_censoring_includes_tail_only_for_infinite_right(self):
        actual = self.likelihood([0., 2., 3., 8., 0., 6.],
                                 [math.inf, 6., 7., math.inf, 4., 8.])
        expected = torch.tensor([1., .66, .546, .168, .4, .072], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)

    def test_survival_matches_interval_likelihood_and_tail(self):
        state, _, _ = self.state()
        times = torch.tensor([0., 1., 2., 4., 6., 8., math.inf], dtype=torch.float64)
        actual = self.tree.survival(state, times).squeeze(0)
        expected = torch.tensor([1., .95, .9, .6, .24, .168, 0.], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[:-1], self.likelihood(times[:-1].tolist(), [math.inf] * 6))
        with self.assertRaisesRegex(ValueError, "survival is defined only"):
            self.tree.survival(state, torch.tensor([9.]))

    def test_observations_beyond_horizon_are_coarsened(self):
        model = RawTimeKLTree(self.arch, 8.)
        left = torch.tensor([9., 9., 7., 9., 8.])
        right = torch.tensor([9., math.inf, 9., 10., 8.])
        prepared = model.prepare(torch.zeros(5, 2), left, right)
        torch.testing.assert_close(prepared.left, torch.tensor([8., 8., 7., 8., 8.], dtype=torch.float64))
        torch.testing.assert_close(prepared.right,
                                   torch.tensor([math.inf] * 4 + [8.], dtype=torch.float64))
        self.assertEqual(prepared.exact.tolist(), [False, False, False, False, True])
        state = model(prepared.x)
        likelihood = model.tree.loglik(state, *prepared[1:]).exp()
        torch.testing.assert_close(likelihood[:2], state.log_tail_mass[:2].double().exp())
        self.assertTrue(torch.isfinite(model.terms(prepared)[0]))

    def test_fixed_and_adaptive_output_dimensions(self):
        for depth in (1, 2, 4):
            for learn in (False, True):
                with self.subTest(depth=depth, learn=learn):
                    model = RawTimeKLTree(replace(self.arch, depth=depth), 8.,
                                          NLLCVOptimization(learn_splits=learn))
                    x = torch.zeros(3, 2)
                    self.assertEqual(model.mass_head.out_features, 2**depth)
                    self.assertEqual(model.split_head.out_features, 2**depth - 1)
                    self.assertEqual(model.rho(x).shape, (3, 2**depth - 1))
                    state = model(x)
                    self.assertEqual(state.width.shape, (3, 2**depth))
                    torch.testing.assert_close(state.log_mass.exp().sum(-1) + state.log_tail_mass.exp(),
                                               torch.ones(3))

    def test_mixed_and_tail_only_likelihood_gradients_are_finite(self):
        state, logits, rho = self.state(4, gradients=True)
        left = torch.tensor([8., 1., 3., 5.], dtype=torch.float64)
        right = torch.tensor([math.inf, 1., 7., math.inf], dtype=torch.float64)
        loss = -self.tree.loglik(state, left, right, left == right, torch.zeros_like(left)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(rho.grad).all())
        self.assertGreater(float(logits.grad[0, -1]), 0.)

    def test_partition_penalty_does_not_include_tail(self):
        model = RawTimeKLTree(self.arch, 8., NLLCVOptimization(warmup_epochs=0))
        observations = model.prepare(torch.zeros(2, 2), torch.ones(2), torch.ones(2))
        with torch.no_grad():
            model.split_head.bias.copy_(torch.tensor([.7, -.3, .4]))
        _, before = model.terms(observations)
        self.assertGreater(float(before.detach()), 0.)
        with torch.no_grad():
            model.mass_head.bias[-1] = 5.
        _, after = model.terms(observations)
        torch.testing.assert_close(before, after)

    def test_cpu_training_fixed_and_adaptive_with_tail_only_rows(self):
        x = torch.linspace(-1, 1, 16).reshape(8, 2)
        left = torch.tensor([1., 2., 0., 3., 4., 5., 8., 8.])
        right = torch.tensor([1., 4., 3., math.inf, 4., 7., 8., math.inf])
        data = (x, left, right)
        training = TrainingConfig(epochs=3, batch_size=4, validation_every=1, cuda_graphs=False)
        for learn in (False, True):
            for cv in (False, True):
                with self.subTest(learn=learn, cv=cv):
                    model, history = fit_raw_time_kl(
                        data, data if cv else None, architecture=self.arch, training=training,
                        optimization=NLLCVOptimization(learn_splits=learn, warmup_epochs=1),
                        device="cpu", fixed_epochs=None if cv else 3)
                    self.assertEqual(model.fit_info["upper_bound"], 8.)
                    self.assertEqual(model.fit_info["max_observation"], 8.)
                    self.assertEqual(model.fit_info["support"], "raw_time_body_tail")
                    self.assertTrue(all(math.isfinite(row["objective"]) for row in history))
                    self.assertNotEqual(float(model.mass_head.bias[-1].detach()), 0.)
                    survival = model.predict_survival(x, torch.tensor([0., 4., 8.]), batch_size=3)
                    self.assertTrue(torch.isfinite(survival).all())
                    self.assertTrue((survival[:, :-1] >= survival[:, 1:]).all())
                    torch.testing.assert_close(survival[:, -1], model(x).log_tail_mass.double().exp(),
                                               rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
