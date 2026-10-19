"""Numerical, data-isolation and export tests; integration is the --smoke run."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from sksurv.metrics import brier_score, integrated_brier_score
from sksurv.util import Surv

from .data import FEATURES, load_support
from .metrics import evaluate, ipcw_brier, evaluation_grid
from .models import breslow_loss, MemoryBoundNNCDE, support_parameters
from .run import METHODS, summarize
from .tuning import default_config
from testing.benchmark.nncde.model import NNCDE
from tuning.main import NNCDE_GRID
from regularization.tuning.main import SearchConfig
from regularization.workflow import writer_lock
from dataclasses import asdict


class RealWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_data_and_train_only_preprocessing(self):
        data = load_support()
        self.assertEqual((len(data.train), len(data.test), data.x.shape[1]), (7098, 1775, 14))
        self.assertEqual(int(data.event[data.train].sum()), 4823)
        self.assertEqual(int(data.event[data.test].sum()), 1213)
        self.assertNotIn("row_in_split", FEATURES)
        train, valid = data.train[:100], data.train[100:200]
        before = data.scaler(train)
        grid, low, high = evaluation_grid(data.outcomes(data.train))
        data.x[valid] = 1e7
        data.x[data.test] = 1e8
        data.time[data.test] = 1e9
        after = data.scaler(train)
        torch.testing.assert_close(before[0], after[0])
        torch.testing.assert_close(before[1], after[1])
        np.testing.assert_array_equal(grid, evaluation_grid(data.outcomes(data.train))[0])
        self.assertEqual((low, high), (0., 2029.))
        self.assertEqual((grid[0], grid[-1]), (0., 2029.))
        self.assertEqual(len(grid), 261)

    def test_evaluation_grid_includes_zero_and_training_maximum(self):
        train = Surv.from_arrays([1, 0, 1], [5., 12., 40.])
        grid, low, high = evaluation_grid(train, size=3)
        np.testing.assert_array_equal(grid, [0., 20., 30., 40.])
        self.assertEqual((low, high), (0., 40.))
        with self.assertRaises(ValueError):
            evaluation_grid(train, size=1)

    def test_full_budget_and_paper_preset(self):
        config = default_config()
        self.assertEqual(config["nrt_search"], asdict(SearchConfig()))
        self.assertEqual(config["nncde_grid"], NNCDE_GRID)
        self.assertEqual((config["folds"], config["nncde_epochs"], config["nncde_patience"]), (4, 200, 20))
        paper = support_parameters()
        self.assertEqual(paper["hidden_layers"], (44,))
        self.assertAlmostEqual(paper["learning_rate"], .047277902766138385)
        self.assertAlmostEqual(paper["l2_reg"], 8.1196630859375)
        self.assertTrue(paper["standardize"])

    def test_ipcw_matches_scikit_survival_and_ibs(self):
        train = Surv.from_arrays([1, 0, 1, 0, 1, 0, 1, 1], [1, 2, 3, 3, 5, 6, 8, 10])
        test = Surv.from_arrays([1, 0, 1, 1, 0], [1.5, 2.5, 4, 7, 9])
        grid = np.array([0., 2., 3., 4., 6.])
        survival = np.exp(-np.arange(1, 6)[:, None] * grid / 20)
        expected = brier_score(train, test, survival[:, 1:], grid[1:])[1]
        actual = ipcw_brier(train, test, survival[:, 1:], grid[1:])[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        result, _, _ = evaluate(train, test, survival, grid, 2.)
        self.assertAlmostEqual(result["integrated_brier_score"],
                               integrated_brier_score(train, test, survival[:, 1:], grid[1:]))
        self.assertAlmostEqual(result["ipcw_brier_integral"],
                               result["integrated_brier_score"] * 4.)

    def test_full_interval_and_zero_censoring_survival_at_endpoint(self):
        train = Surv.from_arrays([1, 0], [1., 2.])
        test = Surv.from_arrays([1, 0], [1., 2.])
        grid = np.array([0., 1., 2.])
        survival = np.array([[1., .5, .25], [1., .75, .5]])
        with np.errstate(divide="raise", invalid="raise"):
            result, rows, risk = evaluate(train, test, survival, grid, 0.)
        # G(1)=1; at t=2, G=0 but the control sum is empty. Only the
        # event at t=1 contributes, with its positive denominator G(1).
        np.testing.assert_allclose([r["brier_ipcw"] for r in rows], [0., .15625, .03125])
        self.assertEqual(rows[-1]["censoring_survival"], 0.)
        self.assertEqual(result["evaluation_start_days"], 0.)
        self.assertEqual(result["evaluation_end_days"], 2.)
        self.assertAlmostEqual(result["ipcw_brier_integral"], .171875)
        self.assertAlmostEqual(result["integrated_brier_score"], .0859375)
        self.assertEqual(result["c_index"], 1.)
        np.testing.assert_allclose(risk, [-1.125, -1.5])

    def test_zero_censoring_survival_for_required_event_is_rejected(self):
        train = Surv.from_arrays([1, 0], [1., 2.])
        test = Surv.from_arrays([True], [2.])
        with self.assertRaisesRegex(ValueError, "required IPCW contribution"):
            ipcw_brier(train, test, np.array([[.5]]), [2.])

    def test_zero_censoring_survival_for_required_control_is_rejected(self):
        train = Surv.from_arrays([1, 0], [1., 2.])
        test = Surv.from_arrays([1, 0], [1.5, 3.])
        with self.assertRaisesRegex(ValueError, "required IPCW contribution"):
            ipcw_brier(train, test, np.array([[.5], [.5]]), [2.])

    def test_support_full_interval_has_finite_ipcw_scores(self):
        data = load_support()
        train, test = data.outcomes(data.train), data.outcomes(data.test)
        grid, low, high = evaluation_grid(train)
        survival = np.broadcast_to(np.exp(-grid / 1000.), (len(test), len(grid)))
        with np.errstate(divide="raise", invalid="raise"):
            metrics, rows, _ = evaluate(train, test, survival, grid, low)
        self.assertTrue(np.isfinite(metrics["integrated_brier_score"]))
        self.assertEqual(metrics["min_censoring_survival"], 0.)
        self.assertEqual(rows[-1]["time_days"], 2029.)
        self.assertAlmostEqual(metrics["ipcw_brier_integral"],
                               metrics["integrated_brier_score"] * high)
        # The public reference API accepts only interior test-follow-up times.
        # Check agreement there; the hand calculation above covers the endpoint.
        interior = (grid >= test["time"].min()) & (grid < test["time"].max())
        expected = brier_score(train, test, survival[:, interior], grid[interior])[1]
        actual = np.array([r["brier_ipcw"] for r in rows])[interior]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_comparison_exports_full_support_and_both_brier_integrals(self):
        train = Surv.from_arrays([1, 0], [1., 2.])
        grid = np.array([0., 1., 2.])
        survival = np.array([[1., .5, .25], [1., .75, .5]])
        metrics, rows, _ = evaluate(train, train, survival, grid, 0.)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "request.json").write_text(json.dumps(dict(request_hash="test", config=dict(profile="smoke"))))
            for method in METHODS:
                directory = output / method
                directory.mkdir()
                (directory / "result.json").write_text(json.dumps(dict(
                    status="ok", method=method, request_hash="test", selection={}, **metrics)))
                with (directory / "brier_curve.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["method", *rows[0]])
                    writer.writeheader()
                    writer.writerows(dict(method=method, **r) for r in rows)
            summarize(output)
            with (output / "comparison.csv").open(newline="") as stream:
                comparison = list(csv.DictReader(stream))
            self.assertEqual(len(comparison), 8)
            for row in comparison:
                self.assertEqual(float(row["c_index"]), 1.)
                self.assertAlmostEqual(float(row["integrated_brier_score"]), .0859375)
                self.assertAlmostEqual(float(row["ipcw_brier_integral"]), .171875)
                self.assertEqual(float(row["evaluation_start_days"]), 0.)
                self.assertEqual(float(row["evaluation_end_days"]), 2.)

    def test_ipcw_uncensored_and_late_test_followup(self):
        train = Surv.from_arrays([True, True, True], [1., 5., 10.])
        test = Surv.from_arrays([True, True], [2., 20.])
        estimate = np.array([[.4, .3], [.8, .7]])
        actual, _ = ipcw_brier(train, test, estimate, [3., 4.])
        np.testing.assert_allclose(actual, [.1, .09])
        censored = Surv.from_arrays([1, 0], [1., 2.])
        with self.assertRaises(ValueError):
            ipcw_brier(censored, test, estimate, [2., 3.])

    def test_breslow_ties_match_full_risk_sets(self):
        risk = torch.tensor([.3, -.2, .8, .4], dtype=torch.float64, requires_grad=True)
        observed = torch.tensor([2., 2., 3., 4.])
        event = torch.tensor([1., 0., 1., 1.])
        expected = -sum(risk[i] - torch.logsumexp(risk[observed >= observed[i]], 0)
                        for i in range(4) if event[i]) / event.sum()
        actual = breslow_loss(risk, observed, event)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, risk)[0], torch.autograd.grad(expected, risk)[0])
        permutation = torch.tensor([1, 0, 3, 2])
        torch.testing.assert_close(actual, breslow_loss(risk[permutation], observed[permutation], event[permutation]))

    def test_memory_bounded_nncde_matches_original(self):
        torch.manual_seed(17)
        x = torch.randn(18, 3)
        left = torch.linspace(.1, 2., 18)
        right = left.clone()
        right[::3] = torch.inf
        kwargs = dict(hidden_nodes=4, batch_size=8, epochs=3, patience=2, seed=18, device="cpu")
        original = NNCDE(**kwargs).fit(x, left, right, censor_type="right")
        bounded = MemoryBoundNNCDE(**kwargs)
        bounded.microbatch = 3
        bounded.fit(x, left, right, censor_type="right")
        grid = torch.linspace(0, 2, 20)
        torch.testing.assert_close(original.predict_survival(x, grid), bounded.predict_survival(x, grid), rtol=2e-5, atol=2e-6)

    def test_incomplete_summary_has_no_comparison_csv(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            output = Path(folder)
            (output / "request.json").write_text(json.dumps(dict(request_hash="test", config=dict(profile="smoke"))))
            with self.assertRaisesRegex(RuntimeError, "Only 0/8"):
                summarize(output)
            self.assertTrue((output / "status.csv").exists())
            self.assertFalse((output / "comparison.csv").exists())

    def test_writer_lock_rejects_concurrent_writer(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            with writer_lock(folder):
                with self.assertRaisesRegex(RuntimeError, "another worker"):
                    with writer_lock(folder):
                        self.fail("second writer acquired the same lock")
            with writer_lock(folder):
                pass


if __name__ == "__main__":
    unittest.main()
