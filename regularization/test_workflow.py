"""Dataset parity, raw likelihood, frozen refits and a small two-stage run."""
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import workflow as existing
from tuning import main as existing_tuning
from testing import simulation as existing_testing
from .model import RawTimeKLTree, observed_upper_bound
from .train import fit_raw_time_kl
from .tuning.main import main as tune, optuna_module
from .testing.simulation import main as simulate, summarize
from .unified_nll.nll_cv_train import NLLCVOptimization
from .unified_nll.train import TrainingConfig, _evaluate
from .unified_nll.tuning_model import Architecture
from .workflow import (METHODS, dataset_hash, evaluate, load_hyperparameters,
                       simulation_data, source_hash, tuning_data)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_same_data(self, actual, expected):
        for key in ("x", "L", "R", "T", "first", "second", "train", "test"):
            self.assertTrue(torch.equal(getattr(actual, key), getattr(expected, key)), key)
        self.assertEqual(actual.metadata(), expected.metadata())
        self.assertEqual(dataset_hash(actual), dataset_hash(expected))

    def test_all_35_tuning_datasets_match_actual_existing_entrypoint(self):
        self.assertEqual(len(existing.ALL_CASES), 35)
        with tempfile.TemporaryDirectory() as tmp:
            for case in existing.ALL_CASES:
                with self.subTest(case=case.id):
                    args = existing_tuning.parser().parse_args([
                        "--method", "NRT", "--output-dir", tmp, "--epochs", "3"])
                    captured = []
                    def capture(*positional, **kwargs):
                        captured.append(existing.generate(*positional, **kwargs))
                        raise StopIteration
                    with patch.object(existing_tuning, "generate", side_effect=capture):
                        with self.assertRaises(StopIteration):
                            existing_tuning.tune_case(args, "NRT_NLL", case, [{}])
                    actual, seed = tuning_data(case)
                    self.assertEqual(seed, args.seed+case.index*10_000)
                    self.assert_same_data(actual, captured[0])

    def test_all_35_simulation_datasets_match_existing_job_seeds(self):
        args = SimpleNamespace(base_seed=123, n_samples=5000, grid_size=256,
                               device="cpu", threads_per_fit=1, force=False, output_dir=Path("unused"))
        for case in existing.ALL_CASES:
            spec = dict(tau=existing.horizon(case), mu=0., r=None)
            for replication in (1, 7):
                with self.subTest(case=case.id, replication=replication):
                    job = existing_testing.make_job(args, "NRT_NLL", case, replication, (spec, {}, {}, "hash"))
                    request = job["request"]
                    expected = existing.generate(case, request["n_samples"], request["data_seed"],
                        request["tau"], mu=request["mu"], r=request["r"])
                    actual, seed = simulation_data(case, spec, replication)
                    self.assertEqual(seed, request["data_seed"])
                    self.assert_same_data(actual, expected)
                    self.assertNotEqual(seed, 123+case.index*10_000)

    def test_custom_generator_settings_and_rng_are_preserved(self):
        for case in existing.select_cases("ph_right_q0.5,multimodal_closer_interval_q0.2"):
            torch.manual_seed(99)
            before = torch.get_rng_state().clone()
            actual, seed = tuning_data(case, n_samples=128, base_seed=555, mu=.3, r=.6)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            expected = existing.generate(case, 128, seed, existing.horizon(case, mu=.3, r=.6), mu=.3, r=.6)
            self.assert_same_data(actual, expected)

    def test_raw_likelihood_and_penalty_scale(self):
        arch = Architecture(depth=3, private_width=8, private_layers=2)
        model = RawTimeKLTree(arch, 11., NLLCVOptimization(warmup_epochs=0))
        x = torch.zeros(7, 6)
        left = torch.tensor([2., 2., 2., 0., 12., 11., 11.], dtype=torch.float64)
        right = torch.tensor([2., torch.inf, 5., 5., 12., 11., torch.inf], dtype=torch.float64)
        observed = model.prepare(x, left, right)
        self.assertTrue(torch.equal(observed.left, left))
        expected = torch.tensor([1/11, 9/11, 3/11, 5/11, 0., 1/11, 0.], dtype=torch.float64)
        torch.testing.assert_close(model.tree.loglik(model(x), *observed[1:]).exp(), expected)
        with torch.no_grad():
            model.split_head.bias.copy_(torch.linspace(-2., 2., 7))
        state = model(x)
        penalty = (-2/arch.depth)*(state.width/(11./8)).log().mean()
        torch.testing.assert_close(model.terms(observed)[1], penalty)
        bigger = RawTimeKLTree(arch, 110., NLLCVOptimization(warmup_epochs=0))
        bigger.load_state_dict(model.state_dict())
        bigger.tree.upper_bound.fill_(110.)
        torch.testing.assert_close(bigger.terms(bigger.prepare(x, left*10, right*10))[1], penalty)
        self.assertTrue(bool((state.width >= 11.*arch.total_width_floor/8).all()))

    def test_support_fixed_epochs_and_no_test_stopping(self):
        train = (torch.randn(32, 6), torch.linspace(.1, 2., 32), torch.linspace(.1, 2., 32))
        arch = Architecture(depth=2, private_width=8, private_layers=1, regularization_strength=1.)
        training = TrainingConfig(epochs=3, batch_size=16, validation_every=1, cuda_graphs=False)
        optimization = NLLCVOptimization(warmup_epochs=1)
        self.assertAlmostEqual(observed_upper_bound(train), 2.2)
        invalid = (torch.zeros(1, 6), torch.tensor([100.]), torch.tensor([100.]))
        with self.assertRaisesRegex(ValueError, "zero likelihood under raw support"):
            fit_raw_time_kl(train, invalid, architecture=arch, training=training,
                            optimization=optimization, device="cpu")
        shared, _ = fit_raw_time_kl(train, invalid, architecture=arch, training=training,
            optimization=optimization, device="cpu", upper_bound=110.)
        self.assertTrue(math.isfinite(shared.fit_info["validation_nll"]))
        self.assertEqual(shared.fit_info["upper_bound"], 110.)
        self.assertAlmostEqual(shared.fit_info["max_observation"], 2.)
        for upper in (2.1, 0., math.inf, math.nan):
            with self.subTest(upper_bound=upper), self.assertRaisesRegex(ValueError, "upper_bound"):
                fit_raw_time_kl(train, invalid, architecture=arch, training=training,
                    optimization=optimization, device="cpu", upper_bound=upper)
        fixed, history = fit_raw_time_kl(train, architecture=replace(arch, regularization_strength=0.),
            training=training, optimization=replace(optimization, learn_splits=False),
            device="cpu", fixed_epochs=3)
        self.assertEqual(len(history), 3)
        self.assertAlmostEqual(fixed.fit_info["upper_bound"], 2.2)
        torch.testing.assert_close(fixed.rho(train[0]), torch.full((32, 3), .5))
        self.assertFalse(any(p.requires_grad for p in fixed.split_head.parameters()))
        self.assertFalse(any("validation_nll" in r for r in history))
        with self.assertRaises(ValueError):
            fit_raw_time_kl(train, train, training=training, device="cpu", fixed_epochs=3)

    def test_common_support_covers_all_35_default_cv_datasets(self):
        arch = Architecture(depth=3, private_width=8, private_layers=1)
        fold_local_failures = []
        for case in existing.ALL_CASES:
            data, _ = tuning_data(case)
            upper = observed_upper_bound(data.bounds(data.train))
            model = RawTimeKLTree(arch, upper)
            for fold, (train, valid) in enumerate(existing.cv_splits(data.train, 321), 1):
                with self.subTest(case=case.id, fold=fold):
                    _, left, right = data.bounds(valid)
                    local_upper = observed_upper_bound(data.bounds(train))
                    outside = torch.where(left == right, left > local_upper, left >= local_upper)
                    if outside.any():
                        fold_local_failures.append((case.id, fold, int(outside.sum())))
                    score = _evaluate(model, model.prepare(*data.bounds(valid)))
                    self.assertTrue(math.isfinite(score))
        self.assertEqual(fold_local_failures, [
            ("ph_exact_q0.2", 3, 2), ("lognormal_exact_q0.2", 4, 1),
            ("lognormal_right_q0.2", 2, 1)])

    def test_tuning_freezes_pool_support_without_test_oracle(self):
        case = existing.select_cases("ph_exact_q0.2")[0]
        data, seed = tuning_data(case, n_samples=128)
        # A lone training-pool outlier must be covered when held out by CV.
        data.L[data.train[0]] = data.R[data.train[0]] = 100.
        # Neither held-out observations nor latent event times may set support.
        data.L[data.test] = data.R[data.test] = 1e6
        data.T.fill_(1e9)
        upper = observed_upper_bound(data.bounds(data.train))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            space = root/"space.json"
            space.write_text(json.dumps(dict(search=dict(depth_min=2, depth_max=2, widths=[8],
                shared_layers_max=1, private_layers_max=1, strength_max=1.),
                baseline=dict(depth=2, shared_width=8, private_width=8, private_layers=1,
                              regularization_strength=1.))))
            with patch("regularization.tuning.main.tuning_data", return_value=(data, seed)):
                tune(["--cases", case.id, "--output-dir", str(root), "--n-samples", "128",
                    "--device", "cpu", "--threads", "1", "--trials", "2", "--initial-trials", "2",
                    "--folds", "4", "--epochs", "3", "--warmup-epochs", "1", "--validation-every", "1",
                    "--batch-size", "64", "--grid-size", "8", "--no-cuda-graphs", "--search-space", str(space)])
            request = json.loads((root/"studies"/case.id/"request.json").read_text())["request"]
            self.assertEqual(request["support"], dict(policy="tuning_train_pool", upper_bound=upper))
            folds = list((root/"studies"/case.id).glob("**/fold_*.json"))
            self.assertEqual(len(folds), 20)
            for path in folds:
                record = json.loads(path.read_text())
                self.assertEqual(record["upper_bound"], upper)
                self.assertTrue(math.isfinite(record["validation_nll"]))
            for method in METHODS:
                result = json.loads((root/method/case.id/"result.json").read_text())
                self.assertEqual(result["status"], "ok")
                self.assertTrue(math.isfinite(result["cv_score"]))
                self.assertEqual(result["fit_info"]["upper_bound"], upper)
                self.assertTrue(result["test_nll_infinite"])
                self.assertIsNone(result["test_nll"])

    def test_infinite_test_nll_is_strict_json_and_sse_remains_available(self):
        case = existing.select_cases("ph_exact_q0.2")[0]
        data, _ = tuning_data(case, n_samples=64)
        model = RawTimeKLTree(upper_bound=.0001)
        metrics = evaluate(model, data, 8)
        self.assertIsNone(metrics["test_nll"])
        self.assertTrue(metrics["test_nll_infinite"])
        json.dumps(metrics, allow_nan=False)

    def test_small_tuning_simulation_resume_and_summary(self):
        try:
            optuna_module()
        except ImportError:
            self.skipTest("Optuna is required for the integration smoke test")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tuning, testing = root/"tuning", root/"testing"
            space = root/"space.json"
            space.write_text(json.dumps(dict(search=dict(depth_min=2, depth_max=2, widths=[8],
                shared_layers_max=1, private_layers_max=1, strength_max=1.),
                baseline=dict(depth=2, shared_width=8, private_width=8, private_layers=1,
                              regularization_strength=1.))))
            case_id = "po_interval_q0.2"
            tune_args = ["--cases", case_id, "--output-dir", str(tuning), "--n-samples", "128",
                "--device", "cpu", "--threads", "1", "--trials", "2", "--initial-trials", "2",
                "--folds", "2", "--epochs", "3", "--warmup-epochs", "1", "--validation-every", "1",
                "--batch-size", "64", "--grid-size", "8", "--no-cuda-graphs", "--search-space", str(space)]
            tune(tune_args)
            cases = existing.select_cases(case_id)
            tables = load_hyperparameters(tuning/"hyper_para", cases, METHODS)
            for method in METHODS:
                spec = tables[method][case_id]
                self.assertGreater(spec["refit_epochs"], 1)
                self.assertEqual(spec["source_sha256"], source_hash())
            with patch("regularization.tuning.main.fit_raw_time_kl", side_effect=AssertionError("unexpected refit")):
                tune(tune_args)
            sim_args = ["run", "--cases", case_id, "--hyper-dir", str(tuning/"hyper_para"),
                "--output-dir", str(testing), "--replications", "2", "--n-samples", "128",
                "--device", "cpu", "--threads", "1", "--grid-size", "8"]
            simulate(sim_args)
            paths = list(testing.glob("*/*/replication_*.json"))
            self.assertEqual(len(paths), 6)
            for rep in (1, 2):
                records = [json.loads((testing/m/case_id/f"replication_{rep:04d}.json").read_text()) for m in METHODS]
                self.assertEqual(len({r["dataset_sha256"] for r in records}), 1)
                self.assertTrue(all(r["status"] == "ok" for r in records))
                self.assertTrue(all(r["fit_info"]["epochs_run"] == r["refit_epochs"] for r in records))
            with (testing/"tables"/f"{case_id}.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["replication"] for r in rows], ["1", "2", "average", "standard_deviation"])
            before = {p: p.read_bytes() for p in paths}
            with patch("regularization.testing.simulation.fit_raw_time_kl", side_effect=AssertionError("unexpected refit")):
                simulate(sim_args)
            self.assertEqual(before, {p: p.read_bytes() for p in paths})
            with self.assertRaisesRegex(ValueError, "match tuning"):
                simulate([*sim_args, "--n-samples", "129"])
            changed = json.loads(paths[0].read_text())
            changed["grid_size"] = 9
            paths[0].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "different datasets|mixed configurations"):
                summarize(testing, cases, METHODS, 2)
            paths[0].write_bytes(before[paths[0]])
            table = testing/"tables"/f"{case_id}.csv"
            report_before = table.read_bytes()
            paths[0].unlink()
            complete, pending = summarize(testing, cases, METHODS, 2)
            self.assertFalse(complete)
            self.assertEqual(pending, [case_id])
            self.assertEqual(table.read_bytes(), report_before)
            summarize(testing, cases, METHODS, 2, allow_incomplete=True)
            with table.open() as stream:
                partial = list(csv.DictReader(stream))
            self.assertEqual(len(partial), 4)


if __name__ == "__main__":
    unittest.main()
