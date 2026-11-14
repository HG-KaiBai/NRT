"""Endpoint, aggregation, pairing and historical-refit regression checks."""
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import simulation as sim
from .compare_structures import compare
from ..unified_nll.nll_cv_train import NLLCVOptimization
from ..unified_nll.train import TrainingConfig
from ..unified_nll.tuning_model import Architecture
from ..workflow import METHODS, select_cases, source_hash


class TauBiasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_absolute_value_precedes_subject_average_and_uses_training_boundary(self):
        data = SimpleNamespace(x=torch.zeros(4, 2), train=torch.tensor([0, 1]),
            test=torch.tensor([2, 3]), tau=99., max_observation=lambda indices: 3.125,
            truth=lambda indices, times: torch.tensor([[.2], [.8]], dtype=torch.float64))

        class Model:
            def predict_survival(self, x, times):
                self.times = times
                return torch.tensor([[.5], [.5]], dtype=torch.float64)

        model = Model()
        metric = sim.evaluate_tau_bias(model, data)
        self.assertEqual(metric["survival_tau"], 3.125)
        self.assertEqual(model.times.dtype, torch.float64)
        self.assertAlmostEqual(metric["survival_tau_absolute_bias"], .3)
        self.assertAlmostEqual(metric["survival_tau_estimate_mean"]-metric["survival_tau_truth_mean"], 0.)
        data.tau = None  # Exact cases use the same training-boundary definition.
        self.assertEqual(metric, sim.evaluate_tau_bias(model, data))

    def test_nonfinite_survival_is_rejected(self):
        data = SimpleNamespace(x=torch.zeros(1, 2), train=torch.tensor([0]), test=torch.tensor([0]),
            max_observation=lambda indices: 1., truth=lambda indices, times: torch.ones(1, 1))
        model = SimpleNamespace(predict_survival=lambda x, times: torch.full((1, 1), float("nan")))
        with self.assertRaisesRegex(ValueError, "finite"):
            sim.evaluate_tau_bias(model, data)

    def records(self, directory, *, bias=True):
        case = select_cases("ph_right_q0.2")[0]
        for replication, value in ((1, .1), (2, .3)):
            for method in METHODS:
                saved = dict(status="ok", case_id=case.id, method=method, replication=replication,
                    tuning_hash=method, parameters={}, training_config={}, optimization={}, refit_epochs=2,
                    source_sha256="model", simulation_source_sha256="simulation", grid_size=3,
                    data_seed=replication, dataset_sha256=str(replication), n_samples=10, tau=9., mu=0., r=None,
                    grid_max=8., right_censoring_rate=.2, average_grid_sse=replication*2., request_hash=str(replication))
                if bias:
                    saved.update(survival_tau=8., survival_tau_absolute_bias=value,
                        survival_tau_estimate_mean=.5, survival_tau_truth_mean=.4,
                        tau_bias_evaluation=sim.bias_provenance(saved["request_hash"]))
                sim.atomic_json(sim.result_path(Path(directory), method, case, replication), saved)
        return case

    def read_table(self, path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_summary_mean_sample_sd_and_legacy_missing_metric(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            case = self.records(output)
            complete, pending = sim.summarize(output, [case], METHODS, 2, metric="survival_tau_absolute_bias")
            self.assertEqual((complete, pending), ([case.id], []))
            path = output/"tables_tau_absolute_bias"/f"{case.id}.csv"
            rows = self.read_table(path)
            self.assertEqual([row["replication"] for row in rows], ["1", "2", "average", "standard_deviation"])
            self.assertAlmostEqual(float(rows[-2][METHODS[0]]), .2)
            self.assertAlmostEqual(float(rows[-1][METHODS[0]]), math.sqrt(.02))
            original = path.read_bytes()
            self.records(output, bias=False)
            self.assertEqual(sim.summarize(output, [case], METHODS, 2, metric="survival_tau_absolute_bias"), ([], [case.id]))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(sim.summarize(output, [case], METHODS, 2), ([case.id], []))
            sim.summarize(output, [case], METHODS, 2, metric="survival_tau_absolute_bias", allow_incomplete=True)
            self.assertEqual(self.read_table(path)[-2][METHODS[0]], "")

    def test_cross_structure_comparison_checks_paired_data(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = {name: root/name for name in ("finite", "body_tail", "root_tail")}
            for run in runs.values():
                case = self.records(run)
            self.assertEqual(compare(runs, root/"comparison", [case], METHODS, 2), [])
            rows = self.read_table(root/"comparison/tables_tau_absolute_bias"/f"{case.id}.csv")
            self.assertEqual(len(rows[-1]), 10)
            self.assertAlmostEqual(float(rows[-2]["finite/"+METHODS[0]]), .2)
            path = sim.result_path(runs["root_tail"], METHODS[0], case, 2)
            saved = json.loads(path.read_text())
            saved["dataset_sha256"] = "unpaired"
            sim.atomic_json(path, saved)
            with self.assertRaisesRegex(ValueError, "different data"):
                compare(runs, root/"comparison", [case], METHODS, 2)

    def test_backfill_refits_preserves_sse_and_resumes_without_fitting(self):
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=Path(tmp), n_samples=40, base_seed=123, grid_size=8, device="cpu")
            case = select_cases("ph_right_q0.2")[0]
            method = METHODS[0]
            spec = dict(n_samples=40, base_seed=123, source_sha256=source_hash(), tuning_hash="frozen",
                parameters=asdict(Architecture(depth=2, x_dim=6, private_width=4, private_layers=1,
                                               shared_width=4, shared_layers=1, regularization_strength=0.)),
                training_config=asdict(TrainingConfig(epochs=2, batch_size=16, cuda_graphs=False)),
                optimization=asdict(NLLCVOptimization(warmup_epochs=0, learn_splits=False)),
                refit_epochs=2, penalty="partition_kl", tau=4.5, mu=0., r=None)
            # Recreate the old driver's request format with the actual numerical model.
            with patch.object(sim, "simulation_source_hash", return_value=sim.LEGACY_SIMULATION_SHA256):
                old = sim.fit_replication(args, method, case, 1, spec)
            self.assertEqual(old["status"], "ok", old.get("error"))
            old = {k: v for k, v in old.items() if not k.startswith("survival_tau") and k != "tau_bias_evaluation"}
            path = sim.result_path(args.output_dir, method, case, 1)
            sim.atomic_json(path, old)
            updated = sim.backfill_tau_bias(args, method, case, 1, spec)
            self.assertTrue(sim.has_tau_bias(updated), updated.get("tau_bias_evaluation"))
            for key, value in old.items():
                self.assertEqual(updated[key], value, key)
            self.assertAlmostEqual(updated["tau_bias_evaluation"]["refit_sse_difference"], 0.)
            with patch.object(sim, "fit_raw_time_kl", side_effect=AssertionError("must reuse")):
                self.assertEqual(sim.backfill_tau_bias(args, method, case, 1, spec), updated)
            old["dataset_sha256"] = "wrong"
            sim.atomic_json(path, old)
            with patch.object(sim, "fit_raw_time_kl", side_effect=AssertionError("must validate before fitting")):
                invalid = sim.backfill_tau_bias(args, method, case, 1, spec)
            self.assertEqual(invalid["status"], "ok")
            self.assertEqual(invalid["average_grid_sse"], old["average_grid_sse"])
            self.assertEqual(invalid["tau_bias_evaluation"]["status"], "error")
            self.assertIn("data do not match", invalid["tau_bias_evaluation"]["error"])


if __name__ == "__main__":
    unittest.main()
