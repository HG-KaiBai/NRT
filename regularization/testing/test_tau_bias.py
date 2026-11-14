"""Body-tail endpoint metrics and summary regression checks."""
import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from . import simulation as sim
from ..workflow import METHODS, select_cases


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

if __name__ == "__main__":
    unittest.main()
