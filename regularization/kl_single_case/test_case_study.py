"""Data identity, no test input to training, and saved KL prediction checks."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pycode.neural_only import testdata2
from pycode.unified_nll import simulation
from pycode.unified_nll.data_compatibility import verify_generator
from pycode.unified_nll.nll_cv_train import NLLCVOptimization
from pycode.unified_nll.raw_time.model import RawTimeKLTree, fit_raw_time_kl
from pycode.unified_nll.train import TrainingConfig
from pycode.unified_nll.tuning_model import Architecture
from .case_study import (CaseConfig, generate_case, load_selected, prepare_output,
                         train_case, evaluate_case, save_case, plot_case)


class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_canonical_generator_and_frozen_kl_all_35_settings(self):
        self.assertTrue(verify_generator()["passed"])
        cases = [(s, m) for s in ("ph", "lognormal", "po") for m in ("separate",)]
        cases += [("multimodal", "separate"), ("multimodal", "closer")]
        count = 0
        for scenario, mode in cases:
            for censor in ("exact", "right", "interval", "current"):
                for q in ((0.2,) if censor == "exact" else (0.2, 0.5)):
                    config = CaseConfig(n_samples=96, scenario=scenario, censor_type=censor,
                                        censor_q=q, multimodal_case=mode)
                    data = generate_case(config)
                    kwargs = dict(n=96, x_dim=6, q=q, censor_type=censor, device="cpu")
                    for module in (testdata2, simulation):
                        torch.manual_seed(config.seed)
                        expected = (module.data_reg_multimodal(**kwargs, multimodal_case=mode)
                                    if scenario == "multimodal" else module.data_reg(**kwargs, scenario=scenario))
                        for actual, target in zip(data["raw"], expected):
                            torch.testing.assert_close(actual, target, rtol=0, atol=0)
                    for actual, source in zip(data["train"], data["raw"][:3]):
                        torch.testing.assert_close(actual, source[:76], rtol=0, atol=0)
                    for actual, source in zip(data["test"], data["raw"][:3]):
                        torch.testing.assert_close(actual, source[76:], rtol=0, atol=0)
                    self.assertEqual(float(data["grid"][-1]), data["max_obs"])
                    count += 1
        self.assertEqual(count, 35)

    def test_generator_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            changed = Path(temp) / "simulation.py"
            changed.write_text(Path(simulation.__file__).read_text()+"\nDRIFT = True\n")
            with patch.object(simulation, "__file__", str(changed)):
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    verify_generator()

    def test_saved_selection_and_dimension_guard(self):
        arch, epochs = load_selected()
        self.assertEqual(epochs, 35)
        self.assertEqual(arch.x_dim, 6)
        with tempfile.TemporaryDirectory() as temp:
            old_path = Path(temp) / "selected_parameters.csv"
            old_path.write_text("setting,method\nph_right_q0.5,adaptive_nll_cv\n")
            old_path.with_name("config.json").write_text('{"support":"old_infinite_normalized"}')
            with self.assertRaisesRegex(ValueError, "raw-time finite support"):
                load_selected(path=old_path)
        with self.assertRaisesRegex(ValueError, "X_DIM"):
            load_selected(CaseConfig(x_dim=5))

    def test_kl_fit_no_test_input_and_reproducible_saved_predictions(self):
        config = CaseConfig(n_samples=60, grid_size=32, scenario="multimodal", multimodal_case="closer")
        data = generate_case(config)
        arch = Architecture(depth=2, private_width=8, private_layers=2)
        training = TrainingConfig(epochs=3, batch_size=16, validation_every=1, cuda_graphs=False)
        opt = NLLCVOptimization(warmup_epochs=1)
        with patch("pycode.kl_single_case.case_study.fit_raw_time_kl", wraps=fit_raw_time_kl) as fit:
            model, history, seconds = train_case(data, config, arch, 3, training=training, optimization=opt)
            self.assertIs(fit.call_args.args[0], data["train"])
            self.assertNotIn("valid", fit.call_args.kwargs)
            self.assertEqual(fit.call_args.kwargs["fixed_epochs"], 3)
        result, estimated, errors = evaluate_case(model, data, seconds)
        self.assertFalse(np.isnan(result.test_nll.iloc[0]))
        self.assertAlmostEqual(float(model.upperbound),1.1*data["max_obs"])
        self.assertAlmostEqual(float(errors.mean()), result.average_grid_sse.iloc[0], places=6)
        self.assertTrue(bool((estimated[:, 1:] <= estimated[:, :-1]+1e-6).all()))
        with tempfile.TemporaryDirectory() as temp:
            prepare_output(temp, config, arch, training, opt, 3, "cpu")
            save_case(temp, config, data, model, history, result, estimated, errors)
            with np.load(Path(temp)/"generated_data.npz") as saved:
                np.testing.assert_array_equal(saved["raw_0"], data["raw"][0].numpy())
            state = torch.load(Path(temp)/"model.pt", weights_only=True)
            restored = RawTimeKLTree(Architecture(**state["architecture"]), state["upper_bound"], opt)
            restored.load_state_dict(state["state_dict"])
            torch.testing.assert_close(restored.predict_survival(data["test"][0], data["grid"]), estimated)
            figures = plot_case(model, data, estimated, history, output=temp)
            self.assertEqual(set(figures), {"survival", "training", "density"})
            import matplotlib.pyplot as plt
            for name, fig in figures.items():
                self.assertTrue((Path(temp)/f"{name}.png").exists())
                plt.close(fig)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                prepare_output(temp, replace(config, seed=999), arch, training, opt, 3, "cpu")


if __name__ == "__main__":
    unittest.main()
