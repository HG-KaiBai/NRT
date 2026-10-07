from __future__ import annotations

import unittest

import torch

from .aft import ParametricAFT, r_backend_available as aft_backend_available
from .common import (
    average_grid_sse,
    benchmark_model,
    lognormal_survival,
    multimodal_survival,
)
from .cox import CoxPH, r_backend_available as cox_backend_available
from .ctree import CTree, r_backend_available as ctree_backend_available
from .deepsurv import (
    DeepSurv,
    DeepSurvCUDA,
    PAPER_GAUSSIAN_PARAMETERS,
)
from .deepsurv.cuda_model import _Configuration
from .deepsurv.losses import cox_partial_log_likelihood
from .deep_distribution_regression import DeepDistributionRegression
from .iccforest import ICCForest, r_backend_available
from .nncde import NNCDE
from ..testdata2 import (
    MULTIMODAL_SCALE_MULTIPLIER_CASES,
    MULTIMODAL_SHAPES,
    data_reg,
    data_reg_multimodal,
)


class BenchmarkMethodTests(unittest.TestCase):
    @staticmethod
    def assert_survival_curve(curve: torch.Tensor, rows: int, columns: int) -> None:
        assert curve.shape == (rows, columns)
        assert bool(torch.isfinite(curve).all())
        assert bool(((curve >= 0.0) & (curve <= 1.0)).all())
        assert bool((curve[:, 1:] <= curve[:, :-1] + 1e-6).all())

    def test_average_grid_sse_matches_notebook_definition(self) -> None:
        grid = torch.tensor([0.0, 1.0, 2.0])
        shape = torch.ones(2)
        scale = torch.ones(2)
        truth = torch.exp(-grid).repeat(2, 1)
        estimated = truth.clone()
        estimated[0, 1] += 0.5
        expected = torch.tensor((0.25 + 0.0) / 2.0)
        self.assertTrue(torch.allclose(average_grid_sse(estimated, shape, scale, grid), expected))

    def test_multimodal_truth_averages_components(self) -> None:
        grid = torch.tensor([0.0, 1.0, 2.0])
        weights = torch.tensor([[0.25, 0.75]])
        scales = torch.tensor([[1.0, 2.0]])
        expected = (
            0.25 * torch.exp(-((grid / 1.0) ** MULTIMODAL_SHAPES[0]))
            + 0.75 * torch.exp(-((grid / 2.0) ** MULTIMODAL_SHAPES[1]))
        )
        self.assertTrue(torch.allclose(multimodal_survival(weights, scales, grid)[0], expected))

    def test_lognormal_survival_matches_distribution(self) -> None:
        grid = torch.tensor([0.0, 0.5, 1.0, 2.0])
        shape = torch.tensor([0.5, 1.0])
        scale = torch.tensor([1.0, 2.0])
        expected = 1.0 - torch.distributions.LogNormal(
            torch.log(scale).unsqueeze(1), shape.unsqueeze(1)
        ).cdf(grid[1:].unsqueeze(0))
        actual = lognormal_survival(shape, scale, grid)
        torch.testing.assert_close(actual[:, 0], torch.ones(2))
        torch.testing.assert_close(actual[:, 1:], expected)

    def test_multimodal_generator_censoring_and_intervals(self) -> None:
        for multimodal_case in MULTIMODAL_SCALE_MULTIPLIER_CASES:
            for censor_type in ("exact", "right", "interval", "current"):
                with self.subTest(multimodal_case=multimodal_case, censor_type=censor_type):
                    torch.manual_seed(13)
                    x, L, R, event_time, weights, scales, component = data_reg_multimodal(
                        n=4000, q=0.3, censor_type=censor_type, multimodal_case=multimodal_case,
                    )
                    self.assertEqual(x.shape, (4000, 6))
                    self.assertEqual(weights.shape, (4000, 2))
                    self.assertEqual(weights.shape, scales.shape)
                    torch.testing.assert_close(weights, torch.full((4000, 2), 0.5))
                    self.assertTrue(bool(((L <= event_time) & (event_time <= R)).all()))
                    self.assertTrue(bool(((component >= 0) & (component < 2)).all()))
                    if censor_type != "exact":
                        self.assertLess(abs(torch.isinf(R).float().mean().item() - 0.3), 0.04)

    def test_multimodal_benchmark_path(self) -> None:
        class ConstantEstimator:
            supported_censor_types = frozenset({"exact"})

            def fit(self, x, L, R, *, censor_type):
                return self

            def predict_survival(self, x, times):
                return torch.exp(-times)[None, :].expand(len(x), -1)

        result = benchmark_model(
            ConstantEstimator(), method="constant", scenario="multimodal",
            censor_type="exact", n_samples=32, grid_size=16
        )
        self.assertTrue(torch.isfinite(torch.tensor(result.average_grid_sse)))
        self.assertGreater(result.average_grid_sse, 0.0)

    def test_lognormal_benchmark_path(self) -> None:
        class ConstantEstimator:
            supported_censor_types = frozenset({"exact"})

            def fit(self, x, L, R, *, censor_type):
                return self

            def predict_survival(self, x, times):
                return torch.exp(-times)[None, :].expand(len(x), -1)

        result = benchmark_model(
            ConstantEstimator(),
            method="constant",
            scenario="lognormal",
            censor_type="exact",
            n_samples=32,
            grid_size=16,
        )
        self.assertTrue(torch.isfinite(torch.tensor(result.average_grid_sse)))
        self.assertGreater(result.average_grid_sse, 0.0)

    def test_nncde_exact_and_right(self) -> None:
        for censor_type in ("exact", "right"):
            torch.manual_seed(7)
            x, L, R, *_ = data_reg(n=32, censor_type=censor_type)
            model = NNCDE(epochs=1, batch_size=8, hidden_nodes=8, device="cpu")
            model.fit(x[:24], L[:24], R[:24], censor_type=censor_type)
            grid = torch.linspace(0.0, 2.0, 12)
            curve = model.predict_survival(x[24:], grid)
            self.assert_survival_curve(curve, 8, 12)

    def test_ddr_exact(self) -> None:
        torch.manual_seed(8)
        x, L, R, *_ = data_reg(n=40, censor_type="exact")
        model = DeepDistributionRegression(
            epochs=1, batch_size=8, hidden_nodes=8, n_cutpoints=5, device="cpu"
        )
        model.fit(x[:32], L[:32], R[:32], censor_type="exact")
        grid = torch.linspace(0.0, 2.0, 12)
        curve = model.predict_survival(x[32:], grid)
        self.assert_survival_curve(curve, 8, 12)

    @unittest.skipUnless(aft_backend_available(), "R AFT backend is unavailable")
    def test_aft_all_observation_schemes(self) -> None:
        for censor_type in ("exact", "right", "interval", "current"):
            with self.subTest(censor_type=censor_type):
                torch.manual_seed(11)
                x, L, R, *_ = data_reg(n=40, censor_type=censor_type)
                with ParametricAFT(max_iter=10) as model:
                    model.fit(x[:32], L[:32], R[:32], censor_type=censor_type)
                    curve = model.predict_survival(
                        x[32:], torch.linspace(0.0, 2.0, 12)
                    )
                self.assert_survival_curve(curve, 8, 12)

    @unittest.skipUnless(aft_backend_available(), "R AFT backend is unavailable")
    def test_aft_lognormal_and_multimodal_single_weibull(self) -> None:
        torch.manual_seed(12)
        x, L, R, *_ = data_reg(n=40, censor_type="right", scenario="lognormal")
        with ParametricAFT(distribution="lognormal", max_iter=10) as lognormal:
            lognormal.fit(x[:32], L[:32], R[:32], censor_type="right")
            curve = lognormal.predict_survival(
                x[32:], torch.linspace(0.0, 2.0, 12)
            )
        self.assert_survival_curve(curve, 8, 12)

        x, L, R, *_ = data_reg_multimodal(n=40, censor_type="right")
        with ParametricAFT(distribution="weibull", max_iter=10) as weibull:
            weibull.fit(x[:32], L[:32], R[:32], censor_type="right")
            curve = weibull.predict_survival(
                x[32:], torch.linspace(0.0, 2.0, 12)
            )
        self.assert_survival_curve(curve, 8, 12)

    def test_deepsurv_exact_and_right(self) -> None:
        for censor_type in ("exact", "right"):
            with self.subTest(censor_type=censor_type):
                torch.manual_seed(14)
                x, L, R, *_ = data_reg(n=40, censor_type=censor_type)
                model = DeepSurv(
                    hidden_layers=(8,), dropout=None, epochs=1, tune=False,
                    device="cpu"
                )
                model.fit(x[:32], L[:32], R[:32], censor_type=censor_type)
                curve = model.predict_survival(x[32:], torch.linspace(0.0, 2.0, 12))
                self.assert_survival_curve(curve, 8, 12)

    def test_deepsurv_uses_paper_gaussian_parameters(self) -> None:
        model = DeepSurv(device="cpu")
        self.assertIs(DeepSurv, DeepSurvCUDA)
        for name, expected in PAPER_GAUSSIAN_PARAMETERS.items():
            actual = getattr(model, name)
            self.assertEqual(actual, expected, name)
        self.assertFalse(model.tuning_parameters["performed_in_benchmark"])

    def test_deepsurv_stg_cox_loss(self) -> None:
        logits = torch.tensor([1001.0, 1000.0, 999.0], requires_grad=True)
        event = torch.tensor([1.0, 0.0, 1.0])
        loss = cox_partial_log_likelihood(logits, event)
        shifted = logits.detach() - logits.detach().max()
        expected = -(
            (shifted - torch.log(torch.cumsum(torch.exp(shifted), dim=0)))
            * event
        ).sum() / event.sum()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(loss.detach(), expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(NotImplementedError):
            cox_partial_log_likelihood(logits.detach(), event, ties="efron")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_deepsurv_cuda_exact_and_right(self) -> None:
        for censor_type in ("exact", "right"):
            with self.subTest(censor_type=censor_type):
                torch.manual_seed(141)
                x, L, R, *_ = data_reg(n=40, censor_type=censor_type)
                with DeepSurvCUDA(
                    hidden_layers=(8,), dropout=None, batch_norm=False,
                    l2_reg=0.01, epochs=2, device="cuda",
                ) as model:
                    model.fit(x[:32], L[:32], R[:32], censor_type=censor_type)
                    curve = model.predict_survival(
                        x[32:], torch.linspace(0.0, 2.0, 12)
                    )
                    assert model._model is not None
                    self.assertTrue(next(model._model.parameters()).is_cuda)
                self.assert_survival_curve(curve, 8, 12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_deepsurv_cuda_keeps_paper_gaussian_parameters(self) -> None:
        model = DeepSurvCUDA(device="cuda", tune=False)
        parameters = model.modeling_parameters
        self.assertEqual(
            parameters["hidden_layers"],
            list(PAPER_GAUSSIAN_PARAMETERS["hidden_layers"]),
        )
        for name in ("activation", "dropout", "batch_norm", "standardize"):
            self.assertEqual(parameters[name], PAPER_GAUSSIAN_PARAMETERS[name])
        for name in (
            "learning_rate", "lr_decay", "momentum", "l2_reg", "l1_reg",
            "epochs",
        ):
            self.assertEqual(parameters[name], PAPER_GAUSSIAN_PARAMETERS[name])
        self.assertEqual(parameters["optimizer"], "full-batch ADAM")
        self.assertEqual(parameters["device"], "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_deepsurv_cuda_matches_official_regularization_scope(self) -> None:
        model = DeepSurvCUDA(
            hidden_layers=(2,), batch_norm=False, dropout=None,
            l1_reg=0.25, l2_reg=999.0, epochs=1, device="cuda",
        )
        configuration = _Configuration(
            hidden_layers=(2,), learning_rate=0.01, lr_decay=0.0,
            momentum=0.0, l2_reg=0.5, dropout=None,
            activation="rectify", batch_norm=False, standardize=False,
        )
        network = model._make_model(3, configuration, seed_offset=0)
        with torch.no_grad():
            network.network[0].weight.fill_(100.0)
            network.network[-1].weight.copy_(
                torch.tensor([[2.0, -3.0]], device="cuda")
            )
        penalty = model._regularization(network, configuration)
        self.assertAlmostEqual(float(penalty.detach()), 7.75)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_deepsurv_cuda_supports_paper_adam_runner(self) -> None:
        torch.manual_seed(144)
        x, L, R, *_ = data_reg(n=32, censor_type="right")
        with DeepSurvCUDA(
            hidden_layers=(6,), activation="selu", dropout=0.1,
            batch_norm=False, l2_reg=0.01, epochs=2,
            optimizer="adam", device="cuda",
        ) as model:
            model.fit(x, L, R, censor_type="right")
            risk = model.predict_risk(x[:4])
            self.assertEqual(model.modeling_parameters["optimizer"], "full-batch ADAM")
            self.assertTrue(torch.isfinite(risk).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_deepsurv_cuda_default_model_is_covariate_dependent(self) -> None:
        torch.manual_seed(143)
        x = 2.0 * torch.rand(256, 10) - 1.0
        log_risk = torch.log(torch.tensor(5.0)) * torch.exp(
            -(x[:, 0].square() + x[:, 1].square()) / (2.0 * 0.5**2)
        )
        latent = torch.empty(256).exponential_(1.0 / 5.0) / torch.exp(log_risk)
        end_time = torch.quantile(latent, 0.5)
        lower = torch.minimum(latent, end_time)
        right = torch.where(latent < end_time, lower, torch.inf)
        with DeepSurvCUDA(seed=143, device="cuda") as model:
            model.fit(x, lower, right, censor_type="right")
            risk = model.predict_risk(x[:16])
            survival = model.predict_survival(
                x[:16], torch.linspace(0.0, float(end_time), 24)
            )
        self.assertGreater(float(risk.max() - risk.min()), 1e-5)
        self.assertGreater(
            float(survival[:, -1].max() - survival[:, -1].min()), 1e-6
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_deepsurv_cuda_tunes_on_gpu(self) -> None:
        torch.manual_seed(142)
        x, L, R, *_ = data_reg(n=36, censor_type="right")
        with DeepSurvCUDA(
            hidden_layers=(6,), dropout=None, batch_norm=False,
            l2_reg=0.01, epochs=1, tune=True,
            tuning_evals=2, cv_folds=2, tuning_epochs=1,
            device="cuda",
        ) as model:
            model.fit(x, L, R, censor_type="right")
            self.assertTrue(model.tuning_parameters["performed_in_benchmark"])
            self.assertEqual(
                len(model.tuning_parameters["cross_validation_fits"]), 4
            )
            self.assertTrue(
                all(
                    fit["device"] == "cuda"
                    for fit in model.tuning_parameters["cross_validation_fits"]
                )
            )
            self.assertEqual(model.modeling_parameters["device"], "cuda")

    def test_method_censoring_boundaries(self) -> None:
        x, L, R, *_ = data_reg(n=8, censor_type="interval")
        with self.assertRaises(ValueError):
            NNCDE(epochs=1).fit(x, L, R, censor_type="interval")
        with self.assertRaises(ValueError):
            DeepDistributionRegression(epochs=1).fit(x, L, R, censor_type="interval")
        with self.assertRaises(ValueError):
            DeepSurv(epochs=1, device="cpu").fit(x, L, R, censor_type="interval")

    @unittest.skipUnless(cox_backend_available(), "R Cox backend is unavailable")
    def test_cox_all_observation_schemes(self) -> None:
        for censor_type in ("exact", "right", "interval", "current"):
            with self.subTest(censor_type=censor_type):
                torch.manual_seed(15)
                x, L, R, *_ = data_reg(n=64, censor_type=censor_type)
                grid = torch.linspace(0.0, 2.0, 12)
                with CoxPH() as model:
                    model.fit(x[:51], L[:51], R[:51], censor_type=censor_type)
                    curve = model.predict_survival(x[51:], grid)
                    risk = model.predict_risk(x[51:])
                self.assert_survival_curve(curve, 13, 12)
                self.assertEqual(risk.shape, (13,))
                self.assertTrue(bool(torch.isfinite(risk).all()))

    @unittest.skipUnless(r_backend_available(), "R ICcforest backend is unavailable")
    def test_iccforest_interval(self) -> None:
        torch.manual_seed(9)
        x, L, R, *_ = data_reg(n=32, censor_type="interval")
        grid = torch.linspace(0.0, 2.0, 12)
        with ICCForest(ntree=2, tune_mtry=False) as model:
            model.fit(x[:24], L[:24], R[:24], censor_type="interval")
            curve = model.predict_survival(x[24:], grid)
        self.assert_survival_curve(curve, 8, 12)

    @unittest.skipUnless(ctree_backend_available(), "R ctree backend is unavailable")
    def test_ctree_all_observation_schemes(self) -> None:
        for censor_type in ("exact", "right", "interval", "current"):
            with self.subTest(censor_type=censor_type):
                torch.manual_seed(10)
                x, L, R, *_ = data_reg(n=40, censor_type=censor_type)
                grid = torch.linspace(0.0, 2.0, 12)
                with CTree(minsplit=10, minbucket=4) as model:
                    model.fit(x[:32], L[:32], R[:32], censor_type=censor_type)
                    curve = model.predict_survival(x[32:], grid)
                self.assert_survival_curve(curve, 8, 12)


if __name__ == "__main__":
    unittest.main()
