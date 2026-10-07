"""Regression checks for the five failed production Cox/AFT fits."""
import subprocess
import unittest

import torch

import workflow as w
from .aft import ParametricAFT, r_backend_available as aft_available
from .cox import CoxPH, r_backend_available as cox_available


class BackendEdgeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @unittest.skipUnless(cox_available(), 'R Cox backend unavailable')
    def test_cox_survival_at_last_support_endpoint(self):
        for q, seed, tau in ((0.2, 1060167, 4.574094667177786),
                             (0.2, 1060554, 4.574094667177786),
                             (0.5, 1070169, 1.3513375990573424)):
            with self.subTest(seed=seed):
                data = w.generate(w.Case('ph', 'interval', q), 5000, seed, tau)
                endpoint = data.max_observation(data.train)
                grid = torch.tensor([0.0, endpoint * 0.99, endpoint, endpoint * 1.01])
                with CoxPH() as model:
                    model.fit(*data.bounds(data.train), censor_type='interval')
                    survival = model.predict_survival(data.x[data.test[:16]], grid)
                self.assertTrue(bool(torch.isfinite(survival).all()))
                torch.testing.assert_close(survival[:, 0], torch.ones(16))
                torch.testing.assert_close(survival[:, -2:], torch.zeros(16, 2))
                self.assertTrue(bool((survival.diff(dim=1) <= 0).all()))

    @unittest.skipUnless(aft_available(), 'R AFT backend unavailable')
    def test_aft_fits_previously_zero_time_censoring_seeds(self):
        for case, seed, tau, distribution in (
            (w.Case('lognormal', 'right', 0.2), 1100608, 3.1234899682837227, 'lognormal'),
            (w.Case('multimodal_separate', 'interval', 0.2), 1300372, 18.006402979690264, 'weibull'),
        ):
            with self.subTest(case=case.id):
                data = w.generate(case, 5000, seed, tau)
                x, left, right = data.bounds(data.train)
                self.assertFalse(bool(((data.L == 0) & torch.isinf(data.R)).any()))
                grid = torch.linspace(0, data.max_observation(data.train), 8)
                with ParametricAFT(distribution=distribution) as model:
                    model.fit(x, left, right, censor_type=case.censor_type)
                    survival = model.predict_survival(data.x[data.test[:16]], grid)
                    process = subprocess.run([
                        model.rscript, '--vanilla', '-e',
                        'a <- readRDS(commandArgs(TRUE)[1]); cat(length(a$fit$linear.predictors))',
                        str(model._model_path),
                    ], capture_output=True, text=True, check=True)
                    self.assertEqual(int(process.stdout), len(x))
                self.assertTrue(bool(torch.isfinite(survival).all()))
                torch.testing.assert_close(survival[:, 0], torch.ones(16))
                self.assertTrue(bool((survival.diff(dim=1) <= 0).all()))


if __name__ == '__main__':
    unittest.main()
