"""Scenario selection and real-R prediction checks for parametric AFT."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import unittest

import torch

from . import ParametricAFT, distribution_for_scenario, r_backend_available
from ...testdata2 import data_reg, data_reg_multimodal


class AFTFamilyTests(unittest.TestCase):
    def test_notebook_uses_matching_families(self):
        path = Path(__file__).parents[2] / "benchmark_methods.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        # Exercise the notebook's actual AFT construction without initializing
        # the other benchmarks or their optional backends.
        source = "".join(notebook["cells"][5]["source"])
        start = source.index('    "AFT": ParametricAFT(')
        end = source.index("    ),", start) + len("    ),")
        expression = "methods = {\n" + source[start:end] + "\n}"
        for scenario, expected in {
            "ph": "exponential", "lognormal": "lognormal",
            "po": "loglogistic", "multimodal": "weibull",
        }.items():
            with self.subTest(scenario=scenario):
                namespace = dict(ParametricAFT=ParametricAFT,
                                 distribution_for_scenario=distribution_for_scenario,
                                 SCENARIO=scenario)
                exec(expression, namespace)
                self.assertEqual(namespace["methods"]["AFT"].distribution, expected)

    def test_invalid_family_is_rejected(self):
        with self.assertRaises(ValueError):
            ParametricAFT(distribution="logistic")  # Would model T instead of log(T).
        with self.assertRaises(ValueError):
            distribution_for_scenario("unknown")

    @unittest.skipUnless(r_backend_available(), "R survival backend unavailable")
    def test_all_schemes_and_predictions_against_r_distribution_oracle(self):
        for scenario in ("ph", "lognormal", "po", "multimodal"):
            for scheme in ("exact", "right", "current", "interval"):
                with self.subTest(scenario=scenario, scheme=scheme):
                    torch.manual_seed(91)
                    kwargs = dict(n=200, q=0.2, censor_type=scheme)
                    if scenario == "multimodal":
                        x, L, R, *_ = data_reg_multimodal(**kwargs)
                    else:
                        x, L, R, *_ = data_reg(scenario=scenario, **kwargs)
                    grid = torch.tensor([0.0, 1e-8, 0.1, 0.5, 1.0, 2.0, 10.0, 1e20, torch.inf])
                    with ParametricAFT(distribution=distribution_for_scenario(scenario)) as model:
                        model.fit(x[:160], L[:160], R[:160], censor_type=scheme)
                        prediction = model.predict_survival(x[160:164], grid)
                        self.assertEqual(tuple(prediction.shape), (4, len(grid)))
                        self.assertTrue(bool(torch.isfinite(prediction).all()))
                        self.assertTrue(bool(((prediction >= 0) & (prediction <= 1)).all()))
                        self.assertTrue(bool((prediction.diff(dim=1) <= 0).all()))
                        torch.testing.assert_close(prediction[:, 0], torch.ones(4))
                        torch.testing.assert_close(prediction[:, -1], torch.zeros(4))

                        directory = Path(model._temporary_directory.name)
                        # Independent oracle: survival::psurvreg rather than the
                        # backend's explicit pexp/pweibull/plnorm/plogis branches.
                        script = directory / "oracle.R"
                        script.write_text(
                            "suppressPackageStartupMessages(library(survival))\n"
                            "args <- commandArgs(trailingOnly=TRUE)\n"
                            "a <- readRDS(args[1])\n"
                            "x <- read.csv(args[2])\n"
                            "g <- scan(args[3], sep=',', quiet=TRUE)\n"
                            "lp <- predict(a$fit, newdata=x, type='lp')\n"
                            "s <- sapply(g, function(t) {\n"
                            "  if (t == 0) return(rep(1, length(lp)))\n"
                            "  if (is.infinite(t)) return(rep(0, length(lp)))\n"
                            "  1-survival::psurvreg(t, mean=lp, scale=a$fit$scale,\n"
                            "    distribution=a$distribution)\n"
                            "})\n"
                            "write.table(s, args[4], sep=',', row.names=FALSE, col.names=FALSE)\n",
                            encoding="utf-8",
                        )
                        oracle_path = directory / "oracle.csv"
                        process = subprocess.run(
                            [model.rscript, "--vanilla", str(script), str(model._model_path),
                             str(directory / "prediction.csv"), str(directory / "grid.csv"),
                             str(oracle_path)], capture_output=True, text=True, check=False,
                        )
                        self.assertEqual(process.returncode, 0, process.stderr)
                        with oracle_path.open(newline="", encoding="utf-8") as source:
                            oracle = torch.tensor([[float(v) for v in row] for row in csv.reader(source)])
                        torch.testing.assert_close(prediction, oracle, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
