"""NLL feedback, fixed/adaptive training, pairing, resumption and artifact audits."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from .cv_train import CVOptimization,fit_cv
from .cv_search import CVSearchConfig
from .depth_experiment import _evaluation_data
from .experiment import ExperimentConfig
from .nll_cv_train import NLLCVOptimization,NLLCVTree,fit_nll_cv
from .nll_cv_search import METHODS,MATCHED,SEARCH_METHODS,tune_nll_cv,evaluate_nll_cv
from .nll_cv_report import audit_nll_cv,report_nll_cv,paired_sse
from .nll_cv_io import write_json
from .nll_cv_status import read_shared
from .train import TrainingConfig
from .tuning_model import Architecture


class NLLCVTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        x = torch.randn(24,6)
        left = torch.linspace(.1,2.,24)
        right = left.clone()
        right[::3] = torch.inf
        right[1::3] += .5
        self.data = x,left,right
        self.arch = Architecture(depth=3,shared_width=5,shared_layers=1,
            private_width=6,private_layers=2,regularization_strength=100000.)
        self.training = TrainingConfig(epochs=4,batch_size=8,validation_every=1,patience=4,cuda_graphs=False)
        self.opt = NLLCVOptimization(warmup_epochs=1)

    def test_atomic_cache_retry_preserves_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cache.json"
            original = Path.replace
            calls = []
            def locked_then_replace(source,target):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError("transient Windows reader lock")
                return original(source,target)
            value = dict(fingerprint="test",folds=[dict(fold=0,validation_nll=.5)])
            with patch.object(Path,"replace",locked_then_replace),patch("pycode.unified_nll.nll_cv_io.time.sleep"):
                write_json(path,value)
            assert len(calls) == 3
            assert json.loads(read_shared(path)) == value
            assert not path.with_suffix(".json.tmp").exists()

    def test_fixed_rho_all_depths_and_matched_mass_initialization(self):
        for depth in range(3,9):
            arch = replace(self.arch,depth=depth)
            torch.manual_seed(17)
            adaptive = NLLCVTree(arch,1.,self.opt)
            torch.manual_seed(17)
            fixed = NLLCVTree(arch,1.,replace(self.opt,learn_splits=False))
            for a,b in zip(adaptive.mass_encoder.parameters(),fixed.mass_encoder.parameters()):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            fixed.split_gate.fill_(1.)
            with torch.no_grad():
                fixed.split_head.bias.fill_(100.)
            assert torch.equal(fixed.rho(self.data[0]),torch.full((24,2**depth-1),.5))
            self.assertAlmostEqual(float(fixed.terms(fixed.prepare(*self.data))[1]),0.,places=10)
            assert all(not p.requires_grad for p in fixed.split_encoder.parameters())
            assert all(p.requires_grad for p in adaptive.parameters())

    def test_adaptive_matches_previous_numerics_and_nll_only_refit(self):
        kwargs = dict(architecture=self.arch,training=self.training,seed=5,device="cpu")
        with patch("pycode.unified_nll.cv_train.score_brier",side_effect=AssertionError("Brier called")):
            model,history = fit_nll_cv(self.data,self.data,optimization=self.opt,**kwargs)
        old,_ = fit_cv(self.data,self.data,torch.linspace(0,3,17),
            optimization=CVOptimization(warmup_epochs=1),**kwargs)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,old.state_dict()[key],atol=0,rtol=0)
        assert all("brier" not in k for row in history for k in row)
        for learned in (False,True):
            with patch("pycode.unified_nll.nll_cv_train._evaluate",side_effect=AssertionError("test stopping")):
                refit,h = fit_nll_cv(self.data,optimization=replace(self.opt,learn_splits=learned),fixed_epochs=3,**kwargs)
            self.assertEqual(refit.fit_info["epochs_run"],3)
            assert all("validation_nll" not in row for row in h)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_eager_parity_for_both_rho_methods(self):
        for learned in (False,True):
            kwargs = dict(architecture=self.arch,optimization=replace(self.opt,learn_splits=learned),seed=5,device="cuda")
            eager,_ = fit_nll_cv(self.data,self.data,training=self.training,**kwargs)
            graph,_ = fit_nll_cv(self.data,self.data,training=replace(self.training,cuda_graphs=True),**kwargs)
            for key,value in eager.state_dict().items():
                torch.testing.assert_close(value,graph.state_dict()[key],atol=2e-6,rtol=2e-5)

    def test_independent_nll_studies_resume_pairing_and_complete_audit(self):
        kwargs = dict(config=ExperimentConfig(scenarios=("ph","multimodal"),censor_types=("current",),
            censor_qs=(.5,),n_samples=60,seeds=(0,1),grid_size=9),
            search=CVSearchConfig(trials=3,initial_trials=2,depth_min=2,depth_max=3,
                widths=(4,8),shared_layers_max=1,private_layers_max=2,strength_max=100000.),
            baseline=Architecture(depth=2,shared_width=4,private_width=4,private_layers=1,regularization_strength=.5),
            training=TrainingConfig(epochs=3,batch_size=32,validation_every=1,patience=3,cuda_graphs=False),
            optimization=self.opt,device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            calls = []
            def interrupted(*args,**fit_kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("simulated interruption")
                return fit_nll_cv(*args,**fit_kwargs)
            with patch("pycode.unified_nll.nll_cv_search.fit_nll_cv",side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,"simulated interruption"):
                    tune_nll_cv(output,**kwargs)
            assert not (output/"selection_complete.json").exists()
            assert len(json.loads(next((output/"fold_cache").glob("*.json")).read_text())["folds"]) == 1
            with patch("pycode.unified_nll.nll_cv_search._evaluation_data",side_effect=AssertionError("test during selection")):
                selected,trials,folds = tune_nll_cv(output,**kwargs)
            self.assertEqual(len(selected),3*len(METHODS))
            self.assertEqual(len(folds),3*(2*3+1)*4)
            assert "multimodal_close_current_q0.5" in set(selected.setting)
            assert "average_grid_sse" not in selected.columns
            np.testing.assert_array_equal(trials.cv_nll,trials.objective_value)
            assert trials[trials.method == SEARCH_METHODS[0]].regularization_strength.eq(0).all()
            data_seeds = []
            def recorded(setting,actual,model_config,seed):
                data_seeds.append(seed)
                return _evaluation_data(setting,actual,model_config,seed)
            with patch("pycode.unified_nll.nll_cv_search._evaluation_data",side_effect=recorded):
                runs,summary = evaluate_nll_cv(output,**kwargs)
            assert sorted(data_seeds) == [0,0,0,1,1,1]
            assert runs.n_fit.eq(48).all() and runs.n_test.eq(12).all() and summary.n_runs.eq(2).all()
            audit = audit_nll_cv(output)
            assert audit["passed"] and audit["independent_nll_studies"] == 6
            overall,family,paired = report_nll_cv(output)
            pivot = runs.pivot(index=["setting","seed"],columns="method",values="average_grid_sse")
            expected = (pivot[SEARCH_METHODS[1]]-pivot[SEARCH_METHODS[0]]).groupby("setting").mean()
            np.testing.assert_allclose(paired.set_index("setting").adaptive_minus_tuned_fixed_mean.sort_index(),expected.sort_index())
            with patch("pycode.unified_nll.nll_cv_search.fit_nll_cv",side_effect=AssertionError("unnecessary refit")):
                tune_nll_cv(output,**kwargs)
                resumed,_ = evaluate_nll_cv(output,**kwargs)
            np.testing.assert_allclose(runs.average_grid_sse,resumed.average_grid_sse)
            # A changed frozen selection must fail before final evaluation.
            with (output/"selected_parameters.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError,"freeze all parameters"):
                evaluate_nll_cv(output,**kwargs)


if __name__ == "__main__":
    unittest.main()
