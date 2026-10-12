"""Raw support, censored likelihoods, checkpointing and current entry smoke."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .model import RawTimeKLTree, observed_upper_bound, predict_density, fit_raw_time_kl
from .search import tune_nll_cv, evaluate_nll_cv
from .report import audit_nll_cv, report_nll_cv
from ..cv_search import CVSearchConfig
from ..experiment import ExperimentConfig
from ..nll_cv_train import NLLCVOptimization
from ..train import TrainingConfig
from ..tuning_model import Architecture


class RawTimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.arch = Architecture(depth=3, private_width=8, private_layers=2,
                                 regularization_strength=20.)
        self.opt = NLLCVOptimization(warmup_epochs=1)
        self.training = TrainingConfig(epochs=3, batch_size=8, validation_every=1,
                                       patience=3, cuda_graphs=False)
        self.x = torch.zeros(7,6)

    def test_raw_likelihood_all_censoring_and_no_clipping(self):
        model = RawTimeKLTree(self.arch, 11., self.opt)
        left = torch.tensor([2.,2.,2.,0.,12.,11.,11.], dtype=torch.float64)
        right = torch.tensor([2.,torch.inf,5.,5.,12.,11.,torch.inf], dtype=torch.float64)
        obs = model.prepare(self.x,left,right)
        torch.testing.assert_close(obs.left,left,rtol=0,atol=0)
        self.assertTrue(torch.equal(obs.log_jacobian,torch.zeros_like(left)))
        likelihood = model.tree.loglik(model(obs.x),*obs[1:]).exp()
        expected = torch.tensor([1/11,9/11,3/11,5/11,0.,1/11,0.],dtype=torch.float64)
        torch.testing.assert_close(likelihood,expected,rtol=1e-6,atol=1e-8)
        times = torch.tensor([-1.,0.,2.,11.,12.,torch.inf])
        survival = model.predict_survival(self.x,times)
        torch.testing.assert_close(survival[0],torch.tensor([1.,1.,9/11,0.,0.,0.],dtype=torch.float64),rtol=1e-6,atol=1e-8)
        density = predict_density(model,self.x,times)
        torch.testing.assert_close(density[0],torch.tensor([0.,1/11,1/11,1/11,0.,0.],dtype=torch.float64),rtol=1e-6,atol=1e-8)

    def test_fit_subset_only_and_impossible_validation_rejected(self):
        train=(torch.randn(24,6),torch.linspace(.1,2.,24),torch.linspace(.1,2.,24))
        self.assertAlmostEqual(observed_upper_bound(train),2.2)
        model,_=fit_raw_time_kl(train,architecture=self.arch,training=self.training,
            optimization=self.opt,fixed_epochs=3,device="cpu")
        self.assertAlmostEqual(float(model.upperbound),2.2)
        state=model(train[0])
        torch.testing.assert_close(state.lower[:,0],torch.zeros(24,dtype=torch.float64))
        torch.testing.assert_close(state.upper[:,-1],torch.full((24,),2.2,dtype=torch.float64))
        invalid=(torch.zeros(1,6),torch.tensor([100.]),torch.tensor([100.]))
        with self.assertRaisesRegex(ValueError,"train-only raw support"):
            fit_raw_time_kl(train,invalid,architecture=self.arch,training=self.training,
                            optimization=self.opt,device="cpu")

    def test_kl_scale_and_density_integral(self):
        model=RawTimeKLTree(self.arch,17.,self.opt)
        model.split_gate.fill_(1.)
        with torch.no_grad(): model.split_head.bias.copy_(torch.linspace(-2.,2.,7))
        obs=model.prepare(self.x,torch.ones(7),torch.ones(7))
        state=model(self.x)
        expected=(-2/self.arch.depth)*torch.log(state.width/(17./8)).mean()
        torch.testing.assert_close(model.terms(obs)[1],expected)
        centers=(state.lower[0]+state.upper[0])/2
        integral=(predict_density(model,self.x[:1],centers)[0]*state.width[0]).sum()
        self.assertAlmostEqual(float(integral.detach()),1.,places=6)
        scaled=RawTimeKLTree(self.arch,170.,self.opt)
        scaled.load_state_dict({k:v for k,v in model.state_dict().items() if k!="upperbound"},strict=False)
        nll,reg=model.terms(obs)
        scaled_nll,scaled_reg=scaled.terms(scaled.prepare(self.x,torch.full((7,),10.),torch.full((7,),10.)))
        torch.testing.assert_close(scaled_reg,reg,atol=1e-12,rtol=1e-12)
        self.assertAlmostEqual(float((scaled_nll-nll).detach()),np.log(10),places=6)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_graph_eager_parity(self):
        train=(torch.randn(24,6),torch.linspace(.1,2.,24),torch.linspace(.1,2.,24))
        for learned in (False,True):
            kwargs=dict(architecture=self.arch,optimization=replace(self.opt,learn_splits=learned),
                        fixed_epochs=3,seed=17,device="cuda")
            eager,_=fit_raw_time_kl(train,training=self.training,**kwargs)
            graph,_=fit_raw_time_kl(train,training=replace(self.training,cuda_graphs=True),**kwargs)
            for key,value in eager.state_dict().items():
                torch.testing.assert_close(value,graph.state_dict()[key],rtol=2e-5,atol=2e-6)

    def test_current_search_audit_and_resume(self):
        kwargs=dict(config=ExperimentConfig(scenarios=("ph",),censor_types=("current",),
            censor_qs=(.5,),n_samples=200,seeds=(0,1),grid_size=9),
            search=CVSearchConfig(trials=2,initial_trials=2,folds=2,depth_min=2,depth_max=3,
                widths=(4,8),shared_layers_max=1,private_layers_max=2,strength_max=100.),
            baseline=Architecture(depth=2,shared_width=4,private_width=4,private_layers=1,
                                   regularization_strength=.5),
            training=replace(self.training,batch_size=64),optimization=self.opt,device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            selected,trials,folds=tune_nll_cv(directory,**kwargs)
            self.assertEqual(len(selected),3)
            self.assertEqual(len(folds),10)
            runs,summary=evaluate_nll_cv(directory,**kwargs)
            self.assertEqual(len(runs),6)
            self.assertTrue(audit_nll_cv(directory)["train_only_upper_bounds_verified"])
            report_nll_cv(directory)
            text=(Path(directory)/"grid_sse_report.md").read_text(encoding="utf-8")
            self.assertIn("共 1 个 setting",text)
            self.assertIn("2 组 train–test",text)
            with patch("pycode.unified_nll.raw_time.search.fit_nll_cv",side_effect=AssertionError("unexpected fit")):
                tune_nll_cv(directory,**kwargs)
                evaluate_nll_cv(directory,**kwargs)
            manifest=json.loads((Path(directory)/"config.json").read_text())
            self.assertEqual(manifest["time_coordinate"],"raw")
            self.assertTrue(runs.upper_bound.gt(0).all())
            with self.assertRaisesRegex(ValueError,"source/config changed"):
                tune_nll_cv(directory,**dict(kwargs,search=replace(kwargs["search"],trials=3)))


if __name__ == "__main__":
    unittest.main()
