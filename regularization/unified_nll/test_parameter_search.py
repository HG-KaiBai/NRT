"""Brier parity, NLL-only optimization, shared architecture, and resumable HPO."""
from dataclasses import replace
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .brier import score_brier
from .experiment import ExperimentConfig
from .parameter_search import SearchConfig, tune_per_setting, evaluate_selected
from .train import TrainingConfig
from .tuning_model import Architecture,TuningTree
from .tuning_train import fit_tuning


class ParameterSearchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_brier_matches_original_implementation_for_mixed_observations(self):
        from pycode.neural_only.mlp import ConditionalTree as OriginalTree
        torch.manual_seed(7)
        model = TuningTree(Architecture(depth=3,private_width=8,private_layers=2),1.3)
        with torch.no_grad():
            model.mass_head.weight.normal_(0,.2)
            model.split_head.weight.normal_(0,.2)
        x = torch.randn(8,6)
        left = torch.tensor([.3,.5,.2,0.,1.,0.,1.1,.6])
        right = torch.tensor([.3,torch.inf,.9,.7,torch.inf,2.,1.100001,.6001])
        grid = torch.linspace(0,3,256)
        state = model(x)

        class DistributionAdapter:
            max_bound = 1e20
            def leaf_masses(self,ignored):
                return state.log_mass.exp()
            def leaf_intervals(self,ignored):
                return state.lower,state.upper
            def _cdf_from_intervals(self,times,lower,upper,masses):
                u = times/(times+model.time_scale)
                fraction = ((u[:,:,None]-lower[:,None,:])/(upper-lower)[:,None,:]).clamp(0,1)
                return (fraction*masses[:,None,:]).sum(-1)

        class OriginalFormula:
            tree = DistributionAdapter()
            max_observation = 3.
            def __call__(self,ignored):
                return SimpleNamespace(left_probs=torch.zeros(8,7),split_props=torch.zeros(8,7))

        original = OriginalTree._brier_loss(OriginalFormula(),x,left,right,256,False,False)
        actual = score_brier(model,(x,left,right),grid,batch_size=3)
        self.assertAlmostEqual(actual,float(original.detach()),places=6)

    def test_brier_is_only_computed_after_training(self):
        torch.manual_seed(4)
        x = torch.randn(19,6)
        left = torch.linspace(.1,2.,19)
        right = left.clone()
        right[::3] = torch.inf
        train,valid = (x,left,right),(x[:5],left[:5],right[:5])
        architecture = Architecture(depth=2,shared_width=5,shared_layers=2,
            private_width=4,private_layers=2,regularization_strength=.5)
        training = TrainingConfig(epochs=3,batch_size=8,validation_every=1,patience=3,cuda_graphs=False)
        kwargs = dict(architecture=architecture,training=training,seed=9,device="cpu")
        with patch("pycode.unified_nll.tuning_train.score_brier",return_value=0.) as first:
            a,ah = fit_tuning(train,valid,torch.linspace(0,2,9),**kwargs)
        with patch("pycode.unified_nll.tuning_train.score_brier",return_value=100.) as second:
            b,bh = fit_tuning(train,valid,torch.linspace(0,2,9),**kwargs)
        self.assertEqual(first.call_count,1)
        self.assertEqual(second.call_count,1)
        self.assertEqual(ah,bh)
        for name,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
        gradients = torch.autograd.grad(sum(a.terms(a.prepare(*train))),tuple(a.parameters()),allow_unused=True)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        if torch.cuda.is_available():
            eager,_ = fit_tuning(train,valid,torch.linspace(0,2,9),**dict(kwargs,device="cuda"))
            graph,_ = fit_tuning(train,valid,torch.linspace(0,2,9),**dict(kwargs,
                device="cuda",training=replace(training,cuda_graphs=True)))
            for name,value in eager.state_dict().items():
                torch.testing.assert_close(value,graph.state_dict()[name],atol=2e-6,rtol=2e-5)

    def test_search_confirmation_test_split_and_resume(self):
        config = ExperimentConfig(scenarios=("ph",),censor_types=("current",),censor_qs=(.5,),
            seeds=(0,1),n_samples=60,grid_size=9)
        baseline = Architecture(depth=2,shared_width=4,private_width=4,private_layers=1,
            regularization_strength=.5)
        search = SearchConfig(trials=3,initial_trials=2,shortlist=2,depth_min=2,depth_max=3,
            widths=(4,8),shared_layers_max=1,private_layers_max=2,confirmation_seeds=(102,103))
        training = TrainingConfig(epochs=2,batch_size=32,validation_every=1,patience=2,cuda_graphs=False)
        kwargs = dict(config=config,search=search,training=training,baseline=baseline,device="cpu",verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            best,trials,confirmed = tune_per_setting(directory,**kwargs)
            self.assertEqual(len(trials),3)
            self.assertEqual(len(confirmed),4)
            scores = confirmed.groupby("trial").validation_brier.mean()
            self.assertAlmostEqual(best.iloc[0].selection_brier,scores.min())
            self.assertNotIn("average_grid_sse",trials.columns)
            self.assertNotIn("test_brier",trials.columns)
            repeated,_,_ = tune_per_setting(directory,**kwargs)
            self.assertEqual(int(best.iloc[0].trial),int(repeated.iloc[0].trial))
            runs,summary = evaluate_selected(directory,**kwargs)
            self.assertEqual(len(runs),4)
            self.assertTrue(summary.n_runs.eq(2).all())
            resumed,_ = evaluate_selected(directory,**kwargs)
            np.testing.assert_allclose(runs.average_grid_sse,resumed.average_grid_sse)
            with self.assertRaisesRegex(ValueError,"different code/configuration"):
                tune_per_setting(directory,**dict(kwargs,training=replace(training,epochs=3)))


if __name__ == "__main__":
    unittest.main()
