"""Checks for the mass-independent penalty and leakage-free depth protocol."""

from dataclasses import replace
import tempfile
import unittest

import numpy as np
import torch

from .depth_experiment import (DepthConfig, _evaluation_data, _generate,
                               run_depth_comparison, tune_global_strength)
from .experiment import ExperimentConfig, Setting
from .model import ModelConfig
from .metrics import average_grid_sse
from .regularization import depth_model_config, fit_regularized, partition_penalty
from .train import TrainingConfig, fit
from .tree import Tree


class DepthRegularizationTests(unittest.TestCase):
    def test_grid_sse_is_a_sum_then_a_subject_mean(self):
        truth=torch.tensor([[1.,.5,0.],[1.,.6,.2]])
        estimated=torch.tensor([[1.,.3,.1],[.9,.5,.3]])
        expected=torch.tensor((.2**2+.1**2+.1**2+.1**2+.1**2)/2)
        torch.testing.assert_close(average_grid_sse(estimated,truth),expected)
        torch.testing.assert_close(average_grid_sse(1-estimated,1-truth),expected)
        with self.assertRaises(ValueError):
            average_grid_sse(estimated,truth[:1])

    def test_grid_matches_original_first_eighty_percent_bounds(self):
        config=ExperimentConfig(n_samples=60,grid_size=256)
        setting=Setting("ph","right",.5)
        model=ModelConfig(x_dim=2)
        raw=_generate(setting,config,model,123)
        pool=torch.cat((raw[1][:48],raw[2][:48]))
        expected=float(pool[torch.isfinite(pool)].max())
        actual=_evaluation_data(setting,config,model,123)
        self.assertEqual(len(actual["test"][0]),12)
        self.assertEqual(actual["grid"].numel(),256)
        self.assertEqual(float(actual["grid"][-1]),expected)

    def test_barrier_is_mass_independent_and_depth_normalized(self):
        expected = -np.log(4 * .3 * .7)
        for depth in (3, 6, 8):
            tree = Tree(depth, min_leaf_width=0).double()
            mass = torch.randn(2, 2**depth-1, dtype=torch.float64, requires_grad=True)
            rho = torch.full_like(mass, .3, requires_grad=True)
            state = tree(mass, rho)
            penalty = partition_penalty(state, depth)
            torch.testing.assert_close(penalty, torch.full_like(penalty, expected))
            mass_gradient, rho_gradient = torch.autograd.grad(penalty.sum(), (mass,rho),
                                                              allow_unused=True)
            self.assertIsNone(mass_gradient)
            self.assertTrue(torch.isfinite(rho_gradient).all())
            alternate = tree(-mass.detach()*20, rho.detach())
            torch.testing.assert_close(penalty, partition_penalty(alternate, depth))
            balanced = tree(mass.detach(), torch.full_like(rho, .5))
            torch.testing.assert_close(partition_penalty(balanced,depth),
                                       torch.zeros(2,dtype=torch.float64),atol=1e-14,rtol=0)

    def test_analytical_barrier_derivative_and_floor_scaling(self):
        tree=Tree(1,0).double()
        rho=torch.tensor([[.2]],dtype=torch.float64,requires_grad=True)
        value=partition_penalty(tree(torch.zeros_like(rho),rho),1).sum()
        gradient,=torch.autograd.grad(value,rho)
        torch.testing.assert_close(gradient,torch.full_like(gradient,-1/.2+1/.8))
        for depth in (3,6,8):
            config=depth_model_config(ModelConfig(),depth)
            self.assertAlmostEqual(2**depth*config.min_leaf_width,8e-4)

    def test_extended_training_matches_original_for_same_penalty(self):
        torch.manual_seed(5)
        x=torch.randn(17,2)
        left=torch.linspace(.1,2,17)
        right=left.clone()
        right[::3]=torch.inf
        right[1::3]+=.3
        config=ModelConfig(depth=2,x_dim=2,h_dim=4,mlp_hidden=5)
        training=TrainingConfig(epochs=3,batch_size=8,validation_every=1,patience=3,
                                cuda_graphs=False)
        kwargs=dict(validation=(x[:5],left[:5],right[:5]),model_config=config,
                    training_config=training,device="cpu",seed=11)
        old,_=fit(x,left,right,**kwargs)
        new,_=fit_regularized(x,left,right,penalty="density_kl",**kwargs)
        for name,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[name],atol=1e-7,rtol=1e-6)
        if torch.cuda.is_available():
            eager,eh=fit_regularized(x,left,right,penalty="partition_kl",
                **dict(kwargs,device="cuda"))
            graph,gh=fit_regularized(x,left,right,penalty="partition_kl",
                **dict(kwargs,device="cuda",training_config=replace(training,cuda_graphs=True)))
            for name,value in eager.state_dict().items():
                torch.testing.assert_close(value,graph.state_dict()[name],atol=2e-6,rtol=2e-5)
            for a,b in zip(eh,gh):
                self.assertAlmostEqual(a["validation_nll"],b["validation_nll"],places=5)

    def test_pilot_depth_selection_and_resume(self):
        config=ExperimentConfig(scenarios=("ph",),censor_types=("current",),censor_qs=(.5,),
            seeds=(0,1),n_samples=60,grid_size=9)
        depth=DepthConfig(depths=(2,3),strengths=(0.,.5))
        model=ModelConfig(x_dim=2,h_dim=4,mlp_hidden=5,mlp_layers=2)
        training=TrainingConfig(epochs=2,batch_size=32,validation_every=1,patience=2,
                                cuda_graphs=False)
        kwargs=dict(config=config,depth_config=depth,model_config=model,
                    training_config=training,device="cpu",verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            selected,scores=tune_global_strength(directory,**kwargs)
            self.assertEqual(selected,float(scores.iloc[0].strength))
            runs=run_depth_comparison(directory,**kwargs)
            resumed=run_depth_comparison(directory,**kwargs)
            self.assertEqual(len(runs),12)
            self.assertEqual(len(resumed),12)
            self.assertTrue(runs.groupby(["depth","method"]).size().eq(2).all())
            actual=runs.groupby(["setting","seed","method"]).validation_nll.min()
            import pandas as pd
            chosen=pd.read_csv(f"{directory}/validation_selected_depth_runs.csv")
            recorded=chosen.set_index(["setting","seed","method"]).validation_nll
            np.testing.assert_allclose(actual.sort_index(),recorded.sort_index(),atol=1e-12)
            with self.assertRaisesRegex(ValueError,"different code/configuration"):
                run_depth_comparison(directory,**dict(kwargs,
                    depth_config=replace(depth,total_width_floor=.001)))
            with self.assertRaisesRegex(ValueError,"must differ"):
                tune_global_strength(directory,**dict(kwargs,
                    depth_config=replace(depth,pilot_data_seed=config.data_seed)))


if __name__=="__main__":
    unittest.main()
