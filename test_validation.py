"""Check CV validation reuse, data isolation, and validation checkpoint recovery."""

from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import workflow as w
from tuning import main as tune


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_sizes_reproducibility_and_rng_isolation(self):
        for stage in ('tuning', 'testing'):
            split = importlib.import_module(f'{stage}.validation').training_validation_split
            for count in (2, 9, 25, 4000):
                with self.subTest(stage=stage, count=count):
                    state = torch.get_rng_state().clone()
                    train, valid = split(count, seed=321)
                    torch.testing.assert_close(torch.get_rng_state(), state)
                    self.assertEqual(len(train), int(0.8 * count))
                    self.assertEqual(len(valid), count - len(train))
                    self.assertFalse(set(train.tolist()) & set(valid.tolist()))
                    self.assertEqual(set(train.tolist()) | set(valid.tolist()), set(range(count)))
                    torch.testing.assert_close(train, split(count, seed=321)[0])
            with self.assertRaisesRegex(ValueError, 'at least two'):
                split(1)

    def test_nrt_stops_on_validation_and_restores_best_epoch(self):
        for stage in ('tuning', 'testing'):
            module = importlib.import_module(f'{stage}.train_mlp')
            for method in ('joint_nll', 'joint_brier', 'alternating'):
                for explicit in (False, True):
                    with self.subTest(stage=stage, method=method, explicit=explicit):
                        torch.manual_seed(321)
                        x = torch.arange(40, dtype=torch.float32).reshape(20, 2) / 40
                        left = torch.linspace(0.1, 1.5, 20)
                        right = left.clone()
                        if explicit:
                            validation = (x[15:], left[15:], right[15:])
                            inputs = (x[:15], left[:15], right[:15])
                            expected_training = x[:15, 0].tolist()
                        else:
                            train, valid = module.training_validation_split(20, seed=321)
                            validation = (x[valid], left[valid], right[valid])
                            inputs = (x, left, right)
                            expected_training = x[train, 0].tolist()
                        config = module.TrainingConfig(
                            epochs=10, patience=2, training_batch_size=5,
                            brier_grid_size=8, training_method=method, cuda_graphs=False,
                        )
                        seen, states = [], []
                        scores = iter((3.0, 1.0, 2.0, 4.0))
                        objective = module.ConditionalTree.objective

                        def record_objective(model, xb, lb, rb, **kwargs):
                            seen.extend(xb[:, 0].tolist())
                            return objective(model, xb, lb, rb, **kwargs)

                        def validation_score(model, xb, lb, rb, *rest):
                            for actual, expected in zip((xb, lb, rb), validation):
                                torch.testing.assert_close(actual, expected)
                            self.assertFalse(model.training)
                            states.append({key: value.clone() for key, value in model.state_dict().items()})
                            return next(scores)

                        evaluator = 'evaluate_brier' if method == 'joint_brier' else 'evaluate_nll'
                        # The real evaluator puts the complete model in eval mode.
                        def evaluate(model, *args):
                            model.eval()
                            return validation_score(model, *args)

                        with patch.object(module.ConditionalTree, 'objective', new=record_objective), \
                             patch.object(module, evaluator, side_effect=evaluate):
                            model, history = module.fit(
                                *inputs, 2.0, module.ModelConfig(depth=2, x_dim=2, h_dim=4, mlp_hidden=4),
                                config, 'cpu', verbose=False,
                                validation_data=validation if explicit else None,
                            )
                        self.assertEqual(len(history), 4)
                        self.assertEqual([row[config.monitor] for row in history], [3.0, 1.0, 2.0, 4.0])
                        self.assertEqual(set(seen), set(expected_training))
                        self.assertFalse(set(seen) & set(validation[0][:, 0].tolist()))
                        for key, value in model.state_dict().items():
                            torch.testing.assert_close(value, states[1][key])

    def test_cv_passes_the_existing_fold_without_nested_split(self):
        data = w.generate(w.Case('ph', 'exact', 0.2), 5000, 123, None)
        with tempfile.TemporaryDirectory() as directory:
            args = tune.parser().parse_args(['--method', 'NRT', '--device', 'cpu', '--epochs', '1'])
            with patch.object(tune, 'fit_model', return_value=object()) as fit, \
                 patch('tuning.train_mlp.evaluate_nll', return_value=1.0), \
                 redirect_stdout(io.StringIO()):
                tune.tune_cv(args, 'NRT_2Loss', data, [{'depth': 2}], Path(directory))
            self.assertEqual(fit.call_count, 4)
            for call, (train, valid) in zip(fit.call_args_list, w.cv_splits(data.train, args.tune_seed)):
                torch.testing.assert_close(call.args[2], train)
                torch.testing.assert_close(call.kwargs['validation_indices'], valid)
                self.assertEqual((len(train), len(valid)), (3000, 1000))
                self.assertFalse(set(data.test.tolist()) & (set(train.tolist()) | set(valid.tolist())))

    def test_alternating_brier_selected_results_require_retuning(self):
        with self.assertRaisesRegex(ValueError, "expected 'nll'"):
            tune.check_tuning_result({'cv_metric': 'brier'}, 'NRT_2Loss', 'old_result.json')

    def test_nncde_uses_cv_validation_for_each_cross_fit_network(self):
        for stage in ('tuning', 'testing'):
            module = importlib.import_module(f'{stage}.benchmark.nncde.model')
            for explicit in (True, False):
                with self.subTest(stage=stage, explicit=explicit):
                    model = module.NNCDE(hidden_nodes=4, batch_size=20, epochs=1, device='cpu')
                    x = torch.arange(24, dtype=torch.float32).reshape(24, 1)
                    times = torch.linspace(0.1, 2.4, 24)
                    validation_x = torch.arange(24, 30, dtype=torch.float32).reshape(6, 1)
                    validation_times = torch.linspace(2.5, 3.0, 6)
                    seen_training, seen_validation = [], []
                    loss_function = model._loss

                    def loss(fitted, xb, *args):
                        if torch.is_grad_enabled():
                            seen_training.append(xb[:, 0].tolist())
                        else:
                            seen_validation.append(xb[:, 0].tolist())
                        return loss_function(fitted, xb, *args)

                    with patch.object(model, '_loss', side_effect=loss):
                        model.fit(x, times, times, censor_type='exact', validation_data=(
                            validation_x, validation_times, validation_times,
                        ) if explicit else None)
                    self.assertEqual(len(seen_validation), 2)
                    for train, valid in zip(seen_training, seen_validation):
                        self.assertEqual(len(train), 12 if explicit else 9)
                        self.assertEqual(len(valid), 6 if explicit else 3)
                        self.assertFalse(set(train) & set(valid))
                        if explicit:
                            self.assertEqual(valid, validation_x[:, 0].tolist())

    def test_deepsurv_uses_fold_validation_and_recovers_checkpoint(self):
        for stage in ('tuning', 'testing'):
            module = importlib.import_module(f'{stage}.benchmark.deepsurv.cuda_model')
            with self.subTest(stage=stage):
                x = torch.randn(25, 2)
                times = torch.arange(1, 26, dtype=torch.float32)
                model = module.DeepSurvCUDA(hidden_layers=(4,), dropout=0, epochs=10,
                                           patience=2, device='cpu', standardize=True)
                states, losses = [], iter((3.0, 1.0, 2.0, 4.0))
                loss_function = module.cox_partial_log_likelihood

                def score(risk, event, **kwargs):
                    if torch.is_grad_enabled():
                        self.assertEqual(len(risk), 20)
                        return loss_function(risk, event, **kwargs)
                    self.assertEqual(len(risk), 5)
                    states.append({name: value.clone() for name, value in network[0].state_dict().items()})
                    return risk.new_tensor(next(losses))

                network = []
                make_model = model._make_model
                def capture(*args):
                    result = make_model(*args)
                    network.append(result)
                    return result

                with patch.object(model, '_make_model', side_effect=capture), \
                     patch.object(module, 'cox_partial_log_likelihood', side_effect=score):
                    model.fit(x[:20], times[:20], times[:20], censor_type='exact',
                              validation_data=(x[20:], times[20:], times[20:]))
                self.assertEqual(len(model.training_history_), 4)
                torch.testing.assert_close(model._input_mean, x[:20].mean(dim=0))
                for name, value in model._model.state_dict().items():
                    torch.testing.assert_close(value, states[1][name])


if __name__ == '__main__':
    unittest.main()
