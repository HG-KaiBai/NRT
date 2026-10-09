"""Check paper Gaussian settings through CV, refitting, and simulation."""
from contextlib import redirect_stdout
import importlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import workflow as w
from testing import simulation as sim
from tuning import main as tune
from tuning.benchmark.deepsurv.cuda_model import (
    OFFICIAL_TUNING_PARAMETERS, PAPER_GAUSSIAN_FIXED_PARAMETERS, PAPER_GAUSSIAN_PARAMETERS,
    TUNABLE_PARAMETER_NAMES,
)


class DeepSurvPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_paper_defaults(self, model):
        for name, expected in PAPER_GAUSSIAN_PARAMETERS.items():
            self.assertEqual(getattr(model, name), expected, name)

    def assert_search_ranges(self, parameters, ranges):
        widths = parameters['hidden_layers']
        self.assertTrue(all(ranges['nodes_per_layer'][0] <= width <= ranges['nodes_per_layer'][1]
                            for width in widths))
        self.assertTrue(ranges['hidden_layers'][0] <= len(widths) <= ranges['hidden_layers'][1])
        lower, upper = ranges['log10_learning_rate']
        self.assertTrue(lower <= math.log10(parameters['learning_rate']) <= upper)
        for name in ('lr_decay', 'momentum', 'l2_reg', 'dropout'):
            lower, upper = ranges[name]
            self.assertTrue(lower <= parameters[name] <= upper, name)

    def test_model_and_cli_defaults_match_paper_in_both_stages(self):
        for stage in ('tuning', 'testing'):
            with self.subTest(stage=stage):
                module = importlib.import_module(f'{stage}.benchmark.deepsurv.cuda_model')
                model = module.DeepSurvCUDA(device='cpu')
                self.assert_paper_defaults(model)
                self.assertEqual(model.tuning_epochs, PAPER_GAUSSIAN_PARAMETERS['epochs'])
                self.assertEqual(model.tuning_parameters['fixed_candidate_parameters'],
                                 PAPER_GAUSSIAN_FIXED_PARAMETERS)
                runner = importlib.import_module(f'{stage}.benchmark.deepsurv.run')
                with patch('sys.argv', ['deepsurv', '--device', 'cpu']), \
                     patch.object(runner, 'benchmark_model',
                                  return_value=SimpleNamespace(average_grid_sse=0.0)) as benchmark, \
                     redirect_stdout(io.StringIO()):
                    runner.main()
                self.assert_paper_defaults(benchmark.call_args.args[0])

    def test_cv_and_final_fit_keep_fixed_settings_in_both_stages(self):
        data = w.generate(w.Case('ph', 'exact', 0.2), 40, 17, None)
        for stage in ('tuning', 'testing'):
            with self.subTest(stage=stage):
                module = importlib.import_module(f'{stage}.benchmark.deepsurv.cuda_model')
                # Shorten only the epoch limits for this real CPU training check.
                with module.DeepSurvCUDA(device='cpu', tune=True, cv_folds=4,
                                         tuning_evals=2, epochs=1, tuning_epochs=1) as model:
                    with patch.object(model, '_train_network', wraps=model._train_network) as train:
                        model.fit(*data.bounds(data.train), censor_type='exact')
                    self.assertEqual(train.call_count, 2 * 4 + 1)
                    for call in train.call_args_list:
                        configuration = call.args[3]
                        for name in ('activation', 'batch_norm', 'standardize'):
                            self.assertEqual(getattr(configuration, name),
                                             PAPER_GAUSSIAN_PARAMETERS[name])
                        self.assertEqual(call.kwargs['epochs'], 1)
                    self.assertEqual(model.optimizer, 'adam')
                    self.assertEqual(model.l1_reg, 0.0)
                    self.assertEqual(model.tuning_parameters['selection_metric'],
                                     'validation Harrell C-index')
                    torch.testing.assert_close(model._input_mean, torch.zeros(data.x.shape[1]))
                    torch.testing.assert_close(model._input_std, torch.ones(data.x.shape[1]))
                    self.assertTrue(torch.isfinite(model.predict_risk(data.x)).all())
                    ranges = OFFICIAL_TUNING_PARAMETERS['reference_search']['ranges']
                    self.assertEqual(model.tuning_parameters['ranges'], ranges)
                    self.assert_search_ranges(model.best_hyperparameters_, ranges)
                    for candidate in model.tuning_parameters['cross_validation_fits']:
                        self.assert_search_ranges(candidate, ranges)

    def test_optimizer_reads_edited_ranges_in_both_stages(self):
        data = w.generate(w.Case('ph', 'exact', 0.2), 40, 17, None)
        ranges = dict(log10_learning_rate=[-4.5, -4.0], nodes_per_layer=[35, 40],
                      hidden_layers=[2, 3], lr_decay=[0.0001, 0.0002],
                      momentum=[0.81, 0.85], l2_reg=[0.1, 0.2], dropout=[0.01, 0.02])
        for stage in ('tuning', 'testing'):
            module = importlib.import_module(f'{stage}.benchmark.deepsurv.cuda_model')
            with self.subTest(stage=stage), \
                 patch.dict(module.OFFICIAL_TUNING_PARAMETERS['reference_search'], ranges=ranges), \
                 module.DeepSurvCUDA(device='cpu', tune=True, cv_folds=2,
                                     tuning_evals=2, epochs=1, tuning_epochs=1) as model:
                model.fit(*data.bounds(data.train), censor_type='exact')
                self.assert_search_ranges(model.best_hyperparameters_, ranges)
                for candidate in model.tuning_parameters['cross_validation_fits']:
                    self.assert_search_ranges(candidate, ranges)

    def test_tuning_and_simulation_use_same_fixed_settings_and_selected_values(self):
        args = tune.parser().parse_args(['--method', 'DeepSurv'])
        self.assertEqual(tune.training_config(args, 'DeepSurv'), PAPER_GAUSSIAN_FIXED_PARAMETERS)
        self.assertEqual(w.DEFAULT_EPOCHS['DeepSurv'], PAPER_GAUSSIAN_PARAMETERS['epochs'])
        case = w.Case('ph', 'exact', 0.2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = tune.parser().parse_args([
                '--method', 'DeepSurv', '--output-dir', str(root / 'tuning'),
                '--n-samples', '40', '--epochs', '1', '--tuning-evals', '2',
                '--grid-size', '8', '--device', 'cpu',
            ])
            with redirect_stdout(io.StringIO()):
                tune.tune_case(args, 'DeepSurv', case, None)
            result = json.loads((args.output_dir / 'DeepSurv' / case.id / 'result.json').read_text())
            self.assertEqual(result['request']['search_ranges'],
                             OFFICIAL_TUNING_PARAMETERS['reference_search']['ranges'])
            self.assertEqual(result['cv_metric'], 'c_index')
            self.assertEqual(result['training_config'], {**PAPER_GAUSSIAN_FIXED_PARAMETERS, 'epochs': 1})
            with redirect_stdout(io.StringIO()):
                tune.write_summary(args, [('DeepSurv', case)])
            testing_args = sim.parser().parse_args([
                'run', '--hyper-dir', str(args.output_dir / 'hyper_para'), '--output-dir', str(root / 'testing'),
                '--n-samples', '40', '--grid-size', '8', '--device', 'cpu',
            ])
            spec = sim.load_spec(testing_args, 'DeepSurv', case)
            expected_training = {name: value for name, value in PAPER_GAUSSIAN_FIXED_PARAMETERS.items()
                                 if name not in result['parameters']}
            self.assertEqual(spec[2], expected_training)
            for name in TUNABLE_PARAMETER_NAMES:
                self.assertEqual(spec[1][name], result['parameters'][name])
            testing_args.epochs = 1
            spec = sim.load_spec(testing_args, 'DeepSurv', case)
            job = sim.make_job(testing_args, 'DeepSurv', case, 1, spec)
            with redirect_stdout(io.StringIO()):
                simulated = sim.fit_job(job)
            self.assertEqual(simulated['status'], 'ok')
            self.assertEqual(simulated['training_config'], {**expected_training, 'epochs': 1})
            self.assertEqual(simulated['parameters'], result['parameters'])
            with patch.object(tune, 'fit_model', side_effect=AssertionError('Unexpected refitting')), \
                 redirect_stdout(io.StringIO()) as output:
                tune.tune_case(args, 'DeepSurv', case, None)
            self.assertIn('Reusing', output.getvalue())
            # Legacy output must not be reused or published as a corrected search.
            result['request'].pop('search_ranges')
            w.atomic_json(args.output_dir / 'DeepSurv' / case.id / 'result.json', result)
            with self.assertRaisesRegex(ValueError, 'DeepSurv search ranges'):
                tune.tune_case(args, 'DeepSurv', case, None)
            with self.assertRaisesRegex(ValueError, 'DeepSurv search ranges'):
                tune.write_summary(args, [('DeepSurv', case)])
            result['request']['search_ranges'] = {
                **OFFICIAL_TUNING_PARAMETERS['reference_search']['ranges'],
                'nodes_per_layer': [2, 20],
            }
            w.atomic_json(args.output_dir / 'DeepSurv' / case.id / 'result.json', result)
            with self.assertRaisesRegex(ValueError, 'DeepSurv search ranges'):
                tune.tune_case(args, 'DeepSurv', case, None)

    def test_csv_parameters_are_used_with_paper_defaults_for_unspecified_settings(self):
        case = w.Case('ph', 'exact', 0.2)
        searched = {name: PAPER_GAUSSIAN_PARAMETERS[name] for name in TUNABLE_PARAMETER_NAMES}
        searched.update(hidden_layers=[5, 5], learning_rate=0.002, dropout=0.1, l2_reg=0.02)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = {**searched, 'activation': 'rectify', 'batch_norm': True, 'standardize': True}
            w.write_csv(root / 'DeepSurv.csv', [dict(
                model_name='DeepSurv', data_types=case.id,
                parameters=json.dumps(saved), tau=None)])
            args = sim.parser().parse_args(['run', '--hyper-dir', str(root)])
            _, parameters, training, _ = sim.load_spec(args, 'DeepSurv', case)
            self.assertEqual(parameters, saved)
            defaults = {name: value for name, value in PAPER_GAUSSIAN_FIXED_PARAMETERS.items()
                        if name not in saved}
            self.assertEqual(training, defaults)
            args.epochs = 2
            self.assertEqual(sim.load_spec(args, 'DeepSurv', case)[2],
                             {**defaults, 'epochs': 2})


if __name__ == '__main__':
    unittest.main()
