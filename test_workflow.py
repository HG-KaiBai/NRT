"""Regression checks for data sharing, fixed tau, CV isolation, and resumption."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import workflow as w
from testing import simulation as sim
from tuning import main as tune
from tuning.main import grid_candidates


class DataWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_case_support_matrix(self):
        self.assertEqual(len(w.ALL_CASES), 35)
        self.assertEqual(len({c.id for c in w.ALL_CASES}), 35)
        exact_cases = [c for c in w.ALL_CASES if c.censor_type == 'exact']
        self.assertEqual(len(exact_cases), 5)
        self.assertEqual({c.setting for c in exact_cases}, set(w.SETTINGS))
        counts = {m: sum(c.censor_type in w.SUPPORTED[m] for c in w.ALL_CASES) for m in w.METHODS}
        self.assertEqual(counts['NRT_NLL'], 35)
        self.assertEqual(counts['NNCDE'], 15)
        self.assertEqual(counts['DeepSurv'], 15)
        self.assertEqual(counts['deep_distribution_regression'], 5)
        self.assertEqual(counts['iccforest'], 30)
        self.assertEqual(sum(counts[m] for m in w.CPU_METHODS), 140)

    def test_all_cases_valid_and_identical_across_methods(self):
        for case in w.ALL_CASES:
            with self.subTest(case=case.id):
                tau = w.horizon(case)
                data = w.generate(case, 64, 123, tau)
                repeated = w.generate(case, 64, 123, tau)
                for key in ('x', 'L', 'R', 'T', 'train', 'test'):
                    torch.testing.assert_close(getattr(data, key), getattr(repeated, key))
                self.assertTrue(bool(((data.L <= data.T) & (data.T <= data.R)).all()))
                self.assertEqual(len(data.train), 51)
                self.assertEqual(len(data.test), 13)
                grid = torch.linspace(0, 20, 32)
                truth = data.truth(data.test, grid)
                self.assertTrue(bool(torch.isfinite(truth).all()))
                self.assertTrue(bool(((truth >= 0) & (truth <= 1)).all()))
                self.assertTrue(bool((truth.diff(dim=1) <= 0).all()))
                if case.censor_type == 'exact':
                    self.assertIsNone(tau)
                    self.assertEqual(data.metadata()['right_censoring_rate'], 0)

    def test_saved_tau_is_used_without_recalibration(self):
        import tuning.testdata2 as generator
        import testing.testdata2 as testing_generator
        for module in (generator, testing_generator):
            with patch.object(module, 'censoring_horizon', side_effect=AssertionError('Unexpected recalibration')):
                for scenario in ('ph', 'lognormal', 'po', 'multimodal'):
                    kwargs = dict(n=32, q=0.5, tau=0.001, censor_type='right')
                    if scenario == 'multimodal':
                        _, left, right, *_ = module.data_reg_multimodal(**kwargs)
                    else:
                        _, left, right, *_ = module.data_reg(scenario=scenario, **kwargs)
                    self.assertTrue(bool((left <= 0.001).all()))
                    self.assertGreater(float(torch.isinf(right).float().mean()), 0.9)

    def test_population_censoring_rates(self):
        for setting in w.SETTINGS:
            for q in (0.2, 0.5):
                case = w.Case(setting, 'right', q)
                with self.subTest(case=case.id):
                    data = w.generate(case, 20_000, 9876, w.horizon(case))
                    self.assertLess(abs(data.metadata()['right_censoring_rate'] - q), 0.015)

    def test_censoring_resamples_zero_draws_in_tuning_and_testing(self):
        import tuning.testdata2 as tuning_generator
        import testing.testdata2 as testing_generator
        for module in (tuning_generator, testing_generator):
            for censor_type in ('right', 'current', 'interval'):
                with self.subTest(module=module.__name__, censor_type=censor_type):
                    events = torch.full((3,), 2.0)
                    draws = [torch.tensor([0.0, 0.25, 0.0]),
                             torch.tensor([0.0, 0.5]), torch.tensor([0.75])]
                    with patch.object(module.torch, 'rand_like', side_effect=draws) as draw:
                        left, right = module._observe(events, censor_type, 1.0)
                    torch.testing.assert_close(left, torch.tensor([0.75, 0.25, 0.5]))
                    self.assertTrue(bool(torch.isinf(right).all()))
                    self.assertEqual(draw.call_count, 3)

    def test_exact_observations_do_not_sample_censoring(self):
        import tuning.testdata2 as tuning_generator
        import testing.testdata2 as testing_generator
        events = torch.tensor([0.5, 1.0, 2.0])
        for module in (tuning_generator, testing_generator):
            with patch.object(module.torch, 'rand_like', side_effect=AssertionError('Unexpected draw')):
                left, right = module._observe(events, 'exact', None)
            torch.testing.assert_close(left, events)
            torch.testing.assert_close(right, events)

    def test_cv_never_contains_held_out_observations(self):
        data = w.generate(w.Case('ph', 'exact', 0.2), 101, 7, None)
        held_out = set(data.test.tolist())
        validation_rows = []
        for train, valid in w.cv_splits(data.train, 321):
            self.assertFalse(held_out.intersection(train.tolist()))
            self.assertFalse(held_out.intersection(valid.tolist()))
            self.assertFalse(set(train.tolist()).intersection(valid.tolist()))
            self.assertEqual(set(train.tolist()) | set(valid.tolist()), set(data.train.tolist()))
            validation_rows.extend(valid.tolist())
        self.assertEqual(sorted(validation_rows), sorted(data.train.tolist()))

    def test_generation_preserves_training_rng(self):
        torch.manual_seed(91)
        expected_state = torch.random.get_rng_state().clone()
        w.generate(w.Case('ph', 'right', 0.2), 64, 12, 5.0)
        torch.testing.assert_close(torch.random.get_rng_state(), expected_state)

    def test_grid_search_visits_every_combination_once_in_stable_order(self):
        # Keep these checks independent of the user's editable tuning grid.
        space = dict(depth=[6, 8], h_dim=[8, 16, 20, 31], mlp_hidden=[8, 16, 20, 31],
                     mlp_layers=[2, 3], shared_layers=[0, 1, 2])
        first = grid_candidates(space)
        self.assertEqual(first, grid_candidates(space))
        self.assertEqual(len({w.fingerprint(c) for c in first}), 192)
        self.assertEqual(first[0], {key: values[0] for key, values in space.items()})
        self.assertEqual(first[-1], {key: values[-1] for key, values in space.items()})
        self.assertEqual({c['shared_layers'] for c in first}, {0, 1, 2})
        for key, choices in space.items():
            self.assertEqual({c[key] for c in first}, set(choices))
        fixed = {key: [choices[0]] for key, choices in space.items()}
        self.assertEqual(len(grid_candidates(fixed)), 1)

    def test_grid_search_uses_only_listed_choices(self):
        space = dict(depth=[8, 6], h_dim=[8, 20], mlp_hidden=[16],
                     mlp_layers=[1], shared_layers=[0])
        candidates = grid_candidates(space)
        self.assertEqual({(c['depth'], c['h_dim']) for c in candidates},
                         {(8, 8), (8, 20), (6, 8), (6, 20)})
        self.assertEqual(len(candidates), 4)

    def test_grid_search_rejects_invalid_choice_lists(self):
        space = dict(depth=[6], h_dim=[8], mlp_hidden=[16],
                     mlp_layers=[1], shared_layers=[0])
        for choices in ([], [8, 8], [0], [-1], [8.0], [True], 8, '8'):
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                grid_candidates(dict(space, h_dim=choices))
        with self.assertRaises(ValueError):
            grid_candidates({'depth': [6]})

    def test_grid_search_loads_all_json_choices_regardless_of_evaluation_limit_or_seed(self):
        space = dict(depth=[6, 8], h_dim=[8, 16, 20, 31], mlp_hidden=[16],
                     mlp_layers=[1], shared_layers=[0, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'grid.json'
            path.write_text(json.dumps(space))
            args = tune.parser().parse_args(['--method', 'NRT', '--search-space', str(path),
                                            '--tuning-evals', '1', '--tune-seed', '321'])
            self.assertEqual(tune.configurations(args), grid_candidates(space))
            self.assertEqual(len(tune.configurations(args)), 16)
            args.tune_seed = 999
            self.assertEqual(tune.configurations(args), grid_candidates(space))


class TuningSelectionTests(unittest.TestCase):
    def test_cv_chooses_nll_optimum_for_nrt_nll_and_brier_for_other_losses(self):
        # Deliberately give NLL and Brier different optimal configurations.
        data = w.generate(w.Case('ph', 'exact', 0.2), 32, 123, None)
        candidates = [dict(depth=2), dict(depth=3)]
        for method, depth, metric in (('NRT_NLL', 2, 'nll'), ('NRT_Brier', 3, 'brier'), ('NRT_2Loss', 3, 'brier')):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                args = tune.parser().parse_args(['--method', 'NRT', '--epochs', '2', '--device', 'cpu'])
                with patch.object(tune, 'fit_model', side_effect=lambda method, data, train, parameters, *rest: parameters), \
                     patch('tuning.train_mlp.evaluate_nll', side_effect=lambda model, *rest: 1.0 if model['depth'] == 2 else 9.0), \
                     patch('tuning.train_mlp.evaluate_brier', side_effect=lambda model, *rest: 9.0 if model['depth'] == 2 else 1.0), \
                     redirect_stdout(io.StringIO()):
                    parameters, training, score, selected_metric = tune.tune_cv(args, method, data, candidates, Path(directory))
                self.assertEqual(parameters['depth'], depth)
                self.assertEqual((score, selected_metric), (1.0, metric))

    def test_tuning_result_with_wrong_metric_requires_retuning(self):
        with tempfile.TemporaryDirectory() as directory:
            args = tune.parser().parse_args(['--method', 'NRT', '--epochs', '2', '--tuning-evals', '1',
                                            '--device', 'cpu', '--n-samples', '32', '--output-dir', directory])
            case = w.Case('ph', 'exact', 0.2)
            candidates = [dict(depth=2)]
            training = tune.training_config(args, 'NRT_NLL')
            with patch.object(tune, 'tune_cv', return_value=({'depth': 2}, training, 1.0, 'nll')), \
                 patch.object(tune, 'fit_model', return_value=object()), \
                 patch.object(tune, 'evaluate', return_value={'average_grid_sse': 1.0}), redirect_stdout(io.StringIO()):
                tune.tune_case(args, 'NRT_NLL', case, candidates)
            path = args.output_dir / 'NRT_NLL' / case.id / 'result.json'
            result = json.loads(path.read_text())
            result['cv_metric'] = 'c_index'
            w.atomic_json(path, result)
            jobs = [('NRT_NLL', case)]
            with patch.object(tune, 'fit_model', side_effect=AssertionError('Unexpected fitting')):
                with self.assertRaisesRegex(ValueError, "expected 'nll'"):
                    tune.tune_case(args, 'NRT_NLL', case, candidates)
                with self.assertRaisesRegex(ValueError, "expected 'nll'"):
                    tune.write_summary(args, jobs)
                result['cv_metric'] = 'nll'
                w.atomic_json(path, result)
                with redirect_stdout(io.StringIO()) as output:
                    tune.tune_case(args, 'NRT_NLL', case, candidates)
                self.assertIn('Reusing', output.getvalue())


class TuningArrayTests(unittest.TestCase):
    def jobs(self, method, task_id, *extra):
        args = tune.parser().parse_args(['--method', method, '--task-id', str(task_id), *extra])
        methods = args.nrt_methods.split(',') if method == 'NRT' else [method]
        return tune.tuning_jobs(args, w.select_cases(args.cases), methods)

    def test_array_covers_each_applicable_model_case_once(self):
        for family, methods, task_count in (
            ('NRT', list(w.NRT_LOSSES), 105), ('NNCDE', ['NNCDE'], 15),
            ('DeepSurv', ['DeepSurv'], 15),
        ):
            with self.subTest(method=family):
                tasks = [self.jobs(family, task) for task in range(task_count)]
                self.assertTrue(all(len(jobs) == 1 for jobs in tasks))
                jobs = [job for task in tasks for job in task]
                expected = {(method, case.id) for method in methods for case in w.ALL_CASES
                            if case.censor_type in w.SUPPORTED[method]}
                self.assertEqual(len(jobs), len(expected))
                self.assertEqual({(method, case.id) for method, case in jobs}, expected)
        for task, method, case in (
            (0, 'NRT_NLL', w.ALL_CASES[0]), (34, 'NRT_NLL', w.ALL_CASES[-1]),
            (35, 'NRT_Brier', w.ALL_CASES[0]), (69, 'NRT_Brier', w.ALL_CASES[-1]),
            (70, 'NRT_2Loss', w.ALL_CASES[0]), (104, 'NRT_2Loss', w.ALL_CASES[-1]),
        ):
            self.assertEqual(self.jobs('NRT', task), [(method, case)])
        for method in ('NNCDE', 'DeepSurv'):
            self.assertEqual(self.jobs(method, 3), [(method, w.Case('lognormal', 'exact', 0.2))])
            self.assertEqual(self.jobs(method, 14), [(method, w.Case('multimodal_closer', 'right', 0.5))])

    def test_filters_preserve_array_task_mapping(self):
        case = w.ALL_CASES[-1]
        filtered = ['--cases', case.id, '--nrt-methods', 'NRT_Brier']
        jobs = [job for task in range(105) for job in self.jobs('NRT', task, *filtered)]
        self.assertEqual(jobs, [('NRT_Brier', case)])
        self.assertEqual(self.jobs('NRT', 69, *filtered), [('NRT_Brier', case)])
        for method in ('NNCDE', 'DeepSurv'):
            case = w.Case('multimodal_closer', 'right', 0.5)
            filtered = ['--cases', case.id]
            jobs = [job for task in range(15) for job in self.jobs(method, task, *filtered)]
            self.assertEqual(jobs, [(method, case)])
            self.assertEqual(self.jobs(method, 14, *filtered), [(method, case)])

    def test_invalid_task_ids_are_rejected(self):
        for method, task_id in (('NRT', -1), ('NRT', 105), ('NNCDE', -1), ('NNCDE', 15), ('DeepSurv', 15)):
            with self.subTest(method=method, task_id=task_id), self.assertRaisesRegex(ValueError, 'task ID'):
                self.jobs(method, task_id)

    def test_excluded_worker_skips_without_cuda_or_fitting(self):
        for method in ('NRT', 'NNCDE', 'DeepSurv'):
            with self.subTest(method=method), patch('sys.argv', ['tuning.main', '--method', method, '--task-id', '3',
                                                               '--cases', 'ph_right_q0.2']), \
                 patch.object(tune, 'node_preflight', side_effect=AssertionError('Unexpected CUDA check')), \
                 patch.object(tune, 'tune_case', side_effect=AssertionError('Unexpected fitting')), \
                 redirect_stdout(io.StringIO()) as output:
                tune.main()
                self.assertIn('Skipping', output.getvalue())

    def test_list_cases_matches_applicable_array_order(self):
        for method, count in (('NRT', 35), ('NNCDE', 15), ('DeepSurv', 15)):
            with self.subTest(method=method), patch('sys.argv', ['tuning.main', '--method', method, '--list-cases']), \
                 redirect_stdout(io.StringIO()) as output:
                tune.main()
            cases = output.getvalue().splitlines()
            self.assertEqual(len(cases), count)
            self.assertEqual(cases, [self.jobs(method, task)[0][1].id for task in range(count)])

    def test_workers_fit_one_case_and_preserve_other_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / 'NRT_NRT_NLL_NRT_Brier_NRT_2Loss_summary.csv'
            combined.write_text('existing combined summary\n')

            def save_result(args, method, case, candidates):
                w.atomic_json(root / method / case.id / 'result.json', dict(
                    method=method, case_id=case.id, tau=None, right_censoring_rate=0,
                    cv_metric='nll' if method == 'NRT_NLL' else 'brier', cv_score=0.2, average_grid_sse=1.0,
                    parameters={'depth': 6, 'shared_layers': 2}))

            with patch.object(tune, 'tune_case', side_effect=save_result) as fit, \
                 patch.object(tune, 'node_preflight'), redirect_stdout(io.StringIO()):
                for task_id in (0, 1, 35):
                    with patch('sys.argv', ['tuning.main', '--method', 'NRT', '--task-id', str(task_id),
                                            '--device', 'cpu', '--output-dir', directory]):
                        tune.main()
            self.assertEqual([(call.args[1], call.args[2]) for call in fit.call_args_list], [
                ('NRT_NLL', w.ALL_CASES[0]), ('NRT_NLL', w.ALL_CASES[1]), ('NRT_Brier', w.ALL_CASES[0])])
            self.assertEqual(combined.read_text(), 'existing combined summary\n')
            self.assertFalse((root / 'hyper_para').exists())
            import csv
            for method, case in (('NRT_NLL', w.ALL_CASES[0]), ('NRT_NLL', w.ALL_CASES[1]), ('NRT_Brier', w.ALL_CASES[0])):
                with (root / method / case.id / 'summary.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual((rows[0]['method'], rows[0]['case_id']), (method, case.id))

            # Consolidation reads all selected results after workers finish.
            argv = ['tuning.main', '--method', 'NRT', '--nrt-methods', 'NRT_NLL',
                    '--cases', ','.join(case.id for case in w.ALL_CASES[:2]),
                    '--output-dir', directory, '--summarize']
            with patch('sys.argv', argv), redirect_stdout(io.StringIO()), \
                 patch.object(tune, 'node_preflight', side_effect=AssertionError('Unexpected CUDA check')), \
                 patch.object(tune, 'tune_case', side_effect=AssertionError('Unexpected fitting')):
                tune.main()
            with (root / 'NRT_NRT_NLL_summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['case_id'] for row in rows], [case.id for case in w.ALL_CASES[:2]])
            with (root / 'hyper_para' / 'NRT_NLL.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['data_types'] for row in rows], [case.id for case in w.ALL_CASES[:2]])
            self.assertEqual(json.loads(rows[0]['parameters']), {'depth': 6, 'shared_layers': 2})
            self.assertEqual(rows[0]['tau'], '')

    def test_last_worker_publishes_method_and_family_hyperparameters(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = w.ALL_CASES[:2]
            combined = root / 'NRT_NRT_NLL_NRT_Brier_NRT_2Loss_summary.csv'
            combined.write_text('existing combined summary\n')

            def save_result(args, method, case, candidates):
                w.atomic_json(root / method / case.id / 'result.json', dict(
                    method=method, case_id=case.id, tau=None, right_censoring_rate=0,
                    cv_metric=tune.CV_METRICS[method], cv_score=0.2, average_grid_sse=1.0,
                    parameters={'depth': 6, 'shared_layers': 2}))

            with patch.object(tune, 'tune_case', side_effect=save_result), \
                 patch.object(tune, 'node_preflight'), redirect_stdout(io.StringIO()):
                for task in (0, 1, 35, 36, 70, 71):
                    with patch('sys.argv', ['tuning.main', '--method', 'NRT', '--task-id', str(task),
                                            '--cases', ','.join(case.id for case in cases),
                                            '--output-dir', directory]):
                        tune.main()
                    for offset, method in enumerate(w.NRT_LOSSES):
                        self.assertEqual((root / 'hyper_para' / f'{method}.csv').exists(),
                                         task >= offset * 35 + 1)
                    if task != 71:
                        self.assertEqual(combined.read_text(), 'existing combined summary\n')
            with combined.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            for method in w.NRT_LOSSES:
                with (root / 'hyper_para' / f'{method}.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row['data_types'] for row in rows], [case.id for case in cases])
                self.assertEqual(json.loads(rows[0]['parameters']), {'depth': 6, 'shared_layers': 2})

    def test_benchmark_resumption_publishes_complete_supported_cases(self):
        import csv
        for method in ('NNCDE', 'DeepSurv'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cases = [case for case in w.ALL_CASES if case.censor_type in w.SUPPORTED[method]]
                for case in cases:
                    w.atomic_json(root / method / case.id / 'result.json', dict(
                        method=method, case_id=case.id, tau=None if case.censor_type == 'exact' else 5.0,
                        right_censoring_rate=0,
                        cv_metric=tune.CV_METRICS[method], cv_score=0.2, average_grid_sse=1.0,
                        parameters={'hidden_nodes': 64},
                        request={'search_ranges': tune.OFFICIAL_TUNING_PARAMETERS['reference_search']['ranges']}
                        if method == 'DeepSurv' else {}))
                # A resumed worker must publish even though it reuses its result.
                with patch('sys.argv', ['tuning.main', '--method', method, '--task-id', '0',
                                        '--output-dir', directory]), \
                     patch.object(tune, 'node_preflight'), \
                     patch.object(tune, 'tune_case') as reuse, redirect_stdout(io.StringIO()):
                    tune.main()
                reuse.assert_called_once()
                with (root / 'hyper_para' / f'{method}.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 15)
                self.assertEqual([row['data_types'] for row in rows], [case.id for case in cases])
                for row, case in zip(rows, cases):
                    self.assertEqual(row['tau'], '' if case.censor_type == 'exact' else '5.0')
                self.assertTrue((root / f'{method}_{method}_summary.csv').is_file())

    def test_summary_rejects_incomplete_results_and_array_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['tuning.main', '--method', 'NNCDE', '--output-dir', directory, '--summarize']
            with patch('sys.argv', argv), self.assertRaisesRegex(RuntimeError, '15 tuning results missing'):
                tune.main()
            self.assertFalse(list(Path(directory).glob('*.csv')))
            with patch('sys.argv', [*argv, '--task-id', '0']), self.assertRaisesRegex(ValueError, 'without --task-id'):
                tune.main()


class SimulationHyperparameterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.case = w.Case('ph', 'right', 0.2)
        self.args = sim.parser().parse_args(['run', '--hyper-dir', str(self.root)])

    def save_csv(self, method, tau=5.0, rows=1):
        row = dict(model_name=method, data_types=self.case.id,
                   parameters=json.dumps({'depth': 3}), tau=tau)
        w.write_csv(self.root / f'{method}.csv', [row] * rows)

    def test_csvs_cover_all_neural_fits_without_tuning_manifests(self):
        self.assertEqual(sim.parser().parse_args(['run']).hyper_dir, w.ROOT / 'testing' / 'hyper_para')
        args = sim.parser().parse_args(['run', '--hyper-dir', str(self.root),
                                      '--tuning-dir', str(self.root / 'missing')])
        for method in (*w.NRT_LOSSES, 'NNCDE', 'DeepSurv'):
            w.write_csv(self.root / f'{method}.csv', [dict(
                model_name=method, data_types=case.id, parameters=json.dumps({'depth': 3}),
                tau=None if case.censor_type == 'exact' else 5.0)
                for case in w.ALL_CASES if case.censor_type in w.SUPPORTED[method]])
        for method in (*w.NRT_LOSSES, 'NNCDE', 'DeepSurv'):
            for case in w.ALL_CASES:
                if case.censor_type not in w.SUPPORTED[method]:
                    continue
                manifest, parameters, training, _ = sim.load_spec(args, method, case)
                self.assertEqual(parameters, {'depth': 3})
                expected_tau = None if case.censor_type == 'exact' else 5.0
                self.assertEqual(manifest['tau'], expected_tau)
                self.assertEqual(training['epochs'], w.DEFAULT_EPOCHS[method])

    def test_csv_specs_work_without_tuning_manifests_and_track_parameter_changes(self):
        self.save_csv('NRT_NLL')
        manifest, parameters, training, original_hash = sim.load_spec(self.args, 'NRT_NLL', self.case)
        self.assertEqual(parameters, {'depth': 3})
        self.assertEqual(manifest['tau'], 5.0)
        self.assertEqual(training['lr'], 1e-3)
        w.write_csv(self.root / 'NRT_NLL.csv', [dict(model_name='NRT_NLL', data_types=self.case.id,
                     parameters=json.dumps({'depth': 4}), tau=5.0)])
        changed_args = sim.parser().parse_args(['run', '--hyper-dir', str(self.root), '--epochs', '7'])
        changed = sim.load_spec(changed_args, 'NRT_NLL', self.case)
        self.assertNotEqual(changed[3], original_hash)
        self.assertEqual(changed[2]['epochs'], 7)

    def test_explicit_csv_uses_new_parameters_and_tau_independently_of_other_files(self):
        self.save_csv('NRT_2Loss', tau=5.0)
        original = sim.load_spec(self.args, 'NRT_2Loss', self.case)
        path = self.root / 'NRT_2Loss_custom.csv'
        w.write_csv(path, [dict(model_name='NRT_2Loss', data_types=self.case.id,
                               parameters=json.dumps({'depth': 7}), tau=6.0)])
        (self.root / 'NRT_NLL.csv').write_text('invalid CSV\n')
        args = sim.parser().parse_args([
            'run', '--methods', 'NRT_2Loss', '--hyper-dir', str(self.root),
            '--hyper-file', str(path),
        ])
        manifest, parameters, training, source_hash = sim.load_spec(args, 'NRT_2Loss', self.case)
        self.assertEqual(manifest['tau'], 6.0)
        self.assertEqual(parameters, {'depth': 7})
        self.assertEqual(training['epochs'], 200)
        self.assertNotEqual(source_hash, original[3])
        self.assertEqual(set(sim.hyperparameter_tables(args)[0]), {'NRT_2Loss'})

    def test_explicit_csv_requires_one_neural_method_and_reports_missing_path(self):
        path = self.root / 'missing.csv'
        for methods in ('NRT_2Loss,NRT_NLL', 'cox'):
            args = sim.parser().parse_args([
                'run', '--methods', methods, '--hyper-file', str(path),
            ])
            with self.subTest(methods=methods), self.assertRaisesRegex(ValueError, 'exactly one neural'):
                sim.hyperparameter_tables(args)
        args = sim.parser().parse_args([
            'run', '--methods', 'NRT_2Loss', '--hyper-file', str(path),
        ])
        with self.assertRaisesRegex(FileNotFoundError, 'missing.csv'):
            sim.load_spec(args, 'NRT_2Loss', self.case)

    def test_conflicting_tau_and_duplicate_rows_are_rejected(self):
        self.save_csv('NRT_NLL')
        self.save_csv('NRT_Brier', tau=6.0)
        with self.assertRaisesRegex(ValueError, 'disagree on tau'):
            sim.load_spec(self.args, 'NRT_NLL', self.case)
        self.save_csv('NRT_Brier', rows=2)
        with self.assertRaisesRegex(ValueError, 'duplicate case'):
            sim.load_spec(self.args, 'NRT_NLL', self.case)

    def test_missing_parameter_file_and_case_are_rejected(self):
        self.save_csv('NRT_NLL')
        with self.assertRaisesRegex(FileNotFoundError, 'NNCDE.csv'):
            sim.load_spec(self.args, 'NNCDE', self.case)
        self.save_csv('NNCDE')
        args = sim.parser().parse_args(['run', '--hyper-dir', str(self.root)])
        with self.assertRaisesRegex(ValueError, 'No hyperparameters'):
            sim.load_spec(args, 'NNCDE', w.Case('ph', 'exact', 0.2))

    def test_invalid_saved_tau_is_rejected(self):
        for tau in (None, -1, float('inf'), float('nan')):
            with self.subTest(tau=tau):
                self.save_csv('NRT_NLL', tau=tau)
                with self.assertRaisesRegex(ValueError, 'invalid tau'):
                    sim.load_spec(self.args, 'NRT_NLL', self.case)


class SimulationCPUManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.args = sim.parser().parse_args([
            'run', '--device', 'cpu', '--tuning-dir', str(self.root),
            '--hyper-dir', str(self.root / 'hyper_para'),
        ])

    def save_manifest(self, case, **changes):
        manifest = dict(schema_version=w.SCHEMA_VERSION, case=asdict(case), case_id=case.id,
                        tau=None if case.censor_type == 'exact' else 7.25, mu=1.5, r=2.0)
        manifest.update(changes)
        path = self.root / 'cases' / f'{case.id}.json'
        w.atomic_json(path, manifest)
        return manifest

    def test_all_cpu_specs_use_saved_cases_without_reading_csvs(self):
        self.assertEqual(sim.parser().parse_args(['run']).tuning_dir, w.ROOT / 'tuning' / 'output')
        # A malformed neural table must not affect CPU simulations.
        self.args.hyper_dir.mkdir()
        (self.args.hyper_dir / 'NRT_NLL.csv').write_text('invalid CSV\n')
        count = 0
        with patch.object(sim, 'hyperparameter_tables', side_effect=AssertionError('Unexpected CSV read')):
            for case in w.ALL_CASES:
                expected = self.save_manifest(case)
                for method in w.CPU_METHODS:
                    if case.censor_type not in w.SUPPORTED[method]:
                        continue
                    with self.subTest(method=method, case=case.id):
                        spec = sim.load_spec(self.args, method, case)
                        self.assertEqual(spec[0], expected)
                        job = sim.make_job(self.args, method, case, 1, spec)
                        for key in ('tau', 'mu', 'r'):
                            self.assertEqual(job['request'][key], expected[key])
                        count += 1
        self.assertEqual(count, 140)

    def test_missing_manifest_reports_the_case_path(self):
        case = w.Case('ph', 'right', 0.2)
        with self.assertRaisesRegex(FileNotFoundError, f'{case.id}.json'):
            sim.load_spec(self.args, 'cox', case)

    def test_invalid_tau_and_mismatched_cases_are_rejected(self):
        case = w.Case('ph', 'right', 0.2)
        for tau in (None, -1, 0, True, '7.25'):
            with self.subTest(tau=tau):
                self.save_manifest(case, tau=tau)
                with self.assertRaisesRegex(ValueError, 'invalid tau'):
                    sim.load_spec(self.args, 'cox', case)
        self.save_manifest(case, case_id='other')
        with self.assertRaisesRegex(ValueError, 'Invalid tuning case manifest'):
            sim.load_spec(self.args, 'cox', case)
        exact = w.Case('ph', 'exact', 0.2)
        self.save_manifest(exact, tau=7.25)
        with self.assertRaisesRegex(ValueError, 'null tau'):
            sim.load_spec(self.args, 'cox', exact)

    def test_changing_saved_tau_changes_the_simulation_hash(self):
        case = w.Case('ph', 'right', 0.2)
        self.save_manifest(case)
        original = sim.make_job(self.args, 'cox', case, 1, sim.load_spec(self.args, 'cox', case))
        self.save_manifest(case, tau=8.0)
        changed = sim.make_job(self.args, 'cox', case, 1, sim.load_spec(self.args, 'cox', case))
        self.assertNotEqual(original['request_hash'], changed['request_hash'])


class ResultWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.case = w.Case('ph', 'right', 0.2)
        self.args = sim.parser().parse_args(['summarize', '--output-dir', str(self.root), '--replications', '2'])

    def save_result(self, method, rep, rate):
        payload = dict(schema_version=w.SCHEMA_VERSION, method=method, case=asdict(self.case),
                       case_id=self.case.id, replication=rep, data_seed=1_000_000 + rep,
                       n_samples=100, grid_size=16, mu=0.0, r=None, tau=5.0,
                       parameters={'depth': 2}, training_config={'epochs': 2}, tuning_hash=method,
                       status='ok', right_censoring_rate=rate, average_grid_sse=1.0,
                       elapsed_seconds=2.0)
        w.atomic_json(sim.result_path(self.root, method, self.case, rep), payload)
        return payload

    def test_summary_counts_data_once_across_methods(self):
        for method in ('NRT_NLL', 'NRT_Brier'):
            self.save_result(method, 1, 0.1)
            self.save_result(method, 2, 0.3)
        with redirect_stdout(io.StringIO()):
            sim.summarize(self.args, [self.case], ['NRT_NLL', 'NRT_Brier'])
        import csv
        with (self.root / 'censoring_summary.csv').open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(int(row['n_replications']), 2)
        self.assertAlmostEqual(float(row['average_right_censoring_rate']), 0.2)
        self.assertEqual(float(row['tau']), 5.0)

    def test_summary_rejects_missing_results(self):
        self.save_result('NRT_NLL', 1, 0.2)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'missing'):
            sim.summarize(self.args, [self.case], ['NRT_NLL'])
        self.assertFalse((self.root / 'tables').exists())

    def test_combined_censoring_report_aggregates_all_replications_by_method(self):
        import csv
        import statistics
        methods = ['NRT_NLL', 'NRT_Brier']
        for method in methods:
            self.save_result(method, 1, 0.1)
            self.save_result(method, 2, 0.3)
        with redirect_stdout(io.StringIO()):
            sim.summarize(self.args, [self.case], methods, censoring_only=True)
        reports = list(self.root.glob('censoring_*.csv'))
        self.assertEqual([path.name for path in reports], ['censoring_NRT_NLL_NRT_Brier_rep1_2.csv'])
        with reports[0].open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['method'] for row in rows], methods)
        for row in rows:
            self.assertEqual(int(row['n_replications']), 2)
            self.assertAlmostEqual(float(row['average_right_censoring_rate']), 0.2)
            self.assertAlmostEqual(float(row['sd_right_censoring_rate']), statistics.stdev([0.1, 0.3]))
            self.assertAlmostEqual(float(row['difference_from_q']), 0.0)
        self.assertFalse((self.root / 'tables').exists())

    def test_combined_censoring_report_preserves_existing_report_until_success(self):
        report = self.root / 'censoring_NRT_NLL_rep1_2.csv'
        report.write_text('existing report\n')
        self.save_result('NRT_NLL', 2, 0.2)
        with redirect_stdout(io.StringIO()):
            sim.summarize(self.args, [self.case], ['NRT_NLL'], complete_only=True, censoring_only=True)
        self.assertEqual(report.read_text(), 'existing report\n')
        failed = self.save_result('NRT_NLL', 1, 0.1)
        failed['status'] = 'error'
        w.atomic_json(sim.result_path(self.root, 'NRT_NLL', self.case, 1), failed)
        with redirect_stdout(io.StringIO()):
            sim.summarize(self.args, [self.case], ['NRT_NLL'], complete_only=True, censoring_only=True)
        self.assertEqual(report.read_text(), 'existing report\n')
        self.save_result('NRT_NLL', 1, 0.1)
        with redirect_stdout(io.StringIO()):
            sim.summarize(self.args, [self.case, w.Case('ph', 'right', 0.5)], ['NRT_NLL'],
                          complete_only=True, censoring_only=True)
        self.assertEqual(report.read_text(), 'existing report\n')
        with redirect_stdout(io.StringIO()):
            sim.summarize(self.args, [self.case], ['NRT_NLL'], complete_only=True, censoring_only=True)
        self.assertNotEqual(report.read_text(), 'existing report\n')

    def test_automatic_summary_waits_for_other_workers_and_successful_replications(self):
        import csv
        self.args.summary_methods = 'NRT_NLL,NNCDE,deep_distribution_regression'
        for rep in (1, 2):
            self.save_result('NRT_NLL', rep, 0.2)
        self.save_result('NNCDE', 2, 0.2)
        with redirect_stdout(io.StringIO()):
            sim.auto_summary(self.args, [self.case], ['NRT_NLL'])
        self.assertFalse((self.root / 'tables').exists())
        failed = self.save_result('NNCDE', 1, 0.2)
        failed['status'] = 'error'
        w.atomic_json(sim.result_path(self.root, 'NNCDE', self.case, 1), failed)
        with redirect_stdout(io.StringIO()):
            sim.auto_summary(self.args, [self.case], ['NNCDE'])
        self.assertFalse((self.root / 'tables').exists())
        self.save_result('NNCDE', 1, 0.2)
        with redirect_stdout(io.StringIO()):
            sim.auto_summary(self.args, [self.case], ['NNCDE'])
        table = self.root / 'tables' / f'{self.case.id}.csv'
        with table.open() as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ['replication', 'NRT_NLL', 'NNCDE'])
            self.assertEqual(len(list(reader)), 4)
        with (self.root / 'censoring_summary.csv').open() as stream:
            self.assertEqual(int(next(csv.DictReader(stream))['n_replications']), 2)
        original = table.read_bytes()
        with redirect_stdout(io.StringIO()):
            sim.auto_summary(self.args, [self.case], ['NRT_NLL'])
        self.assertEqual(table.read_bytes(), original)

    def test_automatic_summary_all_methods_combines_separate_launchers(self):
        import csv
        self.args.summary_methods = 'all'
        applicable = [method for method in w.METHODS if self.case.censor_type in w.SUPPORTED[method]]
        for method in applicable[:-1]:
            for rep in (1, 2):
                self.save_result(method, rep, 0.2)
        with redirect_stdout(io.StringIO()):
            sim.auto_summary(self.args, [self.case], [applicable[-2]])
        self.assertFalse((self.root / 'tables').exists())
        for rep in (1, 2):
            self.save_result(applicable[-1], rep, 0.2)
        with redirect_stdout(io.StringIO()):
            sim.auto_summary(self.args, [self.case], [applicable[-1]])
        with (self.root / 'tables' / f'{self.case.id}.csv').open() as stream:
            self.assertEqual(csv.DictReader(stream).fieldnames, ['replication', *applicable])

    def test_nrt_2loss_summary_has_one_method_column_and_sample_statistics(self):
        import csv
        import statistics
        self.args.summary_methods = 'NRT_2Loss'
        for rep, score in enumerate((1.0, 3.0), start=1):
            payload = self.save_result('NRT_2Loss', rep, 0.2)
            payload['average_grid_sse'] = score
            w.atomic_json(sim.result_path(self.root, 'NRT_2Loss', self.case, rep), payload)
        with redirect_stdout(io.StringIO()):
            sim.auto_summary(self.args, [self.case], ['NRT_2Loss'])
        with (self.root / 'tables' / f'{self.case.id}.csv').open() as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ['replication', 'NRT_2Loss'])
            rows = list(reader)
        self.assertEqual([row['replication'] for row in rows], ['1', '2', 'average', 'standard_deviation'])
        self.assertEqual(float(rows[-2]['NRT_2Loss']), 2.0)
        self.assertAlmostEqual(float(rows[-1]['NRT_2Loss']), statistics.stdev([1.0, 3.0]))

    def test_automatic_summary_publishes_complete_cases_and_preserves_pending_tables(self):
        self.args.summary_methods = None
        pending_case = w.Case('ph', 'right', 0.5)
        pending_table = self.root / 'tables' / f'{pending_case.id}.csv'
        w.write_csv(pending_table, [{'replication': 'existing', 'NRT_NLL': 9}])
        original = pending_table.read_bytes()
        censoring = self.root / 'censoring_summary.csv'
        censoring.write_text('existing censoring report\n')
        for rep in (1, 2):
            self.save_result('NRT_NLL', rep, 0.2)
        with redirect_stdout(io.StringIO()):
            sim.auto_summary(self.args, [self.case, pending_case], ['NRT_NLL'])
        self.assertTrue((self.root / 'tables' / f'{self.case.id}.csv').exists())
        self.assertEqual(pending_table.read_bytes(), original)
        self.assertEqual(censoring.read_text(), 'existing censoring report\n')

    def test_run_and_resumed_array_worker_publish_tables_and_one_censoring_report(self):
        import csv
        args = sim.parser().parse_args([
            'run', '--methods', 'NRT_NLL', '--output-dir', str(self.root),
            '--replications', '2', '--replication', '2', '--n-samples', '32',
            '--grid-size', '16', '--device', 'cpu',
        ])
        case = self.case
        spec = ({'tau': 5.0, 'mu': 0.0, 'r': None}, {}, {}, 'source')
        first = sim.make_job(args, 'NRT_NLL', case, 1, spec)
        data = w.generate(case, args.n_samples, first['request']['data_seed'], 5.0)
        w.atomic_json(first['output'], {**first['request'], **data.metadata(),
                      'case_id': case.id, 'request_hash': first['request_hash'],
                      'status': 'ok', 'average_grid_sse': 1.0})
        with patch.object(sim, 'node_preflight'), patch.object(sim, 'load_spec', return_value=spec), \
             patch.object(sim, 'fit_model', return_value=object()) as fit, \
             patch.object(sim, 'evaluate', return_value={'average_grid_sse': 3.0}), \
             redirect_stdout(io.StringIO()):
            sim.run(args, [case], ['NRT_NLL'])
            sim.run(args, [case], ['NRT_NLL'])
        self.assertEqual(fit.call_count, 1)
        table = self.root / 'tables' / f'{case.id}.csv'
        with table.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(float(rows[-2]['NRT_NLL']), 2.0)
        self.assertFalse((self.root / 'censoring_NRT_NLL_rep2_2.csv').exists())
        with (self.root / 'censoring_NRT_NLL_rep1_2.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['n_replications'], '2')
        self.assertTrue(sim.result_path(self.root, 'NRT_NLL', case, 2).is_file())
        # A launcher waiting for other methods still publishes its own report.
        args.summary_methods = 'all'
        table.unlink()
        report = self.root / 'censoring_NRT_NLL_rep1_2.csv'
        report.unlink()
        with patch.object(sim, 'node_preflight'), patch.object(sim, 'load_spec', return_value=spec), \
             patch.object(sim, 'fit_model', side_effect=AssertionError('Unexpected refitting')), \
             redirect_stdout(io.StringIO()):
            sim.run(args, [case], ['NRT_NLL'])
        self.assertTrue(report.is_file())
        self.assertFalse(table.exists())

    def test_summary_has_500_replications_and_method_means_and_sample_sd(self):
        import csv
        import statistics
        self.args.replications = 500
        methods = ['NRT_NLL', 'NRT_Brier']
        for rep in range(1, 501):
            for multiplier, method in enumerate(methods, start=1):
                payload = self.save_result(method, rep, 0.2)
                payload['average_grid_sse'] = multiplier * rep
                w.atomic_json(sim.result_path(self.root, method, self.case, rep), payload)
        with redirect_stdout(io.StringIO()):
            sim.summarize(self.args, [self.case], [*methods, 'deep_distribution_regression'])
        with (self.root / 'tables' / f'{self.case.id}.csv').open() as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ['replication', *methods])
            rows = list(reader)
        self.assertEqual(len(rows), 502)
        self.assertEqual([row['replication'] for row in rows[:500]], [str(rep) for rep in range(1, 501)])
        self.assertEqual(rows[-2]['replication'], 'average')
        self.assertEqual(rows[-1]['replication'], 'standard_deviation')
        for multiplier, method in enumerate(methods, start=1):
            self.assertEqual(float(rows[0][method]), multiplier)
            self.assertEqual(float(rows[499][method]), 500 * multiplier)
            self.assertEqual(float(rows[-2][method]), 250.5 * multiplier)
            self.assertAlmostEqual(float(rows[-1][method]), statistics.stdev(range(1, 501)) * multiplier)

    def test_partial_summary_preserves_missing_and_failed_replication_rows(self):
        import csv
        self.args.allow_incomplete = True
        self.save_result('NRT_NLL', 2, 0.2)
        failed = self.save_result('NRT_Brier', 1, 0.2)
        failed['status'] = 'error'
        w.atomic_json(sim.result_path(self.root, 'NRT_Brier', self.case, 1), failed)
        with redirect_stdout(io.StringIO()):
            sim.summarize(self.args, [self.case], ['NRT_NLL', 'NRT_Brier'])
        with (self.root / 'tables' / f'{self.case.id}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], {'replication': '1', 'NRT_NLL': '', 'NRT_Brier': ''})
        self.assertEqual(rows[1]['NRT_NLL'], '1.0')
        self.assertEqual(rows[-2]['NRT_Brier'], '')

    def test_summary_rejects_inconsistent_data(self):
        for rep in (1, 2):
            self.save_result('NRT_NLL', rep, 0.2)
            self.save_result('NRT_Brier', rep, 0.3)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'different simulation data'):
            sim.summarize(self.args, [self.case], ['NRT_NLL', 'NRT_Brier'])

    def test_resume_skips_identical_success_and_rejects_changed_request(self):
        path = self.root / 'result.json'
        w.atomic_json(path, {'status': 'ok', 'request_hash': 'same'})
        job = dict(output=path, force=False, request_hash='same', request={})
        with patch.object(sim, 'fit_model', side_effect=AssertionError('Unexpected fitting')):
            self.assertEqual(sim.fit_job(job)['status'], 'ok')
        job['request_hash'] = 'different'
        with self.assertRaisesRegex(ValueError, 'different simulation'):
            sim.fit_job(job)


if __name__ == '__main__':
    unittest.main()
