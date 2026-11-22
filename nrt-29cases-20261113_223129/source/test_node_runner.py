"""Inject node failures without submitting jobs or requiring a GPU."""
from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import call, patch

import torch
import node_runner as runner
import workflow


class NodeRunnerTests(unittest.TestCase):
    def test_precheck_failure_stops_every_entrypoint_with_retry_status(self):
        for module in runner.MODULES:
            for error in (RuntimeError('CPU calculation failed'), RuntimeError('CUDA unavailable'),
                          ImportError('R/Python dependency unavailable'), OSError('temporary disk unavailable')):
                with self.subTest(module=module, error=str(error)), \
                     patch.object(runner, 'precheck', side_effect=error), \
                     patch.object(runner.cuda_retry, 'main') as run, \
                     redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                    runner.main([module, '--device', 'cpu'])
                self.assertEqual(stopped.exception.code, 75)
                run.assert_not_called()

    def test_cpu_and_gpu_checks_precede_backend_loading(self):
        with patch.object(workflow, 'check_device') as device, \
             patch.object(workflow, 'node_preflight') as backend:
            runner.precheck('tuning.main', runner.options('tuning.main', ['--device', 'cuda:1', '--method', 'DeepSurv']))
        self.assertEqual(device.call_args_list, [call('cpu'), call('cuda:1')])
        backend.assert_called_once_with('cpu', ['DeepSurv'], 'tuning')

    def test_realworld_cpu_checks_selected_r_backends_and_python_dependencies(self):
        for method, expected in (('Cox', []), ('Ctree', ['ctree']), ('ICcforest', ['iccforest'])):
            with self.subTest(method=method), patch.object(workflow, 'node_preflight') as backend, \
                 patch.object(runner.importlib, 'import_module') as dependency, redirect_stdout(io.StringIO()):
                runner.precheck('realworld.run', runner.options('realworld.run', ['run', '--device', 'cpu', '--methods', method]))
            backend.assert_called_once_with('cpu', expected, 'testing')
            self.assertEqual(dependency.call_args_list, [call('optuna'), call('lifelines'), call('sksurv')])

    def test_nrt_tuning_checks_optuna(self):
        with patch.object(runner.importlib, 'import_module') as dependency, redirect_stdout(io.StringIO()):
            runner.precheck('regularization.tuning.main', runner.options('regularization.tuning.main', ['--device', 'cpu']))
        dependency.assert_called_once_with('optuna')

    def test_prepare_and_summaries_check_cpu_without_gpu(self):
        for module, args in (('tuning.prepare_cases', []), ('realworld.run', ['prepare']),
                             ('realworld.run', ['summarize']), ('tuning.main', ['--summarize']),
                             ('testing.simulation', ['summarize']),
                             ('regularization.testing.simulation', ['summarize'])):
            with self.subTest(module=module), patch.object(workflow, 'check_device') as device, \
                 patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CPU command accessed CUDA')):
                runner.precheck(module, runner.options(module, args + ['--device', 'cuda:0']))
            device.assert_called_once_with('cpu')

    def test_previews_skip_checks_but_forward_original_arguments(self):
        for flag in ('--help', '-h', '--dry-run', '--list-cases'):
            with self.subTest(flag=flag), patch.object(runner, 'precheck') as probe, \
                 patch.object(runner.cuda_retry, 'main') as run:
                runner.main(['tuning.main', '--method', 'NNCDE', flag])
            probe.assert_not_called()
            run.assert_called_once_with(['tuning.main', '--method', 'NNCDE', flag])

    def test_model_error_is_not_converted_to_a_node_failure(self):
        with patch.object(runner, 'precheck'), \
             patch.object(runner.cuda_retry, 'main', side_effect=ValueError('CSV checksum changed')), \
             self.assertRaisesRegex(ValueError, 'CSV checksum changed'):
            runner.main(['realworld.run', 'run', '--device', 'cpu'])

    def test_invalid_device_is_a_configuration_error(self):
        with patch.object(runner, 'precheck') as probe, redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as stopped:
            runner.main(['tuning.main', '--device', 'typo'])
        self.assertEqual(stopped.exception.code, 2)
        probe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
