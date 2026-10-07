"""Exercise fail-fast node checks without using real GPUs or Slurm jobs."""
from contextlib import redirect_stdout, redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import workflow as w
from testing import simulation as sim
from tuning import main as tune


class NodePreflightTests(unittest.TestCase):
    def test_cpu_executes_and_imports_only_selected_backend(self):
        module = SimpleNamespace(NNCDE=object())
        with patch.object(w.importlib, 'import_module', return_value=module) as imports, \
             redirect_stdout(io.StringIO()) as output:
            w.node_preflight('cpu', ['NNCDE'], 'testing')
        imports.assert_called_once_with('testing.benchmark.nncde.model')
        self.assertIn('CPU execution preflight passed', output.getvalue())

    def test_gpu_probe_executes_and_synchronizes_selected_device(self):
        values = torch.arange(1, 4097, dtype=torch.float32)
        properties = SimpleNamespace(name='test GPU', major=9, minor=0)
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=2), \
             patch.object(torch.cuda, 'set_device') as select, \
             patch.object(torch.cuda, 'synchronize') as synchronize, \
             patch.object(torch.cuda, 'get_device_properties', return_value=properties), \
             patch.object(torch, 'arange', return_value=values) as allocate, \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(w.check_device('cuda:1'), torch.device('cuda:1'))
        select.assert_called_once_with(1)
        synchronize.assert_called_once_with(torch.device('cuda:1'))
        self.assertEqual(allocate.call_args.kwargs['device'], torch.device('cuda:1'))
        self.assertIn('test GPU', output.getvalue())

    def test_gpu_unavailable_invisible_or_execution_failure_uses_startup_exit(self):
        values = torch.ones(4)
        for available, count, sync_error in (
            (False, 2, None), (True, 0, None), (True, 2, RuntimeError('CUDA kernel failed')),
        ):
            with self.subTest(available=available, count=count, sync_error=sync_error), \
                 patch.object(torch.cuda, 'is_available', return_value=available), \
                 patch.object(torch.cuda, 'device_count', return_value=count), \
                 patch.object(torch.cuda, 'set_device'), \
                 patch.object(torch, 'arange', return_value=values), \
                 patch.object(torch.cuda, 'synchronize', side_effect=sync_error), \
                 patch.object(w.importlib, 'import_module') as imports, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                w.node_preflight('cuda:0', ['NNCDE'], 'testing')
            self.assertEqual(error.exception.code, w.PREFLIGHT_EXIT_CODE)
            imports.assert_not_called()

    def test_missing_r_packages_fail_before_fitting(self):
        module = SimpleNamespace(CTree=object(), r_backend_available=lambda: False)
        with patch.object(w.importlib, 'import_module', return_value=module), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as output, \
             self.assertRaises(SystemExit) as error:
            w.node_preflight('cpu', ['ctree'], 'testing')
        self.assertEqual(error.exception.code, w.PREFLIGHT_EXIT_CODE)
        self.assertIn('Rscript and R packages partykit, icenReg, survival', output.getvalue())

    def test_successful_r_dependency_probe_is_required(self):
        module = SimpleNamespace(CoxPH=object(), r_backend_available=Mock(return_value=True))
        with patch.object(w.importlib, 'import_module', return_value=module), redirect_stdout(io.StringIO()):
            w.node_preflight('cpu', ['cox'], 'testing')
        module.r_backend_available.assert_called_once_with()

    def test_deepsurv_tuning_checks_search_dependency(self):
        module = SimpleNamespace(DeepSurvCUDA=object())
        with patch.object(w.importlib, 'import_module', side_effect=[module, ImportError('missing optunity')]) as imports, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as output, \
             self.assertRaises(SystemExit) as error:
            w.node_preflight('cpu', ['DeepSurv'], 'tuning')
        self.assertEqual(error.exception.code, w.PREFLIGHT_EXIT_CODE)
        self.assertEqual(imports.call_args.args, ('optunity',))
        self.assertIn('missing optunity', output.getvalue())

    def test_driver_preflight_failure_prevents_work(self):
        with patch('sys.argv', ['tuning.main', '--method', 'NRT', '--task-id', '0', '--device', 'cpu']), \
             patch.object(tune, 'node_preflight', side_effect=SystemExit(75)) as preflight, \
             patch.object(tune, 'tune_case') as fit, self.assertRaises(SystemExit):
            tune.main()
        preflight.assert_called_once_with('cpu', {'NRT_NLL'}, 'tuning')
        fit.assert_not_called()
        args = sim.parser().parse_args(['run', '--device', 'cpu', '--replications', '1'])
        with patch.object(sim, 'node_preflight', side_effect=SystemExit(75)) as preflight, \
             patch.object(sim, 'load_spec') as load, patch.object(sim, 'fit_job') as fit, \
             self.assertRaises(SystemExit):
            sim.run(args, [w.ALL_CASES[0]], ['ctree'])
        preflight.assert_called_once_with('cpu', {'ctree'}, 'testing')
        load.assert_not_called()
        fit.assert_not_called()

    def test_preview_and_summary_skip_node_checks(self):
        for options in (['--list-cases'], ['--dry-run'], ['--summarize']):
            with self.subTest(options=options), \
                 patch('sys.argv', ['tuning.main', '--method', 'NRT', *options]), \
                 patch.object(tune, 'write_summary'), patch.object(tune, 'node_preflight') as preflight, \
                 redirect_stdout(io.StringIO()):
                tune.main()
            preflight.assert_not_called()
        args = sim.parser().parse_args(['run', '--device', 'cpu', '--dry-run'])
        with patch.object(sim, 'node_preflight') as preflight, redirect_stdout(io.StringIO()):
            sim.run(args, [w.ALL_CASES[0]], ['ctree'])
        preflight.assert_not_called()


if __name__ == '__main__':
    unittest.main()
