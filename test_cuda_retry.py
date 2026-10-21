"""Failure injection for CUDA requeue policy; requires no GPU or torch import."""
from contextlib import redirect_stderr
import io
import sys
import unittest
from unittest.mock import patch

import cuda_retry


class CudaRetryTests(unittest.TestCase):
    def test_forwards_module_arguments_and_restores_argv(self):
        original = sys.argv
        seen = []
        with patch.object(cuda_retry.runpy, 'run_module',
                          side_effect=lambda *a, **k: seen.append(sys.argv[:])) as run:
            cuda_retry.main(['realworld.run', 'run', '--device', 'cuda:0'])
        run.assert_called_once_with('realworld.run', run_name='__main__', alter_sys=True)
        self.assertEqual(seen, [['realworld.run', 'run', '--device', 'cuda:0']])
        self.assertIs(sys.argv, original)

    def test_unavailable_device_exits_75_with_original_error(self):
        for message in cuda_retry.CUDA_UNAVAILABLE:
            with self.subTest(message=message):
                error = RuntimeError('CUDA error: ' + message)
                output = io.StringIO()
                with patch.object(cuda_retry.runpy, 'run_module', side_effect=error), \
                     redirect_stderr(output), self.assertRaises(SystemExit) as stopped:
                    cuda_retry.main(['realworld.run', 'run'])
                self.assertEqual(stopped.exception.code, 75)
                self.assertIn(message, output.getvalue())

    def test_model_errors_and_out_of_memory_do_not_request_requeue(self):
        for error in (ValueError('Data/source/config/environment changed'),
                      RuntimeError('CUDA out of memory'),
                      RuntimeError('CUDA error: device-side assert triggered'),
                      RuntimeError('nonfinite training objective')):
            with self.subTest(error=str(error)), \
                 patch.object(cuda_retry.runpy, 'run_module', side_effect=error), \
                 self.assertRaises(type(error)) as stopped:
                cuda_retry.main(['realworld.run', 'run'])
            self.assertIs(stopped.exception, error)

    def test_driver_exit_codes_are_preserved(self):
        for status in (0, 2, 75):
            with self.subTest(status=status), \
                 patch.object(cuda_retry.runpy, 'run_module', side_effect=SystemExit(status)), \
                 self.assertRaises(SystemExit) as stopped:
                cuda_retry.main(['realworld.run', 'preflight'])
            self.assertEqual(stopped.exception.code, status)


if __name__ == '__main__':
    unittest.main()
