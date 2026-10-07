"""Check CPU phase isolation with mocked fits and worker pools."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import workflow as w
from testing import simulation as sim


class CPUSchedulingTests(unittest.TestCase):
    def check_schedule(self, methods, parallel_fits=8, forest_workers=1):
        args = sim.parser().parse_args([
            'run', '--device', 'cpu', '--replications', '2',
            '--parallel-fits', str(parallel_fits),
            '--iccforest-parallel-fits', str(forest_workers),
        ])
        cases = [w.Case('ph', 'right', 0.2), w.Case('ph', 'current', 0.2)]
        events, pools = [], []
        active_pool = None

        class Pool:
            def __init__(self, **options):
                self.options = options
                pools.append(options)

            def __enter__(self):
                nonlocal active_pool
                active_pool = self
                events.append('pool open')
                return self

            def __exit__(self, *exc):
                nonlocal active_pool
                events.append('pool closed')
                active_pool = None

            def map(self, function, jobs, chunksize):
                return map(function, jobs)

        def fit(job):
            request = job['request']
            method = request['method']
            if method == 'iccforest' and forest_workers == 1:
                self.assertIsNone(active_pool, 'The other worker pool must close before ICcforest')
            events.append((method, job['case'].id, request['replication']))
            return dict(method=method, case_id=job['case'].id, status='ok', right_censoring_rate=0.2)

        spec = ({'tau': 5.0, 'mu': 0.0, 'r': None}, {}, {}, 'source')
        with patch.object(sim, 'node_preflight'), \
             patch.object(sim, 'load_spec', return_value=spec), \
             patch.object(sim, 'fit_job', side_effect=fit), \
             patch.object(sim, 'ProcessPoolExecutor', Pool), \
             patch.object(sim, 'auto_summary'), \
             patch.object(sim, 'write_csv'), redirect_stdout(io.StringIO()):
            sim.run(args, cases, methods)

        actual = [event for event in events if isinstance(event, tuple)]
        expected = {(method, case.id, rep) for method in methods for case in cases
                    if case.censor_type in w.SUPPORTED[method] for rep in (1, 2)}
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected), 'Each requested fit must run exactly once')
        return events, pools

    def test_parallel_methods_finish_before_sequential_forests(self):
        events, pools = self.check_schedule(['iccforest', 'cox', 'aft'])
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]['max_workers'], 8)
        self.assertEqual(pools[0]['max_tasks_per_child'], 1)
        closed = events.index('pool closed')
        self.assertTrue(all(event[0] == 'iccforest' for event in events[closed + 1:]))

    def test_forests_are_separate_when_all_methods_are_sequential(self):
        events, pools = self.check_schedule(['iccforest', 'cox'], parallel_fits=1)
        self.assertFalse(pools)
        self.assertEqual([event[0] for event in events], ['cox'] * 4 + ['iccforest'] * 4)

    def test_forest_only_uses_no_parallel_pool_by_default(self):
        _, pools = self.check_schedule(['iccforest'])
        self.assertFalse(pools)

    def test_other_methods_do_not_create_a_forest_pool(self):
        _, pools = self.check_schedule(['cox'])
        self.assertEqual(len(pools), 1)

    def test_explicit_forest_concurrency_uses_a_separate_pool(self):
        events, pools = self.check_schedule(['cox', 'iccforest'], forest_workers=2)
        self.assertEqual([pool['max_workers'] for pool in pools], [8, 2])
        self.assertEqual(events.count('pool closed'), 2)

    def test_invalid_forest_concurrency_is_rejected_before_running(self):
        with patch('sys.argv', ['simulation', 'run', '--iccforest-parallel-fits', '0']), \
             patch.object(sim, 'run') as run, self.assertRaisesRegex(ValueError, 'Counts must be positive'):
            sim.main()
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
