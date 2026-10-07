"""Exercise launcher startup with Slurm spool copies, without fitting models."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
LAUNCHERS = (
    ('tuning/tune_nrt.sh', 'tuning.main', 'NRT', '0'),
    ('tuning/tune_nncde.sh', 'tuning.main', 'NNCDE', '0'),
    ('tuning/tune_deepsurv.sh', 'tuning.main', 'DeepSurv', '0'),
    ('testing/run_nrt.sh', 'testing.simulation', 'NRT_NLL', '0'),
    ('testing/run_nncde.sh', 'testing.simulation', 'NNCDE', None),
    ('testing/run_deepsurv.sh', 'testing.simulation', 'DeepSurv', None),
    ('testing/run_cpu.sh', 'testing.simulation',
     'deep_distribution_regression,ctree,iccforest,cox,aft', '1'),
)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nrt-launchers-')
        self.addCleanup(self.temp.cleanup)
        self.scratch = Path(self.temp.name)
        self.spool = self.scratch / 'spool' / 'slurmd' / 'job62498329'
        self.spool.mkdir(parents=True)
        self.python = self.scratch / 'python-stub'
        self.python.write_text(
            f'#!{sys.executable}\n'
            'import json, os, sys\n'
            'if sys.argv[1:2] == ["-c"]:\n'
            '    if os.environ.get("STUB_PROBE_LOG"):\n'
            '        with open(os.environ["STUB_PROBE_LOG"], "a") as stream: stream.write("probe\\n")\n'
            '    sys.exit(int(os.environ.get("STUB_PROBE_EXIT", "0")))\n'
            'print(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}))\n'
            'sys.exit(int(os.environ.get("STUB_DRIVER_EXIT", "0")))\n'
        )
        self.python.chmod(0o755)
        # Slurm commands are always stubbed; tests never submit or requeue jobs.
        srun = self.scratch / 'srun'
        srun.write_text(
            f'#!{sys.executable}\n'
            'import json, os, sys\n'
            'if os.environ.get("STUB_STEP_LOG"):\n'
            '    with open(os.environ["STUB_STEP_LOG"], "a") as stream: stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
            'args = sys.argv[1:]\n'
            'while args and args[0].startswith("--"): args.pop(0)\n'
            'os.execvp(args[0], args)\n'
        )
        srun.chmod(0o755)
        scontrol = self.scratch / 'scontrol'
        scontrol.write_text(
            f'#!{sys.executable}\n'
            'import json, os, sys\n'
            'with open(os.environ["STUB_REQUEUE_LOG"], "a") as stream: stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
            'sys.exit(int(os.environ.get("STUB_SCONTROL_EXIT", "0")))\n'
        )
        scontrol.chmod(0o755)
        # Avoid loading the cluster's R module during the CPU startup check.
        module = self.scratch / 'module'
        module.write_text('#!/bin/sh\nexit 0\n')
        module.chmod(0o755)
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(('SLURM_', 'NRT_', 'BASH_FUNC_'))
            and key not in ('BASH_ENV', 'ENV', 'DEVICE', 'OUTDIR', 'HYPER_DIR', 'HYPER_FILE', 'SUMMARY_METHODS',
                            'CASES', 'TUNING_TASK_ID', 'EPOCHS', 'TUNING_EVALS')
        }
        self.env.update(PY=str(self.python), PATH=f'{self.scratch}:{os.environ["PATH"]}',
                        NRT_PYTHON_CHECK_DELAY_SECONDS='0',
                        STUB_REQUEUE_LOG=str(self.scratch / 'requeues.jsonl'))

    def launch(self, script, spec, *, root=ROOT, slurm=False, submit_dir=None,
               override=None, task_id=None):
        env = self.env.copy()
        if slurm:
            env['SLURM_JOB_ID'] = '62498329'
        if submit_dir is not None:
            env['SLURM_SUBMIT_DIR'] = str(submit_dir)
        if override is not None:
            env['NRT_ROOT'] = str(override)
        if task_id is not None:
            env['SLURM_ARRAY_TASK_ID'] = str(task_id)
        result = subprocess.run(
            ['bash', str(script), '--dry-run'], cwd=self.spool, env=env,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        for record in records:
            self.assertEqual(record['cwd'], str(root))
            args = record['args']
            self.assertEqual(args[:3], ['-u', '-m', spec[1]])
            is_tuning = spec[1].startswith('tuning.')
            method_flag = '--method' if is_tuning else '--methods'
            self.assertEqual(args[args.index(method_flag) + 1], spec[2])
            if spec[1] == 'tuning.main' and spec[2] == 'DeepSurv':
                self.assertEqual(args[args.index('--epochs') + 1], '1000')
                self.assertEqual(args[args.index('--tuning-evals') + 1], '50')
            elif is_tuning:
                self.assertNotIn('--tuning-evals', args)
            stage = 'tuning' if is_tuning else 'testing'
            self.assertEqual(args[args.index('--output-dir') + 1], str(root / stage / 'output'))
            if stage == 'testing':
                self.assertEqual(args[args.index('--hyper-dir') + 1], str(root / 'testing' / 'hyper_para'))
                self.assertEqual(args[args.index('--tuning-dir') + 1], str(root / 'tuning' / 'output'))
                self.assertEqual(args[args.index('--summary-methods') + 1], 'all')
            self.assertEqual(args[-1], '--dry-run')
        return records

    def copied_script(self, spec, root=ROOT):
        script = self.spool / 'slurm_script'
        shutil.copyfile(root / spec[0], script)
        return script

    def relocated_project(self):
        root = self.scratch / 'project copy with spaces'
        root.mkdir()
        shutil.copyfile(ROOT / 'runtime.sh', root / 'runtime.sh')
        for spec in LAUNCHERS:
            target = root / spec[0]
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / spec[0], target)
        return root

    def test_all_spool_copies_start_from_any_submission_directory(self):
        for spec in LAUNCHERS:
            script = self.copied_script(spec)
            for submit_dir in (ROOT, ROOT / 'tuning', ROOT / 'testing', self.scratch, None):
                with self.subTest(script=spec[0], submit_dir=submit_dir):
                    records = self.launch(script, spec, slurm=True,
                                          submit_dir=submit_dir, task_id=spec[3])
                    self.assertEqual(len(records), 1)

    def test_local_scripts_start_from_an_unrelated_working_directory(self):
        for spec in LAUNCHERS:
            with self.subTest(script=spec[0]):
                records = self.launch(ROOT / spec[0], spec, task_id=spec[3])
                self.assertEqual(len(records), 1)

    def test_local_relocated_scripts_discover_their_project(self):
        root = self.relocated_project()
        for spec in LAUNCHERS:
            with self.subTest(script=spec[0]):
                self.launch(root / spec[0], spec, root=root, task_id=spec[3])

    def test_explicit_project_override_is_used_by_spool_copies(self):
        root = self.relocated_project()
        for spec in LAUNCHERS:
            with self.subTest(script=spec[0]):
                self.launch(self.copied_script(spec), spec, root=root,
                            slurm=True, submit_dir=ROOT, override=root, task_id=spec[3])

    def test_array_boundaries_preserve_task_and_replication_arguments(self):
        for spec, tasks in (
            (LAUNCHERS[0], (34, 35, 69, 70, 104)),
            (LAUNCHERS[1], (14,)),
            (LAUNCHERS[2], (14,)),
            (LAUNCHERS[3], (1, 2)),
            (LAUNCHERS[6], (500,)),
        ):
            for task in tasks:
                expected = spec
                if spec[0] == 'testing/run_nrt.sh':
                    expected = (spec[0], spec[1], ('NRT_NLL', 'NRT_Brier', 'NRT_2Loss')[task], spec[3])
                with self.subTest(script=spec[0], task=task):
                    records = self.launch(self.copied_script(spec), expected,
                                          slurm=True, submit_dir=ROOT, task_id=task)
                    self.assertEqual(len(records), 1)
                    args = records[0]['args']
                    flag = '--task-id' if spec[1].startswith('tuning.') else '--replication'
                    if spec[0] != 'testing/run_nrt.sh':
                        self.assertEqual(args[args.index(flag) + 1], str(task))

    def run_failure(self, **overrides):
        env = {**self.env, 'SLURM_JOB_ID': '1234', 'SLURM_ARRAY_JOB_ID': '1200',
               'SLURM_ARRAY_TASK_ID': '7', **overrides}
        return subprocess.run(['bash', str(ROOT / 'tuning/tune_nncde.sh'), '--dry-run'],
                              cwd=self.spool, env=env, capture_output=True, text=True, timeout=10)

    def requeues(self):
        path = self.scratch / 'requeues.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_environment_failure_retries_then_requeues_only_array_element(self):
        probes = self.scratch / 'probes.txt'
        result = self.run_failure(STUB_PROBE_EXIT='1', STUB_PROBE_LOG=str(probes))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(probes.read_text().splitlines()), 4)
        self.assertEqual(self.requeues(), [['requeue', '1200_7']])
        self.assertEqual(result.stdout, '')

    def test_node_failure_requeues_but_fitting_failure_does_not(self):
        result = self.run_failure(STUB_DRIVER_EXIT='75')
        self.assertEqual(result.returncode, 75)
        self.assertEqual(self.requeues(), [['requeue', '1200_7']])
        (self.scratch / 'requeues.jsonl').unlink()
        result = self.run_failure(STUB_DRIVER_EXIT='1')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.requeues(), [])

    def test_requeue_limit_disabled_requeue_and_rejected_request(self):
        for settings in ({'SLURM_RESTART_COUNT': '4'}, {'NRT_MAX_PYTHON_REQUEUES': '0'}):
            with self.subTest(settings=settings):
                result = self.run_failure(STUB_DRIVER_EXIT='75', **settings)
                self.assertEqual(result.returncode, 75)
                self.assertEqual(self.requeues(), [])
        result = self.run_failure(STUB_DRIVER_EXIT='75', STUB_SCONTROL_EXIT='1')
        self.assertEqual(result.returncode, 75)
        self.assertIn('Slurm rejected', result.stderr)

    def test_single_job_target_and_local_failure(self):
        result = self.run_failure(STUB_DRIVER_EXIT='75', SLURM_ARRAY_JOB_ID='', SLURM_ARRAY_TASK_ID='')
        self.assertEqual(result.returncode, 75)
        self.assertEqual(self.requeues(), [['requeue', '1234']])
        (self.scratch / 'requeues.jsonl').unlink()
        result = self.run_failure(STUB_DRIVER_EXIT='75', SLURM_JOB_ID='')
        self.assertEqual(result.returncode, 75)
        self.assertEqual(self.requeues(), [])

    def test_probes_and_fits_use_the_allocated_slurm_step(self):
        steps = self.scratch / 'steps.jsonl'
        result = self.run_failure(STUB_STEP_LOG=str(steps))
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in steps.read_text().splitlines()]
        self.assertEqual(len(records), 2)
        for args in records:
            self.assertEqual(args[:3], ['--nodes=1', '--ntasks=1', '--cpu-bind=cores'])
        self.assertEqual(records[0][4:6], ['-c', 'import torch, numpy, pandas'])
        self.assertEqual(records[1][4:7], ['-u', '-m', 'tuning.main'])

    def test_all_launchers_allow_requeue_and_append_logs(self):
        for spec in LAUNCHERS:
            text = (ROOT / spec[0]).read_text()
            self.assertIn('#SBATCH --requeue', text)
            self.assertIn('#SBATCH --open-mode=append', text)


if __name__ == '__main__':
    unittest.main()
