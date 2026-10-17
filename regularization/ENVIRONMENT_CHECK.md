# Regularized NLL environment check

Historical check: 2026-10-11, America/New_York. CPU node: `rorqual2`; GPU node: `rg12901`.
The existing Python 3.11 runtime passed CPU and real GPU workflow validation.
Optuna was installed and launcher paths were corrected without rebuilding the environment.
See [the current snapshot](../environment/current/README.md) for the cleanup-time inventory.

## Runtime and original snapshot

Entry point: `/home/y679zhao/envs/nrt-runtime/bin/python`, wrapping
`/home/y679zhao/envs/nrt-python311/bin/python`. Activation uses
`/home/y679zhao/envs/nrt-runtime/activate.sh`, loading StdEnv/2023, python/3.11.5,
r/4.4.0, and cuda/12.6. PyTorch itself was built with CUDA 12.2.

| Component | Checked runtime | Historical environment/ snapshot |
| --- | --- | --- |
| Python | 3.11.5 | 3.14.2 |
| PyTorch | 2.6.0 | 2.11.0 |
| NumPy | 1.26.4 | 2.4.2 |
| SciPy | 1.15.2 | 1.18.1 |
| pandas | 2.2.3 | 3.0.0 |
| CUDA build | 12.2 | 12.9 |
| cuDNN | 90201 | 91301 |
| Optuna | 4.5.0 | Not included |

Distribution metadata for torch, numpy, scipy, pandas, and optuna includes the
`+computecanada` suffix. Optuna and missing Alembic, SQLAlchemy, colorlog, Mako,
greenlet, PyYAML, and tqdm dependencies were installed from the local wheelhouse.
Core packages were not upgraded. Optuna loads from the main environment; `.deps`
is not needed. All eight original `environment/SHA256SUMS` files passed verification.

The original snapshot had 57 Python distributions, versus 43 in the checked runtime:
17 matching versions, 17 changed versions, 23 original packages absent, and 9 additional
packages. This check established that regularization could run; it did not establish
an exact copy of the entire original Python/R environment. Different builds or hardware
may yield different numerical results.

## Completed historical checks

- `python -m pip check` found no broken requirements.
- All nine `regularization.test_workflow` tests passed, without skips, including
  all 35 datasets, 140 shared-support CV folds, likelihoods, KL scaling, actual
  tuning-to-simulation execution, resumption, and configuration validation.
- Each fixed/adaptive optimizer completed 20 startup trials and one TPE trial,
  testing ask/tell, attributes, SQLite, and reload with lightweight substitute scores.
- Both launchers completed a real CPU workflow: `po_interval_q0.2`, n=128,
  2 trials, 2 folds, 3 epochs, warmup=1, depth=2, width=8, grid=8. All methods
  succeeded: 10 CV fits, 3 tuning refits, 6 simulation fits, and a complete case CSV.
- Shell syntax, default and overridden interpreters, and both Slurm arrays passed
  validation, including `sbatch --test-only` for the account and H100 MIG types.

## Historical GPU validation

Job **23015894** ran for 29 seconds on `rg12901`, completed with exit `0:0`, and
used an NVIDIA H100 80GB HBM3 MIG 2g.20gb device (compute capability 9.0).
With the same small configuration and `cuda_graphs=True`, it exercised fused and
capturable Adam, split unfreezing after warmup, and graph capture/replay for two
batch sizes. All 10 CV fits, 3 tuning refits, and 6 simulation fits passed. All nine
final records had `status=ok`, and the summary CSV was published.

Only this short GPU check was submitted during the historical validation. The full
35-case search and 500-replication experiment were not run as part of that check.
The default testing 40GB MIG request passed submission checks, but that device's
full-run time and memory use were not measured.

## Recheck

```bash
cd /home/y679zhao/links/projects/def-anthree/y679zhao/NRT
export PY=/home/y679zhao/envs/nrt-runtime/bin/python
"$PY" -m pip check
"$PY" -m unittest regularization.test_workflow -v
bash regularization/tuning/tune_nll.sh --dry-run
bash regularization/testing/run_nll.sh --dry-run
# Install the added dependency from the Alliance wheelhouse if needed:
"$PY" -m pip install --no-index -r regularization/requirements.txt
```

Historical evidence is under `/home/y679zhao/.cache/regularization-check-20261011-Suf01o/`:
`snapshot-comparison.json`, `python-freeze.txt`, `check_tpe.py`, `optuna.log`,
`optuna/verification.json`, `space.json`, `cpu/`, `gpu-check.sh`, GPU job logs,
`gpu/verification.json`, and `gpu/`. These are separate from production outputs.
