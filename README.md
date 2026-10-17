# NRT: regularized likelihood and survival benchmarks

`regularization/` is the current NRT implementation. It compares `fixed_nll_cv`,
`adaptive_nll_cv`, and `fixed_at_adaptive_architecture`. The retired NRT_NLL,
NRT_Brier, and NRT_2Loss implementations and their results have been removed.

Benchmark implementations remain in `tuning/benchmark/` and `testing/benchmark/`.
Their retained results remain in `tuning/output(old)/`, `testing/output(old)/`, and
`testing/hyper_para(old)/`. Mixed historical tables retain their benchmark columns.
Current regularization results, selected hyperparameters, studies, and caches stay
in `regularization/tuning/output/` and `regularization/testing/output/`.

## Reproduce the current NRT workflow

Run commands from this repository's root. The tested cluster interpreter is:

```bash
export PY=/home/y679zhao/envs/nrt-runtime/bin/python
"$PY" -m pip check
"$PY" -m unittest regularization.test_workflow test_workflow test_deepsurv test_preflight test_cpu_scheduling -v
bash regularization/tuning/tune_nll.sh --dry-run
bash regularization/testing/run_nll.sh --dry-run
```

See [the current environment snapshot](environment/current/README.md) for exact
installed package versions and recreation commands. The older files immediately
under `environment/` describe a different Python 3.14 environment; they are retained
as historical benchmark provenance. Do not use that snapshot as the current NRT runtime.

The [regularization guide](regularization/README.md) defines the model, support,
search space, seeds, training protocol, and output schema. Defaults are 35 cases,
5000 observations, 200 trials per searched method, four CV folds, and 500 simulation
replications. Tuning uses `123 + case.index*10000`; simulation uses
`123 + 1000000 + case.index*10000 + replication`. Case indices are stable, including
historical gaps. Every method shares its data and 80/20 split. Testing reads the
saved tau, model parameters, optimization settings, and fixed refit epochs.

Resume the saved experiment with unchanged options:

```bash
sbatch regularization/tuning/tune_nll.sh
# Submit testing after all tuning tasks succeed, replacing TUNING_JOB_ID:
sbatch --dependency=afterok:TUNING_JOB_ID regularization/testing/run_nll.sh
```

For an independent full rerun, choose unused output directories for both stages:

```bash
OUTDIR="$PWD/reproduction/tuning" sbatch regularization/tuning/tune_nll.sh
HYPER_DIR="$PWD/reproduction/tuning/hyper_para" OUTDIR="$PWD/reproduction/testing" \
  sbatch --dependency=afterok:TUNING_JOB_ID regularization/testing/run_nll.sh
```

Do not change the search budget or training settings when reproducing full results.
A small executable validation is available separately:

```bash
"$PY" reproduce.py smoke --output-dir /tmp/nrt-smoke-new
```

It uses small architectures and data to check tuning, CSV export, simulation, and
summary generation. It does not reproduce the numerical values of the full experiment.

Rebuild current summaries without training:

```bash
"$PY" -m regularization.tuning.main --summarize
"$PY" -m regularization.testing.simulation summarize
```

Successful results are reused only when fingerprints match. The exact source
fingerprint compatibility entry in `environment/source-compatibility.json` records
this cleanup: shared data generation, numerical models, training, and tuning logic
are unchanged. It preserves existing result fingerprints only for the reviewed
source snapshot. Any subsequent source change produces a new fingerprint.
Reproducibility across different hardware or package builds is not guaranteed.

## Reproduce benchmarks

| Method | Supported observations | Cases |
| --- | --- | --- |
| NNCDE, DeepSurv | exact, right | 15 each |
| deep_distribution_regression | exact | 5 |
| ctree, cox, aft | exact, right, current, interval | 35 each |
| iccforest | right, current, interval | 30 |

Neural tuning and testing remain separate from current NRT:

```bash
sbatch tuning/tune_nncde.sh
sbatch tuning/tune_deepsurv.sh
# After benchmark tuning finishes:
HYPER_DIR="$PWD/tuning/output/hyper_para" sbatch testing/run_nncde.sh
HYPER_DIR="$PWD/tuning/output/hyper_para" sbatch testing/run_deepsurv.sh
# Current NRT manifests supply all 35 shared data cases for CPU methods:
TUNING_DIR="$PWD/regularization/tuning/output" sbatch testing/run_cpu.sh
```

To use retained benchmark CSVs, set `HYPER_DIR="$PWD/testing/hyper_para(old)"`.
Historical benchmark outputs are summary CSVs; missing raw JSON, model weights,
or original environment details cannot be reconstructed from those summaries.
A new fit can rerun the protocol; summary CSVs alone cannot resume individual fits.
Use the current NRT case manifests only when their n, seed, mu, r, and tau match the
benchmark protocol. Pass matching `--mu` and `--r` for custom neural experiments.

See [benchmark tuning](tuning/README.md), [benchmark testing](testing/README.md),
and their backend READMEs for package requirements and controls. R methods require
Rscript and the relevant survival, icenReg, partykit, and ICcforest packages.

## Node checks and requeue

Launchers use `runtime.sh`; set `PY` to override the default interpreter and
`NRT_ROOT` when submitting a relocated project. Device and dependency checks run
before fits. Exit 75 identifies startup failures; Slurm jobs may requeue up to
`NRT_MAX_PYTHON_REQUEUES` (default 4). Fitting errors do not trigger that startup retry.
Lock files coordinate concurrent writers and may be recreated during a run. Remove
stale locks only after all writers have stopped. Python bytecode and locks are ignored by Git.
