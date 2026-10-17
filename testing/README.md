# Benchmark simulation testing

Current NRT simulation is in [regularization/testing](../regularization/testing/README.md).
This directory runs NNCDE, DeepSurv, deep_distribution_regression, ctree, iccforest,
cox, and aft. Benchmark models and retained result values are preserved.

Run from the NRT root after tuning:

```bash
HYPER_DIR="$PWD/tuning/output/hyper_para" sbatch testing/run_nncde.sh
HYPER_DIR="$PWD/tuning/output/hyper_para" sbatch testing/run_deepsurv.sh
TUNING_DIR="$PWD/regularization/tuning/output" sbatch testing/run_cpu.sh
```

Neural methods read parameters and tau from `testing/hyper_para/<method>.csv` by
default. Override with `HYPER_DIR` or `--hyper-dir`; retained CSVs are in
`testing/hyper_para(old)/`. CPU methods read case manifests from `TUNING_DIR/cases/`
(default `tuning/output/cases/`). They do not read neural CSVs. Match n, seed, mu,
r, and tau when choosing manifests. Neural custom generation uses `--mu` and `--r`.

The CPU launcher uses a 500-task array, one replication per task. Each task runs
110 applicable combinations across workers, then 30 forest cases sequentially.
`PARALLEL_FITS` and `ICCFOREST_PARALLEL_FITS` control these pools separately.
Neural launchers use one GPU each and run replications sequentially.

Each fit saves a resumable JSON. Complete cases automatically publish SSE tables
with replication rows, `average`, and sample `standard_deviation`. All benchmark
methods participate in the default `SUMMARY_METHODS=all`; current NRT has separate
tables. For subsets, use the same `SUMMARY_METHODS` in participating launchers.
Combined censoring reports aggregate each method and count shared datasets once.
Incomplete or failed selections preserve complete tables.

```bash
"$PY" -m testing.simulation summarize
"$PY" -m testing.simulation summarize --methods NNCDE,DeepSurv --allow-incomplete
"$PY" -m testing.simulation summarize --methods ctree,iccforest,cox,aft --censoring-only
```

Match output directory, cases, and replication count to the original run.
`main.py` accepts the same CLI. NNCDE defaults to 200 epochs, DeepSurv to 1000;
`--epochs` overrides them. Historical CSV summaries in `output(old)/` retain all
benchmark values, but cannot replace missing per-fit JSON records for resumption.
