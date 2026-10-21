# Benchmark simulation testing

Current NRT simulation is in [regularization/testing](../regularization/testing/README.md).
This directory runs NNCDE, DeepSurv, deep_distribution_regression, ctree, iccforest,
cox, and aft. Benchmark models and retained result values are preserved.

Run from the NRT root after tuning:

Both CPU and GPU launchers use the shared [node checks and requeue policy](../README.md#node-checks-and-requeue).
A failed precheck requests requeue of the current job/array element (up to four
times by default) before simulation fits begin.

```bash
sbatch testing/run_nncde.sh
sbatch testing/run_deepsurv.sh
sbatch testing/run_cpu.sh
```

Neural methods read parameters and tau from `TUNING_DIR/hyper_para/<method>.csv`
(default `tuning/output/hyper_para/`). Set `TUNING_DIR` / `--tuning-dir` to select
a different tuning run; `HYPER_DIR` / `--hyper-dir` can override the CSV directory.
Retained CSVs are in `testing/hyper_para(old)/`.

CPU benchmarks read only tau from `TUNING_DIR/cases/<case_id>.json`, validating the
case identity and tau. They never read neural CSVs or inherit saved model parameters,
mu, r, n, or seed. Simulation arguments supply those data settings. First run
`bash tuning/prepare_cases.sh` to prepare all 29 case records; this does not require
any neural tuning. For custom generation, pass matching `--mu` / `--r` to
preparation, neural tuning, and simulation. Defaults are n=5000, base seed=123,
mu=0, and r=None.

The CPU launcher uses a 500-task array, one replication per task. Each task runs
92 applicable combinations across workers, then 24 forest cases sequentially
(116 fits per replication).
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
Use [the full rerun procedure](../RERUN_SIMULATIONS.md) for fresh outputs and job
dependencies. Both multimodal distributions exclude current q=0.2/q=0.5 and
interval q=0.5; shared case indices retain their original random seeds.
