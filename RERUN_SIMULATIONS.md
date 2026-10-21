# Fresh 29-case simulation rerun

Run from the NRT root using the pinned Python 3.14.2 / R 4.4.0 environment.
This procedure includes all current benchmark methods, the three default NRT
methods, and the two node-mean-KL variants. Real-world experiments are separate.

Both `multimodal_separate` and `multimodal_closer` exclude current status q=0.2,
current status q=0.5, and interval censoring q=0.5: six exclusions, leaving 29
cases. Case seed indices retain their original values. Default data settings are
n=5000, base seed=123, mu=0, r=None, and 500 simulation replications.

| Stage / method | Cases | Input to simulation |
| --- | --- | --- |
| NNCDE, DeepSurv | 15 each (exact/right) | `benchmarks/tuning/hyper_para/<method>.csv`: tuned parameters and tau |
| deep_distribution_regression | 5 exact | `benchmarks/tuning/cases/`: tau only (null for exact) |
| ctree, cox, aft | 29 each | `benchmarks/tuning/cases/`: tau only |
| iccforest | 24 censored | `benchmarks/tuning/cases/`: tau only |
| Default NRT | 29 per method | `partition_kl/tuning/hyper_para/`: parameters, training settings, epochs and tau |
| Node-mean-KL NRT | 29 per method | `node_mean_kl/tuning/hyper_para/`: parameters, training settings, epochs and tau |

Start in a fresh shell without previous experiment overrides (`CASES`, `METHODS`,
`HYPER_DIR`, `N_SAMPLES`, `EPOCHS`, `TRIALS`, etc.). Choose a new output root; these
commands use scratch storage because a full run produces many cache/result files.
Preserve the output and source files on long-term storage after the run.

```bash
set -euo pipefail
source runtime.sh
"$PY" environment/verify.py --check-source
RUN_ROOT="${SCRATCH:?Set SCRATCH to your experiment storage}/nrt-29cases-$(date +%Y%m%d_%H%M%S)"
mkdir "$RUN_ROOT"
mkdir -p tuning/log testing/log regularization/tuning/log regularization/testing/log
cp environment/fir-20261021/SOURCE_SHA256SUMS "$RUN_ROOT/"
```

First prepare all 29 benchmark case manifests. This CPU job calibrates tau and
records the tuning datasets without fitting models. Neural tuning creates CSVs
for its 15 supported cases; the CPU benchmarks can start as soon as preparation
finishes. The shell substitutions below submit actual Slurm jobs when executed.

```bash
prep_job=$(OUTDIR="$RUN_ROOT/benchmarks/tuning" sbatch --parsable tuning/prepare_cases.sh)
prep_job=${prep_job%%;*}
nncde_job=$(OUTDIR="$RUN_ROOT/benchmarks/tuning" sbatch --parsable \
  --dependency="afterok:$prep_job" tuning/tune_nncde.sh)
nncde_job=${nncde_job%%;*}
deepsurv_job=$(OUTDIR="$RUN_ROOT/benchmarks/tuning" sbatch --parsable \
  --dependency="afterok:$prep_job" tuning/tune_deepsurv.sh)
deepsurv_job=${deepsurv_job%%;*}

TUNING_DIR="$RUN_ROOT/benchmarks/tuning" OUTDIR="$RUN_ROOT/benchmarks/testing" \
  sbatch --dependency="afterok:$prep_job" testing/run_cpu.sh
TUNING_DIR="$RUN_ROOT/benchmarks/tuning" OUTDIR="$RUN_ROOT/benchmarks/testing" \
  sbatch --dependency="afterok:$nncde_job" testing/run_nncde.sh
TUNING_DIR="$RUN_ROOT/benchmarks/tuning" OUTDIR="$RUN_ROOT/benchmarks/testing" \
  sbatch --dependency="afterok:$deepsurv_job" testing/run_deepsurv.sh
```

`TUNING_DIR` selects both `cases/` and `hyper_para/`; CSV copying is unnecessary.
Other benchmarks use only tau from the manifests. Their sample size, seed, mu,
and r come from simulation arguments. For custom mu/r, pass matching `--mu` and
`--r` to preparation, neural tuning, and testing. For custom neural epoch limits,
also pass the same `--epochs` to neural testing; defaults remain 200 for NNCDE
and 1000 for DeepSurv. Keep the default search budgets for a full reproduction.

Submit both NRT penalty experiments into separate directories. Each tuning array
has 29 tasks (0..28). Default testing has three method tasks (0..2); node-mean-KL
testing has two (1..2). Each task runs all retained cases and replications.

```bash
nrt_job=$(OUTDIR="$RUN_ROOT/partition_kl/tuning" sbatch --parsable \
  regularization/tuning/tune_nll.sh)
nrt_job=${nrt_job%%;*}
HYPER_DIR="$RUN_ROOT/partition_kl/tuning/hyper_para" OUTDIR="$RUN_ROOT/partition_kl/testing" \
  sbatch --dependency="afterok:$nrt_job" regularization/testing/run_nll.sh

node_job=$(OUTDIR="$RUN_ROOT/node_mean_kl/tuning" sbatch --parsable \
  regularization/tuning/tune_node_mean_kl.sh)
node_job=${node_job%%;*}
HYPER_DIR="$RUN_ROOT/node_mean_kl/tuning/hyper_para" OUTDIR="$RUN_ROOT/node_mean_kl/testing" \
  sbatch --dependency="afterok:$node_job" regularization/testing/run_node_mean_kl.sh
```

Keep code and environment fixed while jobs are queued or running. The 29-case
revision changes NRT's source fingerprint, so start with fresh outputs. Resume
an interrupted run with its original output directories and identical arguments.
Rebuild complete summaries after all fits finish:

```bash
"$PY" -m testing.simulation summarize \
  --output-dir "$RUN_ROOT/benchmarks/testing"
"$PY" -m regularization.testing.simulation summarize \
  --output-dir "$RUN_ROOT/partition_kl/testing"
"$PY" -m regularization.testing.simulation summarize \
  --methods adaptive_nll_cv,fixed_at_adaptive_architecture \
  --output-dir "$RUN_ROOT/node_mean_kl/testing"
```

Launchers contain Fir-specific GPU resource names and log paths. On a different
server, adjust scheduler resources/log paths and set `NRT_ROOT` / `PY` for that
installation while retaining the pinned environment and experiment arguments.
See [ENVIRONMENT_REPLICATION.md](ENVIRONMENT_REPLICATION.md) for environment
verification and numerical comparisons across servers.
