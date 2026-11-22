# Reproducing the 29-case reference experiment

This directory defines the reference experiment
`nrt-29cases-20261113_223129`. It contains the frozen implementation, environment
inventories, selected parameters, and results described in the
[repository overview](../README.md).

The protocol uses 29 cases, 5,000 observations per dataset, an 80/20 train/test
split, four-fold cross-validation, and 500 replications per supported method/case.
There are 73,000 benchmark fits and 43,500 NRT fits in the testing stage.

## Reference artifacts

| Path | Role |
| --- | --- |
| `source/`, `SOURCE_SHA256SUMS` | The 86 original frozen files and their hashes |
| `source/environment/fir-20261021/` | Pinned Python/R packages and native-runtime inventory |
| `environment/`, `validate_environment.py` | Environment restoration, capture, and verification |
| `launchers/` | Eight reference Slurm launchers |
| `benchmarks/tuning/` | Case manifests, neural CV results, selected parameters, and hyperparameter CSVs |
| `partition_kl/tuning/` | NRT case manifests, CV histories, selections, and hyperparameter CSVs |
| `benchmarks/testing/`, `partition_kl/testing/` | Replication records and result tables |
| `submission.json` | Original configuration, stage dependencies, and job provenance |
| `preflight_configuration.json` | Case matrix, search settings, seeds, and original preview checks |
| `environment_validation.json` | Original environment validation and additional unused R packages |

Raw replication JSON files are needed to regenerate summary tables. Saved CSVs
alone cannot reconstruct those records. The original job IDs in
`submission.json` are provenance; each independent run needs new jobs.

## Environment setup

Follow [ENVIRONMENT_REPLICATION.md](ENVIRONMENT_REPLICATION.md) for the exact
Python 3.14.2 / R 4.4.0 environment. From a fresh Bash shell **in this directory**:

```bash
set -euo pipefail
export NRT_REFERENCE="$PWD"
export NRT_ROOT="$NRT_REFERENCE/source"
export PYTHONDONTWRITEBYTECODE=1
source "$NRT_ROOT/runtime.sh"
export PY
sha256sum --quiet -c "$NRT_REFERENCE/SOURCE_SHA256SUMS"
"$PY" "$NRT_REFERENCE/validate_environment.py"
"$PY" -m unittest test_preflight test_node_runner
```

Set `NRT_VENV` or `PY` before sourcing the runtime if the pinned installation is
elsewhere. The runtime changes the working directory to `source/`. The validator
refreshes its environment reports. Use this experiment's top-level checksum
manifest to verify its source; the older manifests inside the environment
snapshot describe the historical environment capture.

## Prepare a new run

Run this preparation once after the environment setup, then choose either the
full reproduction or saved-parameter refit below. The examples use Slurm and the
original Fir resource settings.

Choose a new destination:

```bash
export NRT_REPRO_OUTPUT="$HOME/nrt-runs/nrt-29cases-$(date +%Y%m%d_%H%M%S)"
test ! -e "$NRT_REPRO_OUTPUT"
mkdir -p "$NRT_REPRO_OUTPUT/log" "$NRT_REPRO_OUTPUT/launchers"

# Remove experiment overrides from an earlier shell session.
unset OUTDIR HYPER_DIR TUNING_DIR EPOCHS TRIALS INITIAL_TRIALS DEVICE
unset TUNING_EVALS COMMAND TUNING_TASK_ID NLL_TASK_ID
unset PARALLEL_FITS ICCFOREST_PARALLEL_FITS PYTHONPATH
export CASES=all METHODS=all PENALTY=partition_kl SUMMARY_METHODS=all
export N_SAMPLES=5000 BASE_SEED=123 REPLICATIONS=500 GRID_SIZE=256
export NRT_LAUNCHERS="$NRT_REPRO_OUTPUT/launchers"
```

The reference launchers contain the original absolute source and checksum paths.
Create copies that point to the reference directory in this checkout:

```bash
"$PY" - <<'PY'
import json
import os
from pathlib import Path

reference = Path(os.environ["NRT_REFERENCE"]).resolve()
destination = Path(os.environ["NRT_LAUNCHERS"])
original = json.loads((reference / "submission.json").read_text())["run_root"]
for launcher in sorted((reference / "launchers").glob("*.sh")):
    text = launcher.read_text().replace(original, str(reference))
    (destination / launcher.name).write_text(text)
PY
```

For another cluster, adapt the copies' `#SBATCH --account`, GPU resource names,
time limits, and memory requests to the destination. Keep the scientific
settings and the files under `source/` unchanged. The submission helper below
sets the working directory and log locations explicitly:

```bash
submit_stage() {
  sbatch --parsable --export=ALL --chdir="$NRT_ROOT" \
    --output="$NRT_REPRO_OUTPUT/log/%x-%A_%a.out" \
    --error="$NRT_REPRO_OUTPUT/log/%x-%A_%a.err" "$@"
}
```

Preview dimensions and case selections before submitting:

```bash
bash "$NRT_LAUNCHERS/prepare_cases.sh" --dry-run
bash "$NRT_LAUNCHERS/tune_nncde.sh" --dry-run
bash "$NRT_LAUNCHERS/tune_deepsurv.sh" --dry-run
bash "$NRT_LAUNCHERS/test_cpu.sh" --dry-run
bash "$NRT_LAUNCHERS/test_nncde.sh" --dry-run
bash "$NRT_LAUNCHERS/test_deepsurv.sh" --dry-run
bash "$NRT_LAUNCHERS/tune_nrt.sh" --dry-run
bash "$NRT_LAUNCHERS/test_nrt.sh" --dry-run
```

The previews do not submit jobs or train models. The following two sections are
alternative execution paths for the newly prepared output directory.

## Full reproduction

These commands submit all preparation, tuning, and testing jobs. Dependencies
ensure that each testing stage waits for its required inputs.

```bash
prep_job=$(OUTDIR="$NRT_REPRO_OUTPUT/benchmarks/tuning" \
  submit_stage "$NRT_LAUNCHERS/prepare_cases.sh")
prep_job=${prep_job%%;*}

nncde_job=$(OUTDIR="$NRT_REPRO_OUTPUT/benchmarks/tuning" \
  submit_stage --dependency="afterok:$prep_job" "$NRT_LAUNCHERS/tune_nncde.sh")
nncde_job=${nncde_job%%;*}

deepsurv_job=$(OUTDIR="$NRT_REPRO_OUTPUT/benchmarks/tuning" \
  submit_stage --dependency="afterok:$prep_job" "$NRT_LAUNCHERS/tune_deepsurv.sh")
deepsurv_job=${deepsurv_job%%;*}

TUNING_DIR="$NRT_REPRO_OUTPUT/benchmarks/tuning" OUTDIR="$NRT_REPRO_OUTPUT/benchmarks/testing" \
  submit_stage --dependency="afterok:$prep_job" "$NRT_LAUNCHERS/test_cpu.sh"
TUNING_DIR="$NRT_REPRO_OUTPUT/benchmarks/tuning" OUTDIR="$NRT_REPRO_OUTPUT/benchmarks/testing" \
  submit_stage --dependency="afterok:$nncde_job" "$NRT_LAUNCHERS/test_nncde.sh"
TUNING_DIR="$NRT_REPRO_OUTPUT/benchmarks/tuning" OUTDIR="$NRT_REPRO_OUTPUT/benchmarks/testing" \
  submit_stage --dependency="afterok:$deepsurv_job" "$NRT_LAUNCHERS/test_deepsurv.sh"

nrt_job=$(OUTDIR="$NRT_REPRO_OUTPUT/partition_kl/tuning" \
  submit_stage "$NRT_LAUNCHERS/tune_nrt.sh")
nrt_job=${nrt_job%%;*}

HYPER_DIR="$NRT_REPRO_OUTPUT/partition_kl/tuning/hyper_para" OUTDIR="$NRT_REPRO_OUTPUT/partition_kl/testing" \
  submit_stage --dependency="afterok:$nrt_job" "$NRT_LAUNCHERS/test_nrt.sh"
```

The reference launchers provide these task layouts:

| Stage | Tasks |
| --- | --- |
| Prepare benchmark cases | One CPU job covering all 29 cases |
| Tune NNCDE / DeepSurv | A 15-task array per method |
| Test CPU benchmarks | A 500-task array, one replication per task |
| Test NNCDE / DeepSurv | One GPU job per method, covering 15 cases × 500 replications |
| Tune NRT | A 29-task array, one case per task |
| Test NRT | A three-task array, one NRT variant per task |

NRT searches use 200 trials per searched method/case, 20 initial trials, four CV
folds, and a 200-epoch limit. The matched control uses the adaptive architecture
without a separate search. NNCDE uses a 27-candidate grid and 200 epochs;
DeepSurv uses 50 evaluations and 1,000 epochs. NRT refitting uses the epoch counts
saved by tuning.

## Refit using saved parameters

This alternative skips preparation and model selection while repeating all
testing fits with the reference parameters. Complete **Prepare a new run** above
using a separate fresh destination, then submit:

```bash
TUNING_DIR="$NRT_REFERENCE/benchmarks/tuning" OUTDIR="$NRT_REPRO_OUTPUT/benchmarks/testing" \
  submit_stage "$NRT_LAUNCHERS/test_cpu.sh"
TUNING_DIR="$NRT_REFERENCE/benchmarks/tuning" OUTDIR="$NRT_REPRO_OUTPUT/benchmarks/testing" \
  submit_stage "$NRT_LAUNCHERS/test_nncde.sh"
TUNING_DIR="$NRT_REFERENCE/benchmarks/tuning" OUTDIR="$NRT_REPRO_OUTPUT/benchmarks/testing" \
  submit_stage "$NRT_LAUNCHERS/test_deepsurv.sh"
HYPER_DIR="$NRT_REFERENCE/partition_kl/tuning/hyper_para" OUTDIR="$NRT_REPRO_OUTPUT/partition_kl/testing" \
  submit_stage "$NRT_LAUNCHERS/test_nrt.sh"
```

NNCDE and DeepSurv read `benchmarks/tuning/hyper_para/*.csv`. CPU benchmarks read
the saved censoring horizons from `benchmarks/tuning/cases/*.json`. NRT reads
parameters, optimization settings, refit epochs, and censoring horizons from
`partition_kl/tuning/hyper_para/*.csv`.

Both paths retain n=5,000, base seed=123, mu=0, r=None, grid size=256, and 500
replications. The tuning seed is `123 + case.index * 10000`; the simulation seed
is `123 + 1000000 + case.index * 10000 + replication`. The case indices are
defined in `source/workflow.py` and include gaps for excluded cases.

## Summarize and compare

Once all testing jobs have finished:

```bash
cd "$NRT_ROOT"
"$PY" -m testing.simulation summarize \
  --output-dir "$NRT_REPRO_OUTPUT/benchmarks/testing" --replications 500
"$PY" -m regularization.testing.simulation summarize \
  --output-dir "$NRT_REPRO_OUTPUT/partition_kl/testing" --replications 500
```

To rebuild the saved reference tables without fitting models, use the same
commands with `NRT_REPRO_OUTPUT="$NRT_REFERENCE"`.

| Output directory | Metric | Tables |
| --- | --- | ---: |
| `benchmarks/testing/tables/` | Average grid SSE | 29 |
| `partition_kl/testing/tables/` | Average grid SSE | 29 |
| `partition_kl/testing/tables_tau_absolute_bias/` | Absolute survival error at the training horizon | 29 |

Each table contains 500 replication rows plus the mean and sample standard
deviation. Average grid SSE averages over test subjects and sums over the 256
time points. Tail bias averages the absolute difference between estimated and
true survival probabilities at the maximum finite observed training endpoint.

Compare individual replication metrics and aggregate tables with the reference.
Incomplete or failed replications cause a nonzero exit status; inspect the
corresponding JSON and job log before summarizing again. Reusing an interrupted
run requires the same source, parameters, and execution settings so that cached
results pass the workflow's compatibility checks. Cross-hardware numerical
agreement must be assessed from the outputs even when source and package
inventories match.
