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
source runtime.sh
"$PY" -m pip check
"$PY" -m unittest regularization.test_workflow test_workflow test_deepsurv test_preflight test_cpu_scheduling -v
bash regularization/tuning/tune_nll.sh --dry-run
bash regularization/testing/run_nll.sh --dry-run
```

The required environment is the **current Fir server snapshot from 2026-10-21**:
Python 3.14.2 in `~/envs/pt`, R 4.4.0, and all exact package/build versions in
[environment/fir-20261021](environment/fir-20261021/README.md). Installers for every
workflow refer to the same root [requirements.txt](requirements.txt); alternate
Python versions and silently substituted package builds are not reproduction targets.
Use [ENVIRONMENT_REPLICATION.md](ENVIRONMENT_REPLICATION.md) for both Alliance and
ordinary Linux servers, offline artifacts, R restoration, and numerical validation.
Verify a copied environment with `"$PY" environment/verify.py --check-source` after
loading the runtime. The files under `environment/current/` are historical Rorqual
Python 3.11 records; [the 2026-10-20 check](environment/fir-20261020/README.md)
retains earlier validation and storage estimates.

The [regularization guide](regularization/README.md) defines the model, support,
search space, seeds, training protocol, and output schema. Defaults are 29 cases,
5000 observations, 200 trials per searched method, four CV folds, and 500 simulation
replications. Tuning uses `123 + case.index*10000`; simulation uses
`123 + 1000000 + case.index*10000 + replication`. Case indices are stable, including
historical gaps. Every method shares its data and 80/20 split. Testing reads the
saved tau, model parameters, optimization settings, and fixed refit epochs.

For both `multimodal_separate` and `multimodal_closer`, current status at q=0.2
and q=0.5 and interval censoring at q=0.5 are excluded: six cases in total.
Their interval q=0.2, exact, and right-censoring cases remain. All current methods
use this shared selection; the original seed indices are preserved.
Use [RERUN_SIMULATIONS.md](RERUN_SIMULATIONS.md) to rerun all simulation methods
in fresh output directories with the correct job dependencies.

Resume an interrupted 29-case experiment with unchanged source and options:

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

Successful results are reused only when fingerprints match. The compatibility
entry in `environment/source-compatibility.json` applies only to the historical
cleanup snapshot. The 29-case revision has a new source fingerprint; use fresh
output directories for the requested rerun.
Reproducibility across different hardware or package builds is not guaranteed.

## Reproduce benchmarks

| Method | Supported observations | Cases |
| --- | --- | --- |
| NNCDE, DeepSurv | exact, right | 15 each |
| deep_distribution_regression | exact | 5 |
| ctree, cox, aft | exact, right, current, interval | 29 each |
| iccforest | right, current, interval | 24 |

Neural tuning and testing remain separate from current NRT:

```bash
# Prepare all 29 case manifests and tau values, without fitting models:
bash tuning/prepare_cases.sh
# Tune each neural method, then run its testing job after that tuning array succeeds:
nncde_job=$(sbatch --parsable tuning/tune_nncde.sh)
deepsurv_job=$(sbatch --parsable tuning/tune_deepsurv.sh)
sbatch --dependency="afterok:${nncde_job%%;*}" testing/run_nncde.sh
sbatch --dependency="afterok:${deepsurv_job%%;*}" testing/run_deepsurv.sh
# Other benchmarks need only the prepared cases:
sbatch testing/run_cpu.sh
```

To use retained benchmark CSVs, set `HYPER_DIR="$PWD/testing/hyper_para(old)"`.
Historical benchmark outputs are summary CSVs; missing raw JSON, model weights,
or original environment details cannot be reconstructed from those summaries.
A new fit can rerun the protocol; summary CSVs alone cannot resume individual fits.
Neural testing defaults to `tuning/output/hyper_para`. Set `TUNING_DIR` to use a
different tuning output root; its `hyper_para/` and `cases/` are selected together.
Other benchmarks read only tau from `cases/<case_id>.json`; n, seed, mu, and r come
from simulation arguments. Pass the same custom `--mu` / `--r` during preparation,
tuning, and simulation so the saved tau matches the generation protocol.

See [benchmark tuning](tuning/README.md), [benchmark testing](testing/README.md),
and their backend READMEs for package requirements and controls. R methods require
Rscript and the relevant survival, icenReg, partykit, and ICcforest packages.

## Node checks and requeue

Launchers use `runtime.sh`; set `PY` to override the default interpreter and
`NRT_ROOT` when submitting a relocated project. All current realworld,
regularization, benchmark tuning/testing, preparation, and summary launchers use
`node_runner.py` through `nrt_python`. Before work begins, it checks temporary
storage, CPU tensor/BLAS calculations, selected Python/R backends, and the requested
GPU's allocation, computation, and synchronization. GPU jobs also check the CPU.
Preparation and summary commands need only CPU checks; previews skip them.

Python/R module-load failures, failed Python import probes, and node precheck
failures return 75. `runtime.sh` requests `scontrol requeue` for the current job
or just its failed array element, up to `NRT_MAX_PYTHON_REQUEUES` (default 4), using
`SLURM_RESTART_COUNT`. Logs append across restarts. Local runs exit without a Slurm
request, and a rejected requeue request is reported in stderr. Known propagated
CUDA availability errors during execution also use 75 via `cuda_retry.py`.
Model/data errors and out-of-memory errors retain their normal failure status.

These launcher changes leave scientific source fingerprints unchanged. Existing
completed fits remain reusable under their original experiment settings. All
batch scripts include `--requeue`; placement on a different node is decided by
Slurm. To use the same checks in a manual command, load `runtime.sh` and run, for
example, `nrt_python realworld.run preflight --device cpu`.
Lock files coordinate concurrent writers and may be recreated during a run. Remove
stale locks only after all writers have stopped. Python bytecode and locks are ignored by Git.
