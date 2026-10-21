# Fir runtime validation — 2026-10-20

The NRT launchers now use the existing Fir environment. No packages needed to be
installed or upgraded: all required Python/R imports succeeded and `pip check`
reported no broken requirements. The executable permissions on `~/envs/pt/bin/python`
and `python3` were repaired; launchers use the working versioned executable.

| Component | Verified version |
| --- | --- |
| Python | 3.14.2 (`/home/y679zhao/envs/pt/bin/python3.14`) |
| PyTorch / CUDA build / cuDNN | 2.11.0 / 12.9 / 91301 |
| NumPy / SciPy / pandas | 2.4.2 / 1.18.1 / 3.0.0 |
| Optuna / lifelines / scikit-survival | 4.5.0 / 0.30.0 / 0.28.0 |
| R | 4.4.0 |
| survival / partykit / icenReg / ICcforest | 3.5-8 / 1.3-0 / 2.0.16 / 0.5-1 |

`runtime.sh` loads `python/3.14.2` and `r/4.4.0`, and sets the R user library to
`~/R/x86_64-pc-linux-gnu-library/4.4`. `PY`, `NRT_VENV`, `NRT_PYTHON_MODULE`,
`R_MODULE` and `R_LIBS_USER` can be overridden. Explicit `PY` skips Python module
selection, while R setup still runs for jobs launched by the submission wrapper.
PyTorch provides its CUDA 12.9 runtime; a separate CUDA 12.6 module is not loaded.

The active project path is `/home/y679zhao/projects/def-anthree/y679zhao/NRT`.
All shell launchers were updated from the absent `links/projects/` path. Existing
MIG requests were retained after checking Fir's live scheduler and configuration.
The user's existing realworld search-space edits were preserved.

The Python 3.14 requirements select scikit-survival 0.28.0, which has a
[CPython 3.14 wheel](https://pypi.org/project/scikit-survival/0.28.0/).
The upstream [0.27 release notes](https://scikit-survival.readthedocs.io/en/stable/release_notes/v0.27.html)
already introduced pandas 3 and NumPy 2.4 support. Actual installed dependencies
were checked through imports, package metadata, and end-to-end smoke runs.

## Run

From the NRT root:

```bash
cd /home/y679zhao/projects/def-anthree/y679zhao/NRT
source runtime.sh
"$PY" -m realworld.run preflight --device cpu

# Each wrapper submits GPU/CPU workers and its dependent summary job.
bash realworld/submit_support.sh
bash realworld/submit_metabric.sh

# Testing must wait until every tuning task has succeeded.
tuning_job=$(sbatch --parsable regularization/tuning/tune_node_mean_kl.sh)
tuning_job=${tuning_job%%;*}
sbatch --dependency="afterok:$tuning_job" regularization/testing/run_node_mean_kl.sh
```

These commands run the full experiments. They were not submitted during this check.
Default outputs are `realworld/output/{support,metabric}` and
`regularization/{tuning,testing}/output_node_mean_kl`. The latter uses the adaptive
model and its matched fixed control. Log directories exist before Slurm opens them.
Use new output directories when changing data, code, configuration or environment;
existing result fingerprints remain enforced.

## Validation

- All required Python/R imports and `python -m pip check` passed.
- The selected regression suites contain 98 tests. See `validation.json` for the
  final result. A stale test that required today's source to equal the exact
  historical cleanup snapshot was corrected. Both exact alias reuse and rejection
  of changed-source reuse are still tested; no production fingerprint was aliased.
- SUPPORT and METABRIC each completed all eight models on CPU using `--smoke`,
  including real R tree/forest fits, model saves, predictions and comparison CSVs.
- The actual `tune_node_mean_kl.sh` and `run_node_mean_kl.sh` wrappers completed a
  CPU tuning-to-testing run: one case, 128 samples, two trials, two folds, three
  epochs, two methods and two replications. All four simulation fits succeeded
  and the case summary CSV was generated.
- Shell syntax and launcher spool-copy tests passed. `sbatch --test-only` accepted
  both realworld submission wrappers, its GPU/CPU workers, and both node_mean_kl
  launchers with the configured account/resources.
- A five-minute GPU check was attempted with a 30-second allocation deadline.
  The scheduler reported that the requested nodes were busy, so no GPU training
  ran and no queued job was left behind. CUDA execution and full-run memory/time
  remain unmeasured on Fir; CPU validation does not establish GPU compatibility.

Detailed temporary smoke artifacts are under `/tmp/nrt-environment-check/` on the
login node. They are disposable validation runs, not production results. Compact
validation and environment records are retained in this directory. The previous
Rorqual snapshot under `environment/current/` is retained as historical evidence.

## Storage

The project currently resides on the home filesystem, even though its path contains
`projects`. Live `diskusage_report` showed approximately 9 GiB / 48 GiB and
294,000 / 500,000 files used. `statvfs` showed about 38.7 GiB and 206,095 entries
available at the recorded check. Scratch has a separate 19 TiB / 1,000,000-file
quota; output locations were not moved.

For one run of all requested defaults, budget **150,000 additional filesystem
entries and 10 GiB**. This is a planning allowance, not a measured full run:

- node_mean_kl tuning: 35 cases × 200 trials × four folds × two JSON files =
  56,000 fold/history files, plus about 7,000 trial directories and summaries.
- node_mean_kl testing: 35 cases × two methods × 500 replications = 35,000 result
  JSON files and 35,000 persistent lock files, plus case directories and summaries.
- Both realworld datasets add roughly 7,500–8,000 fold-cache files/directories,
  plus models, predictions, SQLite studies, logs and summaries.
- The allowance includes other manifests and temporary files. Histories contain
  at most 200 epoch records per fold under these defaults. Model sizes vary with
  selected architectures, so the byte allowance should be rechecked for larger runs.

This projects roughly 444,000 / 500,000 entries and 19 / 48 GiB used, leaving
about 56,000 entries and 29 GiB after the allowance. Capacity is sufficient for
one default run, so **no logs, individual results, or summaries were deleted**.
Recheck before concurrent/repeated full runs or increasing trials/replications:

```bash
diskusage_report
df -h .
df -i .
```

Never remove fold caches, SQLite studies, replication JSON or their lock files
while a run is active. Summarization and resumption still depend on these records.
All existing summary files were retained; pre-existing Git deletions were left
as they were at the start of this task.
