# Benchmark tuning

Current NRT tuning is in [regularization/tuning](../regularization/tuning/README.md).
This directory tunes NNCDE and DeepSurv, each on 15 exact/right-censored cases.
Both use an 80/20 split and four CV folds within the training pool.
NNCDE uses the full `NNCDE_GRID` and validation NLL. DeepSurv uses Sobol search and
validation concordance, with 50 candidate evaluations by default.

Run from the NRT root:

All launchers use the shared CPU/GPU [node checks and requeue policy](../README.md#node-checks-and-requeue).
Preparation checks CPU execution and temporary storage; neural jobs also check
the requested GPU and selected backend before any fitting.

```bash
bash tuning/prepare_cases.sh
sbatch tuning/tune_nncde.sh
sbatch tuning/tune_deepsurv.sh
"$PY" -m tuning.main --method NNCDE --list-cases
"$PY" -m tuning.main --method NNCDE --task-id 0 --dry-run
"$PY" -m tuning.main --method NNCDE --summarize
"$PY" -m tuning.main --method DeepSurv --summarize
```

`prepare_cases.sh` creates all 29 retained `output/cases/<case_id>.json` records,
including tau, without fitting models. Neural tuning covers only the 15 exact/right
cases, so this preparation supplies the additional cases needed by CPU benchmarks.
It reuses identical manifests and rejects conflicting settings. For custom runs,
set `OUTDIR`, `N_SAMPLES`, and `BASE_SEED` consistently and pass matching `--mu` /
`--r` to preparation and tuning. To submit preparation to Slurm, use
`sbatch tuning/prepare_cases.sh` and make neural tuning depend on its success.

Array IDs 0..14 retain the complete supported-case order even when filtered.
NNCDE defaults to 200 epochs; DeepSurv defaults to 1000. Override with `EPOCHS`
or `--epochs`. `TUNING_EVALS` controls DeepSurv's search budget; NNCDE evaluates
all listed combinations. Use `--search-space file.json` to override the NNCDE grid.
DeepSurv ranges are in `benchmark/deepsurv/cuda_model.py`; keep its testing copy
synchronized. Requests validate the saved ranges before reuse.

Each worker saves a result JSON, CV records, and a case manifest with tau. Completed
method selections export `output/hyper_para/<method>.csv` and combined metric tables.
Benchmark testing reads these CSVs directly by default; no copying is required.
Its `TUNING_DIR` must match this stage's `OUTDIR` for custom output directories.
Pass matching `--output-dir` and `--cases` when rebuilding summaries. Incomplete
selections preserve existing tables. Summary-only runs do not require CUDA.

Reference tuning results remain in
`nrt-29cases-20261021_105101/benchmarks/tuning/` at the project root.
Model implementations and their README files remain in `benchmark/`.
See [the full rerun procedure](../RERUN_SIMULATIONS.md) for fresh directories and
Slurm dependencies for every simulation method.
