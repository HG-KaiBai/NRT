# Benchmark tuning

Current NRT tuning is in [regularization/tuning](../regularization/tuning/README.md).
This directory tunes NNCDE and DeepSurv, each on 15 exact/right-censored cases.
Both use an 80/20 split and four CV folds within the training pool.
NNCDE uses the full `NNCDE_GRID` and validation NLL. DeepSurv uses Sobol search and
validation concordance, with 50 candidate evaluations by default.

Run from the NRT root:

```bash
sbatch tuning/tune_nncde.sh
sbatch tuning/tune_deepsurv.sh
"$PY" -m tuning.main --method NNCDE --list-cases
"$PY" -m tuning.main --method NNCDE --task-id 0 --dry-run
"$PY" -m tuning.main --method NNCDE --summarize
"$PY" -m tuning.main --method DeepSurv --summarize
```

Array IDs 0..14 retain the complete supported-case order even when filtered.
NNCDE defaults to 200 epochs; DeepSurv defaults to 1000. Override with `EPOCHS`
or `--epochs`. `TUNING_EVALS` controls DeepSurv's search budget; NNCDE evaluates
all listed combinations. Use `--search-space file.json` to override the NNCDE grid.
DeepSurv ranges are in `benchmark/deepsurv/cuda_model.py`; keep its testing copy
synchronized. Requests validate the saved ranges before reuse.

Each worker saves a result JSON, CV records, and a case manifest with tau. Completed
method selections export `output/hyper_para/<method>.csv` and combined metric tables.
Pass matching `--output-dir` and `--cases` when rebuilding summaries. Incomplete
selections preserve existing tables. Summary-only runs do not require CUDA.

Historical benchmark CSVs remain in `output(old)/`; they are not raw resumable
results. Model implementations and their README files remain in `benchmark/`.
