# Current NRT tuning

From the NRT root, run `python -m regularization.tuning.main` or submit
`sbatch regularization/tuning/tune_nll.sh`. The default GPU array has 29 tasks (0..28),
one per case. Each task fits fixed rho, adaptive KL, and the matched architecture
control. Fixed and adaptive each search 200 candidates: 20 startup and 180 TPE.
The matched control uses CV only to determine its training epoch count.

Data comes from `workflow.generate`, sharing benchmark seeds, tau, and the 80/20
split. Selected parameters, complete training settings, frozen refit epochs, and
tau are exported to `output/hyper_para/<method>.csv` for simulation.

Each method/case retains only `cv_results.csv`, `selection.json`, `result.json`,
and `history.json` (the final refit's history), plus the top-level summaries and
hyperparameter CSVs. No individual trial/fold results or fold histories are saved.
An unfinished search uses one temporary SQLite database to resume completed
trials; it is removed after final outputs are complete. `--summarize` also removes
old fold caches from completed cases while preserving their final outputs.

All CV folds and methods share `U=max(finite L/R endpoints)` computed from the
outer 80% training pool before search. The outer test set and true event times do
not determine the body horizon. The root probability split at U separates the
whole body from the tail, and subsequent splits partition only the body. Old
finite-support and terminal-tail caches/checkpoints are incompatible; use new
output directories when changing the protocol:

```bash
OUTDIR="$PWD/reproduction/tuning" sbatch regularization/tuning/tune_nll.sh
# After tuning succeeds:
HYPER_DIR="$PWD/reproduction/tuning/hyper_para" \
OUTDIR="$PWD/reproduction/testing" sbatch regularization/testing/run_nll.sh
```

See the [workflow guide](../README.md) for the algorithm, defaults, resumption,
and small validation runs.
