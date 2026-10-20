# Current NRT tuning

For the alternative **twice the mean internal-node KL**, submit
`sbatch regularization/tuning/tune_node_mean_kl.sh` from the NRT root. It tunes
only adaptive and its fixed-at-adaptive-architecture control, writing to
`regularization/tuning/output_node_mean_kl`. The equivalent option is
`--penalty node_mean_kl --methods adaptive_nll_cv,fixed_at_adaptive_architecture`;
the original penalty remains the default for the existing launcher.

From the NRT root, run `python -m regularization.tuning.main` or submit
`sbatch regularization/tuning/tune_nll.sh`. The default GPU array has 35 tasks,
one per case. Each task fits fixed rho, adaptive KL, and the matched architecture
control. Fixed and adaptive each search 200 candidates: 20 startup and 180 TPE.
The matched control uses CV only to determine its training epoch count.

Data comes from `workflow.generate`, sharing benchmark seeds, tau, and the 80/20
split. Selected parameters, complete training settings, frozen refit epochs, and
tau are exported to `output/hyper_para/<method>.csv` for simulation.

All CV folds and methods share `U=1.1*max(finite L/R endpoints)` computed from the
outer 80% training pool before search. The outer test set and true event times do
not determine support. Old per-fold-support caches are incompatible; use new
output directories when changing the protocol:

```bash
OUTDIR="$PWD/reproduction/tuning" sbatch regularization/tuning/tune_nll.sh
# After tuning succeeds:
HYPER_DIR="$PWD/reproduction/tuning/hyper_para" \
OUTDIR="$PWD/reproduction/testing" sbatch regularization/testing/run_nll.sh
```

See the [workflow guide](../README.md) for the algorithm, defaults, resumption,
and small validation runs.
