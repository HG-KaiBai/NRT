# Shared NRT dependencies

Only components used by the current regularization and realworld workflows remain:

* `model.py`: prepared raw-time observations.
* `tree.py`: body-tree size constraints and survival integration.
* `tuning_model.py`: architecture configuration and shared/private neural branches.
* `nll_cv_train.py`: optimization configuration, warmup gate, and fixed/adaptive rho.
* `train.py`: training configuration, snapshots, batches, NLL evaluation, and CUDA graphs.
* `metrics.py`: the simulation grid SSE metric.

The body/tail model is in `../model.py`; the sole fitting loop is in `../train.py`.
Unused transformed-time models, density-KL/Brier routines, and duplicate CV/refit
loops have been removed. Historical experiment snapshots retain their own source
files for reproducibility; current source fingerprints include only retained modules.

See [the workflow guide](../README.md) for the terminal-tail parameterization and
`bash regularization/submit.sh` for a fresh Fir tuning/testing run.
