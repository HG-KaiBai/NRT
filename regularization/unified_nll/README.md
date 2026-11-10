# Shared NRT model dependencies

This directory retains the modules imported by the current partition-KL and
realworld workflows: `model.py`, `tree.py`, `train.py`, `regularization.py`,
`tuning_model.py`, `cv_train.py`, `nll_cv_train.py`, `brier.py`, and `metrics.py`,
plus the package initializer.

These files retain their original contents because they define shared model
classes, training helpers, and the simulation source fingerprint. `brier.py`
remains an import dependency of `cv_train.py`.

Legacy experiment drivers, searches, reports, notebooks, raw-time variants,
local dependency overlays, and their separate results have been removed.
Use [the current workflow](../README.md) for the retained implementation and
[the rerun guide](../../RERUN_SIMULATIONS.md) for all 29 simulation cases.
