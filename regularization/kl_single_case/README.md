# Historical single-case finite-support KL experiment

This notebook is retained as an exploratory record. The current reproducible NRT
workflow is [regularization](../README.md); its entry points do not require the
external `pycode`, `neural_only`, or `finite_support_kl` packages used here.

In the original source project, open `model_kl.ipynb` and execute cells in order.
It fits observed NLL plus leaf-width KL, retaining `model`, `history`, `results_df`,
`estimated`, and `grid_sse` for inspection.

The tree uses raw time with uniform density in each leaf:

```text
max_observation = max(finite L/R endpoints in the fitting subset)
upper_bound = 1.1 * max_observation
R = -2/(depth * L) * sum(log(width / (upper_bound/L)))
loss = mean(observed NLL) + (a/sqrt(n_fit)) * mean(R)
L = 2**depth
```

Minimum leaf width is `upper_bound * 0.0008/L`. There is no nonlinear time transform.
Support excludes test observations and latent event times. Impossible observations
have zero likelihood and infinite NLL; events are not clipped and support is not
expanded using test data.

The original generator is `pycode.neural_only.testdata2`, with seed=123, n=5000,
6 covariates, PH/right/q=0.5, and the first 80% for training without reshuffling.
SSE uses 256 raw-time points from zero to the maximum training endpoint, summing
squared errors over the grid and then averaging subjects.

`LOAD_SAVED_SELECTION=True` reads the frozen `raw_time_finite_kl` architecture,
strength, and epoch count from the original project's
`../finite_support_kl/results/finite_support_normalization_comparison/selected_parameters.csv`.
The PH/right/q=0.5 refit uses 35 epochs. Pass a different raw-time selection path
to `load_selected` if needed. If X_DIM changes, retune or disable loading and set
architecture and epochs beforehand. Training uses Adam, warmup, and the split
learning rate selected by NLL-only CV. Seed 123 was used for HPO, so this notebook
is illustrative rather than an independent evaluation.

Default outputs are under `results/ph_right_q0.5_seed123_raw_time_finite_v1/`.
They include model weights and configuration, metrics, histories, subject-level
SSE, raw data, predictions, truth, and figures. Multimodal plots use raw-time density.

On 2026-10-11, saved artifacts were cleaned up and notebook outputs cleared. Only
CSV summaries were retained, including `results/summary.csv` for the two runs.
Re-execution in the original source project recreates training artifacts. Current
NRT reproduction commands are provided in the parent guide.
