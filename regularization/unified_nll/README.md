# Current KL: finite support in original time

The current main notebook `tune_nll_cv_rho.ipynb`, server notebook
`tuning/tune_kl_server.ipynb` and single-case notebook
`../kl_single_case/model_kl.ipynb` use raw time on `[0,U]`, where
`U = 1.1 * max(finite observed L/R endpoints in the actual fit subset)`.
Each CV fold estimates its own training-only U; final refits use the first80% pool.
There is no nonlinear time transformation. Equal-width target is U/L, with
`R = -2/(depth*L) * sum log(width/(U/L))` and `lambda = a/sqrt(n_fit)`.
The current implementation is in `raw_time/`; the server guide explains fresh
output paths, cache compatibility and tests. The single case loads the frozen
`raw_time_finite_kl` selection from the completed finite-support comparison.

The completed experiment below used the old infinite-support transform.
Its numerical source files and aggregate CSV tables are retained; its results
do not describe the new default. On 2026-10-11, individual fit artifacts,
logs and saved notebook outputs were deleted at the user's request.
The historical notebook source is archived at
`archive/infinite_support_20261011/tune_nll_cv_rho.ipynb`. The current main
notebook is reset for a fresh raw-time study in `results/nll_cv_raw_time_comparison`.

# Historical retained experiment: infinite-support NLL CV with adaptive rho

Only the KL penalty experiment is retained after the 2026-10-11 cleanup.
The separate single-case notebook is
[`../kl_single_case/model_kl.ipynb`](../kl_single_case/model_kl.ipynb), using
`neural_only.testdata2` directly. For a fresh server study, edit
[`tuning/tune_kl_server.ipynb`](tuning/tune_kl_server.ipynb) and follow the
Chinese [server tuning guide](tuning/README.md). Fresh tuning checks that
the frozen simulator and neural_only generator still have identical logic.
Completed numerical source files remain unchanged; only CSV result summaries remain.

This independent implementation preserves the completed experiment in
`results/nll_cv_rho_comparison/`. Its main notebook is
[`archive/infinite_support_20261011/tune_nll_cv_rho.ipynb`](archive/infinite_support_20261011/tune_nll_cv_rho.ipynb).
All 9 code cells had completed without errors before the saved outputs were cleared. Historical experiment outputs and obsolete notebook entry
points were removed at the user's request on 2026-10-10. The `neural_only`
directory and unrelated workspace changes were preserved.

## Results

Each data seed first averages grid SSE over 35 settings; the table reports
mean and sample SD across seeds 0..9.

| Method | Mean grid SSE | Sample SD |
| --- | ---: | ---: |
| fixed_nll_cv | 0.371505 | 0.041046 |
| adaptive_nll_cv | 0.351877 | 0.030945 |
| fixed_at_adaptive_architecture | 0.389907 | 0.037191 |

Adaptive lowers aggregate SSE by 5.28% against independently tuned fixed rho
and 9.75% against the matched-architecture fixed control. It improves 22/35
settings and worsens 13/35 in each comparison; the winning sets need not
coincide. The matched models have their own CV-derived training durations
and different trainable parameter counts. This is not uniform superiority
or a strict one-factor causal comparison.

Completed budget: 70 independent HPO studies, 4,480 trials, 18,060 CV fits
including the matched controls, and 1,050 final fits. Historical validation
recorded 29 passing tests before the logs were deleted.

## Common loss and geometry penalty

The observed likelihood is density for exact events and interval probability
for censored observations; right censoring uses an infinite upper bound,
and current status is encoded as (0,C] or (C,infinity). Integration uses
interval overlaps and logsumexp. The transformation is u=t/(t+s), with s
estimated only from finite positive endpoints in the fit subset; exact
likelihood includes the Jacobian.

For depth D, K=2^D, leaf widths w_k summing to one:

```
R(x) = -2/(D*K) * sum_k log(K*w_k(x))
     = (2/D) * KL(Uniform_K || w(x))
train_loss = mean observed NLL + (a/sqrt(n_fit)) * mean R(x)
```

The penalty acts directly on geometry, not leaf probability masses. Width
subtraction and penalty logs use float64; the NN/masses use float32.
Adaptive rho uses 0.1+0.8*sigmoid(raw), with a 10-epoch rho=0.5 warmup.
The tree reserves minimum width 0.0008/K for each leaf. The actual geometric
split ratio also reflects that reserve. Fixed rho=0.5 is a separate method,
with frozen split parameters and a=0; it is not an adaptive candidate.

## Frozen search and training protocol

The six adaptive search variables are depth 3..8, shared layers 0..3,
private layers 1..4, shared/private widths in {16,32,64,128}, and a in
[0,100000]. For shared layers=0, shared width is inactive and canonicalized
to 32. Private layer count includes its feature projection, excluding the
output head. Both branches use the same architecture choices and independent
weights. Fixed rho searches the five effective architecture variables.

The implementation uses the existing Optuna 4.5.0 package in `.deps`:
16 startup Sobol/anchor trials plus 48 multivariate/group TPE trials per
setting and method, n_ei_candidates=48, with no epoch pruning.
a is searched using z=log10(1+a), including zero and strong anchors.
Each study uses one HPO dataset (seed=123), 4,000 pooled observations and
four common shuffled folds (seed=20261010): 3,000 fit / 1,000 validation.
CV initialization seed is 101. Mean unpenalized held-out NLL drives
checkpoints, Optuna feedback and selection. Brier does not enter this study.

Epoch is not a search parameter. CV uses early stopping after warmup;
complete-pool refit uses ceil(median(four CV best epochs)). Common settings:
Adam, mass/shared LR=1e-3, rho LR=1e-4, warmup=10, max epochs=200,
patience=50, validation every 5 epochs, min_delta=1e-4, batch=256,
gradient norm clip=1, with split Adam state reset on unfreezing.

Parameters and refit durations freeze before final evaluation. Each data
seed 0..9 generates a separate 5,000-subject dataset: first 4,000 train,
last 1,000 test. Methods share data and evaluation grids. The 256-point
uniform grid runs from zero to the training set's largest finite endpoint.
The metric matches neural_only:

```python
(estimated_survival - true_survival).square().sum(dim=1).mean()
```

The test set does not choose parameters or stopping epochs. However, these
final data seeds were also used in earlier methodological comparisons;
they are not a new untouched external benchmark. The SD conditions on one
frozen HPO result, and 64 trials do not establish a global optimum. The fixed
search repeats some architectures (49.34 distinct on average, range 42..56);
adaptive visited 64 distinct full configurations. Fourteen adaptive winners
have a>1000, so narrowing that upper bound is not supported as a lossless
change by the current records. Retain the existing architecture ranges.

## Retained CSV summaries

The historical result directory now contains only aggregate CSV tables and
frozen selections: `all_methods_grid_sse.csv`, `overall_grid_sse.csv`,
`family_grid_sse.csv`, `paired_grid_sse.csv`, `paired_comparisons.csv`,
`summary.csv`, `search_coverage.csv`, `selected_parameters.csv` and
`selected_adaptive_penalty_rho.csv`.

The combined five-method table is
`../finite_support_kl/outputs/01a12bd8-2275-76f0-9e35-f3542377af67/all_methods_grid_sse.csv`.
Single-case metric summaries are in `../kl_single_case/results/summary.csv`.

Per-fit/CV records, training histories, curves, figures, SQLite storage,
logs, generated reports and execution/audit manifests have been deleted.
The old result snapshot was replaced by the source-only backup
`saved_experiments/nll_cv_best_20261010_code.zip` at repository root.
The retained summaries cannot resume or re-audit the deleted fits. Use a
fresh output directory for a new study.

The preserved numerical source SHA256 is
`0be7177218cf4d7ecc36a9a0235ff1244b575fe306a0274a36be48f94cdf675d`;
the config fingerprint is
`7674a4eb6b3f054b1ca339f09ec69df22f820c27f017a00a4971a3a839b25f2f`.
Do not edit fingerprinted sources and reuse the old cache under a changed
protocol. New numerical methods or search ranges require a new result
folder. Some modules named for earlier workflows remain because the current
trainer, source fingerprint or regression tests import them.

## Entry points

Use `C:/Users/yixing/anaconda3/envs/PolyaPFN/python.exe` with the repository
root as the working directory. Opening the notebook is sufficient for
review. The following commands explicitly execute the retained notebook
or its tests when requested:

```powershell
python -m pycode.unified_nll.run_notebook pycode/unified_nll/tune_nll_cv_rho.ipynb
python -m unittest pycode.unified_nll.test_nll_cv -v
```

The runner's default notebook is now `tune_nll_cv_rho.ipynb`. No training was
launched as part of the CSV-only cleanup. Notebook code remains runnable;
executing it again will generate new fit artifacts.
