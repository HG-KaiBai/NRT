# Real-data analysis: CO, Rotterdam recurrence and death

Real-data adapters, data, logs, model artifacts, tuning caches, predictions and
comparison tables are contained in this directory. The reference experiment is
`../nrt-29cases-20261021_105101`. Models, search samplers and benchmark backends are
imported directly from `../regularization/`, `../tuning/` and `../testing/`.
There is no local copy of simulation source. Run commands from the NRT project
root so these shared modules are importable. Each new run records hashes of both
the real-data code and shared Python/R code in `request.json`, relative to NRT.
The package inventory is referenced directly from `../environment/fir-20261021/`;
`../runtime.sh` loads the shared Python/R runtime.

## Outcomes and methods

| Analysis | Response | Covariates | Evaluation |
| --- | --- | --- | --- |
| `gt_2011` | CO, uncensored | All other 10 columns | Harrell C-index; ordinary Brier curve and IBS |
| `rotterdam_recur` | rtime/recur | Baseline variables; exclude pid and all outcome columns | Uno IPCW C-index; Yang IPCW Brier curve and IBS |
| `rotterdam_death` | dtime/death | Same baseline variables and patient split as recurrence | Uno IPCW C-index; Yang IPCW Brier curve and IBS |

Rotterdam tumour size is dummy coded with `<=20` as reference and indicators for
`20-50` and `>50`; `year` is retained. Event indicators are used exactly as supplied.
The recurrence analysis uses the supplied right-censoring indicator; it is not a
competing-risk cumulative-incidence analysis or a combined disease-free endpoint.
Death is a separate all-cause mortality endpoint using `dtime/death`. Both Rotterdam
analyses exclude `pid`, `rtime`, `recur`, `dtime` and `death` from predictors, and
share the same patient split. Each endpoint has its own training-observation tau.

Each analysis has 9 methods: three NRT variants plus six benchmarks. The common
benchmarks are NNCDE, DeepSurv, ctree, Cox and Weibull AFT. GT adds Deep Distribution
Regression (DDR); both Rotterdam endpoints add ICcforest. No generating family is
known for real data, so AFT uses the declared Weibull working model, without
selecting its distribution from the test set.

The NRT variants are `fixed_nll_cv`, `adaptive_nll_cv`, and
`fixed_at_adaptive_architecture`, using partition KL. The matched control uses
the adaptive architecture and its own CV epoch selection.

## Training protocol

One reproducible 80/20 split uses the simulation's CPU torch permutation with
seed `123+1`. All methods within each analysis share the same split.
Hyperparameters are selected entirely within the 80% pool. This
is one held-out real-data analysis per endpoint, not 500 independently generated
simulation datasets. No simulation tuning result is reused as a real-data optimum.

* NNCDE: all 27 combinations of widths 64/128/256, learning rates
  .01/.001/.0001, and batch sizes 64/128/256; four-fold validation NLL;
  200 epochs, patience 20, two-half reverse cross-fit and averaged predictions.
* DeepSurv: the original Optunity Sobol search, 50 candidate evaluations,
  four-fold validation Harrell C-index and 1,000 epochs; simulation search ranges,
  SELU architecture, Adam, constant learning rate and output-weight regularization.
  This CV criterion follows simulation; final evaluation uses the definitions below.
* NRT: by default, the simulation SearchConfig/Architecture settings, 200 trials
  (20 startup and 180 TPE), four folds, unregularized validation NLL,
  200 epochs, patience 50, validation every 5 epochs, warmup 10,
  mass/shared LR .001 and split LR .0001. Refit epochs are the rounded-up median
  of best CV epochs. The shared body horizon is the largest observed training
  endpoint, fixed before CV and independent of test outcomes. The tree partitions
  the body, and the final probability output splits only the rightmost leaf into
  its body mass and the tail beyond that horizon.
* ctree/Cox/AFT/DDR/ICcforest reuse the simulation backends and defaults:
  DDR 100 epochs/patience 20; forest 100 trees with the original internal mtry
  tuning; AFT max iterations 250. No extra external search is added.

Real-data adaptations are explicit: NRT and conventional backends use training-only
z scores; NRT CV fits each scaler on its inner training fold. DeepSurv uses
`standardize=True` so each internal CV fit learns its own normalization; its Cox
loss uses Breslow ties, since both real datasets have tied outcomes. The upstream
no-ties simulation implementation is unchanged. NNCDE keeps its internal
expanded-input normalization and original time grid; the adapter only streams
moments, validation and gradient accumulation to bound memory. All outcomes retain
their original units. No time jitter, row filtering, or missing-value imputation
is applied to full runs; both supplied CSVs are complete.

## Metrics

All evaluation uses exactly the simulation grid: `torch.linspace(0, tau, 256)`,
including smoke runs. `tau` is the largest **finite observed training endpoint**;
right-censored lower bounds count as observations. The float32 construction
matches simulation's `Dataset.bounds` and `Dataset.max_observation`. There is no
90th-percentile cutoff, censoring-survival threshold, or test-derived horizon.
This also equals NRT's body horizon; survival there equals its fitted tail mass.
Retained runs from before the body/tail change used the old 1.1-times finite
support and require fresh fitting to evaluate the current model.

### C-index

**Uncensored GT: traditional Harrell C-index.** Compare all pairs with distinct
observed outcomes, including outcomes beyond tau. A larger risk should predict
an earlier outcome. Our common scalar risk is `-trapezoid(S_i, grid)`, the negative
restricted mean survival time to tau, computed from each model's fitted curve.
This choice of marker is an evaluation protocol choice; Harrell's statistic does
not prescribe RMST. Equal observed outcomes are excluded and risk ties within
`1e-8` receive half credit. No censoring weights are used.

**Right-censored Rotterdam: Uno et al. (2011) IPCW C-index.** The fixed scalar
risk is the evaluated model's fitted event probability `r_i=1-S_i(tau)`. This
marker is an evaluation protocol choice; Uno's statistic accepts any scalar risk
score. The reported `c_index` is truncated at tau. With observed time T and event
indicator d, the implemented finite-sample convention is:

```
A_ij(t) = I(d_i=1, T_i<t) * [I(T_i<T_j) + I(T_i=T_j, d_j=0)]
w_i = 1 / G(T_i)^2
credit_ij = I(r_i>r_j) + 0.5*I(r_i==r_j)
C_Uno(t) = sum_(i!=j) A_ij(t)*w_i*credit_ij / sum_(i!=j) A_ij(t)*w_i
```

`G` is marginal censoring KM fitted on the **test sample**, under the stated
assumption that censoring is independent of both event time and prediction risk.
Conditional independence given covariates alone would require a conditional
censoring estimator. No Cox censoring model is used.

The implementation is numerically checked against
`sksurv.metrics.concordance_index_ipcw(y_test, y_test, risk, tau=tau)`. Using the
test sample as the first argument explicitly retains this analysis's test-KM
protocol; scikit-survival's usual call instead estimates KM on training outcomes.
Reverse KM removes tied events before calculating the censoring hazard. Tied
event/event pairs are excluded, tied event/censor pairs are usable, and prediction
ties within `1e-8` get half credit. The strict event boundary `T_i<tau` follows
scikit-survival; later subjects remain available as pair comparators. These tie
and endpoint conventions extend the paper's continuous-time formulation.

Zero censoring survival at an included event makes C unavailable, rather than
silently dropping that event. Weights are not clipped and tau is not shortened.
A draw with no comparable pairs also has unavailable C. Metadata records whether
censoring survival is positive at tau. Keeping the requested full grid beyond
its support does not provide the usual IPCW identification guarantee there.

Implementation: `metrics.concordance_pairs`, `concordance_ratio`,
`uno_ipcw_weights` and `uno_concordance_curve`. For right-censored data,
`c_index_curve.csv` saves Uno C for 256 truncation times with the same fixed risk
`1-S(tau)` at every time. The uncensored analysis reports its single global
Harrell C. Reference implementation:
[scikit-survival concordance_index_ipcw](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html).

### Brier score

For Rotterdam, use the **IPCW Brier score of Yang et al. (2026), section 2.4**,
with landmark zero, specialized to one event and exact/right censoring:

```
BS(u) = mean[ I(T_i<=u, event_i=1)*S_i(u)^2/G(T_i)
            + I(T_i>u)*(1-S_i(u))^2/G(u) ]
IBS = trapezoid(BS, grid) / tau
```

`S_i` is always the evaluated model's fitted event survival distribution. `G` is
separately the censoring survival distribution, estimated on the **test sample**
by ordinary Kaplan–Meier with `1-event` as event indicator, matching the authors'
`survival::survfit(Surv(T, 1-event) ~ 1)` code. Thus tied outcomes use ordinary KM
risk sets, not scikit-survival's event-before-censor reverse-KM convention. The
case boundary is `T<=u` and control boundary `T>u`. This endpoint convention has
no effect for continuous times and is stated explicitly for the tied real data.

Already-censored subjects contribute zero; divide by `n_test`, not the sum of
weights. Empty summands stay zero, including beyond the last follow-up where
`G(u)=0` and no subject remains observed past u. No positive weight floor or
shorter evaluation horizon is substituted. At those late points the estimate has
no observed controls and relies only on event contributions; keeping the full
grid does not restore information about unobserved tail outcomes. GT has no
censoring, so its Brier score is simply `mean((I(T>u)-S(u))^2)`, using the usual
binary squared-error normalization. `brier_at_horizon` is BS(tau);
`integrated_brier_score` is the uniform, normalized trapezoidal integral over the
same 256-point grid. Both summaries use the relevant ordinary/IPCW Brier curve.

Implementation: `metrics.brier_contributions`, `censoring_survival`, and `evaluate`.
References: [published Yang et al. paper](https://onlinelibrary.wiley.com/doi/full/10.1002/bimj.70108),
[author implementation](https://github.com/ZhenweiYang96/Predicitve-accuracy-metrics/blob/main/R%20script/Function/time%20dependent%20evaluation%20metrics.R).
The implementation is for these three single-event analyses, not a general
interval-censored competing-risk evaluator.

### Metric references

Every `result.json` and `bootstrap.json` records the relevant metric references.
All choices of grid, scalar risk, tie tolerance, and 500 bootstrap draws are the
declared evaluation protocol, not claims that those choices are required by a paper.

1. **Traditional C-index:** Harrell FE Jr, Califf RM, Pryor DB, Lee KL, Rosati RA.
   (1982). *Evaluating the Yield of Medical Tests*. JAMA, 247(18), 2543–2546.
   [DOI: 10.1001/jama.1982.03320430047030](https://doi.org/10.1001/jama.1982.03320430047030).
2. **Ordinary Brier score:** Brier GW. (1950). *Verification of Forecasts Expressed
   in Terms of Probability*. Monthly Weather Review, 78(1), 1–3.
   [Publisher / DOI: 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml).
3. **Right-censored IPCW C-index:** Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ.
   (2011). *On the C-statistics for evaluating overall adequacy of risk prediction
   procedures with censored survival data*. Statistics in Medicine, 30(10),
   1105–1117. [DOI: 10.1002/sim.4154](https://doi.org/10.1002/sim.4154).
4. **Right-censored IPCW Brier:** Yang Z, Rizopoulos D, Newcomb LF, Erler NS.
   (2026). *Time-Dependent Predictive Accuracy Metrics in the Context of Interval
   Censoring and Competing Risks*. Biometrical Journal, 68(1), e70108; section 2.4,
   single-event/right-censored specialization with landmark zero.
   [DOI: 10.1002/bimj.70108](https://doi.org/10.1002/bimj.70108).
5. **Integrated Brier summary (IBS):** Graf E, Schmoor C, Sauerbrei W, Schumacher M.
   (1999). *Assessment and comparison of prognostic classification schemes for
   survival data*. Statistics in Medicine, 18(17–18), 2529–2545.
   [PubMed / DOI: 10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5](https://pubmed.ncbi.nlm.nih.gov/10474158/).
   Graf supplies the integrated-summary reference; the censored pointwise Brier
   estimator above follows Yang.
6. **Censoring distribution estimator:** Kaplan EL, Meier P. (1958).
   *Nonparametric Estimation from Incomplete Observations*. Journal of the American
   Statistical Association, 53(282), 457–481.
   [DOI: 10.1080/01621459.1958.10501452](https://doi.org/10.1080/01621459.1958.10501452).

## Test-set bootstrap confidence intervals

After tuning/fitting, `realworld.run run` automatically computes **500 test-set
bootstrap replicates and 95% percentile confidence intervals**. Each draw samples
`n_test` subjects with replacement, keeping outcomes and entire prediction curves
together. Training data, hyperparameters, fitted models and the 256-point grid
stay fixed. For censored data, both C-index's reverse KM and Brier's ordinary KM
are re-estimated in each draw. Copies of the same subject have tied observed
outcomes and cannot form comparable pairs. These intervals condition on the fitted prediction model
and do not include training/tuning uncertainty.

The seed is `20261102`; rows are sorted by original row ID, so every method uses
the same draws. Quantiles use linear interpolation at 2.5% and 97.5%. Outputs are
`bootstrap.json`, `bootstrap_replicates.csv`, and `brier_curve_ci.csv`; the latter
contains pointwise intervals, not simultaneous bands. C replicates without
comparable pairs, or with zero G at an event before tau, are excluded only from
C intervals, with valid/invalid
counts reported. Brier still uses every draw. Fewer than two valid C replicates
makes its interval unavailable. Summary tables include C, IBS and Brier(tau)
intervals and valid-draw counts.

To update old fitted models to the new evaluation, **re-predict** on the full grid:

```bash
source runtime.sh
"$PY" -m realworld.run reevaluate \
  --source-output realworld/runs/full_co_recur_death_uno_km_20261102 \
  --output realworld/runs/reevaluated_co_recur_death \
  --bootstrap-resamples 500 --require-complete
```

This loads the existing `.pt`/`.rds` models and saved training preprocessing on
CPU without tuning or refitting. It verifies the original data/split, produces
new predictions and metrics, then bootstraps them. Source training artifacts
stay unchanged. New `reevaluation.json` and per-method `evaluation.json` record
source paths/hashes and evaluation provenance separately from the original
`request_hash`. Neural predictions are streamed in batches; R backends receive
the full test set. A changed input/source requires a new reevaluation directory.

To recompute only intervals for predictions already using the current metrics:

```bash
source runtime.sh
"$PY" -m realworld.run bootstrap \
  --output realworld/runs/reevaluated_co_recur_death \
  --bootstrap-resamples 500 --bootstrap-confidence 0.95 --bootstrap-seed 20261102
```

This checks saved row IDs, outcomes, grid, data hash and point estimates; matching
bootstrap artifacts are reused. Old predictions on the shorter grid must first
be reevaluated with the previous command. Use `--dataset`/`--methods` for subsets.
`status.csv` tracks evaluation and bootstrap completion;
`summarize --require-complete` requires all 27 methods and their intervals.

## Dataset-specific NRT search ranges

Edit `nrt_search_spaces.json` and pass it with `--nrt-search-space`. The dataset
keys are `gt_2011`, `rotterdam_recur` and `rotterdam_death`. The supplied file sets dataset-specific
widths and regularization ranges. Each entry accepts a `search` object and an
optional `baseline` object. A top-level `defaults` entry supplies common overrides;
dataset entries take precedence. Omitted entries/fields use the selected profile's
defaults. This file affects NRT only, including its matched fixed control. Death
initially uses the same search ranges as recurrence: depth 3–8, widths
8/16/32/64, and maximum regularization strength 1000; each endpoint tunes separately.

| `search` field | Full-profile default | Meaning |
| --- | --- | --- |
| `depth_min`, `depth_max` | 3, 8 | Inclusive integer tree-depth bounds |
| `widths` | `[16, 32, 64, 128]` | Candidate widths for shared/private networks |
| `shared_layers_max` | 3 | Shared layer count ranges from 0 to this value |
| `private_layers_max` | 4 | Private layer count ranges from 1 to this value |
| `strength_max` | 10000 | Adaptive regularization ranges from 0 to this value, sampled in log10(1 + strength) |
| `trials`, `initial_trials` | 200, 20 | Total trial budget and Sobol/anchor startup budget |
| `folds`, `fold_seed` | 4, 321 | NRT-only inner CV split settings |
| `search_seed`, `search_fit_seed` | 871, 101 | Search and inner model-fit seeds |

For example, a smaller GT search and a separate recurrence depth range:

```json
{
  "gt_2011": {
    "search": {"depth_min": 2, "depth_max": 5, "widths": [16, 32], "strength_max": 1000}
  },
  "rotterdam_recur": {"search": {"depth_min": 4, "depth_max": 7}}
}
```

The baseline supplies the first startup architecture. Omitted baseline values are
projected into the selected ranges (nearest allowed width; clipped depth, layers
and strength). Explicit `baseline` values must lie inside the ranges. Its allowed
fields are `depth`, `shared_layers`, `shared_width`, `private_layers`,
`private_width`, and `regularization_strength`; input dimension is inferred from
the dataset. Fixed NRT uses strength zero; the matched control uses the adaptive
winner's architecture with strength zero and its own CV epoch selection. Very
small discrete spaces enqueue only as many unique startup points as available.

Preview the resolved settings, then submit into a new output directory:

```bash
source runtime.sh
"$PY" -m realworld.run plan --nrt-search-space realworld/nrt_search_spaces.json
"$PY" -m realworld.submit --nrt-search-space realworld/nrt_search_spaces.json \
  --output realworld/runs/custom_ranges --dry-run
"$PY" -m realworld.submit --nrt-search-space realworld/nrt_search_spaces.json \
  --output realworld/runs/custom_ranges
```

For a single local task, the same option is accepted by `realworld.run run`.
Preparation validates all datasets and freezes the resolved settings in the
output directory's `nrt_search_spaces.json` and `request.json`. Slurm workers and
resume commands automatically read that frozen file. Editing the original input
file after submission does not change queued jobs. Changing the frozen settings,
passing different explicit settings to an existing output, or changing shared
source code triggers a fingerprint mismatch; use a new output directory.
Dataset manifests and NRT selections also record the effective search settings.
Historical completed runs keep their original provenance and remain readable.

## Run and resume

Run from the NRT project root, using the same Python 3.14.2 / R 4.4.0 runtime:

```bash
source runtime.sh
"$PY" -m realworld.run preflight
"$PY" -m realworld.run run --profile smoke --output realworld/runs/smoke_example
"$PY" -m realworld.submit --dry-run
"$PY" -m realworld.submit
```

The submit command prepares data once, submits 24 task groups (13 GPU and 11 CPU),
and schedules a summary after both arrays finish. Each of the three adaptive/control
pairs runs sequentially in one task, giving 27 method results. Arrays have no
user-imposed concurrency cap. GPU tasks use the same
Fir H100 20GB slice and shared node/requeue checks as simulation. Resource settings
are in `job.sh` and `submit.py`. `submission.json` records job IDs and commands.

Outputs default to `runs/full_co_recur_death/`. Each analysis contains a dataset
manifest, saved split, and one folder per method with selection, CV records,
model, preprocessing, test predictions, Brier curve and metrics. Root
`comparison.csv` combines available results; `status.csv` identifies pending or
failed methods. The final summary exits unsuccessfully if any method is missing.
`--profile smoke` uses 80 sampled rows, tiny budgets and separate output paths;
its results must never be reported as full-analysis results.

Resume a task after interruption using its original request and output directory:

```bash
"$PY" -m realworld.run run --task-id 0 --device cuda:0
"$PY" -m realworld.run summarize --require-complete
```

NRT resumes its SQLite study and fold caches; NNCDE resumes cached folds.
An interrupted DeepSurv search restarts that search (completed selections are
reused). Source/configuration/data fingerprints prevent mixing different
experiments in one output directory. Keep code fixed while production jobs run.
Completed results remain readable by `summarize` after a source revision. Use a
new `--output` directory for new training with revised source; historical request
fingerprints are preserved.
Package inventories and Python version are recorded in `request.json`; each
method also records the actual device. Learned `.pt` objects should only be loaded
from this trusted run with its recorded source and environment.
