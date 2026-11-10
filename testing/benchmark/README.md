# Conditional-distribution benchmarks

This directory contains the model backends used by the retained benchmark
workflow. See [the stage guide](../README.md) and
[the full rerun procedure](../../RERUN_SIMULATIONS.md) for the reference protocol.
Shared data generation and evaluation are defined by root `workflow.py`.

```python
average_grid_sse = (estimated_survival - true_survival).square().sum(dim=1).mean()
```

The simulation grid has 256 points from zero to the largest finite training
bound. This score is a pointwise sum, not an integral.

## Supported observations

| Folder | Paper implementation | Supported `censor_type` |
|---|---|---|
| `iccforest` | Yao, Frydman, and Simonoff (2021) conditional inference forest | `right`, `interval`, `current` |
| `ctree` | Fu and Simonoff (2017) interval-censored conditional inference tree | `exact`, `right`, `interval`, `current` |
| `nncde` | Hu and Nan (2023) full-likelihood neural hazard estimator | `exact`, `right` |
| `deep_distribution_regression` | Li, Reich, and Bondell (2021) fixed-partition JBCE network | `exact` |
| `cox` | Linear Cox proportional-hazards model | `exact`, `right`, `interval`, `current` |
| `aft` | Scenario-matched parametric AFT model | `exact`, `right`, `interval`, `current` |
| `deepsurv` | Katzman et al. (2018) Cox deep neural network | `exact`, `right` |

The restrictions are intentional. NNCDE's censored-data likelihood assumes an
observed event/right-censoring indicator, while Deep Distribution Regression
was proposed for fully observed continuous responses. ICcforest is run only on
censored survival inputs; its interval representation also covers ordinary
right censoring and current-status observations. The Fu--Simonoff ctree handles
the complete interval representation, including its exact-event special case.

## Run

Use the pinned Python 3.14.2 / R 4.4.0 environment from
[environment/fir-20261021](../../environment/fir-20261021/README.md).
Run the current tuning and simulation CLIs described in the stage guide;
the historical benchmark notebook was removed during cleanup.

The `lognormal` scenario uses a lognormal event-time distribution with
covariate-dependent median and log-scale standard deviation 0.5. The
`multimodal` scenario uses a two-component Weibull mixture with equal weights
and shapes `(5, 8)`. Its reference survival curve marginalizes over both
components; the sampled component is latent and is never supplied to a
fitted method. See `../../tuning/testdata2.md` for all four generating scenarios.

ICcforest deliberately calls the official R package rather than substituting
a Python random-survival-forest algorithm. It requires R with `ICcforest`,
`icenReg`, and `survival`. Its default settings tune `mtry` out-of-bag and use
the paper's 15%-`minsplit`/6%-`minbucket` rule. Set `tune_mtry=False` only for a
quick smoke test.

The ctree benchmark directly calls `partykit::ctree` with the interval
log-rank response transformation from Fu and Simonoff. It uses the `partykit`
defaults `mincriterion=0.95`, `minsplit=20`, `minbucket=7`, and unlimited tree
depth. Turnbull NPMLE survival curves are fitted separately in the terminal
nodes. These controls are exposed by the model constructor.

NNCDE follows the paper/reference code: two 64-unit ReLU hidden layers,
piecewise full-likelihood hazard estimation, Adam, early stopping after ten
non-improving validation epochs, a 1:1 split, reverse refitting, and averaging
the two survival estimates.

Deep Distribution Regression uses three 100-unit ELU/BatchNorm layers, 50%
dropout, softmax bin probabilities, cumulative probabilities trained with the
joint binary cross-entropy (JBCE), evenly spaced fixed bins, and the authors'
recommendation of about 10% as many cutpoints as training observations.

The Cox benchmark uses R throughout. Exact/right-censored observations use
`survival::coxph`; interval- and current-status observations use
`icenReg::ic_sp(model="ph")`. Both paths return fitted PH survival curves
without adding nonlinear covariate terms. R with `survival` and `icenReg` is
therefore required.

For interval/current-status fits, prediction at or beyond a finite terminal
support endpoint with zero baseline survival is explicitly zero. This avoids
an `icenReg` interpolation roundoff error that can return `NA` at that endpoint.

The AFT benchmark also uses R throughout. Exponential, Weibull, lognormal,
and log-logistic models use
`survival::survreg` with a `Surv(..., type="interval2")` response, which covers
exact, right-, interval-, and current-status data in one representation. The
multimodal scenarios are intentionally fitted with one Weibull AFT model; the
benchmark does not estimate a Weibull mixture. R with `survival` is required.
The PH simulation uses exponential AFT
(minimum extreme-value log-time errors with scale one), lognormal uses normal
log-time errors, PO uses logistic log-time errors (`dist="loglogistic"`), and
multimodal uses a single Weibull working model. The AFT predictor remains
linear in the supplied covariates.

AFT omits observations right-censored at time zero (`L=0`, `R=Inf`) from the
R fit. They contribute a constant likelihood of one for these positive-time
families, while `survreg` cannot log-transform the zero censoring time. The
original simulation dataset and evaluation partition are retained; the saved
R model artifact records the omitted count as `n_uninformative`.

DeepSurv uses one PyTorch implementation of the official network architecture
and training contract, together with the STG PyTorch Cox loss. It runs directly
on CUDA in the normal project environment; no legacy Theano subprocess is
maintained. Exact architecture, optimization values, and paper-reproduction
parameters are listed in `deepsurv/README.md` and written to result JSON. It is
available for exact and right-censored data and is not extended to interval or
current-status data. A standard Breslow Cox baseline turns learned relative
risks into survival curves for the shared `average_grid_sse` evaluation. The
workflow retains the `multimodal_separate` and `multimodal_closer` cases.

`tuning.main --method DeepSurv` executes the Sobol cross-validation fits on
the selected PyTorch device and refits the selected configuration on the training pool.

## Sources

- W. Yao, H. Frydman, and J. S. Simonoff, "An ensemble method for
  interval-censored time-to-event data," *Biostatistics* 22(1), 2021.
- W. Fu and J. S. Simonoff, "Survival trees for interval-censored survival
  data," *Statistics in Medicine* 36(30), 2017.
- B. Hu and B. Nan, "Conditional Distribution Function Estimation Using
  Neural Networks for Censored and Uncensored Data," *JMLR* 24, 2023;
  reference code: <https://github.com/bingqing0729/NNCDE>.
- R. Li, B. J. Reich, and H. D. Bondell, "Deep distribution regression,"
  *Computational Statistics & Data Analysis* 159, 2021; reference code:
  <https://github.com/RLstat/deep-conditional-distribution-regression>.
- J. L. Katzman, U. Shaham, A. Cloninger, J. Bates, T. Jiang, and Y. Kluger,
  "DeepSurv: personalized treatment recommender system using a Cox
  proportional hazards deep neural network," *BMC Medical Research
  Methodology* 18, 24, 2018; reference code:
  <https://github.com/jaredleekatzman/DeepSurv>.
