# New simulation datasets (`testdata2.py`)

The generator implements five simulation cases: proportional hazards (PH),
lognormal, proportional odds (PO), and two multimodal Weibull mixtures with
separate peaks or closer peaks.
Every scenario supports exact, right-censored, current-status, and
interval-censored observations.

## Covariates and predictor

For every subject, draw all six covariates independently:

$$X_{i1},\ldots,X_{i6}\overset{\mathrm{iid}}{\sim}U[-1,1],\qquad
\eta_i=-\frac{X_{i1}^2+X_{i2}^2}{2r^2}.$$

Only X1 and X2 affect the event distribution; X3-X6 are noise covariates.
The defaults are r=0.7 for PH, lognormal, and multimodal, and r=0.5 for PO.
The API accepts an override `r`. All formulas below use the default `mu=0`.
Setting `mu` multiplies event times, component scales, and the censoring horizon
by exp(mu).

## Event-time distributions

### Proportional hazards (`scenario="ph"`)

$$S(t\mid X_i)=\exp\{-\lambda\exp(\eta_i)t\},\qquad
\lambda=-\log(0.1)>0,\quad r=0.7.$$

Draw an exponential event time with rate lambda exp(eta), or equivalently mean
`scale = exp(-eta) / lambda`. The returned `shape` is one.
The original instruction's `lambda = log(0.1)` has been corrected: a negative
rate in the stated survival formula would give S(t)>1 for t>0.

### Lognormal (`scenario="lognormal"`)

$$\log T_i=\eta_i+\epsilon_i,\qquad\epsilon_i\sim N(0,0.5^2),\quad r=0.7.$$

As in the previous generator, `scale = exp(predictor + mu)` is the median;
here the new predictor is eta. The log-scale standard deviation is sigma=0.5,
returned as `shape`, so Var(log T | X)=0.25. Thus, for t>0,

$$S(t\mid X_i)=1-\Phi((\log t-\eta_i)/0.5),\qquad S(0\mid X_i)=1.$$

### Proportional odds (`scenario="po"`)

PO now has **finite support**, with upper endpoint `H = 20 * exp(mu)`.
At the default `mu=0`, every event time is below 20, including exact data.
For `0 <= t < H`, define `v = t / (exp(mu) * (1 - t/H))`. Then

$$S(t\mid X_i)=\frac{1}{1+\exp(\eta_i)v^2},\qquad r=0.5.$$

Survival is zero at and beyond H. The survival odds are still proportional
across covariates. Generate `W = exp(-eta/2) * sqrt((1-U)/U)` and
`T = exp(mu) * W / (1 + W/20)`. This smooth transformation keeps ordinary
times small while bounding the tail; it does not create a point mass at 20.

For PO, the returned `shape` tensor stores H and `scale` stores the
**dimensionless odds scale** `exp(-eta/2)/20`. Pass both directly to
`survival_curves(..., scenario="po")`. These PO parameters are not Weibull
parameters or the old log-logistic shape/time-scale pair.
Censoring calibration integrates the bounded survival using 512-panel Simpson
quadrature, including zero survival beyond H. Scaling mu also scales the
censoring horizon, preserving the target censoring probability.

### Multimodal (`data_reg_multimodal`)

Draw Z independently of X with probabilities (0.5, 0.5). Both cases use
baseline shapes k=(10,16) and r=0.7. These give narrower, sharper peaks,
similar to the sharper components in `testdata.py`, while retaining distinct
shapes and the non-PH mixture structure. Select the baseline scales with
`multimodal_case`:

| Case | Baseline scales a | Density peaks at X1=X2=0, mu=0 (approximately) |
| --- | --- | --- |
| `"separate"` (default) | (0.5, 6.0) | 0.495 and 5.976 |
| `"closer"` | (3.0, 6.0) | 2.969 and 5.976 |

The closer case moves the first peak toward the second while retaining two
conditional density modes across the default covariate range. The weights,
shapes, covariate predictor, and observation procedure are shared. With the
default covariate distribution, the pooled closer density peaks near 3.14
and 6.19, with the valley about 1.2% of the lower peak density. Thus the
two peaks have a pronounced density decrease between them.

$$T_i\mid X_i,Z_i=j\sim\operatorname{Weibull}(k_j,b_{ij}),\qquad
b_{ij}=a_j\exp(-\eta_i/k_j).$$

Each component is a PH model because

$$S_j(t\mid X_i)=\exp\{-\exp(\eta_i)(t/a_j)^{k_j}\}.$$

The conditional mixture survival is

$$S(t\mid X_i)=\tfrac12 S_1(t\mid X_i)+\tfrac12 S_2(t\mid X_i).$$

The distinct shapes and case-specific scales give two conditional density modes.
After marginalizing over Z, the mixture's hazard ratios vary with time; it
violates PH, rather than merely changing a PH baseline hazard. Both notebook
survival and density plots use the distinct shapes. Censoring does not depend
on the sampled component.

## Censoring and inspections

For censored datasets, draw an independent terminal follow-up time
`C ~ Uniform(0, tau)`, with the lower bound open: zero draws are resampled
so `C > 0` and right-censoring at time zero cannot occur.
The target q defaults to 0.2 (moderate); use q=0.5 for
high censoring. The same follow-up distribution is used for current-status
and interval data, retaining the old generator's inspection and bracketing
procedure.

The uniform horizon tau solves the population equation

$$q=P(T>C)=\frac{1}{\tau}\int_0^\tau E_X\{S(c\mid X)\}\,dc
=\frac{E\{\min(T,\tau)\}}{\tau}.$$

`censoring_horizon` evaluates the time integral analytically for PH, lognormal, and mixtures, and numerically for bounded PO,
averages over a deterministic 64x64 midpoint grid for X1 and X2, then solves
for tau by bisection in float64. For the mixture it averages both components
with weights (0.5,0.5), using the selected case's scales. Each case has its own
calibrated horizon; `censoring_horizon("multimodal", q, multimodal_case="closer")`
returns the closer case's horizon. Calibration is cached and consumes no random draws;
it uses neither the realized events nor their latent component labels.
Consequently censoring is independent of event times, covariates, and Z.
The target is a population rate, so finite samples vary around q.

The returned bounds follow the existing `(X, L, R, ...)` convention:

| Observation scheme | Bounds |
| --- | --- |
| Exact | L=R=T; q is unused. |
| Right | L=min(T,C); R=T if T<C, otherwise infinity. |
| Current status | (L,R)=(0,C) if T<C; otherwise (C,infinity). |
| Interval | For T<C, bracket T with inspection visits in [0,C], using 0 and C when no preceding/following visit exists. For T>=C, use (C,infinity). |

For interval observations with T<C, draw G~Gamma(shape=3, rate=1/3),
then K~Poisson(G) and K independent visits from Uniform(0,C).
The terminal visit at C supplies an upper bound even if K=0.
The fraction of infinite upper bounds is approximately q for all three
censored schemes; additional current-status and interval observations have
finite nonzero-width bounds.

## Usage and notebook integration

```python
import torch
from pycode.neural_only.testdata2 import (
    data_reg, data_reg_multimodal, survival_curves, multimodal_survival,
)

torch.manual_seed(123)
x, L, R, T, shape, scale = data_reg(
    n=5000, scenario="po", censor_type="right", q=0.2,
)
grid = torch.linspace(0, 10, 256)
truth = survival_curves(shape, scale, grid, scenario="po")

torch.manual_seed(123)
x, L, R, T, weights, scales, component = data_reg_multimodal(
    n=5000, censor_type="interval", q=0.5, multimodal_case="closer",
)
truth = multimodal_survival(weights, scales, grid)
```

`model_mlp.ipynb` and `benchmark_methods.ipynb` import `testdata2.py` and
share seed=123, n=5000, q=0.2, scenario="multimodal", multimodal_case="separate", censor_type="right",
CPU data generation, and an 80/20 split. Change the same settings in both
notebooks to compare methods on identical data. Choose any of the four
scenarios and four observation schemes; for the multimodal scenario, set
`MULTIMODAL_CASE` to `"separate"` or `"closer"` in both notebooks. Set q to 0.2
or 0.5 for the requested censoring regimes. Benchmark results record the selected
multimodal case and scales.

The notebooks use `survival_curves` for PH/lognormal/PO and
`multimodal_survival` for the mixture, so their SSE calculations use the
correct conditional truth. Parametric AFT uses the matching log-time error
family for each single-distribution simulation:

| Scenario | AFT event-time family | Error in log(T) |
| --- | --- | --- |
| PH | Exponential (Weibull shape one) | Minimum extreme value, scale one |
| Lognormal | Lognormal | Gaussian, with fitted standard deviation |
| PO | Log-logistic working model | Bounded transformed logistic; not a family match |
| Multimodal | Single Weibull working model | Minimum extreme value, with fitted scale |

The PH exponential choice matches the specified constant baseline hazard.
Lognormal has Gaussian log-time error with standard deviation 0.5. Bounded PO is no longer exactly log-logistic; the AFT benchmark uses an unbounded working model.
Multimodal is deliberately fitted with one Weibull,
rather than a mixture. The AFT covariate predictor remains linear in X1-X6;
the error-family match does not add the generator's nonlinear squared terms.

Both active notebooks use this generator. Legacy generators, standalone
runners, and separate test and paper-reproduction scripts have been removed.
