# Current NRT: regularized NLL tuning and simulation testing

This directory organizes regularized NLL (the observed-data NLL with KL regularization)
into two separate stages: first, use the existing tuning dataset for cross-validation
(CV), hyperparameter selection, and fixing the number of training epochs; then refit
and evaluate under the existing simulation dataset protocol. By default, the workflow
retains the three comparison methods from the original code.

| Method | rho and regularization | Hyperparameter selection |
| --- | --- | --- |
| `fixed_nll_cv` | Fixed rho=0.5, a=0 | Search architectures independently |
| `adaptive_nll_cv` | Learn rho, KL strength a≥0 | Search architectures and a independently |
| `fixed_at_adaptive_architecture` | Fixed rho=0.5, a=0 | Use the architecture selected by adaptive, with a separate CV run to determine its own training epochs |

`unified_nll/` retains the shared model, tree, training, and metric dependencies.
Standalone legacy experiments and `kl_single_case/` have been removed. The network architecture is reused from
`unified_nll/tuning_model.py`; the training loop is copied from
`unified_nll/nll_cv_train.py` and connected to the local body-and-tail model on the
original time scale, as specified in `NeuralRT2.tex`.

## Exact consistency with the existing data

The new workflow directly imports `workflow.generate()`, `workflow.horizon()`, and
`workflow.cv_splits()` from the NRT root. The underlying generator remains
`tuning/testdata2.py`; there is no separate implementation that could drift.

| Item | Setting shared with the existing workflow |
| --- | --- |
| Data cases | 3 unimodal settings × 7 combinations + 2 multimodal settings × 4 combinations = 29 cases |
| Settings | ph, lognormal, po, multimodal_separate, multimodal_closer |
| Observations | exact; right q=0.2/0.5; unimodal current/interval q=0.2/0.5; multimodal interval q=0.2 |
| Exact case ID | Keep `_exact_q0.2`; generate it only once, with q unused in generation |
| Each dataset | n=5000 and 6 covariates by default |
| Tuning seed | `base_seed + case.index * 10000`, with base_seed=123 by default |
| Simulation seed | `base_seed + 1000000 + case.index * 10000 + replication` |
| Replication | Starts at 1; defaults to 1..500, matching the existing testing indices |
| tau | Tuning uses the same horizon calibration function; testing uses tau saved in the CSV without recalibration |
| Train/test | Random permutation using an independent CPU Generator with `seed+1`; first 80% / last 20% |
| Tuning CV | Use the original `cv_splits` within the same 80% training pool; defaults to seed=321 and 4 folds |
| mu / r | Defaults to 0 / None; custom values saved during tuning are passed unchanged to testing |

For example, `ph_right_q0.2` has tuning seed=20123 and simulation replication 1
seed=1020124. `case.index` retains the indices from the existing implementation
before case exclusions. Do not replace these with consecutive
indices 0..28 for data generation; array task indices are used only to select cases.

In the same Python/PyTorch environment, with identical n, base seed, mu, r, tau,
case, and replication, the two sets of entry points produce identical X, L, R, T,
generating-distribution parameters, and train/test indices, element by element.
Different methods also share the same data. Each new result stores a
`dataset_sha256` covering these eight tensors. Tests call the existing tuning entry
point directly to capture generated data and use the existing testing `make_job`
to check simulation seeds and datasets, covering all 29 cases with n=5000.

`tuning/output/cases/<case_id>.json` uses the same fields as the original tuning
workflow, including seed, tau, mu, r, sample counts, and observed right-censoring
rates for the full dataset, training set, and test set. Data consistency does not
imply that different training methods produce identical models or SSE values.

## Model, loss, and support

The model partitions the finite body `(0,U]` on the original time scale t and
retains a separate tail mass for `T>U`. Here U is the manuscript's horizon tau;
the data-generation field `tau` in saved CSVs remains the censoring horizon.

```text
U = max(finite L/R endpoints in the 80% training pool), fixed before CV search
K = 2**depth
cut head: K-1 outputs; probability head: K outputs
body masses: K; tail mass: 1
R(x) = -2/(depth*K) * sum_k log(width_k(x)/(U/K))
lambda_n = a/sqrt(n_fit)
objective = mean(observed-data NLL) + lambda_n * mean(R(x))
```

The first probability output splits the root at U into the whole body and the
terminal tail: `p_body = sigmoid(logits[:, 0])` and `s = sigmoid(-logits[:, 0])`.
The remaining K-1 probability outputs follow breadth-first order in a body subtree
with `depth` levels on `(0,U]`, producing conditional body masses `pi_tilde_k` that sum
to one. Every body mass is `pi_k = p_body * pi_tilde_k`, so body masses plus s
sum to one. Here depth counts only body levels, excluding the root body/tail split.
There is no extra learned cut or tail width; the partition penalty uses only body
widths. Network dimensions, initialization, and the fitting procedure are unchanged.

Exact events use the density with terminal intervals always `(a,b]`: a cut point
belongs to its left interval, U belongs to the last body interval, and zero has
no body density. With `q_k` the exact density or interval-overlap fraction in leaf k,
the likelihood is `p_body * sum_k(pi_tilde_k * q_k) + s * I(R=inf)`.
In particular, `(U,inf)` has likelihood s and `S(U)=s`.
The tail's internal distribution is unspecified. Observation preparation maps
`L` to `min(L,U)` and any `R>U` to infinity, so an exact event beyond U becomes
`(U,inf)` and an interval crossing U includes the whole tail. This coarsening also
applies to validation/test observations. Survival prediction supports `[0,U]`
and `+inf` (where survival is zero); finite prediction times beyond U are rejected.
Geometry and leaf widths use float64; the network and log mass use float32.
The minimum leaf width is `U * 0.0008/K`; rho=`0.1+0.8*sigmoid(raw)`.
All CV folds, candidates, and methods for a case share the same U, which is also
used for the final fit.

This protocol fixes the support using the outer training pool: observed endpoints
from the inner validation folds contribute to U, but U is not adjusted by candidate
or validation score. The outer 20% test set and true event times do not contribute.
Simulation testing still determines U from the 80% training pool of each simulated
dataset; it does not reuse the numerical upper bound from the tuning dataset.
When `fit_raw_time_kl` is called directly, U is determined from the current fitting
subset by default. CV explicitly passes the previously fixed support upper bound
through `upper_bound`. The request file records `support.policy=tuning_train_pool`
and `upper_bound`; each fold's `fit_info.max_observation` still records the maximum
observed endpoint in that fold's actual training subset.

Likelihood calculations remain strict: zero probabilities are not clipped to
epsilon. A direct fit still raises an error if a validation observation has zero
likelihood after coarsening. If testing encounters a zero likelihood, the result stores
`test_nll=null, test_nll_infinite=true`, while SSE can still be computed normally.
Infinite NLL is not written as nonstandard JSON.

The shared body horizon keeps all CV finite endpoints within the body while
preserving data generation, the 80/20 split, and CV indices. Fits record
`fit_info.support=raw_time_body_tail`, `upper_bound=U`, and the unscaled maximum
observed endpoint in the fitting subset as `max_observation`.

## Part 1: tuning

1. Generate the dataset and random 80/20 split using the original workflow. Fix
   the shared support from observed endpoints in the first 80%; the remaining
   20% does not contribute to hyperparameter selection or support determination.
2. Run 200 candidates per case independently for fixed and adaptive: 20 Sobol/anchor
   startup candidates, followed by 180 TPE candidates. Adaptive includes a=0 and
   a strong-regularization anchor, with a searched on the `log10(1+a)` scale.
3. Evaluate every candidate using the same 4-fold CV within the training pool.
   Both checkpoint selection and hyperparameter selection use **unregularized
   validation NLL**, without Brier scores or true survival curves.
4. Select the candidate with the lowest weighted mean validation NLL, breaking
   ties by trial order. Set the final number of training epochs to the median
   of the best epoch counts across folds, rounded up.
5. The matched control reuses the adaptive architecture but determines its own
   training epochs through CV with fixed rho.
6. After freezing all requested configurations for the current case, refit on the
   full training pool for the fixed number of epochs. Only then compute tuning
   held-out test SSE/NLL.
7. After all requested cases for a method finish, automatically export its
   hyperparameter CSV and tuning summary.

| Search/training item | Default |
| --- | --- |
| depth | 3..8 |
| shared layers | 0..3; at 0, shared width has no effect and is normalized to the baseline width |
| private layers | 1..4, including feature projection but excluding the output head |
| shared/private width | 16, 32, 64, 128 |
| a (adaptive) | 0..10000 |
| baseline | depth=3, shared layers=0, shared/private width=32, private layers=3, a=500 |
| HPO search / fit seed | 871 / 101 |
| optimizer | Adam; mass/shared LR=1e-3, split LR=1e-4 |
| warmup | Fix rho=0.5 for the first 10 epochs; then reset the split Adam state and unfreeze |
| max epochs / patience | 200 / 50 epochs |
| validation_every / min_delta | 5 / 1e-4 |
| batch size / gradient clip | 256 / 1.0 |
| CUDA graphs | Enabled by default; disable with `--no-cuda-graphs` |

The full default tuning run requires `29 * (2*200*4 + 4) = 46516` CV fits and
87 final fits. Use the JSON supplied through `--search-space` to change the
`search` and `baseline` fields; the baseline must lie within the search space.
Use a new output directory after changing model, search, training, or generation
settings; fingerprint checks reject old caches. Moving the body/tail split to the
root changes source fingerprints and the meaning/order of probability outputs,
even though head dimensions match the previous terminal-tail model. Keep existing
outputs in their original directory and refit in a new directory; neither old
finite-support nor terminal-tail results/checkpoints describe this model.

## Part 2: simulation testing

1. By default, read configurations from
   `regularization/tuning/output/hyper_para/<method>.csv`. Use `--hyper-dir` to
   specify another directory without copying CSV files.
2. Restore hyperparameters, training_config, optimization, refit_epochs, and data
   settings. Check that tau/mu/r/n/base seed agree across methods and verify
   the numerical source-code fingerprint.
3. Generate new data for each case and replication under the existing testing
   protocol, with a random 80/20 split.
4. Train from scratch using the frozen hyperparameters and epoch counts, without
   new CV, hyperparameter selection, or test-based early stopping. The model
   initialization seed is `data_seed+1`; all methods use the same initialization seed.
5. Save one resumable JSON file per fit. Reuse successful results when the request
   matches; retry failed records.
6. When a worker finishes, check the shared JSON files and automatically generate
   SSE tables for complete cases. By default, publish only after all three methods
   finish. For a single-method run, use `--summary-methods` to specify which methods
   the summary waits for.

The default simulation budget is `29 * 3 * 500 = 43500` final fits.
SSE matches the existing evaluation: a uniform 256-point grid over
`[0, maximum finite endpoint in the training set]`, with
`mean_subjects(sum_grid((S_hat-S_true)**2))`. It is neither divided by the number
of grid points nor multiplied by the integration step size. Summary SD uses the
sample standard deviation (n−1). The true data-generating distribution is used
only to evaluate SSE.

## Running the workflow

The active Fir runtime is Python 3.14.2 / PyTorch 2.11.0 in `~/envs/pt`,
with Optuna 4.5.0 and R 4.4.0. See the
[current environment snapshot](../environment/fir-20261021/README.md) for all exact
package/build versions and [cross-server setup](../ENVIRONMENT_REPLICATION.md).

Run all commands below from the **NRT root directory**. The launchers load this
server's existing runtime by default; use the same environment for manual runs.
The existing source environment can be checked without changing installed packages:

```bash
cd /home/y679zhao/projects/def-anthree/y679zhao/NRT
source runtime.sh
"$PY" environment/verify.py
"$PY" -m unittest test_preflight test_node_runner -v
```

`regularization/requirements.txt` includes the complete root environment inventory.
On another server, recreate the recorded Python/R builds and native dependencies
before running these checks. Do not substitute a different core environment or add
an unrecorded `.deps` overlay when reproducing the current results.

Check task configurations without running training:

```bash
"$PY" -m regularization.tuning.main --dry-run
"$PY" -m regularization.testing.simulation run --dry-run
```

Submit the full tuning run, then submit testing after all tuning tasks succeed.
The shared-support fix covers the default CV folds for all 29 cases. Use a new
output directory for new experiments to avoid reusing caches from before the fix.
You can also tune and simulate a subset independently with `--cases`.

```bash
sbatch regularization/tuning/tune_nll.sh
# After the entire tuning array succeeds:
sbatch regularization/testing/run_nll.sh
```

You can also set an explicit success dependency (replace the job ID with the
actual tuning array ID):

```bash
sbatch --dependency=afterok:TUNING_JOB_ID regularization/testing/run_nll.sh
```

Tuning defaults to array=0..28, with one GPU handling all three methods for one case.
Testing defaults to array=0..2, with one GPU handling all cases and replications for
one method. When changing `CASES`, reduce the tuning array range accordingly; its
task-id is the index within the selected case list. The launchers reuse the original
`runtime.sh` and `node_runner.py` checks for CPU computation, temporary storage,
dependencies, and the selected GPU. Both penalty workflows requeue the failed
job/array element on precheck exit 75, with at most four requeues by default.
`PY`, `OUTDIR`, `HYPER_DIR`, `CASES`, `N_SAMPLES`, `BASE_SEED`, and `DEVICE` can
override defaults. Pass custom generation parameters through tuning's `--mu` / `--r`;
testing automatically reads the values saved in the CSV.

Single-case example (still using the full default search):

```bash
"$PY" -m regularization.tuning.main --cases ph_right_q0.2 --device cuda:0
"$PY" -m regularization.testing.simulation run --cases ph_right_q0.2 --device cuda:0
```

Run only adaptive:

```bash
METHODS=adaptive_nll_cv CASES=ph_right_q0.2 \
  sbatch --array=0 regularization/tuning/tune_nll.sh
SUMMARY_METHODS=adaptive_nll_cv CASES=ph_right_q0.2 \
  sbatch --array=1 regularization/testing/run_nll.sh
```

Small CPU validation run using new output directories. Subsequent testing must
use the same n and base seed:

```bash
"$PY" -m regularization.tuning.main --cases po_interval_q0.2 \
  --output-dir /tmp/nll_tuning_smoke --n-samples 128 --device cpu --threads 1 \
  --trials 2 --initial-trials 2 --folds 2 --epochs 3 --warmup-epochs 1 \
  --validation-every 1 --no-cuda-graphs
"$PY" -m regularization.testing.simulation run --cases po_interval_q0.2 \
  --hyper-dir /tmp/nll_tuning_smoke/hyper_para --output-dir /tmp/nll_testing_smoke \
  --n-samples 128 --replications 2 --device cpu --threads 1
```

Rebuild summaries manually without refitting:

```bash
"$PY" -m regularization.tuning.main --summarize
"$PY" -m regularization.testing.simulation summarize
# Publish incomplete simulation results only when explicitly requested; leave missing/failed cells blank:
"$PY" -m regularization.testing.simulation summarize --allow-incomplete
```

For subsets or custom output directories, pass the same `--cases`, `--methods`,
`--replications`, and `--output-dir` values as in the original run. Use `--replication`
or `--start-replication/--end-replication` to run in batches; the summary target
remains 1..replications.

## Outputs

```text
regularization/
├── model.py, train.py, workflow.py
├── tuning/
│   ├── main.py, tune_nll.sh
│   └── output/
│       ├── <method>/<case_id>/cv_results.csv
│       ├── <method>/<case_id>/selection.json, result.json, history.json
│       ├── hyper_para/<method>.csv
│       └── <method>_summary.csv
└── testing/
    ├── simulation.py, run_nll.sh
    └── output/
        ├── <method>/<case_id>/replication_0001.json ... replication_0500.json
        ├── tables/<case_id>.csv
        └── censoring_summary.csv
```

The hyperparameter CSV retains the original workflow's
`model_name,data_types,parameters,test_average_grid_sse,tau` fields and additionally
stores training settings, refit epochs, data configuration, and source/tuning
fingerprints. It is read only by the new regularization testing workflow and is separate from
the retained benchmark hyperparameter CSV files. Each case's SSE table has
500 replication rows plus `average` and `standard_deviation` rows.
`censoring_summary.csv` deduplicates by dataset, counting each case/replication
only once across the three methods.

Tuning stores no per-trial or per-fold files. Each completed candidate updates the
method's single `cv_results.csv`; the matched control has one CV row. Fold records
and histories stay in memory. The three retained JSON files describe the selected
configuration, final fitted result, and final refit's history.

During an unfinished search only, `studies/<case_id>/` contains a request manifest
and one SQLite database per searched method. Completed trials resume from that
database; an interrupted trial reruns its folds. Once the requested methods have
their final outputs, databases and temporary manifests are deleted. Small lock
files remain for concurrent-run protection. `--summarize` also compacts completed
cases from the old layout, moving their CV tables and removing fold caches.

Tuning and each simulation use write locks; result JSON and CSV files are replaced
atomically. After an interruption, resume with the same command: tuning reuses
completed selections/search trials, while testing reuses successful JSON results. Use a
new output directory when changing the protocol. Automatic summarization preserves
existing complete tables if results are missing or failed.

## Preflight and small validation runs

CPU/GPU startup checks are implemented by root `node_runner.py`, with regression
checks in `test_preflight.py` and `test_node_runner.py`. Body/tail likelihood,
boundary, gradient, and training checks are in `regularization/test_model.py`:
`"$PY" -m unittest regularization.test_model -v`. The small tuning and
simulation commands above can verify the full pipeline in fresh directories.
They do not reproduce the full 29-case, 500-replication experiment.
