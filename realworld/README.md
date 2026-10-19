# Real-world survival analysis: SUPPORT

This directory compares eight survival models on the SUPPORT data distributed with DeepSurv. It predicts survival from 14 covariates and evaluates those predictions on the original held-out test set.

**Status as inspected on October 18, 2026:** the directory contains the dataset and analysis code, but no `output/` directory or saved model results. The settings below describe the implemented full-run configuration. Selected hyperparameters, test scores, and model rankings are not yet available here. Dataset statistics were calculated directly from the bundled CSV; no model training was performed to write this document.

## 1. Data used

The input is [`data/support/support_deepsurv.csv`](data/support/support_deepsurv.csv). According to the provenance in [`data.py`](data.py), it is an exact CSV export of `experiments/data/support/support_train_test.h5` from `jaredleekatzman/DeepSurv`, commit `f0803f57bff9395f2bfbb99da2f1031adcf01331`. The recorded paper DOI is `10.1186/s12874-018-0482-1`.

| Partition | Observations | Observed deaths | Right-censored | Censoring rate | Follow-up range (days) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 7,098 | 4,823 | 2,275 | 32.05% | 3–2,029 |
| Test | 1,775 | 1,213 | 562 | 31.66% | 3–2,029 |
| Total | 8,873 | 6,036 | 2,837 | 31.97% | 3–2,029 |

The original train/test partition is preserved. The CSV has 18 columns:

| Columns | Meaning and use |
| --- | --- |
| `x00`–`x13` | Fourteen input covariates in the original HDF5 matrix order. The export does not provide clinical variable names; a verified mapping is needed before interpreting individual features clinically. |
| `time` | Observed event or censoring time, in days. |
| `event` | `1` for observed death; `0` for right censoring. |
| `split` | Original `train` or `test` membership; excluded from model inputs. |
| `row_in_split` | Source row identifier within the partition; excluded from model inputs. |

The export applies no additional imputation, filtering, relabeling, or scaling. All bundled numeric values are finite and present. The loader checks schema, positive times, binary event indicators, unique source row identifiers, partition sizes, and an allowlist of two verified SHA-256 checksums. The original recorded export checksum is:

```text
76916ee819f1e7610d1c4376ead3ef238877058607f5ca291fb3e6a1ba9bf734
```

The current CSV serialization has checksum:

```text
aac5a35ec6de18a1b39bb584f5e1aa22a7c6e3348a1115d8867229ca8a0b31e8
```

Its 14 feature columns, times, event indicators, and source row identifiers were verified exactly, for both partitions, against the [author HDF5 at the recorded commit](https://github.com/jaredleekatzman/DeepSurv/blob/f0803f57bff9395f2bfbb99da2f1031adcf01331/experiments/data/support/support_train_test.h5). Both verified byte representations are accepted; any other checksum is rejected. The actual input-file hash is saved in each run's manifest.

An observed death is encoded as `L = R = time`; right censoring is encoded as `L = time`, `R = infinity`.

NRT standardizes covariates with the fitting subset's means and population standard deviations, flooring standard deviations at `1e-6`. Final NRT fits use full-training statistics. DeepSurv performs its own training-data standardization. NNCDE standardizes expanded covariate-and-time inputs within each internal training half. Cox and the R tree/forest adapters receive the exported covariates without an additional shared preprocessing step.

## 2. Models used

| Method identifier | Model and purpose |
| --- | --- |
| `fixed_nll_cv` | NRT with fixed midpoint splits (`rho = 0.5`), giving equal-width time bins. Architecture is selected independently by validation negative log-likelihood (NLL). |
| `adaptive_nll_cv` | NRT with covariate-dependent time splits and a width regularizer. Architecture and regularization strength are selected by validation NLL. |
| `fixed_at_adaptive_architecture` | Fixed-split NRT using the adaptive model's selected architecture, with regularization strength zero and separate CV selection of refit epochs. This compares fixed and adaptive splits at the same architecture. |
| `DeepSurv` | Neural Cox proportional-hazards model using the embedded SUPPORT paper preset and a local adapter with Breslow handling of event-time ties. |
| `NNCDE` | Neural conditional survival estimator with a log-hazard network. Two networks exchange training/validation halves; their survival predictions are averaged. |
| `Cox` | Unpenalized `lifelines.CoxPHFitter`, using Efron ties for coefficient fitting and a Breslow baseline cumulative hazard. |
| `Ctree` | Conditional inference survival tree through the existing R `partykit`/`icenReg` backend. |
| `ICcforest` | Conditional inference survival forest through the existing R `ICcforest` backend, including training-only `mtry` tuning. |

NRT estimates a conditional distribution with constant density within each leaf interval. Its finite raw-time support ends at `1.1 × max(training follow-up) = 2,231.9 days`. This bound is fixed from the entire outer training pool and shared across CV folds; test outcomes do not determine it. Depth `d` gives `2^d` leaves. Adaptive intervals can change with the covariates, while fixed intervals remain equally spaced.

## 3. Parameters and model selection

### Shared settings

| Setting | Full-run value |
| --- | --- |
| Final fitting seed | `123` |
| CV fold seed | `321` |
| CV folds | `4`, using a seeded random permutation of training rows, without event stratification |
| NRT search seed | `871` |
| NRT fitting seed during search | `101` |
| Test partition | Original 1,775 rows, held out from fitting and selection |
| Main evaluation-grid size | `256`, before early-time points and landmarks are added |
| Default PyTorch threads | `4` |

NRT and NNCDE use CV selection. DeepSurv uses a fixed preset; Cox and Ctree use fixed settings; ICcforest tunes `mtry` internally on training data. Configuration is defined in [`tuning.py`](tuning.py), with inherited defaults from [`../regularization/tuning/main.py`](../regularization/tuning/main.py) and [`../tuning/main.py`](../tuning/main.py).

### NRT

Each independently searched NRT method receives **200 completed trials**, including 20 initial anchor/Sobol configurations followed by Optuna TPE proposals. Each candidate uses the same four training folds. Selection minimizes validation NLL averaged with weights proportional to fold size; the training regularizer is not added to that selection score.

| Parameter | Value or search range |
| --- | --- |
| Input dimension | `14` |
| Tree depth | Integers `3`–`8`, giving 8–256 leaves |
| Shared layers | Integers `0`–`3` |
| Shared width | `16`, `32`, `64`, or `128`; inactive if shared layers are zero |
| Private layers | Integers `1`–`4` |
| Private width | `16`, `32`, `64`, or `128` |
| Activation | SiLU in the shared trunk and between private feature layers |
| Adaptive regularization strength `s` | `0`–`10,000`, searched in `log10(1 + s)` coordinates |
| Fixed-model regularization strength | `0` |
| Split minimum / total width floor | `0.1` / `8e-4` |
| Optimizer / base learning rate | Adam / `0.001` |
| Split-branch learning-rate ratio | `0.1`, giving split learning rate `0.0001` |
| Batch size | `256` |
| Maximum CV epochs | `200` |
| Warmup | `10` epochs; adaptive splits activate afterward |
| Validation frequency | Every `5` epochs and at the final epoch |
| Early-stopping patience | `50` epochs |
| Minimum improvement for patience | `0.0001` |
| Gradient-norm limit | `1.0` |
| CUDA graphs | Enabled when running on CUDA |

The initial architecture anchor has depth 3, no shared layers, shared/private widths 32, and 3 private layers. The adaptive anchor strength is 500; initial adaptive configurations also include strengths 0, 1,000, and 10,000. These are starting points, not selected results.

For `n` fitting observations, the objective is `mean NLL + (s / sqrt(n)) × width penalty`. The effective strength comes from the architecture's `regularization_strength`; the inherited training-config field with the same name does not set this coefficient in `fit_raw_time_kl`.

Only post-warmup checkpoints are eligible. After selection, a full-training refit uses the ceiling of the median of the four best CV epochs. The matched fixed model reuses the adaptive architecture, sets `s = 0`, and performs its own four-fold epoch selection. In total, the full NRT procedure schedules `2 × 200 × 4 + 4 = 1,604` fold fits, followed by three final refits.

### DeepSurv

The preset in [`models.py`](models.py) records the SUPPORT Table-3 parameters; 800 epochs come from the referenced SUPPORT runner. The implementation uses the single-layer `[44]` architecture.

| Parameter | Value |
| --- | --- |
| Hidden layers | One layer with `44` units |
| Activation | `selu` |
| Dropout | `0.2553173828125` |
| Batch normalization / standardization | `False` / `True` |
| Optimizer | Adam, with default beta parameters |
| Initial learning rate | `0.047277902766138385` |
| Learning-rate decay | `0.00257333984375` |
| Learning-rate schedule | `initial_lr / (1 + epoch_index × lr_decay)`, with zero-based epoch index |
| L2 regularization | `8.1196630859375`, using the inherited output-weight L2 convention |
| L1 regularization | `0.0` |
| Epochs | `800`, with full-batch updates |
| Tuning | Disabled |
| Recorded momentum | `0.8589028320312501`; retained as preset metadata, not applied as SGD momentum to Adam |

Training minimizes the Breslow-ties Cox partial-likelihood loss plus regularization. Loss and learning rate are logged at epoch 1, every 50 epochs, and the final epoch.

### NNCDE

The exhaustive grid has `3 × 3 × 3 = 27` candidates:

| Parameter | Value or search range |
| --- | --- |
| Hidden nodes per layer | `64`, `128`, or `256` |
| Learning rate | `0.01`, `0.001`, or `0.0001` |
| Batch size | `64`, `128`, or `256` |
| Network | Two hidden ReLU layers and one log-hazard output |
| Input dimension | `16`: 14 covariates plus interval start and stop |
| Optimizer | Adam |
| Maximum epochs / patience | `200` / `20` per internal network |
| Minimum validation improvement | `1e-7` |
| Subject microbatch | `32`, accumulating gradients within the selected batch size |

Each candidate undergoes four-fold CV. Selection uses the unweighted mean of the four held-out NLL scores. Each fold fit trains two networks through internal two-half cross-fitting; held-out NLL averages their individual NLLs. This gives 108 candidate/fold fits, each containing two internal network fits.

For candidate `trial` and fold `fold`, both numbered from 1, the fold-fit seed is `321 + (trial - 1) × 4 + fold`. The final selected model uses seed 123 and retains internal two-half training/validation within the full 7,098-row training pool. Its predictions average the two survival curves. The memory-bounded adapter retains all training time intervals rather than subsampling them.

### Cox, Ctree, and ICcforest

| Model | Parameters |
| --- | --- |
| Cox | `penalizer=0.0`; Efron ties; Breslow baseline. Baseline predictions use a right-continuous step function. |
| Ctree | `mincriterion=0.95`, `minsplit=20`, `minbucket=7`, no explicit depth limit, `seed=123`. |
| ICcforest | `ntree=100`, `tune_mtry=True`, `tune_ntree=100`, `step_factor=1.5`, `seed=123`. |

The ICcforest R backend additionally uses `teststat="quad"`, `testtype="Univ"`, `mincriterion=0`, and `saveinfo=FALSE`. It sets `minsplit=max(2, floor(0.15 × n))` and `minbucket=max(1, floor(0.06 × n))`, giving 1,064 and 425 for the full training set. Infinite right endpoints are represented as `9,999,999` in that backend. The fitted R model uses the selected numeric `mtry`; Python's `selection.json` records the tuning settings rather than explicitly exporting that selected value.

## 4. Evaluation and reported results

Every model predicts survival for the same test subjects on the same grid. [`metrics.py`](metrics.py) checks prediction shape, finiteness, probability bounds, and nonincreasing survival over time.

The evaluation support is **day 0 through the maximum observed time in the training pool**, including both endpoints: `tau = max(training time) = 2,029 days`. Observed times include both deaths and right-censoring times. The test outcomes do not select or shorten this interval. The prediction/scoring grid combines 256 equally spaced points over `[0, tau]` with landmarks at 30, 90, 180, 365, and 730 days, removing duplicates. For the full dataset, this produces 261 prediction and Brier-scoring times. The NRT model support remains 2,231.9 days; that is a model-fitting bound, distinct from the evaluation endpoint.

| Metric | Meaning | Preferred direction |
| --- | --- | --- |
| `c_index` | Harrell's concordance using a common risk score: negative restricted mean survival time, calculated by trapezoidal integration from day 0 to the evaluation end. Higher risk means shorter predicted survival. | Higher |
| `c_index_native_risk` | Additional concordance using native log-risk predictions, available only for DeepSurv and Cox. | Higher |
| `integrated_brier_score` | Normalized integrated IPCW Brier score: `(1 / tau) × integral_0^tau BS_IPCW(t) dt`, calculated by trapezoidal quadrature. | Lower |
| `ipcw_brier_integral` | Unnormalized integral `integral_0^tau BS_IPCW(t) dt`, in days. Equals `tau × integrated_brier_score`. | Lower |
| `brier_30d`, `brier_90d`, `brier_180d`, `brier_365d`, `brier_730d` | IPCW Brier scores at the named landmarks; null outside the scoring interval. All five lie within the full-data interval. | Lower |
| `evaluation_start_days`, `evaluation_end_days` | Scoring interval endpoints. | Descriptive |
| `min_censoring_survival` | Minimum estimated censoring survival over the scoring grid. | Diagnostic |
| `elapsed_seconds` | Time for the current method execution, including selection work performed then, final fitting, evaluation, and artifact writing before saving the result. Reused caches affect this value. | Descriptive |

Inverse-probability-of-censoring weights (IPCW) use a reverse Kaplan–Meier censoring estimate fitted only on training outcomes. At time `t`, observed deaths with `T <= t` contribute `S(t)^2 / G(T)`, and observations with `T > t` contribute `(1 - S(t))^2 / G(t)`. Censored observations with `T <= t` contribute zero. The sum is divided by the full test sample size. This follows the [IPCW Brier formula](https://scikit-survival.readthedocs.io/en/v0.25.0/api/generated/sksurv.metrics.brier_score.html); the normalized time average follows the [integrated Brier definition](https://scikit-survival.readthedocs.io/en/v0.25.0/api/generated/sksurv.metrics.integrated_brier_score.html).

For this dataset, `G(2029) = 0`, and no test observation extends beyond day 2,029. The terminal control sum is therefore empty; the endpoint Brier score contains only the observed-event contributions with positive event-time weights. The implementation divides only where an indicator term contributes, avoiding undefined `0 / 0` arithmetic. It still raises an error if an observed event needs a zero `G(T)`, or a test observation extends past a scoring time with zero `G(t)`. No weights are clipped, no samples are removed, and the evaluation endpoint is not shortened. A reported `min_censoring_survival=0` is expected for the full SUPPORT interval under this convention.

The custom evaluator supports day 0 and the exact training maximum. The reference library's public scoring function restricts evaluation times to the interior of test follow-up; regression tests compare against that function on admissible interior points and separately verify the endpoint by hand calculation.

The workflow evaluates one original split and one final fitting seed. It does not produce confidence intervals, repeated-split estimates, bootstrap summaries, significance tests, or plots. Validation NLL is a selection diagnostic; no common test-NLL column is exported.

**Measured results are currently unavailable in this directory.** No numerical performance ranking can be inferred from the configuration alone. Once a full run finishes, report its values from `comparison.csv` and its selected architectures from the selection records described below.

## 5. Output files

The default full-run destination is `realworld/output/support/`, relative to the NRT repository root.

| File | Contents |
| --- | --- |
| `request.json` | Data/source fingerprints, full configuration, package versions, feature names, training/test indices, CV validation indices, evaluation grid, and NRT support bound. |
| `dataset_summary.csv` | Partition counts, events, censoring rates, and follow-up ranges. |
| `split_assignments.csv` | CSV row, original split, source row identifier, and training CV validation fold. |
| `status.csv` | Per-method status (`pending`, `failed`, or `ok`) and errors, written by the summarizer. |
| `comparison.csv` | Eight rows of model metrics, dataset/profile, sample sizes, seed, scoring bounds, censoring diagnostic, risk definition, and elapsed time. |
| `selected_parameters.csv` | Each method's selection record as a JSON string, plus the request hash. |
| `brier_curves.csv` | Combined per-time Brier scores and censoring survival, with method labels. |

The three combined files `comparison.csv`, `selected_parameters.csv`, and `brier_curves.csv` are written only after all eight methods succeed with matching request fingerprints. An incomplete summary writes `status.csv` and raises an error instead of publishing a new partial comparison.

Each method has its own subdirectory:

| Artifact | Contents / applicable models |
| --- | --- |
| `selection.json` | Selected or fixed parameters, selection criterion, and applicable CV scores/refit details. |
| `result.json` | Metrics, runtime, device, environment, and request hash on success; error and traceback on failure. |
| `predictions.npz` | Arrays `survival`, `times`, common `risk`, `csv_rows`, and `native_risk` (empty when unavailable). Full-data survival shape: `(1775, 261)`. |
| `brier_curve.csv` | Method, time in days, IPCW Brier score, and estimated censoring survival. |
| `model.pt` | NRT state dictionary; DeepSurv network/normalization/baseline state; or both NNCDE network/normalization states. Reconstruction also requires the matching model code and parameter records. |
| `history.json` | NRT epoch history or DeepSurv logged losses and learning rates. |
| `scaler.json` | NRT training-set feature means and standard deviations. |
| `crossfit_details.json` | NNCDE internal-network seeds, best epochs, validation NLLs, and epochs run. |
| `coefficients.csv`, `baseline_cumulative_hazard.csv` | Cox coefficient summary and baseline. |
| `model.rds` | Fitted Ctree or ICcforest R artifact. |
| `study.sqlite3`, `cv_results.csv`, `fold_cache/` | Search/CV records for independently tuned NRT models; NNCDE has CV CSV/cache records but no Optuna study. |
| `cv/` | Fold scores and histories for the fixed model at the adaptive architecture. |

## 6. Running the workflow

The scripts are configured for this Rorqual server and reuse NRT's existing `runtime.sh` and `/home/y679zhao/envs/nrt-runtime/bin/python`. They use absolute project/log paths and do not depend on your submission directory. The `log/.gitkeep` file preserves the log directory, which Slurm needs before a job starts.

Submit the full workflow from any directory:

```bash
sbatch /home/y679zhao/links/projects/def-anthree/y679zhao/NRT/realworld/submit_support.sh
```

This short CPU job checks dependencies and prepares the manifest, then submits the GPU and CPU arrays and an `afterok` summary job. The child job IDs and final CSV path are printed in `realworld/log/support_submit-<jobid>.out`. If preflight or preparation fails, no child jobs are submitted. The same driver can also be run with `bash` from the login node.

Alternatively, submit the arrays directly from the NRT root:

```bash
cd /home/y679zhao/links/projects/def-anthree/y679zhao/NRT
mkdir -p realworld/log
gpu=$(sbatch --parsable realworld/run_gpu.sh)
cpu=$(sbatch --parsable realworld/run_cpu.sh)
gpu=${gpu%%;*}
cpu=${cpu%%;*}
sbatch --dependency="afterok:$gpu:$cpu" realworld/summarize.sh
```

The direct array workers initialize the shared manifest safely themselves. For a dependency check before submitting, or to rebuild a completed summary without refitting:

```bash
cd /home/y679zhao/links/projects/def-anthree/y679zhao/NRT
source runtime.sh
"$PY" -m realworld.run preflight --device cpu
"$PY" -m realworld.run summarize --output-dir realworld/output/support
```

`NRT_ROOT`, `OUTDIR`, `DATA_CSV`, and `PY` remain overridable. When moving the project to a different server/path, update the `#SBATCH --chdir`, `--output`, and `--error` directives or override them on the `sbatch` command line; Slurm reads these directives before shell variables are evaluated.

The GPU array runs fixed NRT, adaptive NRT plus its matched fixed control, NNCDE, and DeepSurv. Each GPU task requests one H100 40 GB MIG instance (`--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1`), four CPUs, 32 GB host memory, and two days. The CPU array runs Cox, Ctree, and ICcforest, with four CPUs, 32 GB memory, and one day per task. Scheduler logs go to `realworld/log/`.

Python requirements include the existing PyTorch, NumPy, pandas, and SciPy environment plus `optuna==4.5.0`, `lifelines>=0.30,<0.31`, and `scikit-survival==0.25.0`; see [`requirements.txt`](requirements.txt). R requires `survival`, `partykit`, `icenReg`, and `ICcforest`. The scikit-survival pin keeps this server setup compatible with the existing NumPy 1.26.4 environment; newer releases in the former broad range can require NumPy 2.

A separate small validation run is available:

```bash
"$PY" -m realworld.run run --smoke --device cpu
```

The smoke profile uses 160 training and 80 test observations, two CV folds, two NRT trials, three NRT/NNCDE epochs, five DeepSurv epochs, and five forest trees. Its default destination is `realworld/output/support_smoke/`. Smoke outputs validate execution and are not full-data scientific results.

Completed results and tuning caches can be reused. The manifest checks data, source, configuration, and Python environment fingerprints; use a new output directory when these change. The full-training-support evaluation uses manifest schema version 2 and `evaluation_support="zero_to_max_observed_training_time"`; results from the previous quantile interval cannot be reused under this manifest.

## 7. Source map

| Source | Responsibility |
| --- | --- |
| [`data.py`](data.py) | Provenance, integrity checks, features, outcome bounds, and NRT scaling. |
| [`models.py`](models.py) | DeepSurv preset, Cox adapter, and memory-bounded NNCDE adapter. |
| [`tuning.py`](tuning.py) | Full/smoke configurations, CV selection, caching, and refit-epoch selection. |
| [`metrics.py`](metrics.py) | Grid construction, prediction validation, IPCW Brier scores, IBS, and concordance. |
| [`run.py`](run.py) | Preparation, fitting, prediction, artifact saving, and complete-result summary. |
| [`../regularization/train.py`](../regularization/train.py), [`../regularization/model.py`](../regularization/model.py) | NRT optimization, finite support, likelihood, and width penalty. |
| [`../testing/benchmark/`](../testing/benchmark/) | Shared neural and R benchmark implementations. |
