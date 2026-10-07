# NRT tuning and simulation

This implements [reademe.md](reademe.md) using the existing `testdata2.py` data generator and model implementations. Defaults are 5,000 observations, an 80/20 train/test split, 500 simulation replications, and 256 evaluation grid points.

There are 35 cases across five data settings:

- `ph`, `lognormal`, `po`, `multimodal_separate`, `multimodal_closer`;
- One `exact` case per setting, since q does not affect exact data;
- `right`, `current`, and `interval`, each with `q=0.2` and `q=0.5`.

This gives 5 exact cases plus 30 censored cases. Exact cases retain the existing `_exact_q0.2` identifier; q is ignored, their realized censoring rate is zero, and `tau` is null. For the other schemes, q targets the fraction with `R=inf`, including right censoring after the last inspection for interval/current-status data. The generator retains the notebook's two Weibull component shapes (6, 8), separate scales (1, 6), and closer scales (3, 6).

| Method | Applicable observation schemes | Cases per method |
| --- | --- | --- |
| NRT_NLL, NRT_Brier, NRT_2Loss | All four | 35 |
| NNCDE, DeepSurv | Exact, right | 15 |
| Deep distribution regression | Exact | 5 |
| Ctree, Cox, AFT | All four | 35 |
| ICcforest | Right, current, interval | 30 |

## Workflow algorithm

The workflow has two stages: select parameters once for each applicable model/case, then evaluate those fixed parameters on 500 newly generated datasets. A case is one combination of data setting, observation scheme, and q.

1. **Prepare tuning data.** Calibrate the population censoring horizon `tau` for the case, generate 5,000 observations, and split them into 4,000 training and 1,000 held-out test observations. The realized right-censoring rate is `mean(R == inf)` and can differ from q in a finite sample. Exact cases use `tau=null` and rate zero. All methods receive the same data and outer split for a given case.
2. **Select parameters with four-fold CV.** Split only the 4,000 training observations into four folds. For each candidate, fit on three folds (3,000 observations) and validate on the fourth (1,000), rotating through all four folds. Fit preprocessing and model support bounds using that fold's training observations. Select the candidate with the best mean validation criterion. The tuning test set is held aside throughout this search.
3. **Refit and evaluate.** Fit the selected configuration on all 4,000 training observations and evaluate on the held-out 1,000. Save the selected parameters, training configuration, CV score, test metrics, `tau`, and realized censoring rates. One GPU worker handles this entire search and refit for one model/case; candidates and folds run sequentially inside that worker.
4. **Run simulation replications.** For each case and replication, generate a fresh dataset with a seed distinct from tuning, retaining the saved `tau`. Share that dataset and its 80/20 split across applicable methods. Load tuned parameters for NRT, NNCDE, and DeepSurv; use the existing defaults/internal tuning for the CPU benchmarks. Fit each method on the training partition and evaluate on the held-out partition. NRT and NNCDE use a 200-epoch limit by default; DeepSurv uses the fixed paper Gaussian settings and 1,000 epochs.
5. **Summarize automatically after completion.** Tuning summaries collect one selected result per model/case. Testing workers check the shared result JSONs when they finish and publish a table for each complete case, with 500 replication rows and one column per applicable method, followed by means and sample standard deviations. Once all selected cases finish, a separate censoring summary counts each case/replication dataset once across methods to check whether its average realized rate follows q.

NRT_NLL trains with joint NLL and selects hyperparameters by the lowest mean validation **NLL**. NRT_Brier trains with joint Brier loss and NRT_2Loss uses alternating training; both select hyperparameters by the lowest mean validation **Brier score**. NNCDE selects the lowest validation **NLL**; DeepSurv selects the highest validation **Harrell C-index**. These CV criteria have different meanings and scales. The common comparison metric is the held-out survival-grid SSE against the known generating survival curve:

```text
average_grid_sse = (1 / n_test) * sum_i sum_g (S_hat(t_g | X_i) - S_true(t_g | X_i))^2
```

The default evaluation grid has 256 equally spaced points from zero to the largest finite training observation bound. SSE is averaged over test subjects and summed over grid points. NRT also reports held-out observed-data NLL and Brier score.

## Tuning

Run from this directory. The scripts use `/home/y679zhao/envs/pt/bin/python3.14` by default; override with `PY`. Slurm logs go into the existing `tuning/log` and `testing/log` directories. These commands submit the three tuning launchers:

All tuning and simulation launchers load `runtime.sh` from `/home/y679zhao/projects/def-anthree/y679zhao/NRT` under Slurm, since Slurm executes a copied script in its spool directory. Local Bash runs discover the project from the launcher's location. Set `NRT_ROOT=/path/to/NRT` to use another project copy in either mode.

```bash
cd /home/y679zhao/projects/def-anthree/y679zhao/NRT
export PY=${PY:-/home/y679zhao/envs/pt/bin/python3.14}
sbatch tuning/tune_nrt.sh
sbatch tuning/tune_nncde.sh
sbatch tuning/tune_deepsurv.sh
```

| Tuning script | Slurm array | Work per task | Total tasks |
| --- | --- | --- | --- |
| `tuning/tune_nrt.sh` | `0-104%105` | One of 35 cases for one NRT loss | 105 |
| `tuning/tune_nncde.sh` | `0-14%15` | One of 15 exact/right-censored cases | 15 |
| `tuning/tune_deepsurv.sh` | `0-14%15` | One of 15 exact/right-censored cases | 15 |

Each task owns one GPU and runs the complete CV search, final fit, and test evaluation for one case in one model. NRT tasks 0–34 tune NRT_NLL, 35–69 tune NRT_Brier, and 70–104 tune NRT_2Loss. The benchmark arrays contain only supported cases. The concurrency limits allow all 135 tasks to run together when cluster resources are available. The tuning and testing scripts request one `nvidia_h100_80gb_hbm3_2g.20gb` GPU; use `sbatch --gpus=<cluster_gpu_type>:1 ...` to select a different GPU type supported by the cluster.

Task IDs use the supported case order printed by `--list-cases` for that method, starting at zero. For NNCDE and DeepSurv, each data setting occupies three consecutive IDs: exact, right q=0.2, and right q=0.5. For example, task 0 is `ph_exact_q0.2`, task 3 is `lognormal_exact_q0.2`, and task 14 is `multimodal_closer_right_q0.5`. NRT retains seven cases per setting and three blocks of 35 tasks.

`--cases`/`CASES` and `--nrt-methods` filter this fixed matrix; excluded tasks exit without fitting, and filters do not renumber task IDs. Outside Slurm, set `TUNING_TASK_ID` or pass `--task-id` to run one task, for example `TUNING_TASK_ID=35 bash tuning/tune_nrt.sh --dry-run`. Without a task ID, a local tuning invocation processes the requested cases sequentially. The existing local `NRT_TASK_ID=0`, `1`, or `2` loss selector is also retained when no array task ID is set.

Every case uses one generated dataset and four-fold CV within its 80% training partition. All methods share the same case data and outer split. Model support bounds and preprocessing in each CV fit use only that fold's training observations. The held-out 20% is evaluated only after parameter selection.

- NRT grid search evaluates the complete Cartesian grid for each loss separately, in a deterministic order. NRT_NLL selects the lowest mean validation NLL; NRT_Brier and NRT_2Loss select the lowest mean validation Brier score. Edit `NRT_SEARCH_SPACE` in `tuning/main.py` or use `--search-space file.json`. Explicit choices currently use depth `[7, 8]`, h_dim and mlp_hidden `[8, 16, 32]`, and mlp_layers and shared_layers `[1, 2, 3]`. This grid has 162 configurations. The learning rate is 0.001, split_min is 0.1, dropout is zero, and the epoch limit is 200.
- NNCDE retains the previous exhaustive grid: hidden_nodes 64/128/256, learning_rate 0.01/0.001/0.0001, batch_size 64/128/256. It selects validation NLL, retaining the estimator's internal two-way cross-fitting. The epoch limit is 200. A search-space JSON supplies lists of choices for these three fields.
- DeepSurv retains the existing Optunity Sobol search, parameter ranges, and validation concordance criterion, with four folds and 50 evaluations. Settings outside the search come from `PAPER_GAUSSIAN_PARAMETERS`: SELU, no batch normalization, no standardization, zero L1 penalty, Adam, and 1,000 epochs for both CV and the final fit.

By default, each NRT grid search performs 162 × 4 = 648 CV fits, followed by one final fit. NNCDE evaluates 3 × 3 × 3 = 27 configurations, giving 108 CV estimator fits and one final estimator fit; each estimator retains its internal two-way cross-fitting. DeepSurv requests 50 candidate evaluations with four-fold CV and then fits the selected configuration on the full training partition.

NRT grid search and NNCDE always evaluate every grid combination; reduce their choice lists to run smaller searches. `TUNING_EVALS=...` or `--tuning-evals ...` controls DeepSurv's candidate count, which defaults to 50. Grid searches ignore this budget. NRT results from the previous random search have different request hashes; retune with `--force` or a new output directory.

Edit the tuning search spaces in these files:

| Method | File to edit | Setting |
| --- | --- | --- |
| All three NRT losses | [tuning/main.py](tuning/main.py) | `NRT_SEARCH_SPACE`: lists of integer choices for `depth`, `h_dim`, `mlp_hidden`, `mlp_layers`, `shared_layers` |
| NNCDE | [tuning/main.py](tuning/main.py) | `NNCDE_GRID`: lists of choices for `hidden_nodes`, `learning_rate`, `batch_size` |
| DeepSurv | [tuning/benchmark/deepsurv/cuda_model.py](tuning/benchmark/deepsurv/cuda_model.py) | The ranges passed to `optunity.maximize()` inside `_tune()`; keep the `OFFICIAL_TUNING_PARAMETERS` reference metadata synchronized |

For NRT and NNCDE, you can also create a JSON file with the same keys and pass `--search-space /path/to/file.json` to their tuning launchers. Both use lists of grid choices. For NRT, `[8, 16, 20, 31]` means four choices; `[8, 20]` means two choices. Each NRT list must contain distinct integers; `shared_layers` allows zero and other fields must be positive. For example:

```bash
sbatch tuning/tune_nrt.sh --search-space /path/to/nrt_grid.json
sbatch tuning/tune_nncde.sh --search-space /path/to/nncde_grid.json
```

The shared Python epoch defaults are in `DEFAULT_EPOCHS` in [workflow.py](workflow.py). The shell launcher defaults are in `nrt_tune()` in [runtime.sh](runtime.sh). Override tuning epochs with `EPOCHS=...` or `--epochs ...`.

`TUNING_EVALS`, `EPOCHS`, `N_SAMPLES`, `CASES`, `DEVICE`, `OUTDIR`, `PY`, and `TUNING_TASK_ID` can be set in the tuning launcher environment. Slurm supplies `SLURM_ARRAY_TASK_ID` automatically. CLI arguments passed to a launcher override its defaults. List case IDs with:

```bash
"${PY:-/home/y679zhao/envs/pt/bin/python3.14}" -m tuning.main --method NRT --list-cases
"${PY:-/home/y679zhao/envs/pt/bin/python3.14}" -m tuning.main --method NNCDE --list-cases
"${PY:-/home/y679zhao/envs/pt/bin/python3.14}" -m tuning.main --method DeepSurv --list-cases
```

Each tuning result is saved as `tuning/output/<method>/<case_id>/result.json`, containing the selected parameters, training configuration, CV criterion, test metrics, calibrated `tau`, and realized censoring rates for the complete dataset and both partitions. Fold results and each worker's `summary.csv` are saved alongside it. `tuning/output/cases/<case_id>.json` records the common generator settings and `tau`. Data seeds, splits, and search configurations remain identical to sequential tuning.

Tuning workers automatically publish each method's combined metrics and hyperparameter table once all selected cases for that method are complete. Each NRT loss publishes independently; the combined NRT metrics table is published when all selected losses finish. To rebuild summaries from existing results, without fitting or requiring CUDA:

```bash
PY=${PY:-/home/y679zhao/envs/pt/bin/python3.14}
"$PY" -m tuning.main --method NRT --summarize
"$PY" -m tuning.main --method NNCDE --summarize
"$PY" -m tuning.main --method DeepSurv --summarize
```

Use matching `--output-dir`, `--cases`, and `--nrt-methods` arguments for a subset run. Manual summarization requires all requested applicable results. Automatic checks leave existing tables untouched until the selection is complete and publish complete tables atomically. A filtered run replaces tables of the same name with that selection; use separate output directories to retain summaries for different selections. Testing reads the exported hyperparameter CSVs. Summaries and examples of their columns are described below.

NRT_NLL tuning resumption and summarization require `cv_metric=nll`. Results with a different CV metric must be retuned. Retune only NRT_NLL with `sbatch --array=0-34%35 tuning/tune_nrt.sh --nrt-methods NRT_NLL --force`, or use a new output directory.

## Simulation

Prepare the tuned CSVs in `testing/hyper_para` before submitting neural testing. CPU testing reads the saved case manifests in `tuning/output/cases`. Testing reuses saved `tau` without recalibrating the censoring horizon.

NRT and NNCDE simulation fits use a 200-epoch limit by default. DeepSurv retains saved CSV parameters and uses `PAPER_GAUSSIAN_PARAMETERS` defaults for unspecified settings, including Adam and 1,000 epochs. Pass `--epochs ...` to override the limit for a testing run. Changed configurations require `--force` or another output directory for existing results.

```bash
sbatch testing/run_nrt.sh
sbatch testing/run_nncde.sh
sbatch testing/run_deepsurv.sh
sbatch testing/run_cpu.sh
```

`run_nrt.sh` creates three concurrent GPU array tasks, one per NRT loss. Each task fits its 500 replications sequentially and runs all 35 cases in every replication. `run_nncde.sh` and `run_deepsurv.sh` each run 500 replications sequentially on one GPU, using their 15 supported cases.

`run_cpu.sh` creates a 500-task CPU array. Each task owns one replication and eight CPU cores. It first runs the 110 applicable combinations of deep distribution regression, Ctree, Cox, and AFT through an eight-process pool, using fresh worker processes for each fit. After that pool closes, it runs the 30 ICcforest cases sequentially so the R forests have the allocation's memory to themselves. Each fitting process uses one CPU thread. Override `--cpus-per-task` at submission, or lower concurrency with `PARALLEL_FITS`. `ICCFOREST_PARALLEL_FITS` (or `--iccforest-parallel-fits`) controls only the separate forest phase and defaults to 1. R is loaded with `R_MODULE=r/4.4.0`; required packages are `survival`, `icenReg`, `partykit`, and `ICcforest`. AFT uses exponential for PH, lognormal for lognormal, log-logistic as a working model for bounded PO, and one Weibull family for either mixture setting. ICcforest retains its internal mtry tuning; the other CPU methods use their existing defaults.

Simulation reads `testing/hyper_para/<method>.csv` for the three NRT losses, NNCDE, and DeepSurv. The `data_types` column identifies the case, `parameters` contains its JSON model configuration, and `tau` supplies the saved censoring horizon (empty for exact data). CPU benchmarks read `tau`, `mu`, and `r` from `tuning/output/cases/<case_id>.json` and do not require neural CSVs. Neural CSV horizons must agree across methods. Copy the tuning CSV exports into this directory when using new tuning results. NRT training defaults retain learning rate 0.001; DeepSurv uses paper defaults for settings absent from its CSV. For neural methods, generator settings default to `mu=0`, `r=None`; pass `--mu` and `--r` if needed to match tuning. CPU methods reuse the saved generator settings from the case manifests.

Override input/output roots with `HYPER_DIR` (neural CSVs), `TUNING_DIR` (tuning output containing CPU case manifests), and `OUTDIR`. `REPLICATIONS`, `N_SAMPLES`, `CASES`, `BASE_SEED`, and `GRID_SIZE` configure testing. If changing the CPU replication count, also change its Slurm array, for example `REPLICATIONS=10 sbatch --array=1-10 testing/run_cpu.sh`. All testing launchers use `SUMMARY_METHODS=all` by default so their comparison tables combine the separate CPU and GPU jobs. For a subset run, set `SUMMARY_METHODS` to the same comma-separated method list in every participating launcher, or pass `--summary-methods ...`. Direct Python runs default to summarizing their `--methods` selection.

All methods receive identical data and splits for a given case and replication. Simulation seeds differ from tuning seeds. Each successful result includes truth-based average survival-grid SSE, runtime, `tau`, and censoring rates; NRT also reports observed-data NLL and Brier score. Each invocation also prints its average censoring rate and writes a CSV named `censoring_<methods>_rep<first>_<last>.csv`. GPU launchers therefore report the average over their 500 replications automatically. Results are saved atomically as `testing/output/<method>/<case_id>/replication_0001.json`. Identical successful fits are skipped on rerun. Failed fits record their errors and are retried on rerun. A changed configuration requires another output directory or explicit `--force`.

Combined case tables are created automatically once all their applicable methods and replications succeed. Missing or failed results leave existing case tables untouched. The last worker also publishes `censoring_summary.csv` once all selected cases are complete. Writes use atomic replacement, including concurrent finishers and resumed workers. To rebuild tables manually from existing results:

```bash
"${PY:-/home/y679zhao/envs/pt/bin/python3.14}" -m testing.simulation summarize
```

`tables/<case_id>.csv` contains `average_grid_sse` for each replication, with a `replication` index and one column per applicable method. By default each table has 500 replication rows, then `average` and `standard_deviation` rows (sample standard deviations), for 502 rows total. `censoring_summary.csv` reports `tau`, average realized right-censoring rate, its standard deviation, and the difference from q, counting each replication once across methods. Manual summaries require all requested results unless `--allow-incomplete` is supplied; partial tables preserve all replication rows and leave missing/failed values blank. Use matching `--cases`, `--methods`, `--replications`, and `--output-dir` when rebuilding a subset summary.

## Output folders and files

Paths below are relative to the NRT directory. Output folders are created when results are written. Simulation comparison tables are saved in `testing/output/tables/`; tuning summaries and censoring reports are saved directly in their output roots. Each tuning worker also saves a one-row summary inside its own model/case folder.

```text
NRT/
├── tuning/
│   ├── log/
│   │   ├── tune_nrt-<job_id>_<task_id>.out
│   │   ├── tune_nrt-<job_id>_<task_id>.err
│   │   └── ...                         # NNCDE/DeepSurv logs too
│   └── output/
│       ├── cases/
│       │   └── <case_id>.json          # Shared data settings, tau, rates
│       ├── <method>/
│       │   └── <case_id>/
│       │       ├── result.json        # Selected parameters and evaluation
│       │       ├── cv_results.csv     # NRT/NNCDE candidate/fold results
│       │       ├── cv_results.json    # DeepSurv search history instead
│       │       └── summary.csv        # One row for this array worker
│       ├── NRT_NRT_NLL_NRT_Brier_NRT_2Loss_summary.csv
│       ├── hyper_para/              # Automatic selected-parameter tables, one per method
│       ├── NNCDE_NNCDE_summary.csv
│       └── DeepSurv_DeepSurv_summary.csv
└── testing/
    ├── hyper_para/
    │   └── <method>.csv             # Simulation parameters and common tau
    ├── log/
    │   └── ...                         # Simulation stdout/stderr
    └── output/
        ├── <method>/
        │   └── <case_id>/
        │       ├── replication_0001.json
        │       ├── ...
        │       └── replication_0500.json
        ├── censoring_<methods>_rep<first>_<last>.csv
        ├── tables/
        │   └── <case_id>.csv        # 500 replication rows, average, sample SD
        └── censoring_summary.csv
```

`<method>` is one of `NRT_NLL`, `NRT_Brier`, `NRT_2Loss`, `NNCDE`, `DeepSurv`, `deep_distribution_regression`, `ctree`, `iccforest`, `cox`, or `aft`. `<case_id>` is, for example, `ph_right_q0.2`. There are 135 tuned model/case results and 35 common case manifests in a complete tuning run. A complete simulation run has 275 applicable model/case combinations × 500 replications = 137,500 replication JSON files.

| File | Contents and purpose |
| --- | --- |
| `tuning/output/cases/<case_id>.json` | Case definition, generator parameters, seed, sample/split sizes, `tau`, and complete/train/test right-censoring rates; shared by all methods |
| Tuning `result.json` | Selected `parameters`, `training_config`, `cv_metric`, `cv_score`, held-out `average_grid_sse`, NRT `test_nll`/`test_brier`, elapsed time, common data metadata, and request/hash for resumption; exported to hyperparameter CSVs |
| `testing/hyper_para/<method>.csv` | Saved model configurations and `tau`, indexed by `data_types`; simulation input |
| `testing/output/tables/<case_id>.csv` | Per-replication `average_grid_sse` in method columns, followed by means and sample standard deviations |
| NRT/NNCDE `cv_results.csv` | One row per candidate/fold: `trial`, `fold`, candidate hyperparameters, `validation_score`, and `error`; 400 rows for a default NRT search and 108 for NNCDE |
| DeepSurv `cv_results.json` | Search ranges, sampler, evaluation/fold/epoch settings, selected parameters, best CV C-index, and individual fold fit records |
| Simulation `replication_XXXX.json` | Model/case/replication identifiers, seed, fixed parameters and `tau`, sample sizes and rates, metrics, elapsed time, `status`, and any fitting `error` |
| Per-invocation `censoring_<methods>_rep<first>_<last>.csv` | One row per model/case with the successful replication count, average realized right-censoring rate, `tau`, and difference from q for that invocation; a CPU array task covers one replication, whereas a full GPU invocation covers 500 |

For custom roots, set `OUTDIR` when submitting each stage, `HYPER_DIR` for neural simulation input, and `TUNING_DIR` for CPU case manifests. The corresponding direct Python options are `--output-dir`, `--hyper-dir`, and `--tuning-dir`. Pass these paths explicitly to summarization as well; direct Python commands do not automatically read the launchers' `OUTDIR` environment variable. For example:

```bash
OUTDIR=/path/to/tuned sbatch tuning/tune_nncde.sh
# After that array completes:
"$PY" -m tuning.main --method NNCDE --output-dir /path/to/tuned --summarize
HYPER_DIR=/path/to/tuned/hyper_para OUTDIR=/path/to/simulation SUMMARY_METHODS=NNCDE sbatch testing/run_nncde.sh
# Optional: rebuild the automatically generated tables:
"$PY" -m testing.simulation summarize --methods NNCDE \
  --output-dir /path/to/simulation --replications 500
```

## Summary tables

The examples below use illustrative numbers to show the table layouts. They are not production results. CSVs preserve the stored numerical values; previews here round them for readability. JSON `null` values appear as empty CSV cells.

### Tuning summaries

The three tuning `--summarize` commands shown above read the completed `result.json` files and write these tables:

| Combined file in `tuning/output/` | Rows in a complete run |
| --- | --- |
| `NRT_NRT_NLL_NRT_Brier_NRT_2Loss_summary.csv` | 105: one row per loss/case |
| `NNCDE_NNCDE_summary.csv` | 15: one row per supported case |
| `DeepSurv_DeepSurv_summary.csv` | 15: one row per supported case |

All tuning metric summary CSVs, including each worker's one-row `summary.csv`, have exactly these columns:

```csv
method,case_id,tau,right_censoring_rate,cv_metric,cv_score,average_grid_sse
```

| method | case_id | tau | right_censoring_rate | cv_metric | cv_score | average_grid_sse |
| --- | --- | --- | --- | --- | --- | --- |
| NRT_NLL | ph_right_q0.2 | 4.5741 | 0.1980 | nll | 1.1000 | 2.4000 |
| NRT_Brier | ph_right_q0.2 | 4.5741 | 0.1980 | brier | 0.1050 | 2.2000 |
| NRT_2Loss | ph_right_q0.2 | 4.5741 | 0.1980 | brier | 0.1080 | 2.3000 |

NRT_NLL records `cv_metric=nll`; NRT_Brier and NRT_2Loss record `cv_metric=brier`. Each benchmark's table uses the same schema, with `cv_metric=nll` for NNCDE and `cv_metric=c_index` for DeepSurv. `cv_score` is the selected candidate's mean CV score. `average_grid_sse` is the final fit's held-out test SSE from the single tuning dataset. `right_censoring_rate` refers to the complete tuning dataset. Train/test censoring rates remain in `result.json`. These tables contain one tuning result per model/case and do not average over simulation replications.

Selected hyperparameters are automatically exported to `tuning/output/hyper_para/<method>.csv` for NRT_NLL, NRT_Brier, NRT_2Loss, NNCDE, and DeepSurv. A complete run has 35 rows per NRT loss and 15 rows per benchmark. These tables extend NRT_2Loss's compact layout with the saved censoring horizon:

```csv
model_name,data_types,parameters,test_average_grid_sse,tau
```

`data_types` is the NRT case ID, `parameters` is a JSON object containing the selected model parameters, and `test_average_grid_sse` is the same held-out score as in the metric table. `tau` is the saved tuning horizon reused by testing without recalibration; it is empty for exact cases. Copy these CSVs into `testing/hyper_para`, or point `--hyper-dir`/`HYPER_DIR` at the export directory. Testing checks that CSV horizons agree across methods. Each NRT loss also gets its own combined tuning metric table, such as `NRT_NRT_NLL_summary.csv`, when its selected cases finish.

### Simulation performance summary

After all four simulation launchers finish, run:

```bash
"${PY:-/home/y679zhao/envs/pt/bin/python3.14}" -m testing.simulation summarize
```

This writes 35 CSVs in `testing/output/tables/`, one per data case. Each has 500 replication rows, applicable methods as columns, and two final rows named `average` and `standard_deviation`. Values are held-out `average_grid_sse`.

For example, `ph_right_q0.2.csv` has this header:

```csv
replication,NRT_NLL,NRT_Brier,NRT_2Loss,NNCDE,DeepSurv,ctree,iccforest,cox,aft
```

Exact tables include deep distribution regression and omit ICcforest; current/interval tables omit NNCDE, DeepSurv, and deep distribution regression. Method order follows `--methods`.

Means and SDs use successful results only. SD uses the sample definition with denominator `n_completed - 1` and is zero for one successful result. The command refuses to write tables if requested results are missing or failed; use `--allow-incomplete` for an explicitly partial report. Partial tables preserve replication numbers and method columns, leaving missing/failed cells blank. Rerun the relevant simulation launcher to retry failed fits or complete missing results; matching successful results are reused.

### Simulation censoring summary

Automatic summarization and the manual simulation summarization command write `testing/output/censoring_summary.csv`, with 35 rows in a complete run, one per data case. Methods share the same generated data, so each case/replication contributes once, rather than once per model. The summary checks that metadata agree across methods before combining them.

Its columns are:

```csv
case_id,setting,censor_type,q,tau,n_replications,average_right_censoring_rate,sd_right_censoring_rate,difference_from_q
```

| case_id | setting | censor_type | q | tau | n_replications | average_right_censoring_rate | sd_right_censoring_rate | difference_from_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ph_exact_q0.2 | ph | exact | 0.2 | | 500 | 0.0000 | 0.0000 | |
| ph_right_q0.2 | ph | right | 0.2 | 4.5741 | 500 | 0.2010 | 0.0060 | 0.0010 |
| ph_right_q0.5 | ph | right | 0.5 | 1.5000 | 500 | 0.4990 | 0.0070 | -0.0010 |

`difference_from_q = average_right_censoring_rate - q` for censored cases. Exact cases retain q=0.2 in their identifier for compatibility, but q does not control exact data; their `tau` and difference cells are empty. In a partial run, `n_replications` counts distinct replications with a successful result from at least one selected method, so it can differ from an individual model's `n_completed`.

For a subset summary, supply the same case/method selection, output root, and replication count used in the run. For example:

```bash
"$PY" -m testing.simulation summarize --methods NRT_NLL,NNCDE \
  --cases ph_right_q0.2 --replications 500 --output-dir testing/output
```

This subset command writes both combined tables in the specified output root, replacing previous combined tables there. Run each combined summarization once after the relevant jobs complete.

## Small checks and local runs

Preview scheduling without fitting or requiring CUDA:

```bash
bash tuning/tune_nrt.sh --dry-run
TUNING_TASK_ID=34 bash tuning/tune_nrt.sh --dry-run
TUNING_TASK_ID=104 bash tuning/tune_nrt.sh --dry-run
TUNING_TASK_ID=2 bash tuning/tune_nncde.sh --dry-run
TUNING_TASK_ID=14 bash tuning/tune_deepsurv.sh --dry-run
bash testing/run_nrt.sh --dry-run
bash testing/run_cpu.sh --dry-run
```

Outside Slurm, `run_nrt.sh` starts one process on each of `cuda:0`, `cuda:1`, and `cuda:2`. Set `NRT_TASK_ID=0`, `1`, or `2` to run just one loss on `DEVICE` instead. Direct Python commands permit `--device cpu` for small smoke tests; production GPU launchers default to CUDA and fail if no GPU is available.

```bash
PY=/home/y679zhao/envs/pt/bin/python3.14
cat > /tmp/nrt-smoke-grid.json <<'JSON'
{"depth":[2],"h_dim":[8],"mlp_hidden":[8],"mlp_layers":[1],"shared_layers":[1]}
JSON
"$PY" -m tuning.main --method NRT --nrt-methods NRT_NLL \
  --cases ph_right_q0.2 --device cpu --threads 1 --n-samples 64 \
  --epochs 2 --grid-size 16 --search-space /tmp/nrt-smoke-grid.json --output-dir /tmp/nrt-tuning
"$PY" -m testing.simulation run --methods NRT_NLL --cases ph_right_q0.2 \
  --device cpu --n-samples 64 --replications 2 --epochs 2 --grid-size 16 \
  --hyper-dir /tmp/nrt-tuning/hyper_para --output-dir /tmp/nrt-testing
"$PY" -m testing.simulation summarize --methods NRT_NLL --cases ph_right_q0.2 \
  --replications 2 --output-dir /tmp/nrt-testing
"$PY" -m unittest test_workflow
```

## Node checks and requeue

All tuning and testing launchers run a Python environment check on the allocated node through `srun` in Slurm (directly for local runs). The check imports Torch, NumPy, and pandas, with four attempts 15 seconds apart. Before fitting, both Python workflows execute a small calculation on the requested CPU or GPU; CUDA checks also synchronize and report the GPU name and compute capability. They import the selected model backends, and CPU simulations verify the required R packages before starting any workers. DeepSurv tuning also checks Optunity. Dry runs, case listings, and summaries skip execution/backend preflight.

Every launcher enables `--requeue` and appends logs across restarts. Python environment failures after the retries, or GPU/CPU node preflight failures, request requeue of only the current Slurm job or array element, up to four restarts. Set `NRT_PYTHON_CHECK_ATTEMPTS`, `NRT_PYTHON_CHECK_DELAY_SECONDS`, and `NRT_MAX_PYTHON_REQUEUES` to override these defaults; `NRT_MAX_PYTHON_REQUEUES=0` disables automatic requeue. Local runs exit on failure. Ordinary fitting errors do not request automatic requeue. On restart, valid completed tuning results and successful simulation fits are reused unless `--force` is supplied.
