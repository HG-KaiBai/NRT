# GPU tuning

See [the workflow guide](../README.md) for the case matrix, search spaces, output schema, and examples.

Each NRT loss tunes 35 cases: five exact cases and 30 censored cases. NNCDE and DeepSurv each tune 15 applicable cases. Exact data is fitted once per setting, since q does not affect it.

Submit `tune_nrt.sh`, `tune_nncde.sh`, and `tune_deepsurv.sh`. NRT tunes each loss separately by exhaustive grid search; NNCDE retains its grid/NLL selection and DeepSurv its Sobol/concordance selection. All use an 80/20 outer split and four folds in the training partition. Results and the numeric censoring horizon are consumed directly by `../testing/simulation.py`.

All three NRT variants select the lowest mean validation Brier score and record `cv_metric=brier`. NRT_NLL continues to train and monitor early stopping with NLL. Tuning resumption and summarization reject previous NRT_NLL results selected by validation NLL; retune them with `--force` or a new output directory.

NRT submits 105 GPU array tasks, with 35 concurrent tasks per loss. NNCDE and DeepSurv each submit 15 tasks containing only their supported exact/right-censored cases. Every worker owns one GPU and performs all CV fits for one model/case. NRT task IDs 0–34, 35–69, and 70–104 select NRT_NLL, NRT_Brier, and NRT_2Loss respectively. Case positions follow `--list-cases` for that method, starting at zero. Each benchmark has three cases per data setting, ordered exact, right q=0.2, right q=0.5. Filters retain these task IDs.

Preview a single worker with `TUNING_TASK_ID=35 bash tune_nrt.sh --dry-run`, or pass `--task-id` directly to `python -m tuning.main`. Slurm sets the task ID automatically. Local runs without a task ID process the requested cases sequentially.

Each worker writes `output/<method>/<case_id>/summary.csv`, then checks all selected cases for its method. Once they are complete, it automatically publishes a combined metric summary and `output/hyper_para/<method>.csv` with the selected parameters, held-out test SSE, and saved censoring horizon `tau`. This extends NRT_2Loss's four-column layout with a `tau` column; exact cases leave it empty. Testing uses the saved tuning horizon for every replication. Each NRT loss publishes independently; the combined NRT table is also published when all selected losses finish. Writes are atomic, and incomplete selections leave existing tables untouched. This applies to NRT, NNCDE, and DeepSurv, including resumed workers.

To rebuild existing results, run `python -m tuning.main --method NRT --summarize`; repeat for NNCDE and DeepSurv. Pass the same output directory and case/method filters used for tuning. This command requires all requested results and does not fit models or require CUDA. A filtered run publishes tables for that selection, replacing previous tables of the same name; use separate output directories to retain summaries for different selections.

NRT and NNCDE use 200 epochs by default; DeepSurv uses 1,000. DeepSurv's fixed CV and final-fit settings come from `PAPER_GAUSSIAN_PARAMETERS`: SELU, no batch normalization, no standardization, zero L1 penalty, and Adam. Override tuning epochs with `EPOCHS=...` or `--epochs ...`.

NRT grid search evaluates every parameter combination per loss/case (four CV fits per combination plus a final fit). NNCDE also evaluates its complete grid. DeepSurv defaults to 50 candidate evaluations; `TUNING_EVALS=...` or `--tuning-evals ...` changes that count. Grid searches ignore this budget. Use smaller choice lists for smaller NRT or NNCDE grids. NRT results from the previous random search require `--force` or a new output directory because their request hashes differ.

To modify the tuning grid or search ranges:

- **NRT:** edit `NRT_SEARCH_SPACE` in [main.py](main.py). Each value is a nonempty list of distinct integer choices. For example, `[8, 16, 24, 32]` gives four choices, while `[8, 24]` gives only two. Grid search evaluates every combination from the Cartesian product of these lists in a deterministic order.
- **NNCDE:** edit `NNCDE_GRID` in [main.py](main.py). Each value is a list of grid choices.
- **DeepSurv:** edit `OFFICIAL_TUNING_PARAMETERS["reference_search"]["ranges"]` in [benchmark/deepsurv/cuda_model.py](benchmark/deepsurv/cuda_model.py). The optimizer reads these ranges directly. Keep the testing copy synchronized. Learning rate is sampled on the log10 scale, so `[-5, -3]` means learning rates between `0.00001` and `0.001`.

DeepSurv requests record the search ranges for resumption and summary validation. Results from the earlier hard-coded search (2–20 nodes, log10 learning rate −7 to −3, and dropout 0–0.5) require retuning with `--force` or a new output directory; their CV scores cannot be reused for the corrected ranges.

NRT and NNCDE also accept `--search-space /path/to/file.json`, using the same keys and explicit grid choice lists as their Python dictionaries.

Startup checks and requeue are described in [the workflow guide](../README.md#node-checks-and-requeue). All launchers enable requeue and append logs; node checks run before fits or CPU workers begin.
