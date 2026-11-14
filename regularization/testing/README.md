# Current NRT simulation testing

From the NRT root, run `python -m regularization.testing.simulation run` or submit
`sbatch regularization/testing/run_nll.sh`. The default three GPU tasks each run
one method over 29 cases and 500 replications sequentially.

The default hyperparameter directory is `regularization/tuning/output/hyper_para`.
CSV copying is unnecessary. Training settings and fixed refit epochs come from
tuning; testing performs no hyperparameter or stopping-epoch selection.
The generator, seed formula, and random train/test split match the benchmarks.

Each fit saves one JSON. Once all requested methods and replications succeed,
case SSE and S(tau) absolute-bias tables are published automatically, including
`average` and sample `standard_deviation` (n-1). Failed records can be retried.

For each replication, `survival_tau_absolute_bias` is
`mean_test_subjects(abs(S_hat(tau|x) - S_true(tau|x)))`: take the absolute value
**before** averaging subjects. `survival_tau` is the maximum finite observed
endpoint in the training set, including exact cases. It is not the censoring
generator's saved `tau`.
The conditional truth uses the same generator as SSE, including marginalization
over the unobserved mixture component. Truth is used only after fitting.

SSE remains in `tables/<case_id>.csv`; the new metric is in
`tables_tau_absolute_bias/<case_id>.csv`. Each has the method columns and 500
replication rows followed by `average` and `standard_deviation`. Missing historical
bias values remain missing; they are never filled with zero. `summarize` publishes
both metrics by default; `--metric average_grid_sse` requests only the original SSE.

## Running on Fir

The launcher hardcodes `/home/y679zhao/projects/def-anthree/y679zhao/NRT`.
For a complete fresh body-tail run, from that directory:

```bash
bash regularization/submit.sh
```

Testing waits for the entire tuning array to succeed and reads its freshly
exported hyperparameter CSVs. Existing historical outputs remain separate.
To submit testing separately after tuning, set `HYPER_DIR` and `OUTDIR` to the
corresponding absolute paths and run `sbatch regularization/testing/run_nll.sh`.

Rebuild summaries using the same output directory and replication count:

```bash
source runtime.sh
"$PY" -m regularization.testing.simulation summarize --output-dir /absolute/run/testing
```

Only `run` and `summarize` are supported. Cross-structure source restoration,
historical backfill, and comparison drivers have been removed from this folder.
See [the workflow guide](../README.md) for the retained body-tail model and settings.
