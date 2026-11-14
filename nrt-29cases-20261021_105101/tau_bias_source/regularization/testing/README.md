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
generator's saved `tau`, and it is not the finite model's padded `1.1*tau` bound.
The conditional truth uses the same generator as SSE, including marginalization
over the unobserved mixture component. Truth is used only after fitting.

SSE remains in `tables/<case_id>.csv`; the new metric is in
`tables_tau_absolute_bias/<case_id>.csv`. Each has the method columns and 500
replication rows followed by `average` and `standard_deviation`. Missing historical
bias values remain missing; they are never filled with zero. `summarize` publishes
both metrics by default; `--metric average_grid_sse` requests only the original SSE.

## Add the metric to the three retained structures

Historical JSON files contain neither models nor endpoint predictions, so this
requires refitting, using the original frozen tuning parameters, training epochs,
data and seeds. No tuning is repeated. Prepare isolated numerical sources first:

```bash
python3 regularization/testing/prepare_tau_bias.py
```

This creates `tau_bias_source/` inside each of the three run directories. It
restores the finite model from Git `db9aa0b`, the last-split tail model from its
saved archive, and the root-split tail model from its saved source directory.
All 87 frozen specification fingerprints per structure must match. Only testing
code is overlaid; the original source snapshots are retained.

Submit the existing 29 cases, three methods, and 500 replications per structure
from the NRT root, using the same GPU launcher and runtime as the original runs:

```bash
for name in nrt-29cases-20261021_105101 nrt-body-tail-20261110_140942 nrt-root-tail-20261111_010058; do
  run="$PWD/$name"
  mkdir -p "$run/tau_bias_logs"
  NRT_ROOT="$run/tau_bias_source" COMMAND=backfill-tau-bias \
    HYPER_DIR="$run/partition_kl/tuning/hyper_para" OUTDIR="$run/partition_kl/testing" \
    sbatch --chdir="$run/tau_bias_source" \
      --output="$run/tau_bias_logs/%A_%a.out" --error="$run/tau_bias_logs/%A_%a.err" \
      "$run/tau_bias_source/regularization/testing/run_nll.sh"
done
```

`backfill-tau-bias` verifies each original request and regenerated dataset before
refitting. It appends the metric and `tau_bias_evaluation` provenance to the same
JSON, retaining all original SSE/NLL, fingerprints and timing fields. It also saves
the refit's SSE and its difference from the original SSE so hardware-dependent
reproduction differences are visible. Successful additions resume without fitting;
failed additions preserve the original fit and can be retried. This avoids another
130,500 result files. Keep each prepared source unchanged while jobs are running.

After all structures finish, build the paired nine-column comparison tables and
the long `tau_absolute_bias_summary.csv` (case, structure, method, count, mean, SD):

```bash
source runtime.sh
"$PY" -m regularization.testing.compare_structures \
  --run finite_1p1tau=nrt-29cases-20261021_105101/partition_kl/testing \
  --run body_tail=nrt-body-tail-20261110_140942/partition_kl/testing \
  --run root_tail=nrt-root-tail-20261111_010058/partition_kl/testing \
  --output-dir comparison_tau_absolute_bias
```

The comparison checks that data, evaluation endpoints and conditional truth match
across structures for each replication. Incomplete tables are withheld by default;
`--allow-incomplete` explicitly publishes partial counts and blank missing cells.

See the [workflow guide](../README.md) for commands and output formats.
