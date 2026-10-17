# Current NRT simulation testing

From the NRT root, run `python -m regularization.testing.simulation run` or submit
`sbatch regularization/testing/run_nll.sh`. The default three GPU tasks each run
one method over 35 cases and 500 replications sequentially.

The default hyperparameter directory is `regularization/tuning/output/hyper_para`.
CSV copying is unnecessary. Training settings and fixed refit epochs come from
tuning; testing performs no hyperparameter or stopping-epoch selection.
The generator, seed formula, and random train/test split match the benchmarks.

Each fit saves one JSON. Once all requested methods and replications succeed,
case SSE tables are published automatically, including `average` and sample
`standard_deviation`. Failed records can be retried.

See the [workflow guide](../README.md) for commands and output formats.
