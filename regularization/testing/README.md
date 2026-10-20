# Current NRT simulation testing

After the node-mean-KL tuning array succeeds, submit
`sbatch regularization/testing/run_node_mean_kl.sh` from the NRT root. Its two
array tasks (indices 1 and 2) run adaptive and the corresponding fixed-architecture
control. It reads `regularization/tuning/output_node_mean_kl/hyper_para`, writes
to `regularization/testing/output_node_mean_kl`, and summarizes those two methods.
The penalty is restored from tuning's CSV; no testing penalty override is needed.

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
