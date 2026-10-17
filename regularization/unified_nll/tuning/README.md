# Historical raw-time finite-support KL notebook

`tune_kl_server.ipynb` is retained from the original source project. It imports
`pycode.unified_nll.raw_time.search` and `raw_time.report`; those external package
paths are not the current NRT entry point. Use [current NRT tuning](../../tuning/README.md)
for a self-contained workflow in this repository. Historical infinite-support
experiments and their completed results remain available for reference.

In the original source project, edit the configuration cell and execute cells in
order. Fixed rho, adaptive KL, and fixed rho at the adaptive architecture all use
raw-time finite support. For each fitting subset:

```text
max_observation = max(finite training L/R endpoints)
upper_bound = 1.1 * max_observation
support = [0, upper_bound]
R = -2/(depth*L) * sum(log(width/(upper_bound/L)))
lambda_n = a/sqrt(n_fit), L=2**depth
```

Leaf density is uniform on raw time. The width floor is `upper_bound*0.0008/L`.
KL retains the leaf average and depth factor, without another upper-bound factor.
This historical notebook uses each CV fold's fitting subset to set support;
validation impossibility raises an error. Final test zero likelihood is recorded
as infinite NLL. The current workflow instead fixes common support from the outer
training pool, as documented in the current guide.

The historical notebook defaults are:

| Setting | Value |
| --- | --- |
| Cases | All 35 |
| n_samples / grid_size | 5000 / 256 |
| data_seed / final seeds | 123 / 0..9 |
| x_dim | 6 |
| trials / initial_trials / folds | 64 / 16 / 4 |
| fold_seed / search_seed / search_fit_seed | 20261010 / 871 / 101 |
| depth / shared_layers / private_layers | 3..8 / 0..3 / 1..4 |
| shared_width / private_width | 16, 32, 64, 128 |
| strength_max | 100000, searched on log10(1+a) |
| learning rate / split ratio | 0.001 / 0.1 |
| epochs / patience / validation_every / min_delta | 200 / 50 / 5 / 0.0001 |
| warmup / batch size / CUDA graphs | 10 / 256 / enabled |

The first 80% forms the fitting pool; the final 20% is reserved for evaluation.
All search ranges must contain the baseline. Use a new directory for protocol
changes; the historical default is `results/kl_cv_server_raw_time_v1`. Matching
configuration, source, and device fingerprints permit cache resumption, with a
writer lock preventing concurrent modification. `verify_generator()` compares
the frozen simulator with the original generator before fitting.

Checkpoint selection, TPE, and final selection use unregularized validation NLL.
After all selections are frozen, the final epoch count is the ceiling of the median
best fold epoch. SSE uses 256 points up to the maximum observed training endpoint,
sums grid errors, and averages subjects. The default budget is 18060 CV fits and
1050 final fits. Editing the notebook does not start a search. Seeds 0..9 were used
in development; independent confirmation requires newly specified data seeds.
