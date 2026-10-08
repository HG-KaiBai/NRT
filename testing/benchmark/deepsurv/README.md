# DeepSurv: official architecture on PyTorch/CUDA

This directory contains one DeepSurv implementation. It ports the network and
training contract from
[`jaredleekatzman/DeepSurv`](https://github.com/jaredleekatzman/DeepSurv) at
commit `41eed003e5b892c81e7855e400861fa7a2d9da4f` to PyTorch so it runs in the
project's current CUDA environment.

The implementation preserves:

- fully connected hidden layers and one linear log-risk output;
- Xavier-uniform initialization;
- ReLU or SELU activations, dropout, and optional Lasagne-style batch norm;
- descending-time, full-risk-set Cox training;
- the official output-layer L1/L2 regularization scope;
- full-batch SGD or Adam, matching the selected official experiment runner;
- Breslow baseline-hazard estimation for survival-curve evaluation.

The Cox objective is isolated in `losses.py`. It follows the PyTorch
`PartialLogLikelihood` published by
[`runopti/stg`](https://github.com/runopti/stg/blob/master/python/stg/losses.py),
with `torch.logcumsumexp` replacing `log(cumsum(exp(logits)))` for numerical
stability. Both sources assume no tied event times, so unsupported tie modes
are rejected explicitly.

DeepSurv supports exact and right-censored observations. It is not extended to
interval-censored or current-status outcomes.

## Paper Gaussian reproduction

The executable Gaussian experiment in the official repository is defined by:

- `experiments/deepsurv/Dockerfile.gaussian`;
- `experiments/deepsurv/models/gaussian_model_selu_revision.1.json`;
- `experiments/data/gaussian/gaussian_survival_data.h5`.

It uses three 39-unit SELU layers, dropout `0.0735595703125`, Adam with learning
rate `0.001485675043584323`, output-layer L2 coefficient
`2.78821044921875`, and 1,000 epochs. These exact values remain exported as
`PAPER_GAUSSIAN_PARAMETERS` for use with the current PyTorch implementation.

Both tuning and testing use this preset by default. Sobol CV varies the
hidden-layer widths/depth, learning rate, learning-rate decay, momentum,
L2 coefficient, and dropout. CV and the final fits use SELU, no batch
normalization, no standardization, zero L1 penalty, Adam, and 1,000 epochs.
Explicit epoch overrides remain available for shorter runs.

The older `gaussian_model_0.json` values remain exported as reference constants.

## Run

Open `pycode/neural_only/benchmark_methods.ipynb`. It runs DeepSurv for exact
or right-censored observations on CUDA when available. Set
`DEEPSURV_CUDA_TUNE=True` to enable the existing Sobol tuning workflow.

The code uses the normal project environment; no separate Theano environment
or subprocess backend is required.

Every CV fit uses its existing validation fold for early stopping. Outside CV,
each fit splits its available data into 80% training and 20% validation, monitors
validation negative Cox partial log-likelihood, and restores the best checkpoint.
The default patience is 20 epochs. Preprocessing and the Breslow baseline use
only optimization rows; the final test set stays separate.
