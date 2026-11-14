# Neural benchmark tuning backends

This directory retains `nncde/` and `deepsurv/`, the two models searched by
`tuning.main`. Both support exact and right-censored observations, covering
15 cases in the retained experiment. Use [the tuning guide](../README.md) and
[the full rerun procedure](../../RERUN_SIMULATIONS.md).

CPU benchmark models and deep distribution regression require no separate
search in this protocol; their implementations are retained in
[testing/benchmark](../../testing/benchmark/README.md). Old notebook runners,
their shared evaluation wrapper, and unused tuning copies were removed.

The neural model sources retain their original contents. Their testing copies
remain separate to preserve the recorded training protocol. Requirements are
pinned through the root inventory.
