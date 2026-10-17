# Current NRT environment snapshot

Captured on 2026-10-16 from the installed runtime used by regularization.
`python-requirements.txt` contains exact distribution versions, including build
suffixes. `runtime.json` records Python, PyTorch, CUDA/cuDNN, module stack, and
package metadata. `r-packages.csv` and `r-runtime.txt` describe the installed R
packages and session. `SHA256SUMS` verifies this snapshot's files. `SOURCE_SHA256SUMS` records the
current source and documentation, including changes not yet committed to Git.

Use the existing cluster environment:

```bash
export PY=/home/y679zhao/envs/nrt-runtime/bin/python
"$PY" -m pip check
"$PY" -m unittest regularization.test_workflow -v
sha256sum -c environment/current/SHA256SUMS
sha256sum -c environment/current/SOURCE_SHA256SUMS
```

Recreate Python on a compatible Alliance system from the NRT root, using an unused
environment directory. Match the module stack recorded in `runtime.json`:

```bash
module load StdEnv/2023 python/3.11.5 r/4.4.0 cuda/12.6
python -m venv /your/path/nrt-python311
export PY=/your/path/nrt-python311/bin/python
"$PY" -m pip install --no-index -r environment/current/python-requirements.txt
"$PY" -m pip check
"$PY" -m unittest regularization.test_workflow test_workflow test_deepsurv test_preflight test_cpu_scheduling -v
"$PY" reproduce.py smoke --output-dir /tmp/nrt-reproduction-new
```

The exact `+computecanada` builds require the Alliance wheelhouse or archived wheels.
If a pinned wheel is unavailable, retrieve that exact build before continuing;
substituting a different build does not recreate this snapshot. This inventory
captures versions and runtime configuration; it does not bundle binary installers.
For R benchmarks, restore packages at the versions in `r-packages.csv` into a
separate library and export `R_LIBS_USER` to it. The runtime record identifies the
original library and platform. Keep module-provided system packages in their module
libraries. Verify the relevant R backends before launching simulations.

The source files immediately above this directory describe an older Python 3.14
benchmark environment. They remain available for historical provenance, but are not
the current NRT environment. The current snapshot is suitable for regularization's
code and protocol; identical GPU results across hardware and package builds are not
promised. The source compatibility map is narrowly scoped to the exact cleanup
snapshot and keeps existing current-NRT caches usable without changing their values.
