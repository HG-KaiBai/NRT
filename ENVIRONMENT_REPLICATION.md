# Reproduce the current NRT environment on another server

The required baseline is **the Python and R environment installed on Fir on
2026-10-21**, captured in [environment/fir-20261021](environment/fir-20261021/README.md).
Both Alliance clusters and ordinary Linux/GPU servers must use this same baseline.
Keep exact package versions and `+computecanada` build suffixes. Alternate Python
versions, newer packages, and public PyPI replacements are different environments.
No installed package was changed during this update.

The top-level `environment/` inventory, `environment/current/` (Rorqual), and
`environment/fir-20261020/` are historical records. Root [requirements.txt](requirements.txt)
and all four workflow requirements files now include the same current inventory.

## Exact baseline

| Component | Required version/build |
| --- | --- |
| CPython | 3.14.2; GCC 14.3.0; `cpython-314-x86_64-linux-gnu` |
| R | 4.4.0; `x86_64-pc-linux-gnu` |
| torch | `2.11.0+computecanada` |
| CUDA build / cuDNN | 12.9 / 9.13.1 (`91301`) |
| NumPy / SciPy / pandas | `2.4.2+computecanada` / `1.18.1+computecanada` / `3.0.0+computecanada` |
| Optuna / lifelines | `4.5.0+computecanada` / `0.30.0` |
| scikit-learn / scikit-survival | `1.9.1` / `0.28.0` |
| matplotlib / Optunity | `3.10.8+computecanada` / `1.1.1` |
| R survival / partykit / icenReg / ICcforest | `3.5-8` / `1.3-0` / `2.0.16` / `0.5.1` |
| Native BLAS | FlexiBLAS 3.3.1, AOCL-BLAS 5.1.0 (build 20250513) |
| R LAPACK / glibc | 3.11.0 / 2.37 |

**All 103 Python distributions and all 77 visible R packages** are part of the
baseline, including transitive dependencies and R base/recommended packages.
The table is only a summary. Distribution metadata and imported versions are
recorded separately: imported versions may omit `+computecanada`.

Source interpreter: `/home/y679zhao/envs/pt/bin/python3.14`; base prefix:
`/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/python/3.14.2`.
R comes from `r/4.4.0`, with user library
`/home/y679zhao/R/x86_64-pc-linux-gnu-library/4.4`. The snapshot records the module
stack, compiler/ABI, PyTorch build, NumPy/native thread pools, R BLAS/LAPACK,
library search order, and package provenance. Destination installation paths may
differ; binary compatibility and effective library versions still need to match.

## Preserve source, data, and installers

Copy the **working tree**, including uncommitted source edits, not just Git HEAD.
The active source manifest includes the 29-case simulation revision; the earlier
manifest is retained as `SOURCE_SHA256SUMS.before-29-cases`. Python/R package pins
are unchanged. Use [RERUN_SIMULATIONS.md](RERUN_SIMULATIONS.md) for the new full run.
It also includes the shared CPU/GPU launcher prechecks; the preceding source
manifest is retained as `SOURCE_SHA256SUMS.before-node-precheck`.
`SOURCE_SHA256SUMS` records the captured Python/R/shell code; `DATA_SHA256SUMS`
records the exact CSV bytes. METABRIC is in original row order, SHA-256
`ba502e364efab52916d4316e97d6020e924545bef0f88802f4d77f973cf0eeb1`.
Also preserve external search-space files, selected hyperparameters, requests,
split assignments, seeds, and completed reference outputs.

The snapshot is an installed inventory, **not a bundled installer or complete
artifact lock**. Original wheel URLs/hashes are present only where installed
metadata retained them. `SHA256SUMS` hashes snapshot files, not all installed
binaries. Collect the pinned wheels while the matching wheelhouse is available:

```bash
source runtime.sh
export NRT_ARTIFACTS=/your/path/nrt-artifacts
mkdir -p "$NRT_ARTIFACTS/wheels"
"$PY" -m pip download --no-index --only-binary=:all: --no-deps \
  -r requirements.txt --dest "$NRT_ARTIFACTS/wheels"
(cd "$NRT_ARTIFACTS/wheels" && sha256sum -- *.whl > SHA256SUMS)
```

`--no-index` relies on configured Alliance wheelhouse locations; add `--find-links`
for an exact compatible repository if needed. These are collection instructions;
no wheelhouse has been assembled by this update. Recover missing artifacts from
the original cache/repository before treating the environment as transferable.
Collect and hash the R source archives for all user-library rows as well. Preserve
the native software stack or a compatible container image: compiled wheels can
require libraries outside the virtual environment. Pins, artifact hashes and a
wheelhouse provide stronger repeatability than versions alone.
[pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/).

## Alliance destination with matching CVMFS/modules

Use new environment directories. From the copied NRT root:

```bash
set -e
module load StdEnv/2023 python/3.14.2 r/4.4.0
export NRT_ROOT="$PWD"
export NRT_VENV="$HOME/envs/nrt-fir-20261021"
# Stop if this path already exists; do not overwrite an existing environment.
test ! -e "$NRT_VENV"
python3 -m venv "$NRT_VENV"
export PY="$NRT_VENV/bin/python3.14"
export R_MODULE=r/4.4.0
export R_LIBS_USER="$HOME/R/nrt-fir-20261021"
mkdir -p "$R_LIBS_USER"

# Set NRT_ARTIFACTS to the transferred installer directory first.
(cd "$NRT_ARTIFACTS/wheels" && sha256sum -c SHA256SUMS)
"$PY" -m pip install --no-index --find-links "$NRT_ARTIFACTS/wheels" \
  --no-deps -r requirements.txt
"$PY" -m pip check
```

Match `modules.txt` and native-library records as well as module names. Do not
load the historical `cuda/12.6` module: this PyTorch build uses CUDA 12.9. Confirm
CPU compatibility with the source x86-64-v4 base stack. Restore R below before
verification. The source contains both `pillow` and `pillow_simd` metadata; their
files can overlap. The effective `PIL.__version__` must be 12.1.0. If a fresh
installation exposes the older PIL implementation, reinstall the exact pinned
`pillow==12.1.0+computecanada` wheel last, then verify both inventory and imports.

## Ordinary Linux/GPU destination without CVMFS

The required versions/builds remain exactly the same. A stock PyPI torch wheel
with the same public version does not recreate the Alliance build. Do not strip
build suffixes or generate a second, approximate requirements file.

Provide the compatible native software tree and exact artifacts through an
equivalent mounted stack or a container prepared with that stack. Recreate
Python 3.14.2 and R 4.4.0 inside it, install the same wheels, and restore the same
R packages. Check compiler, ABI, CPU features, glibc, BLAS/LAPACK and shared-library
records. Copying `~/envs/pt` alone is insufficient: its interpreter and native
dependencies refer outside that directory.

No portable image has been built or tested by this update. If an exact artifact
or native dependency is unavailable, record the missing component; the fixed
environment requirement has not yet been met. A container preserves user-space
libraries but still needs compatible host hardware and an NVIDIA driver. Save
the image digest, driver, GPU model/MIG profile, and GPU-node validation results.

Once the matching runtime exists, set `PY`, `R_LIBS_USER`, and Rscript on PATH,
and use the same offline pip installation command above. Without `module`,
`runtime.sh` uses the supplied Python and Rscript. If `module` exists, it also
loads `R_MODULE`; ensure that this selects the recorded R build. Absolute project
and environment paths may differ.

## Restore the exact R packages

`r-packages.csv` contains versions, build versions, original library placement,
and Depends/Imports/LinkingTo. Match the 29 system-library packages from R 4.4.0
first, then restore the 48 user-library packages to an empty `R_LIBS_USER`:

```bash
# Verify the transferred R archive directory's own SHA256SUMS first.
Rscript --vanilla environment/restore-r.R \
  environment/fir-20261021/r-packages.csv "$NRT_ARTIFACTS/r-sources"
```

This helper checks system package/build versions and installs only pinned local
`Package_Version.tar.gz` archives in dependency order, with dependencies disabled.
It requires an empty user library, does not modify the system library, and does
not download upgrades. Use the captured compiler/native stack when compiling.
Matching R source versions with a different compiler or BLAS is not proof of
identical binaries. Retain build records and run the environment check afterward.

## Verify before experiments

From the copied NRT root, after restoring Python and R:

```bash
source runtime.sh
(cd environment/fir-20261021 && sha256sum -c SHA256SUMS)
"$PY" environment/verify.py --check-source
"$PY" -m realworld.run preflight --device cpu
"$PY" -m unittest realworld.test_realworld regularization.test_workflow
"$PY" reproduce.py smoke --output-dir /tmp/nrt-reproduction-new
```

`verify.py` exits nonzero for missing/extra packages, changed exact distribution
or imported versions, Python/compiler/ABI and torch/CUDA/cuDNN/numeric-setting
mismatches, R package/build/native-version changes, dependency/import failures,
or changed datasets. `--check-source` also checks the captured working tree.
Paths, module stack, CPU/GPU inventory and other native-library differences are
reported for review. Passing this check establishes the recorded fields it
compares; it does not prove binary identity or equal scores. Run source checks
before adapting launcher paths; review intentional shell edits separately.

The capture follows `source runtime.sh`: OpenMP/BLAS defaults are one thread;
PyTorch has one intra-op and 32 inter-op threads in this process. Actual training
entry points set torch intra-op threads from `--threads`, normally four in Slurm.
Match the reference run's threads, device, precision, fused optimizer and CUDA
Graph settings. Preserve the runtime's recorded defaults: deterministic algorithms
false, cuDNN deterministic false, cuDNN benchmark false, matmul TF32 false,
cuDNN TF32 true, and float32 matmul precision `highest`.

On an **allocated GPU node**, also run:

```bash
"$PY" environment/verify.py --require-cuda
"$PY" -m realworld.run preflight --device cuda:0
nvidia-smi
"$PY" environment/capture.py --output-dir /your/path/new-gpu-node-snapshot
```

This performs a small CUDA calculation; a model smoke run is still needed. A
login-node capture does not establish GPU execution. Current launchers request
H100 MIG devices. Slurm account/GPU resource names, absolute `--chdir` and log
paths, and excluded node names are site-specific: update or override them on the
destination, including child scripts submitted by `realworld/submit_support.sh`.
Create log directories before submission. Slurm reads `#SBATCH` headers before
shell variables expand. Setup validation does not require production jobs.

## Match numerical results

Alongside the fixed environment, preserve:

1. Exact code and CSV bytes, including row order. Sorting unchanged subjects can
   change seeded CV/smoke selections. Keep split assignments and external
   search-space files rather than regenerating from different defaults.
2. Cases, data sizes, seeds, folds, trial budgets and trial order, penalty,
   selected architecture, optimizer settings, stopping rules, fixed refit epochs,
   time units and evaluation grid. Keep the case manifests, hyperparameters and
   realworld `request.json` with the original results.
3. Device type, GPU model/MIG profile, driver, numeric precision, thread counts,
   BLAS implementation, CUDA graphs and fused optimizer choices. Existing code
   seeds random generators but does not globally enable deterministic algorithms.
   Enabling a new deterministic mode changes the protocol; validate it as a new
   baseline instead of silently applying it to an old experiment.
4. Fresh output directories for independent reruns. Do not overwrite manifests
   or introduce fingerprint aliases to force reuse of incompatible cached results.

First compare a small same-seed run on both servers: selected parameters/epochs,
individual predictions, per-replication metrics, and summaries. Exclude elapsed
time, timestamps, paths and device labels from numeric comparisons. Define numeric
tolerances before reviewing differences. A changed selected model is substantive
even if its final score is close. Smoke success validates execution, not equality
of the full experiment. Matching dependencies cannot guarantee bitwise equality
across hardware or between CPU and GPU; retain an actual output-comparison report.
[PyTorch 2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).

## Prompt for the destination-server assistant

```text
Recreate the exact NRT source-server environment using ENVIRONMENT_REPLICATION.md
and environment/fir-20261021/. Python 3.14.2, R 4.4.0 and every captured package
version/build are fixed. Preserve +computecanada suffixes. Inspect the target
first, use new environment/library directories, and obtain matching artifacts.
Do not substitute public builds or alternate versions. Report any unavailable
artifact/native dependency; do not describe an approximation as equivalent.
Both Alliance and ordinary Linux targets use this same baseline. Restore pinned
R archives, run environment/verify.py, pip check, CPU/R and allocated-GPU checks,
and smoke validation. Compare source/data hashes and numeric outputs with the
reference run. Report native/hardware differences and untested checks. Do not
launch production experiments during setup.
```
