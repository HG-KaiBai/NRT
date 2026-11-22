# Reproduce the simulation environment

The baseline is the installed Fir environment captured on 2026-10-21, retained
unchanged in `source/environment/fir-20261021/`. The original run's
`environment_validation.json` records additional unused R packages present at
execution. Python/R package inventories, build suffixes, native libraries,
hardware settings, and the frozen experiment protocol all matter.

| Component | Recorded version |
| --- | --- |
| Python / R | 3.14.2 / 4.4.0 |
| torch | 2.11.0+computecanada |
| CUDA build / cuDNN | 12.9 / 9.13.1 |
| NumPy / SciPy / pandas | 2.4.2+computecanada / 1.18.1+computecanada / 3.0.0+computecanada |
| Optuna / lifelines | 4.5.0+computecanada / 0.30.0 |
| scikit-learn / scikit-survival | 1.9.1 / 0.28.0 |
| R survival / partykit / icenReg / ICcforest | 3.5-8 / 1.3-0 / 2.0.16 / 0.5.1 |

Use the complete 103-package Python and 77-package R inventories, not just this
table. The snapshot includes compiler/ABI, BLAS/LAPACK, module stack, numeric
settings, package metadata, and library search paths. The original interpreter
is `/home/y679zhao/envs/pt/bin/python3.14`; the R user library is
`/home/y679zhao/R/x86_64-pc-linux-gnu-library/4.4`.

## Existing Fir installation

From this experiment directory:

```bash
set -euo pipefail
export NRT_REFERENCE="$PWD"
export NRT_ROOT="$NRT_REFERENCE/source"
export PYTHONDONTWRITEBYTECODE=1
source "$NRT_ROOT/runtime.sh"
sha256sum --quiet -c "$NRT_REFERENCE/SOURCE_SHA256SUMS"
(cd "$NRT_ROOT/environment/fir-20261021" && sha256sum --quiet -c SHA256SUMS)
"$PY" "$NRT_REFERENCE/validate_environment.py"
"$PY" -m unittest test_preflight test_node_runner
```

The experiment validator accepts additional R packages only when none is loaded
by the benchmark backends; recorded package/runtime changes still fail. It
refreshes the experiment's environment reports. For an exact full-inventory
comparison, including any extra installed packages:

```bash
"$PY" "$NRT_REFERENCE/environment/verify.py" --check-source
```

The simulation-specific verifier checks the frozen experiment source when
requested. It does not check unrelated data files listed in the old snapshot's
historical `DATA_SHA256SUMS`; simulation data are generated from saved seeds and
case settings. Run GPU checks on an allocated node:

```bash
"$PY" "$NRT_REFERENCE/environment/verify.py" --require-cuda
```

## Recreate on another server

The snapshot contains inventories, not an installer bundle or container. Obtain
the exact compatible Python wheels and R source archives before installing.
Preserve `+computecanada` suffixes. Match the native stack and interpreter ABI;
a public wheel with the same upstream version is a different build.

On an Alliance installation with the matching CVMFS stack, create new environment
directories. Set `NRT_ARTIFACTS` to the transferred, independently checksummed
artifact bundle and `NRT_REFERENCE` to the copied experiment directory:

```bash
set -euo pipefail
module load StdEnv/2023 python/3.14.2 r/4.4.0
export NRT_VENV="$HOME/envs/nrt-fir-20261021"
test ! -e "$NRT_VENV"
python3 -m venv "$NRT_VENV"
export PY="$NRT_VENV/bin/python3.14"
export R_LIBS_USER="$HOME/R/nrt-fir-20261021"
mkdir -p "$R_LIBS_USER"
"$PY" -m pip install --no-index --find-links "$NRT_ARTIFACTS/wheels" --no-deps \
  -r "$NRT_REFERENCE/source/requirements.txt"
"$PY" -m pip check
Rscript --vanilla "$NRT_REFERENCE/environment/restore-r.R" \
  "$NRT_REFERENCE/source/environment/fir-20261021/r-packages.csv" \
  "$NRT_ARTIFACTS/r-sources"
```

The R helper requires an empty user library, checks the recorded system package
versions, and installs the 48 pinned user packages from local source archives in
dependency order. It does not download upgrades. Match the recorded compiler and
native libraries when compiling. Both `pillow` and `pillow_simd` metadata were
present in the baseline; verify that the effective `PIL.__version__` is 12.1.0.

On a server without CVMFS, provide a compatible native stack or container and
the same binary artifacts. Copying the virtual environment alone does not copy
its external interpreter and native-library dependencies. No portable image is
bundled here. Record unavailable artifacts rather than substituting versions.

Run the source, environment, backend, and allocated-GPU checks before the full
experiment. Capture a destination inventory into a new directory with:

```bash
"$PY" "$NRT_REFERENCE/environment/capture.py" \
  --output-dir /your/path/new-target-environment
```

The capture tool records simulation source and environment only. Matching
inventories does not establish bitwise-identical results across hardware. Keep
the original seeds, model/search settings, selected parameters, epochs, thread
counts, numeric precision, and GPU configuration. Compare a same-seed small run
before undertaking the full protocol in [README.md](README.md).
