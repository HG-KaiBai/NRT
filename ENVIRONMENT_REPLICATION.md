# Recreate the NRT Python and R environments on another server

This guide includes a prompt to give an assistant on the destination server. Copy this file and the entire `environment/` directory with the project. The manifests describe the **installed environment inspected on 2026-10-09**, including transitive dependencies and package build suffixes.

## Copy-and-paste prompt

```text
Recreate the NRT project's source Python and R environments on this server.
Match the interpreter versions, absolute paths, and every installed package
version using the attached ENVIRONMENT_REPLICATION.md and environment/ files.
Inspect the destination and perform the setup and verification; do not stop
at installation advice.

Required paths and runtime versions:
- Project: /home/y679zhao/projects/def-anthree/y679zhao/NRT
- Python environment: /home/y679zhao/envs/pt
- Python executable: /home/y679zhao/envs/pt/bin/python3.14
- CPython: 3.14.2, GCC 14.3.0, cpython-314-x86_64-linux-gnu
- Python base prefix:
  /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/python/3.14.2
- Original environment creator: virtualenv 20.36.1
- R: 4.4.0, x86_64-pc-linux-gnu; source module: r/4.4.0
- R executable:
  /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.4.0/bin/R
- Rscript executable:
  /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.4.0/bin/Rscript
- R_LIBS_USER: /home/y679zhao/R/x86_64-pc-linux-gnu-library/4.4
- R system library:
  /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.4.0/lib64/R/library

Treat the captured manifests as the authority:
- environment/python-requirements.txt: all 57 pinned Python distributions,
  including pip and setuptools; preserve every +computecanada suffix.
- environment/python-packages.csv: original distribution locations and
  direct wheel URLs/hashes where installed metadata recorded them.
- environment/python-freeze.txt and python-pyvenv.cfg: original provenance.
- environment/python-runtime.txt: compiler, ABI, OS, CUDA/cuDNN build,
  actual imported versions, and PyTorch build configuration.
- environment/r-packages.csv: all 77 R packages, library placement, build
  versions, package sources, and Depends/Imports/LinkingTo metadata.
- environment/r-runtime.txt and r-modules.txt: R paths, library search order,
  module stack, BLAS/LAPACK, native libraries, and session information.

Implementation requirements:
1. Inspect the target OS, CPU architecture/features, available modules,
   CVMFS mounts, compilers, NVIDIA driver, paths, and existing environments.
   The source host is AlmaLinux 9.7; its software stack reports glibc 2.37.
   Python/R base installations use x86-64-v4 paths. Check CPU compatibility.
2. Prefer the matching Alliance software stack when available:
   StdEnv/2023, python/3.14.2, and r/4.4.0. Compare actual loaded modules
   and native libraries with the captured files rather than relying on names.
3. Recreate Python at its required path from the matching base interpreter.
   Reinstall the exact distributions from matching wheels. An ordinary PyPI
   build without +computecanada is a different build. Do not silently replace
   unavailable packages or use the project's short benchmark requirements
   files as the full environment specification.
4. Restore all R package versions. The source has 48 user-library packages
   and 29 packages in the R system library. Preserve .libPaths() precedence
   and each package's library placement. Check bundled base/recommended
   packages too. Restore user packages in dependency order using the pinned
   CSV; do not allow automatic dependency upgrades. Keep bootstrap tools
   such as remotes or renv in a separate tooling library.
5. Preserve existing destination environments. Make setup repeatable and
   use explicit interpreter paths. If exact paths, package artifacts, or
   native dependencies cannot be reproduced, report the specific mismatch
   and proposed alternative; do not describe an approximation as identical.
6. Configure NRT_ROOT, PY, R_MODULE, R_LIBS_USER, and Rscript on PATH for
   both interactive shells and batch jobs. Create tuning/log and testing/log.
   Inspect Slurm account/GPU names on this server and document any necessary
   launcher overrides. Validate submission with --test-only.
7. Run the verification commands below, pip check, NRT CPU/backend preflight,
   and a GPU calculation on an allocated GPU node. The source uses CUDA 12.9
   and cuDNN 91301. No GPU on a login node is not a GPU execution test.
   Do not submit production tuning or simulation jobs as part of setup.
8. Deliver a repeatable setup script, activation instructions, and a report
   comparing versions, paths, complete package inventories, native runtime,
   CPU/backend checks, and GPU checks. State any unverified checks explicitly.

Important source details:
- Installed lifelines is 0.30.0, even though the benchmark requirements
  files say 0.30.3. Use installed 0.30.0 for this replication.
- Both pillow and pillow_simd distribution metadata are installed. They can
  overlap at the PIL import. Match the captured imported PIL version 12.1.0
  as well as the package inventory; installation order can matter.
- R's 4.5 user-library directory is outside this R 4.4 environment.
```

## Captured versions and files

| Component | Exact installed distribution/package version |
| --- | --- |
| Python | 3.14.2 |
| pip | 26.0.1+computecanada |
| setuptools | 81.0.0+computecanada |
| torch | 2.11.0+computecanada |
| NumPy | 2.4.2+computecanada |
| pandas | 3.0.0+computecanada |
| SciPy | 1.18.1+computecanada |
| lifelines | 0.30.0 |
| Optunity | 1.1.1 |
| R | 4.4.0 |
| survival | 3.5-8 |
| icenReg | 2.0.16 |
| partykit | 1.3-0 |
| ICcforest | 0.5.1 |

Python distribution metadata preserves `+computecanada`; imported `torch.__version__`, `numpy.__version__`, and similar attributes can omit that suffix. Compare both the package manifests and imported versions. All four required R packages loaded successfully during capture, and Python's `pip check` passed.

The full inventories are [python-requirements.txt](environment/python-requirements.txt), [python-packages.csv](environment/python-packages.csv), and [r-packages.csv](environment/r-packages.csv). Runtime details are in [python-runtime.txt](environment/python-runtime.txt), [python-pyvenv.cfg](environment/python-pyvenv.cfg), [python-freeze.txt](environment/python-freeze.txt), [r-runtime.txt](environment/r-runtime.txt), and [r-modules.txt](environment/r-modules.txt). [SHA256SUMS](environment/SHA256SUMS) verifies these captured files; it does not hash every installed package binary.

## Preparing the destination

On a destination with the same Alliance modules, begin with:

```bash
module load StdEnv/2023 python/3.14.2 r/4.4.0
export NRT_ROOT=/home/y679zhao/projects/def-anthree/y679zhao/NRT
export PY=/home/y679zhao/envs/pt/bin/python3.14
export R_MODULE=r/4.4.0
export R_LIBS_USER=/home/y679zhao/R/x86_64-pc-linux-gnu-library/4.4
cd "$NRT_ROOT"
mkdir -p tuning/log testing/log "$R_LIBS_USER"
sha256sum -c environment/SHA256SUMS
```

Create a fresh environment at `/home/y679zhao/envs/pt` using the matching Python base installation and virtualenv version, then install [python-requirements.txt](environment/python-requirements.txt). Existing environments should first be inspected and compared. Python environments contain absolute interpreter references and are intended to be recreated at the destination. [Python venv documentation](https://docs.python.org/3.14/library/venv.html).

The snapshot contains package manifests, not a transferred wheelhouse or R source archives. Obtain the matching artifacts from the source software repository/cache before assuming an unrelated server can install them. `python-freeze.txt` includes source-specific `file:///cvmfs/...` references; `python-requirements.txt` provides version pins without those absolute wheel URLs. A freeze is an inventory rather than a complete solver lockfile. [pip freeze documentation](https://pip.pypa.io/en/stable/cli/pip_freeze/).

After collecting the exact compatible wheels into a transferred wheelhouse, installation into the new environment can use:

```bash
"$PY" -m pip install --no-index --find-links /path/to/transferred-wheelhouse \
  --no-deps -r "$NRT_ROOT/environment/python-requirements.txt"
"$PY" -m pip check
```

All distributions are pinned, so this command installs the supplied inventory without resolving additional packages. Preserve artifact hashes when collecting wheels; the snapshot has original wheel hashes for only the packages whose installed metadata retained them. [pip download documentation](https://pip.pypa.io/en/stable/cli/pip_download/).

For R, match the module's 29 system-library packages first, then restore the 48 user-library packages to `R_LIBS_USER`. The CSV records the dependencies needed to determine installation order. A setup script can use `remotes::install_version(..., dependencies=FALSE, upgrade="never", lib=Sys.getenv("R_LIBS_USER"))` for each pinned package after its pinned dependencies exist. Install `remotes` in a separate tooling library so it does not become an extra package in the replicated library search path. [remotes install_version documentation](https://remotes.r-lib.org/reference/install_version.html).

On a server without the matching CVMFS software stack, the source's R executable paths and Python base prefix will require an equivalent mounted software tree or a suitably configured container. Rebuilding packages from source can match versions while changing binary dependencies. Document those differences when exact replication is unavailable.

## Verify the recreated environment

Run these commands from the NRT project root on the destination. They can also check the source against its captured inventory.

```bash
"$PY" - <<'PY'
import csv
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()

runtime = json.loads(Path("environment/python-runtime.txt").read_text())
with Path("environment/python-packages.csv").open(newline="") as stream:
    expected = {canonical(row["Name"]): (row["Version"], row["Location"])
                for row in csv.DictReader(stream)}
actual = {canonical(dist.metadata["Name"]): (dist.version, str(dist.locate_file("")))
          for dist in importlib.metadata.distributions()}
differences = {name: (expected.get(name), actual.get(name))
               for name in expected.keys() | actual.keys()
               if expected.get(name) != actual.get(name)}
assert not differences, differences
assert sys.version_info[:3] == (3, 14, 2)
assert sys.executable == runtime["executable"]
assert sys.prefix == runtime["prefix"]
assert sys.base_prefix == runtime["base_prefix"]
for name, version in runtime["import_versions"].items():
    assert importlib.import_module(name).__version__ == version, name
import torch
assert torch.version.cuda == runtime["torch_cuda_build"]
assert torch.backends.cudnn.version() == runtime["cudnn_version"]
print("Python versions, paths, complete package inventory, and imports match.")
PY

Rscript --vanilla - <<'RS'
expected <- read.csv("environment/r-packages.csv", stringsAsFactors=FALSE)
actual <- as.data.frame(installed.packages(), stringsAsFactors=FALSE)
stopifnot(as.character(getRversion()) == "4.4.0")
stopifnot(nrow(actual) == nrow(expected), !anyDuplicated(actual$Package))
index <- match(expected$Package, actual$Package)
stopifnot(!anyNA(index))
for (field in c("Package", "Version", "LibPath", "Built")) {
  stopifnot(identical(as.character(expected[[field]]),
                      as.character(actual[[field]][index])))
}
r_root <- "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/r/4.4.0"
stopifnot(unname(Sys.which("R")) == paste0(r_root, "/bin/R"))
stopifnot(unname(Sys.which("Rscript")) == paste0(r_root, "/bin/Rscript"))
stopifnot(R.home() == paste0(r_root, "/lib64/R"))
stopifnot(identical(.libPaths(), c(
  "/home/y679zhao/R/x86_64-pc-linux-gnu-library/4.4",
  paste0(r_root, "/lib64/R/library"))))
for (pkg in c("survival", "icenReg", "partykit", "ICcforest")) {
  stopifnot(requireNamespace(pkg, quietly=TRUE))
}
cat("R versions, paths, complete package inventory, and backend imports match.\n")
RS

"$PY" -m pip check
"$PY" - <<'PY'
from workflow import METHODS, node_preflight
node_preflight("cpu", set(METHODS), "testing")
PY
TUNING_TASK_ID=0 bash tuning/tune_nrt.sh --dry-run
```

On an allocated GPU node, additionally run:

```bash
"$PY" - <<'PY'
from workflow import NRT_LOSSES, node_preflight
node_preflight("cuda:0", set(NRT_LOSSES), "tuning")
PY
```

These checks establish the recorded versions, paths, package inventory, imports, and execution capabilities. Compare the captured compiler, PyTorch build configuration, R native libraries, and module stack separately before claiming matching native builds. Identical inventories alone do not establish identical binary artifacts or numerical results across hardware.
