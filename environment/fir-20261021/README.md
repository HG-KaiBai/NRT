# Fixed source-server environment — 2026-10-21

This is NRT's required environment, captured from the installed Fir runtime after
`source runtime.sh`. No package was installed, upgraded, or removed during capture.
Both Alliance and ordinary Linux destinations must match this same baseline.
See [the reproduction guide](../../ENVIRONMENT_REPLICATION.md) for setup and limits.

| Component | Required version |
| --- | --- |
| Python / R | 3.14.2 / 4.4.0 |
| torch | 2.11.0+computecanada |
| CUDA build / cuDNN | 12.9 / 9.13.1 (91301) |
| NumPy / SciPy / pandas | 2.4.2 / 1.18.1 / 3.0.0, all +computecanada |
| Optuna / lifelines | 4.5.0+computecanada / 0.30.0 |
| scikit-learn / scikit-survival | 1.9.1 / 0.28.0 |
| R survival / partykit / icenReg / ICcforest | 3.5-8 / 1.3-0 / 2.0.16 / 0.5.1 |

The baseline includes **103 Python distributions and 77 R packages**. The version
lists include transitive dependencies; the table is only a summary.

| File | Purpose |
| --- | --- |
| `python-requirements.txt` | Exact full Python inventory, preserving build suffixes; root requirements includes this file. |
| `python-packages.json` | Distribution locations, requirements, wheel metadata, and original artifact URL/hash when available. RECORD hashes describe metadata, not a fresh audit of installed binaries. |
| `python-freeze.txt`, `python-pyvenv.cfg` | Original installation provenance and Python base environment. |
| `runtime.json`, `modules.txt` | Python/compiler/ABI, imports, PyTorch/CUDA/cuDNN, CPU features, NumPy/native thread pools, modules and numerical settings. |
| `r-packages.csv` | All R versions, build versions, library placement, dependencies and package metadata. |
| `r-version.csv`, `r-runtime.txt` | R version, BLAS/LAPACK, RNG, library order, session and loaded native libraries. |
| `provenance.json` | Capture time, host, Git HEAD and working-tree status. |
| `SOURCE_SHA256SUMS`, `DATA_SHA256SUMS` | Captured working-tree code/requirements and exact bundled dataset bytes; run from the NRT root. |
| `TOOLS_SHA256SUMS` | Capture, verification and R restoration helper integrity; run from the NRT root. |
| `SHA256SUMS` | Snapshot-file integrity; run from this snapshot directory. |
| `validation.json` | Actual environment probe results; GPU model training is not certified by a login-node capture. |

The source manifest was refreshed for the 29-case simulation revision on
2026-10-21, with no change to the captured Python/R versions or builds.
`SOURCE_SHA256SUMS.before-29-cases` preserves the earlier source manifest;
`provenance.json` records the revision and six excluded case IDs. The active
`SOURCE_SHA256SUMS` is the code baseline for the new simulation rerun.
See [the rerun procedure](../../RERUN_SIMULATIONS.md) for job dependencies and
fresh output directories.

The subsequent CPU/GPU precheck update changes launchers and adds `node_runner.py`.
`SOURCE_SHA256SUMS.before-node-precheck` preserves the 29-case source manifest
before that update. The active manifest includes the shared node checks; model,
data-generation, tuning, and simulation Python/R sources remain unchanged by this
launcher update, preserving existing scientific result fingerprints.

Check the current or recreated environment from the NRT root:

```bash
source runtime.sh
(cd environment/fir-20261021 && sha256sum -c SHA256SUMS)
"$PY" environment/verify.py --check-source
```

Capture a target server for comparison without overwriting this baseline:

```bash
"$PY" environment/capture.py --output-dir /your/path/new-target-snapshot
```

Do not substitute the old Python 3.11 inventory in `environment/current/`.
This snapshot contains inventories, not a transferable wheelhouse/container.
Package/build equality and a successful verification do not establish bitwise
identical training results across CPUs/GPUs; preserve the run protocol and compare
outputs as described in the main reproduction guide.

Validation in this update: the complete source-server environment comparison,
`pip check`, four R backend imports, and a pip dry run for all 103 pinned Python
packages passed. Nine verifier fixtures cover a matching environment plus
changed/missing/extra Python packages, changed cuDNN/BLAS/R versions, changed R
packages, and snapshot tampering. Both R helpers parse, and the archive restoration
dependency order covers all 48 user packages. No destination installation or GPU
training was performed. Earlier model smoke results remain in the historical
2026-10-20 record and are not presented as newly run tests.

The 29-case revision passed the 74-test core suite and the updated 14-test launcher
suite (two additional launcher tests, 76 distinct tests in total). CPU smoke runs
exported NNCDE/DeepSurv tuning CSVs, completed four neural simulation fits using
the default tuning-directory lookup, and completed one AFT current-status fit
using only saved tau. Complete summary tables were produced. The current
environment and source verification passed with no differences; no full GPU
experiment was submitted. Details are in `validation.json`.
