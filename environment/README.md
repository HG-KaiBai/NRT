# NRT environment records

The required source-server baseline is [fir-20261021](fir-20261021/README.md):
Python 3.14.2, R 4.4.0, and the full exact installed package inventories.
[ENVIRONMENT_REPLICATION.md](../ENVIRONMENT_REPLICATION.md) covers both Alliance
and ordinary Linux servers using this same fixed baseline.

From the NRT root, with the intended runtime loaded:

```bash
"$PY" environment/verify.py --check-source
"$PY" environment/capture.py --output-dir /your/path/new-target-snapshot
```

`current/` is a **historical Rorqual Python 3.11 snapshot**, despite its directory
name. Top-level package manifests are the 2026-10-09 benchmark inventory, and
`fir-20261020/` records an earlier Fir validation. Keep those historical files as
provenance; root `requirements.txt` selects only `fir-20261021/` for new reproductions.
