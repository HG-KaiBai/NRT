# NRT reproduction environment

The required baseline is [fir-20261021](fir-20261021/README.md): Python 3.14.2,
R 4.4.0, and the exact installed package inventories.
[ENVIRONMENT_REPLICATION.md](../ENVIRONMENT_REPLICATION.md) covers setup on
Alliance and ordinary Linux servers.

From the NRT root, with the intended runtime loaded:

```bash
"$PY" environment/verify.py --check-source
"$PY" environment/capture.py --output-dir /your/path/new-target-snapshot
```

The 2026-11-10 cleanup removed earlier environment snapshots unrelated to the
two retained result directories. The Fir package/runtime inventory is unchanged;
source, tools, and dataset checksums now describe the cleaned working tree and
its GT 2011 / Rotterdam data. Historical result manifests remain in the result
directories. `source-compatibility.json` and `partition-cleanup-validation.json`
retain evidence for the existing exact-source compatibility mapping.
