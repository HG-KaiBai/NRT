"""Compare this server with the exact captured Python/R environment; no installation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from capture import ROOT, canonical, command, python_packages, python_runtime


def compare_maps(label, expected, actual):
    return [f"{label} {key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
            for key in sorted(expected.keys() | actual.keys()) if expected.get(key) != actual.get(key)]


def checksum_errors(path, root):
    errors = []
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        item = root / name
        if not item.is_file() or hashlib.sha256(item.read_bytes()).hexdigest() != digest:
            errors.append(f"Checksum mismatch: {item}")
    return errors


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def r_inventory(path):
    rows = read_csv(path)
    # Preserve search-path precedence for duplicates: installed.packages reports
    # library order, but the snapshot is sorted. Treat duplicate names explicitly.
    names = [row["Package"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate R packages across libraries require manual resolution: {path}")
    return {row["Package"]: (row["Version"], row["Built"], row["Priority"]) for row in rows}


def verify(snapshot, check_source=False, require_cuda=False):
    failures = checksum_errors(snapshot / "SHA256SUMS", snapshot)
    if failures:
        return dict(passed=False, failures=failures, differences=[])
    expected_packages = json.loads((snapshot / "python-packages.json").read_text())
    actual_packages = python_packages()
    for label, rows in (("expected", expected_packages), ("actual", actual_packages)):
        names = [canonical(row["name"]) for row in rows]
        if len(names) != len(set(names)):
            failures.append(f"Duplicate Python distribution names in {label} inventory")
    versions = lambda rows: {canonical(row["name"]): row["version"] for row in rows}
    failures += compare_maps("Python package", versions(expected_packages), versions(actual_packages))
    expected = json.loads((snapshot / "runtime.json").read_text())
    actual = python_runtime()
    for key in ("python", "compiler", "soabi", "machine", "libc", "cuda_build", "cudnn", "torch_git_version", "torch_build", "numeric_settings"):
        if expected[key] != actual[key]:
            failures.append(f"Runtime {key}: expected {expected[key]!r}, got {actual[key]!r}")
    failures += compare_maps("Imported module", {k: v["version"] for k, v in expected["imports"].items()},
                             {k: v["version"] for k, v in actual["imports"].items()})
    def native_versions(runtime):
        keys = ("internal_api", "prefix", "version", "threading_layer", "user_api",
                "current_backend", "loaded_backends", "num_threads")
        return sorted(json.dumps({key: row.get(key) for key in keys}, sort_keys=True)
                      for row in runtime["threadpools"])
    if native_versions(expected) != native_versions(actual):
        failures.append(f"Native thread-pool libraries differ: expected {native_versions(expected)!r}, got {native_versions(actual)!r}")
    # Installation roots and host/device inventory can differ legitimately.
    differences = [f"{key} differs (review runtime records)" for key in
                   ("executable", "prefix", "base_prefix", "platform", "cpu", "modules", "gpu_devices", "environment", "numpy_config")
                   if expected[key] != actual[key]]
    # Dynamic library enumeration order is not a runtime difference.
    if (sorted(json.dumps(row, sort_keys=True) for row in expected["threadpools"]) !=
            sorted(json.dumps(row, sort_keys=True) for row in actual["threadpools"])):
        differences.append("threadpools differs (review runtime records)")
    with tempfile.TemporaryDirectory(prefix="nrt-r-verify-") as folder:
        r_check = command(["Rscript", "--vanilla", str(ROOT / "environment/capture.R"), folder])
        if r_check["returncode"]:
            failures.append(f"R probe failed: {r_check['stderr']}")
        else:
            failures += compare_maps("R package", r_inventory(snapshot / "r-packages.csv"),
                                     r_inventory(Path(folder) / "r-packages.csv"))
            failures += compare_maps("R runtime", read_csv(snapshot / "r-version.csv")[0],
                                     read_csv(Path(folder) / "r-version.csv")[0])
    pip_check = command([sys.executable, "-m", "pip", "check"])
    if pip_check["returncode"]:
        failures.append(f"pip check failed: {pip_check['stdout']}{pip_check['stderr']}")
    failures += checksum_errors(snapshot / "DATA_SHA256SUMS", ROOT)
    if check_source:
        failures += checksum_errors(snapshot / "SOURCE_SHA256SUMS", ROOT)
    if require_cuda:
        import torch
        try:
            x = torch.arange(16., device="cuda").reshape(4, 4)
            result = (x @ x.T).cpu()
            torch.testing.assert_close(result, x.cpu() @ x.cpu().T)
        except Exception as error:
            failures.append(f"CUDA calculation failed: {error}")
    return dict(passed=not failures, failures=failures, differences=differences,
                python_packages=len(actual_packages), require_cuda=require_cuda,
                note="Matching inventories/build settings do not prove identical installed binaries or numerical results. Review native-library/hardware differences and compare experiment outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "environment/fir-20261021")
    parser.add_argument("--check-source", action="store_true")
    parser.add_argument("--require-cuda", action="store_true", help="Run only on an allocated GPU node")
    args = parser.parse_args()
    report = verify(args.snapshot.resolve(), args.check_source, args.require_cuda)
    print(json.dumps(report, indent=2))
    raise SystemExit(not report["passed"])
