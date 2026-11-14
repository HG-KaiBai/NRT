"""Restore the three frozen numerical sources with the new testing-only evaluator.

Run with any Python 3 interpreter from the NRT root. No fitting or job submission.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "finite_1p1tau": "nrt-29cases-20261021_105101",
    "body_tail": "nrt-body-tail-20261110_140942",
    "root_tail": "nrt-root-tail-20261111_010058",
}
NUMERICAL_FILES = ["regularization/model.py", "regularization/train.py", "regularization/workflow.py",
    "workflow.py", "tuning/testdata2.py", "regularization/tuning/main.py"] + [
    f"regularization/unified_nll/{name}.py" for name in
    ("model", "tree", "train", "regularization", "tuning_model", "cv_train", "nll_cv_train")]
FILES = list(dict.fromkeys(NUMERICAL_FILES + ["runtime.sh", "node_runner.py", "cuda_retry.py", "requirements.txt",
    "regularization/unified_nll/__init__.py", "regularization/unified_nll/metrics.py",
    "regularization/unified_nll/brier.py", "regularization/tuning/__init__.py",
    "regularization/testing/__init__.py"]))
OVERLAY = [f"regularization/testing/{name}" for name in
           ("simulation.py", "run_nll.sh", "test_tau_bias.py", "compare_structures.py", "README.md")]


def numerical_hash(files):
    digest = hashlib.sha256()
    for name in NUMERICAL_FILES:
        digest.update(name.encode())
        digest.update(files[name])
    raw = digest.hexdigest()
    aliases = json.loads(files["environment/source-compatibility.json"])["aliases"]
    return raw, aliases.get(raw, raw)


def prepare():
    for label, directory in RUNS.items():
        run = ROOT/directory
        if label == "finite_1p1tau":
            names = FILES + ["environment/source-compatibility.json"]
            files = {name: subprocess.check_output(["git", "show", f"db9aa0b:{name}"], cwd=ROOT) for name in names}
        elif label == "body_tail":
            with tarfile.open(run/"source_snapshot.tar.gz") as archive:
                files = {name: archive.extractfile(name).read() for name in FILES}
            # Exact-source aliases only; the snapshot's own numerical hash is checked below.
            files["environment/source-compatibility.json"] = (ROOT/"environment/source-compatibility.json").read_bytes()
        else:
            files = {name: (run/"source"/name).read_bytes() for name in FILES + ["environment/source-compatibility.json"]}
        raw_hash, source_hash = numerical_hash(files)
        specs = []
        for path in sorted((run/"partition_kl/tuning/hyper_para").glob("*.csv")):
            with path.open(newline="") as stream:
                specs.extend(csv.DictReader(stream))
        if len(specs) != 87 or {spec["source_sha256"] for spec in specs} != {source_hash}:
            raise ValueError(f"{directory}: restored sources do not match all 87 frozen specifications")
        files.update({name: (ROOT/name).read_bytes() for name in OVERLAY})
        manifest = dict(structure=label, original_run=directory, numerical_source_sha256=source_hash,
            raw_numerical_source_sha256=raw_hash, frozen_specifications=len(specs),
            files={name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())})
        files["tau_bias_source_manifest.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
        destination = run/"tau_bias_source"
        if destination.exists():
            if any(not (destination/name).is_file() or (destination/name).read_bytes() != value for name, value in files.items()):
                raise ValueError(f"existing source differs: {destination}; do not modify a running snapshot")
        else:
            staging = Path(tempfile.mkdtemp(prefix=".tau-bias-source-", dir=run))
            try:
                for name, value in files.items():
                    path = staging/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(value)
                staging.rename(destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        print(f"{label}: {destination}; matched {len(specs)} frozen specifications ({source_hash})")


if __name__ == "__main__":
    prepare()
