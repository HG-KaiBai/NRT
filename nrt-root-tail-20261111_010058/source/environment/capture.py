"""Capture the active Python/R environment without installing or changing packages."""
import argparse
import contextlib
from datetime import datetime
import hashlib
import importlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = ("torch", "numpy", "scipy", "pandas", "optuna", "lifelines",
           "sklearn", "sksurv", "matplotlib", "PIL")
ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE", "R_LIBS_USER", "FLEXIBLAS", "LANG", "LC_ALL")


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def command(args):
    try:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=180)
        return dict(command=args, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=args, returncode=-1, stdout="", stderr=str(error))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def python_packages():
    packages = []
    for dist in metadata.distributions():
        direct = dist.read_text("direct_url.json")
        record = dist.read_text("RECORD")
        packages.append(dict(name=dist.metadata["Name"], version=dist.version,
                             location=str(dist.locate_file("")), requires=dist.requires or [],
                             direct_url=json.loads(direct) if direct else None,
                             wheel=dist.read_text("WHEEL"),
                             record_sha256=hashlib.sha256(record.encode()).hexdigest() if record else None))
    return sorted(packages, key=lambda row: canonical(row["name"]))


def python_runtime():
    modules = {name: importlib.import_module(name) for name in IMPORTS}
    torch = modules["torch"]
    numpy_config = io.StringIO()
    with contextlib.redirect_stdout(numpy_config):
        modules["numpy"].show_config()
    from threadpoolctl import threadpool_info
    cpu = {}
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().split("\n\n")[0].splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in ("vendor_id", "model name", "flags"):
                    cpu[key.strip()] = value.strip()
    return dict(
        python=platform.python_version(), python_build=sys.version,
        executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
        compiler=platform.python_compiler(), soabi=sysconfig.get_config_var("SOABI"),
        platform=platform.platform(), machine=platform.machine(),
        libc=list(platform.libc_ver()), cpu=cpu,
        os_release=Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else None,
        modules=os.environ.get("LOADEDMODULES", "").split(":"),
        environment={key: os.environ.get(key) for key in ENV_KEYS},
        imports={name: dict(version=str(module.__version__), path=module.__file__)
                 for name, module in modules.items()},
        torch_build=torch.__config__.show(), torch_git_version=torch.version.git_version,
        cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        cuda_available=torch.cuda.is_available(),
        nvidia_smi=command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        gpu_devices=[dict(name=torch.cuda.get_device_name(i),
                          capability=list(torch.cuda.get_device_capability(i)))
                     for i in range(torch.cuda.device_count())],
        numeric_settings=dict(default_dtype=str(torch.get_default_dtype()),
                              intraop_threads=torch.get_num_threads(),
                              interop_threads=torch.get_num_interop_threads(),
                              deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                              cudnn_deterministic=torch.backends.cudnn.deterministic,
                              cudnn_benchmark=torch.backends.cudnn.benchmark,
                              matmul_precision=torch.get_float32_matmul_precision(),
                              matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                              cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
        numpy_config=numpy_config.getvalue(), threadpools=threadpool_info())


def source_files():
    """Avoid walking experiment outputs, caches, and old environment snapshots."""
    files = [p for p in ROOT.iterdir() if p.is_file() and
             (p.suffix in (".py", ".sh") or p.name == "requirements.txt")]
    files.append(ROOT / "environment/source-compatibility.json")
    for name in ("regularization", "realworld", "testing", "tuning"):
        for folder, dirs, names in os.walk(ROOT / name):
            dirs[:] = [d for d in dirs if not d.startswith((".", "output", "log", "hyper_para", "__pycache__"))
                       and d not in ("runs", "results", "archive")]
            for filename in names:
                path = Path(folder) / filename
                if (path.suffix in (".py", ".R", ".sh") or filename == "nrt_search_spaces.json"
                        or filename.startswith("requirements") and path.suffix == ".txt"):
                    files.append(path)
    return sorted(set(files))


def write_checksums(destination, files, root):
    destination.write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(files)))


def capture(output):
    output.mkdir(parents=True, exist_ok=False)
    packages = python_packages()
    write_json(output / "python-packages.json", packages)
    (output / "python-requirements.txt").write_text("".join(
        f"{row['name']}=={row['version']}\n" for row in packages))
    print(f"Captured {len(packages)} Python distributions; probing imports and native libraries...", flush=True)
    runtime = python_runtime()
    write_json(output / "runtime.json", runtime)
    (output / "modules.txt").write_text("\n".join(runtime["modules"]) + "\n")
    venv = Path(sys.prefix) / "pyvenv.cfg"
    if venv.exists():
        (output / "python-pyvenv.cfg").write_bytes(venv.read_bytes())
    pip_check = command([sys.executable, "-m", "pip", "check"])
    freeze = command([sys.executable, "-m", "pip", "freeze", "--all"])
    (output / "python-freeze.txt").write_text(freeze["stdout"])
    r_check = command(["Rscript", "--vanilla", str(ROOT / "environment/capture.R"), str(output)])
    write_json(output / "validation.json", dict(pip_check=pip_check, pip_freeze=freeze,
                                               r_capture=r_check, gpu_training_tested=False))
    write_json(output / "provenance.json", dict(
        captured_at=datetime.now(ZoneInfo("America/New_York")).isoformat(),
        host=platform.node(), git_head=command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        git_status=command(["git", "status", "--short", "--untracked-files=normal"])["stdout"],
        note="SOURCE_SHA256SUMS includes working-tree edits; Git HEAD alone is insufficient."))
    write_checksums(output / "SOURCE_SHA256SUMS", source_files(), ROOT)
    write_checksums(output / "DATA_SHA256SUMS", [
        ROOT / "realworld/data/gt_2011.csv",
        ROOT / "realworld/data/rotterdam.csv"], ROOT)
    write_checksums(output / "SHA256SUMS", [p for p in output.iterdir() if p.is_file()], output)
    if any(check["returncode"] for check in (pip_check, freeze, r_check)):
        raise RuntimeError(f"An environment probe failed; inspect {output / 'validation.json'}")
    print(f"Snapshot saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing snapshots are never overwritten")
    args = parser.parse_args()
    capture(args.output_dir.resolve())
