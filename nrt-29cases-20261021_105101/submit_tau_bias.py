"""Submit the authorized test-only absolute-bias backfills and their comparison."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "finite_1p1tau": ROOT/"nrt-29cases-20261021_105101",
    "body_tail": ROOT/"nrt-body-tail-20261110_140942",
    "root_tail": ROOT/"nrt-root-tail-20261111_010058",
}


def now():
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def save(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(stream.name, path)


def main():
    quota = os.statvfs(ROOT)
    available_bytes = quota.f_bavail*quota.f_frsize
    if quota.f_favail < 1000 or available_bytes < 2*1024**3:
        raise RuntimeError("insufficient quota for the conservative backfill budget")
    # Check every source and preflight before the first scheduler mutation.
    for run in RUNS.values():
        if (run/"tau_bias_submission.json").exists():
            raise RuntimeError(f"submission record already exists: {run}; inspect it before resubmitting")
        preflight = json.loads((run/"tau_bias_preflight.json").read_text())
        if preflight["records_validated"] != 43500 or not preflight["all_requests_match_frozen_specs"]:
            raise RuntimeError(f"incomplete preflight: {run}")
        source = run/"tau_bias_source"
        manifest = json.loads((source/"tau_bias_source_manifest.json").read_text())
        for name, expected in manifest["files"].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"source changed after validation: {source/name}")
    records, ids = {}, []
    for structure, run in RUNS.items():
        source = run/"tau_bias_source"
        exported = dict(NRT_ROOT=str(source), COMMAND="backfill-tau-bias",
            HYPER_DIR=str(run/"partition_kl/tuning/hyper_para"), OUTDIR=str(run/"partition_kl/testing"),
            METHODS="all", SUMMARY_METHODS="all", CASES="all", N_SAMPLES="5000", BASE_SEED="123",
            REPLICATIONS="500", GRID_SIZE="256", DEVICE="cuda:0")
        environment = os.environ.copy()
        environment.pop("PY", None)
        environment.update(exported)
        command = ["sbatch", "--parsable", f"--job-name=tau_abs_bias_{structure}",
            f"--chdir={source}", "--array=0-2%3", "--export=ALL",
            f"--output={run}/tau_bias_logs/test-%A_%a.out", f"--error={run}/tau_bias_logs/test-%A_%a.err",
            str(source/"regularization/testing/run_nll.sh")]
        record = dict(created_at=now(), structure=structure, command="backfill-tau-bias", status="submitting",
            execution_source=str(source), output_dir=exported["OUTDIR"], environment=exported,
            submit_command=command, replications=500, cases=29, methods=3, expected_refits=43500,
            retuning=False, original_results_preserved=True,
            quota=dict(available_inodes=quota.f_favail, available_bytes=available_bytes,
                       additional_inode_budget=1000, additional_byte_budget=2*1024**3))
        path = run/"tau_bias_submission.json"
        save(path, record)
        try:
            submitted = subprocess.run(command, env=environment, text=True, capture_output=True, check=True, timeout=45)
            job_id = submitted.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"unexpected sbatch response: {submitted.stdout!r}")
        except Exception as error:
            record.update(status="submission_needs_inspection", error=str(error))
            save(path, record)
            raise
        record.update(status="submitted", testing_job_id=job_id, submitted_at=now(), scheduler_stderr=submitted.stderr)
        save(path, record)
        records[structure] = record
        ids.append(job_id)
        print(f"{structure}: {job_id} (three test tasks)", flush=True)
    first = next(iter(RUNS.values()))
    dependency = "afterok:"+":".join(ids)
    compare_command = ["sbatch", "--parsable", f"--chdir={first}/tau_bias_source", f"--dependency={dependency}",
        "--kill-on-invalid-dep=yes", "--export=ALL",
        f"--output={first}/tau_bias_logs/compare-%j.out", f"--error={first}/tau_bias_logs/compare-%j.err",
        str(first/"tau_bias_compare.sh")]
    compare_environment = os.environ.copy()
    compare_environment.pop("PY", None)
    compared = subprocess.run(compare_command, env=compare_environment,
                             text=True, capture_output=True, check=True, timeout=45)
    compare_id = compared.stdout.strip().split(";", 1)[0]
    if not compare_id.isdigit():
        raise RuntimeError(f"unexpected sbatch response: {compared.stdout!r}")
    for structure, run in RUNS.items():
        records[structure].update(comparison_job_id=compare_id, comparison_dependency=dependency,
            comparison_submit_command=compare_command, comparison_submitted_at=now(),
            comparison_output_dir=str(run/"partition_kl/testing/structure_comparison"))
        save(run/"tau_bias_submission.json", records[structure])
    print(f"comparison: {compare_id}; {dependency}; outputs saved in all three original run directories", flush=True)


if __name__ == "__main__":
    main()
