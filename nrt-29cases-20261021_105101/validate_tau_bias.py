"""Read-only validation of every retained request before test-only backfill."""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from submit_tau_bias import ROOT, RUNS, save

FIELDS = ("schema_version", "method", "case", "case_id", "replication", "data_seed", "n_samples", "base_seed",
    "grid_size", "simulation_source_sha256", "parameters", "training_config", "optimization", "refit_epochs",
    "penalty", "tau", "mu", "r", "source_sha256", "tuning_hash")
SETTINGS = ("ph", "lognormal", "po", "multimodal_separate", "multimodal_closer")
CENSORS = ("exact", "right", "current", "interval")


def read(path):
    return json.loads(path.read_text())


def main():
    paired = {}
    with ThreadPoolExecutor(max_workers=32) as pool:
        for structure, run in RUNS.items():
            count, bias, hours = 0, 0, {}
            for csv_path in sorted((run/"partition_kl/tuning/hyper_para").glob("*.csv")):
                method = csv_path.stem
                elapsed = 0.
                with csv_path.open(newline="") as stream:
                    specs = list(csv.DictReader(stream))
                for spec in specs:
                    case_id = spec["data_types"]
                    paths = [run/"partition_kl/testing"/method/case_id/f"replication_{rep:04d}.json" for rep in range(1, 501)]
                    expected_json = {k: json.loads(spec[k]) for k in ("parameters", "training_config", "optimization")}
                    for replication, (path, saved) in enumerate(zip(paths, pool.map(read, paths)), 1):
                        assert saved["status"] == "ok", str(path)
                        assert (saved["method"], saved["case_id"], saved["replication"]) == (method, case_id, replication), str(path)
                        request = {k: saved[k] for k in FIELDS}
                        assert hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest() == saved["request_hash"], str(path)
                        assert saved["simulation_source_sha256"] == "cef0e3a012b09902d875c2a8cbe7937954849de204e8569bd64bde36e4e3beff", str(path)
                        for k, v in expected_json.items():
                            assert saved[k] == v, (str(path), k)
                        for k in ("source_sha256", "tuning_hash", "penalty"):
                            assert saved[k] == spec[k], (str(path), k)
                        assert saved["refit_epochs"] == int(spec["refit_epochs"]), str(path)
                        assert saved["n_samples"] == int(spec["n_samples"]) == 5000 and saved["base_seed"] == int(spec["base_seed"]) == 123 and saved["grid_size"] == 256, str(path)
                        assert saved["tau"] == (float(spec["tau"]) if spec["tau"] else None), str(path)
                        assert saved["mu"] == float(spec["mu"]) and saved["r"] == (float(spec["r"]) if spec["r"] else None), str(path)
                        case = saved["case"]
                        index = SETTINGS.index(case["setting"])*8+CENSORS.index(case["censor_type"])*2+(0 if case["q"] == .2 else 1)
                        assert saved["data_seed"] == 123+1000000+index*10000+replication, str(path)
                        key = case_id, replication
                        data = {k: saved[k] for k in ("dataset_sha256", "data_seed", "n_samples", "tau", "mu", "r", "grid_max")}
                        assert key not in paired or paired[key] == data, str(path)
                        paired[key] = data
                        count += 1
                        elapsed += saved["elapsed_seconds"]
                        bias += saved.get("tau_bias_evaluation", {}).get("status") == "ok"
                hours[method] = round(elapsed/3600, 2)
                print(f"{structure}: validated {count}/43500 requests", flush=True)
            assert count == 43500, (structure, count)
            report = dict(checked_at=datetime.now(ZoneInfo("America/New_York")).isoformat(), records_validated=count,
                existing_bias_records=bias, all_requests_match_frozen_specs=True,
                datasets_paired_across_structures_and_methods=True, original_fit_hours_by_method=hours,
                quota=dict(inode_limit=500000, used_inodes_before=362003, available_inodes_before=137997,
                           additional_inode_budget=1000, additional_storage_budget_gib=2, available_storage_gib_approx=37))
            save(run/"tau_bias_preflight.json", report)
            print(f"{structure}: {json.dumps(report)}", flush=True)


if __name__ == "__main__":
    main()
