"""Refit frozen parameters on paired NRT simulation datasets and summarize."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

from ..train import fit_raw_time_kl
from ..unified_nll.nll_cv_train import NLLCVOptimization
from ..unified_nll.train import TrainingConfig
from ..unified_nll.tuning_model import Architecture
from ..workflow import (ROOT, SCHEMA_VERSION, atomic_json, dataset_hash, evaluate,
                        fingerprint, load_hyperparameters, preflight, select_cases,
                        select_methods, simulation_data, source_hash, write_csv, writer_lock)


def result_path(output, method, case, replication):
    return output/method/case.id/f"replication_{replication:04d}.json"


def make_request(args, method, case, replication, spec):
    if args.n_samples != spec["n_samples"] or args.base_seed != spec["base_seed"]:
        raise ValueError("n-samples/base-seed must match tuning to preserve the existing dataset protocol")
    if spec["source_sha256"] != source_hash():
        raise ValueError("tuning used different numerical sources; retune into a fresh directory")
    return dict(schema_version=SCHEMA_VERSION, method=method, case=asdict(case), case_id=case.id,
        replication=replication,
        data_seed=args.base_seed+1_000_000+case.index*10_000+replication,
        n_samples=args.n_samples, base_seed=args.base_seed, grid_size=args.grid_size,
        simulation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **{
            key: spec[key] for key in ("parameters", "training_config", "optimization", "refit_epochs", "penalty",
                                      "tau", "mu", "r", "source_sha256", "tuning_hash")})


def fit_replication(args, method, case, replication, spec):
    request = make_request(args, method, case, replication, spec)
    request_hash = fingerprint(request)
    path = result_path(args.output_dir, method, case, replication)
    with writer_lock(path.parent, f".replication_{replication:04d}.lock"):
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("request_hash") != request_hash:
                raise ValueError(f"{path} belongs to a different simulation; use a fresh output directory")
            if old.get("status") == "ok":
                return old
        start = time.monotonic()
        result = dict(request, request_hash=request_hash, status="error")
        try:
            data, seed = simulation_data(case, spec, replication,
                n_samples=args.n_samples, base_seed=args.base_seed)
            result.update(data.metadata(), dataset_sha256=dataset_hash(data))
            # No validation/test data enter fitting, and no epoch/search override is accepted.
            model, _ = fit_raw_time_kl(data.bounds(data.train), architecture=Architecture(**spec["parameters"]),
                training=TrainingConfig(**spec["training_config"]),
                optimization=NLLCVOptimization(**spec["optimization"]),
                fixed_epochs=spec["refit_epochs"], seed=seed+1, device=args.device, penalty=spec["penalty"])
            result.update(evaluate(model, data, args.grid_size), fit_info=model.fit_info, status="ok")
            del model
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed_seconds"] = time.monotonic()-start
        atomic_json(path, result)
        print(f"{method} {case.id} replication={replication} status={result['status']}", flush=True)
        return result


def summarize(output, cases, methods, replications, *, allow_incomplete=False):
    """Publish complete case tables; preserve previous reports while fits are pending."""
    output = Path(output)
    complete_cases, pending, all_rates = [], [], []
    for case in cases:
        rows, contracts, datasets, rates = [], {}, {}, []
        complete = True
        for replication in range(1, replications+1):
            row = dict(replication=replication)
            for method in methods:
                path = result_path(output, method, case, replication)
                saved = json.loads(path.read_text()) if path.exists() else {}
                if saved.get("status") != "ok":
                    complete = False
                    row[method] = None
                    continue
                if (saved["case_id"] != case.id or saved["method"] != method
                        or saved["replication"] != replication):
                    raise ValueError(f"invalid simulation identity in {path}")
                contract = {k: saved[k] for k in ("tuning_hash", "parameters", "training_config",
                    "optimization", "refit_epochs", "source_sha256", "simulation_source_sha256", "grid_size")}
                contract["penalty"] = saved.get("penalty", "partition_kl")
                if method in contracts and contracts[method] != contract:
                    raise ValueError(f"mixed configurations for {method} {case.id}")
                contracts[method] = contract
                dataset = {k: saved[k] for k in ("data_seed", "dataset_sha256", "n_samples", "tau", "mu", "r",
                                                "grid_size", "grid_max")}
                if replication in datasets and datasets[replication] != dataset:
                    raise ValueError(f"methods have different datasets for {case.id} replication {replication}")
                if replication not in datasets:
                    rates.append(saved["right_censoring_rate"])
                    datasets[replication] = dataset
                row[method] = saved["average_grid_sse"]
            rows.append(row)
        if not complete:
            pending.append(case.id)
            if not allow_incomplete:
                continue
        means, deviations = dict(replication="average"), dict(replication="standard_deviation")
        for method in methods:
            values = [row[method] for row in rows if row[method] is not None]
            means[method] = statistics.fmean(values) if values else None
            deviations[method] = statistics.stdev(values) if len(values) > 1 else (0. if values else None)
        write_csv(output/"tables"/f"{case.id}.csv", rows+[means, deviations])
        if complete:
            complete_cases.append(case.id)
            all_rates.append(dict(case_id=case.id, replications=replications,
                tau=next(iter(datasets.values()))["tau"],
                mean_right_censoring_rate=statistics.fmean(rates),
                std_right_censoring_rate=statistics.stdev(rates) if len(rates) > 1 else 0.))
    if not pending:
        write_csv(output/"censoring_summary.csv", all_rates)
    return complete_cases, pending


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "summarize"))
    p.add_argument("--methods", default="all")
    p.add_argument("--summary-methods", help="methods required for automatically published tables")
    p.add_argument("--cases", default="all")
    p.add_argument("--hyper-dir", type=Path, default=ROOT/"tuning/output/hyper_para")
    p.add_argument("--output-dir", type=Path, default=ROOT/"testing/output")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--replications", type=int, default=500)
    p.add_argument("--replication", type=int)
    p.add_argument("--start-replication", type=int, default=1)
    p.add_argument("--end-replication", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cases, methods = select_cases(args.cases), select_methods(args.methods)
    if args.replications < 1 or args.n_samples < 10 or args.grid_size < 3:
        raise ValueError("invalid sample/replication/grid counts")
    if args.command == "summarize":
        with writer_lock(args.output_dir, ".summary.lock"):
            complete, pending = summarize(args.output_dir, cases, methods, args.replications,
                                          allow_incomplete=args.allow_incomplete)
        print(f"published {len(complete)} complete case tables; pending={len(pending)}")
        if pending and not args.allow_incomplete:
            raise SystemExit("missing or failed replications; complete case tables were retained")
        return
    last = args.end_replication if args.end_replication is not None else args.replications
    replications = [args.replication] if args.replication is not None else range(args.start_replication, last+1)
    if not replications or any(not 1 <= rep <= args.replications for rep in replications):
        raise ValueError("replication range must lie within 1..replications")
    if args.dry_run:
        print(f"{len(cases)} cases x {len(methods)} methods x {len(replications)} replications")
        return
    tables = load_hyperparameters(args.hyper_dir, cases, methods)
    # Validate every frozen input before any fitting or partial output.
    for method in methods:
        for case in cases:
            make_request(args, method, case, replications[0], tables[method][case.id])
    preflight(args.device, args.threads)
    failures = 0
    for case in cases:
        for replication in replications:
            for method in methods:
                result = fit_replication(args, method, case, replication, tables[method][case.id])
                failures += result["status"] != "ok"
    summary_methods = select_methods(args.summary_methods) if args.summary_methods else methods
    # A blocking reporting lock lets the final finisher see every prior atomic JSON.
    import fcntl
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/".summary.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        complete, pending = summarize(args.output_dir, cases, summary_methods, args.replications)
    print(f"published {len(complete)} complete case tables; pending={len(pending)}; failures={failures}")
    if failures:
        raise SystemExit(f"{failures} fits failed; rerun to retry saved errors")


if __name__ == "__main__":
    main()
