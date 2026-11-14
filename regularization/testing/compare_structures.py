"""Compare paired S(tau) absolute bias across frozen NRT structures."""
import argparse
import json
from pathlib import Path
import statistics

from .simulation import directory_lock, has_tau_bias, result_path
from ..workflow import select_cases, select_methods, write_csv


def compare(runs, output, cases, methods, replications, *, allow_incomplete=False):
    """Write one nine-column case table and a long summary; never fill missing bias with zero."""
    summary, pending = [], []
    for case in cases:
        rows, contracts = [], {}
        complete = True
        for replication in range(1, replications+1):
            row = dict(replication=replication)
            paired = None
            for structure, run in runs.items():
                for method in methods:
                    column = f"{structure}/{method}"
                    path = result_path(run, method, case, replication)
                    saved = json.loads(path.read_text()) if path.exists() else {}
                    row[column] = None
                    if saved.get("status") != "ok" or not has_tau_bias(saved):
                        complete = False
                        continue
                    if (saved["case_id"], saved["method"], saved["replication"]) != (case.id, method, replication):
                        raise ValueError(f"invalid simulation identity in {path}")
                    dataset = {k: saved[k] for k in ("data_seed", "dataset_sha256", "n_samples", "tau", "mu", "r",
                                                   "survival_tau", "survival_tau_truth_mean")}
                    if paired is not None and paired != dataset:
                        raise ValueError(f"structures have different data or tau: {case.id} replication {replication}")
                    paired = dataset
                    contract = {k: saved[k] for k in ("tuning_hash", "source_sha256", "parameters", "training_config",
                                                     "optimization", "refit_epochs", "simulation_source_sha256")}
                    contract["evaluation_source_sha256"] = saved["tau_bias_evaluation"]["evaluation_source_sha256"]
                    if column in contracts and contracts[column] != contract:
                        raise ValueError(f"mixed configurations in {column} {case.id}")
                    contracts[column] = contract
                    row[column] = saved["survival_tau_absolute_bias"]
            rows.append(row)
        if not complete:
            pending.append(case.id)
            if not allow_incomplete:
                continue
        means, deviations = dict(replication="average"), dict(replication="standard_deviation")
        for structure in runs:
            for method in methods:
                column = f"{structure}/{method}"
                values = [row[column] for row in rows if row[column] is not None]
                means[column] = statistics.fmean(values) if values else None
                deviations[column] = statistics.stdev(values) if len(values) > 1 else (0. if values else None)
                summary.append(dict(case_id=case.id, structure=structure, method=method,
                    n_replications=len(values), expected_replications=replications,
                    mean_absolute_bias=means[column], std_absolute_bias=deviations[column], complete=complete))
        write_csv(Path(output)/"tables_tau_absolute_bias"/f"{case.id}.csv", rows+[means, deviations])
    # Preserve an existing complete global summary while jobs are pending.
    if not pending or allow_incomplete:
        write_csv(Path(output)/"tau_absolute_bias_summary.csv", summary)
    return pending


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=TESTING_DIRECTORY")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cases", default="all")
    p.add_argument("--methods", default="all")
    p.add_argument("--replications", type=int, default=500)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args(argv)
    runs = {}
    for value in args.run:
        label, separator, directory = value.partition("=")
        if not separator or not label or label in runs or not Path(directory).is_dir():
            p.error("each --run needs a unique label and an existing testing directory")
        runs[label] = Path(directory)
    if len(runs) < 2 or args.replications < 1:
        p.error("need at least two structures and positive replications")
    with directory_lock(args.output_dir, blocking=True):
        pending = compare(runs, args.output_dir, select_cases(args.cases), select_methods(args.methods),
                          args.replications, allow_incomplete=args.allow_incomplete)
    print(f"S(tau) absolute bias comparison: pending={len(pending)}")
    if pending and not args.allow_incomplete:
        raise SystemExit("missing absolute bias; run backfill-tau-bias for each structure first")


if __name__ == "__main__":
    main()
