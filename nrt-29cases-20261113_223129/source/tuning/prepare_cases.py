"""Save all selected simulation cases and tau without fitting any model."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from workflow import ROOT, SCHEMA_VERSION, atomic_json, generate, horizon, select_cases


def prepare(output, cases, *, n_samples=5000, seed=123, mu=0., r=None):
    for case in cases:
        data_seed = seed + case.index * 10_000
        tau = horizon(case, mu=mu, r=r)
        data = generate(case, n_samples, data_seed, tau, mu=mu, r=r)
        manifest = dict(schema_version=SCHEMA_VERSION, case=asdict(case), case_id=case.id,
                        seed=data_seed, mu=mu, r=r, **data.metadata())
        path = Path(output) / 'cases' / f'{case.id}.json'
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError(f'{path} contains different data settings; use a new output directory')
        else:
            atomic_json(path, manifest)
    print(f'Prepared {len(cases)} case manifests in {Path(output) / "cases"}; no models fitted.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tuning/output')
    parser.add_argument('--cases', default='all')
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--mu', type=float, default=0.)
    parser.add_argument('--r', type=float)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.n_samples < 10 or args.threads < 1:
        parser.error('n-samples must be at least 10 and threads must be positive')
    cases = select_cases(args.cases)
    if args.dry_run:
        print('\n'.join(case.id for case in cases))
        return
    torch.set_num_threads(args.threads)
    prepare(args.output_dir, cases, n_samples=args.n_samples, seed=args.seed, mu=args.mu, r=args.r)


if __name__ == '__main__':
    main()
