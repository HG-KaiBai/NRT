"""Run the Fu--Simonoff interval-censored ctree benchmark."""

from __future__ import annotations

import argparse

from ..common import benchmark_model
from .model import CTree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--censor-type",
        choices=("exact", "right", "interval", "current"),
        default="interval",
    )
    parser.add_argument(
        "--scenario",
        choices=("ph", "lognormal", "po", "multimodal"),
        default="ph",
    )
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--censor-q", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--mincriterion", type=float, default=0.95)
    parser.add_argument("--minsplit", type=int, default=20)
    parser.add_argument("--minbucket", type=int, default=7)
    parser.add_argument("--maxdepth", type=int, default=None)
    parser.add_argument("--result-path", default="")
    parser.add_argument("--multimodal-case", choices=("separate", "closer"), default="separate")
    args = parser.parse_args()
    with CTree(
        mincriterion=args.mincriterion,
        minsplit=args.minsplit,
        minbucket=args.minbucket,
        maxdepth=args.maxdepth,
        seed=args.seed,
    ) as model:
        result = benchmark_model(
            model,
            method="ctree",
            censor_type=args.censor_type,
            scenario=args.scenario,
        multimodal_case=args.multimodal_case,
            n_samples=args.n_samples,
            censor_q=args.censor_q,
            seed=args.seed,
            grid_size=args.grid_size,
            device="cpu",
        )
    print(f"Average grid_sse: {result.average_grid_sse:.6f}")
    if args.result_path:
        result.save_json(args.result_path)


if __name__ == "__main__":
    main()
