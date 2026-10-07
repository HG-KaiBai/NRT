"""Run Deep Distribution Regression on uncensored synthetic data."""

from __future__ import annotations

import argparse

import torch

from ..common import benchmark_model
from .model import DeepDistributionRegression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--censor-type", choices=("exact",), default="exact")
    parser.add_argument(
        "--scenario",
        choices=("ph", "lognormal", "po", "multimodal"),
        default="ph",
    )
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-cutpoints", type=int, default=None)
    parser.add_argument("--cut-fraction", type=float, default=0.10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-path", default="")
    parser.add_argument("--multimodal-case", choices=("separate", "closer"), default="separate")
    args = parser.parse_args()
    model = DeepDistributionRegression(
        n_cutpoints=args.n_cutpoints,
        cut_fraction=args.cut_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    result = benchmark_model(
        model,
        method="Deep Distribution Regression",
        scenario=args.scenario,
        multimodal_case=args.multimodal_case,
        censor_type="exact",
        n_samples=args.n_samples,
        seed=args.seed,
        grid_size=args.grid_size,
        device=args.device,
    )
    print(f"Average grid_sse: {result.average_grid_sse:.6f}")
    if args.result_path:
        result.save_json(args.result_path)


if __name__ == "__main__":
    main()
