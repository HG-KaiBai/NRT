"""Run the NNCDE benchmark on exact or right-censored synthetic data."""

from __future__ import annotations

import argparse

import torch

from ..common import benchmark_model
from .model import NNCDE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--censor-type", choices=("exact", "right"), default="right")
    parser.add_argument(
        "--scenario",
        choices=("ph", "lognormal", "po", "multimodal"),
        default="ph",
    )
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--censor-q", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-nodes", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-path", default="")
    parser.add_argument("--multimodal-case", choices=("separate", "closer"), default="separate")
    args = parser.parse_args()
    model = NNCDE(
        hidden_nodes=args.hidden_nodes,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    result = benchmark_model(
        model,
        method="NNCDE",
        scenario=args.scenario,
        multimodal_case=args.multimodal_case,
        censor_type=args.censor_type,
        n_samples=args.n_samples,
        censor_q=args.censor_q,
        seed=args.seed,
        grid_size=args.grid_size,
        device=args.device,
    )
    print(f"Average grid_sse: {result.average_grid_sse:.6f}")
    if args.result_path:
        result.save_json(args.result_path)


if __name__ == "__main__":
    main()
