"""Run the linear Cox proportional-hazards benchmark."""

from __future__ import annotations

import argparse

from ..common import benchmark_model
from ...testdata2 import MULTIMODAL_SCALE_MULTIPLIER_CASES
from .model import CoxPH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--censor-type",
        choices=("exact", "right", "interval", "current"),
        default="right",
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
    parser.add_argument(
        "--multimodal-case",
        choices=tuple(MULTIMODAL_SCALE_MULTIPLIER_CASES),
        default="separate",
        help="Multimodal peaks: separate or closer",
    )
    parser.add_argument("--result-path", default="")
    args = parser.parse_args()

    with CoxPH() as model:
        result = benchmark_model(
            model,
            method="Cox PH",
            scenario=args.scenario,
            censor_type=args.censor_type,
            n_samples=args.n_samples,
            censor_q=args.censor_q,
            seed=args.seed,
            grid_size=args.grid_size,
            device="cpu",
            multimodal_case=args.multimodal_case,
        )
    print(f"Average grid_sse: {result.average_grid_sse:.6f}")
    if args.result_path:
        result.save_json(args.result_path)


if __name__ == "__main__":
    main()
