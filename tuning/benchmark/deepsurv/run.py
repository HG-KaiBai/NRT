"""Run the official-architecture PyTorch DeepSurv benchmark on CUDA."""

from __future__ import annotations

import argparse

import torch

from ..common import benchmark_model
from ...testdata2 import MULTIMODAL_SCALE_MULTIPLIER_CASES
from .cuda_model import DeepSurvCUDA, PAPER_GAUSSIAN_PARAMETERS


def _parse_hidden_layers(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated integer widths"
        ) from error
    if not widths or any(width < 1 for width in widths):
        raise argparse.ArgumentTypeError("hidden-layer widths must be positive")
    return widths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--censor-type", choices=("exact", "right"), default="right")
    parser.add_argument(
        "--scenario",
        choices=("ph", "lognormal", "po", "multimodal"),
        default="ph",
    )
    parser.add_argument(
        "--hidden-layers", type=_parse_hidden_layers
    )
    parser.add_argument(
        "--learning-rate", type=float
    )
    parser.add_argument("--lr-decay", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--l2-reg", type=float)
    parser.add_argument("--l1-reg", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--optimizer", choices=("sgd", "adam"))
    parser.add_argument("--tuning-evals", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--tuning-epochs", type=int)
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
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device (the paper reproduction uses CUDA)",
    )
    parser.add_argument("--result-path", default="")
    parser.set_defaults(
        **PAPER_GAUSSIAN_PARAMETERS,
        tuning_epochs=PAPER_GAUSSIAN_PARAMETERS["epochs"],
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")
    device = args.device
    model = DeepSurvCUDA(
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        momentum=args.momentum,
        l2_reg=args.l2_reg,
        l1_reg=args.l1_reg,
        dropout=args.dropout,
        epochs=args.epochs,
        optimizer=args.optimizer,
        seed=args.seed,
        device=device,
        tune=args.tune,
        tuning_evals=args.tuning_evals,
        cv_folds=args.cv_folds,
        tuning_epochs=args.tuning_epochs,
    )
    result = benchmark_model(
        model,
        method="DeepSurv (official architecture, PyTorch)",
        scenario=args.scenario,
        censor_type=args.censor_type,
        n_samples=args.n_samples,
        censor_q=args.censor_q,
        seed=args.seed,
        grid_size=args.grid_size,
        device=device,
        multimodal_case=args.multimodal_case,
    )
    print(f"Modeling parameters: {model.modeling_parameters}")
    print(f"Tuning parameters: {model.tuning_parameters}")
    print(f"Average grid_sse: {result.average_grid_sse:.6f}")
    if args.result_path:
        result.save_json(
            args.result_path,
            modeling_parameters=model.modeling_parameters,
            tuning_parameters=model.tuning_parameters,
        )


if __name__ == "__main__":
    main()
