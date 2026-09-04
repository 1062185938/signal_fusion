"""CLI for region-feature dataset construction and classifier training."""

from __future__ import annotations

import argparse
import json

from signal_fusion.feature_classifier.dataset import build_region_feature_dataset
from signal_fusion.feature_classifier.trainer import train_feature_classifier


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signal-feature-classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-dataset")
    build.add_argument("--dataset-dir", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--train-awgn-copies", type=int, default=1)
    build.add_argument("--awgn-snr-min", type=float, default=5.0)
    build.add_argument("--awgn-snr-max", type=float, default=20.0)
    build.add_argument("--test-awgn-snr", type=float, default=5.0)
    build.add_argument("--seed", type=int, default=44)
    build.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train")
    train.add_argument("--dataset-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-2)
    train.add_argument("--max-epochs", type=int, default=300)
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--seed", type=int, default=44)
    train.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "build-dataset":
        result = build_region_feature_dataset(
            args.dataset_dir,
            args.output_dir,
            train_awgn_copies=args.train_awgn_copies,
            awgn_snr_min=args.awgn_snr_min,
            awgn_snr_max=args.awgn_snr_max,
            test_awgn_snr=args.test_awgn_snr,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = train_feature_classifier(
        args.dataset_dir,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "main"]
