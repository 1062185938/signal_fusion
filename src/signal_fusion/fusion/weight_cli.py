"""CLI for validation-only fusion-weight selection and frozen test evaluation."""

from __future__ import annotations

import argparse
import sys

from signal_fusion.fusion.selection import select_and_evaluate_fusion


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signal-select-fusion-weight")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--label_map_path", required=True)
    parser.add_argument("--feature_classifier_manifest", required=True)
    parser.add_argument("--feature_dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stress_snr_db", type=float, default=5.0)
    parser.add_argument("--weight_step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = select_and_evaluate_fusion(
            dataset_dir=args.dataset_dir,
            model_path=args.model_path,
            label_map_path=args.label_map_path,
            feature_classifier_manifest=args.feature_classifier_manifest,
            feature_dataset_dir=args.feature_dataset_dir,
            output_dir=args.output_dir,
            stress_snr_db=args.stress_snr_db,
            weight_step=args.weight_step,
            seed=args.seed,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    weights = result["selected_manifest"]["weights"]
    print("Fusion weight selection complete")
    print(f"  IQ model weight:          {weights['iq_model']:.3f}")
    print(
        "  feature classifier weight: "
        f"{weights['feature_classifier']:.3f}"
    )
    for split in ("validation", "test"):
        metrics = result[split]["overall"]
        print(
            f"  {split:<10} IQ={metrics['iq_model']['accuracy_percent']:.3f}% "
            f"feature={metrics['feature_classifier']['accuracy_percent']:.3f}% "
            f"fusion={metrics['fusion']['accuracy_percent']:.3f}%"
        )
    print(f"  manifest: {result['outputs']['manifest']}")
    print(f"  report:   {result['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "main"]
