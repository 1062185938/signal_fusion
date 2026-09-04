"""CLI for controlled SNR robustness evaluation."""

from __future__ import annotations

import argparse
import sys

from signal_fusion.evaluation import evaluate_snr_robustness


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-evaluate-snr",
        description=(
            "Evaluate a trained IQ classifier on clean and controlled complex-AWGN "
            "versions of a fixed test split."
        ),
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--snr_db",
        nargs="+",
        type=float,
        default=[20.0, 15.0, 10.0, 5.0, 0.0, -5.0, -10.0],
        help="Added complex-AWGN SNR values in dB.",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model_name", default="deepconvnet_1d")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = evaluate_snr_robustness(
            dataset_dir=args.dataset_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            snr_db_values=args.snr_db,
            trials=args.trials,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
            model_name=args.model_name,
            overwrite=args.overwrite,
            plot=not args.no_plot,
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

    print("\nSNR robustness evaluation complete")
    print(f"  dataset_id: {result['dataset_id']}")
    print(f"  device:     {result['device']}")
    print(f"  windows:    {result['sample_count']}")
    print(f"  regions:    {result['group_count']}")
    print("  condition       window accuracy       region accuracy")
    for condition in result["conditions"]:
        label = (
            "clean"
            if condition["snr_db"] is None
            else f"{condition['snr_db']:g} dB"
        )
        window = condition["aggregate"]["window_accuracy_percent"]
        group = condition["aggregate"]["group_accuracy_percent"]
        print(
            f"  {label:<12}  {window['mean']:7.3f}% ± {window['std']:.3f}  "
            f"{group['mean']:7.3f}% ± {group['std']:.3f}"
        )
    print(f"  JSON: {result['outputs']['json']}")
    print(f"  CSV:  {result['outputs']['csv']}")
    if result["outputs"]["plot"] is not None:
        print(f"  plot: {result['outputs']['plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "main"]
