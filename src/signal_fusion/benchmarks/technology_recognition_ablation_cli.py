"""CLI for the public-dataset location/window ablation evaluation."""

from __future__ import annotations

import argparse

from .technology_recognition_ablation import evaluate_ablation_matrix


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the 4-fold by 5-window public benchmark matrix."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=(128, 512, 1024, 2048, 4096),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = evaluate_ablation_matrix(
        dataset_root=args.dataset_root,
        model_root=args.model_root,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        window_sizes=tuple(args.window_sizes),
    )
    print("Window  Window accuracy       Region accuracy")
    for window_size in result["window_sizes"]:
        summary = result["summaries"][str(window_size)]
        window = summary["window_accuracy_percent"]
        region = summary["region_accuracy_percent"]
        print(
            f"{window_size:>6}  {window['mean']:7.3f}% ± {window['std']:.3f}  "
            f"{region['mean']:7.3f}% ± {region['std']:.3f}"
        )
    print(f"JSON: {result['outputs']['json']}")
    print(f"CSV:  {result['outputs']['csv']}")
    print(f"Plot: {result['outputs']['plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
