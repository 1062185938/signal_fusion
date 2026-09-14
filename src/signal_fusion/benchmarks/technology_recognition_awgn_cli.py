"""CLI for summarizing the selected public-dataset AWGN evaluations."""

from __future__ import annotations

import argparse

from .technology_recognition_awgn import write_awgn_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize selected AWGN evaluations across folds."
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=(512, 4096),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = write_awgn_summary(
        model_root=args.model_root,
        output_dir=args.output_dir,
        window_sizes=tuple(args.window_sizes),
        plot=not args.no_plot,
    )
    print("Window  Condition  Window accuracy       Region accuracy")
    for window_size in result["window_sizes"]:
        conditions = result["summaries"][str(window_size)]["conditions"]
        for condition in conditions.values():
            window = condition["window_accuracy_percent"]
            region = condition["region_accuracy_percent"]
            print(
                f"{window_size:>6}  {condition['label']:<9}  "
                f"{window['mean']:7.3f}% ± {window['std']:.3f}  "
                f"{region['mean']:7.3f}% ± {region['std']:.3f}"
            )
    print(f"JSON:        {result['outputs']['json']}")
    print(f"Summary CSV: {result['outputs']['summary_csv']}")
    print(f"Fold CSV:    {result['outputs']['fold_csv']}")
    if result["outputs"]["plot"] is not None:
        print(f"Plot:        {result['outputs']['plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "main"]
