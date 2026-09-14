"""CLI for the public-dataset location/window ablation evaluation."""

from __future__ import annotations

import argparse

from .technology_recognition_ablation import (
    evaluate_ablation_matrix,
    evaluate_external_test_set,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the 4-fold by 5-window public benchmark matrix."
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset-root")
    dataset_group.add_argument(
        "--external-test",
        help="Evaluate all four checkpoints against one external test NPZ.",
    )
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
    if args.external_test is not None:
        result = evaluate_external_test_set(
            dataset_path=args.external_test,
            model_root=args.model_root,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        print("Model          Accuracy   Macro accuracy")
        for model_result in [*result["models"], result["ensemble"]]:
            metrics = model_result["subsets"]["all"]
            print(
                f"{model_result['model_id']:<14}  "
                f"{metrics['accuracy_percent']:7.3f}%  "
                f"{metrics['macro_accuracy_percent']:7.3f}%"
            )
        agreement = result["model_agreement"]["all"]
        print(
            "All-four agreement: "
            f"{agreement['all_four_agree_ratio'] * 100.0:.3f}%"
        )
        print(f"JSON:       {result['outputs']['json']}")
        print(f"Summary CSV:{result['outputs']['summary_csv']}")
        print(f"Source CSV: {result['outputs']['source_csv']}")
        return 0

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
