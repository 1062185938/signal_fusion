"""Command line entry point for training-dataset assembly."""

from __future__ import annotations

import argparse
import json

from .assembly import assemble_training_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble prepared IQ slices into fixed, balanced train/validation/test "
            "datasets."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the JSON split manifest.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for train.npz, validation.npz, test.npz, and the audit report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace assembly outputs that already exist.",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    result = assemble_training_dataset(
        args.manifest,
        args.output_dir,
        overwrite=args.overwrite,
    )
    summary = {
        "dataset_id": result.report["dataset_id"],
        "splits": {
            split_name: {
                "path": str(path),
                "shape": result.report["splits"][split_name]["shape"],
                "class_counts": result.report["splits"][split_name][
                    "class_counts"
                ],
            }
            for split_name, path in result.split_paths.items()
        },
        "report_path": str(result.report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
