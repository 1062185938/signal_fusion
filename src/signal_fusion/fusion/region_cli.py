"""CLI for rebuilding complete continuous regions."""

from __future__ import annotations

import argparse
import json
import sys

from signal_fusion.fusion.region_dataset import rebuild_continuous_region_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-rebuild-regions",
        description=(
            "Rebuild complete regions from unnormalized prepared-source windows, "
            "then apply one region-level DC removal and RMS normalization."
        ),
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--prepared_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = rebuild_continuous_region_dataset(
            args.dataset_path,
            args.prepared_root,
            args.output_path,
            overwrite=args.overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "main"]
