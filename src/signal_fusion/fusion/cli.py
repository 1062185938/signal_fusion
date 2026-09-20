"""Build blind ensemble-plus-feature evidence and a private audit manifest."""

from __future__ import annotations

import argparse
import json
import sys

from signal_fusion.fusion.ensemble_evidence import build_ensemble_evidence_files
from signal_fusion.model_inference import load_ensemble_services


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-build-evidence",
        description=(
            "Combine a three-model IQ ensemble, a frozen LTE/DVB-T periodicity "
            "gate, and one 64-dimensional feature vector per complete region. "
            "Writes anonymous explanation cases and a private audit manifest."
        ),
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--region_dataset_path", required=True)
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--periodicity_gate_manifest", required=True)
    parser.add_argument(
        "--model_path",
        required=True,
        nargs="+",
        help="Exactly three ONNX checkpoints with the same input and label order.",
    )
    parser.add_argument("--label_map_path", required=True)
    parser.add_argument("--output_dir", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--group_id", type=int)
    target.add_argument("--all_groups", action="store_true")
    parser.add_argument("--case_id_start", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        services = load_ensemble_services(args.model_path, args.label_map_path)
        report = build_ensemble_evidence_files(
            dataset_path=args.dataset_path,
            region_dataset_path=args.region_dataset_path,
            feature_path=args.feature_path,
            periodicity_gate_manifest_path=args.periodicity_gate_manifest,
            services=services,
            output_dir=args.output_dir,
            group_id=None if args.all_groups else args.group_id,
            case_id_start=args.case_id_start,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        IndexError,
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
