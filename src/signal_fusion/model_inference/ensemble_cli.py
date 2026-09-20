"""CLI for region-level inference with multiple independently trained models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from signal_fusion.io import load_prepared_dataset
from signal_fusion.model_inference.ensemble import (
    RegionEnsembleInferenceService,
    load_ensemble_services,
)
from signal_fusion.model_inference.ensemble_evaluation import (
    evaluate_ensemble_results,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-infer-ensemble",
        description=(
            "Run multiple ONNX IQ classifiers on region windows, average "
            "their probabilities, and infer one region or evaluate a labeled "
            "dataset."
        ),
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--model_path",
        required=True,
        nargs="+",
        help="Exactly three ONNX checkpoints with the same input and label order.",
    )
    parser.add_argument("--label_map_path", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--group_id", type=int)
    target.add_argument(
        "--all_groups",
        action="store_true",
        help="Evaluate every labeled region and report ensemble/gate metrics.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_payload(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_ensemble_inference(
    *,
    dataset_path: str | Path,
    model_paths: list[str | Path],
    label_map_path: str | Path,
    group_id: int,
    output_path: str | Path,
    batch_size: int = 64,
    overwrite: bool = False,
) -> dict:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"ensemble output exists; use --overwrite to replace it: {output}"
        )

    services = load_ensemble_services(model_paths, label_map_path)

    dataset = load_prepared_dataset(dataset_path)
    result = RegionEnsembleInferenceService(services).predict_group(
        dataset,
        group_id=group_id,
        batch_size=batch_size,
    )
    payload = result.to_dict()
    payload["input"] = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "model_paths": [str(Path(path).resolve()) for path in model_paths],
        "label_map_path": str(Path(label_map_path).resolve()),
    }
    _write_payload(payload, output)
    return payload


def run_ensemble_evaluation(
    *,
    dataset_path: str | Path,
    model_paths: list[str | Path],
    label_map_path: str | Path,
    output_path: str | Path,
    batch_size: int = 64,
    overwrite: bool = False,
) -> dict:
    """Evaluate the ensemble and unanimous-member gate on every region."""

    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"ensemble output exists; use --overwrite to replace it: {output}"
        )
    services = load_ensemble_services(model_paths, label_map_path)
    dataset = load_prepared_dataset(dataset_path)
    results = [
        service.predict(dataset, batch_size=batch_size) for service in services
    ]
    payload = evaluate_ensemble_results(dataset, results)
    payload["input"] = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "model_paths": [str(Path(path).resolve()) for path in model_paths],
        "label_map_path": str(Path(label_map_path).resolve()),
    }
    _write_payload(payload, output)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.all_groups:
            result = run_ensemble_evaluation(
                dataset_path=args.dataset_path,
                model_paths=args.model_path,
                label_map_path=args.label_map_path,
                output_path=args.output,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
        else:
            result = run_ensemble_inference(
                dataset_path=args.dataset_path,
                model_paths=args.model_path,
                label_map_path=args.label_map_path,
                group_id=args.group_id,
                output_path=args.output,
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

    if args.all_groups:
        metrics = result["ensemble"]["metrics"]["region"]
        gate = result["risk_gate"]
        print("Dataset ensemble evaluation complete")
        print(f"  regions:  {result['region_count']}")
        print(f"  accuracy: {metrics['accuracy_percent']:.6f}%")
        print(f"  macro:    {metrics['macro_accuracy_percent']:.6f}%")
        print(
            "  review:   "
            f"{gate['review_required']['region_count']} "
            f"({gate['review_required']['coverage_percent']:.6f}%)"
        )
        print(f"  output:   {args.output}")
        return 0

    top1 = result["ensemble"]["region_top3"][0]
    print("Region ensemble inference complete")
    print(f"  group_id: {result['group_id']}")
    print(f"  members:  {result['risk_gate']['member_count']}")
    print(
        f"  ensemble: {top1['label']} "
        f"({top1['probability']:.6f})"
    )
    print(f"  decision: {result['decision_status']}")
    print(f"  output:   {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_arg_parser",
    "main",
    "run_ensemble_evaluation",
    "run_ensemble_inference",
]
