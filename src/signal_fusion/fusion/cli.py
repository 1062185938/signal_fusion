"""Build separated blind-input and private-audit evidence files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from signal_fusion.feature_extraction import FeatureExtractionService
from signal_fusion.feature_classifier import FeatureClassifierService
from signal_fusion.fusion import (
    FusionManifest,
    analyze_group,
    build_hermes_input,
    load_complete_region,
)
from signal_fusion.io import load_prepared_dataset
from signal_fusion.io.writers import json_safe
from signal_fusion.model_inference import (
    ModelInferenceService,
    ONNXRuntimeBackend,
    load_label_map,
)


BLIND_OUTPUT_DIR = Path("outputs/hermes_blind_inputs")
AUDIT_OUTPUT_DIR = Path("outputs/hermes_audit")


def _case_name(case_id: int) -> str:
    if isinstance(case_id, bool) or int(case_id) != case_id:
        raise TypeError("case_id must be an integer")
    case_id = int(case_id)
    if case_id <= 0:
        raise ValueError("case_id must be positive")
    return f"case_{case_id:04d}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-build-evidence",
        description=(
            "Read one complete signal region, run the local-window IQ model "
            "and region-level 62-feature classifier, and write separated "
            "blind-input and private-audit JSON files."
        ),
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--label_map_path", required=True)
    parser.add_argument("--feature_classifier_manifest", required=True)
    parser.add_argument("--fusion_manifest", required=True)
    parser.add_argument("--group_id", type=int, required=True)
    parser.add_argument(
        "--case_id",
        type=int,
        required=True,
        help="positive anonymous case number used in blind-facing identifiers",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--snr_db", type=float, default=None)
    parser.add_argument("--noise_seed", type=int, default=44)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_evidence_file(
    *,
    dataset_path: str | Path,
    model_path: str | Path,
    label_map_path: str | Path,
    feature_classifier_manifest: str | Path,
    fusion_manifest: str | Path,
    group_id: int,
    case_id: int,
    batch_size: int = 64,
    snr_db: float | None = None,
    noise_seed: int = 44,
    overwrite: bool = False,
) -> dict:
    case_name = _case_name(case_id)
    output = BLIND_OUTPUT_DIR / f"{case_name}.json"
    audit_output = AUDIT_OUTPUT_DIR / f"{case_name}.json"
    existing = [path for path in (output, audit_output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output already exists; pass --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
    )
    dataset = load_prepared_dataset(dataset_path)
    region = load_complete_region(dataset_path, dataset, group_id=group_id)
    labels = load_label_map(label_map_path)
    backend = ONNXRuntimeBackend(model_path)
    manifest = backend.build_manifest(
        model_id=Path(model_path).stem,
        labels=labels,
    )
    bundle = analyze_group(
        dataset,
        region,
        group_id=group_id,
        model_service=ModelInferenceService(manifest, backend=backend),
        feature_classifier_service=FeatureClassifierService(
            feature_classifier_manifest
        ),
        fusion_manifest=FusionManifest.load(fusion_manifest),
        feature_service=FeatureExtractionService(),
        batch_size=batch_size,
        snr_db=snr_db,
        noise_seed=noise_seed,
    )
    hermes_input = build_hermes_input(bundle, case_id=case_id)
    bundle["outputs"] = {
        "hermes_input": str(output),
        "audit": str(audit_output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(hermes_input), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_output.write_text(
        json.dumps(json_safe(bundle), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return bundle


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        bundle = build_evidence_file(
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            label_map_path=args.label_map_path,
            feature_classifier_manifest=args.feature_classifier_manifest,
            fusion_manifest=args.fusion_manifest,
            group_id=args.group_id,
            case_id=args.case_id,
            batch_size=args.batch_size,
            snr_db=args.snr_db,
            noise_seed=args.noise_seed,
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

    iq_prediction = bundle["iq_model_evidence"]["region_top3"][0]
    feature_prediction = bundle["feature_model_evidence"]["region_top3"][0]
    fusion_prediction = bundle["fusion_result"]["region_top3"][0]
    agreement = bundle["iq_model_evidence"]["window_agreement"]
    print("Evidence bundle complete")
    print(f"  case_id:    {_case_name(args.case_id)}")
    print(f"  group_id:   {bundle['region']['group_id']}")
    print(
        "  IQ model:   "
        f"{iq_prediction['label']} ({iq_prediction['probability']:.6f})"
    )
    print(
        "  feature:    "
        f"{feature_prediction['label']} "
        f"({feature_prediction['probability']:.6f})"
    )
    print(
        "  fused:      "
        f"{fusion_prediction['label']} "
        f"({fusion_prediction['probability']:.6f})"
    )
    print(
        "  agreement:  "
        f"{agreement['agreeing_windows']}/{agreement['window_count']}"
    )
    print(f"  Hermes:     {bundle['outputs']['hermes_input']}")
    print(f"  audit:      {bundle['outputs']['audit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "build_evidence_file", "main"]
