"""Build paired blind inputs for the P6-B LLM review pilot."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from signal_fusion.io.writers import json_safe


FEATURE_REFERENCE_DOCUMENTS = (
    "time_domain_iq_features.md",
    "frequency_domain_iq_features.md",
    "time_frequency_iq_features.md",
)
TECHNOLOGY_REFERENCE_DOCUMENT = "technology_reference_lte_wifi_dvbt.md"
PERIODICITY_REFERENCE_DOCUMENT = "ofdm_periodicity_gate_reference.md"
LABELS = ("LTE", "WiFi", "DVB-T")


def _validate_source_case(source: Mapping[str, Any]) -> None:
    if source.get("schema_version") != 3:
        raise ValueError("source evidence must use schema_version=3")
    if source.get("bundle_type") != "hermes_signal_fusion_input":
        raise ValueError("source evidence has an unexpected bundle_type")
    result = source.get("deterministic_fusion_result", {})
    if result.get("decision_status") != "review_required":
        raise ValueError("source evidence is not review_required")
    if result.get("resolution") != "unresolved_review":
        raise ValueError("source evidence is not an unresolved review")
    if result.get("final_label") is not None:
        raise ValueError("unresolved source evidence must not contain a final label")
    if result.get("provisional_label") not in LABELS:
        raise ValueError("source evidence has an invalid provisional label")
    gate = source.get("periodicity_evidence", {}).get("frozen_gate", {})
    if gate.get("resolved") is not False or gate.get("changed") is not False:
        raise ValueError("source periodicity gate must be unresolved and unchanged")
    risk_gate = source.get("iq_ensemble_evidence", {}).get("risk_gate", {})
    if risk_gate.get("unanimous") is not False:
        raise ValueError("source review must contain member disagreement")
    features = source.get("global_feature_evidence", {})
    if features.get("feature_count") != 64:
        raise ValueError("source evidence must contain 64 global features")
    values = features.get("values", {})
    if not isinstance(values, dict) or len(values) != 64:
        raise ValueError("source evidence must contain 64 named feature values")


def _reference_paths(
    source: Mapping[str, Any],
    *,
    include_global_features: bool,
) -> dict[str, str]:
    global_features = source["global_feature_evidence"]
    feature_paths = global_features["reference_document_paths"]
    paths = {
        TECHNOLOGY_REFERENCE_DOCUMENT: str(
            feature_paths[TECHNOLOGY_REFERENCE_DOCUMENT]
        ),
        PERIODICITY_REFERENCE_DOCUMENT: str(
            source["periodicity_evidence"]["reference_document_path"]
        ),
    }
    if include_global_features:
        paths.update(
            {name: str(feature_paths[name]) for name in FEATURE_REFERENCE_DOCUMENTS}
        )
    for name, raw_path in paths.items():
        if not Path(raw_path).is_file():
            raise FileNotFoundError(f"reference document does not exist: {name}")
    return paths


def build_review_experiment_case(
    source: Mapping[str, Any],
    analysis_id: str,
    *,
    include_global_features: bool,
) -> dict[str, Any]:
    """Convert one unresolved production case into a blind review recommendation."""

    _validate_source_case(source)
    if not analysis_id.startswith("review_") or not analysis_id[7:].isdigit():
        raise ValueError("analysis_id must match review_<digits>")
    result = source["deterministic_fusion_result"]
    periodicity = copy.deepcopy(source["periodicity_evidence"])
    periodicity.pop("reference_document", None)
    periodicity.pop("reference_document_path", None)
    case: dict[str, Any] = {
        "schema_version": 1,
        "bundle_type": "signal_fusion_review_experiment_input",
        "evidence_type": "blind_unresolved_review_recommendation",
        "analysis_id": analysis_id,
        "signal_context": copy.deepcopy(source["signal_context"]),
        "review_context": {
            "decision_status": "review_required",
            "provisional_label": result["provisional_label"],
            "review_reason": result["review_reason"],
            "automatic_resolution": "unresolved",
        },
        "iq_ensemble_evidence": copy.deepcopy(source["iq_ensemble_evidence"]),
        "periodicity_evidence": periodicity,
        "reference_document_paths": _reference_paths(
            source,
            include_global_features=include_global_features,
        ),
        "objective": (
            "Provide one blind, non-operational class recommendation or abstain "
            "using only the evidence and references in this file"
        ),
        "decision_contract": {
            "ground_truth_available": False,
            "recommendation_is_operational_final": False,
            "allowed_labels": list(LABELS),
            "abstention_allowed": True,
            "requirements": [
                "do not read audit files, sibling files, other cases, or provenance",
                "do not assume evidence that is absent from this case",
                "do not invent feature thresholds or fusion weights",
                "return recommend only for coherent evidence; otherwise abstain",
            ],
        },
        "required_output": {
            "analysis_id": "exactly this input analysis_id",
            "recommendation_status": "recommend or abstain",
            "recommended_label": "LTE, WiFi, DVB-T, or null",
            "confidence_level": "medium or low",
            "summary": "concise blind recommendation or abstention reason",
            "model_evidence": "list of model observations",
            "periodicity_evidence": "list of periodicity observations",
            "feature_evidence": "list; empty when no global features are supplied",
            "conflicting_evidence": "list of material conflicts",
            "limitations": "list of limitations",
        },
    }
    if include_global_features:
        global_features = copy.deepcopy(source["global_feature_evidence"])
        global_features.pop("reference_documents", None)
        global_features.pop("reference_document_paths", None)
        case["global_feature_evidence"] = global_features
    return case


def _load_source_records(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_path = source_root / "audit.json"
    blind_dir = source_root / "blind"
    if not audit_path.is_file() or not blind_dir.is_dir():
        raise FileNotFoundError("source evidence requires blind/ and audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("audit_type") != "private_region_periodicity_fusion_evidence_audit":
        raise ValueError("source audit has an unexpected audit_type")
    selected = [
        item for item in audit.get("cases", ())
        if item.get("decision_status") == "review_required"
    ]
    if not selected:
        raise ValueError("source audit contains no review_required cases")
    records = []
    for item in selected:
        source_id = str(item["analysis_id"])
        path = blind_dir / f"{source_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"blind source case not found: {path}")
        source = json.loads(path.read_text(encoding="utf-8"))
        _validate_source_case(source)
        records.append({"audit": item, "source": source})
    return records, audit


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_review_experiment_files(
    source_evidence_dir: str | Path,
    public_output_dir: str | Path,
    private_audit_path: str | Path,
    *,
    random_seed: int = 44,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write paired feature-absent/present cases and a separate private mapping."""

    source_root = Path(source_evidence_dir)
    public_root = Path(public_output_dir)
    private_path = Path(private_audit_path)
    records, source_audit = _load_source_records(source_root)
    count = len(records)
    rng = np.random.default_rng(random_seed)
    identifiers = rng.permutation(np.arange(1, 2 * count + 1))
    set_1_ids = identifiers[:count]
    set_2_ids = identifiers[count:]
    public_paths: list[Path] = []
    pairs: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        set_1_analysis_id = f"review_{int(set_1_ids[index]):04d}"
        set_2_analysis_id = f"review_{int(set_2_ids[index]):04d}"
        set_1_path = public_root / "set_1" / f"{set_1_analysis_id}.json"
        set_2_path = public_root / "set_2" / f"{set_2_analysis_id}.json"
        public_paths.extend((set_1_path, set_2_path))
        pairs.append(
            {
                "pair_index": index + 1,
                "source_analysis_id": record["audit"]["analysis_id"],
                "set_1_analysis_id": set_1_analysis_id,
                "set_2_analysis_id": set_2_analysis_id,
                "set_1_evidence": "iq_and_periodicity",
                "set_2_evidence": "iq_periodicity_and_global_features",
                "private_source_audit": record["audit"],
            }
        )

    submission_path = public_root / "submission_order.json"
    existing = [
        path for path in (*public_paths, submission_path, private_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "review experiment output exists; use --overwrite: "
            + ", ".join(str(path) for path in existing[:5])
        )

    for index, record in enumerate(records):
        set_1_analysis_id = pairs[index]["set_1_analysis_id"]
        set_2_analysis_id = pairs[index]["set_2_analysis_id"]
        _write_json(
            public_root / "set_1" / f"{set_1_analysis_id}.json",
            build_review_experiment_case(
                record["source"],
                set_1_analysis_id,
                include_global_features=False,
            ),
        )
        _write_json(
            public_root / "set_2" / f"{set_2_analysis_id}.json",
            build_review_experiment_case(
                record["source"],
                set_2_analysis_id,
                include_global_features=True,
            ),
        )

    order = rng.permutation(np.arange(2 * count))
    relative_paths = [
        str(path.relative_to(public_root)) for path in public_paths
    ]
    submission = {
        "schema_version": 1,
        "manifest_type": "blind_review_experiment_submission_order",
        "case_count": 2 * count,
        "fresh_session_required_per_case": True,
        "instruction": (
            "Use signal-fusion-review-experiment, read only the listed case and "
            "its declared references, and return one JSON object."
        ),
        "cases": [relative_paths[int(position)] for position in order],
    }
    _write_json(submission_path, submission)
    private_audit = {
        "schema_version": 1,
        "audit_type": "private_p6b_clean_review_pairing_audit",
        "selection": {
            "criterion": "all clean cases with decision_status=review_required",
            "selected_count": count,
            "selection_uses_ground_truth": False,
            "source_audit_schema_version": source_audit.get("schema_version"),
        },
        "random_seed": random_seed,
        "pairs": pairs,
    }
    _write_json(private_path, private_audit)
    return {
        "public_output_dir": str(public_root),
        "set_1_dir": str(public_root / "set_1"),
        "set_2_dir": str(public_root / "set_2"),
        "submission_order_path": str(submission_path),
        "private_audit_path": str(private_path),
        "pair_count": count,
        "public_case_count": 2 * count,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paired blind inputs for the P6-B LLM review pilot."
    )
    parser.add_argument("--source-evidence-dir", required=True)
    parser.add_argument("--public-output-dir", required=True)
    parser.add_argument("--private-audit-path", required=True)
    parser.add_argument("--random-seed", type=int, default=44)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_review_experiment_files(
        args.source_evidence_dir,
        args.public_output_dir,
        args.private_audit_path,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_arg_parser",
    "build_review_experiment_case",
    "build_review_experiment_files",
    "main",
]
