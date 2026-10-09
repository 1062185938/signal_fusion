"""Build paired blind IQ/feature-probe adjudication inputs.

Set A contains only branch outputs. Set B adds the complete-region 64-feature
vector. Ground truth, provenance, selection role, and the equal-weight result
are written exclusively to the private audit file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from signal_fusion.benchmarks.technology_recognition_cross_location_fusion import (
    LABEL_NAMES,
    load_full_feature_bank,
)
from signal_fusion.feature_extraction import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_ID,
    asset_path,
)
from signal_fusion.fusion.references import technology_reference_path
from signal_fusion.io.writers import json_safe


BUNDLE_TYPE = "signal_fusion_cross_location_review_input"
EVIDENCE_TYPE = "blind_iq_feature_disagreement_adjudication"
MANIFEST_TYPE = "blind_fusion_adjudication_submission"
AUDIT_TYPE = "private_cross_location_llm_adjudication_audit"
FEATURE_REFERENCE_DOCUMENTS = (
    "time_domain_iq_features.md",
    "frequency_domain_iq_features.md",
    "time_frequency_iq_features.md",
)
TECHNOLOGY_REFERENCE_DOCUMENT = "technology_reference_lte_wifi_dvbt.md"


def _validate_probabilities(
    probabilities: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABEL_NAMES):
        raise ValueError(
            f"{name} must have shape [N, {len(LABEL_NAMES)}], got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError(f"{name} rows must sum to one")
    return values


def _normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    entropy = -np.sum(values * np.log(np.maximum(values, 1e-12)))
    return float(entropy / math.log(values.size))


def _branch_evidence(name: str, probabilities: np.ndarray) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(LABEL_NAMES),):
        raise ValueError("one branch probability vector has an invalid shape")
    order = np.argsort(-values, kind="stable")
    top1 = int(order[0])
    return {
        "branch_name": name,
        "probabilities": {
            label: float(values[index])
            for index, label in enumerate(LABEL_NAMES)
        },
        "top3": [
            {"label": LABEL_NAMES[int(index)], "probability": float(values[index])}
            for index in order
        ],
        "top1": {
            "label": LABEL_NAMES[top1],
            "probability": float(values[top1]),
        },
        "top1_top2_margin": float(values[order[0]] - values[order[1]]),
        "normalized_entropy": _normalized_entropy(values),
    }


def _match_rows(
    proposal_indices: np.ndarray,
    shadow_indices: np.ndarray,
    signatures: np.ndarray,
    folds: np.ndarray,
    *,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match controls by six-probability signature, preferring the same fold."""

    rng = np.random.default_rng(random_seed)
    unmatched_proposals = set(int(value) for value in proposal_indices)
    unused_shadows = set(int(value) for value in shadow_indices)
    pairs: list[tuple[int, int]] = []

    for fold in sorted(set(str(value) for value in folds[proposal_indices])):
        left = np.asarray(
            [index for index in unmatched_proposals if str(folds[index]) == fold],
            dtype=np.int64,
        )
        right = np.asarray(
            [index for index in unused_shadows if str(folds[index]) == fold],
            dtype=np.int64,
        )
        if left.size == 0 or right.size == 0:
            continue
        left = left[rng.permutation(left.size)]
        right = right[rng.permutation(right.size)]
        distance = np.sum(
            (signatures[left, None, :] - signatures[right][None, :, :]) ** 2,
            axis=2,
        )
        row_indices, column_indices = linear_sum_assignment(distance)
        for row_index, column_index in zip(
            row_indices, column_indices, strict=True
        ):
            proposal = int(left[row_index])
            shadow = int(right[column_index])
            pairs.append((proposal, shadow))
            unmatched_proposals.remove(proposal)
            unused_shadows.remove(shadow)

    if unmatched_proposals:
        left = np.asarray(sorted(unmatched_proposals), dtype=np.int64)
        right = np.asarray(sorted(unused_shadows), dtype=np.int64)
        if right.size < left.size:
            raise ValueError("not enough shadow rows to match all proposals")
        left = left[rng.permutation(left.size)]
        right = right[rng.permutation(right.size)]
        distance = np.sum(
            (signatures[left, None, :] - signatures[right][None, :, :]) ** 2,
            axis=2,
        )
        row_indices, column_indices = linear_sum_assignment(distance)
        for row_index, column_index in zip(
            row_indices, column_indices, strict=True
        ):
            pairs.append((int(left[row_index]), int(right[column_index])))

    pairs.sort(key=lambda pair: pair[0])
    matched_proposals = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    matched_shadows = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    distances = np.linalg.norm(
        signatures[matched_proposals] - signatures[matched_shadows], axis=1
    )
    return matched_proposals, matched_shadows, distances


def select_adjudication_rows(
    iq_probabilities: np.ndarray,
    feature_probabilities: np.ndarray,
    folds: np.ndarray,
    *,
    pair_count: int | None = None,
    random_seed: int = 44,
) -> dict[str, np.ndarray]:
    """Select proposal/shadow rows without accepting or consulting labels.

    A proposal is an IQ/feature disagreement for which an equal 0.5/0.5 mean
    changes the IQ top-1 label.  Shadows come from the remaining disagreements
    and are nearest-neighbour matched on the two three-class probability vectors.
    """

    iq = _validate_probabilities(iq_probabilities, name="iq_probabilities")
    feature = _validate_probabilities(
        feature_probabilities, name="feature_probabilities"
    )
    folds = np.asarray(folds).astype(str)
    if iq.shape != feature.shape or folds.shape != (iq.shape[0],):
        raise ValueError("probabilities and folds must describe the same rows")

    iq_labels = iq.argmax(axis=1)
    feature_labels = feature.argmax(axis=1)
    equal_labels = ((iq + feature) * 0.5).argmax(axis=1)
    disagreement = iq_labels != feature_labels
    proposal_pool = np.flatnonzero(disagreement & (equal_labels != iq_labels))
    shadow_pool = np.flatnonzero(disagreement & (equal_labels == iq_labels))
    if proposal_pool.size == 0:
        raise ValueError("no equal-weight label-changing disagreements were found")

    selected_count = proposal_pool.size if pair_count is None else int(pair_count)
    if selected_count <= 0 or selected_count > proposal_pool.size:
        raise ValueError("pair_count must be positive and no larger than proposal count")
    if shadow_pool.size < selected_count:
        raise ValueError("not enough non-changing disagreements for shadow matching")

    rng = np.random.default_rng(random_seed)
    if selected_count == proposal_pool.size:
        selected_proposals = proposal_pool
    else:
        selected_proposals = np.sort(
            rng.choice(proposal_pool, size=selected_count, replace=False)
        )
    signatures = np.concatenate((iq, feature), axis=1)
    proposals, shadows, distances = _match_rows(
        selected_proposals,
        shadow_pool,
        signatures,
        folds,
        random_seed=random_seed,
    )
    return {
        "proposal_indices": proposals,
        "shadow_indices": shadows,
        "matching_distances": distances,
        "same_fold": folds[proposals] == folds[shadows],
        "proposal_pool_indices": proposal_pool,
        "shadow_pool_indices": shadow_pool,
        "disagreement_indices": np.flatnonzero(disagreement),
    }


def _reference_paths(*, include_physical_features: bool) -> dict[str, str]:
    paths = {
        TECHNOLOGY_REFERENCE_DOCUMENT: str(technology_reference_path().resolve())
    }
    if include_physical_features:
        paths.update(
            {
                name: str(asset_path(name).resolve())
                for name in FEATURE_REFERENCE_DOCUMENTS
            }
        )
    for name, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"reference document does not exist: {name}")
    return paths


def build_adjudication_case(
    analysis_id: str,
    iq_probabilities: np.ndarray,
    feature_probabilities: np.ndarray,
    *,
    include_physical_features: bool,
    physical_features: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    sample_rate_hz: float = 1_000_000.0,
    sample_count: int = 4096,
) -> dict[str, Any]:
    """Build one public blind case; no private metadata is accepted."""

    if not analysis_id.startswith("review_") or not analysis_id[7:].isdigit():
        raise ValueError("analysis_id must match review_<digits>")
    iq = _validate_probabilities(
        np.asarray(iq_probabilities)[None, :], name="iq_probabilities"
    )[0]
    feature = _validate_probabilities(
        np.asarray(feature_probabilities)[None, :], name="feature_probabilities"
    )[0]
    if int(iq.argmax()) == int(feature.argmax()):
        raise ValueError("public review cases require branch top-1 disagreement")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    case: dict[str, Any] = {
        "schema_version": 2,
        "bundle_type": BUNDLE_TYPE,
        "evidence_type": EVIDENCE_TYPE,
        "analysis_id": analysis_id,
        "output_language": "zh-CN",
        "signal_context": {
            "effective_sample_rate_hz": float(sample_rate_hz),
            "sample_count": int(sample_count),
            "duration_ms": float(sample_count / sample_rate_hz * 1000.0),
        },
        "iq_branch": _branch_evidence("iq_model", iq),
        "feature_probe_branch": _branch_evidence(
            "global_64_feature_linear_probe", feature
        ),
        "reference_document_paths": _reference_paths(
            include_physical_features=include_physical_features
        ),
        "required_output": {
            "analysis_id": "exactly this input analysis_id",
            "decision": "keep, change, or abstain",
            "recommended_label": "LTE, WiFi, DVB-T, or null",
            "confidence_level": "medium or low",
            "summary": "concise Chinese blind adjudication summary",
            "iq_evidence": "list of IQ-branch observations",
            "feature_probe_evidence": "list of feature-probe observations",
            "physical_feature_evidence": (
                "list of raw-feature observations; empty when raw features are absent"
            ),
            "conflicting_evidence": "list of material conflicts",
            "limitations": "list of limitations",
        },
    }
    if include_physical_features:
        if physical_features is None or feature_names is None:
            raise ValueError("physical features and names are required for set B")
        values = np.asarray(physical_features, dtype=np.float64)
        names = tuple(str(name) for name in feature_names)
        if values.shape != (FEATURE_COUNT,) or len(names) != FEATURE_COUNT:
            raise ValueError(f"physical feature evidence must contain {FEATURE_COUNT} values")
        if len(set(names)) != FEATURE_COUNT or not np.all(np.isfinite(values)):
            raise ValueError("physical feature names must be unique and values finite")
        case["physical_feature_evidence"] = {
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "feature_count": FEATURE_COUNT,
            "scope": "complete_4096_sample_region",
            "values": {
                name: float(value)
                for name, value in zip(names, values, strict=True)
            },
        }
    elif physical_features is not None or feature_names is not None:
        raise ValueError("set A must not receive physical feature values or names")
    return case


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    required = (
        "fold",
        "location",
        "sample_source_id",
        "source_region_id",
        "global_group_id",
        "y",
        "iq_model_probabilities",
        "feature_probe_probabilities",
        "equal_weight_fusion_probabilities",
    )
    with np.load(path, allow_pickle=False) as raw:
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"prediction artifact is missing fields: {missing}")
        predictions = {name: np.asarray(raw[name]).copy() for name in required}
    count = predictions["y"].size
    for name in required[:5]:
        if predictions[name].shape != (count,):
            raise ValueError(f"prediction field {name!r} has an invalid shape")
    if predictions["y"].shape != (count,):
        raise ValueError("prediction labels have an invalid shape")
    iq = _validate_probabilities(
        predictions["iq_model_probabilities"], name="iq_model_probabilities"
    )
    feature = _validate_probabilities(
        predictions["feature_probe_probabilities"],
        name="feature_probe_probabilities",
    )
    equal = _validate_probabilities(
        predictions["equal_weight_fusion_probabilities"],
        name="equal_weight_fusion_probabilities",
    )
    if iq.shape != feature.shape or iq.shape != equal.shape or iq.shape[0] != count:
        raise ValueError("prediction probability arrays do not align")
    if not np.allclose(equal, (iq + feature) * 0.5, rtol=1e-7, atol=1e-9):
        raise ValueError("stored equal-weight probabilities do not match 0.5/0.5 mean")
    labels = predictions["y"].astype(np.int64)
    if np.any((labels < 0) | (labels >= len(LABEL_NAMES))):
        raise ValueError("prediction labels are outside the declared label space")
    predictions["y"] = labels
    return predictions


def _aligned_feature_rows(
    predictions: Mapping[str, np.ndarray],
    feature_bank: Mapping[str, Any],
) -> np.ndarray:
    lookup = feature_bank["row_lookup"]
    keys = zip(
        np.asarray(predictions["sample_source_id"]).astype(str),
        np.asarray(predictions["source_region_id"], dtype=np.int64),
        strict=True,
    )
    try:
        rows = np.asarray(
            [lookup[(str(source_id), int(region_id))] for source_id, region_id in keys],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"prediction row is absent from feature bank: {exc}") from exc
    if np.unique(rows).size != rows.size:
        raise ValueError("prediction rows do not map one-to-one to feature rows")
    return rows


def build_review_experiment_files(
    prediction_path: str | Path,
    dataset_root: str | Path,
    feature_root: str | Path,
    public_output_dir: str | Path,
    private_audit_path: str | Path,
    *,
    random_seed: int = 44,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write 226 paired A/B signals (452 public files) and one private audit."""

    predictions = _load_predictions(Path(prediction_path))
    iq = predictions["iq_model_probabilities"]
    feature = predictions["feature_probe_probabilities"]
    folds = np.asarray(predictions["fold"]).astype(str)
    selection = select_adjudication_rows(
        iq, feature, folds, random_seed=random_seed
    )
    proposals = selection["proposal_indices"]
    shadows = selection["shadow_indices"]
    if proposals.size != shadows.size:
        raise RuntimeError("proposal/shadow selection is not balanced")

    feature_bank = load_full_feature_bank(dataset_root, feature_root)
    feature_rows = _aligned_feature_rows(predictions, feature_bank)
    feature_values = np.asarray(feature_bank["features"])[feature_rows]
    sample_rates = np.asarray(feature_bank["sample_rate"], dtype=np.float64)[
        feature_rows
    ]
    feature_names = tuple(str(name) for name in feature_bank["feature_names"])
    if feature_values.shape != (iq.shape[0], FEATURE_COUNT):
        raise ValueError("aligned physical feature matrix has an invalid shape")

    rng = np.random.default_rng(random_seed)
    records: list[dict[str, Any]] = []
    for matching_pair_index, (proposal, shadow, distance) in enumerate(
        zip(
            proposals,
            shadows,
            selection["matching_distances"],
            strict=True,
        ),
        start=1,
    ):
        for selection_role, row in (("candidate", proposal), ("shadow", shadow)):
            records.append(
                {
                    "row": int(row),
                    "selection_role": selection_role,
                    "matching_pair_index": matching_pair_index,
                    "matching_distance": float(distance),
                }
            )
    records = [records[int(index)] for index in rng.permutation(len(records))]
    public_case_count = 2 * len(records)
    identifiers = rng.permutation(np.arange(1, public_case_count + 1))
    public_root = Path(public_output_dir)
    private_path = Path(private_audit_path)

    public_paths: list[Path] = []
    audit_cases: list[dict[str, Any]] = []
    equal = predictions["equal_weight_fusion_probabilities"]
    labels = predictions["y"]
    for pair_index, record in enumerate(records, start=1):
        row = record["row"]
        set_a_id = f"review_{int(identifiers[pair_index - 1]):04d}"
        set_b_id = f"review_{int(identifiers[len(records) + pair_index - 1]):04d}"
        set_a_path = public_root / "set_a" / f"{set_a_id}.json"
        set_b_path = public_root / "set_b" / f"{set_b_id}.json"
        public_paths.extend((set_a_path, set_b_path))
        audit_cases.append(
            {
                "signal_case_id": f"signal_{pair_index:04d}",
                "pair_index": pair_index,
                "matching_pair_index": record["matching_pair_index"],
                "matching_distance": record["matching_distance"],
                "selection_role": record["selection_role"],
                "set_a_analysis_id": set_a_id,
                "set_b_analysis_id": set_b_id,
                "true_label": LABEL_NAMES[int(labels[row])],
                "iq_label": LABEL_NAMES[int(iq[row].argmax())],
                "feature_label": LABEL_NAMES[int(feature[row].argmax())],
                "equal_fusion_label": LABEL_NAMES[int(equal[row].argmax())],
                "fold": str(predictions["fold"][row]),
                "location": str(predictions["location"][row]),
                "sample_source_id": str(predictions["sample_source_id"][row]),
                "source_region_id": int(predictions["source_region_id"][row]),
                "global_group_id": int(predictions["global_group_id"][row]),
            }
        )

    manifest_path = public_root / "submission_order.json"
    targets = [*public_paths, manifest_path, private_path]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "adjudication outputs exist; use --overwrite: "
            + ", ".join(str(path) for path in existing[:5])
        )

    for pair_index, record in enumerate(records, start=1):
        row = record["row"]
        set_a_id = audit_cases[pair_index - 1]["set_a_analysis_id"]
        set_b_id = audit_cases[pair_index - 1]["set_b_analysis_id"]
        _write_json(
            public_root / "set_a" / f"{set_a_id}.json",
            build_adjudication_case(
                set_a_id,
                iq[row],
                feature[row],
                include_physical_features=False,
                sample_rate_hz=float(sample_rates[row]),
            ),
        )
        _write_json(
            public_root / "set_b" / f"{set_b_id}.json",
            build_adjudication_case(
                set_b_id,
                iq[row],
                feature[row],
                include_physical_features=True,
                physical_features=feature_values[row],
                feature_names=feature_names,
                sample_rate_hz=float(sample_rates[row]),
            ),
        )

    relative_paths = [str(path.relative_to(public_root)) for path in public_paths]
    submission_order = rng.permutation(len(relative_paths))
    manifest = {
        "schema_version": 2,
        "manifest_type": MANIFEST_TYPE,
        "case_count": public_case_count,
        "signal_case_count": len(records),
        "session_mode": "shared_resumable",
        "instruction": (
            "Use signal-fusion-review-experiment, read only each listed case and "
            "its declared references, and return one JSON object per case."
        ),
        "cases": [relative_paths[int(index)] for index in submission_order],
    }
    _write_json(manifest_path, manifest)

    iq_labels = iq.argmax(axis=1)
    population_summary = {
        "region_count": int(labels.size),
        "iq_correct_count": int(np.count_nonzero(iq_labels == labels)),
        "iq_error_count": int(np.count_nonzero(iq_labels != labels)),
        "candidate_count": int(proposals.size),
        "shadow_count": int(shadows.size),
    }
    audit = {
        "schema_version": 2,
        "audit_type": AUDIT_TYPE,
        "population_summary": population_summary,
        "selection": {
            "selection_uses_ground_truth": False,
            "proposal_rule": (
                "iq_top1 != feature_top1 and equal_weight_top1 != iq_top1"
            ),
            "shadow_rule": (
                "remaining branch disagreements matched on concatenated "
                "six-probability signature, preferring the same fold"
            ),
            "disagreement_count": int(selection["disagreement_indices"].size),
            "proposal_pool_count": int(selection["proposal_pool_indices"].size),
            "shadow_pool_count": int(selection["shadow_pool_indices"].size),
            "same_fold_match_count": int(np.count_nonzero(selection["same_fold"])),
            "random_seed": int(random_seed),
        },
        "cases": audit_cases,
    }
    _write_json(private_path, audit)
    return {
        "public_output_dir": str(public_root),
        "set_a_dir": str(public_root / "set_a"),
        "set_b_dir": str(public_root / "set_b"),
        "submission_order_path": str(manifest_path),
        "private_audit_path": str(private_path),
        "signal_case_count": len(records),
        "public_case_count": public_case_count,
        "proposal_count": int(proposals.size),
        "shadow_count": int(shadows.size),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paired blind cross-location IQ/feature adjudication cases."
    )
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--public-output-dir", required=True)
    parser.add_argument("--private-audit-path", required=True)
    parser.add_argument("--random-seed", type=int, default=44)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_review_experiment_files(
        args.prediction_path,
        args.dataset_root,
        args.feature_root,
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
    "AUDIT_TYPE",
    "BUNDLE_TYPE",
    "EVIDENCE_TYPE",
    "FEATURE_REFERENCE_DOCUMENTS",
    "MANIFEST_TYPE",
    "TECHNOLOGY_REFERENCE_DOCUMENT",
    "build_adjudication_case",
    "build_arg_parser",
    "build_review_experiment_files",
    "main",
    "select_adjudication_rows",
]
