"""Blind evidence generation for an IQ ensemble plus whole-region features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.feature_extraction import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_ID,
    asset_path,
    feature_code_names,
    load_feature_map,
)
from signal_fusion.fusion.references import (
    PERIODICITY_REFERENCE_FILENAME,
    TECHNOLOGY_REFERENCE_FILENAME,
    periodicity_reference_path,
    technology_reference_path,
)
from signal_fusion.fusion.periodicity_gate import (
    apply_periodicity_gate,
    load_periodicity_gate_manifest,
    score_technology_periodicity,
)
from signal_fusion.io import load_prepared_dataset
from signal_fusion.io.writers import json_safe
from signal_fusion.model_inference import (
    ModelInferenceService,
    RegionEnsembleInferenceService,
    RegionEnsembleResult,
)


FEATURE_REFERENCE_DOCUMENTS = (
    "time_domain_iq_features.md",
    "frequency_domain_iq_features.md",
    "time_frequency_iq_features.md",
)
REFERENCE_DOCUMENTS = (
    *FEATURE_REFERENCE_DOCUMENTS,
    TECHNOLOGY_REFERENCE_FILENAME,
)


def _reference_document_paths() -> dict[str, str]:
    paths = {
        name: str(asset_path(name).resolve())
        for name in FEATURE_REFERENCE_DOCUMENTS
    }
    paths[TECHNOLOGY_REFERENCE_FILENAME] = str(
        technology_reference_path().resolve()
    )
    return paths


def load_feature_artifact(
    feature_path: str | Path,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    float,
    int,
    dict[str, np.ndarray | str],
]:
    """Load and validate the stable output of signal-extract-features."""

    with np.load(feature_path, allow_pickle=False) as artifact:
        required = {
            "features",
            "sample_rate",
            "seq_len",
            "feature_count",
            "feature_schema_id",
            "feature_names",
            "source_dataset_id",
            "group_id",
            "sample_source_id",
            "source_region_id",
        }
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(
                "feature artifact is missing fields: " + ", ".join(missing)
            )
        features = np.asarray(artifact["features"], dtype=np.float32)
        sample_rate = float(np.asarray(artifact["sample_rate"]).item())
        seq_len = int(np.asarray(artifact["seq_len"]).item())
        feature_count = int(np.asarray(artifact["feature_count"]).item())
        feature_schema_id = str(
            np.asarray(artifact["feature_schema_id"]).reshape(()).item()
        )
        feature_names = tuple(np.asarray(artifact["feature_names"]).astype(str))
        row_identity: dict[str, np.ndarray | str] = {
            "source_dataset_id": str(
                np.asarray(artifact["source_dataset_id"]).reshape(()).item()
            ),
            "group_id": np.asarray(artifact["group_id"]).copy(),
            "sample_source_id": np.asarray(artifact["sample_source_id"])
            .astype(str)
            .copy(),
            "source_region_id": np.asarray(artifact["source_region_id"]).copy(),
        }
    if feature_schema_id != FEATURE_SCHEMA_ID:
        raise ValueError(
            "feature artifact schema does not match the runtime: "
            f"{feature_schema_id!r} != {FEATURE_SCHEMA_ID!r}"
        )
    if feature_count != FEATURE_COUNT or features.ndim != 2:
        raise ValueError(
            f"feature artifact must contain an [N, {FEATURE_COUNT}] matrix"
        )
    if features.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"feature artifact must contain {FEATURE_COUNT} columns, "
            f"got {features.shape[1]}"
        )
    canonical_names = feature_code_names(load_feature_map())
    if feature_names != canonical_names:
        raise ValueError(
            "feature_names do not match the canonical feature_map.json order"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("feature artifact contains non-finite values")
    if not np.isfinite(sample_rate) or sample_rate <= 0 or seq_len <= 0:
        raise ValueError("feature artifact sample_rate and seq_len must be positive")
    for field_name in ("group_id", "sample_source_id", "source_region_id"):
        if np.asarray(row_identity[field_name]).shape != (features.shape[0],):
            raise ValueError(
                f"feature artifact {field_name} must contain one value per row"
            )
    return features, feature_names, sample_rate, seq_len, row_identity


def _region_vector(dataset: PreparedDataset, name: str) -> np.ndarray:
    if name not in dataset.meta:
        raise ValueError(f"continuous-region dataset is missing {name} metadata")
    values = np.asarray(dataset.meta[name])
    if values.shape != (dataset.num_samples,):
        raise ValueError(
            f"continuous-region {name} must contain one value per region"
        )
    return values


def _region_sample_rate(dataset: PreparedDataset) -> float:
    if "sample_rate" not in dataset.meta:
        raise ValueError("continuous-region dataset is missing sample_rate")
    values = np.asarray(dataset.meta["sample_rate"], dtype=np.float64)
    if values.shape not in {(), (dataset.num_samples,)}:
        raise ValueError(
            "continuous-region sample_rate must be scalar or contain one "
            "value per region"
        )
    unique_values = np.unique(values.reshape(-1))
    if unique_values.size != 1:
        raise ValueError(
            "continuous-region sample_rate must be identical for every region"
        )
    value = float(unique_values[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("continuous-region sample_rate must be positive")
    return value


def _validate_region_preprocessing(dataset: PreparedDataset) -> None:
    for field_name in ("remove_dc", "rms_normalize"):
        if field_name not in dataset.meta:
            raise ValueError(
                f"continuous-region dataset is missing {field_name} metadata"
            )
        value = np.asarray(dataset.meta[field_name])
        if value.shape != () or not bool(value.item()):
            raise ValueError(
                f"continuous-region {field_name} must be scalar true"
            )


def _validate_input_alignment(
    window_dataset: PreparedDataset,
    region_dataset: PreparedDataset,
    feature_row_identity: Mapping[str, np.ndarray | str],
) -> None:
    region_group_ids = _region_vector(region_dataset, "group_id").astype(
        np.int64, copy=False
    )
    region_source_ids = _region_vector(region_dataset, "sample_source_id").astype(str)
    region_source_region_ids = _region_vector(
        region_dataset, "source_region_id"
    ).astype(np.int64, copy=False)
    window_group_ids = _region_vector(window_dataset, "group_id").astype(
        np.int64, copy=False
    )
    window_source_ids = _region_vector(window_dataset, "sample_source_id").astype(str)
    window_source_region_ids = _region_vector(
        window_dataset, "source_region_id"
    ).astype(np.int64, copy=False)

    if not np.array_equal(
        np.asarray(feature_row_identity["group_id"], dtype=np.int64),
        region_group_ids,
    ):
        raise ValueError("feature rows do not match continuous-region group_id order")
    if not np.array_equal(
        np.asarray(feature_row_identity["sample_source_id"]).astype(str),
        region_source_ids,
    ):
        raise ValueError("feature rows do not match continuous-region source IDs")
    if not np.array_equal(
        np.asarray(feature_row_identity["source_region_id"], dtype=np.int64),
        region_source_region_ids,
    ):
        raise ValueError(
            "feature rows do not match continuous-region source_region_id order"
        )
    if feature_row_identity["source_dataset_id"] != region_dataset.source_id:
        raise ValueError(
            "feature source_dataset_id does not match the continuous-region dataset"
        )

    region_row_by_group = {
        int(group_id): row for row, group_id in enumerate(region_group_ids)
    }
    for group_id in np.unique(window_group_ids):
        indices = np.flatnonzero(window_group_ids == group_id)
        region_row = region_row_by_group[int(group_id)]
        if not np.all(window_source_ids[indices] == region_source_ids[region_row]):
            raise ValueError(
                f"window and continuous-region source IDs differ for group {group_id}"
            )
        if not np.all(
            window_source_region_ids[indices]
            == region_source_region_ids[region_row]
        ):
            raise ValueError(
                "window and continuous-region source_region_id differ for "
                f"group {group_id}"
            )
        if window_dataset.y is not None and region_dataset.y is not None:
            if not np.all(window_dataset.y[indices] == region_dataset.y[region_row]):
                raise ValueError(
                    f"window and continuous-region labels differ for group {group_id}"
                )


def _case_name(case_id: int) -> str:
    if isinstance(case_id, bool) or int(case_id) != case_id:
        raise TypeError("case_id must be an integer")
    case_id = int(case_id)
    if case_id <= 0:
        raise ValueError("case_id must be positive")
    return f"case_{case_id:04d}"


def _blind_ensemble(result: RegionEnsembleResult) -> dict[str, Any]:
    payload = result.to_dict()
    members: list[dict[str, Any]] = []
    for member_index, member in enumerate(payload["members"], start=1):
        members.append(
            {
                "member_id": f"iq_member_{member_index}",
                "region_top3": member["region_top3"],
                "window_aggregation": member["window_aggregation"],
                "window_agreement": member["window_agreement"],
                "window_predictions": [
                    {
                        "window_index": window_index,
                        "label": prediction["label"],
                        "confidence": prediction["confidence"],
                    }
                    for window_index, prediction in enumerate(
                        member["window_predictions"]
                    )
                ],
            }
        )
    return {
        "decision_status": payload["decision_status"],
        "risk_gate": payload["risk_gate"],
        "ensemble": payload["ensemble"],
        "members": members,
    }


def _private_value(
    dataset: PreparedDataset,
    name: str,
    row_index: int,
) -> Any | None:
    if name not in dataset.meta:
        return None
    values = np.asarray(dataset.meta[name])
    if values.shape != (dataset.num_samples,):
        raise ValueError(f"private metadata {name} must contain one value per region")
    return values[row_index]


def build_evidence_records(
    window_dataset: PreparedDataset,
    region_dataset: PreparedDataset,
    features: np.ndarray,
    feature_names: Sequence[str],
    feature_row_identity: Mapping[str, np.ndarray | str],
    ensemble_service: RegionEnsembleInferenceService,
    periodicity_gate_manifest_path: str | Path,
    *,
    group_id: int | None = None,
    case_id_start: int = 1,
    batch_size: int = 64,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build anonymous evidence and a separately held private audit mapping."""

    feature_values = np.asarray(features, dtype=np.float32)
    names = tuple(str(name) for name in feature_names)
    if feature_values.shape != (region_dataset.num_samples, FEATURE_COUNT):
        raise ValueError(
            f"features must contain one {FEATURE_COUNT}-dimensional row "
            "per continuous region"
        )
    if names != feature_code_names(load_feature_map()):
        raise ValueError(
            "feature_names do not match the canonical feature_map.json order"
        )
    if not np.all(np.isfinite(feature_values)):
        raise ValueError("features contain non-finite values")
    if len(ensemble_service.members) != 3:
        raise ValueError("evidence generation requires exactly three IQ models")
    _validate_region_preprocessing(region_dataset)

    region_group_ids = _region_vector(region_dataset, "group_id").astype(
        np.int64, copy=False
    )
    if np.unique(region_group_ids).size != region_dataset.num_samples:
        raise ValueError("continuous-region dataset must contain one row per group_id")
    window_group_ids = np.asarray(window_dataset.meta.get("group_id"))
    if window_group_ids.shape != (window_dataset.num_samples,):
        raise ValueError("window dataset must contain one group_id per window")
    if not np.array_equal(
        np.unique(window_group_ids.astype(np.int64, copy=False)),
        np.sort(region_group_ids),
    ):
        raise ValueError("window and continuous-region datasets contain different groups")
    _validate_input_alignment(
        window_dataset,
        region_dataset,
        feature_row_identity,
    )

    if group_id is None:
        ensemble_results = ensemble_service.predict_all_groups(
            window_dataset, batch_size=batch_size
        )
    else:
        ensemble_results = (
            ensemble_service.predict_group(
                window_dataset,
                group_id=int(group_id),
                batch_size=batch_size,
            ),
        )
    result_by_group = {result.group_id: result for result in ensemble_results}
    selected_group_ids = sorted(result_by_group)
    row_by_group = {
        int(group): row for row, group in enumerate(region_group_ids.tolist())
    }
    sample_rate = _region_sample_rate(region_dataset)
    gate_manifest, reliability_margin = load_periodicity_gate_manifest(
        periodicity_gate_manifest_path
    )
    if not np.isclose(sample_rate, float(gate_manifest["sample_rate_hz"])):
        raise ValueError(
            "continuous-region sample rate does not match the periodicity gate"
        )
    if region_dataset.seq_len != int(gate_manifest["region_sample_count"]):
        raise ValueError(
            "continuous-region length does not match the periodicity gate"
        )
    periodicity = score_technology_periodicity(
        region_dataset,
        batch_size=batch_size,
    )
    source_ids = _region_vector(region_dataset, "sample_source_id").astype(str)
    source_region_ids = _region_vector(region_dataset, "source_region_id").astype(
        np.int64, copy=False
    )

    blind_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    labels = ensemble_results[0].labels
    if tuple(labels) != tuple(gate_manifest["labels"]):
        raise ValueError("ensemble labels do not match the periodicity gate manifest")

    ordered_results = [result_by_group[group] for group in selected_group_ids]
    selected_rows = np.asarray(
        [row_by_group[group] for group in selected_group_ids], dtype=np.int64
    )
    member_predictions = np.stack(
        [result.member_region_probabilities.argmax(axis=1) for result in ordered_results]
    )
    ensemble_predictions = np.asarray(
        [int(np.argmax(result.ensemble_probabilities)) for result in ordered_results],
        dtype=np.int64,
    )
    periodicity_predictions = np.asarray(periodicity["prediction"])[selected_rows]
    periodicity_margins = np.asarray(periodicity["margin"])[selected_rows]
    gated = apply_periodicity_gate(
        member_predictions,
        ensemble_predictions,
        periodicity_predictions,
        periodicity_margins,
        reliability_margin,
    )

    for offset, selected_group_id in enumerate(selected_group_ids):
        row = row_by_group[selected_group_id]
        result = result_by_group[selected_group_id]
        case_name = _case_name(case_id_start + offset)
        values = {
            name: float(value)
            for name, value in zip(names, feature_values[row], strict=True)
        }
        ensemble_index = int(ensemble_predictions[offset])
        periodicity_index = int(periodicity_predictions[offset])
        prediction_index = int(gated["prediction"][offset])
        eligible = bool(gated["eligible"][offset])
        reliable = bool(gated["reliable"][offset])
        resolved = bool(gated["resolved"][offset])
        changed = bool(gated["changed"][offset])
        member_disagreement = result.decision_status == "review_required"
        if not member_disagreement:
            decision_status = "accept"
            resolution = "ensemble_unanimous"
            review_reason = None
        elif resolved and changed:
            decision_status = "accept"
            resolution = "periodicity_changed_label"
            review_reason = None
        elif resolved:
            decision_status = "accept"
            resolution = "ensemble_confirmed_by_periodicity"
            review_reason = None
        else:
            decision_status = "review_required"
            resolution = "unresolved_review"
            review_reason = (
                "periodicity_below_threshold"
                if eligible
                else "outside_periodicity_gate_scope"
            )
        prediction_label = labels[prediction_index]
        final_label = prediction_label if decision_status == "accept" else None
        provisional_label = (
            prediction_label if decision_status == "review_required" else None
        )
        periodicity_measurement = periodicity["measurement"].evidence_for_sample(row)
        blind_records.append(
            {
                "schema_version": 3,
                "bundle_type": "hermes_signal_fusion_input",
                "evidence_type": (
                    "blind_region_ensemble_with_periodicity_gate_and_global_features"
                ),
                "analysis_id": case_name,
                "signal_context": {
                    "effective_sample_rate_hz": sample_rate,
                    "observation_sample_count": region_dataset.seq_len,
                    "observation_duration_seconds": (
                        region_dataset.seq_len / sample_rate
                    ),
                    "iq_window_sample_count": window_dataset.seq_len,
                    "iq_window_count": len(result.member_results[0].sample_indices),
                },
                "iq_ensemble_evidence": _blind_ensemble(result),
                "periodicity_evidence": {
                    **periodicity_measurement,
                    "scores": {
                        "lte_score": float(periodicity["lte_score"][row]),
                        "dvbt_2k_score": float(
                            periodicity["dvbt_2k_score"][row]
                        ),
                        "dvbt_8k_score": float(
                            periodicity["dvbt_8k_score"][row]
                        ),
                        "dvbt_score": float(periodicity["dvbt_score"][row]),
                        "prediction_label": labels[periodicity_index],
                        "margin": float(periodicity_margins[offset]),
                    },
                    "frozen_gate": {
                        "reliability_margin": reliability_margin,
                        "eligible": eligible,
                        "reliable": reliable,
                        "resolved": resolved,
                        "changed": changed,
                    },
                    "reference_document": PERIODICITY_REFERENCE_FILENAME,
                    "reference_document_path": str(
                        periodicity_reference_path().resolve()
                    ),
                },
                "deterministic_fusion_result": {
                    "method": "iq_ensemble_with_frozen_periodicity_gate",
                    "decision_status": decision_status,
                    "prediction_label": prediction_label,
                    "final_label": final_label,
                    "provisional_label": provisional_label,
                    "resolution": resolution,
                    "review_reason": review_reason,
                    "ensemble_label": labels[ensemble_index],
                    "periodicity_label": labels[periodicity_index],
                },
                "global_feature_evidence": {
                    "feature_schema_id": FEATURE_SCHEMA_ID,
                    "extraction_scope": "complete_continuous_region",
                    "extraction_count": 1,
                    "preprocessing": {
                        "scope": "complete_region",
                        "remove_dc": True,
                        "rms_normalize": True,
                    },
                    "feature_count": FEATURE_COUNT,
                    "values": values,
                    "reference_documents": list(REFERENCE_DOCUMENTS),
                    "reference_document_paths": _reference_document_paths(),
                },
                "objective": (
                    "Explain the frozen deterministic fusion result and its "
                    "review disposition without changing either one"
                ),
                "decision_contract": {
                    "result_source": "deterministic_fusion_result",
                    "result_is_frozen": True,
                    "llm_role": (
                        "explain model, periodicity, and global-feature evidence; "
                        "identify limitations and anomalies without reclassification"
                    ),
                    "requirements": [
                        "inspect all 64 feature values and all three ensemble members",
                        "read and ground interpretations in all five referenced documents",
                        "treat member disagreement as uncertainty, not as a class vote count",
                        "do not recalculate scores, change the frozen threshold, or rerun the gate",
                        "copy decision_status, prediction_label, final_label, and provisional_label exactly",
                        "do not use the 64 features to change a label or resolve a review case",
                        "use low confidence_level whenever decision_status is review_required",
                    ],
                },
                "required_output": {
                    "decision_status": "exactly deterministic_fusion_result.decision_status",
                    "prediction_label": "exactly deterministic_fusion_result.prediction_label",
                    "final_label": "exactly deterministic_fusion_result.final_label",
                    "provisional_label": "exactly deterministic_fusion_result.provisional_label",
                    "confidence_level": "high, medium, or low",
                    "summary": "concise explanation of the frozen result",
                    "model_evidence": "list of model-based reasons",
                    "periodicity_evidence": "list of periodicity-gate reasons",
                    "feature_evidence": "list of feature-based reasons",
                    "conflicting_evidence": (
                        "list; empty when no material conflict exists"
                    ),
                    "limitations": "list of limitations",
                },
            }
        )

        true_index = None
        if region_dataset.y is not None:
            true_index = int(region_dataset.y[row])
        audit_records.append(
            {
                "analysis_id": case_name,
                "group_id": selected_group_id,
                "source_id": str(source_ids[row]),
                "source_region_id": int(source_region_ids[row]),
                "true_class_index": true_index,
                "true_label": None if true_index is None else labels[true_index],
                "ensemble_class_index": ensemble_index,
                "ensemble_label": labels[ensemble_index],
                "ensemble_correct": (
                    None if true_index is None else ensemble_index == true_index
                ),
                "periodicity_class_index": periodicity_index,
                "periodicity_label": labels[periodicity_index],
                "periodicity_margin": float(periodicity_margins[offset]),
                "periodicity_gate": {
                    "eligible": eligible,
                    "reliable": reliable,
                    "resolved": resolved,
                    "changed": changed,
                },
                "prediction_class_index": prediction_index,
                "prediction_label": prediction_label,
                "prediction_correct": (
                    None if true_index is None else prediction_index == true_index
                ),
                "final_label": final_label,
                "final_correct": (
                    None
                    if true_index is None or final_label is None
                    else prediction_index == true_index
                ),
                "decision_status": decision_status,
                "resolution": resolution,
                "review_reason": review_reason,
                "source_path": _private_value(
                    region_dataset, "sample_source_path", row
                ),
                "location": _private_value(region_dataset, "sample_location", row),
                "filename_location": _private_value(
                    region_dataset, "sample_filename_location", row
                ),
                "gain_db": _private_value(region_dataset, "gain_db", row),
                "center_frequency_hz": _private_value(
                    region_dataset, "center_frequency", row
                ),
                "run": _private_value(region_dataset, "run", row),
            }
        )
    return blind_records, audit_records


def write_evidence_records(
    blind_records: Sequence[dict[str, Any]],
    audit_records: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write anonymous case files and one private audit manifest."""

    output = Path(output_dir)
    blind_dir = output / "blind"
    audit_path = output / "audit.json"
    case_paths = [blind_dir / f"{record['analysis_id']}.json" for record in blind_records]
    existing = [path for path in (*case_paths, audit_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "evidence outputs already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing[:5])
        )

    blind_dir.mkdir(parents=True, exist_ok=True)
    for record, path in zip(blind_records, case_paths, strict=True):
        path.write_text(
            json.dumps(json_safe(record), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    audit_payload = {
        "schema_version": 2,
        "audit_type": "private_region_periodicity_fusion_evidence_audit",
        "case_count": len(audit_records),
        "cases": list(audit_records),
    }
    output.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(json_safe(audit_payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output),
        "blind_dir": str(blind_dir),
        "audit_path": str(audit_path),
        "case_count": len(blind_records),
    }


def build_ensemble_evidence_files(
    *,
    dataset_path: str | Path,
    region_dataset_path: str | Path,
    feature_path: str | Path,
    periodicity_gate_manifest_path: str | Path,
    services: Sequence[ModelInferenceService],
    output_dir: str | Path,
    group_id: int | None = None,
    case_id_start: int = 1,
    batch_size: int = 64,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Load artifacts, build evidence, and write blind/audit outputs."""

    window_dataset = load_prepared_dataset(dataset_path)
    region_dataset = load_prepared_dataset(region_dataset_path)
    (
        features,
        feature_names,
        feature_sample_rate,
        feature_seq_len,
        feature_row_identity,
    ) = (
        load_feature_artifact(feature_path)
    )
    region_sample_rate = _region_sample_rate(region_dataset)
    if feature_sample_rate != region_sample_rate:
        raise ValueError(
            "feature and continuous-region sample rates do not match: "
            f"{feature_sample_rate} != {region_sample_rate}"
        )
    if feature_seq_len != region_dataset.seq_len:
        raise ValueError(
            "feature and continuous-region lengths do not match: "
            f"{feature_seq_len} != {region_dataset.seq_len}"
        )
    ensemble_service = RegionEnsembleInferenceService(services)
    blind_records, audit_records = build_evidence_records(
        window_dataset,
        region_dataset,
        features,
        feature_names,
        feature_row_identity,
        ensemble_service,
        periodicity_gate_manifest_path,
        group_id=group_id,
        case_id_start=case_id_start,
        batch_size=batch_size,
    )
    return write_evidence_records(
        blind_records,
        audit_records,
        output_dir,
        overwrite=overwrite,
    )


__all__ = [
    "FEATURE_REFERENCE_DOCUMENTS",
    "REFERENCE_DOCUMENTS",
    "build_ensemble_evidence_files",
    "build_evidence_records",
    "load_feature_artifact",
    "write_evidence_records",
]
