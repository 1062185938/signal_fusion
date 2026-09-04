"""Fuse one region-level handcrafted feature vector with local model results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.evaluation import add_complex_awgn, standardize_iq_windows
from signal_fusion.feature_extraction import (
    FEATURE_SCHEMA_ID,
    MAX_SIGNAL_LENGTH,
    FeatureBackend,
    FeatureExtractionService,
)
from signal_fusion.fusion.region import CompleteRegion
from signal_fusion.fusion.weights import FusionManifest
from signal_fusion.model_inference import ModelInferenceService


if TYPE_CHECKING:
    from signal_fusion.feature_classifier.service import FeatureClassifierService


MAX_REGION_FEATURE_SAMPLES = MAX_SIGNAL_LENGTH
FEATURE_REFERENCE_ROOT = "src/signal_fusion/feature_extraction/assets"


def _sample_field(dataset: PreparedDataset, name: str) -> np.ndarray:
    if name not in dataset.meta:
        raise ValueError(f"dataset is missing per-sample metadata {name!r}")
    values = np.asarray(dataset.meta[name])
    if values.shape != (dataset.num_samples,):
        raise ValueError(
            f"metadata {name!r} must have shape [{dataset.num_samples}], "
            f"got {values.shape}"
        )
    return values


def _one_selected_value(
    dataset: PreparedDataset,
    name: str,
    indices: np.ndarray,
    *,
    required: bool = True,
) -> Any | None:
    if name not in dataset.meta:
        if required:
            raise ValueError(f"dataset is missing metadata {name!r}")
        return None
    values = np.asarray(dataset.meta[name])
    if values.size == 1:
        return values.reshape(()).item()
    if values.shape != (dataset.num_samples,):
        raise ValueError(
            f"metadata {name!r} must be scalar or have shape "
            f"[{dataset.num_samples}], got {values.shape}"
        )
    unique = np.unique(values[indices])
    if unique.size != 1:
        raise ValueError(
            f"selected group contains multiple {name!r} values: "
            f"{unique.tolist()}"
        )
    return unique[0].item()


def _complex_to_x(samples: np.ndarray) -> np.ndarray:
    iq = np.asarray(samples, dtype=np.complex64).reshape(-1)
    x = np.empty((1, 2, iq.size), dtype=np.float32)
    x[0, 0] = iq.real
    x[0, 1] = iq.imag
    return x


def _canonical_region(
    region: CompleteRegion,
    *,
    snr_db: float | None,
    noise_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    region_x = _complex_to_x(region.samples)
    if snr_db is None:
        analysis_x = standardize_iq_windows(region_x)
        noise = {
            "applied": False,
            "type": "clean",
            "scope": "complete_region_before_windowing",
            "requested_snr_db": None,
            "seed": None,
            "achieved_snr_db": None,
        }
    else:
        if isinstance(noise_seed, bool) or int(noise_seed) != noise_seed:
            raise TypeError("noise_seed must be an integer")
        resolved_seed = int(noise_seed)
        analysis_x, achieved_snr = add_complex_awgn(
            region_x,
            float(snr_db),
            rng=np.random.default_rng(resolved_seed),
            remove_dc=True,
            rms_normalize=True,
        )
        noise = {
            "applied": True,
            "type": "circular_complex_awgn",
            "scope": "complete_region_before_windowing",
            "requested_snr_db": float(snr_db),
            "seed": resolved_seed,
            "achieved_snr_db": float(achieved_snr[0]),
            "interpretation": (
                "complete_input_region_power_to_newly_added_noise_power"
            ),
        }
    canonical = (
        analysis_x[0, 0].astype(np.float32, copy=False)
        + 1j * analysis_x[0, 1].astype(np.float32, copy=False)
    ).astype(np.complex64, copy=False)
    return canonical, noise


def _model_windows(
    dataset: PreparedDataset,
    indices: np.ndarray,
    region: CompleteRegion,
    canonical_region: np.ndarray,
) -> np.ndarray:
    starts = _sample_field(dataset, "window_start_sample")[indices].astype(
        np.int64, copy=False
    )
    ends = _sample_field(dataset, "window_end_sample")[indices].astype(
        np.int64, copy=False
    )
    lengths = ends - starts
    if not np.all(lengths == dataset.seq_len):
        raise ValueError(
            "selected model windows must all match the assembled seq_len"
        )

    windows = np.empty((indices.size, 2, dataset.seq_len), dtype=np.float32)
    for output_index, (start, end) in enumerate(zip(starts, ends)):
        local_start = int(start) - region.start_sample
        local_end = int(end) - region.start_sample
        if local_start < 0 or local_end > region.sample_count:
            raise ValueError(
                f"model window [{start}, {end}) falls outside the complete region"
            )
        window = canonical_region[local_start:local_end]
        windows[output_index, 0] = window.real
        windows[output_index, 1] = window.imag

    remove_dc = bool(_one_selected_value(dataset, "remove_dc", indices))
    rms_normalize = bool(
        _one_selected_value(dataset, "rms_normalize", indices)
    )
    rms_epsilon = float(
        _one_selected_value(dataset, "rms_epsilon", indices)
    )
    return standardize_iq_windows(
        windows,
        remove_dc=remove_dc,
        rms_normalize=rms_normalize,
        epsilon=rms_epsilon,
    )


def _feature_evidence(
    feature_vector: np.ndarray,
    feature_map: dict[str, Any],
    *,
    original_sample_count: int,
) -> dict[str, Any]:
    values = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
    groups: dict[str, list[dict[str, Any]]] = {
        "time_domain": [],
        "frequency_domain": [],
        "time_frequency": [],
    }
    for item in feature_map["features"]:
        index = int(item["index"])
        groups[str(item["group"])].append(
            {
                "index": index,
                "code_name": str(item["code_name"]),
                "display_name": str(item["display_name"]),
                "display_name_zh": str(item["display_name_zh"]),
                "value": float(values[index]),
                "reference": str(item["reference"]),
            }
        )
    used_sample_count = min(original_sample_count, MAX_REGION_FEATURE_SAMPLES)
    return {
        "producer": FEATURE_SCHEMA_ID,
        "scope": "complete_continuous_region",
        "feature_count": int(values.size),
        "extraction_count": 1,
        "reference_root": FEATURE_REFERENCE_ROOT,
        "input": {
            "original_sample_count": int(original_sample_count),
            "used_sample_count": int(used_sample_count),
            "maximum_sample_count": MAX_REGION_FEATURE_SAMPLES,
            "truncated": bool(original_sample_count > MAX_REGION_FEATURE_SAMPLES),
            "truncation_policy": "keep_first_samples",
        },
        "groups": groups,
    }


def _iq_model_evidence(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    *,
    model_id: str,
    provider: str,
) -> dict[str, Any]:
    mean_probabilities = probabilities.mean(axis=0)
    region_top1 = int(np.argmax(mean_probabilities))
    window_top1 = probabilities.argmax(axis=1)
    agreement_count = int(np.count_nonzero(window_top1 == region_top1))
    ranked = np.argsort(mean_probabilities)[::-1][: min(3, len(labels))]
    return {
        "producer": model_id,
        "provider": provider,
        "labels": list(labels),
        "region_probabilities": [float(value) for value in mean_probabilities],
        "region_top3": [
            {
                "label": labels[int(class_index)],
                "probability": float(mean_probabilities[class_index]),
            }
            for class_index in ranked
        ],
        "region_aggregation": "mean_probability_across_selected_windows",
        "window_agreement": {
            "consensus_label": labels[region_top1],
            "agreeing_windows": agreement_count,
            "window_count": int(len(window_top1)),
            "ratio": float(agreement_count / len(window_top1)),
        },
        "window_predictions": [
            {
                "label": labels[int(class_index)],
                "confidence": float(probability[class_index]),
            }
            for class_index, probability in zip(window_top1, probabilities)
        ],
    }


def _feature_model_evidence(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    *,
    model_id: str,
    provider: str,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (1, len(labels)):
        raise ValueError(
            "feature classifier probabilities must have shape "
            f"[1, {len(labels)}], got {values.shape}"
        )
    ranked = np.argsort(values[0])[::-1][: min(3, len(labels))]
    return {
        "producer": model_id,
        "provider": provider,
        "labels": list(labels),
        "region_probabilities": [float(value) for value in values[0]],
        "region_top3": [
            {
                "label": labels[int(class_index)],
                "probability": float(values[0, class_index]),
            }
            for class_index in ranked
        ],
        "input_scope": "complete_continuous_region_62_features",
    }


def analyze_group_branches(
    dataset: PreparedDataset,
    region: CompleteRegion,
    *,
    group_id: int,
    model_service: ModelInferenceService,
    feature_classifier_service: FeatureClassifierService,
    feature_service: FeatureExtractionService | None = None,
    feature_backend: FeatureBackend | None = None,
    batch_size: int = 64,
    snr_db: float | None = None,
    noise_seed: int = 44,
) -> dict[str, Any]:
    """Analyze one complete region and its selected local model windows."""

    group_ids = _sample_field(dataset, "group_id").astype(np.int64, copy=False)
    selected_indices = np.flatnonzero(group_ids == int(group_id)).astype(np.int64)
    if selected_indices.size == 0:
        raise ValueError(f"group_id={group_id} does not exist in the dataset")

    expected_source_id = str(
        _one_selected_value(dataset, "sample_source_id", selected_indices)
    )
    expected_region_id = int(
        _one_selected_value(dataset, "source_region_id", selected_indices)
    )
    expected_start = int(
        _one_selected_value(dataset, "region_start_sample", selected_indices)
    )
    expected_end = int(
        _one_selected_value(dataset, "region_end_sample", selected_indices)
    )
    expected_rate = float(
        _one_selected_value(dataset, "sample_rate", selected_indices)
    )
    if (
        region.source_id != expected_source_id
        or region.source_region_id != expected_region_id
        or region.start_sample != expected_start
        or region.end_sample != expected_end
        or not np.isclose(region.sample_rate, expected_rate)
    ):
        raise ValueError("complete region does not match the selected dataset group")

    canonical_region, noise_evidence = _canonical_region(
        region,
        snr_db=snr_db,
        noise_seed=noise_seed,
    )
    model_x = _model_windows(
        dataset,
        selected_indices,
        region,
        canonical_region,
    )
    model_result = model_service.predict(
        PreparedDataset(X=model_x, source_id=region.source_id),
        source_id=region.source_id,
        batch_size=batch_size,
    )

    feature_samples = canonical_region[:MAX_REGION_FEATURE_SAMPLES]
    feature_service = feature_service or FeatureExtractionService()
    feature_result = feature_service.extract(
        PreparedDataset(X=_complex_to_x(feature_samples), source_id=region.source_id),
        sample_rate=region.sample_rate,
        backend=feature_backend,
        progress_every=0,
    )
    if feature_result.features.shape[0] != 1:
        raise RuntimeError("region feature extraction must return exactly one vector")
    if not np.all(np.isfinite(feature_result.features)):
        raise ValueError("feature extraction produced non-finite values")

    if tuple(feature_result.feature_names) != tuple(
        feature_classifier_service.feature_names
    ):
        raise ValueError(
            "feature classifier feature order does not match extracted features"
        )
    feature_model_result = feature_classifier_service.predict(
        feature_result.features
    )
    if tuple(feature_model_result.labels) != tuple(model_result.labels):
        raise ValueError(
            "IQ model and feature classifier labels must match in the same order"
        )

    iq_model_evidence = _iq_model_evidence(
        model_result.probabilities,
        model_result.labels,
        model_id=model_result.model_id,
        provider=model_result.provider,
    )
    feature_model_evidence = _feature_model_evidence(
        feature_model_result.probabilities,
        feature_model_result.labels,
        model_id=feature_model_result.model_id,
        provider=feature_model_result.provider,
    )
    ground_truth = None
    if dataset.y is not None:
        selected_labels = np.unique(dataset.y[selected_indices])
        if selected_labels.size != 1:
            raise ValueError("selected group contains multiple ground-truth labels")
        class_index = int(selected_labels[0])
        if class_index < 0 or class_index >= len(model_result.labels):
            raise ValueError(
                f"ground-truth class index {class_index} is outside model labels"
            )
        ground_truth = {
            "class_index": class_index,
            "label": model_result.labels[class_index],
        }

    return {
        "schema_version": 4,
        "bundle_type": "hermes_signal_evidence_audit",
        "source": {
            "source_id": region.source_id,
            "raw_source_path": region.raw_source_path,
            "source_dataset_path": region.source_dataset_path,
        },
        "ground_truth": ground_truth,
        "region": {
            "group_id": int(group_id),
            "source_region_id": region.source_region_id,
            "start_sample": region.start_sample,
            "end_sample": region.end_sample,
            "sample_count": region.sample_count,
            "duration_ms": region.sample_count / region.sample_rate * 1000.0,
            "sample_rate": region.sample_rate,
            "selected_window_count": int(selected_indices.size),
            "window_length": dataset.seq_len,
        },
        "noise": noise_evidence,
        "iq_model_evidence": iq_model_evidence,
        "feature_model_evidence": feature_model_evidence,
        "feature_evidence": _feature_evidence(
            feature_result.features[0],
            feature_service.feature_map,
            original_sample_count=region.sample_count,
        ),
        "interpretation_notes": [
            "source provenance is audit-only and must not be classification evidence",
            "the 62 features are extracted once from the continuous region",
            "the IQ model uses local windows cut from that same continuous region",
            "the feature classifier consumes the same single 62-feature vector",
            "no probability fusion rule is applied in this phase",
        ],
    }


def _fusion_result(
    audit_bundle: dict[str, Any],
    manifest: FusionManifest,
) -> dict[str, Any]:
    iq = audit_bundle["iq_model_evidence"]
    feature = audit_bundle["feature_model_evidence"]
    if tuple(iq["labels"]) != manifest.labels:
        raise ValueError("IQ model labels do not match the fusion manifest")
    if tuple(feature["labels"]) != manifest.labels:
        raise ValueError("feature classifier labels do not match the fusion manifest")
    probabilities = manifest.combine(
        np.asarray(iq["region_probabilities"], dtype=np.float64),
        np.asarray(feature["region_probabilities"], dtype=np.float64),
    )[0]
    ranked = np.argsort(probabilities)[::-1][: min(3, len(manifest.labels))]
    return {
        "method": "weighted_probability_average",
        "weights": {
            "iq_model": manifest.iq_model_weight,
            "feature_classifier": manifest.feature_classifier_weight,
        },
        "region_top3": [
            {
                "label": manifest.labels[int(class_index)],
                "probability": float(probabilities[class_index]),
            }
            for class_index in ranked
        ],
        "final_label": manifest.labels[int(ranked[0])],
    }


def analyze_group(
    dataset: PreparedDataset,
    region: CompleteRegion,
    *,
    group_id: int,
    model_service: ModelInferenceService,
    feature_classifier_service: FeatureClassifierService,
    fusion_manifest: FusionManifest,
    feature_service: FeatureExtractionService | None = None,
    feature_backend: FeatureBackend | None = None,
    batch_size: int = 64,
    snr_db: float | None = None,
    noise_seed: int = 44,
) -> dict[str, Any]:
    """Analyze both branches and apply a validated frozen fusion manifest."""

    bundle = analyze_group_branches(
        dataset,
        region,
        group_id=group_id,
        model_service=model_service,
        feature_classifier_service=feature_classifier_service,
        feature_service=feature_service,
        feature_backend=feature_backend,
        batch_size=batch_size,
        snr_db=snr_db,
        noise_seed=noise_seed,
    )
    bundle["fusion_result"] = _fusion_result(bundle, fusion_manifest)
    bundle["interpretation_notes"][-1] = (
        "the frozen fusion manifest is applied after both branches"
    )
    return bundle


def build_hermes_input(
    audit_bundle: dict[str, Any],
    *,
    case_id: int,
) -> dict[str, Any]:
    """Project an internal audit bundle into a blind Hermes input."""

    if isinstance(case_id, bool) or int(case_id) != case_id:
        raise TypeError("case_id must be an integer")
    case_id = int(case_id)
    if case_id <= 0:
        raise ValueError("case_id must be positive")

    region = audit_bundle["region"]
    iq_model = audit_bundle["iq_model_evidence"]
    feature_model = audit_bundle["feature_model_evidence"]
    fusion = audit_bundle["fusion_result"]
    features = audit_bundle["feature_evidence"]
    return {
        "schema_version": 5,
        "bundle_type": "hermes_signal_fusion_input",
        "analysis_id": f"case_{case_id:04d}",
        "signal_context": {
            "duration_ms": region["duration_ms"],
            "sample_rate": region["sample_rate"],
            "region_sample_count": region["sample_count"],
            "selected_window_count": region["selected_window_count"],
            "window_length": region["window_length"],
        },
        "iq_model_evidence": {
            "region_top3": iq_model["region_top3"],
            "region_aggregation": iq_model["region_aggregation"],
            "window_agreement": iq_model["window_agreement"],
            "window_predictions": iq_model["window_predictions"],
        },
        "feature_model_evidence": {
            "region_top3": feature_model["region_top3"],
            "input_scope": feature_model["input_scope"],
        },
        "fusion_result": {
            "method": fusion["method"],
            "weights": fusion["weights"],
            "region_top3": fusion["region_top3"],
            "final_label": fusion["final_label"],
        },
        "feature_evidence": {
            "producer": features["producer"],
            "scope": features["scope"],
            "feature_count": features["feature_count"],
            "extraction_count": features["extraction_count"],
            "reference_root": features["reference_root"],
            "input": features["input"],
            "groups": features["groups"],
        },
        "objective": (
            "Explain the deterministic fusion result using the two classifier "
            "branches and the global 62-dimensional feature observations"
        ),
        "decision_rules": [
            "Use fusion_result.final_label as the final classification",
            "Treat the two classifier Top3 outputs as separate evidence branches",
            "Use window agreement only to describe IQ-model local stability",
            "Use the 62 values as the global physical description of the region",
            "Read all three feature documents under reference_root completely before interpreting the feature values",
            "Do not treat an individual handcrafted feature as a class signature",
            "State material conflicts between either classifier and the global features",
            "Explain agreement or conflict without changing the frozen fusion result",
            "Use only high, medium, or low for the final confidence level",
            "Treat this as a blind case and do not inspect surrounding files",
            "Do not assume hidden acquisition or experiment conditions",
            "Use only observations explicitly present in this input",
        ],
        "required_output": {
            "final_label": "exactly fusion_result.final_label",
            "confidence_level": "high, medium, or low",
            "summary": "concise fusion conclusion",
            "model_evidence": "list of model-based reasons",
            "feature_evidence": "list of feature-based reasons",
            "conflicting_evidence": "list; empty when no material conflict exists",
            "limitations": "list of limitations",
        },
    }


__all__ = [
    "FEATURE_REFERENCE_ROOT",
    "MAX_REGION_FEATURE_SAMPLES",
    "analyze_group",
    "analyze_group_branches",
    "build_hermes_input",
]
