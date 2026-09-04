"""Select a frozen fusion weight on validation data and evaluate test data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.feature_classifier import (
    FeatureClassifierService,
    load_region_feature_split,
)
from signal_fusion.feature_extraction import (
    FeatureExtractionService,
    IQFeatureCtypesBackend,
)
from signal_fusion.fusion.region import load_complete_region
from signal_fusion.fusion.service import analyze_group_branches
from signal_fusion.fusion.weights import FUSION_METHOD, FusionManifest
from signal_fusion.io import load_prepared_dataset
from signal_fusion.model_inference import (
    ModelInferenceService,
    ONNXRuntimeBackend,
    load_label_map,
)


def _sample_field(dataset, name: str) -> np.ndarray:
    if name not in dataset.meta:
        raise ValueError(f"dataset is missing per-sample metadata {name!r}")
    values = np.asarray(dataset.meta[name])
    if values.shape != (dataset.num_samples,):
        raise ValueError(
            f"metadata {name!r} must have shape [{dataset.num_samples}], "
            f"got {values.shape}"
        )
    return values


def _probabilities(
    evidence: dict[str, Any],
    labels: tuple[str, ...],
) -> np.ndarray:
    if tuple(evidence["labels"]) != labels:
        raise ValueError("evidence labels do not match the evaluation labels")
    values = np.asarray(evidence["region_probabilities"], dtype=np.float64)
    if values.shape != (len(labels),):
        raise ValueError("evidence probability vector has an invalid shape")
    return values


class _PrecomputedFeatureBackend:
    def __init__(self, feature_vector: np.ndarray) -> None:
        self.feature_vector = np.asarray(feature_vector, dtype=np.float32)

    def extract_features(self, i_data, q_data, sample_rate) -> np.ndarray:
        return self.feature_vector.copy()


def _precomputed_condition(
    feature_split: dict[str, np.ndarray],
    *,
    group_id: int,
    snr_db: float | None,
) -> tuple[np.ndarray, float | None, int | None] | None:
    group_mask = feature_split["group_id"] == group_id
    if snr_db is None:
        mask = group_mask & (feature_split["condition"] == "clean")
    else:
        requested = feature_split.get("requested_snr_db")
        if requested is None:
            return None
        mask = group_mask & np.isclose(requested, snr_db, equal_nan=False)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    if indices.size != 1:
        raise ValueError(
            f"feature split has multiple matching rows for group_id={group_id}"
        )
    index = int(indices[0])
    if snr_db is None:
        return feature_split["features"][index], None, None
    seeds = feature_split.get("noise_seed")
    if seeds is None:
        raise ValueError("precomputed stress features are missing noise_seed")
    return feature_split["features"][index], float(snr_db), int(seeds[index])


def _collect_split(
    split_path: Path,
    *,
    feature_split: dict[str, np.ndarray],
    stress_snr_db: float,
    seed: int,
    model_service: ModelInferenceService,
    feature_classifier_service: FeatureClassifierService,
    feature_service: FeatureExtractionService,
    feature_backend,
    batch_size: int,
) -> dict[str, Any]:
    dataset = load_prepared_dataset(split_path)
    if dataset.y is None:
        raise ValueError(f"split has no labels: {split_path}")
    group_ids = _sample_field(dataset, "group_id").astype(np.int64, copy=False)
    unique_groups = np.unique(group_ids)
    if not np.array_equal(np.unique(feature_split["group_id"]), unique_groups):
        raise ValueError("IQ and feature split group_id values do not match")
    if tuple(str(name) for name in feature_split["feature_names"]) != tuple(
        feature_service.feature_names
    ):
        raise ValueError("feature split order does not match the runtime feature map")
    labels = tuple(model_service.manifest.labels)
    records: list[dict[str, Any]] = []
    reused_feature_count = 0
    extracted_feature_count = 0

    for position, group_id in enumerate(unique_groups, start=1):
        iq_indices = np.flatnonzero(group_ids == group_id)
        feature_indices = np.flatnonzero(feature_split["group_id"] == group_id)
        iq_labels = np.unique(dataset.y[iq_indices])
        feature_labels = np.unique(feature_split["y"][feature_indices])
        if (
            iq_labels.size != 1
            or feature_labels.size != 1
            or int(iq_labels[0]) != int(feature_labels[0])
        ):
            raise ValueError(
                f"IQ and feature labels differ for group_id={int(group_id)}"
            )
        region = load_complete_region(
            split_path,
            dataset,
            group_id=int(group_id),
        )
        for condition, snr_db in (("clean", None), ("stress", stress_snr_db)):
            precomputed = _precomputed_condition(
                feature_split,
                group_id=int(group_id),
                snr_db=snr_db,
            )
            if precomputed is None:
                resolved_snr = snr_db
                resolved_seed = seed + int(group_id)
                resolved_backend = feature_backend
                extracted_feature_count += 1
            else:
                feature_vector, resolved_snr, resolved_seed = precomputed
                resolved_backend = _PrecomputedFeatureBackend(feature_vector)
                reused_feature_count += 1
            bundle = analyze_group_branches(
                dataset,
                region,
                group_id=int(group_id),
                model_service=model_service,
                feature_classifier_service=feature_classifier_service,
                feature_service=feature_service,
                feature_backend=resolved_backend,
                batch_size=batch_size,
                snr_db=resolved_snr,
                noise_seed=(
                    seed + int(group_id)
                    if resolved_seed is None
                    else resolved_seed
                ),
            )
            truth = bundle["ground_truth"]
            if truth is None:
                raise ValueError("weight selection requires region labels")
            records.append(
                {
                    "group_id": int(group_id),
                    "condition": condition,
                    "label": int(truth["class_index"]),
                    "iq": _probabilities(bundle["iq_model_evidence"], labels),
                    "feature": _probabilities(
                        bundle["feature_model_evidence"], labels
                    ),
                }
            )
        if position % 25 == 0 or position == len(unique_groups):
            print(
                f"[fusion-weight] {split_path.stem}: "
                f"{position}/{len(unique_groups)} regions"
            )

    return {
        "labels": labels,
        "group_id": np.asarray([item["group_id"] for item in records]),
        "condition": np.asarray([item["condition"] for item in records]),
        "y": np.asarray([item["label"] for item in records], dtype=np.int64),
        "iq": np.asarray([item["iq"] for item in records], dtype=np.float64),
        "feature": np.asarray(
            [item["feature"] for item in records], dtype=np.float64
        ),
        "feature_usage": {
            "reused_count": reused_feature_count,
            "extracted_count": extracted_feature_count,
        },
    }


def _classification_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    class_count = len(labels)
    confusion = [
        [
            int(np.count_nonzero((targets == truth) & (predictions == predicted)))
            for predicted in range(class_count)
        ]
        for truth in range(class_count)
    ]
    per_class = {
        labels[index]: float(
            np.mean(predictions[targets == index] == index) * 100.0
        )
        for index in range(class_count)
    }
    selected = probabilities[np.arange(targets.size), targets]
    return {
        "sample_count": int(targets.size),
        "accuracy_percent": float(np.mean(predictions == targets) * 100.0),
        "macro_accuracy_percent": float(np.mean(list(per_class.values()))),
        "negative_log_likelihood": float(
            -np.mean(np.log(np.maximum(selected, 1e-12)))
        ),
        "confusion_matrix": confusion,
        "per_class_accuracy_percent": per_class,
    }


def _disagreement_metrics(
    iq: np.ndarray,
    feature: np.ndarray,
    fused: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    iq_labels = np.argmax(iq, axis=1)
    feature_labels = np.argmax(feature, axis=1)
    mask = iq_labels != feature_labels
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "sample_count": 0,
            "iq_accuracy_percent": None,
            "feature_accuracy_percent": None,
            "fused_accuracy_percent": None,
        }
    return {
        "sample_count": count,
        "iq_accuracy_percent": float(
            np.mean(iq_labels[mask] == targets[mask]) * 100.0
        ),
        "feature_accuracy_percent": float(
            np.mean(feature_labels[mask] == targets[mask]) * 100.0
        ),
        "fused_accuracy_percent": float(
            np.mean(np.argmax(fused[mask], axis=1) == targets[mask]) * 100.0
        ),
    }


def _weight_candidates(step: float) -> list[float]:
    step = float(step)
    if not math.isfinite(step) or step <= 0.0 or step > 1.0:
        raise ValueError("weight_step must be within (0, 1]")
    count = round(1.0 / step)
    if not np.isclose(count * step, 1.0):
        raise ValueError("weight_step must divide 1.0 exactly")
    return [index / count for index in range(count + 1)]


def select_fusion_weight(
    iq_probabilities: np.ndarray,
    feature_probabilities: np.ndarray,
    targets: np.ndarray,
    labels: tuple[str, ...],
    *,
    weight_step: float = 0.1,
) -> tuple[float, list[dict[str, Any]]]:
    """Select IQ weight using validation labels only."""

    iq_probabilities = np.asarray(iq_probabilities, dtype=np.float64)
    feature_probabilities = np.asarray(feature_probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if iq_probabilities.ndim != 2 or feature_probabilities.ndim != 2:
        raise ValueError("validation probabilities must be two-dimensional")
    if targets.ndim != 1 or targets.size != iq_probabilities.shape[0]:
        raise ValueError("validation targets must match the probability rows")
    if np.unique(targets).tolist() != list(range(len(labels))):
        raise ValueError("validation targets must contain every manifest class")
    candidates: list[dict[str, Any]] = []
    for iq_weight in _weight_candidates(weight_step):
        manifest = FusionManifest(
            labels=labels,
            iq_model_weight=iq_weight,
            feature_classifier_weight=1.0 - iq_weight,
        )
        fused = manifest.combine(iq_probabilities, feature_probabilities)
        metrics = _classification_metrics(fused, targets, labels)
        disagreement = _disagreement_metrics(
            iq_probabilities,
            feature_probabilities,
            fused,
            targets,
        )
        candidates.append(
            {
                "iq_model_weight": iq_weight,
                "feature_classifier_weight": 1.0 - iq_weight,
                "metrics": metrics,
                "disagreement": disagreement,
            }
        )

    def rank(item: dict[str, Any]) -> tuple[float, float, float, float]:
        disagreement_accuracy = item["disagreement"]["fused_accuracy_percent"]
        if disagreement_accuracy is None:
            disagreement_accuracy = -1.0
        return (
            item["metrics"]["macro_accuracy_percent"],
            item["metrics"]["accuracy_percent"],
            disagreement_accuracy,
            -abs(item["iq_model_weight"] - 0.5),
        )

    selected = max(candidates, key=rank)
    return float(selected["iq_model_weight"]), candidates


def _evaluate_split(
    collected: dict[str, Any],
    manifest: FusionManifest,
) -> dict[str, Any]:
    fused = manifest.combine(collected["iq"], collected["feature"])

    def evaluate(mask: np.ndarray) -> dict[str, Any]:
        targets = collected["y"][mask]
        return {
            "iq_model": _classification_metrics(
                collected["iq"][mask], targets, manifest.labels
            ),
            "feature_classifier": _classification_metrics(
                collected["feature"][mask], targets, manifest.labels
            ),
            "fusion": _classification_metrics(
                fused[mask], targets, manifest.labels
            ),
            "disagreement": _disagreement_metrics(
                collected["iq"][mask],
                collected["feature"][mask],
                fused[mask],
                targets,
            ),
        }

    result = {"overall": evaluate(np.ones(collected["y"].shape, dtype=bool))}
    result["conditions"] = {
        str(condition): evaluate(collected["condition"] == condition)
        for condition in np.unique(collected["condition"])
    }
    return result


def select_and_evaluate_fusion(
    *,
    dataset_dir: str | Path,
    model_path: str | Path,
    label_map_path: str | Path,
    feature_classifier_manifest: str | Path,
    feature_dataset_dir: str | Path,
    output_dir: str | Path,
    stress_snr_db: float = 5.0,
    weight_step: float = 0.1,
    seed: int = 44,
    batch_size: int = 64,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select on validation, freeze, then evaluate test."""

    stress_snr_db = float(stress_snr_db)
    if not math.isfinite(stress_snr_db):
        raise ValueError("stress_snr_db must be finite")
    if isinstance(seed, bool) or int(seed) != seed:
        raise TypeError("seed must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset_directory = Path(dataset_dir)
    split_paths = {
        split: dataset_directory / f"{split}.npz"
        for split in ("validation", "test")
    }
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing assembled splits: " + ", ".join(missing))
    output_directory = Path(output_dir)
    manifest_path = output_directory / "fusion_manifest.json"
    report_path = output_directory / "fusion_evaluation.json"
    existing = [path for path in (manifest_path, report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "fusion outputs exist; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    labels = load_label_map(label_map_path)
    backend = ONNXRuntimeBackend(model_path)
    model_manifest = backend.build_manifest(
        model_id=Path(model_path).stem,
        labels=labels,
    )
    model_service = ModelInferenceService(model_manifest, backend=backend)
    feature_classifier = FeatureClassifierService(feature_classifier_manifest)
    feature_service = FeatureExtractionService()
    feature_dataset_directory = Path(feature_dataset_dir)
    feature_splits = {
        split: load_region_feature_split(
            feature_dataset_directory / f"{split}.npz"
        )
        for split in ("validation", "test")
    }

    collected: dict[str, dict[str, Any]] = {}
    with IQFeatureCtypesBackend() as feature_backend:
        for split, path in split_paths.items():
            collected[split] = _collect_split(
                path,
                feature_split=feature_splits[split],
                stress_snr_db=stress_snr_db,
                seed=int(seed),
                model_service=model_service,
                feature_classifier_service=feature_classifier,
                feature_service=feature_service,
                feature_backend=feature_backend,
                batch_size=batch_size,
            )
    validation = collected["validation"]
    selected_weight, candidates = select_fusion_weight(
        validation["iq"],
        validation["feature"],
        validation["y"],
        validation["labels"],
        weight_step=weight_step,
    )
    manifest = FusionManifest(
        labels=validation["labels"],
        iq_model_weight=selected_weight,
        feature_classifier_weight=1.0 - selected_weight,
    )
    manifest_json = {
        "schema_version": 1,
        "method": FUSION_METHOD,
        "labels": list(manifest.labels),
        "weights": {
            "iq_model": manifest.iq_model_weight,
            "feature_classifier": manifest.feature_classifier_weight,
        },
        "selection": {
            "split": "validation",
            "conditions": ["clean", f"controlled_{stress_snr_db:g}_db"],
            "weight_step": float(weight_step),
            "primary_metric": "macro_accuracy_percent",
            "tie_breakers": [
                "accuracy_percent",
                "disagreement_accuracy_percent",
                "closest_to_equal_weights",
            ],
        },
    }
    report = {
        "schema_version": 1,
        "evaluation_type": "validation_selected_weighted_probability_fusion",
        "dataset_dir": str(dataset_directory),
        "feature_dataset_dir": str(feature_dataset_directory),
        "labels": list(manifest.labels),
        "configuration": {
            "stress_snr_db": stress_snr_db,
            "seed": int(seed),
            "batch_size": int(batch_size),
            "weight_step": float(weight_step),
        },
        "validation_candidates": candidates,
        "selected_manifest": manifest_json,
        "validation": _evaluate_split(validation, manifest),
        "test": _evaluate_split(collected["test"], manifest),
        "feature_usage": {
            split: collected[split]["feature_usage"]
            for split in ("validation", "test")
        },
        "outputs": {
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


__all__ = ["select_and_evaluate_fusion", "select_fusion_weight"]
