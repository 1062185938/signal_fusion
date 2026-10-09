"""Cross-location IQ/64-feature fusion benchmark.

The benchmark keeps every fold location-isolated:

* the 64-feature linear probe is fitted on the fold's training locations;
* early stopping and fusion parameters use only the validation location;
* the held-out test location is evaluated once with frozen parameters.

The feature probe is an experimental baseline, not a production dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from signal_fusion.benchmarks.technology_recognition import LABELS, LOCATION_FOLDS
from signal_fusion.evaluation.snr_runner import (
    load_evaluation_model,
    predict_probabilities,
    resolve_evaluation_device,
)
from signal_fusion.feature_classifier import FeatureClassifierService
from signal_fusion.feature_classifier.trainer import train_feature_classifier
from signal_fusion.feature_extraction import FEATURE_COUNT, FEATURE_SCHEMA_ID
from signal_fusion.fusion.weights import FusionManifest
from signal_fusion.io import load_prepared_dataset
from signal_fusion.io.writers import json_safe


LABEL_NAMES = tuple(LABELS)
WEIGHT_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))
GATE_THRESHOLD_GRID = tuple(
    float(value) for value in np.linspace(-1.0, 1.0, 41)
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


def _scalar(container: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in container:
        raise ValueError(f"artifact is missing scalar {name!r}")
    value = np.asarray(container[name])
    if value.size != 1:
        raise ValueError(f"artifact field {name!r} must be scalar")
    return value.reshape(()).item()


def _row_keys(source_ids: np.ndarray, source_region_ids: np.ndarray) -> list[tuple[str, int]]:
    return [
        (str(source_id), int(region_id))
        for source_id, region_id in zip(source_ids, source_region_ids, strict=True)
    ]


def load_full_feature_bank(
    dataset_root: str | Path,
    feature_root: str | Path,
) -> dict[str, Any]:
    """Load and validate one extracted feature file per held-out location."""

    dataset_root = Path(dataset_root)
    feature_root = Path(feature_root)
    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "features",
            "y",
            "global_group_id",
            "source_group_id",
            "sample_source_id",
            "source_region_id",
            "sample_rate",
            "location",
            "normalization_scale",
            "window_start_sample",
            "window_end_sample",
        )
    }
    reference_names: np.ndarray | None = None
    reference_label_map: str | None = None
    offset = 0

    for fold_name, split_locations in LOCATION_FOLDS.items():
        location = split_locations["test"][0]
        dataset_path = dataset_root / fold_name / "test.npz"
        feature_path = feature_root / f"{fold_name}_test_features.npz"
        if not dataset_path.is_file() or not feature_path.is_file():
            raise FileNotFoundError(
                f"missing location artifacts: {dataset_path}, {feature_path}"
            )
        with np.load(dataset_path, allow_pickle=False) as raw, np.load(
            feature_path, allow_pickle=False
        ) as feature:
            features = np.asarray(feature["features"])
            labels = np.asarray(raw["y"])
            if features.shape != (labels.size, FEATURE_COUNT):
                raise ValueError(
                    f"{fold_name} feature shape {features.shape} does not match "
                    f"{labels.size} labels"
                )
            if features.dtype != np.float32 or not np.all(np.isfinite(features)):
                raise ValueError(f"{fold_name} features must be finite float32")
            if str(_scalar(feature, "feature_schema_id")) != FEATURE_SCHEMA_ID:
                raise ValueError(f"{fold_name} feature schema is incompatible")
            names = np.asarray(feature["feature_names"]).astype(str)
            if names.shape != (FEATURE_COUNT,) or len(set(names.tolist())) != FEATURE_COUNT:
                raise ValueError(f"{fold_name} feature_names are invalid")
            if reference_names is None:
                reference_names = names.copy()
            elif not np.array_equal(reference_names, names):
                raise ValueError("feature_names differ across locations")

            for identity in ("group_id", "sample_source_id", "source_region_id"):
                if not np.array_equal(raw[identity], feature[identity]):
                    raise ValueError(
                        f"{fold_name} feature rows do not align on {identity}"
                    )
            label_map_json = str(_scalar(raw, "label_map_json"))
            if reference_label_map is None:
                reference_label_map = label_map_json
            elif reference_label_map != label_map_json:
                raise ValueError("label maps differ across locations")

            count = labels.size
            parts["features"].append(features.copy())
            parts["y"].append(labels.astype(np.int64, copy=True))
            parts["global_group_id"].append(
                np.arange(offset, offset + count, dtype=np.int64)
            )
            parts["source_group_id"].append(
                np.asarray(raw["group_id"], dtype=np.int64)
            )
            parts["sample_source_id"].append(
                np.asarray(raw["sample_source_id"]).astype(str)
            )
            parts["source_region_id"].append(
                np.asarray(raw["source_region_id"], dtype=np.int64)
            )
            parts["sample_rate"].append(
                np.asarray(raw["sample_rate"], dtype=np.float64)
            )
            parts["location"].append(np.full(count, location))
            for field in (
                "normalization_scale",
                "window_start_sample",
                "window_end_sample",
            ):
                parts[field].append(np.asarray(raw[field]))
            offset += count

    bank = {name: np.concatenate(values) for name, values in parts.items()}
    bank["feature_names"] = reference_names
    bank["label_map_json"] = reference_label_map
    keys = _row_keys(bank["sample_source_id"], bank["source_region_id"])
    if len(set(keys)) != len(keys):
        raise ValueError("feature bank contains duplicate source/region identities")
    bank["row_lookup"] = {key: index for index, key in enumerate(keys)}
    return bank


def align_feature_rows(
    dataset,
    bank: Mapping[str, Any],
    expected_locations: tuple[str, ...],
) -> np.ndarray:
    """Align a prepared split to the global feature bank without group-id joins."""

    source_ids = _sample_field(dataset, "sample_source_id").astype(str)
    source_region_ids = _sample_field(dataset, "source_region_id").astype(np.int64)
    lookup = bank["row_lookup"]
    try:
        indices = np.asarray(
            [lookup[key] for key in _row_keys(source_ids, source_region_ids)],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"prepared split row is absent from feature bank: {exc}") from exc
    if np.unique(indices).size != dataset.num_samples:
        raise ValueError("prepared split maps multiple rows to one feature row")
    actual_locations = tuple(sorted(set(np.asarray(bank["location"])[indices].tolist())))
    if actual_locations != tuple(sorted(expected_locations)):
        raise ValueError(
            f"split locations {actual_locations} do not match {expected_locations}"
        )
    if dataset.y is None or not np.array_equal(dataset.y, bank["y"][indices]):
        raise ValueError("prepared split labels do not align with feature rows")

    metadata_pairs = (
        ("normalization_scale", np.float64),
        ("window_start_sample", np.int64),
        ("window_end_sample", np.int64),
    )
    for field, dtype in metadata_pairs:
        values = _sample_field(dataset, field).astype(dtype)
        reference = np.asarray(bank[field])[indices].astype(dtype)
        if np.issubdtype(dtype, np.floating):
            matches = np.allclose(values, reference, rtol=1e-6, atol=1e-7)
        else:
            matches = np.array_equal(values, reference)
        if not matches:
            raise ValueError(f"prepared split differs from feature source on {field}")
    return indices


def _write_feature_split(
    path: Path,
    dataset,
    bank: Mapping[str, Any],
    indices: np.ndarray,
    *,
    dataset_id: str,
    split_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(bank["features"])[indices],
        y=np.asarray(bank["y"])[indices],
        group_id=np.asarray(bank["global_group_id"])[indices],
        source_group_id=np.asarray(bank["source_group_id"])[indices],
        source_region_id=np.asarray(bank["source_region_id"])[indices],
        sample_source_id=np.asarray(bank["sample_source_id"])[indices],
        sample_rate=np.asarray(bank["sample_rate"])[indices],
        location=np.asarray(bank["location"])[indices],
        condition=np.full(indices.size, "clean"),
        feature_names=np.asarray(bank["feature_names"]),
        feature_schema_id=np.asarray(FEATURE_SCHEMA_ID),
        feature_dataset_version=np.asarray("cross_location_features_v1"),
        source_dataset_id=np.asarray(str(dataset.source_id)),
        dataset_id=np.asarray(dataset_id),
        label_map_json=np.asarray(str(bank["label_map_json"])),
        split=np.asarray(split_name),
    )


def build_fold_feature_dataset(
    dataset_root: Path,
    bank: Mapping[str, Any],
    fold_name: str,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    """Create train/validation/test feature splits from the shared location bank."""

    paths = {name: output_dir / f"{name}.npz" for name in LOCATION_FOLDS[fold_name]}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "feature split outputs already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    aligned: dict[str, np.ndarray] = {}
    dataset_id = f"technology_recognition_cross_location_64_v2_{fold_name}"
    for split_name, locations in LOCATION_FOLDS[fold_name].items():
        dataset = load_prepared_dataset(
            dataset_root / fold_name / f"{split_name}.npz"
        )
        indices = align_feature_rows(dataset, bank, locations)
        _write_feature_split(
            paths[split_name],
            dataset,
            bank,
            indices,
            dataset_id=dataset_id,
            split_name=split_name,
        )
        aligned[split_name] = indices
    return aligned


def classification_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    per_class = {
        label_name: (
            float(
                np.mean(predictions[targets == label_index] == label_index)
                * 100.0
            )
            if np.any(targets == label_index)
            else None
        )
        for label_index, label_name in enumerate(LABEL_NAMES)
    }
    observed_per_class = [
        value for value in per_class.values() if value is not None
    ]
    selected = probabilities[np.arange(targets.size), targets]
    return {
        "sample_count": int(targets.size),
        "correct_count": int(np.count_nonzero(predictions == targets)),
        "error_count": int(np.count_nonzero(predictions != targets)),
        "accuracy_percent": float(np.mean(predictions == targets) * 100.0),
        "macro_accuracy_percent": float(np.mean(observed_per_class)),
        "negative_log_likelihood": float(
            -np.mean(np.log(np.maximum(selected, 1e-12)))
        ),
        "per_class_accuracy_percent": per_class,
    }


def change_metrics(
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    targets = np.asarray(targets, dtype=np.int64)
    changed = candidate != baseline
    corrected = (baseline != targets) & (candidate == targets)
    harmed = (baseline == targets) & (candidate != targets)
    changed_count = int(np.count_nonzero(changed))
    return {
        "changed_count": changed_count,
        "changed_percent": float(np.mean(changed) * 100.0),
        "corrected_count": int(np.count_nonzero(corrected)),
        "harmed_count": int(np.count_nonzero(harmed)),
        "net_correction_count": int(np.count_nonzero(corrected) - np.count_nonzero(harmed)),
        "override_precision_percent": (
            float(np.count_nonzero(corrected) / changed_count * 100.0)
            if changed_count
            else None
        ),
    }


def combine_probabilities(
    iq_probabilities: np.ndarray,
    feature_probabilities: np.ndarray,
    iq_weight: float,
) -> np.ndarray:
    manifest = FusionManifest(
        labels=LABEL_NAMES,
        iq_model_weight=float(iq_weight),
        feature_classifier_weight=1.0 - float(iq_weight),
    )
    return manifest.combine(iq_probabilities, feature_probabilities)


def select_safe_fusion_weight(
    iq_probabilities: np.ndarray,
    feature_probabilities: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    """Select on validation, preferring the IQ baseline on metric ties."""

    candidates: list[dict[str, Any]] = []
    for iq_weight in WEIGHT_GRID:
        fused = combine_probabilities(
            iq_probabilities, feature_probabilities, iq_weight
        )
        candidates.append(
            {
                "iq_weight": iq_weight,
                "feature_weight": 1.0 - iq_weight,
                "metrics": classification_metrics(fused, targets),
                "changes_from_iq": change_metrics(
                    iq_probabilities, fused, targets
                ),
            }
        )

    def rank(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
        metrics = item["metrics"]
        return (
            float(metrics["macro_accuracy_percent"]),
            float(metrics["accuracy_percent"]),
            float(item["iq_weight"]),
            -float(metrics["negative_log_likelihood"]),
        )

    selected = max(candidates, key=rank)
    return float(selected["iq_weight"]), candidates


def apply_selective_gate(
    iq_probabilities: np.ndarray,
    feature_probabilities: np.ndarray,
    threshold: float | None,
) -> np.ndarray:
    """Use feature probabilities only for sufficiently strong disagreements."""

    iq = np.asarray(iq_probabilities, dtype=np.float64)
    feature = np.asarray(feature_probabilities, dtype=np.float64)
    output = iq.copy()
    if threshold is None:
        return output
    disagreement = iq.argmax(axis=1) != feature.argmax(axis=1)
    relative_confidence = feature.max(axis=1) - iq.max(axis=1)
    override = disagreement & (relative_confidence >= float(threshold))
    output[override] = feature[override]
    return output


def select_safe_gate_threshold(
    iq_probabilities: np.ndarray,
    feature_probabilities: np.ndarray,
    targets: np.ndarray,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Select one validation threshold; metric ties keep more IQ decisions."""

    thresholds: tuple[float | None, ...] = (None, *GATE_THRESHOLD_GRID)
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        fused = apply_selective_gate(
            iq_probabilities, feature_probabilities, threshold
        )
        candidates.append(
            {
                "threshold": threshold,
                "metrics": classification_metrics(fused, targets),
                "changes_from_iq": change_metrics(
                    iq_probabilities, fused, targets
                ),
            }
        )

    def rank(item: Mapping[str, Any]) -> tuple[float, float, int, float]:
        metrics = item["metrics"]
        changes = item["changes_from_iq"]
        threshold = item["threshold"]
        return (
            float(metrics["macro_accuracy_percent"]),
            float(metrics["accuracy_percent"]),
            -int(changes["changed_count"]),
            math.inf if threshold is None else float(threshold),
        )

    selected = max(candidates, key=rank)
    return selected["threshold"], candidates


def disagreement_router_metrics(
    iq_probabilities: np.ndarray,
    feature_probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    """Measure the truth-blind IQ/feature Top1 disagreement review router."""

    iq = np.asarray(iq_probabilities).argmax(axis=1)
    feature = np.asarray(feature_probabilities).argmax(axis=1)
    targets = np.asarray(targets, dtype=np.int64)
    review = iq != feature
    iq_errors = iq != targets
    review_count = int(np.count_nonzero(review))
    error_count = int(np.count_nonzero(iq_errors))
    reviewed_errors = int(np.count_nonzero(review & iq_errors))
    return {
        "rule": "iq_top1_differs_from_feature_top1",
        "sample_count": int(targets.size),
        "review_count": review_count,
        "review_coverage_percent": float(np.mean(review) * 100.0),
        "iq_error_count": error_count,
        "reviewed_iq_error_count": reviewed_errors,
        "iq_error_recall_percent": (
            float(reviewed_errors / error_count * 100.0) if error_count else None
        ),
        "review_error_rate_percent": (
            float(reviewed_errors / review_count * 100.0) if review_count else None
        ),
        "iq_correct_feature_wrong_count": int(
            np.count_nonzero(review & (iq == targets) & (feature != targets))
        ),
        "iq_wrong_feature_correct_count": int(
            np.count_nonzero(review & (iq != targets) & (feature == targets))
        ),
        "both_wrong_count": int(
            np.count_nonzero(review & (iq != targets) & (feature != targets))
        ),
    }


def _evaluate_methods(
    iq: np.ndarray,
    feature: np.ndarray,
    targets: np.ndarray,
    *,
    selected_weight: float,
    selected_threshold: float | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    probabilities = {
        "iq_model": iq,
        "feature_probe": feature,
        "equal_weight_fusion": combine_probabilities(iq, feature, 0.5),
        "validation_weighted_fusion": combine_probabilities(
            iq, feature, selected_weight
        ),
        "validation_selective_gate": apply_selective_gate(
            iq, feature, selected_threshold
        ),
    }
    report = {
        name: {
            "metrics": classification_metrics(values, targets),
            "changes_from_iq": change_metrics(iq, values, targets),
        }
        for name, values in probabilities.items()
    }
    report["disagreement_router"] = disagreement_router_metrics(
        iq, feature, targets
    )
    return report, probabilities


def run_cross_location_fusion_benchmark(
    dataset_root: str | Path,
    feature_root: str | Path,
    model_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 44,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train full-data feature probes and evaluate frozen per-fold fusion."""

    dataset_root = Path(dataset_root)
    model_root = Path(model_root)
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    prediction_path = output_dir / "test_predictions.npz"
    validation_prediction_path = output_dir / "validation_predictions.npz"
    existing = [
        path
        for path in (result_path, prediction_path, validation_prediction_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "benchmark outputs already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = load_full_feature_bank(dataset_root, feature_root)
    resolved_device = resolve_evaluation_device(device)

    fold_reports: dict[str, Any] = {}
    split_records: dict[str, list[dict[str, np.ndarray]]] = {
        "validation": [],
        "test": [],
    }
    for fold_name, fold_locations in LOCATION_FOLDS.items():
        fold_output = output_dir / fold_name
        feature_dataset_dir = fold_output / "feature_dataset"
        aligned = build_fold_feature_dataset(
            dataset_root,
            bank,
            fold_name,
            feature_dataset_dir,
            overwrite=overwrite,
        )
        feature_model_dir = fold_output / "feature_probe"
        training = train_feature_classifier(
            feature_dataset_dir,
            feature_model_dir,
            device=device,
            batch_size=256,
            learning_rate=1e-2,
            max_epochs=300,
            patience=30,
            seed=seed,
            overwrite=overwrite,
        )
        feature_service = FeatureClassifierService(
            feature_model_dir / "feature_classifier_manifest.json"
        )
        if tuple(feature_service.labels) != LABEL_NAMES:
            raise ValueError("feature classifier label order differs from IQ labels")

        model = load_evaluation_model(
            model_root / fold_name / "best_model.pth",
            model_name="deepconvnet_1d",
            class_num=len(LABEL_NAMES),
            input_channels=2,
            seq_len=4096,
            device=resolved_device,
        )
        collected: dict[str, dict[str, np.ndarray]] = {}
        for split_name in ("validation", "test"):
            dataset = load_prepared_dataset(
                dataset_root / fold_name / f"{split_name}.npz"
            )
            indices = aligned[split_name]
            iq_probabilities = predict_probabilities(
                model,
                dataset.X,
                batch_size=batch_size,
                device=resolved_device,
            )
            feature_probabilities = feature_service.predict(
                np.asarray(bank["features"])[indices]
            ).probabilities
            collected[split_name] = {
                "fold": np.full(dataset.num_samples, fold_name),
                "location": np.asarray(bank["location"])[indices],
                "sample_source_id": np.asarray(bank["sample_source_id"])[indices],
                "source_region_id": np.asarray(bank["source_region_id"])[indices],
                "global_group_id": np.asarray(bank["global_group_id"])[indices],
                "y": dataset.y,
                "iq_probabilities": iq_probabilities,
                "feature_probabilities": feature_probabilities,
            }

        validation = collected["validation"]
        selected_weight, weight_candidates = select_safe_fusion_weight(
            validation["iq_probabilities"],
            validation["feature_probabilities"],
            validation["y"],
        )
        selected_threshold, gate_candidates = select_safe_gate_threshold(
            validation["iq_probabilities"],
            validation["feature_probabilities"],
            validation["y"],
        )

        evaluation: dict[str, Any] = {}
        for split_name, record in collected.items():
            report, method_probabilities = _evaluate_methods(
                record["iq_probabilities"],
                record["feature_probabilities"],
                record["y"],
                selected_weight=selected_weight,
                selected_threshold=selected_threshold,
            )
            evaluation[split_name] = report
            enriched = dict(record)
            for method_name, values in method_probabilities.items():
                enriched[f"{method_name}_probabilities"] = values
            split_records[split_name].append(enriched)
            np.savez_compressed(
                fold_output / f"{split_name}_predictions.npz",
                **enriched,
            )

        fold_reports[fold_name] = {
            "locations": {name: list(value) for name, value in fold_locations.items()},
            "feature_probe_training": {
                "best_epoch": training["best_epoch"],
                "epochs_completed": training["epochs_completed"],
                "train_count": training["metrics"]["train"]["sample_count"],
                "validation_count": training["metrics"]["validation"]["sample_count"],
                "test_count": training["metrics"]["test"]["sample_count"],
            },
            "selection": {
                "iq_weight": selected_weight,
                "feature_weight": 1.0 - selected_weight,
                "selective_gate_threshold": selected_threshold,
                "weight_candidates": weight_candidates,
                "gate_candidates": gate_candidates,
            },
            "evaluation": evaluation,
        }
        print(
            f"[cross-location-fusion] {fold_name}: "
            f"weight={selected_weight:.2f}, gate={selected_threshold}, "
            f"IQ={evaluation['test']['iq_model']['metrics']['accuracy_percent']:.3f}%, "
            f"weighted={evaluation['test']['validation_weighted_fusion']['metrics']['accuracy_percent']:.3f}%, "
            f"selective={evaluation['test']['validation_selective_gate']['metrics']['accuracy_percent']:.3f}%"
        )

    combined: dict[str, dict[str, np.ndarray]] = {}
    for split_name, records in split_records.items():
        keys = records[0].keys()
        combined[split_name] = {
            key: np.concatenate([record[key] for record in records], axis=0)
            for key in keys
        }
        destination = (
            validation_prediction_path if split_name == "validation" else prediction_path
        )
        np.savez_compressed(destination, **combined[split_name])

    test = combined["test"]
    aggregate_methods: dict[str, Any] = {}
    for method_name in (
        "iq_model",
        "feature_probe",
        "equal_weight_fusion",
        "validation_weighted_fusion",
        "validation_selective_gate",
    ):
        values = test[f"{method_name}_probabilities"]
        aggregate_methods[method_name] = {
            "metrics": classification_metrics(values, test["y"]),
            "changes_from_iq": change_metrics(
                test["iq_probabilities"], values, test["y"]
            ),
        }
    aggregate = {
        "methods": aggregate_methods,
        "disagreement_router": disagreement_router_metrics(
            test["iq_probabilities"],
            test["feature_probabilities"],
            test["y"],
        ),
    }
    report = {
        "schema_version": 1,
        "analysis_type": "location_isolated_iq_64_feature_fusion",
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "scope": {
            "test_region_count": int(test["y"].size),
            "locations": sorted(set(test["location"].tolist())),
            "classes": list(LABEL_NAMES),
            "condition": "clean",
        },
        "methodology": {
            "feature_probe": "standardized_linear_64_to_3",
            "per_fold_protocol": (
                "train locations fit the probe; validation location selects "
                "fusion parameters; held-out test location is evaluated once"
            ),
            "weighted_fusion_grid": list(WEIGHT_GRID),
            "weight_tie_break": "prefer_more_iq_weight",
            "selective_gate_score": "feature_top1_confidence_minus_iq_top1_confidence",
            "selective_gate_tie_break": "prefer_fewer_overrides",
            "llm_used": False,
        },
        "folds": fold_reports,
        "aggregate_test": aggregate,
        "interpretation_limits": [
            "The 64-feature classifier is an experimental linear probe, not a frozen production branch.",
            "Adjacent regions from one recording are correlated; publication intervals must cluster by sample_source_id.",
            "The four locations are development cross-validation; UZ/iGent remain necessary for final external validation.",
            "No LLM decision is evaluated in this stage.",
        ],
        "outputs": {
            "test_predictions": str(prediction_path),
            "validation_predictions": str(validation_prediction_path),
            "result": str(result_path),
        },
    }
    result_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate location-isolated IQ/64-feature fusion baselines."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_cross_location_fusion_benchmark(
        args.dataset_root,
        args.feature_root,
        args.model_root,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(report["aggregate_test"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "align_feature_rows",
    "apply_selective_gate",
    "change_metrics",
    "classification_metrics",
    "disagreement_router_metrics",
    "load_full_feature_bank",
    "run_cross_location_fusion_benchmark",
    "select_safe_fusion_weight",
    "select_safe_gate_threshold",
]
