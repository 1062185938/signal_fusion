"""Labeled-dataset evaluation for the region IQ-model ensemble."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.evaluation.snr import classification_metrics
from signal_fusion.model_inference.contracts import ModelInferenceResult


def _macro_accuracy(metrics: dict[str, Any]) -> float:
    accuracies = [
        item["accuracy_percent"]
        for item in metrics["per_class"].values()
        if item["accuracy_percent"] is not None
    ]
    if not accuracies:
        raise ValueError("metrics do not contain any represented classes")
    return float(np.mean(accuracies))


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "window": metrics["window"],
        "region": {
            **metrics["group"],
            "macro_accuracy_percent": _macro_accuracy(metrics["group"]),
        },
    }


def _bucket_summary(mask: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    correct_count = int(np.count_nonzero(correct[mask]))
    return {
        "region_count": count,
        "coverage_percent": float(count / mask.size * 100.0),
        "correct_count": correct_count,
        "error_count": count - correct_count,
        "accuracy_percent": (
            float(correct_count / count * 100.0) if count else None
        ),
    }


def evaluate_ensemble_results(
    dataset: PreparedDataset,
    member_results: Sequence[ModelInferenceResult],
) -> dict[str, Any]:
    """Evaluate full-dataset member predictions and their region ensemble.

    Each member is run on every window before this function is called. Window
    probabilities are averaged within each region for each member, then the
    member region probabilities are averaged to produce the ensemble decision.
    """

    members = tuple(member_results)
    if len(members) < 2:
        raise ValueError("ensemble evaluation requires at least two members")
    if dataset.y is None:
        raise ValueError("ensemble evaluation requires labeled data")

    sample_count = dataset.num_samples
    group_ids = np.asarray(dataset.meta.get("group_id"))
    source_ids = np.asarray(dataset.meta.get("sample_source_id")).astype(str)
    for field_name, values in (
        ("group_id", group_ids),
        ("sample_source_id", source_ids),
    ):
        if values.shape != (sample_count,):
            raise ValueError(
                f"{field_name} metadata must contain one value per sample"
            )

    reference = members[0]
    expected_indices = np.arange(sample_count, dtype=np.int64)
    model_ids: list[str] = []
    for result in members:
        if result.labels != reference.labels:
            raise ValueError("all ensemble members must use the same label order")
        if result.num_samples != sample_count:
            raise ValueError("every ensemble member must predict every window")
        if not np.array_equal(result.sample_indices, expected_indices):
            raise ValueError(
                "ensemble member sample_indices must cover the dataset in order"
            )
        model_ids.append(result.model_id)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("ensemble model_id values must be unique")

    labels = reference.labels
    label_map = {str(index): label for index, label in enumerate(labels)}
    probabilities = np.stack(
        [result.probabilities for result in members], axis=0
    ).astype(np.float64, copy=False)
    ensemble_window_probabilities = probabilities.mean(axis=0)

    unique_group_ids = np.unique(group_ids)
    region_labels: list[int] = []
    region_sources: list[str] = []
    member_region_probabilities: list[np.ndarray] = []
    for group_id in unique_group_ids:
        mask = group_ids == group_id
        group_labels = np.unique(dataset.y[mask])
        group_sources = np.unique(source_ids[mask])
        if group_labels.size != 1 or group_sources.size != 1:
            raise ValueError(
                f"group_id={group_id!r} must belong to one label and source"
            )
        region_labels.append(int(group_labels[0]))
        region_sources.append(str(group_sources[0]))
        member_region_probabilities.append(probabilities[:, mask, :].mean(axis=1))

    member_region = np.stack(member_region_probabilities, axis=1)
    ensemble_region = member_region.mean(axis=0)
    region_labels_array = np.asarray(region_labels, dtype=np.int64)
    region_sources_array = np.asarray(region_sources)
    ensemble_predictions = ensemble_region.argmax(axis=1)
    correct = ensemble_predictions == region_labels_array
    member_predictions = member_region.argmax(axis=2)
    accepted = np.all(
        member_predictions == member_predictions[0:1, :], axis=0
    )
    review_required = ~accepted

    member_summaries: list[dict[str, Any]] = []
    for result in members:
        metrics = classification_metrics(
            result.probabilities,
            dataset.y,
            group_ids,
            source_ids,
            label_map,
        )
        member_summaries.append(
            {
                "model_id": result.model_id,
                "provider": result.provider,
                "metrics": _compact_metrics(metrics),
            }
        )

    ensemble_metrics = classification_metrics(
        ensemble_window_probabilities,
        dataset.y,
        group_ids,
        source_ids,
        label_map,
    )

    per_source: dict[str, dict[str, Any]] = {}
    source_accuracies_by_class: dict[int, list[float]] = {
        index: [] for index in range(len(labels))
    }
    for source_id in sorted(set(region_sources)):
        mask = region_sources_array == source_id
        source_labels = np.unique(region_labels_array[mask])
        if source_labels.size != 1:
            raise ValueError(f"source_id={source_id!r} contains multiple labels")
        source_label = int(source_labels[0])
        source_correct = int(np.count_nonzero(correct[mask]))
        source_count = int(np.count_nonzero(mask))
        source_accuracy = float(source_correct / source_count * 100.0)
        source_accuracies_by_class[source_label].append(source_accuracy)
        per_source[source_id] = {
            "label": source_label,
            "class_name": labels[source_label],
            "region_count": source_count,
            "correct_count": source_correct,
            "error_count": source_count - source_correct,
            "accuracy_percent": source_accuracy,
        }

    source_accuracies = [
        item["accuracy_percent"] for item in per_source.values()
    ]
    class_source_macro = {
        str(class_index): {
            "class_name": labels[class_index],
            "source_count": len(accuracies),
            "accuracy_percent": (
                float(np.mean(accuracies)) if accuracies else None
            ),
        }
        for class_index, accuracies in source_accuracies_by_class.items()
    }
    represented_class_source_accuracies = [
        item["accuracy_percent"]
        for item in class_source_macro.values()
        if item["accuracy_percent"] is not None
    ]

    total_errors = int(np.count_nonzero(~correct))
    reviewed_errors = int(np.count_nonzero(review_required & ~correct))
    review_ids = unique_group_ids[review_required]
    return {
        "schema_version": 1,
        "result_type": "labeled_region_iq_model_ensemble_evaluation",
        "labels": list(labels),
        "sample_count": sample_count,
        "region_count": int(unique_group_ids.size),
        "source_count": len(per_source),
        "aggregation": {
            "within_member": "mean_window_probability_per_region",
            "across_members": "mean_region_probability",
        },
        "members": member_summaries,
        "ensemble": {
            "metrics": _compact_metrics(ensemble_metrics),
            "source_macro_accuracy_percent": float(np.mean(source_accuracies)),
            "class_balanced_source_macro_accuracy_percent": float(
                np.mean(represented_class_source_accuracies)
            ),
            "per_class_source_macro": class_source_macro,
            "per_source": per_source,
        },
        "risk_gate": {
            "method": "unanimous_member_region_top1",
            "accept": _bucket_summary(accepted, correct),
            "review_required": _bucket_summary(review_required, correct),
            "error_detection": {
                "total_ensemble_errors": total_errors,
                "errors_sent_to_review": reviewed_errors,
                "error_recall_percent": (
                    float(reviewed_errors / total_errors * 100.0)
                    if total_errors
                    else None
                ),
            },
            "review_group_ids": [
                int(group_id) for group_id in review_ids.tolist()
            ],
        },
    }


__all__ = ["evaluate_ensemble_results"]
