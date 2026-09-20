"""Region-level probability averaging across independently trained IQ models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.model_inference.backend import ONNXRuntimeBackend
from signal_fusion.model_inference.contracts import ModelInferenceResult
from signal_fusion.model_inference.labels import load_label_map
from signal_fusion.model_inference.service import ModelInferenceService


ACCEPT = "accept"
REVIEW_REQUIRED = "review_required"


def load_ensemble_services(
    model_paths: Sequence[str | Path],
    label_map_path: str | Path,
) -> tuple[ModelInferenceService, ...]:
    """Load exactly three independently trained ONNX IQ classifiers."""

    paths = tuple(Path(path) for path in model_paths)
    if len(paths) != 3:
        raise ValueError("ensemble inference requires exactly three model paths")
    labels = load_label_map(label_map_path)
    services: list[ModelInferenceService] = []
    model_ids: set[str] = set()
    for model_path in paths:
        model_id = model_path.parent.name
        if model_id in model_ids:
            raise ValueError(f"derived ensemble model_id is not unique: {model_id}")
        model_ids.add(model_id)
        backend = ONNXRuntimeBackend(model_path)
        manifest = backend.build_manifest(model_id=model_id, labels=labels)
        services.append(ModelInferenceService(manifest, backend=backend))
    return tuple(services)


def _top_k(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    count: int = 3,
) -> list[dict[str, Any]]:
    ranked = np.argsort(-probabilities, kind="stable")[: min(count, len(labels))]
    return [
        {
            "rank": rank,
            "class_index": int(class_index),
            "label": labels[int(class_index)],
            "probability": float(probabilities[class_index]),
        }
        for rank, class_index in enumerate(ranked, start=1)
    ]


def _uncertainty(probabilities: np.ndarray) -> dict[str, float]:
    ranked = np.sort(probabilities)
    if ranked.size == 1:
        return {
            "top1_probability": float(ranked[-1]),
            "top1_top2_margin": 1.0,
            "normalized_entropy": 0.0,
        }
    clipped = np.clip(probabilities, 1e-12, 1.0)
    normalized_entropy = -float(np.sum(clipped * np.log(clipped))) / math.log(
        probabilities.size
    )
    return {
        "top1_probability": float(ranked[-1]),
        "top1_top2_margin": float(ranked[-1] - ranked[-2]),
        "normalized_entropy": normalized_entropy,
    }


@dataclass(slots=True)
class RegionEnsembleResult:
    """One region decision derived from multiple model/window predictions."""

    group_id: int
    source_id: str
    labels: tuple[str, ...]
    member_results: tuple[ModelInferenceResult, ...]
    member_region_probabilities: np.ndarray
    ensemble_probabilities: np.ndarray
    decision_status: str

    def __post_init__(self) -> None:
        self.labels = tuple(self.labels)
        self.member_results = tuple(self.member_results)
        if len(self.member_results) < 2:
            raise ValueError("region ensemble requires at least two member results")
        expected_member_shape = (len(self.member_results), len(self.labels))
        if self.member_region_probabilities.shape != expected_member_shape:
            raise ValueError(
                "member_region_probabilities must have shape "
                f"{expected_member_shape}, got {self.member_region_probabilities.shape}"
            )
        if self.ensemble_probabilities.shape != (len(self.labels),):
            raise ValueError(
                "ensemble_probabilities must contain one value per label"
            )
        if self.decision_status not in {ACCEPT, REVIEW_REQUIRED}:
            raise ValueError(f"unsupported decision_status={self.decision_status!r}")

    def to_dict(self) -> dict[str, Any]:
        member_top1 = self.member_region_probabilities.argmax(axis=1)
        unique_top1 = np.unique(member_top1)
        members: list[dict[str, Any]] = []
        for result, region_probabilities, region_top1 in zip(
            self.member_results,
            self.member_region_probabilities,
            member_top1,
        ):
            window_top1 = result.probabilities.argmax(axis=1)
            agreeing_windows = int(np.count_nonzero(window_top1 == region_top1))
            members.append(
                {
                    "model_id": result.model_id,
                    "provider": result.provider,
                    "region_probabilities": [
                        float(value) for value in region_probabilities
                    ],
                    "region_top3": _top_k(region_probabilities, self.labels),
                    "window_aggregation": "mean_probability",
                    "window_agreement": {
                        "agreeing_windows": agreeing_windows,
                        "window_count": result.num_samples,
                        "ratio": float(agreeing_windows / result.num_samples),
                    },
                    "window_predictions": [
                        {
                            "sample_index": int(sample_index),
                            "label": self.labels[int(class_index)],
                            "confidence": float(probability[class_index]),
                        }
                        for sample_index, class_index, probability in zip(
                            result.sample_indices,
                            window_top1,
                            result.probabilities,
                        )
                    ],
                }
            )

        return {
            "schema_version": 1,
            "result_type": "region_iq_model_ensemble",
            "group_id": self.group_id,
            "source_id": self.source_id,
            "labels": list(self.labels),
            "decision_status": self.decision_status,
            "risk_gate": {
                "method": "unanimous_member_region_top1",
                "member_count": len(self.member_results),
                "unanimous": bool(unique_top1.size == 1),
                "unique_top1_labels": [
                    self.labels[int(class_index)] for class_index in unique_top1
                ],
                "reason": (
                    None
                    if unique_top1.size == 1
                    else "member_region_top1_disagreement"
                ),
            },
            "ensemble": {
                "aggregation": "mean_probability_across_members",
                "region_probabilities": [
                    float(value) for value in self.ensemble_probabilities
                ],
                "region_top3": _top_k(
                    self.ensemble_probabilities,
                    self.labels,
                ),
                "uncertainty": _uncertainty(self.ensemble_probabilities),
            },
            "members": members,
        }


def aggregate_region_predictions(
    group_id: int,
    results: Sequence[ModelInferenceResult],
) -> RegionEnsembleResult:
    """Average windows within each model, then average across models."""

    members = tuple(results)
    if len(members) < 2:
        raise ValueError("region ensemble requires at least two model results")
    model_ids = [result.model_id for result in members]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("ensemble model_id values must be unique")

    reference = members[0]
    for result in members[1:]:
        if result.labels != reference.labels:
            raise ValueError("all ensemble members must use the same label order")
        if result.source_id != reference.source_id:
            raise ValueError("all ensemble members must use the same source_id")
        if not np.array_equal(result.sample_indices, reference.sample_indices):
            raise ValueError("all ensemble members must use the same windows")

    member_probabilities = np.stack(
        [result.probabilities.mean(axis=0) for result in members],
        axis=0,
    ).astype(np.float64, copy=False)
    ensemble_probabilities = member_probabilities.mean(axis=0)
    member_top1 = member_probabilities.argmax(axis=1)
    decision_status = (
        ACCEPT
        if np.unique(member_top1).size == 1
        else REVIEW_REQUIRED
    )
    return RegionEnsembleResult(
        group_id=int(group_id),
        source_id=reference.source_id,
        labels=reference.labels,
        member_results=members,
        member_region_probabilities=member_probabilities,
        ensemble_probabilities=ensemble_probabilities,
        decision_status=decision_status,
    )


class RegionEnsembleInferenceService:
    """Run multiple IQ model services on every window of one dataset group."""

    def __init__(self, members: Sequence[ModelInferenceService]) -> None:
        self.members = tuple(members)
        if len(self.members) < 2:
            raise ValueError("region ensemble requires at least two model services")

    @staticmethod
    def _group_indices_and_source_id(
        dataset: PreparedDataset,
        group_id: int,
    ) -> tuple[np.ndarray, str]:
        """Return source-ordered windows after validating complete coverage."""

        if "group_id" not in dataset.meta:
            raise ValueError("ensemble dataset is missing group_id metadata")
        group_ids = np.asarray(dataset.meta["group_id"])
        if group_ids.shape != (dataset.num_samples,):
            raise ValueError(
                "group_id metadata must contain one value per dataset sample"
            )
        indices = np.flatnonzero(group_ids == int(group_id)).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"group_id={group_id} does not exist in the dataset")

        region_fields: dict[str, np.ndarray] = {}
        for field_name in (
            "window_start_sample",
            "window_end_sample",
            "region_start_sample",
            "region_end_sample",
        ):
            if field_name not in dataset.meta:
                raise ValueError(
                    f"ensemble dataset is missing {field_name} metadata"
                )
            values = np.asarray(dataset.meta[field_name])
            if values.shape != (dataset.num_samples,):
                raise ValueError(
                    f"{field_name} metadata must contain one value per sample"
                )
            region_fields[field_name] = values[indices].astype(
                np.int64,
                copy=False,
            )

        region_starts = np.unique(region_fields["region_start_sample"])
        region_ends = np.unique(region_fields["region_end_sample"])
        starts = region_fields["window_start_sample"]
        ends = region_fields["window_end_sample"]
        order = np.argsort(starts, kind="stable")
        starts = starts[order]
        ends = ends[order]
        complete_region = (
            region_starts.size == 1
            and region_ends.size == 1
            and region_ends[0] > region_starts[0]
            and np.all(ends - starts == dataset.seq_len)
            and starts[0] == region_starts[0]
            and ends[-1] == region_ends[0]
            and np.array_equal(starts[1:], ends[:-1])
        )
        if not complete_region:
            raise ValueError(
                "selected group windows must form one complete "
                "non-overlapping region"
            )
        indices = indices[order]

        source_id = dataset.source_id
        if "sample_source_id" in dataset.meta:
            sample_source_ids = np.asarray(dataset.meta["sample_source_id"])
            if sample_source_ids.shape != (dataset.num_samples,):
                raise ValueError(
                    "sample_source_id metadata must contain one value per sample"
                )
            source_ids = np.unique(sample_source_ids[indices].astype(str))
            if source_ids.size != 1:
                raise ValueError(
                    "selected group contains multiple sample_source_id values"
                )
            source_id = str(source_ids[0])
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("ensemble inference requires a source_id")
        return indices, source_id

    @staticmethod
    def _select_result(
        result: ModelInferenceResult,
        indices: np.ndarray,
        source_id: str,
    ) -> ModelInferenceResult:
        if not np.array_equal(
            result.sample_indices,
            np.arange(result.num_samples, dtype=np.int64),
        ):
            raise ValueError(
                "full-dataset ensemble inference must preserve dataset sample order"
            )
        return ModelInferenceResult(
            source_id=source_id,
            model_id=result.model_id,
            labels=result.labels,
            logits=result.logits[indices].copy(),
            probabilities=result.probabilities[indices].copy(),
            auxiliary_outputs={
                name: values[indices].copy()
                for name, values in result.auxiliary_outputs.items()
            },
            sample_indices=indices.copy(),
            provider=result.provider,
            metadata=result.metadata,
        )

    def predict_group(
        self,
        dataset: PreparedDataset,
        *,
        group_id: int,
        batch_size: int = 64,
    ) -> RegionEnsembleResult:
        indices, source_id = self._group_indices_and_source_id(dataset, group_id)

        results = [
            service.predict(
                dataset,
                source_id=source_id,
                sample_indices=indices,
                batch_size=batch_size,
            )
            for service in self.members
        ]
        return aggregate_region_predictions(group_id, results)

    def predict_all_groups(
        self,
        dataset: PreparedDataset,
        *,
        batch_size: int = 64,
    ) -> tuple[RegionEnsembleResult, ...]:
        """Run each model once, then aggregate every complete dataset region."""

        if "group_id" not in dataset.meta:
            raise ValueError("ensemble dataset is missing group_id metadata")
        group_ids = np.asarray(dataset.meta["group_id"])
        if group_ids.shape != (dataset.num_samples,):
            raise ValueError(
                "group_id metadata must contain one value per dataset sample"
            )
        full_results = [
            service.predict(dataset, batch_size=batch_size) for service in self.members
        ]
        region_results: list[RegionEnsembleResult] = []
        for group_id in np.unique(group_ids.astype(np.int64, copy=False)):
            indices, source_id = self._group_indices_and_source_id(dataset, group_id)
            selected = [
                self._select_result(result, indices, source_id)
                for result in full_results
            ]
            region_results.append(
                aggregate_region_predictions(int(group_id), selected)
            )
        return tuple(region_results)


__all__ = [
    "ACCEPT",
    "REVIEW_REQUIRED",
    "RegionEnsembleInferenceService",
    "RegionEnsembleResult",
    "aggregate_region_predictions",
    "load_ensemble_services",
]
