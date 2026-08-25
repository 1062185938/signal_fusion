"""Structured contracts for small-model inference results."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from signal_fusion.contracts import Evidence


@dataclass(frozen=True, slots=True)
class RankedPrediction:
    """One class in a sample's confidence-ordered prediction list."""

    rank: int
    class_index: int
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, Integral):
            raise TypeError("rank must be an integer")
        if int(self.rank) <= 0:
            raise ValueError("rank must be positive")
        if isinstance(self.class_index, bool) or not isinstance(
            self.class_index, Integral
        ):
            raise TypeError("class_index must be an integer")
        if int(self.class_index) < 0:
            raise ValueError("class_index must be non-negative")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, Real):
            raise TypeError("confidence must be a real number")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")

        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "class_index", int(self.class_index))
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "class_index": self.class_index,
            "label": self.label,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class ModelInferenceResult:
    """Raw model outputs plus probabilities and explainable projections."""

    source_id: str
    model_id: str
    labels: tuple[str, ...]
    logits: np.ndarray
    probabilities: np.ndarray
    auxiliary_outputs: dict[str, np.ndarray]
    sample_indices: np.ndarray
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("source_id", "model_id", "provider"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        self.labels = tuple(self.labels)
        if not self.labels or any(
            not isinstance(label, str) or not label.strip() for label in self.labels
        ):
            raise ValueError("labels must contain non-empty strings")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique and index ordered")

        self._validate_matrix(self.logits, "logits")
        self._validate_matrix(self.probabilities, "probabilities")
        if self.logits.shape != self.probabilities.shape:
            raise ValueError(
                "logits and probabilities must have the same shape; "
                f"got {self.logits.shape} and {self.probabilities.shape}"
            )
        if self.logits.shape[1] != len(self.labels):
            raise ValueError(
                f"labels={len(self.labels)} does not match model classes="
                f"{self.logits.shape[1]}"
            )
        if not np.all(np.isfinite(self.logits)):
            raise ValueError("logits must contain only finite values")
        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("probabilities must contain only finite values")
        if np.any(self.probabilities < 0.0) or np.any(self.probabilities > 1.0):
            raise ValueError("probabilities must be within [0, 1]")
        if not np.allclose(
            self.probabilities.sum(axis=1),
            1.0,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("probability rows must sum to one")

        if not isinstance(self.sample_indices, np.ndarray):
            raise TypeError("sample_indices must be a numpy.ndarray")
        if self.sample_indices.ndim != 1:
            raise ValueError("sample_indices must have shape [N]")
        if self.sample_indices.dtype != np.int64:
            raise TypeError(
                f"sample_indices must have dtype int64, got {self.sample_indices.dtype}"
            )
        if self.sample_indices.shape[0] != self.logits.shape[0]:
            raise ValueError("sample_indices length must match inference batch size")
        if np.any(self.sample_indices < 0):
            raise ValueError("sample_indices must be non-negative")

        normalized_outputs: dict[str, np.ndarray] = {}
        for name, values in dict(self.auxiliary_outputs).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("auxiliary output names must be non-empty strings")
            if not isinstance(values, np.ndarray):
                raise TypeError(f"auxiliary_outputs[{name!r}] must be a numpy.ndarray")
            if values.ndim == 0 or values.shape[0] != self.logits.shape[0]:
                raise ValueError(
                    f"auxiliary_outputs[{name!r}] first dimension must match "
                    "the inference batch"
                )
            normalized_outputs[name] = values
        self.auxiliary_outputs = normalized_outputs
        self.metadata = dict(self.metadata)

    @staticmethod
    def _validate_matrix(values: np.ndarray, name: str) -> None:
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(f"{name} must have non-empty shape [N, C]")
        if values.dtype != np.float32:
            raise TypeError(f"{name} must have dtype float32, got {values.dtype}")

    @property
    def num_samples(self) -> int:
        return int(self.logits.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.logits.shape[1])

    def top_k_for_sample(
        self,
        sample_index: int = 0,
        top_k: int = 5,
    ) -> tuple[RankedPrediction, ...]:
        index = int(sample_index)
        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"sample_index={index} is outside [0, {self.num_samples - 1}]"
            )
        requested_top_k = int(top_k)
        if requested_top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        resolved_top_k = min(requested_top_k, self.num_classes)
        indices = np.argsort(self.probabilities[index])[-resolved_top_k:][::-1]
        return tuple(
            RankedPrediction(
                rank=rank,
                class_index=int(class_index),
                label=self.labels[int(class_index)],
                confidence=float(self.probabilities[index, class_index]),
            )
            for rank, class_index in enumerate(indices, start=1)
        )

    def evidence_for_sample(
        self,
        sample_index: int = 0,
        top_k: int = 5,
    ) -> Evidence:
        predictions = self.top_k_for_sample(sample_index, top_k)
        original_index = int(self.sample_indices[int(sample_index)])
        return Evidence(
            source_id=self.source_id,
            kind="modulation_prediction",
            producer=self.model_id,
            payload={
                "sample_index": original_index,
                "top_k": [prediction.to_dict() for prediction in predictions],
            },
            confidence=predictions[0].confidence,
            metadata={
                "model_id": self.model_id,
                "provider": self.provider,
                **self.metadata,
            },
        )


__all__ = ["ModelInferenceResult", "RankedPrediction"]
