"""Validated weighted-probability fusion contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


FUSION_METHOD = "weighted_probability_average"


@dataclass(frozen=True, slots=True)
class FusionManifest:
    labels: tuple[str, ...]
    iq_model_weight: float
    feature_classifier_weight: float

    def __post_init__(self) -> None:
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("fusion manifest labels must be non-empty and unique")
        if (
            not np.isfinite(self.iq_model_weight)
            or not np.isfinite(self.feature_classifier_weight)
            or self.iq_model_weight < 0.0
            or self.feature_classifier_weight < 0.0
            or not np.isclose(
                self.iq_model_weight + self.feature_classifier_weight,
                1.0,
            )
        ):
            raise ValueError(
                "fusion weights must be finite, non-negative, and sum to 1"
            )

    @classmethod
    def load(cls, path: str | Path) -> "FusionManifest":
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("fusion manifest schema_version must be 1")
        if manifest.get("method") != FUSION_METHOD:
            raise ValueError(f"fusion manifest method must be {FUSION_METHOD!r}")
        labels = tuple(str(label) for label in manifest["labels"])
        weights = manifest["weights"]
        iq_weight = float(weights["iq_model"])
        feature_weight = float(weights["feature_classifier"])
        return cls(
            labels=labels,
            iq_model_weight=iq_weight,
            feature_classifier_weight=feature_weight,
        )

    def combine(
        self,
        iq_probabilities: np.ndarray,
        feature_probabilities: np.ndarray,
    ) -> np.ndarray:
        iq = _probabilities(iq_probabilities, len(self.labels), "IQ model")
        feature = _probabilities(
            feature_probabilities,
            len(self.labels),
            "feature classifier",
        )
        if iq.shape != feature.shape:
            raise ValueError(
                "IQ model and feature classifier probability shapes must match"
            )
        return (
            self.iq_model_weight * iq
            + self.feature_classifier_weight * feature
        )


def _probabilities(values: np.ndarray, class_count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] != class_count:
        raise ValueError(
            f"{name} probabilities must have shape [N, {class_count}], "
            f"got {array.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} probabilities must be finite and non-negative")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{name} probability rows must sum to 1")
    return array


__all__ = ["FUSION_METHOD", "FusionManifest"]
