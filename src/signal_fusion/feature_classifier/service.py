"""ONNX runtime service for the region-level feature classifier."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.feature_extraction import FEATURE_COUNT, FEATURE_SCHEMA_ID


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class FeatureClassifierResult:
    probabilities: np.ndarray
    labels: tuple[str, ...]
    model_id: str
    provider: str

    def top_k(self, sample_index: int = 0, k: int = 3) -> list[dict[str, Any]]:
        if sample_index < 0 or sample_index >= self.probabilities.shape[0]:
            raise IndexError("sample_index is outside the classifier result")
        count = min(max(int(k), 1), len(self.labels))
        ranked = np.argsort(self.probabilities[sample_index])[::-1][:count]
        return [
            {
                "label": self.labels[int(index)],
                "probability": float(self.probabilities[sample_index, index]),
            }
            for index in ranked
        ]


class FeatureClassifierService:
    """Standardize 64 features and run the exported linear ONNX model."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        providers: list[str] | None = None,
    ) -> None:
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("feature classifier manifest schema_version must be 1")
        if manifest.get("feature_schema_id") != FEATURE_SCHEMA_ID:
            raise ValueError("feature classifier uses an incompatible feature schema")
        feature_names = tuple(str(name) for name in manifest["feature_names"])
        if len(feature_names) != FEATURE_COUNT or len(set(feature_names)) != FEATURE_COUNT:
            raise ValueError(f"feature_names must contain {FEATURE_COUNT} unique names")
        labels = tuple(str(label) for label in manifest["labels"])
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("feature classifier labels must be non-empty and unique")

        scaler_path = (path.parent / manifest["scaler_path"]).resolve()
        with np.load(scaler_path, allow_pickle=False) as scaler:
            self.mean = np.asarray(scaler["mean"], dtype=np.float32)
            self.scale = np.asarray(scaler["scale"], dtype=np.float32)
        if self.mean.shape != (FEATURE_COUNT,) or self.scale.shape != (FEATURE_COUNT,):
            raise ValueError(
                f"feature scaler must contain {FEATURE_COUNT}-dimensional mean and scale"
            )
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("feature scaler contains non-finite values")
        if np.any(self.scale <= 0.0):
            raise ValueError("feature scaler scale values must be positive")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "feature classifier inference requires onnxruntime"
            ) from exc
        model_path = (path.parent / manifest["model_path"]).resolve()
        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = str(manifest["input_name"])
        self.output_name = str(manifest["output_name"])
        self.labels = labels
        self.feature_names = feature_names
        self.model_id = str(manifest["model_id"])
        self.provider = self.session.get_providers()[0]

    def predict(self, features: np.ndarray) -> FeatureClassifierResult:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
            raise ValueError(f"features must have shape [N, {FEATURE_COUNT}]")
        if not np.all(np.isfinite(values)):
            raise ValueError("features contain NaN or Inf")
        standardized = ((values - self.mean) / self.scale).astype(np.float32)
        logits = self.session.run(
            [self.output_name], {self.input_name: standardized}
        )[0]
        probabilities = _softmax(np.asarray(logits, dtype=np.float64))
        return FeatureClassifierResult(
            probabilities=probabilities,
            labels=self.labels,
            model_id=self.model_id,
            provider=self.provider,
        )


__all__ = ["FeatureClassifierResult", "FeatureClassifierService"]
