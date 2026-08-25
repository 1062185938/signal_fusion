"""Contracts for the 62-dimensional IQ feature extraction runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from signal_fusion.contracts import Evidence


FEATURE_COUNT = 62
FEATURE_SCHEMA_ID = "matlab_iq_features_62_v1"


@dataclass(slots=True)
class FeatureResult:
    """Numerical feature output kept separate from its Evidence projection."""

    source_id: str
    features: np.ndarray
    feature_names: tuple[str, ...]
    sample_rate: float
    seq_len: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.features, np.ndarray):
            raise TypeError("features must be a numpy.ndarray")
        if self.features.ndim != 2 or self.features.shape[1] != FEATURE_COUNT:
            raise ValueError(
                f"features must have shape [N, {FEATURE_COUNT}], "
                f"got {self.features.shape}"
            )
        if self.features.dtype != np.float32:
            raise TypeError(
                f"features must have dtype float32, got {self.features.dtype}"
            )

        self.feature_names = tuple(self.feature_names)
        if len(self.feature_names) != FEATURE_COUNT:
            raise ValueError(
                f"feature_names must contain {FEATURE_COUNT} names, "
                f"got {len(self.feature_names)}"
            )
        if any(not isinstance(name, str) or not name for name in self.feature_names):
            raise ValueError("feature_names must contain non-empty strings")
        if len(set(self.feature_names)) != FEATURE_COUNT:
            raise ValueError("feature_names must be unique and index ordered")

        self.sample_rate = float(self.sample_rate)
        if not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        self.seq_len = int(self.seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.metadata = dict(self.metadata)

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_count(self) -> int:
        return FEATURE_COUNT

    def evidence_for_sample(self, sample_index: int = 0) -> Evidence:
        """Create one JSON-friendly Evidence item without hiding raw results."""

        index = int(sample_index)
        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"sample_index={index} is outside [0, {self.num_samples - 1}]"
            )
        values = self.features[index]
        return Evidence(
            source_id=self.source_id,
            kind="iq_features",
            producer=FEATURE_SCHEMA_ID,
            payload={
                "sample_index": index,
                "feature_count": FEATURE_COUNT,
                "values": {
                    name: float(value)
                    for name, value in zip(self.feature_names, values)
                },
            },
            metadata={
                "sample_rate": self.sample_rate,
                "seq_len": self.seq_len,
                "feature_schema_id": FEATURE_SCHEMA_ID,
            },
        )


__all__ = ["FEATURE_COUNT", "FEATURE_SCHEMA_ID", "FeatureResult"]
