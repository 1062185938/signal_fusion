"""Shared data contracts for the signal analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Mapping

import numpy as np


ShapeDimension = int | str | None


def _require_non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _normalize_shape(shape: tuple[ShapeDimension, ...], field_name: str) -> tuple[ShapeDimension, ...]:
    normalized = tuple(shape)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    for dimension in normalized:
        if isinstance(dimension, bool):
            raise TypeError(f"{field_name} contains an invalid boolean dimension")
        if isinstance(dimension, int) and dimension <= 0:
            raise ValueError(f"{field_name} integer dimensions must be positive")
        if isinstance(dimension, str) and not dimension.strip():
            raise ValueError(f"{field_name} symbolic dimensions must not be empty")
        if dimension is not None and not isinstance(dimension, (int, str)):
            raise TypeError(
                f"{field_name} dimensions must be int, str, or None; got {dimension!r}"
            )
    return normalized


@dataclass(slots=True)
class PreparedDataset:
    """Canonical in-memory IQ dataset used by analysis components.

    ``X`` always uses channel-first IQ layout ``[N, 2, L]`` and ``float32``.
    Optional labels use ``[N]`` and ``int64``. ``source_id`` is a readable,
    caller-managed identifier; it is deliberately independent of file hashes.
    """

    X: np.ndarray
    y: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")
        if self.X.ndim != 3 or self.X.shape[1] != 2:
            raise ValueError(
                f"X must have shape [N, 2, L], got shape={self.X.shape}"
            )
        if self.X.dtype != np.float32:
            raise TypeError(f"X must have dtype float32, got {self.X.dtype}")
        if self.X.shape[2] <= 0:
            raise ValueError("X sequence length must be positive")

        if self.y is not None:
            if not isinstance(self.y, np.ndarray):
                raise TypeError("y must be a numpy.ndarray or None")
            if self.y.ndim != 1 or self.y.shape[0] != self.X.shape[0]:
                raise ValueError(
                    "y must have shape [N] matching X; "
                    f"got y.shape={self.y.shape}, X.shape={self.X.shape}"
                )
            if self.y.dtype != np.int64:
                raise TypeError(f"y must have dtype int64, got {self.y.dtype}")

        self.meta = dict(self.meta)
        if self.source_id is not None:
            self.source_id = _require_non_empty_text(self.source_id, "source_id")

    @property
    def num_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.X.shape[2])

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the dictionary shape consumed by the existing CLIs."""

        meta = dict(self.meta)
        if self.source_id is not None:
            meta.setdefault("source_id", self.source_id)
        return {"X": self.X, "y": self.y, "meta": meta}


@dataclass(slots=True)
class Evidence:
    """One explainable observation produced from a signal source."""

    source_id: str
    kind: str
    producer: str
    payload: dict[str, Any]
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_id = _require_non_empty_text(self.source_id, "source_id")
        self.kind = _require_non_empty_text(self.kind, "kind")
        self.producer = _require_non_empty_text(self.producer, "producer")
        self.payload = dict(self.payload)
        self.metadata = dict(self.metadata)

        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, Real):
                raise TypeError("confidence must be a real number or None")
            value = float(self.confidence)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("confidence must be finite and within [0, 1]")
            self.confidence = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "producer": self.producer,
            "payload": dict(self.payload),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ModelManifest:
    """Runtime-facing description of a small signal model."""

    model_id: str
    model_format: str
    model_path: str
    input_name: str
    input_shape: tuple[ShapeDimension, ...]
    outputs: Mapping[str, tuple[ShapeDimension, ...]]
    labels: tuple[str, ...] = ()
    input_dtype: str = "float32"
    model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_id = _require_non_empty_text(self.model_id, "model_id")
        self.model_format = _require_non_empty_text(
            self.model_format, "model_format"
        ).lower()
        self.model_path = _require_non_empty_text(self.model_path, "model_path")
        self.input_name = _require_non_empty_text(self.input_name, "input_name")
        self.input_dtype = _require_non_empty_text(self.input_dtype, "input_dtype")
        self.input_shape = _normalize_shape(tuple(self.input_shape), "input_shape")

        normalized_outputs: dict[str, tuple[ShapeDimension, ...]] = {}
        for name, shape in dict(self.outputs).items():
            output_name = _require_non_empty_text(name, "output name")
            normalized_outputs[output_name] = _normalize_shape(
                tuple(shape), f"outputs[{output_name!r}]"
            )
        if not normalized_outputs:
            raise ValueError("outputs must contain at least one model output")
        self.outputs = normalized_outputs

        self.labels = tuple(_require_non_empty_text(label, "label") for label in self.labels)
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique and index ordered")
        if self.model_version is not None:
            self.model_version = _require_non_empty_text(
                self.model_version, "model_version"
            )
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_format": self.model_format,
            "model_path": self.model_path,
            "model_version": self.model_version,
            "input": {
                "name": self.input_name,
                "shape": list(self.input_shape),
                "dtype": self.input_dtype,
            },
            "outputs": {
                name: list(shape) for name, shape in self.outputs.items()
            },
            "labels": list(self.labels),
            "metadata": dict(self.metadata),
        }

