"""Dataset-facing service for MATLAB-generated IQ features."""

from __future__ import annotations

import math
from os import PathLike
from typing import Protocol

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.feature_extraction.backend import IQFeatureCtypesBackend
from signal_fusion.feature_extraction.contracts import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_ID,
    FeatureResult,
)
from signal_fusion.feature_extraction.feature_map import (
    feature_code_names,
    load_feature_map,
)


class FeatureBackend(Protocol):
    def extract_features(
        self,
        i_data: np.ndarray,
        q_data: np.ndarray,
        sample_rate: float,
    ) -> np.ndarray: ...


def _positive_scalar(value: object, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(
            f"{name} must be a scalar, got shape={np.asarray(value).shape}"
        )
    number = float(array[0])
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive, got {number!r}")
    return number


def resolve_sample_rate(
    dataset: PreparedDataset,
    explicit_sample_rate: float | None = None,
) -> float:
    """Resolve Hz from an explicit value or canonical dataset metadata."""

    if explicit_sample_rate is not None:
        return _positive_scalar(explicit_sample_rate, "sample_rate")
    for key in ("sample_rate", "sampling_rate", "fs", "Fs"):
        if key in dataset.meta:
            return _positive_scalar(dataset.meta[key], key)
    raise ValueError(
        "MATLAB 特征提取需要采样率。请显式传入 sample_rate，"
        "或在 PreparedDataset.meta 中提供 sample_rate/fs。"
    )


def extract_feature_matrix(
    x: np.ndarray,
    sample_rate: float,
    backend: FeatureBackend,
    progress_every: int = 100,
) -> np.ndarray:
    """Extract one stable 62-dimensional row for each canonical IQ sample."""

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected X shape [N, 2, seq_len], got {x.shape}")
    num_samples = int(x.shape[0])
    features = np.empty((num_samples, FEATURE_COUNT), dtype=np.float32)
    for index in range(num_samples):
        features[index] = backend.extract_features(
            x[index, 0, :],
            x[index, 1, :],
            sample_rate,
        )
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"[feature] extracted {index + 1}/{num_samples} samples")
    return features


class FeatureExtractionService:
    """Convert a PreparedDataset into numerical features and Evidence."""

    def __init__(self, feature_map_path: str | PathLike[str] | None = None) -> None:
        self.feature_map = load_feature_map(feature_map_path)
        self.feature_names = feature_code_names(self.feature_map)

    def extract(
        self,
        dataset: PreparedDataset,
        *,
        sample_rate: float | None = None,
        source_id: str | None = None,
        max_samples: int | None = None,
        backend: FeatureBackend | None = None,
        dll_dir: str | PathLike[str] | None = None,
        dll_path: str | PathLike[str] | None = None,
        dependency_dirs: list[str | PathLike[str]] | None = None,
        progress_every: int = 100,
    ) -> FeatureResult:
        resolved_sample_rate = resolve_sample_rate(dataset, sample_rate)
        resolved_source_id = source_id or dataset.source_id
        if resolved_source_id is None:
            metadata_source_id = dataset.meta.get("source_id")
            if metadata_source_id is not None:
                resolved_source_id = str(np.asarray(metadata_source_id).item())
        if not isinstance(resolved_source_id, str) or not resolved_source_id.strip():
            raise ValueError(
                "Feature extraction requires dataset.source_id or source_id"
            )

        x = dataset.X
        if max_samples is not None:
            limit = int(max_samples)
            if limit <= 0:
                raise ValueError(f"max_samples must be positive, got {max_samples}")
            x = x[:limit]

        if backend is None:
            with IQFeatureCtypesBackend(
                dll_dir=dll_dir,
                dll_path=dll_path,
                dependency_dirs=dependency_dirs,
            ) as owned_backend:
                features = extract_feature_matrix(
                    x,
                    resolved_sample_rate,
                    owned_backend,
                    progress_every,
                )
        else:
            features = extract_feature_matrix(
                x,
                resolved_sample_rate,
                backend,
                progress_every,
            )

        return FeatureResult(
            source_id=resolved_source_id,
            features=features,
            feature_names=self.feature_names,
            sample_rate=resolved_sample_rate,
            seq_len=dataset.seq_len,
            metadata={
                "feature_schema_id": FEATURE_SCHEMA_ID,
                "feature_map_path": self.feature_map["feature_map_path"],
                "input_num_samples": dataset.num_samples,
                "selected_num_samples": int(features.shape[0]),
            },
        )


__all__ = [
    "FeatureBackend",
    "FeatureExtractionService",
    "extract_feature_matrix",
    "resolve_sample_rate",
]
