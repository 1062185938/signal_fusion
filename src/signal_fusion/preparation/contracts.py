"""Contracts used by the optional raw-signal preparation path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


SampleReader = Callable[[int, int], np.ndarray]


def _non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(slots=True)
class RawSignal:
    """Lazy, random-access view of one continuous raw IQ recording."""

    source_id: str
    source_path: str
    sample_rate: float
    sample_count: int
    sample_format: str
    _sample_reader: SampleReader = field(repr=False, compare=False)
    center_frequency: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_id = _non_empty_text(self.source_id, "source_id")
        self.source_path = _non_empty_text(self.source_path, "source_path")
        self.sample_format = _non_empty_text(self.sample_format, "sample_format")
        self.sample_rate = float(self.sample_rate)
        if not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        self.sample_count = int(self.sample_count)
        if self.sample_count < 0:
            raise ValueError("sample_count must not be negative")
        if not callable(self._sample_reader):
            raise TypeError("_sample_reader must be callable")
        if self.center_frequency is not None:
            self.center_frequency = float(self.center_frequency)
            if not math.isfinite(self.center_frequency):
                raise ValueError("center_frequency must be finite or None")
        self.metadata = dict(self.metadata)

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    def read_samples(self, start_sample: int = 0, count: int | None = None) -> np.ndarray:
        """Read a bounded interval as one-dimensional ``complex64`` IQ."""

        start = int(start_sample)
        if start < 0 or start > self.sample_count:
            raise ValueError(
                f"start_sample must be within [0, {self.sample_count}], got {start}"
            )
        requested = self.sample_count - start if count is None else int(count)
        if requested < 0:
            raise ValueError(f"count must not be negative, got {requested}")
        actual_count = min(requested, self.sample_count - start)
        if actual_count == 0:
            return np.empty(0, dtype=np.complex64)

        samples = np.asarray(self._sample_reader(start, actual_count))
        if samples.ndim != 1 or samples.shape[0] != actual_count:
            raise ValueError(
                "Raw sample reader violated its contract: "
                f"expected ({actual_count},), got {samples.shape}"
            )
        return samples.astype(np.complex64, copy=False)


@dataclass(slots=True)
class SignalRegion:
    """Half-open signal interval ``[start_sample, end_sample)``."""

    start_sample: int
    end_sample: int
    detector: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.start_sample = int(self.start_sample)
        self.end_sample = int(self.end_sample)
        if self.start_sample < 0:
            raise ValueError("start_sample must not be negative")
        if self.end_sample <= self.start_sample:
            raise ValueError("end_sample must be greater than start_sample")
        self.detector = _non_empty_text(self.detector, "detector")
        if self.score is not None:
            self.score = float(self.score)
            if not math.isfinite(self.score):
                raise ValueError("score must be finite or None")
        self.metadata = dict(self.metadata)

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(slots=True)
class PreparationConfig:
    """Format-independent segmentation, windowing, and normalization settings."""

    seq_len: int
    hop_len: int | None = None
    remainder: str = "drop"
    normalization: str = "none"
    remove_dc: bool = False
    min_region_samples: int = 1
    merge_gap_samples: int = 0
    pad_before_samples: int = 0
    pad_after_samples: int = 0
    label: int | None = None
    class_name: str | None = None

    def __post_init__(self) -> None:
        self.seq_len = int(self.seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.hop_len = self.seq_len if self.hop_len is None else int(self.hop_len)
        if self.hop_len <= 0:
            raise ValueError("hop_len must be positive")
        self.remainder = str(self.remainder).lower()
        if self.remainder not in {"drop", "pad"}:
            raise ValueError("remainder must be 'drop' or 'pad'")
        self.normalization = str(self.normalization).lower()
        if self.normalization not in {"none", "rms", "peak"}:
            raise ValueError("normalization must be 'none', 'rms', or 'peak'")
        self.remove_dc = bool(self.remove_dc)

        self.min_region_samples = int(self.min_region_samples)
        self.merge_gap_samples = int(self.merge_gap_samples)
        self.pad_before_samples = int(self.pad_before_samples)
        self.pad_after_samples = int(self.pad_after_samples)
        if self.min_region_samples <= 0:
            raise ValueError("min_region_samples must be positive")
        if self.merge_gap_samples < 0:
            raise ValueError("merge_gap_samples must not be negative")
        if self.pad_before_samples < 0 or self.pad_after_samples < 0:
            raise ValueError("region padding must not be negative")

        if self.label is not None:
            self.label = int(self.label)
        if self.class_name is not None:
            self.class_name = _non_empty_text(self.class_name, "class_name")

