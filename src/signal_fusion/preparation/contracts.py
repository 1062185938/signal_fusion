"""Contracts used by the optional raw-signal preparation path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any

import numpy as np


SampleReader = Callable[[int, int], np.ndarray]


RESAMPLING_PROFILE_V1 = "polyphase_kaiser5_v1"
RESAMPLING_MAX_FACTOR = 4_096


def _non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _positive_integer(value: int, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{field_name} must be positive")
    return integer


@dataclass(frozen=True, slots=True)
class ResamplingConfig:
    """Versioned target-rate contract for optional IQ resampling.

    The profile fixes the signal-processing choices used by a dataset version.
    Numerical limits are explicit so an invalid rate approximation or an
    unexpectedly large input or output buffer (including region guard context)
    fails instead of silently changing behavior.
    """

    target_sample_rate: float = 4_000_000.0
    profile: str = RESAMPLING_PROFILE_V1
    max_denominator: int = 1_000
    rate_tolerance_ppm: float = 0.1
    max_resampling_factor: int = RESAMPLING_MAX_FACTOR
    max_input_samples: int = 10_000_000
    max_output_samples: int = 10_000_000

    def __post_init__(self) -> None:
        target_sample_rate = float(self.target_sample_rate)
        if not math.isfinite(target_sample_rate) or target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be finite and positive")
        object.__setattr__(self, "target_sample_rate", target_sample_rate)

        profile = _non_empty_text(self.profile, "profile")
        if profile != RESAMPLING_PROFILE_V1:
            raise ValueError(
                f"unsupported resampling profile {profile!r}; "
                f"expected {RESAMPLING_PROFILE_V1!r}"
            )
        object.__setattr__(self, "profile", profile)

        max_denominator = _positive_integer(
            self.max_denominator, "max_denominator"
        )
        max_resampling_factor = _positive_integer(
            self.max_resampling_factor, "max_resampling_factor"
        )
        max_input_samples = _positive_integer(
            self.max_input_samples, "max_input_samples"
        )
        max_output_samples = _positive_integer(
            self.max_output_samples, "max_output_samples"
        )
        if max_resampling_factor > RESAMPLING_MAX_FACTOR:
            raise ValueError(
                "max_resampling_factor exceeds the safety limit for "
                f"{RESAMPLING_PROFILE_V1!r}: {RESAMPLING_MAX_FACTOR}"
            )
        object.__setattr__(self, "max_denominator", max_denominator)
        object.__setattr__(
            self, "max_resampling_factor", max_resampling_factor
        )
        object.__setattr__(self, "max_input_samples", max_input_samples)
        object.__setattr__(self, "max_output_samples", max_output_samples)

        rate_tolerance_ppm = float(self.rate_tolerance_ppm)
        if not math.isfinite(rate_tolerance_ppm) or rate_tolerance_ppm < 0:
            raise ValueError("rate_tolerance_ppm must be finite and non-negative")
        object.__setattr__(self, "rate_tolerance_ppm", rate_tolerance_ppm)


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
    """Format-independent segmentation, windowing, and normalization settings.

    With optional pipeline resampling, region filtering/padding fields remain
    in native source samples while ``seq_len`` and ``hop_len`` apply to the
    effective output-rate grid.
    """

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
