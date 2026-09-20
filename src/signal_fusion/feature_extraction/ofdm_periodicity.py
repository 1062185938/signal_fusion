"""Sample-rate-aware OFDM periodicity measurements.

The extractor measures physical repetition at caller-provided periods.  It does
not attach protocol labels or make classification decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


PERIODICITY_SCHEMA_ID = "ofdm_periodicity_v1"
DEFAULT_SEARCH_RADIUS_SAMPLES = 2


@dataclass(frozen=True, slots=True)
class PeriodicityCandidate:
    """One physical period to inspect, expressed independently of sample rate."""

    candidate_id: str
    period_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        period = float(self.period_seconds)
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("period_seconds must be finite and positive")
        object.__setattr__(self, "period_seconds", period)


@dataclass(slots=True)
class PeriodicityBatchResult:
    """Fixed-shape measurements for a batch of equally sized IQ regions."""

    candidates: tuple[PeriodicityCandidate, ...]
    sample_rate_hz: float
    sample_count: int
    search_radius_samples: int
    nominal_lags: np.ndarray
    resolvable: np.ndarray
    peak_lags: np.ndarray
    normalized_correlations: np.ndarray

    def __post_init__(self) -> None:
        candidate_count = len(self.candidates)
        if candidate_count == 0:
            raise ValueError("at least one periodicity candidate is required")
        if self.nominal_lags.shape != (candidate_count,):
            raise ValueError("nominal_lags must contain one value per candidate")
        if self.resolvable.shape != (candidate_count,):
            raise ValueError("resolvable must contain one value per candidate")
        if self.peak_lags.ndim != 2 or self.peak_lags.shape[1] != candidate_count:
            raise ValueError("peak_lags must have shape [N, candidate_count]")
        if self.normalized_correlations.shape != self.peak_lags.shape:
            raise ValueError(
                "normalized_correlations must have the same shape as peak_lags"
            )
        if self.peak_lags.dtype != np.int64:
            raise TypeError("peak_lags must have dtype int64")
        if self.normalized_correlations.dtype != np.float32:
            raise TypeError("normalized_correlations must have dtype float32")
        if not np.all(np.isfinite(self.normalized_correlations)):
            raise ValueError("normalized_correlations must be finite")

    @property
    def num_samples(self) -> int:
        return int(self.peak_lags.shape[0])

    def evidence_for_sample(self, sample_index: int) -> dict[str, Any]:
        """Return one JSON-friendly, protocol-neutral evidence record."""

        index = int(sample_index)
        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"sample_index={index} is outside [0, {self.num_samples - 1}]"
            )
        measurements: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(self.candidates):
            is_resolvable = bool(self.resolvable[candidate_index])
            peak_lag = int(self.peak_lags[index, candidate_index])
            measurements.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "nominal_period_seconds": candidate.period_seconds,
                    "nominal_lag_samples": float(
                        self.nominal_lags[candidate_index]
                    ),
                    "search_radius_samples": self.search_radius_samples,
                    "resolvable": is_resolvable,
                    "peak_lag_samples": peak_lag if is_resolvable else None,
                    "peak_period_seconds": (
                        peak_lag / self.sample_rate_hz if is_resolvable else None
                    ),
                    "normalized_correlation": (
                        float(
                            self.normalized_correlations[
                                index, candidate_index
                            ]
                        )
                        if is_resolvable
                        else None
                    ),
                }
            )
        return {
            "schema_id": PERIODICITY_SCHEMA_ID,
            "method": "energy_normalized_complex_autocorrelation",
            "scope": "complete_continuous_region",
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
            "candidates": measurements,
        }


def _validate_candidates(
    candidates: Sequence[PeriodicityCandidate],
) -> tuple[PeriodicityCandidate, ...]:
    values = tuple(candidates)
    if not values:
        raise ValueError("at least one periodicity candidate is required")
    if any(not isinstance(item, PeriodicityCandidate) for item in values):
        raise TypeError("candidates must contain PeriodicityCandidate values")
    candidate_ids = [item.candidate_id for item in values]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id values must be unique")
    return values


def _validate_sample_rate(sample_rate_hz: float) -> float:
    value = float(sample_rate_hz)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    return value


def measure_periodicity_batch(
    signals: np.ndarray,
    sample_rate_hz: float,
    candidates: Sequence[PeriodicityCandidate],
    *,
    search_radius_samples: int = DEFAULT_SEARCH_RADIUS_SAMPLES,
    batch_size: int = 256,
) -> PeriodicityBatchResult:
    """Measure normalized complex autocorrelation near candidate periods.

    ``signals`` must have shape ``[N, sample_count]`` and a complex dtype.  A
    candidate is marked unresolved when its complete search neighborhood does
    not fit inside the observed region.
    """

    values = np.asarray(signals)
    if values.ndim != 2:
        raise ValueError(f"signals must have shape [N, sample_count], got {values.shape}")
    if not np.iscomplexobj(values):
        raise TypeError("signals must use a complex dtype")
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("signals must contain at least one region of length >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("signals contain NaN or Inf")

    sample_rate = _validate_sample_rate(sample_rate_hz)
    candidate_values = _validate_candidates(candidates)
    if isinstance(search_radius_samples, bool):
        raise TypeError("search_radius_samples must be an integer")
    radius = int(search_radius_samples)
    if radius != search_radius_samples or radius < 0:
        raise ValueError("search_radius_samples must be a non-negative integer")
    if isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    chunk_size = int(batch_size)
    if chunk_size != batch_size or chunk_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    num_samples, sample_count = values.shape
    candidate_count = len(candidate_values)
    nominal_lags = np.asarray(
        [sample_rate * item.period_seconds for item in candidate_values],
        dtype=np.float64,
    )
    center_lags = np.rint(nominal_lags).astype(np.int64)
    resolvable = (
        (center_lags - radius >= 1)
        & (center_lags + radius < sample_count)
    )
    peak_lags = np.zeros((num_samples, candidate_count), dtype=np.int64)
    correlations = np.zeros((num_samples, candidate_count), dtype=np.float32)

    for start in range(0, num_samples, chunk_size):
        stop = min(start + chunk_size, num_samples)
        chunk = np.asarray(values[start:stop], dtype=np.complex64)
        centered = chunk - np.mean(chunk, axis=1, keepdims=True)

        for candidate_index, center_lag in enumerate(center_lags):
            if not resolvable[candidate_index]:
                continue
            best = np.full(stop - start, -1.0, dtype=np.float64)
            best_lag = np.zeros(stop - start, dtype=np.int64)
            for lag in range(int(center_lag) - radius, int(center_lag) + radius + 1):
                reference = centered[:, :-lag]
                delayed = centered[:, lag:]
                numerator = np.abs(
                    np.sum(np.conj(reference) * delayed, axis=1)
                ).astype(np.float64)
                denominator = np.sqrt(
                    np.sum(np.abs(reference) ** 2, axis=1).astype(np.float64)
                    * np.sum(np.abs(delayed) ** 2, axis=1).astype(np.float64)
                )
                current = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 0.0,
                )
                current = np.clip(current, 0.0, 1.0)
                replace = current > best
                best[replace] = current[replace]
                best_lag[replace] = lag
            correlations[start:stop, candidate_index] = best.astype(np.float32)
            peak_lags[start:stop, candidate_index] = best_lag

    return PeriodicityBatchResult(
        candidates=candidate_values,
        sample_rate_hz=sample_rate,
        sample_count=sample_count,
        search_radius_samples=radius,
        nominal_lags=nominal_lags,
        resolvable=resolvable.astype(bool),
        peak_lags=peak_lags,
        normalized_correlations=correlations,
    )


def measure_periodicity(
    signal: np.ndarray,
    sample_rate_hz: float,
    candidates: Sequence[PeriodicityCandidate],
    *,
    search_radius_samples: int = DEFAULT_SEARCH_RADIUS_SAMPLES,
) -> PeriodicityBatchResult:
    """Measure one complex IQ region and return a one-row batch result."""

    values = np.asarray(signal)
    if values.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got {values.shape}")
    return measure_periodicity_batch(
        values[np.newaxis, :],
        sample_rate_hz,
        candidates,
        search_radius_samples=search_radius_samples,
        batch_size=1,
    )


__all__ = [
    "DEFAULT_SEARCH_RADIUS_SAMPLES",
    "PERIODICITY_SCHEMA_ID",
    "PeriodicityBatchResult",
    "PeriodicityCandidate",
    "measure_periodicity",
    "measure_periodicity_batch",
]
