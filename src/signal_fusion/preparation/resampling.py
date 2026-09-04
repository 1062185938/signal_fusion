"""Versioned polyphase resampling for complete signal regions.

This module is intentionally independent from the preparation pipeline.  It
provides the V2-A numerical foundation without changing any legacy CLI or
dataset behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import math
from numbers import Integral
from typing import Any

import numpy as np
import scipy
from scipy.signal import firwin, resample_poly

from signal_fusion.preparation.contracts import (
    RESAMPLING_MAX_FACTOR,
    RESAMPLING_PROFILE_V1,
    RawSignal,
    ResamplingConfig,
    SignalRegion,
)


_TAPS_PER_PHASE = 10
_KAISER_BETA = 5.0
_PAD_TYPE = "constant"
_IMPLEMENTATION = "scipy.signal.resample_poly"


def _finite_positive_rate(value: float, field_name: str) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return rate


def _non_negative_integer(value: int, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{field_name} must not be negative")
    return integer


def _ceil_ratio(value: int, numerator: int, denominator: int) -> int:
    """Return ``ceil(value * numerator / denominator)`` using integers."""

    return (value * numerator + denominator - 1) // denominator


@dataclass(frozen=True, slots=True)
class ResamplingPlan:
    """Resolved rational rate conversion and its reproducibility metadata."""

    source_sample_rate: float
    target_sample_rate: float
    effective_sample_rate: float
    up: int
    down: int
    rate_error_ppm: float
    profile: str = RESAMPLING_PROFILE_V1
    max_input_samples: int = 10_000_000
    max_output_samples: int = 10_000_000

    def __post_init__(self) -> None:
        source_rate = _finite_positive_rate(
            self.source_sample_rate, "source_sample_rate"
        )
        target_rate = _finite_positive_rate(
            self.target_sample_rate, "target_sample_rate"
        )
        effective_rate = _finite_positive_rate(
            self.effective_sample_rate, "effective_sample_rate"
        )
        object.__setattr__(self, "source_sample_rate", source_rate)
        object.__setattr__(self, "target_sample_rate", target_rate)
        object.__setattr__(self, "effective_sample_rate", effective_rate)

        up = _non_negative_integer(self.up, "up")
        down = _non_negative_integer(self.down, "down")
        if up == 0 or down == 0:
            raise ValueError("up and down must be positive")
        if math.gcd(up, down) != 1:
            raise ValueError("up and down must be reduced to coprime factors")
        if max(up, down) > RESAMPLING_MAX_FACTOR:
            raise ValueError(
                "up/down exceeds the resampling profile safety limit: "
                f"{up}/{down}"
            )
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "down", down)

        expected_effective_rate = source_rate * up / down
        if not math.isclose(
            effective_rate,
            expected_effective_rate,
            rel_tol=1e-12,
            abs_tol=max(1e-12, abs(expected_effective_rate) * 1e-12),
        ):
            raise ValueError(
                "effective_sample_rate does not match source_sample_rate * up/down"
            )

        rate_error_ppm = float(self.rate_error_ppm)
        if not math.isfinite(rate_error_ppm) or rate_error_ppm < 0:
            raise ValueError("rate_error_ppm must be finite and non-negative")
        expected_error_ppm = (
            abs(effective_rate - target_rate) / target_rate * 1_000_000.0
        )
        if not math.isclose(
            rate_error_ppm,
            expected_error_ppm,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "rate_error_ppm does not match effective and target sample rates"
            )
        object.__setattr__(self, "rate_error_ppm", rate_error_ppm)

        if self.profile != RESAMPLING_PROFILE_V1:
            raise ValueError(f"unsupported resampling profile {self.profile!r}")
        max_input_samples = _non_negative_integer(
            self.max_input_samples, "max_input_samples"
        )
        max_output_samples = _non_negative_integer(
            self.max_output_samples, "max_output_samples"
        )
        if max_input_samples == 0 or max_output_samples == 0:
            raise ValueError("sample buffer limits must be positive")
        object.__setattr__(self, "max_input_samples", max_input_samples)
        object.__setattr__(self, "max_output_samples", max_output_samples)

    @property
    def is_identity(self) -> bool:
        return self.up == 1 and self.down == 1

    @property
    def method(self) -> str:
        return "identity" if self.is_identity else "polyphase"

    @property
    def filter_half_length(self) -> int:
        if self.is_identity:
            return 0
        return _TAPS_PER_PHASE * max(self.up, self.down)

    @property
    def filter_tap_count(self) -> int:
        return 1 if self.is_identity else 2 * self.filter_half_length + 1

    @property
    def guard_source_samples(self) -> int:
        """Native-rate context required on each side of a region."""

        if self.is_identity:
            return 0
        return _ceil_ratio(self.filter_half_length, 1, self.up)

    def output_length(self, input_sample_count: int) -> int:
        count = _non_negative_integer(input_sample_count, "input_sample_count")
        return _ceil_ratio(count, self.up, self.down)

    def source_boundary_to_target(self, source_sample: int) -> int:
        """Map a native boundary onto the effective output-rate grid."""

        sample = _non_negative_integer(source_sample, "source_sample")
        return _ceil_ratio(sample, self.up, self.down)

    def target_interval_to_source(
        self, target_start_sample: int, target_end_sample: int
    ) -> tuple[int, int]:
        """Return a conservative native enclosure of a target interval."""

        start = _non_negative_integer(target_start_sample, "target_start_sample")
        end = _non_negative_integer(target_end_sample, "target_end_sample")
        if end < start:
            raise ValueError("target_end_sample must not precede target_start_sample")
        source_start = start * self.down // self.up
        if end == start:
            return source_start, source_start
        source_end = _ceil_ratio(end, self.down, self.up)
        return source_start, source_end

    def to_metadata(self) -> dict[str, Any]:
        """Return stable scalar provenance for a prepared dataset."""

        return {
            "source_sample_rate": self.source_sample_rate,
            "target_sample_rate": self.target_sample_rate,
            "effective_sample_rate": self.effective_sample_rate,
            "resampling_profile": self.profile,
            "resampling_implementation": _IMPLEMENTATION,
            "resampling_library": "scipy",
            "resampling_library_version": scipy.__version__,
            "resampling_method": self.method,
            "resampling_applied": not self.is_identity,
            "resample_up": self.up,
            "resample_down": self.down,
            "resampling_rate_error_ppm": self.rate_error_ppm,
            "resampling_filter_tap_count": self.filter_tap_count,
            "resampling_taps_per_phase": _TAPS_PER_PHASE,
            "resampling_kaiser_beta": _KAISER_BETA,
            "resampling_pad_type": _PAD_TYPE,
            "resampling_guard_source_samples": self.guard_source_samples,
            "resampling_max_input_samples": self.max_input_samples,
            "resampling_max_output_samples": self.max_output_samples,
        }


@dataclass(frozen=True, slots=True)
class ResampledRegion:
    """One resampled region with native and effective-grid coordinates.

    A non-empty native region may contain no sample on a lower-rate output
    grid.  In that case the target interval and ``samples`` are both empty.
    """

    samples: np.ndarray = field(repr=False, compare=False)
    plan: ResamplingPlan
    source_region_start_sample: int
    source_region_end_sample: int
    source_read_start_sample: int
    source_read_end_sample: int
    target_region_start_sample: int
    target_region_end_sample: int
    target_read_start_sample: int
    target_read_end_sample: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1 or not np.iscomplexobj(samples):
            raise ValueError("samples must be one-dimensional complex IQ")
        samples = np.array(samples, dtype=np.complex64, copy=True, order="C")
        object.__setattr__(self, "samples", samples)

        coordinate_names = (
            "source_region_start_sample",
            "source_region_end_sample",
            "source_read_start_sample",
            "source_read_end_sample",
            "target_region_start_sample",
            "target_region_end_sample",
            "target_read_start_sample",
            "target_read_end_sample",
        )
        for name in coordinate_names:
            object.__setattr__(self, name, _non_negative_integer(getattr(self, name), name))

        if self.source_region_end_sample <= self.source_region_start_sample:
            raise ValueError("source region must be non-empty")
        if self.source_read_end_sample <= self.source_read_start_sample:
            raise ValueError("source read interval must be non-empty")
        if not (
            self.source_read_start_sample <= self.source_region_start_sample
            and self.source_region_end_sample <= self.source_read_end_sample
        ):
            raise ValueError("source read interval must contain the source region")
        if self.source_read_start_sample % self.plan.down != 0:
            raise ValueError(
                "source_read_start_sample must align to the plan down factor"
            )

        expected_target_region_start = self.plan.source_boundary_to_target(
            self.source_region_start_sample
        )
        expected_target_region_end = self.plan.source_boundary_to_target(
            self.source_region_end_sample
        )
        if (
            self.target_region_start_sample != expected_target_region_start
            or self.target_region_end_sample != expected_target_region_end
        ):
            raise ValueError("target region coordinates do not match the plan")
        expected_target_read_start = self.plan.source_boundary_to_target(
            self.source_read_start_sample
        )
        expected_target_read_end = self.plan.source_boundary_to_target(
            self.source_read_end_sample
        )
        if (
            self.target_read_start_sample != expected_target_read_start
            or self.target_read_end_sample != expected_target_read_end
        ):
            raise ValueError("target read coordinates do not match the plan")
        if not (
            self.target_read_start_sample <= self.target_region_start_sample
            and self.target_region_end_sample <= self.target_read_end_sample
        ):
            raise ValueError("target read interval must contain the target region")
        if (
            self.target_region_end_sample - self.target_region_start_sample
            != samples.size
        ):
            raise ValueError("sample count does not match target region coordinates")

    @property
    def sample_count(self) -> int:
        return int(self.samples.size)

    def to_metadata(self) -> dict[str, Any]:
        metadata = self.plan.to_metadata()
        metadata.update(
            {
                "source_region_start_sample": self.source_region_start_sample,
                "source_region_end_sample": self.source_region_end_sample,
                "source_read_start_sample": self.source_read_start_sample,
                "source_read_end_sample": self.source_read_end_sample,
                "target_region_start_sample": self.target_region_start_sample,
                "target_region_end_sample": self.target_region_end_sample,
                "target_read_start_sample": self.target_read_start_sample,
                "target_read_end_sample": self.target_read_end_sample,
            }
        )
        return metadata


def build_resampling_plan(
    source_sample_rate: float,
    config: ResamplingConfig,
) -> ResamplingPlan:
    """Resolve an accurate, bounded rational conversion plan."""

    source_rate = _finite_positive_rate(source_sample_rate, "source_sample_rate")
    exact_ratio = Fraction(str(config.target_sample_rate)) / Fraction(str(source_rate))
    ratio = exact_ratio.limit_denominator(config.max_denominator)
    up = ratio.numerator
    down = ratio.denominator
    if max(up, down) > config.max_resampling_factor:
        raise ValueError(
            "resolved resampling factor exceeds max_resampling_factor: "
            f"{up}/{down}"
        )

    effective_rate = source_rate * up / down
    error_ppm = (
        abs(effective_rate - config.target_sample_rate)
        / config.target_sample_rate
        * 1_000_000.0
    )
    if error_ppm > config.rate_tolerance_ppm:
        raise ValueError(
            "target sample rate cannot be represented within tolerance: "
            f"resolved {up}/{down}, error={error_ppm:.9g} ppm, "
            f"tolerance={config.rate_tolerance_ppm:.9g} ppm"
        )

    return ResamplingPlan(
        source_sample_rate=source_rate,
        target_sample_rate=config.target_sample_rate,
        effective_sample_rate=effective_rate,
        up=up,
        down=down,
        rate_error_ppm=error_ppm,
        profile=config.profile,
        max_input_samples=config.max_input_samples,
        max_output_samples=config.max_output_samples,
    )


@lru_cache(maxsize=32)
def _filter_taps(profile: str, up: int, down: int) -> np.ndarray:
    if profile != RESAMPLING_PROFILE_V1:
        raise ValueError(f"unsupported resampling profile {profile!r}")
    if up == 1 and down == 1:
        taps = np.ones(1, dtype=np.float32)
    else:
        max_rate = max(up, down)
        half_length = _TAPS_PER_PHASE * max_rate
        taps = firwin(
            2 * half_length + 1,
            1.0 / max_rate,
            window=("kaiser", _KAISER_BETA),
        ).astype(np.float32)
    taps.setflags(write=False)
    return taps


def resample_iq(samples: np.ndarray, *, plan: ResamplingPlan) -> np.ndarray:
    """Resample one complete complex-IQ array according to ``plan``."""

    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(f"samples must be one-dimensional, got shape={values.shape}")
    if not np.iscomplexobj(values):
        raise TypeError("samples must contain complex IQ values")
    if values.size > plan.max_input_samples:
        raise ValueError(
            "resampling input would exceed max_input_samples: "
            f"{values.size} > {plan.max_input_samples}"
        )
    expected_count = plan.output_length(values.size)
    if expected_count > plan.max_output_samples:
        raise ValueError(
            "resampled output would exceed max_output_samples: "
            f"{expected_count} > {plan.max_output_samples}"
        )
    if values.size == 0:
        return np.empty(0, dtype=np.complex64)

    complex_values = values.astype(np.complex64, copy=False)
    if plan.is_identity:
        return np.array(complex_values, dtype=np.complex64, copy=True, order="C")

    output = resample_poly(
        complex_values,
        plan.up,
        plan.down,
        window=_filter_taps(plan.profile, plan.up, plan.down),
        padtype=_PAD_TYPE,
        cval=0.0,
    )
    if output.shape != (expected_count,):
        raise RuntimeError(
            "polyphase resampler returned an unexpected shape: "
            f"expected {(expected_count,)}, got {output.shape}"
        )
    return np.array(output, dtype=np.complex64, copy=True, order="C")


def resample_region(
    signal: RawSignal,
    region: SignalRegion,
    *,
    plan: ResamplingPlan,
) -> ResampledRegion:
    """Read and resample a native-rate region with sufficient FIR context.

    The read start is aligned to the reduced downsampling factor.  This keeps
    independently processed regions on the same global target sampling grid.
    """

    if region.end_sample > signal.sample_count:
        raise ValueError(
            "region end exceeds recording sample count: "
            f"{region.end_sample} > {signal.sample_count}"
        )
    if not math.isclose(
        signal.sample_rate,
        plan.source_sample_rate,
        rel_tol=1e-12,
        abs_tol=max(1e-12, abs(plan.source_sample_rate) * 1e-12),
    ):
        raise ValueError(
            "plan source_sample_rate does not match RawSignal.sample_rate"
        )

    target_start = plan.source_boundary_to_target(region.start_sample)
    target_end = plan.source_boundary_to_target(region.end_sample)
    if target_end - target_start > plan.max_output_samples:
        raise ValueError(
            "resampled region would exceed max_output_samples: "
            f"{target_end - target_start} > {plan.max_output_samples}"
        )

    if plan.is_identity:
        read_start = region.start_sample
        read_end = region.end_sample
    else:
        guard = plan.guard_source_samples
        read_start = max(0, region.start_sample - guard)
        read_start -= read_start % plan.down
        read_end = min(signal.sample_count, region.end_sample + guard)

    context_output_count = plan.output_length(read_end - read_start)
    if read_end - read_start > plan.max_input_samples:
        raise ValueError(
            "guarded native buffer would exceed max_input_samples: "
            f"{read_end - read_start} > {plan.max_input_samples}"
        )
    if context_output_count > plan.max_output_samples:
        raise ValueError(
            "guarded resampling buffer would exceed max_output_samples: "
            f"{context_output_count} > {plan.max_output_samples}"
        )
    native_samples = signal.read_samples(read_start, read_end - read_start)
    resampled_context = resample_iq(native_samples, plan=plan)
    target_read_start = plan.source_boundary_to_target(read_start)
    target_read_end = target_read_start + resampled_context.size

    expected_target_read_end = plan.source_boundary_to_target(read_end)
    if target_read_end != expected_target_read_end:
        raise RuntimeError(
            "resampled context is not aligned to the global target grid"
        )
    crop_start = target_start - target_read_start
    crop_end = target_end - target_read_start
    if crop_start < 0 or crop_end > resampled_context.size:
        raise RuntimeError("resampled region crop falls outside its context")

    return ResampledRegion(
        samples=resampled_context[crop_start:crop_end],
        plan=plan,
        source_region_start_sample=region.start_sample,
        source_region_end_sample=region.end_sample,
        source_read_start_sample=read_start,
        source_read_end_sample=read_end,
        target_region_start_sample=target_start,
        target_region_end_sample=target_end,
        target_read_start_sample=target_read_start,
        target_read_end_sample=target_read_end,
    )


__all__ = [
    "RESAMPLING_PROFILE_V1",
    "ResampledRegion",
    "ResamplingConfig",
    "ResamplingPlan",
    "build_resampling_plan",
    "resample_iq",
    "resample_region",
]
