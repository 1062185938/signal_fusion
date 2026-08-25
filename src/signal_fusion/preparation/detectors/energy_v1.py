"""Versioned migration of the legacy LoRa hysteresis energy detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from signal_fusion.preparation.contracts import RawSignal, SignalRegion
from signal_fusion.preparation.detectors.base import SignalDetector


EPS = 1e-20


def milliseconds_to_samples(sample_rate: float, value_ms: float) -> int:
    return max(0, int(round(sample_rate * value_ms / 1000.0)))


def moving_average_power_v1(iq: np.ndarray, window_size: int) -> np.ndarray:
    if len(iq) == 0:
        return np.empty(0, dtype=np.float64)
    power = np.abs(iq).astype(np.float64) ** 2
    if window_size <= 1:
        return power
    if len(power) < window_size:
        return np.empty(0, dtype=np.float64)
    cumulative = np.empty(len(power) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(power, out=cumulative[1:])
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(
        window_size
    )


def _uniform_probe_starts(
    total_samples: int, probe_size: int, probe_count: int
) -> list[int]:
    if total_samples <= 0 or probe_size <= 0 or probe_count <= 0:
        return []
    max_start = max(0, total_samples - probe_size)
    if probe_count == 1 or max_start == 0:
        return [0]
    starts = np.linspace(0, max_start, num=probe_count)
    return sorted({int(round(value)) for value in starts})


@dataclass(slots=True)
class EnergyDetectorV1Config:
    """Legacy energy detector parameters expressed in the original units."""

    chunk_size: int = 1_000_000
    window_ms: float = 1.0
    start_threshold_db: float = 6.0
    end_threshold_db: float = 5.0
    min_signal_ms: float = 8.0
    min_gap_ms: float = 2.0
    pad_before_ms: float = 0.5
    pad_after_ms: float = 0.2
    release_windows: int = 2
    noise_percentile: float = 20.0
    noise_probe_count: int = 8
    ignore_initial_ms: float = 0.0
    window_power_ratio: float = 0.05

    def __post_init__(self) -> None:
        self.chunk_size = int(self.chunk_size)
        self.release_windows = int(self.release_windows)
        self.noise_probe_count = int(self.noise_probe_count)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.window_ms < 0:
            raise ValueError("window_ms must not be negative")
        if self.release_windows <= 0:
            raise ValueError("release_windows must be positive")
        if self.noise_probe_count <= 0:
            raise ValueError("noise_probe_count must be positive")
        if not 0.0 <= self.noise_percentile <= 100.0:
            raise ValueError("noise_percentile must be within [0, 100]")
        if self.ignore_initial_ms < 0:
            raise ValueError("ignore_initial_ms must not be negative")
        if self.window_power_ratio <= 0:
            raise ValueError("window_power_ratio must be positive")
        if self.min_signal_ms < 0 or self.min_gap_ms < 0:
            raise ValueError("minimum signal and gap durations must not be negative")
        if self.pad_before_ms < 0 or self.pad_after_ms < 0:
            raise ValueError("padding durations must not be negative")


@dataclass(frozen=True, slots=True)
class _SampleConfig:
    window_size: int
    stride: int
    start_threshold_linear: float
    end_threshold_linear: float
    release_windows: int
    min_signal_samples: int
    min_gap_samples: int
    pad_before_samples: int
    pad_after_samples: int


def _estimate_noise_floor(
    signal: RawSignal,
    *,
    window_size: int,
    chunk_size: int,
    noise_percentile: float,
    noise_probe_count: int,
    start_sample: int,
) -> tuple[float, float]:
    start_sample = min(max(0, int(start_sample)), signal.sample_count)
    available_samples = signal.sample_count - start_sample
    if available_samples == 0:
        return EPS, 10.0 * math.log10(EPS)

    probe_size = min(200_000, max(1, chunk_size), available_samples)
    starts = [
        start_sample + offset
        for offset in _uniform_probe_starts(
            available_samples, probe_size, noise_probe_count
        )
    ]
    smoothed_blocks: list[np.ndarray] = []
    for start in starts:
        iq = signal.read_samples(start, probe_size)
        smoothed = moving_average_power_v1(
            iq, min(window_size, max(1, len(iq)))
        )
        if len(smoothed) > 0:
            smoothed_blocks.append(smoothed)

    if smoothed_blocks:
        values = np.concatenate(smoothed_blocks)
        noise_floor = float(np.percentile(values, noise_percentile))
    else:
        iq = signal.read_samples(start_sample, min(available_samples, probe_size))
        if len(iq) == 0:
            noise_floor = EPS
        else:
            noise_floor = float(
                np.percentile(
                    np.abs(iq).astype(np.float64) ** 2, noise_percentile
                )
            )
    noise_floor = max(noise_floor, EPS)
    return noise_floor, 10.0 * math.log10(noise_floor)


def _detect_candidates(
    signal: RawSignal,
    config: _SampleConfig,
    *,
    chunk_size: int,
    start_sample: int,
) -> list[tuple[int, int]]:
    overlap_samples = max(config.window_size, 1)
    start_sample = min(max(0, int(start_sample)), signal.sample_count)
    next_window_start = start_sample
    in_signal = False
    current_region_start: int | None = None
    release_count = 0
    candidates: list[tuple[int, int]] = []

    for chunk_start in range(start_sample, signal.sample_count, chunk_size):
        read_start = max(start_sample, chunk_start - overlap_samples)
        read_end = min(signal.sample_count, chunk_start + chunk_size)
        iq = signal.read_samples(read_start, read_end - read_start)
        smoothed = moving_average_power_v1(iq, config.window_size)
        if len(smoothed) == 0:
            continue

        win_start = max(next_window_start, read_start)
        if win_start < read_start:
            offset_steps = math.ceil((read_start - win_start) / config.stride)
            win_start += offset_steps * config.stride

        while win_start + config.window_size <= read_end:
            local_start = win_start - read_start
            if local_start < 0 or local_start >= len(smoothed):
                break
            win_end = win_start + config.window_size
            if win_end > read_end:
                break
            average_power = float(smoothed[local_start])
            next_window_start = win_start + config.stride

            if not in_signal:
                if average_power >= config.start_threshold_linear:
                    in_signal = True
                    current_region_start = win_start
                    release_count = 0
                win_start = next_window_start
                continue

            if average_power < config.end_threshold_linear:
                release_count += 1
                if release_count >= config.release_windows:
                    candidates.append(
                        (int(current_region_start or 0), int(win_end))
                    )
                    in_signal = False
                    current_region_start = None
                    release_count = 0
            else:
                release_count = 0
            win_start = next_window_start

    if in_signal and current_region_start is not None:
        candidates.append((int(current_region_start), int(signal.sample_count)))
    return candidates


def _merge_and_filter(
    candidates: list[tuple[int, int]],
    *,
    min_gap_samples: int,
    min_signal_samples: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    sorted_candidates = sorted((int(s), int(e)) for s, e in candidates if e > s)
    if not sorted_candidates:
        return [], [], 0
    merged = [sorted_candidates[0]]
    for start, end in sorted_candidates[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end < min_gap_samples:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    valid = [
        (start, end)
        for start, end in merged
        if end - start >= min_signal_samples
    ]
    return merged, valid, len(merged) - len(valid)


def _apply_padding(
    regions: list[tuple[int, int]],
    *,
    total_samples: int,
    before: int,
    after: int,
) -> list[dict[str, int]]:
    bursts: list[dict[str, int]] = []
    for raw_start, raw_end in regions:
        start = max(0, raw_start - before)
        end = min(total_samples, raw_end + after)
        if end > start:
            bursts.append(
                {
                    "raw_start": raw_start,
                    "raw_end": raw_end,
                    "start": start,
                    "end": end,
                }
            )
    return bursts


def _refine_regions(
    signal: RawSignal,
    bursts: list[dict[str, int]],
    *,
    window_size: int,
    window_power_ratio: float,
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    refined_bursts: list[dict[str, int]] = []
    refinement_info: list[dict[str, Any]] = []
    window_size = max(1, int(window_size))

    for burst_id, burst in enumerate(bursts):
        burst_length = burst["end"] - burst["start"]
        iq = signal.read_samples(burst["start"], burst_length)
        if burst_length <= 0 or len(iq) == 0:
            refined_bursts.append(dict(burst))
            refinement_info.append(
                {
                    "burst_id": burst_id,
                    "original_start": burst["start"],
                    "original_end": burst["end"],
                    "refined_start": burst["start"],
                    "refined_end": burst["end"],
                    "pref_power": None,
                    "threshold_power": None,
                    "window_powers": [],
                    "valid_windows": [],
                    "valid_window_count": 0,
                    "window_count": 0,
                }
            )
            continue

        starts = list(range(0, len(iq), window_size))
        window_starts = np.asarray(
            [burst["start"] + start for start in starts], dtype=np.int64
        )
        window_ends = np.asarray(
            [
                burst["start"] + min(start + window_size, len(iq))
                for start in starts
            ],
            dtype=np.int64,
        )
        powers = np.asarray(
            [
                float(
                    np.mean(
                        np.abs(
                            iq[start : min(start + window_size, len(iq))]
                        ).astype(np.float64)
                        ** 2
                    )
                )
                for start in starts
            ],
            dtype=np.float64,
        )
        preferred_power = float(np.percentile(powers, 90))
        threshold_power = preferred_power * window_power_ratio
        valid = powers >= threshold_power
        refined = dict(burst)

        if not np.any(valid):
            valid_window_count = 0
        else:
            valid_indices = np.flatnonzero(valid)
            runs: list[tuple[int, int]] = []
            run_start = int(valid_indices[0])
            previous = int(valid_indices[0])
            for index in valid_indices[1:]:
                index = int(index)
                if index == previous + 1:
                    previous = index
                else:
                    runs.append((run_start, previous))
                    run_start = index
                    previous = index
            runs.append((run_start, previous))
            best_start, best_end = max(
                runs, key=lambda item: item[1] - item[0] + 1
            )
            refined_start = int(window_starts[best_start])
            refined_end = int(window_ends[best_end])
            if refined_end > refined_start:
                refined["start"] = refined_start
                refined["end"] = refined_end
            valid_window_count = int(np.sum(valid))

        refined_bursts.append(refined)
        refinement_info.append(
            {
                "burst_id": burst_id,
                "original_start": int(burst["start"]),
                "original_end": int(burst["end"]),
                "refined_start": int(refined["start"]),
                "refined_end": int(refined["end"]),
                "pref_power": preferred_power,
                "threshold_power": float(threshold_power),
                "window_powers": powers.astype(float).tolist(),
                "valid_windows": valid.astype(bool).tolist(),
                "valid_window_count": valid_window_count,
                "window_count": int(len(powers)),
            }
        )
    return refined_bursts, refinement_info


class EnergyDetectorV1(SignalDetector):
    """Exact versioned form of the legacy LoRa energy-region algorithm."""

    name = "energy_v1"

    def __init__(self, config: EnergyDetectorV1Config | None = None):
        self.config = config or EnergyDetectorV1Config()
        self.last_report: dict[str, Any] | None = None

    def detect(self, signal: RawSignal) -> list[SignalRegion]:
        sample_rate = signal.sample_rate
        window_size = max(
            1, milliseconds_to_samples(sample_rate, self.config.window_ms)
        )
        stride = max(1, window_size // 2)
        ignored_samples = min(
            signal.sample_count,
            int(sample_rate * self.config.ignore_initial_ms / 1000.0),
        )
        noise_floor_linear, noise_floor_db = _estimate_noise_floor(
            signal,
            window_size=window_size,
            chunk_size=self.config.chunk_size,
            noise_percentile=self.config.noise_percentile,
            noise_probe_count=self.config.noise_probe_count,
            start_sample=ignored_samples,
        )
        start_threshold_linear = noise_floor_linear * (
            10.0 ** (self.config.start_threshold_db / 10.0)
        )
        end_threshold_linear = noise_floor_linear * (
            10.0 ** (self.config.end_threshold_db / 10.0)
        )
        sample_config = _SampleConfig(
            window_size=window_size,
            stride=stride,
            start_threshold_linear=start_threshold_linear,
            end_threshold_linear=end_threshold_linear,
            release_windows=self.config.release_windows,
            min_signal_samples=milliseconds_to_samples(
                sample_rate, self.config.min_signal_ms
            ),
            min_gap_samples=milliseconds_to_samples(
                sample_rate, self.config.min_gap_ms
            ),
            pad_before_samples=milliseconds_to_samples(
                sample_rate, self.config.pad_before_ms
            ),
            pad_after_samples=milliseconds_to_samples(
                sample_rate, self.config.pad_after_ms
            ),
        )
        candidates = _detect_candidates(
            signal,
            sample_config,
            chunk_size=self.config.chunk_size,
            start_sample=ignored_samples,
        )
        merged, valid, skipped_short = _merge_and_filter(
            candidates,
            min_gap_samples=sample_config.min_gap_samples,
            min_signal_samples=sample_config.min_signal_samples,
        )
        padded = _apply_padding(
            valid,
            total_samples=signal.sample_count,
            before=sample_config.pad_before_samples,
            after=sample_config.pad_after_samples,
        )
        refined, refinement_info = _refine_regions(
            signal,
            padded,
            window_size=window_size,
            window_power_ratio=self.config.window_power_ratio,
        )

        regions = [
            SignalRegion(
                start_sample=burst["start"],
                end_sample=burst["end"],
                detector=self.name,
                metadata={
                    "raw_start_sample": burst["raw_start"],
                    "raw_end_sample": burst["raw_end"],
                    "pre_refinement_start_sample": padded[index]["start"],
                    "pre_refinement_end_sample": padded[index]["end"],
                    "refinement": refinement_info[index],
                },
            )
            for index, burst in enumerate(refined)
        ]
        self.last_report = {
            "detector": self.name,
            "config": asdict(self.config),
            "ignored_samples": ignored_samples,
            "noise_floor_linear": noise_floor_linear,
            "noise_floor_db": noise_floor_db,
            "window_size": window_size,
            "stride": stride,
            "start_threshold_linear": start_threshold_linear,
            "end_threshold_linear": end_threshold_linear,
            "sample_config": asdict(sample_config),
            "coarse_candidates": candidates,
            "merged_candidates": merged,
            "valid_candidates": valid,
            "skipped_short": skipped_short,
            "bursts_before_refinement": padded,
            "burst_regions": refined,
            "burst_refinement": refinement_info,
        }
        return regions


__all__ = [
    "EnergyDetectorV1",
    "EnergyDetectorV1Config",
    "milliseconds_to_samples",
    "moving_average_power_v1",
]

