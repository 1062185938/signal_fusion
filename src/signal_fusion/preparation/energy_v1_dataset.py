"""Dataset production path that preserves legacy energy-v1 behavior."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.io.writers import write_dataset_summary, write_prepared_dataset
from signal_fusion.preparation.contracts import RawSignal, SignalRegion
from signal_fusion.preparation.detectors.energy_v1 import (
    EPS,
    EnergyDetectorV1,
    EnergyDetectorV1Config,
)
from signal_fusion.preparation.readers import open_raw_signal


_INT_METADATA_FIELDS = (
    "burst_id",
    "window_id",
    "window_start_sample",
    "window_end_sample",
    "burst_start_sample",
    "burst_end_sample",
    "raw_burst_start_sample",
    "raw_burst_end_sample",
)
_FLOAT_METADATA_FIELDS = (
    "burst_rms",
    "burst_mean_power",
    "burst_peak_power",
    "normalization_scale",
)


@dataclass(frozen=True, slots=True)
class _BurstStats:
    burst_rms: float
    burst_mean_power: float
    burst_peak_power: float
    normalization_scale: float


def _preprocess_burst(
    iq: np.ndarray, *, remove_dc: bool, normalize: str
) -> tuple[np.ndarray, _BurstStats]:
    if remove_dc and len(iq) > 0:
        iq = iq - np.mean(iq)
    power = np.abs(iq).astype(np.float64) ** 2
    mean_power = float(np.mean(power)) if len(power) else 0.0
    peak_power = float(np.max(power)) if len(power) else 0.0
    rms = float(math.sqrt(max(mean_power, 0.0)))
    if normalize == "rms":
        scale = max(rms, EPS)
        iq = iq / scale
    elif normalize == "none":
        scale = 1.0
    else:
        raise ValueError("normalize must be 'rms' or 'none'")
    return iq.astype(np.complex64, copy=False), _BurstStats(
        burst_rms=rms,
        burst_mean_power=mean_power,
        burst_peak_power=peak_power,
        normalization_scale=scale,
    )


def _window_starts(
    sample_count: int, *, seq_len: int, hop_len: int, remainder: str
) -> list[int]:
    starts: list[int] = []
    if sample_count >= seq_len:
        starts = list(range(0, sample_count - seq_len + 1, hop_len))
        tail_start = starts[-1] + hop_len if starts else 0
        if remainder == "zero_pad" and tail_start < sample_count:
            starts.append(tail_start)
    elif remainder == "zero_pad" and sample_count > 0:
        starts = [0]
    return starts


def prepare_energy_v1(
    signal: RawSignal,
    *,
    detector: EnergyDetectorV1,
    seq_len: int = 128,
    hop_len: int = 128,
    remove_dc: bool = True,
    normalize: str = "rms",
    remainder: str = "drop",
    label: int = 0,
    class_name: str = "LoRa",
) -> PreparedDataset:
    """Run energy-v1 and retain its burst-level preprocessing semantics."""

    seq_len = int(seq_len)
    hop_len = int(hop_len)
    if seq_len <= 0 or hop_len <= 0:
        raise ValueError("seq_len and hop_len must be positive")
    if remainder not in {"drop", "zero_pad"}:
        raise ValueError("remainder must be 'drop' or 'zero_pad'")
    if normalize not in {"rms", "none"}:
        raise ValueError("normalize must be 'rms' or 'none'")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be a non-empty string")

    regions = detector.detect(signal)
    all_windows: list[np.ndarray] = []
    metadata_lists: dict[str, list[Any]] = {
        field: [] for field in _INT_METADATA_FIELDS + _FLOAT_METADATA_FIELDS
    }

    for burst_id, region in enumerate(regions):
        iq = signal.read_samples(region.start_sample, region.sample_count)
        processed, stats = _preprocess_burst(
            iq, remove_dc=remove_dc, normalize=normalize
        )
        raw_start = int(region.metadata["raw_start_sample"])
        raw_end = int(region.metadata["raw_end_sample"])
        for window_id, offset in enumerate(
            _window_starts(
                len(processed),
                seq_len=seq_len,
                hop_len=hop_len,
                remainder=remainder,
            )
        ):
            segment = processed[offset : offset + seq_len]
            actual_end = min(
                region.start_sample + offset + len(segment), region.end_sample
            )
            if len(segment) < seq_len:
                if remainder == "drop":
                    continue
                padded = np.zeros(seq_len, dtype=np.complex64)
                padded[: len(segment)] = segment
                segment = padded
            window = np.stack([segment.real, segment.imag], axis=0).astype(
                np.float32, copy=False
            )
            all_windows.append(window)
            metadata_lists["burst_id"].append(burst_id)
            metadata_lists["window_id"].append(window_id)
            metadata_lists["window_start_sample"].append(
                int(region.start_sample + offset)
            )
            metadata_lists["window_end_sample"].append(int(actual_end))
            metadata_lists["burst_start_sample"].append(region.start_sample)
            metadata_lists["burst_end_sample"].append(region.end_sample)
            metadata_lists["raw_burst_start_sample"].append(raw_start)
            metadata_lists["raw_burst_end_sample"].append(raw_end)
            metadata_lists["burst_rms"].append(stats.burst_rms)
            metadata_lists["burst_mean_power"].append(stats.burst_mean_power)
            metadata_lists["burst_peak_power"].append(stats.burst_peak_power)
            metadata_lists["normalization_scale"].append(
                stats.normalization_scale
            )

    if all_windows:
        x = np.stack(all_windows, axis=0).astype(np.float32, copy=False)
    else:
        x = np.empty((0, 2, seq_len), dtype=np.float32)
    y = np.full(x.shape[0], int(label), dtype=np.int64)
    metadata_arrays = {
        field: np.asarray(metadata_lists[field], dtype=np.int64)
        for field in _INT_METADATA_FIELDS
    }
    metadata_arrays.update(
        {
            field: np.asarray(metadata_lists[field], dtype=np.float64)
            for field in _FLOAT_METADATA_FIELDS
        }
    )
    report = detector.last_report
    if report is None:
        raise RuntimeError("EnergyDetectorV1 did not produce a detection report")
    meta: dict[str, Any] = {
        **metadata_arrays,
        "sample_rate": float(signal.sample_rate),
        "center_frequency": signal.center_frequency,
        "seq_len": seq_len,
        "hop_len": hop_len,
        "label": int(label),
        "class_name": class_name,
        "normalization_mode": normalize,
        "remove_dc": bool(remove_dc),
        "remainder": remainder,
        "energy_v1_report": report,
    }
    return PreparedDataset(X=x, y=y, meta=meta, source_id=signal.source_id)


def _save_dataset_npz(
    dataset: PreparedDataset,
    *,
    output_path: str | Path,
    signal: RawSignal,
    source_data_path: str,
    source_meta_path: str | None,
    include_source_id: bool,
) -> None:
    extra_fields: dict[str, Any] = {
        "sample_rate": np.asarray(signal.sample_rate, dtype=np.float64),
        "center_frequency": np.asarray(
            np.nan
            if signal.center_frequency is None
            else float(signal.center_frequency),
            dtype=np.float64,
        ),
        "seq_len": np.asarray(dataset.meta["seq_len"], dtype=np.int64),
        "hop_len": np.asarray(dataset.meta["hop_len"], dtype=np.int64),
        "label": np.asarray(dataset.meta["label"], dtype=np.int64),
        "class_name": np.asarray(dataset.meta["class_name"]),
        "source_data_path": np.asarray(source_data_path),
    }
    if source_meta_path is not None:
        extra_fields["source_meta_path"] = np.asarray(source_meta_path)
    write_prepared_dataset(
        dataset,
        output_path,
        metadata_fields=_INT_METADATA_FIELDS + _FLOAT_METADATA_FIELDS,
        extra_fields=extra_fields,
        include_source_id=include_source_id,
    )


def _build_summary(
    dataset: PreparedDataset,
    *,
    signal: RawSignal,
    output_path: str,
    source_data_path: str,
    source_meta_path: str | None,
    include_source_id: bool,
) -> dict[str, Any]:
    report = dataset.meta["energy_v1_report"]
    detector_config = report["config"]
    summary: dict[str, Any] = {
        "source_data_path": source_data_path,
        "source_meta_path": source_meta_path,
        "sigmf_datatype": signal.metadata.get("sigmf_datatype"),
        "integer_scaled": signal.metadata.get("integer_scaled", False),
        "integer_scale": signal.metadata.get("integer_scale"),
        "sample_rate": signal.sample_rate,
        "center_frequency": signal.center_frequency,
        "description": signal.metadata.get("description"),
        "datetime": signal.metadata.get("datetime"),
        "total_raw_samples": signal.sample_count,
        "recording_duration": signal.duration_seconds,
        "ignore_initial_ms": detector_config["ignore_initial_ms"],
        "ignored_samples": report["ignored_samples"],
        "noise_floor_linear": report["noise_floor_linear"],
        "noise_floor_db": report["noise_floor_db"],
        "start_threshold_db": detector_config["start_threshold_db"],
        "end_threshold_db": detector_config["end_threshold_db"],
        "start_threshold_linear": report["start_threshold_linear"],
        "end_threshold_linear": report["end_threshold_linear"],
        "window_ms": detector_config["window_ms"],
        "window_size": report["window_size"],
        "window_power_ratio": detector_config["window_power_ratio"],
        "release_windows": detector_config["release_windows"],
        "min_signal_ms": detector_config["min_signal_ms"],
        "min_gap_ms": detector_config["min_gap_ms"],
        "pad_before_ms": detector_config["pad_before_ms"],
        "pad_after_ms": detector_config["pad_after_ms"],
        "number_of_coarse_candidates": len(report["coarse_candidates"]),
        "number_of_merged_candidates": len(report["merged_candidates"]),
        "number_of_valid_bursts": len(report["burst_regions"]),
        "number_of_bursts_before_refinement": len(
            report["bursts_before_refinement"]
        ),
        "number_of_skipped_short_bursts": report["skipped_short"],
        "number_of_generated_windows": dataset.num_samples,
        "seq_len": dataset.seq_len,
        "hop_len": dataset.meta["hop_len"],
        "normalization_mode": dataset.meta["normalization_mode"],
        "remove_dc": dataset.meta["remove_dc"],
        "remainder": dataset.meta["remainder"],
        "label": dataset.meta["label"],
        "class_name": dataset.meta["class_name"],
        "output_path": output_path,
        "output_shape": list(dataset.X.shape),
        "output_dtype": str(dataset.X.dtype),
        "burst_regions": report["burst_regions"],
        "burst_refinement": report["burst_refinement"],
    }
    if include_source_id:
        summary["source_id"] = dataset.source_id
        summary["source_format"] = signal.metadata.get("data_format")
        summary["detector"] = "energy_v1"
    return summary


def build_energy_v1_dataset(
    input_path: str,
    output_path: str,
    *,
    source_id: str,
    data_format: str = "auto",
    reader_options: dict[str, Any] | None = None,
    detector_config: EnergyDetectorV1Config | None = None,
    detector: EnergyDetectorV1 | None = None,
    label: int = 0,
    class_name: str = "LoRa",
    seq_len: int = 128,
    hop_len: int = 128,
    remove_dc: bool = True,
    normalize: str = "rms",
    remainder: str = "drop",
    include_source_id: bool = True,
) -> dict[str, Any]:
    """Prepare and save a dataset through the format-independent energy-v1 path."""

    signal = open_raw_signal(
        input_path,
        source_id=source_id,
        data_format=data_format,
        **dict(reader_options or {}),
    )
    if detector is not None and detector_config is not None:
        raise ValueError("Pass detector or detector_config, not both")
    detector = detector or EnergyDetectorV1(detector_config)
    dataset = prepare_energy_v1(
        signal,
        detector=detector,
        seq_len=seq_len,
        hop_len=hop_len,
        remove_dc=remove_dc,
        normalize=normalize,
        remainder=remainder,
        label=label,
        class_name=class_name,
    )
    source_meta_path = signal.metadata.get("metadata_path")
    _save_dataset_npz(
        dataset,
        output_path=output_path,
        signal=signal,
        source_data_path=input_path,
        source_meta_path=source_meta_path,
        include_source_id=include_source_id,
    )
    summary = _build_summary(
        dataset,
        signal=signal,
        output_path=output_path,
        source_data_path=input_path,
        source_meta_path=source_meta_path,
        include_source_id=include_source_id,
    )
    summary_path = write_dataset_summary(summary, output_path)
    if dataset.num_samples == 0:
        print("No fixed-length IQ windows were generated. Empty NPZ was saved.")
    return {
        "output_path": output_path,
        "num_bursts": len(detector.last_report["burst_regions"]),
        "num_samples": dataset.num_samples,
        "x_shape": tuple(dataset.X.shape),
        "label": int(label),
        "class_name": class_name,
        "sample_rate": signal.sample_rate,
        "noise_floor_db": detector.last_report["noise_floor_db"],
        "summary_path": str(summary_path),
    }


def build_sigmf_dataset(
    data_path: str,
    meta_path: str,
    output_path: str,
    label: int = 0,
    class_name: str = "LoRa",
    seq_len: int = 128,
    hop_len: int = 128,
    chunk_size: int = 1_000_000,
    window_ms: float = 1.0,
    start_threshold_db: float = 6.0,
    end_threshold_db: float = 3.0,
    min_signal_ms: float = 5.0,
    min_gap_ms: float = 2.0,
    pad_before_ms: float = 0.5,
    pad_after_ms: float = 0.2,
    remove_dc: bool = True,
    normalize: str = "rms",
    remainder: str = "drop",
    release_windows: int = 2,
    noise_percentile: float = 20.0,
    noise_probe_count: int = 8,
    ignore_initial_ms: float = 0.0,
    window_power_ratio: float = 0.05,
) -> dict[str, Any]:
    """Compatibility API matching the legacy SigMF builder signature."""

    return build_energy_v1_dataset(
        input_path=data_path,
        output_path=output_path,
        source_id=Path(data_path).stem,
        data_format="sigmf",
        reader_options={"metadata_path": meta_path},
        detector_config=EnergyDetectorV1Config(
            chunk_size=chunk_size,
            window_ms=window_ms,
            start_threshold_db=start_threshold_db,
            end_threshold_db=end_threshold_db,
            min_signal_ms=min_signal_ms,
            min_gap_ms=min_gap_ms,
            pad_before_ms=pad_before_ms,
            pad_after_ms=pad_after_ms,
            release_windows=release_windows,
            noise_percentile=noise_percentile,
            noise_probe_count=noise_probe_count,
            ignore_initial_ms=ignore_initial_ms,
            window_power_ratio=window_power_ratio,
        ),
        label=label,
        class_name=class_name,
        seq_len=seq_len,
        hop_len=hop_len,
        remove_dc=remove_dc,
        normalize=normalize,
        remainder=remainder,
        include_source_id=False,
    )


__all__ = [
    "build_energy_v1_dataset",
    "build_sigmf_dataset",
    "prepare_energy_v1",
]
