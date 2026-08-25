"""Composition pipeline from raw recording to PreparedDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.preparation.contracts import PreparationConfig, RawSignal
from signal_fusion.preparation.detectors.base import SignalDetector
from signal_fusion.preparation.normalization import normalize_iq
from signal_fusion.preparation.readers import open_raw_signal
from signal_fusion.preparation.segmentation import segment_regions
from signal_fusion.preparation.windowing import window_spans


def prepare_signal(
    signal: RawSignal,
    *,
    detector: SignalDetector,
    config: PreparationConfig,
) -> PreparedDataset:
    """Detect, segment, window, and normalize one raw signal recording."""

    detected_regions = detector.detect(signal)
    regions = segment_regions(
        detected_regions,
        recording_sample_count=signal.sample_count,
        config=config,
    )

    windows: list[np.ndarray] = []
    region_ids: list[int] = []
    window_ids: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    valid_counts: list[int] = []
    region_starts: list[int] = []
    region_ends: list[int] = []
    normalization_scales: list[float] = []
    dc_offsets_i: list[float] = []
    dc_offsets_q: list[float] = []

    for region_id, region in enumerate(regions):
        for window_id, span in enumerate(window_spans(region, config)):
            raw_window = signal.read_samples(span.start_sample, span.valid_samples)
            normalized = normalize_iq(
                raw_window,
                mode=config.normalization,
                remove_dc=config.remove_dc,
            )
            padded = np.zeros(config.seq_len, dtype=np.complex64)
            padded[: span.valid_samples] = normalized.samples
            windows.append(padded)
            region_ids.append(region_id)
            window_ids.append(window_id)
            starts.append(span.start_sample)
            ends.append(span.end_sample)
            valid_counts.append(span.valid_samples)
            region_starts.append(region.start_sample)
            region_ends.append(region.end_sample)
            normalization_scales.append(normalized.scale)
            dc_offsets_i.append(float(normalized.dc_offset.real))
            dc_offsets_q.append(float(normalized.dc_offset.imag))

    x = np.empty((len(windows), 2, config.seq_len), dtype=np.float32)
    for index, window in enumerate(windows):
        x[index, 0, :] = window.real
        x[index, 1, :] = window.imag
    y = None
    if config.label is not None:
        y = np.full(len(windows), config.label, dtype=np.int64)

    meta: dict[str, Any] = {
        "source_id": signal.source_id,
        "source_path": signal.source_path,
        "sample_rate": float(signal.sample_rate),
        "center_frequency": signal.center_frequency,
        "source_sample_format": signal.sample_format,
        "recording_sample_count": int(signal.sample_count),
        "detector": detector.name,
        "number_of_detected_regions": len(detected_regions),
        "number_of_regions": len(regions),
        "number_of_windows": len(windows),
        "seq_len": config.seq_len,
        "hop_len": config.hop_len,
        "remainder": config.remainder,
        "normalization": config.normalization,
        "remove_dc": config.remove_dc,
        "label": config.label,
        "class_name": config.class_name,
        "region_id": np.asarray(region_ids, dtype=np.int64),
        "window_id": np.asarray(window_ids, dtype=np.int64),
        "window_start_sample": np.asarray(starts, dtype=np.int64),
        "window_end_sample": np.asarray(ends, dtype=np.int64),
        "valid_sample_count": np.asarray(valid_counts, dtype=np.int64),
        "region_start_sample": np.asarray(region_starts, dtype=np.int64),
        "region_end_sample": np.asarray(region_ends, dtype=np.int64),
        "normalization_scale": np.asarray(normalization_scales, dtype=np.float64),
        "dc_offset_i": np.asarray(dc_offsets_i, dtype=np.float64),
        "dc_offset_q": np.asarray(dc_offsets_q, dtype=np.float64),
        "source_metadata": dict(signal.metadata),
    }
    return PreparedDataset(X=x, y=y, meta=meta, source_id=signal.source_id)


def prepare_file(
    path: str | Path,
    *,
    source_id: str,
    detector: SignalDetector,
    config: PreparationConfig,
    data_format: str = "auto",
    reader_options: dict[str, Any] | None = None,
) -> PreparedDataset:
    """Open a raw file through the registry and run :func:`prepare_signal`."""

    signal = open_raw_signal(
        path,
        source_id=source_id,
        data_format=data_format,
        **dict(reader_options or {}),
    )
    return prepare_signal(signal, detector=detector, config=config)


__all__ = ["prepare_file", "prepare_signal"]

