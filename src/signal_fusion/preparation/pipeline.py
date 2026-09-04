"""Composition pipeline from raw recording to PreparedDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.preparation.contracts import (
    PreparationConfig,
    RawSignal,
    ResamplingConfig,
    SignalRegion,
)
from signal_fusion.preparation.detectors.base import SignalDetector
from signal_fusion.preparation.normalization import normalize_iq
from signal_fusion.preparation.readers import open_raw_signal
from signal_fusion.preparation.resampling import (
    ResamplingPlan,
    build_resampling_plan,
    resample_region,
)
from signal_fusion.preparation.segmentation import segment_regions
from signal_fusion.preparation.windowing import window_spans


def prepare_signal(
    signal: RawSignal,
    *,
    detector: SignalDetector,
    config: PreparationConfig,
    resampling: ResamplingConfig | None = None,
) -> PreparedDataset:
    """Detect, segment, optionally resample, window, and normalize a recording.

    Detection and segmentation always use the source recording and its native
    sample rate.  When ``resampling`` is provided, complete segmented regions
    are converted before target-grid windowing and normalization.
    """

    detected_regions = detector.detect(signal)
    regions = segment_regions(
        detected_regions,
        recording_sample_count=signal.sample_count,
        config=config,
    )
    resampling_plan: ResamplingPlan | None = None
    if resampling is not None:
        resampling_plan = build_resampling_plan(signal.sample_rate, resampling)

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
    source_window_starts: list[int] = []
    source_window_ends: list[int] = []
    source_region_starts: list[int] = []
    source_region_ends: list[int] = []
    source_read_starts: list[int] = []
    source_read_ends: list[int] = []
    target_read_starts: list[int] = []
    target_read_ends: list[int] = []

    for region_id, region in enumerate(regions):
        processing_region = region
        resampled = None
        if resampling_plan is not None:
            resampled = resample_region(signal, region, plan=resampling_plan)
            if resampled.sample_count == 0:
                continue
            processing_region = SignalRegion(
                start_sample=resampled.target_region_start_sample,
                end_sample=resampled.target_region_end_sample,
                detector=region.detector,
                score=region.score,
                metadata=region.metadata,
            )

        for window_id, span in enumerate(window_spans(processing_region, config)):
            if resampled is None:
                raw_window = signal.read_samples(
                    span.start_sample, span.valid_samples
                )
            else:
                local_start = span.start_sample - resampled.target_region_start_sample
                local_end = local_start + span.valid_samples
                raw_window = resampled.samples[local_start:local_end]
                if raw_window.shape != (span.valid_samples,):
                    raise RuntimeError(
                        "target window falls outside its resampled region"
                    )
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
            region_starts.append(processing_region.start_sample)
            region_ends.append(processing_region.end_sample)
            normalization_scales.append(normalized.scale)
            dc_offsets_i.append(float(normalized.dc_offset.real))
            dc_offsets_q.append(float(normalized.dc_offset.imag))

            if resampled is not None and resampling_plan is not None:
                source_window_start, source_window_end = (
                    resampling_plan.target_interval_to_source(
                        span.start_sample, span.end_sample
                    )
                )
                source_window_starts.append(
                    max(region.start_sample, source_window_start)
                )
                source_window_ends.append(
                    min(region.end_sample, source_window_end)
                )
                source_region_starts.append(region.start_sample)
                source_region_ends.append(region.end_sample)
                source_read_starts.append(resampled.source_read_start_sample)
                source_read_ends.append(resampled.source_read_end_sample)
                target_read_starts.append(resampled.target_read_start_sample)
                target_read_ends.append(resampled.target_read_end_sample)

    x = np.empty((len(windows), 2, config.seq_len), dtype=np.float32)
    for index, window in enumerate(windows):
        x[index, 0, :] = window.real
        x[index, 1, :] = window.imag
    y = None
    if config.label is not None:
        y = np.full(len(windows), config.label, dtype=np.int64)

    output_sample_rate = (
        float(signal.sample_rate)
        if resampling_plan is None
        else resampling_plan.effective_sample_rate
    )
    output_recording_sample_count = (
        int(signal.sample_count)
        if resampling_plan is None
        else resampling_plan.output_length(signal.sample_count)
    )
    meta: dict[str, Any] = {
        "source_id": signal.source_id,
        "source_path": signal.source_path,
        "sample_rate": output_sample_rate,
        "center_frequency": signal.center_frequency,
        "source_sample_format": signal.sample_format,
        "recording_sample_count": output_recording_sample_count,
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
    if resampling_plan is not None:
        meta.update(resampling_plan.to_metadata())
        meta.update(
            {
                "resampling_enabled": True,
                "coordinate_schema": "dual_rate_v1",
                "detector_coordinate_system": "source",
                "segmentation_coordinate_system": "source",
                "region_coordinate_system": "target",
                "window_coordinate_system": "target",
                "source_recording_sample_count": int(signal.sample_count),
                "target_recording_sample_count": output_recording_sample_count,
                "source_window_start_sample": np.asarray(
                    source_window_starts, dtype=np.int64
                ),
                "source_window_end_sample": np.asarray(
                    source_window_ends, dtype=np.int64
                ),
                "source_region_start_sample": np.asarray(
                    source_region_starts, dtype=np.int64
                ),
                "source_region_end_sample": np.asarray(
                    source_region_ends, dtype=np.int64
                ),
                "source_read_start_sample": np.asarray(
                    source_read_starts, dtype=np.int64
                ),
                "source_read_end_sample": np.asarray(
                    source_read_ends, dtype=np.int64
                ),
                "target_window_start_sample": np.asarray(
                    starts, dtype=np.int64
                ),
                "target_window_end_sample": np.asarray(ends, dtype=np.int64),
                "target_region_start_sample": np.asarray(
                    region_starts, dtype=np.int64
                ),
                "target_region_end_sample": np.asarray(
                    region_ends, dtype=np.int64
                ),
                "target_read_start_sample": np.asarray(
                    target_read_starts, dtype=np.int64
                ),
                "target_read_end_sample": np.asarray(
                    target_read_ends, dtype=np.int64
                ),
            }
        )
    return PreparedDataset(X=x, y=y, meta=meta, source_id=signal.source_id)


def prepare_file(
    path: str | Path,
    *,
    source_id: str,
    detector: SignalDetector,
    config: PreparationConfig,
    resampling: ResamplingConfig | None = None,
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
    return prepare_signal(
        signal,
        detector=detector,
        config=config,
        resampling=resampling,
    )


__all__ = ["prepare_file", "prepare_signal"]
