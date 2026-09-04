"""File-level orchestration for the generic preparation pipeline."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from signal_fusion.io.writers import write_dataset_summary, write_prepared_dataset
from signal_fusion.preparation.contracts import PreparationConfig, ResamplingConfig
from signal_fusion.preparation.detectors.base import SignalDetector
from signal_fusion.preparation.pipeline import prepare_signal
from signal_fusion.preparation.readers import open_raw_signal


def build_prepared_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_id: str,
    detector: SignalDetector,
    config: PreparationConfig,
    resampling: ResamplingConfig | None = None,
    data_format: str = "auto",
    reader_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare and save data with the format-independent standard pipeline."""

    signal = open_raw_signal(
        input_path,
        source_id=source_id,
        data_format=data_format,
        **dict(reader_options or {}),
    )
    dataset = prepare_signal(
        signal,
        detector=detector,
        config=config,
        resampling=resampling,
    )
    written_path = write_prepared_dataset(dataset, output_path)
    detector_config: dict[str, Any] = {}
    if is_dataclass(detector):
        detector_config = asdict(detector)
    else:
        config_value = getattr(detector, "config", None)
        if is_dataclass(config_value):
            detector_config = asdict(config_value)
    detector_report = getattr(detector, "last_report", None)
    summary = {
        "source_id": source_id,
        "source_data_path": str(input_path),
        "source_format": signal.metadata.get("data_format"),
        "source_metadata": signal.metadata,
        "sample_rate": dataset.meta["sample_rate"],
        "center_frequency": signal.center_frequency,
        "total_raw_samples": signal.sample_count,
        "recording_duration": signal.duration_seconds,
        "detector": detector.name,
        "detector_config": detector_config,
        "preparation_config": asdict(config),
        "number_of_detected_regions": dataset.meta["number_of_detected_regions"],
        "number_of_regions": dataset.meta["number_of_regions"],
        "number_of_generated_windows": dataset.num_samples,
        "seq_len": dataset.seq_len,
        "hop_len": dataset.meta["hop_len"],
        "normalization_mode": dataset.meta["normalization"],
        "remove_dc": dataset.meta["remove_dc"],
        "remainder": dataset.meta["remainder"],
        "label": dataset.meta["label"],
        "class_name": dataset.meta["class_name"],
        "output_path": str(written_path),
        "output_shape": list(dataset.X.shape),
        "output_dtype": str(dataset.X.dtype),
    }
    if detector_report is not None:
        summary["detector_report"] = detector_report
    if resampling is not None:
        summary["resampling_config"] = asdict(resampling)
        summary["source_sample_rate"] = signal.sample_rate
        summary["resampling"] = {
            key: dataset.meta[key]
            for key in (
                "target_sample_rate",
                "effective_sample_rate",
                "resampling_profile",
                "resampling_method",
                "resampling_applied",
                "resample_up",
                "resample_down",
            )
        }
    summary_path = write_dataset_summary(summary, written_path)
    return {
        "output_path": str(written_path),
        "num_regions": int(dataset.meta["number_of_regions"]),
        "num_samples": dataset.num_samples,
        "x_shape": tuple(dataset.X.shape),
        "label": config.label,
        "class_name": config.class_name,
        "sample_rate": dataset.meta["sample_rate"],
        "detector": detector.name,
        "summary_path": str(summary_path),
    }


__all__ = ["build_prepared_dataset"]
