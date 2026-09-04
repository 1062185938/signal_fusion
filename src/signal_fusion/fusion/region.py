"""Load one complete continuous region referenced by an assembled dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.preparation.contracts import SignalRegion
from signal_fusion.preparation.readers import open_raw_signal, resolve_raw_format
from signal_fusion.preparation.resampling import ResamplingPlan, resample_region


@dataclass(frozen=True, slots=True)
class CompleteRegion:
    """One region on the effective sample-rate grid before normalization."""

    samples: np.ndarray = field(repr=False, compare=False)
    sample_rate: float
    start_sample: int
    end_sample: int
    source_id: str
    source_region_id: int
    raw_source_path: str
    source_dataset_path: str

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1 or not np.iscomplexobj(samples):
            raise ValueError("samples must be one-dimensional complex IQ")
        samples = np.array(samples, dtype=np.complex64, copy=True, order="C")
        if samples.size == 0:
            raise ValueError("complete region must not be empty")
        if not np.isfinite(samples.real).all() or not np.isfinite(samples.imag).all():
            raise ValueError("complete region contains NaN or Inf")
        object.__setattr__(self, "samples", samples)

        sample_rate = float(self.sample_rate)
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        object.__setattr__(self, "sample_rate", sample_rate)

        start = int(self.start_sample)
        end = int(self.end_sample)
        if start < 0 or end <= start:
            raise ValueError("region coordinates must define a non-empty interval")
        if end - start != samples.size:
            raise ValueError("region coordinates do not match the IQ sample count")
        object.__setattr__(self, "start_sample", start)
        object.__setattr__(self, "end_sample", end)
        object.__setattr__(self, "source_region_id", int(self.source_region_id))

        for name in ("source_id", "raw_source_path", "source_dataset_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def sample_count(self) -> int:
        return int(self.samples.size)


def _sample_field(dataset: PreparedDataset, name: str) -> np.ndarray:
    if name not in dataset.meta:
        raise ValueError(f"dataset is missing per-sample metadata {name!r}")
    values = np.asarray(dataset.meta[name])
    if values.shape != (dataset.num_samples,):
        raise ValueError(
            f"metadata {name!r} must have shape [{dataset.num_samples}], "
            f"got {values.shape}"
        )
    return values


def _one_group_value(
    dataset: PreparedDataset,
    name: str,
    indices: np.ndarray,
) -> Any:
    values = _sample_field(dataset, name)[indices]
    unique = np.unique(values)
    if unique.size != 1:
        raise ValueError(
            f"selected group contains multiple {name!r} values: "
            f"{unique.tolist()}"
        )
    return unique[0].item()


def _scalar(container: Any, name: str, *, required: bool = True) -> Any | None:
    if name not in container:
        if required:
            raise ValueError(f"source dataset is missing scalar metadata {name!r}")
        return None
    values = np.asarray(container[name])
    if values.size != 1:
        raise ValueError(f"source metadata {name!r} must be scalar")
    return values.reshape(()).item()


def _region_scalar(
    container: Any,
    name: str,
    region_mask: np.ndarray,
) -> Any:
    if name not in container:
        raise ValueError(f"source dataset is missing region metadata {name!r}")
    values = np.asarray(container[name])
    if values.ndim != 1 or values.shape != region_mask.shape:
        raise ValueError(f"source region metadata {name!r} has an invalid shape")
    unique = np.unique(values[region_mask])
    if unique.size != 1:
        raise ValueError(
            f"source region contains multiple {name!r} values: {unique.tolist()}"
        )
    return unique[0].item()


def _project_root(path: Path) -> Path:
    for directory in (path.parent, *path.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise FileNotFoundError(
        f"cannot locate project root containing pyproject.toml from {path}"
    )


def _project_path(recorded_path: str, source_dataset_path: Path) -> Path:
    path = Path(recorded_path)
    resolved = path if path.is_absolute() else _project_root(source_dataset_path) / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"recorded source file does not exist: {resolved}")
    return resolved


def _source_dataset_from_report(
    dataset_path: Path,
    dataset: PreparedDataset,
    source_id: str,
) -> Path:
    report_path = dataset_path.parent / "assembly_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            "complete region loading requires assembly_report.json next to the "
            f"assembled dataset: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split = str(np.asarray(dataset.meta.get("split", "")).reshape(()).item())
    try:
        sources = report["splits"][split]["sources"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"assembly report does not describe dataset split {split!r}"
        ) from exc
    matches = [item for item in sources if item.get("source_id") == source_id]
    if len(matches) != 1:
        raise ValueError(
            f"assembly report must contain exactly one source_id={source_id!r}"
        )
    source_dataset_path = Path(matches[0]["resolved_path"]).resolve()
    if not source_dataset_path.is_file():
        raise FileNotFoundError(
            f"source slice dataset does not exist: {source_dataset_path}"
        )
    return source_dataset_path


def _resampling_plan(container: Any) -> ResamplingPlan:
    return ResamplingPlan(
        source_sample_rate=float(_scalar(container, "source_sample_rate")),
        target_sample_rate=float(_scalar(container, "target_sample_rate")),
        effective_sample_rate=float(_scalar(container, "effective_sample_rate")),
        up=int(_scalar(container, "resample_up")),
        down=int(_scalar(container, "resample_down")),
        rate_error_ppm=float(_scalar(container, "resampling_rate_error_ppm")),
        profile=str(_scalar(container, "resampling_profile")),
        max_input_samples=int(_scalar(container, "resampling_max_input_samples")),
        max_output_samples=int(_scalar(container, "resampling_max_output_samples")),
    )


def load_complete_region(
    dataset_path: str | Path,
    dataset: PreparedDataset,
    *,
    group_id: int,
) -> CompleteRegion:
    """Reconstruct one complete region from its raw recording and provenance."""

    assembled_path = Path(dataset_path).resolve()
    group_ids = _sample_field(dataset, "group_id").astype(np.int64, copy=False)
    indices = np.flatnonzero(group_ids == int(group_id)).astype(np.int64)
    if indices.size == 0:
        raise ValueError(f"group_id={group_id} does not exist in the dataset")

    source_id = str(_one_group_value(dataset, "sample_source_id", indices))
    source_region_id = int(
        _one_group_value(dataset, "source_region_id", indices)
    )
    source_region_field = str(
        _one_group_value(dataset, "source_region_field", indices)
    )
    target_start = int(_one_group_value(dataset, "region_start_sample", indices))
    target_end = int(_one_group_value(dataset, "region_end_sample", indices))
    expected_sample_rate = float(
        _one_group_value(dataset, "sample_rate", indices)
    )

    source_dataset_path = _source_dataset_from_report(
        assembled_path, dataset, source_id
    )
    with np.load(source_dataset_path, allow_pickle=False) as source_dataset:
        if source_region_field not in source_dataset:
            raise ValueError(
                f"source dataset is missing region field {source_region_field!r}"
            )
        source_region_values = np.asarray(
            source_dataset[source_region_field], dtype=np.int64
        )
        region_mask = source_region_values == source_region_id
        if not np.any(region_mask):
            raise ValueError(
                f"source region {source_region_id} does not exist in "
                f"{source_dataset_path}"
            )

        recorded_source_path = _scalar(
            source_dataset, "source_path", required=False
        )
        if recorded_source_path is None:
            recorded_source_path = _scalar(source_dataset, "source_data_path")
        raw_path = _project_path(str(recorded_source_path), source_dataset_path)
        raw_format = resolve_raw_format(raw_path)
        reader_options: dict[str, Any] = {}
        if raw_format == "sigmf":
            recorded_meta_path = _scalar(
                source_dataset, "source_meta_path", required=False
            )
            if recorded_meta_path is not None:
                reader_options["metadata_path"] = _project_path(
                    str(recorded_meta_path), source_dataset_path
                )
        elif raw_format in {"dat", "bin"}:
            native_rate = _scalar(
                source_dataset, "source_sample_rate", required=False
            )
            reader_options["sample_rate"] = float(
                expected_sample_rate if native_rate is None else native_rate
            )
            center_frequency = _scalar(
                source_dataset, "center_frequency", required=False
            )
            if center_frequency is not None:
                reader_options["center_frequency"] = float(center_frequency)
        elif raw_format == "mat":
            native_rate = _scalar(
                source_dataset, "source_sample_rate", required=False
            )
            if native_rate is not None:
                reader_options["sample_rate"] = float(native_rate)

        raw_signal = open_raw_signal(
            raw_path,
            source_id=source_id,
            data_format=raw_format,
            **reader_options,
        )
        resampling_enabled = bool(
            _scalar(source_dataset, "resampling_enabled", required=False) or False
        )
        if resampling_enabled:
            native_start = int(
                _region_scalar(
                    source_dataset,
                    "source_region_start_sample",
                    region_mask,
                )
            )
            native_end = int(
                _region_scalar(
                    source_dataset,
                    "source_region_end_sample",
                    region_mask,
                )
            )
            plan = _resampling_plan(source_dataset)
            resampled = resample_region(
                raw_signal,
                SignalRegion(native_start, native_end, detector="recorded_region"),
                plan=plan,
            )
            samples = resampled.samples
            actual_start = resampled.target_region_start_sample
            actual_end = resampled.target_region_end_sample
            actual_sample_rate = resampled.plan.effective_sample_rate
        else:
            native_start = target_start
            native_end = target_end
            samples = raw_signal.read_samples(native_start, native_end - native_start)
            actual_start = native_start
            actual_end = native_end
            actual_sample_rate = raw_signal.sample_rate

    if actual_start != target_start or actual_end != target_end:
        raise ValueError(
            "reconstructed region coordinates do not match assembled metadata: "
            f"reconstructed=[{actual_start}, {actual_end}), "
            f"assembled=[{target_start}, {target_end})"
        )
    if not np.isclose(actual_sample_rate, expected_sample_rate):
        raise ValueError(
            "reconstructed region sample rate does not match assembled metadata: "
            f"{actual_sample_rate} != {expected_sample_rate}"
        )
    return CompleteRegion(
        samples=samples,
        sample_rate=actual_sample_rate,
        start_sample=actual_start,
        end_sample=actual_end,
        source_id=source_id,
        source_region_id=source_region_id,
        raw_source_path=str(raw_path),
        source_dataset_path=str(source_dataset_path),
    )


__all__ = ["CompleteRegion", "load_complete_region"]
