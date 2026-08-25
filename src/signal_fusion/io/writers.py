"""Shared writers for canonical prepared IQ datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from signal_fusion.contracts import PreparedDataset


_RESERVED_FIELDS = frozenset({"X", "y", "source_id"})


def json_safe(value: Any) -> Any:
    """Convert NumPy and path values into JSON-compatible Python values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dataset_summary_path(output_path: str | Path) -> Path:
    """Return the conventional JSON summary path next to an NPZ dataset."""

    output = Path(output_path)
    return output.with_name(f"{output.stem}_dataset_summary.json")


def _npz_value(value: Any) -> np.ndarray | None:
    """Return a pickle-free NPZ value, or ``None`` for structured metadata."""

    if value is None:
        return None
    if isinstance(value, Path):
        return np.asarray(str(value))
    if isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        return None if value.dtype.hasobject else value
    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        return None if array.dtype.hasobject else array
    return None


def prepared_dataset_payload(
    dataset: PreparedDataset,
    *,
    metadata_fields: Iterable[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
    include_source_id: bool = True,
) -> dict[str, np.ndarray]:
    """Build a stable, ``allow_pickle=False`` compatible NPZ payload.

    When ``metadata_fields`` is omitted, every flat serializable value from
    ``dataset.meta`` is included. Nested dictionaries remain in the JSON
    summary instead of becoming object arrays in the NPZ file.
    """

    payload: dict[str, np.ndarray] = {"X": dataset.X}
    if dataset.y is not None:
        payload["y"] = dataset.y

    automatic_fields = metadata_fields is None
    selected_fields = (
        tuple(dataset.meta) if automatic_fields else tuple(metadata_fields)
    )
    for field_name in selected_fields:
        if field_name in _RESERVED_FIELDS:
            if automatic_fields:
                continue
            raise ValueError(f"metadata field {field_name!r} is reserved")
        if field_name not in dataset.meta:
            raise KeyError(f"PreparedDataset.meta is missing {field_name!r}")
        value = _npz_value(dataset.meta[field_name])
        if value is not None:
            payload[field_name] = value

    for field_name, raw_value in dict(extra_fields or {}).items():
        if field_name in _RESERVED_FIELDS:
            raise ValueError(f"extra field {field_name!r} is reserved")
        value = _npz_value(raw_value)
        if value is None:
            raise TypeError(
                f"extra field {field_name!r} is not pickle-free NPZ metadata"
            )
        payload[field_name] = value

    if include_source_id and dataset.source_id is not None:
        payload["source_id"] = np.asarray(dataset.source_id)
    return payload


def write_prepared_dataset(
    dataset: PreparedDataset,
    output_path: str | Path,
    *,
    metadata_fields: Iterable[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
    include_source_id: bool = True,
) -> Path:
    """Write one canonical dataset as a compressed, pickle-free NPZ file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **prepared_dataset_payload(
            dataset,
            metadata_fields=metadata_fields,
            extra_fields=extra_fields,
            include_source_id=include_source_id,
        ),
    )
    return output


def write_dataset_summary(
    summary: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a human-readable JSON summary next to a prepared dataset."""

    summary_path = dataset_summary_path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(json_safe(dict(summary)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


__all__ = [
    "dataset_summary_path",
    "json_safe",
    "prepared_dataset_payload",
    "write_dataset_summary",
    "write_prepared_dataset",
]
