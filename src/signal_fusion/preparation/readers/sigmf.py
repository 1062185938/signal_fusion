"""Lazy SigMF raw-IQ reader."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.preparation.contracts import RawSignal
from signal_fusion.preparation.readers.base import RawSignalReader


@dataclass(frozen=True, slots=True)
class _DataTypeSpec:
    name: str
    dtype: np.dtype
    integer_scale: float | None


_DATATYPES = {
    "cf32_le": _DataTypeSpec(
        "cf32_le", np.dtype([("i", "<f4"), ("q", "<f4")]), None
    ),
    "cf32": _DataTypeSpec(
        "cf32", np.dtype([("i", "<f4"), ("q", "<f4")]), None
    ),
    "cf32_be": _DataTypeSpec(
        "cf32_be", np.dtype([("i", ">f4"), ("q", ">f4")]), None
    ),
    "ci16_le": _DataTypeSpec(
        "ci16_le", np.dtype([("i", "<i2"), ("q", "<i2")]), 1.0 / 32768.0
    ),
    "ci16": _DataTypeSpec(
        "ci16", np.dtype([("i", "<i2"), ("q", "<i2")]), 1.0 / 32768.0
    ),
    "ci16_be": _DataTypeSpec(
        "ci16_be", np.dtype([("i", ">i2"), ("q", ">i2")]), 1.0 / 32768.0
    ),
    "ci8_le": _DataTypeSpec(
        "ci8_le", np.dtype([("i", "i1"), ("q", "i1")]), 1.0 / 128.0
    ),
    "ci8": _DataTypeSpec(
        "ci8", np.dtype([("i", "i1"), ("q", "i1")]), 1.0 / 128.0
    ),
}


def _parse_datatype(value: str) -> _DataTypeSpec:
    datatype = str(value).strip().lower()
    try:
        return _DATATYPES[datatype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SigMF datatype {value!r}. "
            f"Supported: {sorted(_DATATYPES)}"
        ) from exc


def _resolve_paths(
    path: str | Path, metadata_path: str | Path | None
) -> tuple[Path, Path]:
    supplied = Path(path)
    if metadata_path is not None:
        return supplied, Path(metadata_path)
    if supplied.name.endswith(".sigmf-meta"):
        return supplied.with_suffix(".sigmf-data"), supplied
    if supplied.name.endswith(".sigmf-data"):
        return supplied, supplied.with_suffix(".sigmf-meta")
    raise ValueError(
        "SigMF path must end with '.sigmf-data' or '.sigmf-meta', or "
        "metadata_path must be provided explicitly"
    )


def _convert_raw(raw: np.ndarray, spec: _DataTypeSpec) -> np.ndarray:
    i_values = raw["i"].astype(np.float32, copy=False)
    q_values = raw["q"].astype(np.float32, copy=False)
    if spec.integer_scale is not None:
        i_values = i_values * spec.integer_scale
        q_values = q_values * spec.integer_scale
    return (i_values + 1j * q_values).astype(np.complex64, copy=False)


class SigMFReader(RawSignalReader):
    format_name = "sigmf"

    def open(
        self,
        path: str | Path,
        *,
        source_id: str,
        metadata_path: str | Path | None = None,
    ) -> RawSignal:
        data_path, meta_path = _resolve_paths(path, metadata_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"SigMF data file not found: {data_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"SigMF metadata file not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as handle:
            raw_metadata: dict[str, Any] = json.load(handle)
        global_metadata = raw_metadata.get("global")
        if not isinstance(global_metadata, dict):
            raise ValueError("SigMF metadata must contain a 'global' object")
        if "core:sample_rate" not in global_metadata:
            raise ValueError("SigMF global metadata is missing 'core:sample_rate'")
        if "core:datatype" not in global_metadata:
            raise ValueError("SigMF global metadata is missing 'core:datatype'")

        sample_rate = float(global_metadata["core:sample_rate"])
        datatype = str(global_metadata["core:datatype"])
        spec = _parse_datatype(datatype)
        file_size = data_path.stat().st_size
        if file_size % spec.dtype.itemsize != 0:
            raise ValueError(
                "SigMF file size is not aligned to its datatype: "
                f"file_size={file_size}, bytes_per_sample={spec.dtype.itemsize}"
            )
        sample_count = file_size // spec.dtype.itemsize

        captures = raw_metadata.get("captures")
        first_capture = captures[0] if isinstance(captures, list) and captures else {}
        center_frequency = first_capture.get("core:frequency")
        datetime_value = first_capture.get(
            "core:datetime", global_metadata.get("core:datetime")
        )

        def read_samples(start: int, count: int) -> np.ndarray:
            with data_path.open("rb") as handle:
                handle.seek(start * spec.dtype.itemsize)
                raw = np.fromfile(handle, dtype=spec.dtype, count=count)
            return _convert_raw(raw, spec)

        return RawSignal(
            source_id=source_id,
            source_path=str(data_path),
            sample_rate=sample_rate,
            sample_count=sample_count,
            sample_format=datatype,
            center_frequency=center_frequency,
            _sample_reader=read_samples,
            metadata={
                "data_format": self.format_name,
                "metadata_path": str(meta_path),
                "sigmf_datatype": datatype,
                "integer_scaled": spec.integer_scale is not None,
                "integer_scale": spec.integer_scale,
                "description": global_metadata.get("core:description"),
                "datetime": datetime_value,
                "sigmf_metadata": raw_metadata,
            },
        )


__all__ = ["SigMFReader"]
