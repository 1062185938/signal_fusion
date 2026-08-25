"""Raw-file reader registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from signal_fusion.preparation.contracts import RawSignal
from signal_fusion.preparation.readers.base import RawSignalReader
from signal_fusion.preparation.readers.dat import ComplexDatReader
from signal_fusion.preparation.readers.mat import MatReader
from signal_fusion.preparation.readers.sigmf import SigMFReader


RAW_READER_REGISTRY: dict[str, RawSignalReader] = {
    "sigmf": SigMFReader(),
    "mat": MatReader(),
    "dat": ComplexDatReader(),
    "bin": ComplexDatReader(),
}


def resolve_raw_format(path: str | Path, data_format: str = "auto") -> str:
    requested = str(data_format or "auto").lower()
    if requested != "auto":
        if requested not in RAW_READER_REGISTRY:
            raise ValueError(
                f"Unsupported raw data format {data_format!r}. "
                f"Supported: {sorted(RAW_READER_REGISTRY)}"
            )
        return requested

    name = Path(path).name.lower()
    if name.endswith((".sigmf-data", ".sigmf-meta")):
        return "sigmf"
    suffix = Path(path).suffix.lower()
    suffix_map = {".mat": "mat", ".dat": "dat", ".bin": "bin"}
    try:
        return suffix_map[suffix]
    except KeyError as exc:
        raise ValueError(f"Cannot infer raw data format from path: {path}") from exc


def open_raw_signal(
    path: str | Path,
    *,
    source_id: str,
    data_format: str = "auto",
    **reader_options: Any,
) -> RawSignal:
    """Open a raw recording through the format registry."""

    resolved_format = resolve_raw_format(path, data_format)
    return RAW_READER_REGISTRY[resolved_format].open(
        path, source_id=source_id, **reader_options
    )


__all__ = [
    "ComplexDatReader",
    "MatReader",
    "RAW_READER_REGISTRY",
    "RawSignalReader",
    "SigMFReader",
    "open_raw_signal",
    "resolve_raw_format",
]

