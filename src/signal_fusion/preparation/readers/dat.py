"""Lazy reader for headerless complex64 DAT/BIN recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from signal_fusion.preparation.contracts import RawSignal
from signal_fusion.preparation.readers.base import RawSignalReader


class ComplexDatReader(RawSignalReader):
    format_name = "dat"

    def open(
        self,
        path: str | Path,
        *,
        source_id: str,
        sample_rate: float,
        center_frequency: float | None = None,
        iq_format: str = "complex64",
    ) -> RawSignal:
        data_path = Path(path)
        if not data_path.is_file():
            raise FileNotFoundError(f"DAT/BIN file not found: {data_path}")
        if iq_format != "complex64":
            raise ValueError("Phase 2A DAT/BIN reader supports only complex64 IQ")

        dtype = np.dtype("<c8")
        file_size = data_path.stat().st_size
        if file_size % dtype.itemsize != 0:
            raise ValueError(
                "DAT/BIN file size is not aligned to complex64: "
                f"file_size={file_size}, bytes_per_sample={dtype.itemsize}"
            )
        sample_count = file_size // dtype.itemsize

        def read_samples(start: int, count: int) -> np.ndarray:
            with data_path.open("rb") as handle:
                handle.seek(start * dtype.itemsize)
                return np.fromfile(handle, dtype=dtype, count=count).astype(
                    np.complex64, copy=False
                )

        return RawSignal(
            source_id=source_id,
            source_path=str(data_path),
            sample_rate=sample_rate,
            sample_count=sample_count,
            sample_format=iq_format,
            center_frequency=center_frequency,
            _sample_reader=read_samples,
            metadata={
                "data_format": self.format_name,
                "bytes_per_sample": dtype.itemsize,
            },
        )


__all__ = ["ComplexDatReader"]

