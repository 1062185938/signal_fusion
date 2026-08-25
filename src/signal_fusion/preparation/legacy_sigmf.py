"""Temporary API and CLI compatibility for sigmf_dataset_builder.py."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from signal_fusion.preparation.energy_v1_dataset import build_sigmf_dataset


@dataclass(frozen=True)
class DataTypeSpec:
    name: str
    dtype: np.dtype
    is_integer: bool
    scale: float


@dataclass(frozen=True)
class SigMFMetadata:
    sample_rate: float
    datatype: str
    dtype_spec: DataTypeSpec
    description: str | None
    center_frequency: float | None
    datetime: str | None
    raw: dict[str, Any]


def parse_sigmf_datatype(datatype: str) -> DataTypeSpec:
    name = str(datatype).lower().strip()
    formats = {
        "cf32_le": DataTypeSpec(name, np.dtype([("r", "<f4"), ("i", "<f4")]), False, 1.0),
        "cf32": DataTypeSpec(name, np.dtype([("r", "<f4"), ("i", "<f4")]), False, 1.0),
        "cf32_be": DataTypeSpec(name, np.dtype([("r", ">f4"), ("i", ">f4")]), False, 1.0),
        "ci16_le": DataTypeSpec(name, np.dtype([("r", "<i2"), ("i", "<i2")]), True, 1.0 / 32768.0),
        "ci16": DataTypeSpec(name, np.dtype([("r", "<i2"), ("i", "<i2")]), True, 1.0 / 32768.0),
        "ci16_be": DataTypeSpec(name, np.dtype([("r", ">i2"), ("i", ">i2")]), True, 1.0 / 32768.0),
        "ci8_le": DataTypeSpec(name, np.dtype([("r", "i1"), ("i", "i1")]), True, 1.0 / 128.0),
        "ci8": DataTypeSpec(name, np.dtype([("r", "i1"), ("i", "i1")]), True, 1.0 / 128.0),
    }
    try:
        return formats[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SigMF datatype {datatype!r}. Supported: {sorted(formats)}"
        ) from exc


def load_sigmf_metadata(meta_path: str) -> SigMFMetadata:
    path = Path(meta_path)
    if not path.is_file():
        raise FileNotFoundError(f"SigMF metadata file not found: {meta_path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    global_metadata = raw.get("global")
    if not isinstance(global_metadata, dict):
        raise ValueError("SigMF metadata must contain a 'global' object.")
    if "core:sample_rate" not in global_metadata:
        raise ValueError("SigMF metadata global section is missing 'core:sample_rate'.")
    if "core:datatype" not in global_metadata:
        raise ValueError("SigMF metadata global section is missing 'core:datatype'.")
    sample_rate = float(global_metadata["core:sample_rate"])
    if sample_rate <= 0:
        raise ValueError("SigMF 'core:sample_rate' must be positive.")
    datatype = str(global_metadata["core:datatype"])
    captures = raw.get("captures")
    capture = captures[0] if isinstance(captures, list) and captures else {}
    center_frequency = capture.get("core:frequency")
    return SigMFMetadata(
        sample_rate=sample_rate,
        datatype=datatype,
        dtype_spec=parse_sigmf_datatype(datatype),
        description=global_metadata.get("core:description"),
        center_frequency=None if center_frequency is None else float(center_frequency),
        datetime=capture.get("core:datetime", global_metadata.get("core:datetime")),
        raw=raw,
    )


class SigMFReader:
    """Constructor-compatible adapter retained for the old test/API surface."""

    def __init__(self, data_path: str, dtype_spec: DataTypeSpec):
        self.data_path = data_path
        self.dtype_spec = dtype_spec
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"SigMF data file not found: {data_path}")
        size = path.stat().st_size
        if size % dtype_spec.dtype.itemsize != 0:
            raise ValueError("SigMF data file size is not aligned to datatype")
        self.total_samples = size // dtype_spec.dtype.itemsize

    def read_samples(self, start_sample: int, count: int) -> np.ndarray:
        start = max(0, int(start_sample))
        count = min(max(0, int(count)), self.total_samples - start)
        if count <= 0:
            return np.empty(0, dtype=np.complex64)
        with open(self.data_path, "rb") as handle:
            handle.seek(start * self.dtype_spec.dtype.itemsize)
            raw = np.fromfile(handle, dtype=self.dtype_spec.dtype, count=count)
        real = raw["r"].astype(np.float32, copy=False)
        imag = raw["i"].astype(np.float32, copy=False)
        if self.dtype_spec.is_integer:
            real = real * self.dtype_spec.scale
            imag = imag * self.dtype_spec.scale
        return (real + 1j * imag).astype(np.complex64, copy=False)


def build_legacy_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fixed-length IQ dataset from one SigMF LoRa recording."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--class_name", default="LoRa")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--hop_len", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--window_ms", type=float, default=1.0)
    parser.add_argument("--start_threshold_db", type=float, default=6.0)
    parser.add_argument("--end_threshold_db", type=float, default=5.0)
    parser.add_argument("--min_signal_ms", type=float, default=8.0)
    parser.add_argument("--min_gap_ms", type=float, default=2.0)
    parser.add_argument("--pad_before_ms", type=float, default=0.5)
    parser.add_argument("--pad_after_ms", type=float, default=0.2)
    parser.add_argument("--normalize", choices=["rms", "none"], default="rms")
    dc_group = parser.add_mutually_exclusive_group()
    dc_group.add_argument("--remove_dc", dest="remove_dc", action="store_true")
    dc_group.add_argument("--no_remove_dc", dest="remove_dc", action="store_false")
    parser.set_defaults(remove_dc=True)
    parser.add_argument("--remainder", choices=["drop", "zero_pad"], default="drop")
    parser.add_argument("--release_windows", type=int, default=2)
    parser.add_argument("--noise_percentile", type=float, default=20.0)
    parser.add_argument("--noise_probe_count", type=int, default=8)
    parser.add_argument("--ignore_initial_ms", type=float, default=0.0)
    parser.add_argument("--window_power_ratio", type=float, default=0.05)
    return parser


def legacy_main(argv: list[str] | None = None) -> int:
    args = build_legacy_arg_parser().parse_args(argv)
    try:
        result = build_sigmf_dataset(**vars(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("\nSigMF dataset build complete")
    print(f"  output_path:    {result['output_path']}")
    print(f"  summary_path:   {result['summary_path']}")
    print(f"  sample_rate:    {result['sample_rate']}")
    print(f"  noise_floor_db: {result['noise_floor_db']:.2f}")
    print(f"  bursts:         {result['num_bursts']}")
    print(f"  windows:        {result['num_samples']}")
    print(f"  X shape:        {result['x_shape']}")
    print(f"  label/class:    {result['label']} / {result['class_name']}")
    return 0


__all__ = [
    "DataTypeSpec",
    "SigMFMetadata",
    "SigMFReader",
    "build_sigmf_dataset",
    "legacy_main",
    "load_sigmf_metadata",
    "parse_sigmf_datatype",
]

