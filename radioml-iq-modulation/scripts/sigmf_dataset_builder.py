"""Temporary compatibility wrapper for signal_fusion preparation energy_v1."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.preparation.legacy_sigmf import (  # noqa: E402
    DataTypeSpec,
    SigMFMetadata,
    SigMFReader,
    build_sigmf_dataset,
    legacy_main,
    load_sigmf_metadata,
    parse_sigmf_datatype,
)


main = legacy_main

__all__ = [
    "DataTypeSpec",
    "SigMFMetadata",
    "SigMFReader",
    "build_sigmf_dataset",
    "load_sigmf_metadata",
    "main",
    "parse_sigmf_datatype",
]


if __name__ == "__main__":
    raise SystemExit(main())
