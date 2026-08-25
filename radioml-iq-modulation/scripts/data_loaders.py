"""Temporary compatibility wrapper for the shared signal_fusion loader.

The legacy training and inference CLIs import this module by its historical
path. Keep that path working until those CLIs become package entry points.
"""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.io.loaders import (  # noqa: E402
    SUPPORTED_FORMATS,
    load_prepared_dataset,
    load_signal_dataset,
    load_signal_for_inference,
)


__all__ = [
    "SUPPORTED_FORMATS",
    "load_prepared_dataset",
    "load_signal_dataset",
    "load_signal_for_inference",
]

