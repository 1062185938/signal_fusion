"""Temporary compatibility wrapper for the shared signal_fusion loader.

The legacy feature-extraction CLI imports this module by its historical path.
Keep that path working until the CLI itself is replaced by a package entry point.
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

