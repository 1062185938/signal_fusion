"""Temporary compatibility wrapper for the core feature backend."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.feature_extraction.backend import (  # noqa: E402
    FEATURE_COUNT,
    IQFeatureCtypesBackend,
    MAX_SIGNAL_LENGTH,
    MIN_SIGNAL_LENGTH,
    STATUS_MESSAGES,
    default_library_dir,
    default_library_name,
)


__all__ = [
    "FEATURE_COUNT",
    "IQFeatureCtypesBackend",
    "MAX_SIGNAL_LENGTH",
    "MIN_SIGNAL_LENGTH",
    "STATUS_MESSAGES",
    "default_library_dir",
    "default_library_name",
]
