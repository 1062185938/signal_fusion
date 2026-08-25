"""Temporary compatibility wrapper for the core 62-feature map."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.feature_extraction.feature_map import (  # noqa: E402
    EXPECTED_GROUP_COUNTS,
    REQUIRED_FEATURE_FIELDS,
    default_feature_map_path,
    feature_code_names,
    load_feature_map,
)


__all__ = [
    "EXPECTED_GROUP_COUNTS",
    "REQUIRED_FEATURE_FIELDS",
    "default_feature_map_path",
    "feature_code_names",
    "load_feature_map",
]
