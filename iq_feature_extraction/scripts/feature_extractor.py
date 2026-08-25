"""Temporary CLI/API wrapper for the core feature extraction runtime."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.feature_extraction.backend import (  # noqa: E402
    FEATURE_COUNT,
    IQFeatureCtypesBackend,
)
from signal_fusion.feature_extraction.cli import (  # noqa: E402
    build_arg_parser,
    extract_features_from_dataset,
    main,
    write_feature_result,
)
from signal_fusion.feature_extraction.feature_map import (  # noqa: E402
    feature_code_names,
    load_feature_map,
)
from signal_fusion.feature_extraction.service import (  # noqa: E402
    FeatureExtractionService,
    extract_feature_matrix,
)


__all__ = [
    "FEATURE_COUNT",
    "FeatureExtractionService",
    "IQFeatureCtypesBackend",
    "build_arg_parser",
    "extract_feature_matrix",
    "extract_features_from_dataset",
    "feature_code_names",
    "load_feature_map",
    "main",
    "write_feature_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
