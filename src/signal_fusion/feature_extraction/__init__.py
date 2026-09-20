"""Core runtime for stable 64-dimensional IQ feature extraction."""

from .backend import (
    IQFeatureCtypesBackend,
    MAX_SIGNAL_LENGTH,
    MIN_SIGNAL_LENGTH,
    STATUS_MESSAGES,
    default_library_dir,
    default_library_name,
)
from .contracts import FEATURE_COUNT, FEATURE_SCHEMA_ID, FeatureResult
from .feature_map import (
    EXPECTED_GROUP_COUNTS,
    default_feature_map_path,
    feature_code_names,
    load_feature_map,
)
from .ofdm_periodicity import (
    DEFAULT_SEARCH_RADIUS_SAMPLES,
    PERIODICITY_SCHEMA_ID,
    PeriodicityBatchResult,
    PeriodicityCandidate,
    measure_periodicity,
    measure_periodicity_batch,
)
from .resource_paths import (
    asset_path,
    c_api_header_path,
    feature_map_resource_path,
    native_resource_dir,
)
from .service import (
    FeatureBackend,
    FeatureExtractionService,
    extract_feature_matrix,
    resolve_sample_rate,
)


__all__ = [
    "EXPECTED_GROUP_COUNTS",
    "FEATURE_COUNT",
    "FEATURE_SCHEMA_ID",
    "FeatureBackend",
    "FeatureExtractionService",
    "FeatureResult",
    "IQFeatureCtypesBackend",
    "MAX_SIGNAL_LENGTH",
    "MIN_SIGNAL_LENGTH",
    "DEFAULT_SEARCH_RADIUS_SAMPLES",
    "PERIODICITY_SCHEMA_ID",
    "PeriodicityBatchResult",
    "PeriodicityCandidate",
    "STATUS_MESSAGES",
    "asset_path",
    "c_api_header_path",
    "default_feature_map_path",
    "default_library_dir",
    "default_library_name",
    "extract_feature_matrix",
    "feature_code_names",
    "feature_map_resource_path",
    "load_feature_map",
    "measure_periodicity",
    "measure_periodicity_batch",
    "native_resource_dir",
    "resolve_sample_rate",
]
