"""Small-model inference runtime for canonical IQ datasets."""

from .backend import ONNXRuntimeBackend, preferred_execution_providers
from .contracts import ModelInferenceResult, RankedPrediction
from .ensemble import (
    ACCEPT,
    REVIEW_REQUIRED,
    RegionEnsembleInferenceService,
    RegionEnsembleResult,
    aggregate_region_predictions,
    load_ensemble_services,
)
from .ensemble_evaluation import evaluate_ensemble_results
from .labels import load_label_map, normalize_label_map
from .legacy import (
    _sync_signal_inference,
    load_mod_labels,
    recognize_iq_modulation,
    run_onnx_inference,
)
from .service import ModelBackend, ModelInferenceService, stable_softmax


__all__ = [
    "ACCEPT",
    "ModelBackend",
    "ModelInferenceResult",
    "ModelInferenceService",
    "ONNXRuntimeBackend",
    "RankedPrediction",
    "REVIEW_REQUIRED",
    "RegionEnsembleInferenceService",
    "RegionEnsembleResult",
    "_sync_signal_inference",
    "load_label_map",
    "load_mod_labels",
    "normalize_label_map",
    "preferred_execution_providers",
    "recognize_iq_modulation",
    "run_onnx_inference",
    "aggregate_region_predictions",
    "evaluate_ensemble_results",
    "load_ensemble_services",
    "stable_softmax",
]
