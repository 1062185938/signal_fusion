"""Small-model inference runtime for canonical IQ datasets."""

from .backend import ONNXRuntimeBackend, preferred_execution_providers
from .contracts import ModelInferenceResult, RankedPrediction
from .labels import load_label_map, normalize_label_map
from .legacy import (
    _sync_signal_inference,
    load_mod_labels,
    recognize_iq_modulation,
    run_onnx_inference,
)
from .service import ModelBackend, ModelInferenceService, stable_softmax


__all__ = [
    "ModelBackend",
    "ModelInferenceResult",
    "ModelInferenceService",
    "ONNXRuntimeBackend",
    "RankedPrediction",
    "_sync_signal_inference",
    "load_label_map",
    "load_mod_labels",
    "normalize_label_map",
    "preferred_execution_providers",
    "recognize_iq_modulation",
    "run_onnx_inference",
    "stable_softmax",
]
