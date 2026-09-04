"""Core contracts and shared infrastructure for signal_fusion."""

from .contracts import Evidence, ModelManifest, PreparedDataset
from .feature_extraction import FeatureExtractionService, FeatureResult
from .io import load_prepared_dataset, load_signal_dataset, load_signal_for_inference
from .model_inference import (
    ModelInferenceResult,
    ModelInferenceService,
    ONNXRuntimeBackend,
    RankedPrediction,
)
from .preparation import (
    BlePacketDetectorV1,
    BlePacketDetectorV1Config,
    EnergyDetectorV1,
    EnergyDetectorV1Config,
    FullSignalDetector,
    PreparationConfig,
    RawSignal,
    SignalRegion,
    build_energy_v1_dataset,
    open_raw_signal,
    prepare_file,
    prepare_signal,
)

__all__ = [
    "Evidence",
    "BlePacketDetectorV1",
    "BlePacketDetectorV1Config",
    "EnergyDetectorV1",
    "EnergyDetectorV1Config",
    "ModelManifest",
    "ModelInferenceResult",
    "ModelInferenceService",
    "ONNXRuntimeBackend",
    "PreparedDataset",
    "RankedPrediction",
    "FullSignalDetector",
    "FeatureExtractionService",
    "FeatureResult",
    "PreparationConfig",
    "RawSignal",
    "SignalRegion",
    "build_energy_v1_dataset",
    "load_prepared_dataset",
    "load_signal_dataset",
    "load_signal_for_inference",
    "open_raw_signal",
    "prepare_file",
    "prepare_signal",
]

__version__ = "0.1.0"
