"""Signal-region detector interfaces and format-independent strategies."""

from .base import SignalDetector
from .energy_v1 import EnergyDetectorV1, EnergyDetectorV1Config
from .full_signal import FullSignalDetector
from .registry import DETECTOR_REGISTRY, available_detectors, build_detector

__all__ = [
    "EnergyDetectorV1",
    "EnergyDetectorV1Config",
    "FullSignalDetector",
    "SignalDetector",
    "DETECTOR_REGISTRY",
    "available_detectors",
    "build_detector",
]
