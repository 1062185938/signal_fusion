"""Signal-region detector interfaces and format-independent strategies."""

from .base import SignalDetector
from .ble_packet_v1 import BlePacketDetectorV1, BlePacketDetectorV1Config
from .energy_v1 import EnergyDetectorV1, EnergyDetectorV1Config
from .fixed_blocks import FixedBlockDetector
from .full_signal import FullSignalDetector
from .registry import DETECTOR_REGISTRY, available_detectors, build_detector

__all__ = [
    "EnergyDetectorV1",
    "EnergyDetectorV1Config",
    "FixedBlockDetector",
    "BlePacketDetectorV1",
    "BlePacketDetectorV1Config",
    "FullSignalDetector",
    "SignalDetector",
    "DETECTOR_REGISTRY",
    "available_detectors",
    "build_detector",
]
