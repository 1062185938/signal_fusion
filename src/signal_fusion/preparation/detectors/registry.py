"""Registry and factory for signal-region detection strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from signal_fusion.preparation.detectors.base import SignalDetector
from signal_fusion.preparation.detectors.ble_packet_v1 import (
    BlePacketDetectorV1,
    BlePacketDetectorV1Config,
)
from signal_fusion.preparation.detectors.energy_v1 import (
    EnergyDetectorV1,
    EnergyDetectorV1Config,
)
from signal_fusion.preparation.detectors.fixed_blocks import FixedBlockDetector
from signal_fusion.preparation.detectors.full_signal import FullSignalDetector


DetectorFactory = Callable[[Mapping[str, Any]], SignalDetector]


def _full_signal_factory(options: Mapping[str, Any]) -> SignalDetector:
    return FullSignalDetector(**dict(options))


def _energy_v1_factory(options: Mapping[str, Any]) -> SignalDetector:
    return EnergyDetectorV1(EnergyDetectorV1Config(**dict(options)))


def _ble_packet_v1_factory(options: Mapping[str, Any]) -> SignalDetector:
    return BlePacketDetectorV1(BlePacketDetectorV1Config(**dict(options)))


def _fixed_blocks_factory(options: Mapping[str, Any]) -> SignalDetector:
    return FixedBlockDetector(**dict(options))


DETECTOR_REGISTRY: dict[str, DetectorFactory] = {
    "full_signal": _full_signal_factory,
    "energy_v1": _energy_v1_factory,
    "ble_packet_v1": _ble_packet_v1_factory,
    "fixed_blocks": _fixed_blocks_factory,
}


def available_detectors() -> tuple[str, ...]:
    """Return stable detector names accepted by the public preparation API."""

    return tuple(DETECTOR_REGISTRY)


def build_detector(
    name: str,
    options: Mapping[str, Any] | None = None,
) -> SignalDetector:
    """Construct a detector without coupling it to any source-file reader."""

    detector_name = str(name).strip().lower()
    try:
        factory = DETECTOR_REGISTRY[detector_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown detector {name!r}. Available: {available_detectors()}"
        ) from exc
    try:
        return factory(dict(options or {}))
    except TypeError as exc:
        raise ValueError(
            f"Invalid options for detector {detector_name!r}: {exc}"
        ) from exc


__all__ = [
    "DETECTOR_REGISTRY",
    "DetectorFactory",
    "available_detectors",
    "build_detector",
]
