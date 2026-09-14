"""Detector interface independent of raw file format."""

from __future__ import annotations

from abc import ABC, abstractmethod

from signal_fusion.preparation.contracts import RawSignal, SignalRegion


class SignalDetector(ABC):
    """Locate candidate signal regions in one raw recording."""

    name: str
    preserve_region_boundaries: bool = False

    @abstractmethod
    def detect(self, signal: RawSignal) -> list[SignalRegion]:
        raise NotImplementedError
