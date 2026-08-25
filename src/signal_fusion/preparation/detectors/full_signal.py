"""Detector that treats a configured recording interval as signal."""

from __future__ import annotations

from dataclasses import dataclass

from signal_fusion.preparation.contracts import RawSignal, SignalRegion
from signal_fusion.preparation.detectors.base import SignalDetector


@dataclass(slots=True)
class FullSignalDetector(SignalDetector):
    """Select the complete recording or one explicit half-open interval."""

    start_sample: int = 0
    end_sample: int | None = None
    name: str = "full_signal"

    def detect(self, signal: RawSignal) -> list[SignalRegion]:
        start = int(self.start_sample)
        end = signal.sample_count if self.end_sample is None else int(self.end_sample)
        if start < 0 or start > signal.sample_count:
            raise ValueError(
                f"start_sample must be within the recording, got {start}"
            )
        if end <= start or end > signal.sample_count:
            raise ValueError(
                f"end_sample must be within ({start}, {signal.sample_count}], got {end}"
            )
        return [
            SignalRegion(
                start_sample=start,
                end_sample=end,
                detector=self.name,
                score=1.0,
            )
        ]

