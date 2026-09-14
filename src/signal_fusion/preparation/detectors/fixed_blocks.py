"""Deterministic fixed-length region selection for continuous recordings."""

from __future__ import annotations

from dataclasses import dataclass, field

from signal_fusion.preparation.contracts import RawSignal, SignalRegion
from signal_fusion.preparation.detectors.base import SignalDetector


def _uniform_indices(total: int, requested: int) -> tuple[int, ...]:
    if requested > total:
        raise ValueError(
            f"cannot select {requested} fixed blocks from only {total}"
        )
    if requested == total:
        return tuple(range(total))
    return tuple(
        ((2 * index + 1) * total) // (2 * requested)
        for index in range(requested)
    )


@dataclass(slots=True)
class FixedBlockDetector(SignalDetector):
    """Return complete blocks, optionally sampled uniformly across a recording."""

    block_size_samples: int = 4096
    block_count: int | None = None
    name: str = field(default="fixed_blocks", init=False)
    preserve_region_boundaries: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.block_size_samples = int(self.block_size_samples)
        if self.block_size_samples <= 0:
            raise ValueError("block_size_samples must be positive")
        if self.block_count is not None:
            self.block_count = int(self.block_count)
            if self.block_count <= 0:
                raise ValueError("block_count must be positive or None")

    def detect(self, signal: RawSignal) -> list[SignalRegion]:
        total_blocks = signal.sample_count // self.block_size_samples
        requested = total_blocks if self.block_count is None else self.block_count
        indices = _uniform_indices(total_blocks, requested)
        return [
            SignalRegion(
                start_sample=block_index * self.block_size_samples,
                end_sample=(block_index + 1) * self.block_size_samples,
                detector=self.name,
                score=1.0,
                metadata={"source_block_index": block_index},
            )
            for block_index in indices
        ]


__all__ = ["FixedBlockDetector"]
