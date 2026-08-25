"""Fixed-length window coordinate generation."""

from __future__ import annotations

from dataclasses import dataclass

from signal_fusion.preparation.contracts import PreparationConfig, SignalRegion


@dataclass(frozen=True, slots=True)
class WindowSpan:
    """One window's absolute source coordinates and valid sample count."""

    start_sample: int
    end_sample: int
    valid_samples: int


def window_spans(
    region: SignalRegion, config: PreparationConfig
) -> list[WindowSpan]:
    """Generate windows without reading or modifying IQ samples."""

    spans: list[WindowSpan] = []
    start = region.start_sample
    if config.remainder == "drop":
        last_start = region.end_sample - config.seq_len
        while start <= last_start:
            spans.append(
                WindowSpan(
                    start_sample=start,
                    end_sample=start + config.seq_len,
                    valid_samples=config.seq_len,
                )
            )
            start += config.hop_len
        return spans

    while start < region.end_sample:
        end = min(start + config.seq_len, region.end_sample)
        spans.append(
            WindowSpan(
                start_sample=start,
                end_sample=end,
                valid_samples=end - start,
            )
        )
        start += config.hop_len
    return spans


__all__ = ["WindowSpan", "window_spans"]

