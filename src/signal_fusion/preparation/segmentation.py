"""Format-independent region filtering, padding, and merging."""

from __future__ import annotations

from collections.abc import Iterable

from signal_fusion.preparation.contracts import PreparationConfig, SignalRegion


def _merge_pair(left: SignalRegion, right: SignalRegion) -> SignalRegion:
    detectors = []
    for detector in (left.detector, right.detector):
        if detector not in detectors:
            detectors.append(detector)
    scores = [score for score in (left.score, right.score) if score is not None]
    metadata = dict(left.metadata)
    metadata["merged_region_count"] = int(
        metadata.get("merged_region_count", 1)
    ) + int(right.metadata.get("merged_region_count", 1))
    return SignalRegion(
        start_sample=min(left.start_sample, right.start_sample),
        end_sample=max(left.end_sample, right.end_sample),
        detector="+".join(detectors),
        score=max(scores) if scores else None,
        metadata=metadata,
    )


def segment_regions(
    regions: Iterable[SignalRegion],
    *,
    recording_sample_count: int,
    config: PreparationConfig,
    preserve_region_boundaries: bool = False,
) -> list[SignalRegion]:
    """Clamp, filter, pad, sort, and merge detector output regions."""

    total = int(recording_sample_count)
    if total < 0:
        raise ValueError("recording_sample_count must not be negative")

    prepared: list[SignalRegion] = []
    for region in regions:
        raw_start = max(0, int(region.start_sample))
        raw_end = min(total, int(region.end_sample))
        if raw_start >= raw_end:
            continue
        if raw_end - raw_start < config.min_region_samples:
            continue
        start = max(0, raw_start - config.pad_before_samples)
        end = min(total, raw_end + config.pad_after_samples)
        metadata = dict(region.metadata)
        metadata.setdefault("detected_start_sample", raw_start)
        metadata.setdefault("detected_end_sample", raw_end)
        metadata.setdefault("merged_region_count", 1)
        prepared.append(
            SignalRegion(
                start_sample=start,
                end_sample=end,
                detector=region.detector,
                score=region.score,
                metadata=metadata,
            )
        )

    prepared.sort(key=lambda region: (region.start_sample, region.end_sample))
    merged: list[SignalRegion] = []
    for region in prepared:
        if not merged:
            merged.append(region)
            continue
        previous = merged[-1]
        gap = region.start_sample - previous.end_sample
        touching_preserved = preserve_region_boundaries and gap == 0
        if gap <= config.merge_gap_samples and not touching_preserved:
            merged[-1] = _merge_pair(previous, region)
        else:
            merged.append(region)
    return merged


__all__ = ["segment_regions"]
