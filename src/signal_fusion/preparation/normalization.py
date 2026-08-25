"""Window-level IQ normalization independent of detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NormalizedIQ:
    samples: np.ndarray
    scale: float
    dc_offset: complex


def normalize_iq(
    samples: np.ndarray,
    *,
    mode: str = "none",
    remove_dc: bool = False,
) -> NormalizedIQ:
    """Normalize one non-empty complex IQ window and report applied values."""

    iq = np.asarray(samples, dtype=np.complex64).reshape(-1).copy()
    if iq.size == 0:
        raise ValueError("Cannot normalize an empty IQ window")
    if not np.isfinite(iq.real).all() or not np.isfinite(iq.imag).all():
        raise ValueError("IQ window contains NaN or Inf")

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"none", "rms", "peak"}:
        raise ValueError("mode must be 'none', 'rms', or 'peak'")

    dc_offset = complex(np.mean(iq, dtype=np.complex128)) if remove_dc else 0j
    if remove_dc:
        iq -= np.complex64(dc_offset)

    if normalized_mode == "rms":
        scale = float(np.sqrt(np.mean(np.abs(iq) ** 2, dtype=np.float64)))
    elif normalized_mode == "peak":
        scale = float(np.max(np.abs(iq)))
    else:
        scale = 1.0
    if not np.isfinite(scale):
        raise ValueError("Normalization scale is not finite")
    if scale <= np.finfo(np.float32).eps:
        scale = 1.0
    if normalized_mode != "none":
        iq /= np.float32(scale)

    return NormalizedIQ(
        samples=iq.astype(np.complex64, copy=False),
        scale=scale,
        dc_offset=dc_offset,
    )


__all__ = ["NormalizedIQ", "normalize_iq"]

