"""Training-time augmentation for complex IQ windows."""

from __future__ import annotations

import math

import torch


def _validate_iq(inputs: torch.Tensor) -> None:
    if inputs.ndim != 3 or inputs.shape[1] != 2 or inputs.shape[2] == 0:
        raise ValueError(f"IQ must have shape [N, 2, L], got {tuple(inputs.shape)}")
    if not inputs.is_floating_point():
        raise TypeError("IQ inputs must be a floating-point tensor")


def _standardize_iq(inputs: torch.Tensor, epsilon: float) -> torch.Tensor:
    standardized = inputs - inputs.mean(dim=2, keepdim=True)
    rms = torch.sqrt(standardized.square().sum(dim=1).mean(dim=1))
    if torch.any(rms <= epsilon):
        raise ValueError("augmentation produced zero-energy IQ windows")
    return standardized / rms[:, None, None]


def add_random_frequency_shift(
    inputs: torch.Tensor,
    *,
    probability: float,
    max_shift_fraction: float,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Randomly translate complex spectra during training.

    ``max_shift_fraction`` is measured in cycles per sample, equivalently as a
    fraction of the sampling rate.  A value of ``0.1`` samples one independent
    shift per selected window from ``[-0.1 Fs, +0.1 Fs]``.  Selected windows are
    DC-removed and complex-RMS normalized after translation.
    """

    probability = float(probability)
    max_shift_fraction = float(max_shift_fraction)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and between 0 and 1")
    if (
        not math.isfinite(max_shift_fraction)
        or not 0.0 <= max_shift_fraction <= 0.5
    ):
        raise ValueError("max_shift_fraction must be finite and within [0, 0.5]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if probability == 0.0 or max_shift_fraction == 0.0:
        return inputs
    _validate_iq(inputs)

    selected = torch.rand(inputs.shape[0], device=inputs.device) < probability
    if not torch.any(selected):
        return inputs

    values = inputs[selected]
    shifts = torch.empty(
        values.shape[0], device=inputs.device, dtype=inputs.dtype
    ).uniform_(-max_shift_fraction, max_shift_fraction)
    sample_index = torch.arange(
        inputs.shape[2], device=inputs.device, dtype=inputs.dtype
    )
    angles = 2.0 * math.pi * shifts[:, None] * sample_index[None, :]
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    shifted = torch.stack(
        (
            values[:, 0] * cosine - values[:, 1] * sine,
            values[:, 0] * sine + values[:, 1] * cosine,
        ),
        dim=1,
    )
    shifted = _standardize_iq(shifted, epsilon)

    augmented = inputs.clone()
    augmented[selected] = shifted
    return augmented


def add_random_spectral_inversion(
    inputs: torch.Tensor,
    *,
    probability: float,
) -> torch.Tensor:
    """Randomly conjugate complex IQ windows during training."""

    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and between 0 and 1")
    if probability == 0.0:
        return inputs
    _validate_iq(inputs)

    selected = torch.rand(inputs.shape[0], device=inputs.device) < probability
    if not torch.any(selected):
        return inputs
    augmented = inputs.clone()
    augmented[selected, 1] = -augmented[selected, 1]
    return augmented


def add_random_complex_awgn(
    inputs: torch.Tensor,
    *,
    probability: float,
    snr_min_db: float,
    snr_max_db: float,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Add exact per-window complex AWGN to a random part of one batch.

    ``probability`` is evaluated independently for every window.  Selected
    windows receive an SNR sampled uniformly between ``snr_min_db`` and
    ``snr_max_db``.  The noisy windows are then DC-removed
    and complex-RMS normalized, matching the prepared training data contract.
    """

    probability = float(probability)
    snr_min_db = float(snr_min_db)
    snr_max_db = float(snr_max_db)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and between 0 and 1")
    if not math.isfinite(snr_min_db) or not math.isfinite(snr_max_db):
        raise ValueError("SNR bounds must be finite")
    if snr_min_db > snr_max_db:
        raise ValueError("snr_min_db must not exceed snr_max_db")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if probability == 0.0:
        return inputs
    _validate_iq(inputs)

    selected = torch.rand(inputs.shape[0], device=inputs.device) < probability
    if not torch.any(selected):
        return inputs

    values = inputs[selected]
    signal_power = values.square().sum(dim=1).mean(dim=1)
    if torch.any(signal_power <= epsilon):
        raise ValueError("cannot add controlled noise to zero-energy IQ windows")

    noise = torch.randn_like(values)
    raw_noise_power = noise.square().sum(dim=1).mean(dim=1)
    if snr_min_db == snr_max_db:
        snr_db = torch.full_like(signal_power, snr_min_db)
    else:
        snr_db = torch.empty_like(signal_power).uniform_(snr_min_db, snr_max_db)
    target_ratio = torch.pow(10.0, snr_db / 10.0)
    scale = torch.sqrt(signal_power / (target_ratio * raw_noise_power))

    noisy = _standardize_iq(values + noise * scale[:, None, None], epsilon)

    augmented = inputs.clone()
    augmented[selected] = noisy
    return augmented


__all__ = [
    "add_random_complex_awgn",
    "add_random_frequency_shift",
    "add_random_spectral_inversion",
]
