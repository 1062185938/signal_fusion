"""Training-time augmentation for complex IQ windows."""

from __future__ import annotations

import math

import torch


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
    if inputs.ndim != 3 or inputs.shape[1] != 2 or inputs.shape[2] == 0:
        raise ValueError(f"IQ must have shape [N, 2, L], got {tuple(inputs.shape)}")
    if not inputs.is_floating_point():
        raise TypeError("IQ inputs must be a floating-point tensor")

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

    noisy = values + noise * scale[:, None, None]
    noisy = noisy - noisy.mean(dim=2, keepdim=True)
    rms = torch.sqrt(noisy.square().sum(dim=1).mean(dim=1))
    if torch.any(rms <= epsilon):
        raise ValueError("AWGN augmentation produced zero-energy IQ windows")
    noisy = noisy / rms[:, None, None]

    augmented = inputs.clone()
    augmented[selected] = noisy
    return augmented


__all__ = ["add_random_complex_awgn"]
