"""Evaluation utilities for trained signal classifiers."""

from .snr import (
    add_complex_awgn,
    classification_metrics,
    standardize_iq_windows,
)


def evaluate_snr_robustness(*args, **kwargs):
    """Load the optional PyTorch runtime only when evaluation is requested."""

    from .snr_runner import evaluate_snr_robustness as _evaluate

    return _evaluate(*args, **kwargs)


__all__ = [
    "add_complex_awgn",
    "classification_metrics",
    "evaluate_snr_robustness",
    "standardize_iq_windows",
]
