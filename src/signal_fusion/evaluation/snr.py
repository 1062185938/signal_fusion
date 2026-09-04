"""Numerical building blocks for controlled complex-AWGN evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _validated_iq(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"IQ must have shape [N, 2, L], got {values.shape}")
    if values.shape[0] == 0 or values.shape[2] == 0:
        raise ValueError("IQ must contain at least one non-empty window")
    if not np.all(np.isfinite(values)):
        raise ValueError("IQ contains non-finite values")
    return values.astype(np.float64, copy=False)


def standardize_iq_windows(
    x: np.ndarray,
    *,
    remove_dc: bool = True,
    rms_normalize: bool = True,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Apply the same per-window DC removal and complex RMS normalization as training."""

    values = _validated_iq(x).copy()
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if remove_dc:
        values -= values.mean(axis=2, keepdims=True)
    if rms_normalize:
        rms = np.sqrt(np.mean(np.square(values).sum(axis=1), axis=1))
        invalid = np.flatnonzero(rms <= epsilon)
        if invalid.size:
            raise ValueError(
                "IQ contains zero-energy windows after DC removal; "
                f"first indices={invalid[:10].tolist()}"
            )
        values /= rms[:, np.newaxis, np.newaxis]
    standardized = values.astype(np.float32)
    if not np.all(np.isfinite(standardized)):
        raise ValueError("IQ standardization produced non-finite values")
    return standardized


def add_complex_awgn(
    x: np.ndarray,
    snr_db: float,
    *,
    rng: np.random.Generator,
    remove_dc: bool = True,
    rms_normalize: bool = True,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Add circular complex AWGN at an exact per-window incremental SNR.

    The input already contains acquisition noise.  Therefore ``snr_db`` describes
    the ratio between the complete input window power and newly added noise, not
    an estimate of the physical over-the-air SNR.
    """

    values = _validated_iq(x)
    snr_db = float(snr_db)
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")

    signal_power = np.mean(np.square(values).sum(axis=1), axis=1)
    invalid = np.flatnonzero(signal_power <= epsilon)
    if invalid.size:
        raise ValueError(
            "cannot add controlled noise to zero-energy windows; "
            f"first indices={invalid[:10].tolist()}"
        )

    noise = rng.standard_normal(values.shape)
    raw_noise_power = np.mean(np.square(noise).sum(axis=1), axis=1)
    target_ratio = 10.0 ** (snr_db / 10.0)
    scale = np.sqrt(signal_power / (target_ratio * raw_noise_power))
    noise *= scale[:, np.newaxis, np.newaxis]
    achieved_snr_db = 10.0 * np.log10(
        signal_power / np.mean(np.square(noise).sum(axis=1), axis=1)
    )
    noisy = standardize_iq_windows(
        values + noise,
        remove_dc=remove_dc,
        rms_normalize=rms_normalize,
        epsilon=epsilon,
    )
    return noisy, achieved_snr_db.astype(np.float64)


def _confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, class_num: int
) -> list[list[int]]:
    return [
        [
            int(np.count_nonzero((labels == true_label) & (predictions == predicted)))
            for predicted in range(class_num)
        ]
        for true_label in range(class_num)
    ]


def _accuracy_summary(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_map: dict[str, str],
) -> dict[str, Any]:
    class_num = probabilities.shape[1]
    correct = predictions == labels
    confidence = probabilities[np.arange(len(predictions)), predictions]
    per_class: dict[str, dict[str, Any]] = {}
    for label in range(class_num):
        mask = labels == label
        label_count = int(np.count_nonzero(mask))
        per_class[str(label)] = {
            "class_name": label_map[str(label)],
            "sample_count": label_count,
            "correct_count": int(np.count_nonzero(correct[mask])),
            "accuracy_percent": (
                float(np.mean(correct[mask]) * 100.0)
                if label_count
                else None
            ),
        }
    return {
        "sample_count": int(len(labels)),
        "correct_count": int(np.count_nonzero(correct)),
        "error_count": int(np.count_nonzero(~correct)),
        "accuracy_percent": float(np.mean(correct) * 100.0),
        "mean_predicted_confidence": float(np.mean(confidence)),
        "confusion_matrix": _confusion_matrix(labels, predictions, class_num),
        "per_class": per_class,
    }


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    source_ids: np.ndarray,
    label_map: dict[str, str],
) -> dict[str, Any]:
    """Compute window, source, and probability-averaged region metrics."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    group_ids = np.asarray(group_ids)
    source_ids = np.asarray(source_ids).astype(str)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must have shape [N, class_num]")
    sample_count, class_num = probabilities.shape
    for name, values in (
        ("labels", labels),
        ("group_ids", group_ids),
        ("source_ids", source_ids),
    ):
        if values.shape != (sample_count,):
            raise ValueError(f"{name} must have shape [{sample_count}]")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(labels < 0) or np.any(labels >= class_num):
        raise ValueError("labels fall outside the probability class range")
    labels = labels.astype(np.int64, copy=False)
    if set(label_map) != {str(index) for index in range(class_num)}:
        raise ValueError("label_map must contain continuous string indices")

    predictions = probabilities.argmax(axis=1).astype(np.int64)
    result = {
        "window": _accuracy_summary(
            labels, predictions, probabilities, label_map
        )
    }

    per_source: dict[str, dict[str, Any]] = {}
    for source_id in sorted(set(source_ids.tolist())):
        mask = source_ids == source_id
        source_labels = np.unique(labels[mask])
        if source_labels.size != 1:
            raise ValueError(f"source_id={source_id!r} contains multiple labels")
        per_source[source_id] = {
            "label": int(source_labels[0]),
            "class_name": label_map[str(int(source_labels[0]))],
            **_accuracy_summary(
                labels[mask], predictions[mask], probabilities[mask], label_map
            ),
        }
    result["per_source"] = per_source

    group_labels: list[int] = []
    group_probabilities: list[np.ndarray] = []
    group_sources: list[str] = []
    for group_id in sorted(np.unique(group_ids).tolist()):
        mask = group_ids == group_id
        unique_labels = np.unique(labels[mask])
        unique_sources = np.unique(source_ids[mask])
        if unique_labels.size != 1 or unique_sources.size != 1:
            raise ValueError(
                f"group_id={group_id!r} must belong to one label and source"
            )
        group_labels.append(int(unique_labels[0]))
        group_probabilities.append(probabilities[mask].mean(axis=0))
        group_sources.append(str(unique_sources[0]))

    grouped_probabilities = np.asarray(group_probabilities, dtype=np.float64)
    grouped_labels = np.asarray(group_labels, dtype=np.int64)
    grouped_predictions = grouped_probabilities.argmax(axis=1).astype(np.int64)
    result["group"] = _accuracy_summary(
        grouped_labels,
        grouped_predictions,
        grouped_probabilities,
        label_map,
    )
    result["group"]["vote_method"] = "mean_probability"
    result["group"]["source_count"] = len(set(group_sources))
    return result


__all__ = [
    "add_complex_awgn",
    "classification_metrics",
    "standardize_iq_windows",
]
