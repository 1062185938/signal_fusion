"""Conservative LTE/DVB-T correction gate for IQ-model ensembles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.feature_extraction import (
    PERIODICITY_SCHEMA_ID,
    PeriodicityCandidate,
    measure_periodicity_batch,
)


LTE_LABEL = 0
WIFI_LABEL = 1
DVBT_LABEL = 2
TECHNOLOGY_PERIODICITY_CANDIDATES = (
    PeriodicityCandidate("period_66p67us", 1.0 / 15_000.0),
    PeriodicityCandidate("period_224us", 224e-6),
    PeriodicityCandidate("period_896us", 896e-6),
)


def _constant_sample_rate(dataset: PreparedDataset) -> float:
    if "sample_rate" not in dataset.meta:
        raise ValueError("dataset is missing sample_rate metadata")
    values = np.asarray(dataset.meta["sample_rate"], dtype=np.float64).reshape(-1)
    unique = np.unique(values)
    if unique.size != 1 or not np.isfinite(unique[0]) or unique[0] <= 0.0:
        raise ValueError("dataset must use one finite positive sample rate")
    return float(unique[0])


def score_technology_periodicity(
    dataset: PreparedDataset,
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Measure the frozen LTE/DVB-T candidate periods for each complete region."""

    if dataset.X.ndim != 3 or dataset.X.shape[1] != 2:
        raise ValueError(f"expected X shape [N, 2, seq_len], got {dataset.X.shape}")
    sample_rate = _constant_sample_rate(dataset)
    signals = (
        np.asarray(dataset.X[:, 0, :], dtype=np.float32)
        + 1j * np.asarray(dataset.X[:, 1, :], dtype=np.float32)
    ).astype(np.complex64, copy=False)
    measured = measure_periodicity_batch(
        signals,
        sample_rate,
        TECHNOLOGY_PERIODICITY_CANDIDATES,
        search_radius_samples=2,
        batch_size=batch_size,
    )
    correlations = measured.normalized_correlations
    lte_score = correlations[:, 0]
    dvbt_2k_score = correlations[:, 1]
    dvbt_8k_score = correlations[:, 2]
    dvbt_score = np.maximum(dvbt_2k_score, dvbt_8k_score)
    prediction = np.where(lte_score >= dvbt_score, LTE_LABEL, DVBT_LABEL).astype(
        np.int64
    )
    return {
        "sample_rate": sample_rate,
        "measurement": measured,
        "correlations": correlations,
        "peak_lags": measured.peak_lags,
        "lte_score": lte_score,
        "dvbt_2k_score": dvbt_2k_score,
        "dvbt_8k_score": dvbt_8k_score,
        "dvbt_score": dvbt_score,
        "prediction": prediction,
        "margin": np.abs(lte_score - dvbt_score).astype(np.float32),
    }


def load_periodicity_gate_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], float]:
    """Load the frozen technology-specific gate and validate its exact contract."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"periodicity gate manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("periodicity gate manifest schema_version must be 1")
    if payload.get("result_type") != "technology_recognition_periodicity_gate":
        raise ValueError("unexpected periodicity gate manifest result_type")
    if tuple(payload.get("labels", ())) != ("LTE", "WiFi", "DVB-T"):
        raise ValueError("periodicity gate manifest has an unexpected label order")
    if payload.get("sample_rate_hz") != 1_000_000:
        raise ValueError("periodicity gate manifest requires a 1 MS/s sample rate")
    if payload.get("region_sample_count") != 4096:
        raise ValueError("periodicity gate manifest requires 4096-sample regions")
    if payload.get("periodicity_schema_id") != PERIODICITY_SCHEMA_ID:
        raise ValueError("periodicity gate manifest schema does not match the code")

    expected_candidates = [
        (item.candidate_id, float(item.period_seconds))
        for item in TECHNOLOGY_PERIODICITY_CANDIDATES
    ]
    recorded_candidates = [
        (str(item["candidate_id"]), float(item["period_seconds"]))
        for item in payload.get("candidates", ())
    ]
    if recorded_candidates != expected_candidates:
        raise ValueError("periodicity gate candidates do not match the frozen code")
    try:
        margin = float(payload["frozen_rule"]["reliability_margin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "periodicity gate manifest has no valid reliability margin"
        ) from exc
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(
            "periodicity gate reliability margin must be finite and non-negative"
        )
    return payload, margin


def _label_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {array.shape}")
    return array


def lte_dvbt_disagreement_mask(
    member_predictions: np.ndarray,
    ensemble_predictions: np.ndarray,
) -> np.ndarray:
    """Select only ensemble conflicts containing both LTE and DVB-T.

    Any region with a WiFi member vote is excluded.  An ensemble WiFi decision
    is also excluded, so this branch can never change a WiFi decision.
    """

    members = np.asarray(member_predictions, dtype=np.int64)
    ensemble = _label_vector(ensemble_predictions, "ensemble_predictions")
    if members.ndim != 2 or members.shape[0] != ensemble.size:
        raise ValueError(
            "member_predictions must have shape [N, M] matching the ensemble"
        )
    if members.shape[1] < 2:
        raise ValueError("periodicity gating requires at least two members")

    valid_member = np.all(np.isin(members, (LTE_LABEL, DVBT_LABEL)), axis=1)
    has_lte = np.any(members == LTE_LABEL, axis=1)
    has_dvbt = np.any(members == DVBT_LABEL, axis=1)
    valid_ensemble = np.isin(ensemble, (LTE_LABEL, DVBT_LABEL))
    return valid_member & has_lte & has_dvbt & valid_ensemble


def select_periodicity_gate_margin(
    true_labels: np.ndarray,
    periodicity_predictions: np.ndarray,
    periodicity_margins: np.ndarray,
) -> float:
    """Fit a reject threshold on all labeled LTE/DVB-T validation regions."""

    truth = _label_vector(true_labels, "true_labels")
    periodic = _label_vector(periodicity_predictions, "periodicity_predictions")
    margins = np.asarray(periodicity_margins, dtype=np.float64)
    if periodic.shape != truth.shape or margins.shape != truth.shape:
        raise ValueError("labels, periodicity predictions, and margins must align")
    if not np.all(np.isfinite(margins)) or np.any(margins < 0.0):
        raise ValueError("periodicity_margins must be finite and non-negative")

    eligible = np.isin(truth, (LTE_LABEL, DVBT_LABEL))
    if not np.any(eligible):
        raise ValueError("validation data contains no LTE/DVB-T regions")
    wrong = eligible & (periodic != truth)
    if not np.any(wrong):
        return 0.0
    return float(np.nextafter(np.max(margins[wrong]), np.inf))


def apply_periodicity_gate(
    member_predictions: np.ndarray,
    ensemble_predictions: np.ndarray,
    periodicity_predictions: np.ndarray,
    periodicity_margins: np.ndarray,
    reliability_margin: float,
) -> dict[str, np.ndarray]:
    """Apply a frozen periodicity threshold without changing other regions."""

    ensemble = _label_vector(ensemble_predictions, "ensemble_predictions")
    periodic = _label_vector(periodicity_predictions, "periodicity_predictions")
    margins = np.asarray(periodicity_margins, dtype=np.float64)
    threshold = float(reliability_margin)
    if periodic.shape != ensemble.shape or margins.shape != ensemble.shape:
        raise ValueError("ensemble, periodicity predictions, and margins must align")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("reliability_margin must be finite and non-negative")
    if not np.all(np.isfinite(margins)) or np.any(margins < 0.0):
        raise ValueError("periodicity_margins must be finite and non-negative")

    eligible = lte_dvbt_disagreement_mask(member_predictions, ensemble)
    reliable = margins >= threshold
    resolved = eligible & reliable
    fused = ensemble.copy()
    fused[resolved] = periodic[resolved]
    changed = resolved & (fused != ensemble)
    return {
        "eligible": eligible,
        "reliable": reliable,
        "resolved": resolved,
        "changed": changed,
        "prediction": fused,
    }


def summarize_periodicity_gate(
    true_labels: np.ndarray,
    member_predictions: np.ndarray,
    ensemble_predictions: np.ndarray,
    gated: dict[str, Any],
) -> dict[str, Any]:
    """Return compact correction and remaining-review metrics."""

    truth = _label_vector(true_labels, "true_labels")
    ensemble = _label_vector(ensemble_predictions, "ensemble_predictions")
    fused = _label_vector(np.asarray(gated["prediction"]), "prediction")
    if truth.shape != ensemble.shape or fused.shape != truth.shape:
        raise ValueError("truth, ensemble, and fused predictions must align")

    members = np.asarray(member_predictions, dtype=np.int64)
    if members.ndim != 2 or members.shape[0] != truth.size:
        raise ValueError("member_predictions must have shape [N, M]")
    member_disagreement = np.any(members != members[:, :1], axis=1)
    resolved = np.asarray(gated["resolved"], dtype=bool)
    changed = np.asarray(gated["changed"], dtype=bool)
    eligible = np.asarray(gated["eligible"], dtype=bool)
    for name, values in (
        ("eligible", eligible),
        ("resolved", resolved),
        ("changed", changed),
    ):
        if values.shape != truth.shape:
            raise ValueError(f"{name} must have shape [N]")

    baseline_correct = ensemble == truth
    fused_correct = fused == truth
    corrected = ~baseline_correct & fused_correct
    introduced = baseline_correct & ~fused_correct
    remaining_review = member_disagreement & ~resolved
    accepted = ~remaining_review

    count = int(truth.size)
    baseline_correct_count = int(np.sum(baseline_correct))
    fused_correct_count = int(np.sum(fused_correct))
    accepted_count = int(np.sum(accepted))
    accepted_correct_count = int(np.sum(accepted & fused_correct))
    return {
        "region_count": count,
        "baseline_ensemble": {
            "correct_count": baseline_correct_count,
            "error_count": count - baseline_correct_count,
            "accuracy_percent": 100.0 * baseline_correct_count / count,
        },
        "member_disagreement_count": int(np.sum(member_disagreement)),
        "lte_dvbt_gate_candidate_count": int(np.sum(eligible)),
        "periodicity_resolved_count": int(np.sum(resolved)),
        "changed_label_count": int(np.sum(changed)),
        "corrected_error_count": int(np.sum(corrected)),
        "introduced_error_count": int(np.sum(introduced)),
        "net_correction_count": int(np.sum(corrected) - np.sum(introduced)),
        "fused": {
            "correct_count": fused_correct_count,
            "error_count": count - fused_correct_count,
            "accuracy_percent": 100.0 * fused_correct_count / count,
        },
        "review_required": {
            "region_count": int(np.sum(remaining_review)),
            "coverage_percent": 100.0 * float(np.mean(remaining_review)),
        },
        "accepted": {
            "region_count": accepted_count,
            "coverage_percent": 100.0 * accepted_count / count,
            "correct_count": accepted_correct_count,
            "error_count": accepted_count - accepted_correct_count,
            "accuracy_percent": (
                100.0 * accepted_correct_count / accepted_count
                if accepted_count
                else None
            ),
        },
    }


__all__ = [
    "DVBT_LABEL",
    "LTE_LABEL",
    "TECHNOLOGY_PERIODICITY_CANDIDATES",
    "WIFI_LABEL",
    "apply_periodicity_gate",
    "load_periodicity_gate_manifest",
    "lte_dvbt_disagreement_mask",
    "score_technology_periodicity",
    "select_periodicity_gate_margin",
    "summarize_periodicity_gate",
]
