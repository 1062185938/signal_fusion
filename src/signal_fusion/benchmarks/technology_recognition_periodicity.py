"""Location-isolated OFDM periodicity evaluation for LTE and DVB-T."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.feature_extraction import PERIODICITY_SCHEMA_ID
from signal_fusion.fusion.periodicity_gate import (
    DVBT_LABEL,
    LTE_LABEL,
    TECHNOLOGY_PERIODICITY_CANDIDATES,
    WIFI_LABEL,
    score_technology_periodicity,
)
from signal_fusion.io import load_prepared_dataset
from signal_fusion.io.writers import json_safe


LABEL_NAMES = {LTE_LABEL: "LTE", WIFI_LABEL: "WiFi", DVBT_LABEL: "DVB-T"}


def _metadata_vector(dataset: PreparedDataset, name: str) -> np.ndarray:
    if name not in dataset.meta:
        raise ValueError(f"dataset is missing {name} metadata")
    values = np.asarray(dataset.meta[name])
    if values.shape != (dataset.num_samples,):
        raise ValueError(f"{name} must contain one value per region")
    return values


def _source_location(source_id: str) -> str:
    parts = str(source_id).split("_")
    if len(parts) < 3 or parts[0] != "tr":
        raise ValueError(f"cannot resolve location from source_id={source_id!r}")
    return parts[1]


def score_periodicity_dataset(
    dataset: PreparedDataset,
    *,
    batch_size: int = 256,
) -> dict[str, np.ndarray | float]:
    """Measure the three frozen physical periods for one 4096-region dataset."""

    if dataset.y is None:
        raise ValueError("periodicity evaluation requires labels")
    scores = score_technology_periodicity(dataset, batch_size=batch_size)
    return {
        "sample_rate": scores["sample_rate"],
        "y": np.asarray(dataset.y, dtype=np.int64),
        "group_id": _metadata_vector(dataset, "group_id").astype(np.int64),
        "source_id": _metadata_vector(dataset, "sample_source_id").astype(str),
        **{
            name: scores[name]
            for name in (
                "correlations",
                "peak_lags",
                "lte_score",
                "dvbt_2k_score",
                "dvbt_8k_score",
                "dvbt_score",
                "prediction",
                "margin",
            )
        },
    }


def select_reliability_margin(scores: dict[str, Any]) -> float:
    """Select the smallest margin that rejects every validation error.

    The threshold is fit only on labeled LTE/DVB-T validation regions.  It is
    intentionally a reject option, not a learned classifier.
    """

    y = np.asarray(scores["y"], dtype=np.int64)
    prediction = np.asarray(scores["prediction"], dtype=np.int64)
    margin = np.asarray(scores["margin"], dtype=np.float64)
    eligible = np.isin(y, (LTE_LABEL, DVBT_LABEL))
    if not np.any(eligible):
        raise ValueError("validation data contains no LTE/DVB-T regions")
    wrong = eligible & (prediction != y)
    if not np.any(wrong):
        return 0.0
    return float(np.nextafter(np.max(margin[wrong]), np.inf))


def summarize_periodicity_scores(
    scores: dict[str, Any],
    reliability_margin: float,
) -> dict[str, Any]:
    """Summarize LTE/DVB-T discrimination and WiFi counterfactual flags."""

    y = np.asarray(scores["y"], dtype=np.int64)
    prediction = np.asarray(scores["prediction"], dtype=np.int64)
    margin = np.asarray(scores["margin"], dtype=np.float64)
    reliable = margin >= float(reliability_margin)
    eligible = np.isin(y, (LTE_LABEL, DVBT_LABEL))
    accepted = eligible & reliable
    accepted_count = int(np.sum(accepted))
    accepted_correct = int(np.sum(accepted & (prediction == y)))
    eligible_count = int(np.sum(eligible))
    full_correct = int(np.sum(eligible & (prediction == y)))
    wifi = y == WIFI_LABEL

    per_class: dict[str, Any] = {}
    for label in (LTE_LABEL, DVBT_LABEL):
        class_mask = y == label
        class_accepted = class_mask & reliable
        class_accepted_count = int(np.sum(class_accepted))
        per_class[str(label)] = {
            "class_name": LABEL_NAMES[label],
            "region_count": int(np.sum(class_mask)),
            "raw_correct_count": int(
                np.sum(class_mask & (prediction == label))
            ),
            "reliable_count": class_accepted_count,
            "reliable_correct_count": int(
                np.sum(class_accepted & (prediction == label))
            ),
        }

    return {
        "reliability_margin": float(reliability_margin),
        "lte_dvbt_region_count": eligible_count,
        "raw_correct_count": full_correct,
        "raw_accuracy_percent": (
            100.0 * full_correct / eligible_count if eligible_count else None
        ),
        "reliable_count": accepted_count,
        "reliable_coverage_percent": (
            100.0 * accepted_count / eligible_count if eligible_count else None
        ),
        "reliable_correct_count": accepted_correct,
        "reliable_error_count": accepted_count - accepted_correct,
        "reliable_accuracy_percent": (
            100.0 * accepted_correct / accepted_count
            if accepted_count
            else None
        ),
        "insufficient_evidence_count": eligible_count - accepted_count,
        "wifi_region_count": int(np.sum(wifi)),
        "wifi_counterfactual_reliable_count": int(np.sum(wifi & reliable)),
        "wifi_counterfactual_reliable_percent": (
            100.0 * int(np.sum(wifi & reliable)) / int(np.sum(wifi))
            if np.any(wifi)
            else None
        ),
        "per_class": per_class,
    }


def _source_rows(
    fold_name: str,
    scores: dict[str, Any],
    reliability_margin: float,
) -> list[dict[str, Any]]:
    y = np.asarray(scores["y"], dtype=np.int64)
    prediction = np.asarray(scores["prediction"], dtype=np.int64)
    margin = np.asarray(scores["margin"], dtype=np.float64)
    source_ids = np.asarray(scores["source_id"]).astype(str)
    reliable = margin >= reliability_margin
    rows: list[dict[str, Any]] = []
    for source_id in sorted(np.unique(source_ids)):
        mask = source_ids == source_id
        labels = np.unique(y[mask])
        if labels.size != 1:
            raise ValueError(f"source {source_id!r} contains multiple labels")
        label = int(labels[0])
        eligible = label in (LTE_LABEL, DVBT_LABEL)
        accepted = mask & reliable if eligible else np.zeros_like(mask)
        accepted_count = int(np.sum(accepted))
        accepted_correct = int(np.sum(accepted & (prediction == y)))
        rows.append(
            {
                "fold": fold_name,
                "location": _source_location(source_id),
                "source_id": source_id,
                "class_name": LABEL_NAMES[label],
                "region_count": int(np.sum(mask)),
                "raw_binary_accuracy_percent": (
                    100.0 * float(np.mean(prediction[mask] == y[mask]))
                    if eligible
                    else ""
                ),
                "reliable_count": accepted_count if eligible else "",
                "reliable_coverage_percent": (
                    100.0 * accepted_count / int(np.sum(mask)) if eligible else ""
                ),
                "reliable_accuracy_percent": (
                    100.0 * accepted_correct / accepted_count
                    if eligible and accepted_count
                    else ""
                ),
                "wifi_counterfactual_reliable_percent": (
                    100.0 * float(np.mean(reliable[mask]))
                    if not eligible
                    else ""
                ),
            }
        )
    return rows


def _baseline_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    region = payload["ensemble"]["metrics"]["region"]
    risk = payload["risk_gate"]
    return {
        "source_path": str(path.resolve()),
        "labels": payload["labels"],
        "region_count": int(payload["region_count"]),
        "source_count": int(payload["source_count"]),
        "correct_count": int(region["correct_count"]),
        "error_count": int(region["error_count"]),
        "accuracy_percent": float(region["accuracy_percent"]),
        "unanimous_region_count": int(risk["accept"]["region_count"]),
        "unanimous_error_count": int(risk["accept"]["error_count"]),
        "review_region_count": int(risk["review_required"]["region_count"]),
        "review_error_count": int(risk["review_required"]["error_count"]),
    }


def run_location_fold_evaluation(
    fold_root: str | Path,
    baseline_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Run Phase P0-P2 and write compact JSON/CSV/NPZ artifacts."""

    root = Path(fold_root)
    baseline = Path(baseline_path)
    output = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"fold root not found: {root}")
    if not baseline.is_file():
        raise FileNotFoundError(f"baseline not found: {baseline}")
    output.mkdir(parents=True, exist_ok=True)

    fold_summaries: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    prediction_parts: dict[str, list[np.ndarray]] = {
        "fold": [],
        "group_id": [],
        "source_id": [],
        "y": [],
        "correlations": [],
        "peak_lags": [],
        "prediction": [],
        "margin": [],
        "reliable": [],
    }

    for fold_index in range(1, 5):
        fold_name = f"fold{fold_index}"
        fold_dir = root / fold_name
        train_scores = score_periodicity_dataset(
            load_prepared_dataset(fold_dir / "train.npz"),
            batch_size=batch_size,
        )
        validation_scores = score_periodicity_dataset(
            load_prepared_dataset(fold_dir / "validation.npz"),
            batch_size=batch_size,
        )
        reliability_margin = select_reliability_margin(validation_scores)
        test_scores = score_periodicity_dataset(
            load_prepared_dataset(fold_dir / "test.npz"),
            batch_size=batch_size,
        )
        fold_summaries.append(
            {
                "fold": fold_name,
                "threshold_selection": (
                    "smallest absolute LTE/DVB score margin strictly above "
                    "every validation misclassification"
                ),
                "reliability_margin": reliability_margin,
                "train": summarize_periodicity_scores(
                    train_scores, reliability_margin
                ),
                "validation": summarize_periodicity_scores(
                    validation_scores, reliability_margin
                ),
                "test": summarize_periodicity_scores(
                    test_scores, reliability_margin
                ),
            }
        )
        source_rows.extend(
            _source_rows(fold_name, test_scores, reliability_margin)
        )

        count = len(np.asarray(test_scores["y"]))
        prediction_parts["fold"].append(np.full(count, fold_name))
        for name in (
            "group_id",
            "source_id",
            "y",
            "correlations",
            "peak_lags",
            "prediction",
            "margin",
        ):
            prediction_parts[name].append(np.asarray(test_scores[name]))
        prediction_parts["reliable"].append(
            np.asarray(test_scores["margin"]) >= reliability_margin
        )

    test_counts = [item["test"] for item in fold_summaries]
    total_regions = sum(item["lte_dvbt_region_count"] for item in test_counts)
    total_raw_correct = sum(item["raw_correct_count"] for item in test_counts)
    total_reliable = sum(item["reliable_count"] for item in test_counts)
    total_reliable_correct = sum(
        item["reliable_correct_count"] for item in test_counts
    )
    total_wifi = sum(item["wifi_region_count"] for item in test_counts)
    total_wifi_flagged = sum(
        item["wifi_counterfactual_reliable_count"] for item in test_counts
    )

    report = {
        "schema_version": 1,
        "result_type": "technology_recognition_periodicity_phase_p0_p2",
        "scope": {
            "labels": ["LTE", "WiFi", "DVB-T"],
            "sample_rate_hz": 1_000_000,
            "region_sample_count": 4096,
            "periodicity_role": "LTE_vs_DVB-T_physical_evidence_only",
            "wifi_is_not_classified_by_periodicity": True,
        },
        "phase_p0_baseline": _baseline_summary(baseline),
        "phase_p1_measurement": {
            "schema_id": PERIODICITY_SCHEMA_ID,
            "method": "energy_normalized_complex_autocorrelation",
            "search_radius_samples": 2,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "period_seconds": item.period_seconds,
                }
                for item in TECHNOLOGY_PERIODICITY_CANDIDATES
            ],
            "classification_rule": (
                "LTE when period_66p67us exceeds max(period_224us, "
                "period_896us), otherwise DVB-T"
            ),
        },
        "phase_p2_location_isolated_folds": fold_summaries,
        "aggregate_outer_test": {
            "lte_dvbt_region_count": total_regions,
            "raw_correct_count": total_raw_correct,
            "raw_accuracy_percent": 100.0 * total_raw_correct / total_regions,
            "reliable_count": total_reliable,
            "reliable_coverage_percent": 100.0 * total_reliable / total_regions,
            "reliable_correct_count": total_reliable_correct,
            "reliable_error_count": total_reliable - total_reliable_correct,
            "reliable_accuracy_percent": (
                100.0 * total_reliable_correct / total_reliable
                if total_reliable
                else None
            ),
            "wifi_region_count": total_wifi,
            "wifi_counterfactual_reliable_count": total_wifi_flagged,
            "wifi_counterfactual_reliable_percent": (
                100.0 * total_wifi_flagged / total_wifi if total_wifi else None
            ),
        },
        "interpretation_limits": [
            "The four folds are development cross-validation, not a never-used final test.",
            "The periodicity branch does not identify WiFi and must be gated by IQ-model disagreement.",
            "The external 44-file result is retrospective and is not part of this P2 score.",
        ],
    }

    report_path = output / "periodicity_evaluation.json"
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output / "periodicity_by_source.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    prediction_path = output / "periodicity_predictions.npz"
    np.savez_compressed(
        prediction_path,
        **{
            name: np.concatenate(parts, axis=0)
            for name, parts in prediction_parts.items()
        },
        candidate_ids=np.asarray(
            [item.candidate_id for item in TECHNOLOGY_PERIODICITY_CANDIDATES]
        ),
        candidate_period_seconds=np.asarray(
            [item.period_seconds for item in TECHNOLOGY_PERIODICITY_CANDIDATES],
            dtype=np.float64,
        ),
        periodicity_schema_id=np.asarray(PERIODICITY_SCHEMA_ID),
    )
    return {
        "report_path": str(report_path),
        "source_csv_path": str(csv_path),
        "predictions_path": str(prediction_path),
        "report": report,
    }


__all__ = [
    "DVBT_LABEL",
    "LTE_LABEL",
    "TECHNOLOGY_PERIODICITY_CANDIDATES",
    "WIFI_LABEL",
    "run_location_fold_evaluation",
    "score_periodicity_dataset",
    "select_reliability_margin",
    "summarize_periodicity_scores",
]
