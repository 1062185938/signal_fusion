"""Validation-frozen Phase P3 evaluation of the LTE/DVB-T periodicity gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.evaluation import add_complex_awgn, standardize_iq_windows
from signal_fusion.fusion.periodicity_gate import (
    TECHNOLOGY_PERIODICITY_CANDIDATES,
    apply_periodicity_gate,
    load_periodicity_gate_manifest,
    select_periodicity_gate_margin,
    summarize_periodicity_gate,
)
from signal_fusion.io import load_prepared_dataset
from signal_fusion.io.writers import json_safe
from signal_fusion.model_inference.ensemble import (
    RegionEnsembleInferenceService,
    load_ensemble_services,
)
from signal_fusion.model_inference.service import ModelInferenceService
from signal_fusion.benchmarks.technology_recognition_periodicity import (
    LABEL_NAMES,
    score_periodicity_dataset,
    summarize_periodicity_scores,
)
from signal_fusion.feature_extraction import PERIODICITY_SCHEMA_ID


def _ensemble_region_arrays(
    dataset: PreparedDataset,
    services: Sequence[ModelInferenceService],
    *,
    batch_size: int,
) -> dict[str, Any]:
    if dataset.y is None:
        raise ValueError("ensemble evaluation requires labels")
    results = RegionEnsembleInferenceService(services).predict_all_groups(
        dataset,
        batch_size=batch_size,
    )
    if not results:
        raise ValueError("ensemble evaluation produced no regions")

    group_ids = np.asarray([item.group_id for item in results], dtype=np.int64)
    if np.unique(group_ids).size != group_ids.size:
        raise ValueError("ensemble region group_id values must be unique")
    labels = results[0].labels
    if tuple(labels) != ("LTE", "WiFi", "DVB-T"):
        raise ValueError(
            "periodicity fusion requires label order ('LTE', 'WiFi', 'DVB-T')"
        )

    truth: list[int] = []
    for item in results:
        indices = item.member_results[0].sample_indices
        group_labels = np.unique(dataset.y[indices])
        if group_labels.size != 1:
            raise ValueError(f"group_id={item.group_id} contains multiple labels")
        truth.append(int(group_labels[0]))

    member_probabilities = np.stack(
        [item.member_region_probabilities for item in results], axis=0
    ).astype(np.float32)
    ensemble_probabilities = np.stack(
        [item.ensemble_probabilities for item in results], axis=0
    ).astype(np.float32)
    return {
        "labels": labels,
        "group_id": group_ids,
        "source_id": np.asarray([item.source_id for item in results]),
        "y": np.asarray(truth, dtype=np.int64),
        "member_region_probabilities": member_probabilities,
        "ensemble_region_probabilities": ensemble_probabilities,
        "member_predictions": member_probabilities.argmax(axis=2),
        "ensemble_prediction": ensemble_probabilities.argmax(axis=1),
    }


def _align_periodicity(
    scores: dict[str, Any],
    target_group_ids: np.ndarray,
) -> dict[str, Any]:
    score_group_ids = np.asarray(scores["group_id"], dtype=np.int64)
    if np.unique(score_group_ids).size != score_group_ids.size:
        raise ValueError("periodicity group_id values must be unique")
    positions = {int(group_id): index for index, group_id in enumerate(score_group_ids)}
    if set(positions) != set(np.asarray(target_group_ids, dtype=np.int64).tolist()):
        raise ValueError("window and continuous-region datasets contain different groups")
    order = np.asarray(
        [positions[int(group_id)] for group_id in target_group_ids],
        dtype=np.int64,
    )
    aligned: dict[str, Any] = {}
    for name, values in scores.items():
        array = np.asarray(values)
        if array.ndim > 0 and array.shape[0] == score_group_ids.size:
            aligned[name] = array[order]
        else:
            aligned[name] = values
    return aligned


def _validate_alignment(
    ensemble: dict[str, Any],
    periodicity: dict[str, Any],
) -> None:
    if not np.array_equal(ensemble["group_id"], periodicity["group_id"]):
        raise ValueError("ensemble and periodicity group_id values do not align")
    if not np.array_equal(ensemble["y"], periodicity["y"]):
        raise ValueError("ensemble and periodicity labels do not align")
    if not np.array_equal(
        np.asarray(ensemble["source_id"]).astype(str),
        np.asarray(periodicity["source_id"]).astype(str),
    ):
        raise ValueError("ensemble and periodicity source_id values do not align")


def _evaluate_split(
    window_dataset: PreparedDataset,
    region_dataset: PreparedDataset,
    services: Sequence[ModelInferenceService],
    *,
    reliability_margin: float | None,
    batch_size: int,
) -> dict[str, Any]:
    ensemble = _ensemble_region_arrays(
        window_dataset,
        services,
        batch_size=batch_size,
    )
    periodicity = _align_periodicity(
        score_periodicity_dataset(region_dataset, batch_size=batch_size),
        ensemble["group_id"],
    )
    _validate_alignment(ensemble, periodicity)
    if reliability_margin is None:
        reliability_margin = select_periodicity_gate_margin(
            ensemble["y"],
            periodicity["prediction"],
            periodicity["margin"],
        )
    gated = apply_periodicity_gate(
        ensemble["member_predictions"],
        ensemble["ensemble_prediction"],
        periodicity["prediction"],
        periodicity["margin"],
        reliability_margin,
    )
    return {
        "ensemble": ensemble,
        "periodicity": periodicity,
        "gated": gated,
        "reliability_margin": float(reliability_margin),
    }


def _subset_summary(result: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    ensemble = result["ensemble"]
    gated = result["gated"]
    return summarize_periodicity_gate(
        np.asarray(ensemble["y"])[mask],
        np.asarray(ensemble["member_predictions"])[mask],
        np.asarray(ensemble["ensemble_prediction"])[mask],
        {name: np.asarray(values)[mask] for name, values in gated.items()},
    )


def _split_summary(result: dict[str, Any]) -> dict[str, Any]:
    ensemble = result["ensemble"]
    gated = result["gated"]
    truth = np.asarray(ensemble["y"], dtype=np.int64)
    source_ids = np.asarray(ensemble["source_id"]).astype(str)
    member_predictions = np.asarray(ensemble["member_predictions"])
    ensemble_prediction = np.asarray(ensemble["ensemble_prediction"])
    fused = np.asarray(gated["prediction"])
    changed = np.asarray(gated["changed"], dtype=bool)
    unanimous = np.all(member_predictions == member_predictions[:, :1], axis=1)

    summary = summarize_periodicity_gate(
        truth,
        member_predictions,
        ensemble_prediction,
        gated,
    )
    summary["reliability_margin"] = result["reliability_margin"]
    summary["invariants"] = {
        "unanimous_changed_count": int(np.sum(unanimous & changed)),
        "ensemble_wifi_changed_count": int(
            np.sum((ensemble_prediction == 1) & changed)
        ),
        "ineligible_changed_count": int(
            np.sum(~np.asarray(gated["eligible"], dtype=bool) & changed)
        ),
    }
    if any(summary["invariants"].values()):
        raise RuntimeError("periodicity gate violated a protected decision invariant")

    summary["per_class"] = {
        str(label): {
            "class_name": LABEL_NAMES[label],
            **_subset_summary(result, truth == label),
        }
        for label in sorted(LABEL_NAMES)
    }
    summary["per_source"] = {
        source_id: _subset_summary(result, source_ids == source_id)
        for source_id in sorted(np.unique(source_ids))
    }
    summary["changed_wrong_to_wrong_count"] = int(
        np.sum(changed & (ensemble_prediction != truth) & (fused != truth))
    )
    return summary


def _source_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    ensemble = result["ensemble"]
    source_ids = np.asarray(ensemble["source_id"]).astype(str)
    truth = np.asarray(ensemble["y"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for source_id in sorted(np.unique(source_ids)):
        mask = source_ids == source_id
        labels = np.unique(truth[mask])
        if labels.size != 1:
            raise ValueError(f"source_id={source_id!r} contains multiple labels")
        item = _subset_summary(result, mask)
        rows.append(
            {
                "source_id": source_id,
                "class_name": LABEL_NAMES[int(labels[0])],
                "region_count": item["region_count"],
                "baseline_accuracy_percent": item["baseline_ensemble"][
                    "accuracy_percent"
                ],
                "gate_candidate_count": item["lte_dvbt_gate_candidate_count"],
                "periodicity_resolved_count": item["periodicity_resolved_count"],
                "changed_label_count": item["changed_label_count"],
                "corrected_error_count": item["corrected_error_count"],
                "introduced_error_count": item["introduced_error_count"],
                "fused_accuracy_percent": item["fused"]["accuracy_percent"],
            }
        )
    return rows


def _prediction_artifact_arrays(
    split_name: str,
    result: dict[str, Any],
) -> dict[str, np.ndarray]:
    ensemble = result["ensemble"]
    periodicity = result["periodicity"]
    gated = result["gated"]
    count = len(np.asarray(ensemble["y"]))
    return {
        "split": np.full(count, split_name),
        "group_id": np.asarray(ensemble["group_id"]),
        "source_id": np.asarray(ensemble["source_id"]).astype(str),
        "y": np.asarray(ensemble["y"]),
        "member_region_probabilities": np.asarray(
            ensemble["member_region_probabilities"]
        ),
        "ensemble_region_probabilities": np.asarray(
            ensemble["ensemble_region_probabilities"]
        ),
        "member_predictions": np.asarray(ensemble["member_predictions"]),
        "ensemble_prediction": np.asarray(ensemble["ensemble_prediction"]),
        "periodicity_correlations": np.asarray(periodicity["correlations"]),
        "periodicity_peak_lags": np.asarray(periodicity["peak_lags"]),
        "periodicity_prediction": np.asarray(periodicity["prediction"]),
        "periodicity_margin": np.asarray(periodicity["margin"]),
        "gate_eligible": np.asarray(gated["eligible"]),
        "gate_resolved": np.asarray(gated["resolved"]),
        "label_changed": np.asarray(gated["changed"]),
        "fused_prediction": np.asarray(gated["prediction"]),
    }


def run_periodicity_fusion_evaluation(
    window_dataset_dir: str | Path,
    region_dataset_dir: str | Path,
    model_paths: Sequence[str | Path],
    label_map_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze the gate on validation and evaluate it once on test."""

    window_root = Path(window_dataset_dir)
    region_root = Path(region_dataset_dir)
    output = Path(output_dir)
    manifest_path = output / "periodicity_gate_manifest.json"
    report_path = output / "periodicity_fusion_evaluation.json"
    source_path = output / "periodicity_fusion_test_by_source.csv"
    prediction_path = output / "periodicity_fusion_predictions_private.npz"
    existing = [
        path
        for path in (manifest_path, report_path, source_path, prediction_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "periodicity fusion output exists; use overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )

    for root, description in (
        (window_root, "2048-window dataset directory"),
        (region_root, "4096-region dataset directory"),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{description} not found: {root}")
        for split in ("validation", "test"):
            if not (root / f"{split}.npz").is_file():
                raise FileNotFoundError(f"missing {split} dataset under {root}")

    services = load_ensemble_services(model_paths, label_map_path)
    validation = _evaluate_split(
        load_prepared_dataset(window_root / "validation.npz"),
        load_prepared_dataset(region_root / "validation.npz"),
        services,
        reliability_margin=None,
        batch_size=batch_size,
    )
    frozen_margin = validation["reliability_margin"]
    test = _evaluate_split(
        load_prepared_dataset(window_root / "test.npz"),
        load_prepared_dataset(region_root / "test.npz"),
        services,
        reliability_margin=frozen_margin,
        batch_size=batch_size,
    )

    report = {
        "schema_version": 1,
        "result_type": "technology_recognition_periodicity_phase_p3",
        "scope": {
            "labels": list(validation["ensemble"]["labels"]),
            "sample_rate_hz": 1_000_000,
            "iq_window_sample_count": 2048,
            "region_sample_count": 4096,
            "ensemble_member_count": 3,
        },
        "input": {
            "window_dataset_dir": str(window_root.resolve()),
            "region_dataset_dir": str(region_root.resolve()),
            "model_paths": [str(Path(path).resolve()) for path in model_paths],
            "label_map_path": str(Path(label_map_path).resolve()),
        },
        "frozen_rule": {
            "selected_on": "validation",
            "reliability_margin": frozen_margin,
            "threshold_selection": (
                "smallest periodicity margin strictly above every LTE/DVB-T "
                "validation misclassification"
            ),
            "validation_periodicity_calibration": summarize_periodicity_scores(
                validation["periodicity"], frozen_margin
            ),
            "eligibility": (
                "all member top1 labels are LTE or DVB-T, both labels occur, "
                "and ensemble top1 is not WiFi"
            ),
            "action": (
                "use periodicity prediction only when eligible and reliable; "
                "otherwise preserve ensemble prediction"
            ),
            "protected_decisions": [
                "all unanimous member decisions",
                "all ensemble WiFi decisions",
                "all conflicts containing a WiFi member vote",
            ],
        },
        "validation": _split_summary(validation),
        "test": _split_summary(test),
    }
    manifest = {
        "schema_version": 1,
        "result_type": "technology_recognition_periodicity_gate",
        "labels": list(validation["ensemble"]["labels"]),
        "sample_rate_hz": 1_000_000,
        "region_sample_count": 4096,
        "periodicity_schema_id": PERIODICITY_SCHEMA_ID,
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "period_seconds": item.period_seconds,
            }
            for item in TECHNOLOGY_PERIODICITY_CANDIDATES
        ],
        "frozen_rule": report["frozen_rule"],
        "validation_dataset_path": str(
            (region_root / "validation.npz").resolve()
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = _source_rows(test)
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    validation_arrays = _prediction_artifact_arrays("validation", validation)
    test_arrays = _prediction_artifact_arrays("test", test)
    np.savez_compressed(
        prediction_path,
        **{
            name: np.concatenate(
                (validation_arrays[name], test_arrays[name]), axis=0
            )
            for name in validation_arrays
        },
        reliability_margin=np.asarray(frozen_margin, dtype=np.float64),
    )
    return {
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "source_csv_path": str(source_path),
        "predictions_path": str(prediction_path),
        "report": report,
    }


def run_frozen_periodicity_gate_evaluation(
    window_dataset_path: str | Path,
    region_dataset_path: str | Path,
    model_paths: Sequence[str | Path],
    label_map_path: str | Path,
    gate_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate a previously frozen periodicity gate on one external dataset."""

    window_path = Path(window_dataset_path)
    region_path = Path(region_dataset_path)
    output = Path(output_dir)
    report_path = output / "periodicity_frozen_gate_evaluation.json"
    source_path = output / "periodicity_frozen_gate_by_source.csv"
    prediction_path = output / "periodicity_frozen_gate_predictions_private.npz"
    existing = [
        path for path in (report_path, source_path, prediction_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "frozen periodicity evaluation output exists; use overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    for path, description in (
        (window_path, "2048-window dataset"),
        (region_path, "4096-region dataset"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    manifest, frozen_margin = load_periodicity_gate_manifest(gate_manifest_path)
    window_dataset = load_prepared_dataset(window_path)
    region_dataset = load_prepared_dataset(region_path)
    if window_dataset.seq_len != 2048:
        raise ValueError("IQ ensemble dataset must use 2048-sample windows")
    if region_dataset.seq_len != int(manifest["region_sample_count"]):
        raise ValueError("continuous-region length does not match the frozen manifest")

    services = load_ensemble_services(model_paths, label_map_path)
    result = _evaluate_split(
        window_dataset,
        region_dataset,
        services,
        reliability_margin=frozen_margin,
        batch_size=batch_size,
    )
    report = {
        "schema_version": 1,
        "result_type": "technology_recognition_periodicity_phase_p4_external",
        "frozen_gate": {
            "manifest_path": str(Path(gate_manifest_path).resolve()),
            "reliability_margin": frozen_margin,
            "threshold_was_refit": False,
            "eligibility": manifest["frozen_rule"]["eligibility"],
            "action": manifest["frozen_rule"]["action"],
            "protected_decisions": manifest["frozen_rule"][
                "protected_decisions"
            ],
        },
        "input": {
            "window_dataset_path": str(window_path.resolve()),
            "region_dataset_path": str(region_path.resolve()),
            "model_paths": [str(Path(path).resolve()) for path in model_paths],
            "label_map_path": str(Path(label_map_path).resolve()),
        },
        "external_test": _split_summary(result),
    }

    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = _source_rows(result)
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arrays = _prediction_artifact_arrays("external_test", result)
    np.savez_compressed(
        prediction_path,
        **arrays,
        reliability_margin=np.asarray(frozen_margin, dtype=np.float64),
    )
    return {
        "report_path": str(report_path),
        "source_csv_path": str(source_path),
        "predictions_path": str(prediction_path),
        "report": report,
    }


def _windowize_continuous_regions(region_x: np.ndarray) -> np.ndarray:
    values = np.asarray(region_x)
    if values.ndim != 3 or values.shape[1:] != (2, 4096):
        raise ValueError(
            "continuous region IQ must have shape [N, 2, 4096], "
            f"got {values.shape}"
        )
    windows = values.reshape(values.shape[0], 2, 2, 2048)
    windows = windows.transpose(0, 2, 1, 3).reshape(-1, 2, 2048)
    return standardize_iq_windows(windows)


def _compact_awgn_trial(
    snr_db: float,
    seed: int,
    achieved_snr_db: np.ndarray,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "snr_db": float(snr_db),
        "noise_seed": int(seed),
        "achieved_snr_db_mean": float(np.mean(achieved_snr_db)),
        "achieved_snr_db_std": float(np.std(achieved_snr_db)),
        "baseline_accuracy_percent": summary["baseline_ensemble"][
            "accuracy_percent"
        ],
        "baseline_error_count": summary["baseline_ensemble"]["error_count"],
        "fused_accuracy_percent": summary["fused"]["accuracy_percent"],
        "fused_error_count": summary["fused"]["error_count"],
        "member_disagreement_count": summary["member_disagreement_count"],
        "lte_dvbt_gate_candidate_count": summary[
            "lte_dvbt_gate_candidate_count"
        ],
        "periodicity_resolved_count": summary["periodicity_resolved_count"],
        "changed_label_count": summary["changed_label_count"],
        "corrected_error_count": summary["corrected_error_count"],
        "introduced_error_count": summary["introduced_error_count"],
        "net_correction_count": summary["net_correction_count"],
        "review_required_count": summary["review_required"]["region_count"],
        "accepted_accuracy_percent": summary["accepted"]["accuracy_percent"],
        "per_class": {
            label: {
                "class_name": item["class_name"],
                "baseline_accuracy_percent": item["baseline_ensemble"][
                    "accuracy_percent"
                ],
                "fused_accuracy_percent": item["fused"]["accuracy_percent"],
                "corrected_error_count": item["corrected_error_count"],
                "introduced_error_count": item["introduced_error_count"],
            }
            for label, item in summary["per_class"].items()
        },
    }


def _aggregate_awgn_trials(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("AWGN aggregation requires at least one trial")
    scalar_fields = (
        "baseline_accuracy_percent",
        "baseline_error_count",
        "fused_accuracy_percent",
        "fused_error_count",
        "member_disagreement_count",
        "lte_dvbt_gate_candidate_count",
        "periodicity_resolved_count",
        "changed_label_count",
        "corrected_error_count",
        "introduced_error_count",
        "net_correction_count",
        "review_required_count",
        "accepted_accuracy_percent",
    )
    aggregate: dict[str, Any] = {"trial_count": len(trials)}
    for field in scalar_fields:
        values = np.asarray([trial[field] for trial in trials], dtype=np.float64)
        aggregate[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return aggregate


def run_frozen_periodicity_gate_awgn_evaluation(
    window_dataset_path: str | Path,
    region_dataset_path: str | Path,
    model_paths: Sequence[str | Path],
    label_map_path: str | Path,
    gate_manifest_path: str | Path,
    clean_evaluation_path: str | Path,
    output_dir: str | Path,
    *,
    snr_db_values: Sequence[float] = (10.0, 7.5, 5.0),
    noise_seeds: Sequence[int] = (44, 45, 46, 47, 48),
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate the frozen gate under region-consistent incremental AWGN."""

    window_path = Path(window_dataset_path)
    region_path = Path(region_dataset_path)
    clean_path = Path(clean_evaluation_path)
    output = Path(output_dir)
    report_path = output / "periodicity_gate_awgn_evaluation.json"
    trial_path = output / "periodicity_gate_awgn_trials.csv"
    prediction_path = output / "periodicity_gate_awgn_predictions_private.npz"
    existing = [
        path for path in (report_path, trial_path, prediction_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "periodicity AWGN output exists; use overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    for path, description in (
        (window_path, "2048-window dataset"),
        (region_path, "4096-region dataset"),
        (clean_path, "clean frozen-gate evaluation"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    snr_values = tuple(float(value) for value in snr_db_values)
    seeds = tuple(int(value) for value in noise_seeds)
    if not snr_values or not np.all(np.isfinite(snr_values)):
        raise ValueError("snr_db_values must contain finite values")
    if len(set(snr_values)) != len(snr_values):
        raise ValueError("snr_db_values must not contain duplicates")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("noise_seeds must be non-empty and unique")

    manifest, frozen_margin = load_periodicity_gate_manifest(gate_manifest_path)
    clean_report = json.loads(clean_path.read_text(encoding="utf-8"))
    if clean_report.get("result_type") != (
        "technology_recognition_periodicity_phase_p4_external"
    ):
        raise ValueError("clean evaluation has an unexpected result_type")
    if not np.isclose(
        clean_report["frozen_gate"]["reliability_margin"],
        frozen_margin,
    ):
        raise ValueError("clean evaluation does not use the frozen gate margin")

    window_dataset = load_prepared_dataset(window_path)
    region_dataset = load_prepared_dataset(region_path)
    if window_dataset.seq_len != 2048 or region_dataset.seq_len != 4096:
        raise ValueError("AWGN evaluation requires 2048 windows and 4096 regions")
    reconstructed_clean = _windowize_continuous_regions(region_dataset.X)
    if not np.array_equal(reconstructed_clean, window_dataset.X):
        raise ValueError(
            "continuous regions do not reconstruct the clean IQ-model windows"
        )

    expected_group_ids = np.repeat(
        np.asarray(region_dataset.meta["group_id"], dtype=np.int64), 2
    )
    if not np.array_equal(
        expected_group_ids,
        np.asarray(window_dataset.meta["group_id"], dtype=np.int64),
    ):
        raise ValueError("continuous regions and IQ windows have different ordering")

    services = load_ensemble_services(model_paths, label_map_path)
    trials: list[dict[str, Any]] = []
    prediction_parts: list[dict[str, np.ndarray]] = []
    for snr_db in snr_values:
        for seed in seeds:
            noisy_region_x, achieved = add_complex_awgn(
                region_dataset.X,
                snr_db,
                rng=np.random.default_rng(seed),
            )
            noisy_window_x = _windowize_continuous_regions(noisy_region_x)
            noisy_windows = PreparedDataset(
                X=noisy_window_x,
                y=np.asarray(window_dataset.y).copy(),
                source_id=window_dataset.source_id,
                meta=dict(window_dataset.meta),
            )
            noisy_regions = PreparedDataset(
                X=noisy_region_x,
                y=np.asarray(region_dataset.y).copy(),
                source_id=region_dataset.source_id,
                meta=dict(region_dataset.meta),
            )
            result = _evaluate_split(
                noisy_windows,
                noisy_regions,
                services,
                reliability_margin=frozen_margin,
                batch_size=batch_size,
            )
            summary = _split_summary(result)
            trial = _compact_awgn_trial(snr_db, seed, achieved, summary)
            trials.append(trial)

            arrays = _prediction_artifact_arrays(
                f"awgn_{snr_db:g}_db_seed_{seed}", result
            )
            count = len(arrays["y"])
            arrays["snr_db"] = np.full(count, snr_db, dtype=np.float64)
            arrays["noise_seed"] = np.full(count, seed, dtype=np.int64)
            arrays["achieved_snr_db"] = np.asarray(achieved, dtype=np.float64)
            prediction_parts.append(arrays)
            print(
                "[periodicity-awgn] "
                f"snr={snr_db:g} seed={seed} "
                f"baseline={trial['baseline_accuracy_percent']:.6f}% "
                f"fused={trial['fused_accuracy_percent']:.6f}% "
                f"corrected={trial['corrected_error_count']} "
                f"introduced={trial['introduced_error_count']}",
                flush=True,
            )

    conditions = []
    for snr_db in snr_values:
        selected = [trial for trial in trials if trial["snr_db"] == snr_db]
        conditions.append(
            {
                "snr_db": snr_db,
                "trials": selected,
                "aggregate": _aggregate_awgn_trials(selected),
            }
        )
    report = {
        "schema_version": 1,
        "result_type": "technology_recognition_periodicity_phase_p5_awgn",
        "noise_definition": (
            "per-region circular complex AWGN relative to the complete captured "
            "region power; this is incremental SNR, not physical over-the-air SNR"
        ),
        "noise_application": (
            "add noise once to each 4096-sample region, then derive the two "
            "standardized 2048-sample IQ-model windows from the same noisy region"
        ),
        "frozen_gate": {
            "manifest_path": str(Path(gate_manifest_path).resolve()),
            "reliability_margin": frozen_margin,
            "threshold_was_refit": False,
            "model_was_retrained": False,
            "eligibility": manifest["frozen_rule"]["eligibility"],
        },
        "input": {
            "window_dataset_path": str(window_path.resolve()),
            "region_dataset_path": str(region_path.resolve()),
            "clean_evaluation_path": str(clean_path.resolve()),
            "model_paths": [str(Path(path).resolve()) for path in model_paths],
            "label_map_path": str(Path(label_map_path).resolve()),
            "snr_db_values": list(snr_values),
            "noise_seeds": list(seeds),
        },
        "clean_reference": clean_report["external_test"],
        "awgn_conditions": conditions,
    }

    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = [
        {key: value for key, value in trial.items() if key != "per_class"}
        for trial in trials
    ]
    with trial_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        prediction_path,
        **{
            name: np.concatenate([part[name] for part in prediction_parts], axis=0)
            for name in prediction_parts[0]
        },
        reliability_margin=np.asarray(frozen_margin, dtype=np.float64),
    )
    return {
        "report_path": str(report_path),
        "trials_csv_path": str(trial_path),
        "predictions_path": str(prediction_path),
        "report": report,
    }


__all__ = [
    "run_frozen_periodicity_gate_awgn_evaluation",
    "run_frozen_periodicity_gate_evaluation",
    "run_periodicity_fusion_evaluation",
]
