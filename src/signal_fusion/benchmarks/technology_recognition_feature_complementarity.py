"""Small cross-location pilot for IQ/64-feature complementarity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.benchmarks.technology_recognition import LABELS, LOCATION_FOLDS
from signal_fusion.evaluation.snr import standardize_iq_windows
from signal_fusion.evaluation.snr_runner import (
    load_evaluation_model,
    predict_probabilities,
    resolve_evaluation_device,
)
from signal_fusion.feature_classifier import FeatureClassifierService
from signal_fusion.feature_classifier.trainer import train_feature_classifier
from signal_fusion.feature_extraction import (
    FEATURE_SCHEMA_ID,
    FeatureExtractionService,
    IQFeatureCtypesBackend,
    extract_feature_matrix,
)
from signal_fusion.io import load_prepared_dataset
from signal_fusion.io.writers import json_safe


LABEL_MAP = {str(label): name for name, label in LABELS.items()}


def uniform_indices_per_source(
    source_ids: np.ndarray,
    count_per_source: int,
) -> np.ndarray:
    """Select deterministic, uniformly spaced rows within every source."""

    source_ids = np.asarray(source_ids).astype(str)
    if source_ids.ndim != 1 or source_ids.size == 0:
        raise ValueError("source_ids must be a non-empty one-dimensional array")
    if isinstance(count_per_source, bool) or int(count_per_source) != count_per_source:
        raise TypeError("count_per_source must be an integer")
    count_per_source = int(count_per_source)
    if count_per_source <= 0:
        raise ValueError("count_per_source must be positive")

    selected: list[np.ndarray] = []
    for source_id in sorted(set(source_ids.tolist())):
        candidates = np.flatnonzero(source_ids == source_id)
        if candidates.size < count_per_source:
            raise ValueError(
                f"source {source_id!r} contains only {candidates.size} rows; "
                f"cannot select {count_per_source}"
            )
        positions = np.floor(
            (np.arange(count_per_source, dtype=np.float64) + 0.5)
            * candidates.size
            / count_per_source
        ).astype(np.int64)
        selected.append(candidates[positions])
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def select_matched_correct_controls(
    labels: np.ndarray,
    source_ids: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Match every IQ error to a correct row from the same source and class."""

    labels = np.asarray(labels, dtype=np.int64)
    source_ids = np.asarray(source_ids).astype(str)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.ndim != 1 or source_ids.shape != labels.shape or predictions.shape != labels.shape:
        raise ValueError("labels, source_ids, and predictions must share shape [N]")

    error_mask = predictions != labels
    correct_mask = ~error_mask
    selected: list[np.ndarray] = []
    for source_id in sorted(set(source_ids[error_mask].tolist())):
        source_errors = np.flatnonzero(error_mask & (source_ids == source_id))
        unique_labels = np.unique(labels[source_errors])
        if unique_labels.size != 1:
            raise ValueError(f"source {source_id!r} contains multiple true labels")
        candidates = np.flatnonzero(
            correct_mask
            & (source_ids == source_id)
            & (labels == int(unique_labels[0]))
        )
        requested = source_errors.size
        if candidates.size < requested:
            raise ValueError(
                f"source {source_id!r} has {requested} errors but only "
                f"{candidates.size} correct controls"
            )
        positions = np.floor(
            (np.arange(requested, dtype=np.float64) + 0.5)
            * candidates.size
            / requested
        ).astype(np.int64)
        selected.append(candidates[positions])
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def _accuracy(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    count = int(labels.size)
    correct = int(np.count_nonzero(labels == predictions))
    return {
        "count": count,
        "correct_count": correct,
        "error_count": count - correct,
        "accuracy_percent": 100.0 * correct / count if count else None,
        "per_class_accuracy_percent": {
            name: (
                100.0
                * float(np.mean(predictions[labels == label] == label))
                if np.any(labels == label)
                else None
            )
            for name, label in LABELS.items()
        },
    }


def summarize_fold_complementarity(
    labels: np.ndarray,
    iq_predictions: np.ndarray,
    feature_predictions: np.ndarray,
    uniform_mask: np.ndarray,
    error_mask: np.ndarray,
    control_mask: np.ndarray,
) -> dict[str, Any]:
    """Summarize cross-location accuracy and diagnostic rescue/harm counts."""

    labels = np.asarray(labels, dtype=np.int64)
    iq_predictions = np.asarray(iq_predictions, dtype=np.int64)
    feature_predictions = np.asarray(feature_predictions, dtype=np.int64)
    masks = {
        "uniform": np.asarray(uniform_mask, dtype=bool),
        "error": np.asarray(error_mask, dtype=bool),
        "control": np.asarray(control_mask, dtype=bool),
    }
    for name, values in (
        ("iq_predictions", iq_predictions),
        ("feature_predictions", feature_predictions),
        *masks.items(),
    ):
        if values.shape != labels.shape:
            raise ValueError(f"{name} must have shape {labels.shape}")
    if np.any(iq_predictions[masks["error"]] == labels[masks["error"]]):
        raise ValueError("error_mask contains an IQ-correct row")
    if np.any(iq_predictions[masks["control"]] != labels[masks["control"]]):
        raise ValueError("control_mask contains an IQ-error row")

    rescued = int(
        np.count_nonzero(
            feature_predictions[masks["error"]] == labels[masks["error"]]
        )
    )
    error_count = int(np.count_nonzero(masks["error"]))
    introduced = int(
        np.count_nonzero(
            feature_predictions[masks["control"]] != labels[masks["control"]]
        )
    )
    control_count = int(np.count_nonzero(masks["control"]))
    return {
        "uniform_cross_location": {
            "iq": _accuracy(
                labels[masks["uniform"]], iq_predictions[masks["uniform"]]
            ),
            "feature_probe": _accuracy(
                labels[masks["uniform"]], feature_predictions[masks["uniform"]]
            ),
        },
        "iq_error_diagnostic": {
            "error_count": error_count,
            "feature_rescued_count": rescued,
            "feature_rescue_rate_percent": (
                100.0 * rescued / error_count if error_count else None
            ),
            "feature_predictions": _accuracy(
                labels[masks["error"]], feature_predictions[masks["error"]]
            ),
        },
        "matched_correct_control": {
            "control_count": control_count,
            "feature_correct_count": control_count - introduced,
            "feature_wrong_count": introduced,
            "feature_wrong_rate_percent": (
                100.0 * introduced / control_count if control_count else None
            ),
            "feature_predictions": _accuracy(
                labels[masks["control"]], feature_predictions[masks["control"]]
            ),
        },
    }


def _sample_field(dataset, name: str) -> np.ndarray:
    if name not in dataset.meta:
        raise ValueError(f"test dataset is missing metadata {name!r}")
    values = np.asarray(dataset.meta[name])
    if values.shape != (dataset.num_samples,):
        raise ValueError(
            f"test metadata {name!r} must have shape [{dataset.num_samples}]"
        )
    return values


def _write_feature_split(
    path: Path,
    bank: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    dataset_id: str,
    split_name: str,
) -> None:
    count = int(indices.size)
    np.savez_compressed(
        path,
        features=bank["features"][indices],
        y=bank["y"][indices],
        group_id=np.arange(count, dtype=np.int64),
        source_region_id=bank["source_region_id"][indices],
        sample_source_id=bank["sample_source_id"][indices],
        condition=np.full(count, "clean"),
        requested_snr_db=np.full(count, np.nan, dtype=np.float64),
        achieved_snr_db=np.full(count, np.nan, dtype=np.float64),
        noise_seed=np.full(count, -1, dtype=np.int64),
        original_sample_count=np.full(count, 4096, dtype=np.int64),
        used_sample_count=np.full(count, 4096, dtype=np.int64),
        sample_rate=np.full(count, 1_000_000.0, dtype=np.float64),
        feature_names=bank["feature_names"],
        feature_schema_id=np.asarray(FEATURE_SCHEMA_ID),
        feature_dataset_version=np.asarray("region_features_v1"),
        split=np.asarray(split_name),
        source_dataset_id=np.asarray(dataset_id),
        dataset_id=np.asarray(dataset_id),
        label_map_json=np.asarray(json.dumps(LABEL_MAP, ensure_ascii=False)),
    )


def _save_case_audit(
    path: Path,
    bank: dict[str, np.ndarray],
    feature_probabilities: np.ndarray,
) -> None:
    fields = (
        "fold",
        "test_location",
        "test_index",
        "sample_source_id",
        "source_region_id",
        "role",
        "true_label",
        "iq_prediction",
        "iq_confidence",
        "feature_prediction",
        "feature_confidence",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        iq_prediction = bank["iq_probabilities"].argmax(axis=1)
        feature_prediction = feature_probabilities.argmax(axis=1)
        for index in range(bank["y"].size):
            roles = []
            if bool(bank["is_uniform_probe"][index]):
                roles.append("uniform_probe")
            if bool(bank["is_iq_error"][index]):
                roles.append("iq_error")
            if bool(bank["is_matched_control"][index]):
                roles.append("matched_control")
            writer.writerow(
                {
                    "fold": str(bank["fold"][index]),
                    "test_location": str(bank["test_location"][index]),
                    "test_index": int(bank["test_index"][index]),
                    "sample_source_id": str(bank["sample_source_id"][index]),
                    "source_region_id": int(bank["source_region_id"][index]),
                    "role": "+".join(roles),
                    "true_label": LABEL_MAP[str(int(bank["y"][index]))],
                    "iq_prediction": LABEL_MAP[str(int(iq_prediction[index]))],
                    "iq_confidence": float(np.max(bank["iq_probabilities"][index])),
                    "feature_prediction": LABEL_MAP[
                        str(int(feature_prediction[index]))
                    ],
                    "feature_confidence": float(
                        np.max(feature_probabilities[index])
                    ),
                }
            )


def run_feature_complementarity_pilot(
    dataset_root: str | Path,
    model_root: str | Path,
    output_dir: str | Path,
    *,
    regions_per_source: int = 8,
    batch_size: int = 256,
    device: str = "cuda",
    seed: int = 44,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the first-round, location-isolated 64-feature diagnostic."""

    dataset_root = Path(dataset_root)
    model_root = Path(model_root)
    output = Path(output_dir)
    result_path = output / "result.json"
    prediction_path = output / "iq_predictions.npz"
    bank_path = output / "feature_bank.npz"
    audit_path = output / "case_audit.csv"
    if not overwrite:
        existing = [
            path
            for path in (result_path, prediction_path, bank_path, audit_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "pilot outputs already exist; use --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_evaluation_device(device)
    feature_service = FeatureExtractionService()

    prediction_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "fold",
            "test_location",
            "test_index",
            "group_id",
            "source_region_id",
            "sample_source_id",
            "y",
            "probabilities",
            "prediction",
        )
    }
    bank_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "fold",
            "test_location",
            "test_index",
            "group_id",
            "source_region_id",
            "sample_source_id",
            "y",
            "iq_probabilities",
            "is_uniform_probe",
            "is_iq_error",
            "is_matched_control",
            "features",
        )
    }
    fold_base: dict[str, dict[str, Any]] = {}

    with IQFeatureCtypesBackend() as backend:
        for fold_name, locations in LOCATION_FOLDS.items():
            test_location = locations["test"][0]
            test_path = dataset_root / fold_name / "test.npz"
            model_path = model_root / fold_name / "best_model.pth"
            dataset = load_prepared_dataset(test_path)
            if dataset.y is None:
                raise ValueError(f"test dataset has no labels: {test_path}")
            source_ids = _sample_field(dataset, "sample_source_id").astype(str)
            group_ids = _sample_field(dataset, "group_id").astype(np.int64)
            source_region_ids = _sample_field(
                dataset, "source_region_id"
            ).astype(np.int64)
            sample_rates = _sample_field(dataset, "sample_rate").astype(np.float64)
            if not np.allclose(sample_rates, 1_000_000.0):
                raise ValueError(f"{fold_name} test data is not uniformly 1 MS/s")

            model = load_evaluation_model(
                model_path,
                model_name="deepconvnet_1d",
                class_num=len(LABELS),
                input_channels=int(dataset.X.shape[1]),
                seq_len=dataset.seq_len,
                device=resolved_device,
            )
            probabilities = predict_probabilities(
                model,
                dataset.X,
                batch_size=batch_size,
                device=resolved_device,
            )
            predictions = probabilities.argmax(axis=1).astype(np.int64)
            error_indices = np.flatnonzero(predictions != dataset.y)
            control_indices = select_matched_correct_controls(
                dataset.y, source_ids, predictions
            )
            uniform_indices = uniform_indices_per_source(
                source_ids, regions_per_source
            )
            selected_indices = np.unique(
                np.concatenate((uniform_indices, error_indices, control_indices))
            )
            selected_x = standardize_iq_windows(dataset.X[selected_indices])
            features = extract_feature_matrix(
                selected_x,
                1_000_000.0,
                backend,
                progress_every=100,
            )

            count = dataset.num_samples
            prediction_parts["fold"].append(np.full(count, fold_name))
            prediction_parts["test_location"].append(
                np.full(count, test_location)
            )
            prediction_parts["test_index"].append(
                np.arange(count, dtype=np.int64)
            )
            prediction_parts["group_id"].append(group_ids)
            prediction_parts["source_region_id"].append(source_region_ids)
            prediction_parts["sample_source_id"].append(source_ids)
            prediction_parts["y"].append(dataset.y.astype(np.int64, copy=False))
            prediction_parts["probabilities"].append(probabilities)
            prediction_parts["prediction"].append(predictions)

            selected_count = selected_indices.size
            bank_parts["fold"].append(np.full(selected_count, fold_name))
            bank_parts["test_location"].append(
                np.full(selected_count, test_location)
            )
            bank_parts["test_index"].append(selected_indices)
            bank_parts["group_id"].append(group_ids[selected_indices])
            bank_parts["source_region_id"].append(
                source_region_ids[selected_indices]
            )
            bank_parts["sample_source_id"].append(source_ids[selected_indices])
            bank_parts["y"].append(dataset.y[selected_indices])
            bank_parts["iq_probabilities"].append(probabilities[selected_indices])
            bank_parts["is_uniform_probe"].append(
                np.isin(selected_indices, uniform_indices)
            )
            bank_parts["is_iq_error"].append(
                np.isin(selected_indices, error_indices)
            )
            bank_parts["is_matched_control"].append(
                np.isin(selected_indices, control_indices)
            )
            bank_parts["features"].append(features)
            fold_base[fold_name] = {
                "test_location": test_location,
                "test_region_count": count,
                "iq_error_count": int(error_indices.size),
                "matched_control_count": int(control_indices.size),
                "uniform_probe_count": int(uniform_indices.size),
                "unique_extracted_count": int(selected_indices.size),
            }
            print(
                f"[complementarity] {fold_name}/{test_location}: "
                f"errors={error_indices.size}, controls={control_indices.size}, "
                f"unique_features={selected_indices.size}"
            )

    predictions_all = {
        name: np.concatenate(parts, axis=0)
        for name, parts in prediction_parts.items()
    }
    bank = {
        name: np.concatenate(parts, axis=0)
        for name, parts in bank_parts.items()
    }
    bank["feature_names"] = np.asarray(feature_service.feature_names)
    bank["feature_schema_id"] = np.asarray(FEATURE_SCHEMA_ID)
    np.savez_compressed(prediction_path, **predictions_all)
    np.savez_compressed(bank_path, **bank)

    fold_summaries: dict[str, Any] = {}
    feature_probability_parts = np.full(
        (bank["y"].size, len(LABELS)), np.nan, dtype=np.float64
    )
    for fold_name, locations in LOCATION_FOLDS.items():
        fold_dir = output / "folds" / fold_name
        feature_dataset_dir = fold_dir / "feature_dataset"
        feature_model_dir = fold_dir / "feature_probe"
        feature_dataset_dir.mkdir(parents=True, exist_ok=True)
        uniform = bank["is_uniform_probe"].astype(bool)
        split_locations = {
            "train": locations["train"],
            "validation": locations["validation"],
            "test": locations["test"],
        }
        dataset_id = f"technology_recognition_64_v2_complementarity_{fold_name}"
        for split_name, selected_locations in split_locations.items():
            indices = np.flatnonzero(
                uniform & np.isin(bank["test_location"], selected_locations)
            )
            _write_feature_split(
                feature_dataset_dir / f"{split_name}.npz",
                bank,
                indices,
                dataset_id=dataset_id,
                split_name=split_name,
            )

        training = train_feature_classifier(
            feature_dataset_dir,
            feature_model_dir,
            device=device,
            batch_size=64,
            learning_rate=1e-2,
            max_epochs=300,
            patience=30,
            seed=seed,
            overwrite=overwrite,
        )
        service = FeatureClassifierService(
            feature_model_dir / "feature_classifier_manifest.json"
        )
        fold_mask = bank["fold"] == fold_name
        feature_result = service.predict(bank["features"][fold_mask])
        feature_probability_parts[fold_mask] = feature_result.probabilities
        feature_predictions = feature_result.probabilities.argmax(axis=1)
        fold_summary = summarize_fold_complementarity(
            bank["y"][fold_mask],
            bank["iq_probabilities"][fold_mask].argmax(axis=1),
            feature_predictions,
            bank["is_uniform_probe"][fold_mask],
            bank["is_iq_error"][fold_mask],
            bank["is_matched_control"][fold_mask],
        )
        fold_summary.update(fold_base[fold_name])
        fold_summary["feature_probe_training"] = {
            "train_count": training["metrics"]["train"]["sample_count"],
            "validation_count": training["metrics"]["validation"]["sample_count"],
            "test_count": training["metrics"]["test"]["sample_count"],
            "best_epoch": training["best_epoch"],
            "model_type": training["model_type"],
        }
        fold_summaries[fold_name] = fold_summary

    if not np.all(np.isfinite(feature_probability_parts)):
        raise RuntimeError("some selected regions did not receive a feature prediction")
    np.savez_compressed(
        output / "feature_predictions.npz",
        fold=bank["fold"],
        test_location=bank["test_location"],
        test_index=bank["test_index"],
        y=bank["y"],
        probabilities=feature_probability_parts,
        prediction=feature_probability_parts.argmax(axis=1).astype(np.int64),
    )
    _save_case_audit(audit_path, bank, feature_probability_parts)

    uniform_mask = bank["is_uniform_probe"].astype(bool)
    error_mask = bank["is_iq_error"].astype(bool)
    control_mask = bank["is_matched_control"].astype(bool)
    aggregate = summarize_fold_complementarity(
        bank["y"],
        bank["iq_probabilities"].argmax(axis=1),
        feature_probability_parts.argmax(axis=1),
        uniform_mask,
        error_mask,
        control_mask,
    )
    total_regions = int(predictions_all["y"].size)
    total_iq_errors = int(
        np.count_nonzero(
            predictions_all["prediction"] != predictions_all["y"]
        )
    )
    rescued = aggregate["iq_error_diagnostic"]["feature_rescued_count"]
    aggregate["oracle_upper_bound"] = {
        "definition": (
            "IQ prediction plus a hypothetical perfect selector that uses the "
            "feature probe only when the feature probe is correct"
        ),
        "base_iq_correct_count": total_regions - total_iq_errors,
        "additional_correct_count": rescued,
        "oracle_correct_count": total_regions - total_iq_errors + rescued,
        "oracle_accuracy_percent": (
            100.0 * (total_regions - total_iq_errors + rescued) / total_regions
        ),
        "gain_over_iq_percent_points": 100.0 * rescued / total_regions,
    }

    report = {
        "schema_version": 1,
        "analysis_type": "cross_location_64_feature_complementarity_pilot",
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "configuration": {
            "regions_per_source": regions_per_source,
            "seed": seed,
            "device": str(resolved_device),
            "feature_probe": "standardized_linear_64_to_3",
            "feature_extraction_condition": "clean",
            "control_matching": "same_test_fold_same_source_same_true_class",
        },
        "scope": {
            "full_iq_prediction_region_count": total_regions,
            "full_iq_error_count": total_iq_errors,
            "uniform_feature_probe_region_count": int(np.count_nonzero(uniform_mask)),
            "iq_error_feature_region_count": int(np.count_nonzero(error_mask)),
            "matched_control_feature_region_count": int(
                np.count_nonzero(control_mask)
            ),
            "unique_extracted_feature_region_count": int(bank["y"].size),
        },
        "folds": fold_summaries,
        "aggregate": aggregate,
        "interpretation_limits": [
            (
                "The error/control subset is outcome-selected and is only a "
                "diagnostic; its accuracy is not population accuracy."
            ),
            (
                "The oracle upper bound assumes a perfect selector and is not an "
                "implemented fusion rule."
            ),
            (
                "The linear probe measures whether the 64 features retain "
                "cross-location class information; it is not a proposed final model."
            ),
        ],
        "outputs": {
            "iq_predictions": str(prediction_path),
            "feature_bank": str(bank_path),
            "feature_predictions": str(output / "feature_predictions.npz"),
            "case_audit_csv": str(audit_path),
            "result": str(result_path),
        },
    }
    result_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the first-round 64-feature cross-location pilot."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regions-per-source", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_feature_complementarity_pilot(
        args.dataset_root,
        args.model_root,
        args.output_dir,
        regions_per_source=args.regions_per_source,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_feature_complementarity_pilot",
    "select_matched_correct_controls",
    "summarize_fold_complementarity",
    "uniform_indices_per_source",
]
