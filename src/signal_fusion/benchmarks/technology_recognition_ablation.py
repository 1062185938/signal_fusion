"""Evaluation for the public-dataset location and window-length matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.evaluation.snr import classification_metrics
from signal_fusion.evaluation.snr_runner import (
    load_evaluation_model,
    predict_probabilities,
    resolve_evaluation_device,
)
from signal_fusion.io import load_prepared_dataset

from .technology_recognition import LABELS, LOCATION_FOLDS, WINDOW_SIZES


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_ablation_runs(
    runs: list[dict[str, Any]],
    *,
    window_sizes: tuple[int, ...] = WINDOW_SIZES,
) -> dict[str, Any]:
    """Aggregate per-fold clean evaluation metrics by window length."""

    summaries: dict[str, Any] = {}
    for window_size in window_sizes:
        selected = [run for run in runs if run["window_size"] == window_size]
        if len(selected) != len(LOCATION_FOLDS):
            raise ValueError(
                f"window_size={window_size} must contain "
                f"{len(LOCATION_FOLDS)} folds"
            )
        summaries[str(window_size)] = {
            "fold_count": len(selected),
            "window_accuracy_percent": _stats(
                [run["window"]["accuracy_percent"] for run in selected]
            ),
            "region_accuracy_percent": _stats(
                [run["region"]["accuracy_percent"] for run in selected]
            ),
            "window_mean_confidence": _stats(
                [run["window"]["mean_predicted_confidence"] for run in selected]
            ),
            "per_class_window_accuracy_percent": {
                name: _stats(
                    [
                        run["window"]["per_class"][str(label)][
                            "accuracy_percent"
                        ]
                        for run in selected
                    ]
                )
                for name, label in LABELS.items()
            },
        }
    return summaries


def _write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "window_size",
        "fold",
        "validation_location",
        "test_location",
        "best_epoch",
        "best_validation_accuracy_percent",
        "window_accuracy_percent",
        "region_accuracy_percent",
        "window_mean_confidence",
        "window_accuracy_LTE_percent",
        "window_accuracy_WiFi_percent",
        "window_accuracy_DVB-T_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "window_size": run["window_size"],
                "fold": run["fold"],
                "validation_location": run["locations"]["validation"][0],
                "test_location": run["locations"]["test"][0],
                "best_epoch": run["training"]["best_epoch"],
                "best_validation_accuracy_percent": run["training"][
                    "best_validation_accuracy_percent"
                ],
                "window_accuracy_percent": run["window"]["accuracy_percent"],
                "region_accuracy_percent": run["region"]["accuracy_percent"],
                "window_mean_confidence": run["window"][
                    "mean_predicted_confidence"
                ],
            }
            for name, label in LABELS.items():
                row[f"window_accuracy_{name}_percent"] = run["window"][
                    "per_class"
                ][str(label)]["accuracy_percent"]
            writer.writerow(row)


def _write_plot(
    path: Path,
    summaries: dict[str, Any],
    window_sizes: tuple[int, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = np.asarray(window_sizes)
    window_mean = [
        summaries[str(size)]["window_accuracy_percent"]["mean"] for size in sizes
    ]
    window_min = [
        summaries[str(size)]["window_accuracy_percent"]["min"] for size in sizes
    ]
    window_max = [
        summaries[str(size)]["window_accuracy_percent"]["max"] for size in sizes
    ]
    region_mean = [
        summaries[str(size)]["region_accuracy_percent"]["mean"] for size in sizes
    ]
    region_min = [
        summaries[str(size)]["region_accuracy_percent"]["min"] for size in sizes
    ]
    region_max = [
        summaries[str(size)]["region_accuracy_percent"]["max"] for size in sizes
    ]
    window_error = np.asarray(
        [
            np.asarray(window_mean) - np.asarray(window_min),
            np.asarray(window_max) - np.asarray(window_mean),
        ]
    )
    region_error = np.asarray(
        [
            np.asarray(region_mean) - np.asarray(region_min),
            np.asarray(region_max) - np.asarray(region_mean),
        ]
    )

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.errorbar(
        sizes,
        window_mean,
        yerr=window_error,
        marker="o",
        capsize=4,
        label="Window accuracy (fold min–max)",
    )
    axis.errorbar(
        sizes,
        region_mean,
        yerr=region_error,
        marker="s",
        capsize=4,
        label="4096-point region accuracy (fold min–max)",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes, [str(size) for size in sizes])
    axis.set_ylim(65.0, 101.0)
    axis.set_xlabel("Window length (complex samples)")
    axis.set_ylabel("Four-fold accuracy (%)")
    axis.set_title("Technology Recognition window-length ablation")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_ablation_matrix(
    *,
    dataset_root: str | Path,
    model_root: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    batch_size: int = 256,
    window_sizes: tuple[int, ...] = WINDOW_SIZES,
) -> dict[str, Any]:
    """Evaluate clean models at window and 4096-point region levels."""

    dataset_base = Path(dataset_root)
    model_base = Path(model_root)
    output = Path(output_dir)
    json_path = output / "ablation_evaluation.json"
    csv_path = output / "ablation_runs.csv"
    plot_path = output / "ablation_accuracy.png"
    existing = [path for path in (json_path, csv_path, plot_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "ablation evaluation outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not window_sizes:
        raise ValueError("window_sizes must not be empty")
    if len(set(window_sizes)) != len(window_sizes):
        raise ValueError("window_sizes must not contain duplicates")
    unsupported = [size for size in window_sizes if size not in WINDOW_SIZES]
    if unsupported:
        raise ValueError(f"unsupported window sizes: {unsupported}")

    resolved_device = resolve_evaluation_device(device)
    expected_label_map = {
        str(label): name for name, label in LABELS.items()
    }
    runs: list[dict[str, Any]] = []
    for window_size in window_sizes:
        for fold_name, locations in LOCATION_FOLDS.items():
            dataset_dir = (
                dataset_base / f"window_{window_size}" / fold_name
            )
            model_dir = model_base / f"window_{window_size}" / fold_name
            dataset = load_prepared_dataset(dataset_dir / "test.npz")
            label_map = json.loads(
                str(np.asarray(dataset.meta["label_map_json"]).item())
            )
            if label_map != expected_label_map:
                raise ValueError(
                    f"unexpected label map for window={window_size}, {fold_name}"
                )
            model = load_evaluation_model(
                model_dir / "best_model.pth",
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
            metrics = classification_metrics(
                probabilities,
                dataset.y,
                np.asarray(dataset.meta["group_id"]),
                np.asarray(dataset.meta["sample_source_id"]),
                label_map,
            )
            validation_loss = np.load(model_dir / "val_loss.npy")
            validation_accuracy = np.load(model_dir / "val_acc.npy")
            best_index = int(np.argmin(validation_loss))
            training_result = json.loads(
                (model_dir / "training_result.json").read_text(encoding="utf-8")
            )
            runs.append(
                {
                    "window_size": window_size,
                    "windows_per_region": 4096 // window_size,
                    "fold": fold_name,
                    "locations": {
                        split: list(split_locations)
                        for split, split_locations in locations.items()
                    },
                    "dataset_id": training_result["dataset_id"],
                    "training": {
                        "best_epoch": best_index + 1,
                        "best_validation_loss": float(
                            validation_loss[best_index]
                        ),
                        "best_validation_accuracy_percent": float(
                            validation_accuracy[best_index]
                        ),
                        "epochs_completed": training_result[
                            "epochs_completed"
                        ],
                        "elapsed_sec": training_result["elapsed_sec"],
                    },
                    "window": metrics["window"],
                    "region": metrics["group"],
                }
            )

    summaries = summarize_ablation_runs(runs, window_sizes=window_sizes)
    result = {
        "schema_version": 1,
        "evaluation_type": "location_rotated_clean_window_length_ablation",
        "device": str(resolved_device),
        "model_name": "deepconvnet_1d",
        "region_size": 4096,
        "window_sizes": list(window_sizes),
        "folds": {
            name: {
                split: list(locations)
                for split, locations in fold.items()
            }
            for name, fold in LOCATION_FOLDS.items()
        },
        "training_controls": {
            "artificial_awgn": False,
            "seed": 44,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "cross_entropy",
            "batch_size": 256,
            "max_epoch": 30,
            "early_stopping_patience": 8,
        },
        "aggregation": "mean window probabilities within each 4096-point region",
        "summaries": summaries,
        "runs": runs,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "plot": str(plot_path),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(csv_path, runs)
    _write_plot(plot_path, summaries, window_sizes)
    return result


def _compact_classification_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    window = metrics["window"]
    per_class = window["per_class"]
    class_accuracies = [
        item["accuracy_percent"]
        for item in per_class.values()
        if item["accuracy_percent"] is not None
    ]
    return {
        "sample_count": window["sample_count"],
        "correct_count": window["correct_count"],
        "accuracy_percent": window["accuracy_percent"],
        "macro_accuracy_percent": float(np.mean(class_accuracies)),
        "mean_predicted_confidence": window["mean_predicted_confidence"],
        "confusion_matrix": window["confusion_matrix"],
        "per_class": per_class,
    }


def evaluate_external_test_set(
    *,
    dataset_path: str | Path,
    model_root: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate all four location-fold checkpoints on one external test set."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = Path(output_dir)
    json_path = output / "external_evaluation.json"
    summary_csv_path = output / "external_model_results.csv"
    source_csv_path = output / "external_source_results.csv"
    existing = [
        path
        for path in (json_path, summary_csv_path, source_csv_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "external evaluation outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )

    dataset = load_prepared_dataset(dataset_path)
    if dataset.y is None or dataset.seq_len != 4096:
        raise ValueError("external test dataset must contain labeled 4096-point IQ")
    label_map = json.loads(
        str(np.asarray(dataset.meta["label_map_json"]).item())
    )
    expected_label_map = {
        str(label): name for name, label in LABELS.items()
    }
    if label_map != expected_label_map:
        raise ValueError("external test label map does not match the benchmark")

    sample_count = dataset.num_samples
    group_ids = np.asarray(dataset.meta["group_id"], dtype=np.int64)
    source_ids = np.asarray(dataset.meta["sample_source_id"]).astype(str)
    locations = np.asarray(dataset.meta["sample_location"]).astype(str)
    evaluation_groups = np.asarray(
        dataset.meta["sample_evaluation_group"]
    ).astype(str)
    center_frequencies = np.asarray(
        dataset.meta["center_frequency"], dtype=np.float64
    )
    for name, values in (
        ("group_id", group_ids),
        ("sample_source_id", source_ids),
        ("sample_location", locations),
        ("sample_evaluation_group", evaluation_groups),
        ("center_frequency", center_frequencies),
    ):
        if values.shape != (sample_count,):
            raise ValueError(f"external metadata {name!r} has invalid shape")

    subset_masks: dict[str, np.ndarray] = {
        "all": np.ones(sample_count, dtype=bool),
        "unseen_location": evaluation_groups == "unseen_location",
        "unseen_frequency": evaluation_groups == "unseen_frequency",
    }
    subset_masks.update(
        {
            f"location_{location}": locations == location
            for location in sorted(set(locations.tolist()))
        }
    )
    if any(not np.any(mask) for mask in subset_masks.values()):
        raise ValueError("external evaluation contains an empty reporting subset")

    resolved_device = resolve_evaluation_device(device)
    probability_sets: list[np.ndarray] = []
    model_results: list[dict[str, Any]] = []
    compact_source_results: list[dict[str, Any]] = []

    def evaluate_probabilities(
        model_id: str, probabilities: np.ndarray
    ) -> dict[str, Any]:
        subsets: dict[str, Any] = {}
        full_metrics: dict[str, Any] | None = None
        for subset_name, mask in subset_masks.items():
            metrics = classification_metrics(
                probabilities[mask],
                dataset.y[mask],
                group_ids[mask],
                source_ids[mask],
                label_map,
            )
            subsets[subset_name] = _compact_classification_metrics(metrics)
            if subset_name == "all":
                full_metrics = metrics
        if full_metrics is None:
            raise RuntimeError("external all-data metrics were not computed")
        for source_id, source_metrics in full_metrics["per_source"].items():
            mask = source_ids == source_id
            unique_location = np.unique(locations[mask])
            unique_group = np.unique(evaluation_groups[mask])
            unique_frequency = np.unique(center_frequencies[mask])
            if (
                unique_location.size != 1
                or unique_group.size != 1
                or unique_frequency.size != 1
            ):
                raise ValueError(f"source metadata is inconsistent: {source_id}")
            compact_source_results.append(
                {
                    "model_id": model_id,
                    "source_id": source_id,
                    "location": str(unique_location[0]),
                    "evaluation_group": str(unique_group[0]),
                    "center_frequency_hz": float(unique_frequency[0]),
                    "class_name": source_metrics["class_name"],
                    "sample_count": source_metrics["sample_count"],
                    "accuracy_percent": source_metrics["accuracy_percent"],
                    "mean_predicted_confidence": source_metrics[
                        "mean_predicted_confidence"
                    ],
                }
            )
        return {"model_id": model_id, "subsets": subsets}

    model_base = Path(model_root)
    for fold_name, fold_locations in LOCATION_FOLDS.items():
        model_path = model_base / fold_name / "best_model.pth"
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
        probability_sets.append(probabilities)
        result = evaluate_probabilities(fold_name, probabilities)
        result["training_locations"] = list(fold_locations["train"])
        model_results.append(result)

    ensemble_probabilities = np.mean(probability_sets, axis=0)
    ensemble_result = evaluate_probabilities("ensemble_mean", ensemble_probabilities)
    predictions = np.stack(
        [probabilities.argmax(axis=1) for probabilities in probability_sets],
        axis=0,
    )
    agreement = {
        subset_name: {
            "sample_count": int(np.count_nonzero(mask)),
            "all_four_agree_count": int(
                np.count_nonzero(
                    np.all(predictions[:, mask] == predictions[0:1, mask], axis=0)
                )
            ),
        }
        for subset_name, mask in subset_masks.items()
    }
    for values in agreement.values():
        values["all_four_agree_ratio"] = (
            values["all_four_agree_count"] / values["sample_count"]
        )

    result = {
        "schema_version": 1,
        "evaluation_type": "external_clean_multi_checkpoint",
        "dataset_id": str(np.asarray(dataset.meta["dataset_id"]).item()),
        "dataset_path": str(dataset_path),
        "device": str(resolved_device),
        "model_name": "deepconvnet_1d",
        "window_size": 4096,
        "source_count": len(set(source_ids.tolist())),
        "sample_count": sample_count,
        "label_map": label_map,
        "models": model_results,
        "ensemble": ensemble_result,
        "model_agreement": agreement,
        "outputs": {
            "json": str(json_path),
            "summary_csv": str(summary_csv_path),
            "source_csv": str(source_csv_path),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_fields = [
        "model_id",
        "subset",
        "sample_count",
        "accuracy_percent",
        "macro_accuracy_percent",
        "mean_predicted_confidence",
        "accuracy_LTE_percent",
        "accuracy_WiFi_percent",
        "accuracy_DVB-T_percent",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for model_result in [*model_results, ensemble_result]:
            for subset_name, metrics in model_result["subsets"].items():
                row = {
                    "model_id": model_result["model_id"],
                    "subset": subset_name,
                    "sample_count": metrics["sample_count"],
                    "accuracy_percent": metrics["accuracy_percent"],
                    "macro_accuracy_percent": metrics["macro_accuracy_percent"],
                    "mean_predicted_confidence": metrics[
                        "mean_predicted_confidence"
                    ],
                }
                for name, label in LABELS.items():
                    row[f"accuracy_{name}_percent"] = metrics["per_class"][
                        str(label)
                    ]["accuracy_percent"]
                writer.writerow(row)

    source_fields = [
        "model_id",
        "source_id",
        "location",
        "evaluation_group",
        "center_frequency_hz",
        "class_name",
        "sample_count",
        "accuracy_percent",
        "mean_predicted_confidence",
    ]
    with source_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(compact_source_results)
    return result


__all__ = [
    "evaluate_ablation_matrix",
    "evaluate_external_test_set",
    "summarize_ablation_runs",
]
