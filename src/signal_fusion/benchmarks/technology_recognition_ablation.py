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


__all__ = ["evaluate_ablation_matrix", "summarize_ablation_runs"]
