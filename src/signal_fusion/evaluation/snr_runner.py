"""End-to-end SNR robustness evaluation for fixed prepared test splits."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.evaluation.snr import add_complex_awgn, classification_metrics
from signal_fusion.io import load_prepared_dataset


def _metadata_scalar(meta: dict[str, Any], field_name: str) -> Any:
    if field_name not in meta:
        raise ValueError(f"test dataset is missing metadata {field_name!r}")
    value = np.asarray(meta[field_name])
    if value.size != 1:
        raise ValueError(f"metadata {field_name!r} must be scalar")
    return value.reshape(()).item()


def _sample_metadata(
    meta: dict[str, Any], field_name: str, sample_count: int
) -> np.ndarray:
    if field_name not in meta:
        raise ValueError(f"test dataset is missing metadata {field_name!r}")
    value = np.asarray(meta[field_name])
    if value.shape != (sample_count,):
        raise ValueError(
            f"metadata {field_name!r} must have shape [{sample_count}], "
            f"got {value.shape}"
        )
    return value


def _parse_label_map(meta: dict[str, Any]) -> dict[str, str]:
    raw = _metadata_scalar(meta, "label_map_json")
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError("test dataset label_map_json is invalid JSON") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("test dataset label_map_json must be a non-empty object")
    normalized = {str(key): str(value) for key, value in parsed.items()}
    if set(normalized) != {
        str(index) for index in range(len(normalized))
    }:
        raise ValueError("label_map_json must use continuous indices from zero")
    return normalized


def resolve_evaluation_device(requested: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "SNR model evaluation requires PyTorch. Install signal-fusion[training]."
        ) from exc
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def load_evaluation_model(
    model_path: Path,
    *,
    model_name: str,
    class_num: int,
    input_channels: int,
    seq_len: int,
    device,
):
    import torch

    from signal_fusion.modeling import build_model

    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {model_path}")
    model = build_model(
        model_name=model_name,
        class_num=class_num,
        input_channels=input_channels,
        seq_len=seq_len,
    )
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_probabilities(model, x: np.ndarray, *, batch_size: int, device):
    import torch

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            output = model(batch)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)


def _aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    labels = sorted(trials[0]["metrics"]["window"]["per_class"])
    return {
        "trial_count": len(trials),
        "window_accuracy_percent": stats(
            [trial["metrics"]["window"]["accuracy_percent"] for trial in trials]
        ),
        "group_accuracy_percent": stats(
            [trial["metrics"]["group"]["accuracy_percent"] for trial in trials]
        ),
        "window_mean_confidence": stats(
            [
                trial["metrics"]["window"]["mean_predicted_confidence"]
                for trial in trials
            ]
        ),
        "per_class_window_accuracy_percent": {
            label: {
                "class_name": trials[0]["metrics"]["window"]["per_class"][label][
                    "class_name"
                ],
                **stats(
                    [
                        trial["metrics"]["window"]["per_class"][label][
                            "accuracy_percent"
                        ]
                        for trial in trials
                    ]
                ),
            }
            for label in labels
        },
    }


def _write_csv(
    path: Path,
    conditions: list[dict[str, Any]],
    label_map: dict[str, str],
) -> None:
    class_fields = [
        f"window_accuracy_{label_map[str(index)]}_percent"
        for index in range(len(label_map))
    ]
    fields = [
        "condition",
        "snr_db",
        "trial_index",
        "seed",
        "window_accuracy_percent",
        "group_accuracy_percent",
        "window_mean_confidence",
        "achieved_snr_db_mean",
        "achieved_snr_db_std",
        *class_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in conditions:
            for trial in condition["trials"]:
                metrics = trial["metrics"]
                row: dict[str, Any] = {
                    "condition": condition["condition"],
                    "snr_db": condition["snr_db"],
                    "trial_index": trial["trial_index"],
                    "seed": trial["seed"],
                    "window_accuracy_percent": metrics["window"][
                        "accuracy_percent"
                    ],
                    "group_accuracy_percent": metrics["group"][
                        "accuracy_percent"
                    ],
                    "window_mean_confidence": metrics["window"][
                        "mean_predicted_confidence"
                    ],
                    "achieved_snr_db_mean": trial.get("achieved_snr_db_mean"),
                    "achieved_snr_db_std": trial.get("achieved_snr_db_std"),
                }
                for index in range(len(label_map)):
                    row[class_fields[index]] = metrics["window"]["per_class"][
                        str(index)
                    ]["accuracy_percent"]
                writer.writerow(row)


def _write_plot(path: Path, conditions: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "SNR curve generation requires matplotlib. "
            "Install signal-fusion[training]."
        ) from exc

    clean = next(item for item in conditions if item["snr_db"] is None)
    noisy = sorted(
        (item for item in conditions if item["snr_db"] is not None),
        key=lambda item: item["snr_db"],
    )
    snr_values = np.asarray([item["snr_db"] for item in noisy], dtype=np.float64)
    window_means = np.asarray(
        [item["aggregate"]["window_accuracy_percent"]["mean"] for item in noisy]
    )
    window_stds = np.asarray(
        [item["aggregate"]["window_accuracy_percent"]["std"] for item in noisy]
    )
    group_means = np.asarray(
        [item["aggregate"]["group_accuracy_percent"]["mean"] for item in noisy]
    )
    group_stds = np.asarray(
        [item["aggregate"]["group_accuracy_percent"]["std"] for item in noisy]
    )

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.errorbar(
        snr_values,
        window_means,
        yerr=window_stds,
        marker="o",
        capsize=3,
        label="Window accuracy",
    )
    axis.errorbar(
        snr_values,
        group_means,
        yerr=group_stds,
        marker="s",
        capsize=3,
        label="Region vote accuracy",
    )
    axis.axhline(
        clean["aggregate"]["window_accuracy_percent"]["mean"],
        color="tab:blue",
        linestyle="--",
        alpha=0.55,
        label="Clean window baseline",
    )
    axis.axhline(
        clean["aggregate"]["group_accuracy_percent"]["mean"],
        color="tab:orange",
        linestyle=":",
        alpha=0.65,
        label="Clean region baseline",
    )
    axis.set_xlabel("Added complex-AWGN SNR (dB)")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Signal classifier SNR robustness")
    axis.set_ylim(0.0, 101.0)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_snr_robustness(
    *,
    dataset_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    snr_db_values: list[float] | tuple[float, ...] = (
        20.0,
        15.0,
        10.0,
        5.0,
        0.0,
        -5.0,
        -10.0,
    ),
    trials: int = 5,
    seed: int = 44,
    batch_size: int = 256,
    device: str = "auto",
    model_name: str = "deepconvnet_1d",
    overwrite: bool = False,
    plot: bool = True,
) -> dict[str, Any]:
    """Evaluate a trained checkpoint on clean and controlled-noise test windows."""

    dataset_directory = Path(dataset_dir)
    test_path = dataset_directory / "test.npz"
    model_file = Path(model_path)
    output_directory = Path(output_dir)
    if not test_path.is_file():
        raise FileNotFoundError(f"fixed test split does not exist: {test_path}")
    if isinstance(trials, bool) or int(trials) != trials or trials <= 0:
        raise ValueError("trials must be a positive integer")
    trials = int(trials)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    snr_values = [float(value) for value in snr_db_values]
    if not snr_values or any(not math.isfinite(value) for value in snr_values):
        raise ValueError("snr_db_values must contain finite values")
    if len(set(snr_values)) != len(snr_values):
        raise ValueError("snr_db_values must not contain duplicates")

    json_path = output_directory / "snr_evaluation.json"
    csv_path = output_directory / "snr_accuracy.csv"
    plot_path = output_directory / "snr_accuracy_curve.png"
    planned = [json_path, csv_path, *([plot_path] if plot else [])]
    existing = [str(path) for path in planned if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "evaluation outputs already exist; use overwrite=True: "
            + ", ".join(existing)
        )

    dataset = load_prepared_dataset(test_path)
    if dataset.y is None:
        raise ValueError("test dataset has no labels")
    label_map = _parse_label_map(dataset.meta)
    class_num = len(label_map)
    labels = dataset.y.astype(np.int64, copy=False)
    if np.unique(labels).tolist() != list(range(class_num)):
        raise ValueError("test labels do not match label_map_json")
    group_ids = _sample_metadata(
        dataset.meta, "group_id", dataset.num_samples
    ).astype(np.int64, copy=False)
    source_ids = _sample_metadata(
        dataset.meta, "sample_source_id", dataset.num_samples
    ).astype(str)
    dataset_id = str(_metadata_scalar(dataset.meta, "dataset_id"))

    resolved_device = resolve_evaluation_device(device)
    model = load_evaluation_model(
        model_file,
        model_name=model_name,
        class_num=class_num,
        input_channels=int(dataset.X.shape[1]),
        seq_len=dataset.seq_len,
        device=resolved_device,
    )

    conditions: list[dict[str, Any]] = []
    clean_probabilities = predict_probabilities(
        model, dataset.X, batch_size=batch_size, device=resolved_device
    )
    clean_trial = {
        "trial_index": 0,
        "seed": None,
        "metrics": classification_metrics(
            clean_probabilities,
            labels,
            group_ids,
            source_ids,
            label_map,
        ),
    }
    conditions.append(
        {
            "condition": "clean",
            "snr_db": None,
            "trials": [clean_trial],
            "aggregate": _aggregate_trials([clean_trial]),
        }
    )

    for snr_db in snr_values:
        condition_trials: list[dict[str, Any]] = []
        for trial_index in range(trials):
            trial_seed = int(seed) + trial_index
            noisy_x, achieved = add_complex_awgn(
                dataset.X,
                snr_db,
                rng=np.random.default_rng(trial_seed),
                remove_dc=True,
                rms_normalize=True,
            )
            probabilities = predict_probabilities(
                model, noisy_x, batch_size=batch_size, device=resolved_device
            )
            condition_trials.append(
                {
                    "trial_index": trial_index,
                    "seed": trial_seed,
                    "achieved_snr_db_mean": float(np.mean(achieved)),
                    "achieved_snr_db_std": float(np.std(achieved)),
                    "metrics": classification_metrics(
                        probabilities,
                        labels,
                        group_ids,
                        source_ids,
                        label_map,
                    ),
                }
            )
        conditions.append(
            {
                "condition": "added_awgn",
                "snr_db": snr_db,
                "trials": condition_trials,
                "aggregate": _aggregate_trials(condition_trials),
            }
        )

    result = {
        "schema_version": 1,
        "evaluation_type": "incremental_complex_awgn_snr_sweep",
        "dataset_id": dataset_id,
        "test_path": str(test_path),
        "model_path": str(model_file),
        "model_name": model_name,
        "device": str(resolved_device),
        "sample_count": dataset.num_samples,
        "group_count": int(np.unique(group_ids).size),
        "seq_len": dataset.seq_len,
        "input_channels": int(dataset.X.shape[1]),
        "label_map": label_map,
        "configuration": {
            "snr_db_values": snr_values,
            "trials_per_snr": trials,
            "base_seed": int(seed),
            "batch_size": int(batch_size),
            "noise": "circular_complex_gaussian_exact_per_window_power",
            "snr_interpretation": "input_window_power_to_newly_added_noise_power",
            "post_noise_remove_dc": True,
            "post_noise_rms_normalize": True,
        },
        "conditions": conditions,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "plot": str(plot_path) if plot else None,
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(csv_path, conditions, label_map)
    if plot:
        _write_plot(plot_path, conditions)
    return result


__all__ = [
    "evaluate_snr_robustness",
    "load_evaluation_model",
    "predict_probabilities",
    "resolve_evaluation_device",
]
