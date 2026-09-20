"""Train, evaluate, and export the linear region-feature classifier."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from signal_fusion.feature_classifier.dataset import load_region_feature_split
from signal_fusion.feature_extraction import FEATURE_COUNT, FEATURE_SCHEMA_ID


def _scalar(arrays: dict[str, np.ndarray], name: str) -> str:
    if name not in arrays or arrays[name].size != 1:
        raise ValueError(f"feature split field {name!r} must be scalar")
    return str(arrays[name].reshape(()).item())


def _parse_label_map(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    raw = json.loads(_scalar(arrays, "label_map_json"))
    label_map = {str(index): str(value) for index, value in raw.items()}
    if set(label_map) != {str(index) for index in range(len(label_map))}:
        raise ValueError("label map must use continuous indices from zero")
    return label_map


def _resolve_device(requested: str):
    import torch

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    label_map: dict[str, str],
) -> dict[str, Any]:
    predictions = np.asarray(logits).argmax(axis=1)
    class_count = len(label_map)
    confusion = [
        [
            int(np.count_nonzero((labels == true) & (predictions == predicted)))
            for predicted in range(class_count)
        ]
        for true in range(class_count)
    ]
    return {
        "sample_count": int(labels.size),
        "correct_count": int(np.count_nonzero(predictions == labels)),
        "accuracy_percent": float(np.mean(predictions == labels) * 100.0),
        "confusion_matrix": confusion,
        "per_class_accuracy_percent": {
            label_map[str(label)]: float(
                np.mean(predictions[labels == label] == label) * 100.0
            )
            for label in range(class_count)
        },
    }


def train_feature_classifier(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    batch_size: int = 64,
    learning_rate: float = 1e-2,
    max_epochs: int = 300,
    patience: int = 30,
    seed: int = 44,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train a standardized Linear(64, class_count) baseline."""

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from signal_fusion.feature_classifier.model import LinearFeatureClassifier

    if batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("batch_size, max_epochs, and patience must be positive")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(seed, bool) or int(seed) != seed:
        raise TypeError("seed must be an integer")
    seed = int(seed)

    source_directory = Path(dataset_dir).resolve()
    splits = {
        name: load_region_feature_split(source_directory / f"{name}.npz")
        for name in ("train", "validation", "test")
    }
    dataset_ids = {_scalar(value, "dataset_id") for value in splits.values()}
    if len(dataset_ids) != 1:
        raise ValueError("feature split dataset_id values differ")
    feature_names = tuple(str(name) for name in splits["train"]["feature_names"])
    label_map = _parse_label_map(splits["train"])
    for split_name, arrays in splits.items():
        if tuple(str(name) for name in arrays["feature_names"]) != feature_names:
            raise ValueError(f"{split_name} feature_names differ from train")
        if _parse_label_map(arrays) != label_map:
            raise ValueError(f"{split_name} label map differs from train")

    output_directory = Path(output_dir)
    paths = {
        "checkpoint": output_directory / "best_model.pth",
        "onnx": output_directory / "feature_classifier.onnx",
        "scaler": output_directory / "feature_scaler.npz",
        "manifest": output_directory / "feature_classifier_manifest.json",
        "result": output_directory / "training_result.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "feature classifier outputs exist; use overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    train_features = splits["train"]["features"].astype(np.float64)
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    constant_mask = scale <= 1e-12
    scale[constant_mask] = 1.0
    mean = mean.astype(np.float32)
    scale = scale.astype(np.float32)
    np.savez_compressed(
        paths["scaler"],
        mean=mean,
        scale=scale,
        constant_feature_mask=constant_mask,
    )

    standardized = {
        name: ((arrays["features"] - mean) / scale).astype(np.float32)
        for name, arrays in splits.items()
    }
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = _resolve_device(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(standardized["train"]),
            torch.from_numpy(splits["train"]["y"]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = LinearFeatureClassifier(FEATURE_COUNT, len(label_map)).to(resolved_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    def evaluate(split_name: str) -> tuple[float, np.ndarray]:
        model.eval()
        x = torch.from_numpy(standardized[split_name]).to(resolved_device)
        y = torch.from_numpy(splits[split_name]["y"]).to(resolved_device)
        with torch.no_grad():
            logits = model(x)
            loss = float(criterion(logits, y).item())
        return loss, logits.cpu().numpy()

    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(resolved_device)
            y = y.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        validation_loss, validation_logits = evaluate("validation")
        validation_accuracy = float(
            np.mean(
                validation_logits.argmax(axis=1) == splits["validation"]["y"]
            )
            * 100.0
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), paths["checkpoint"])
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[feature-classifier] epoch={epoch} "
                f"val_loss={validation_loss:.6f} val_acc={validation_accuracy:.2f}%"
            )
        if stale_epochs >= patience:
            break

    model.load_state_dict(
        torch.load(paths["checkpoint"], map_location=resolved_device, weights_only=True)
    )
    split_metrics: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        _, logits = evaluate(split_name)
        split_metrics[split_name] = _metrics(
            logits, splits[split_name]["y"], label_map
        )
        if split_name == "test":
            split_metrics[split_name]["conditions"] = {}
            for condition in np.unique(splits[split_name]["condition"]):
                mask = splits[split_name]["condition"] == condition
                split_metrics[split_name]["conditions"][str(condition)] = _metrics(
                    logits[mask], splits[split_name]["y"][mask], label_map
                )

    model.eval()
    dummy = torch.zeros(1, FEATURE_COUNT, device=resolved_device)
    torch.onnx.export(
        model,
        dummy,
        paths["onnx"],
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    model_id = output_directory.name
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "model_type": f"linear_{FEATURE_COUNT}_to_classes",
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_names": list(feature_names),
        "labels": [label_map[str(index)] for index in range(len(label_map))],
        "input_name": "features",
        "input_shape": [None, FEATURE_COUNT],
        "output_name": "logits",
        "output_shape": [None, len(label_map)],
        "model_path": paths["onnx"].name,
        "scaler_path": paths["scaler"].name,
        "training_dataset_id": next(iter(dataset_ids)),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = {
        "schema_version": 1,
        "model_id": model_id,
        "model_type": manifest["model_type"],
        "device": resolved_device.type,
        "feature_count": FEATURE_COUNT,
        "class_count": len(label_map),
        "label_map": label_map,
        "dataset_dir": str(source_directory),
        "dataset_id": next(iter(dataset_ids)),
        "configuration": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "patience": patience,
            "seed": seed,
            "standardization": "train_split_mean_std",
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": epoch,
        "elapsed_sec": time.time() - start_time,
        "constant_feature_count": int(np.count_nonzero(constant_mask)),
        "metrics": split_metrics,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["result"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


__all__ = ["train_feature_classifier"]
