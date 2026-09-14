"""Evaluation of a trained classifier on extracted 62-feature datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.evaluation.snr import classification_metrics
from signal_fusion.io.writers import json_safe

from .dataset import load_region_feature_split
from .service import FeatureClassifierService


def evaluate_feature_classifier(
    dataset_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate one ONNX feature classifier without changing its inputs."""

    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"feature evaluation output exists; use overwrite=True: {output}"
        )
    arrays = load_region_feature_split(dataset_path)
    label_map = {
        str(index): str(name)
        for index, name in json.loads(
            str(arrays["label_map_json"].reshape(()).item())
        ).items()
    }
    service = FeatureClassifierService(manifest_path)
    expected_labels = tuple(
        label_map[str(index)] for index in range(len(label_map))
    )
    if service.labels != expected_labels:
        raise ValueError(
            "feature dataset labels do not match classifier labels: "
            f"dataset={expected_labels}, classifier={service.labels}"
        )
    result = service.predict(arrays["features"])
    metrics = classification_metrics(
        result.probabilities,
        arrays["y"],
        arrays["group_id"],
        arrays["sample_source_id"].astype(str),
        label_map,
    )
    report = {
        "schema_version": 1,
        "evaluation_type": "feature_classifier",
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_id": str(arrays["dataset_id"].reshape(()).item()),
        "model_manifest": str(Path(manifest_path).resolve()),
        "model_id": result.model_id,
        "provider": result.provider,
        "sample_count": int(arrays["features"].shape[0]),
        "label_map": label_map,
        "metrics": metrics,
        "output_path": str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


__all__ = ["evaluate_feature_classifier"]
