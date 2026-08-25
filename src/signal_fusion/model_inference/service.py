"""PreparedDataset-facing service for small-model inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from signal_fusion.contracts import ModelManifest, PreparedDataset
from signal_fusion.model_inference.backend import ONNXRuntimeBackend
from signal_fusion.model_inference.contracts import ModelInferenceResult


class ModelBackend(Protocol):
    input_name: str
    input_shape: tuple[int | str | None, ...]
    input_dtype: str
    output_names: tuple[str, ...]
    provider: str

    def run(self, batch: np.ndarray) -> Mapping[str, np.ndarray]: ...


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Apply the same max-shifted softmax used by the legacy runtime."""

    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"logits must have non-empty shape [N, C], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must contain only finite values")
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return (exp_values / np.sum(exp_values, axis=1, keepdims=True)).astype(
        np.float32,
        copy=False,
    )


def _validate_input_shape(
    actual_shape: tuple[int, ...],
    expected_shape: tuple[int | str | None, ...],
) -> None:
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"Model input rank mismatch: expected {expected_shape}, got {actual_shape}"
        )
    for axis, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if axis == 0 or expected is None or isinstance(expected, str):
            continue
        if int(expected) != int(actual):
            raise ValueError(
                f"Model input shape mismatch at axis {axis}: "
                f"expected {expected_shape}, got {actual_shape}"
            )


def _normalize_sample_indices(
    dataset: PreparedDataset,
    sample_indices: np.ndarray | None,
    max_samples: int | None,
) -> np.ndarray:
    if sample_indices is not None and max_samples is not None:
        raise ValueError("sample_indices and max_samples cannot be used together")
    if sample_indices is None:
        count = dataset.num_samples
        if max_samples is not None:
            limit = int(max_samples)
            if limit <= 0:
                raise ValueError(f"max_samples must be positive, got {max_samples}")
            count = min(count, limit)
        indices = np.arange(count, dtype=np.int64)
    else:
        indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("Model inference requires at least one sample")
    if np.any(indices < 0) or np.any(indices >= dataset.num_samples):
        raise IndexError(
            f"sample_indices must be within [0, {dataset.num_samples - 1}]"
        )
    return indices


class ModelInferenceService:
    """Run a ModelManifest against canonical IQ samples."""

    def __init__(
        self,
        manifest: ModelManifest,
        backend: ModelBackend | None = None,
    ) -> None:
        if manifest.model_format != "onnx":
            raise ValueError(
                f"Unsupported model_format={manifest.model_format!r}; expected 'onnx'"
            )
        self.manifest = manifest
        self.backend = backend or ONNXRuntimeBackend(manifest.model_path)
        if self.backend.input_name != manifest.input_name:
            raise ValueError(
                f"Manifest input_name={manifest.input_name!r} does not match "
                f"backend input_name={self.backend.input_name!r}"
            )
        if self.backend.input_dtype != manifest.input_dtype:
            raise ValueError(
                f"Manifest input_dtype={manifest.input_dtype!r} does not match "
                f"backend input_dtype={self.backend.input_dtype!r}"
            )
        if set(self.backend.output_names) != set(manifest.outputs):
            raise ValueError(
                "Manifest outputs do not match backend outputs: "
                f"manifest={tuple(manifest.outputs)}, "
                f"backend={tuple(self.backend.output_names)}"
            )

    def predict(
        self,
        dataset: PreparedDataset,
        *,
        source_id: str | None = None,
        batch_size: int = 64,
        max_samples: int | None = None,
        sample_indices: np.ndarray | None = None,
    ) -> ModelInferenceResult:
        resolved_source_id = source_id or dataset.source_id
        if resolved_source_id is None and "source_id" in dataset.meta:
            resolved_source_id = str(np.asarray(dataset.meta["source_id"]).item())
        if not isinstance(resolved_source_id, str) or not resolved_source_id.strip():
            raise ValueError("Model inference requires dataset.source_id or source_id")

        resolved_batch_size = int(batch_size)
        if resolved_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        indices = _normalize_sample_indices(dataset, sample_indices, max_samples)
        selected_x = np.ascontiguousarray(dataset.X[indices], dtype=np.float32)
        _validate_input_shape(tuple(selected_x.shape), self.manifest.input_shape)
        _validate_input_shape(tuple(selected_x.shape), self.backend.input_shape)

        output_chunks: dict[str, list[np.ndarray]] = {
            name: [] for name in self.backend.output_names
        }
        for start in range(0, selected_x.shape[0], resolved_batch_size):
            batch_outputs = dict(
                self.backend.run(selected_x[start : start + resolved_batch_size])
            )
            if set(batch_outputs) != set(self.backend.output_names):
                raise RuntimeError(
                    "Backend output names changed: "
                    f"expected={self.backend.output_names}, "
                    f"actual={tuple(batch_outputs)}"
                )
            for name in self.backend.output_names:
                values = batch_outputs[name]
                array = np.asarray(values)
                if array.ndim == 0:
                    raise ValueError(f"Model output {name!r} must include a batch axis")
                output_chunks[name].append(array)

        outputs = {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in output_chunks.items()
        }
        logits_name = str(
            self.manifest.metadata.get("logits_output", self.backend.output_names[0])
        )
        if logits_name not in outputs:
            raise ValueError(f"Configured logits output does not exist: {logits_name}")
        logits = np.asarray(outputs.pop(logits_name), dtype=np.float32)
        if logits.ndim != 2:
            raise ValueError(
                f"Logits output {logits_name!r} must have shape [N, C], "
                f"got {logits.shape}"
            )
        if logits.shape[1] != len(self.manifest.labels):
            raise ValueError(
                "label_map 与模型输出标签类别不匹配 "
                f"(label_map={len(self.manifest.labels)}, "
                f"model_outputs={logits.shape[1]})"
            )
        probabilities = stable_softmax(logits)

        return ModelInferenceResult(
            source_id=resolved_source_id,
            model_id=self.manifest.model_id,
            labels=self.manifest.labels,
            logits=logits,
            probabilities=probabilities,
            auxiliary_outputs=outputs,
            sample_indices=indices,
            provider=self.backend.provider,
            metadata={
                "input_name": self.manifest.input_name,
                "input_shape": list(self.manifest.input_shape),
                "logits_output": logits_name,
            },
        )


__all__ = ["ModelBackend", "ModelInferenceService", "stable_softmax"]
