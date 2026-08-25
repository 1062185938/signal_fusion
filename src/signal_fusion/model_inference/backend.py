"""Optional ONNX Runtime backend for IQ small-model inference."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import ModelManifest, ShapeDimension


ONNX_DTYPE_NAMES = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
}


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX inference requires onnxruntime; install signal-fusion[inference]"
        ) from exc
    return ort


def preferred_execution_providers() -> list[str]:
    """Preserve the legacy CUDA-first, CPU-fallback provider policy."""

    ort = _import_onnxruntime()
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _normalize_runtime_shape(shape: Sequence[Any]) -> tuple[ShapeDimension, ...]:
    normalized: list[ShapeDimension] = []
    for dimension in shape:
        if isinstance(dimension, int):
            normalized.append(int(dimension) if dimension > 0 else None)
        elif isinstance(dimension, str) and dimension:
            normalized.append(dimension)
        else:
            normalized.append(None)
    return tuple(normalized)


class ONNXRuntimeBackend:
    """Load one ONNX session and return named raw output arrays."""

    def __init__(
        self,
        model_path: str | Path,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        ort = _import_onnxruntime()
        requested_providers = (
            list(providers) if providers is not None else preferred_execution_providers()
        )
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=requested_providers,
        )
        model_inputs = self.session.get_inputs()
        if len(model_inputs) != 1:
            raise ValueError(
                f"Expected exactly one ONNX input, got {len(model_inputs)}"
            )
        model_input = model_inputs[0]
        self.input_name = str(model_input.name)
        self.input_shape = _normalize_runtime_shape(model_input.shape)
        runtime_input_dtype = str(model_input.type)
        self.input_dtype = ONNX_DTYPE_NAMES.get(
            runtime_input_dtype,
            runtime_input_dtype,
        )

        model_outputs = self.session.get_outputs()
        if not model_outputs:
            raise ValueError("ONNX model must expose at least one output")
        self.output_names = tuple(str(output.name) for output in model_outputs)
        self.output_shapes = {
            str(output.name): _normalize_runtime_shape(output.shape)
            for output in model_outputs
        }
        active_providers = tuple(self.session.get_providers())
        self.providers = active_providers or tuple(requested_providers)
        self.provider = (
            "GPU" if "CUDAExecutionProvider" in self.providers else "CPU"
        )

    def run(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        inputs = np.ascontiguousarray(batch, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: inputs})
        if len(outputs) != len(self.output_names):
            raise RuntimeError(
                f"ONNX returned {len(outputs)} outputs; expected {len(self.output_names)}"
            )
        return {
            name: np.asarray(values)
            for name, values in zip(self.output_names, outputs)
        }

    def build_manifest(
        self,
        *,
        model_id: str,
        labels: Sequence[str],
        model_version: str | None = None,
        logits_output: str | None = None,
    ) -> ModelManifest:
        resolved_logits_output = logits_output or self.output_names[0]
        if resolved_logits_output not in self.output_names:
            raise ValueError(
                f"logits_output={resolved_logits_output!r} is not an ONNX output"
            )
        return ModelManifest(
            model_id=model_id,
            model_format="onnx",
            model_path=str(self.model_path),
            input_name=self.input_name,
            input_shape=self.input_shape,
            outputs=self.output_shapes,
            labels=tuple(labels),
            input_dtype=self.input_dtype,
            model_version=model_version,
            metadata={
                "logits_output": resolved_logits_output,
                "providers": list(self.providers),
            },
        )


__all__ = [
    "ONNXRuntimeBackend",
    "ONNX_DTYPE_NAMES",
    "preferred_execution_providers",
]
