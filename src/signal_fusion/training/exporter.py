"""ONNX export and runtime verification for trained IQ models."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import torch


def export_onnx(model, filename, seq_len, device, input_channels=2):
    model.eval()
    dummy_input = torch.randn(1, input_channels, seq_len).to(device)
    export_kwargs = dict(
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output", "feature"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
            "feature": {0: "batch_size"},
        },
    )
    try:
        torch.onnx.export(model, dummy_input, filename, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, filename, **export_kwargs)
    print(f"\n[ONNX] Model exported successfully to: {filename}")

    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model)
    print("[ONNX] Model verification OK!")

    session = ort.InferenceSession(filename, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, input_channels, seq_len).astype(np.float32)
    y, feat = session.run(None, {"input": x})
    print(f"[ONNX] Test inference output shape: y={y.shape}, feature={feat.shape}")


__all__ = ["export_onnx"]
