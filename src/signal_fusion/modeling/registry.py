"""Model registry shared by training and model-development workflows."""

from __future__ import annotations

from signal_fusion.modeling.architectures import DeepConvNet1D, LSTMIQ


MODEL_REGISTRY = {
    "deepconvnet_1d": DeepConvNet1D,
    "lstm_iq": LSTMIQ,
}


def build_model(
    model_name,
    class_num,
    input_channels=2,
    seq_len=128,
    **kwargs,
):
    normalized_name = str(model_name).lower()
    if normalized_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Available models: {available}"
        )
    model_class = MODEL_REGISTRY[normalized_name]
    return model_class(
        class_num=class_num,
        input_channels=input_channels,
        seq_len=seq_len,
        **kwargs,
    )


__all__ = ["MODEL_REGISTRY", "build_model"]
