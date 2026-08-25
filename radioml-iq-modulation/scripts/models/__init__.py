"""Temporary wrapper for canonical signal_fusion model definitions."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.modeling import (  # noqa: E402
    DeepConvNet1D,
    DeepConvNet_1D,
    LSTMIQ,
    MODEL_REGISTRY,
    build_model,
)


__all__ = [
    "DeepConvNet1D",
    "DeepConvNet_1D",
    "LSTMIQ",
    "MODEL_REGISTRY",
    "build_model",
]
