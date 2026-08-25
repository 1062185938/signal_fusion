"""Offline model-training components.

Training dependencies are optional and intentionally are not imported when the
top-level :mod:`signal_fusion` package is loaded.
"""

from .exporter import export_onnx
from .losses import LabelSmoothingCrossEntropy, LogitNormLoss
from .trainer import EarlyStopping, eval_epoch, run_training, train_epoch


__all__ = [
    "EarlyStopping",
    "LabelSmoothingCrossEntropy",
    "LogitNormLoss",
    "eval_epoch",
    "export_onnx",
    "run_training",
    "train_epoch",
]
