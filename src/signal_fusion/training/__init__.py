"""Offline model-training components.

Training dependencies are optional and intentionally are not imported when the
top-level :mod:`signal_fusion` package is loaded.
"""

from .augmentation import add_random_complex_awgn
from .exporter import export_onnx
from .fixed_splits import FixedSplitBundle, load_fixed_split_bundle
from .losses import LabelSmoothingCrossEntropy, LogitNormLoss
from .trainer import EarlyStopping, eval_epoch, run_training, train_epoch


__all__ = [
    "EarlyStopping",
    "FixedSplitBundle",
    "LabelSmoothingCrossEntropy",
    "LogitNormLoss",
    "add_random_complex_awgn",
    "eval_epoch",
    "export_onnx",
    "load_fixed_split_bundle",
    "run_training",
    "train_epoch",
]
