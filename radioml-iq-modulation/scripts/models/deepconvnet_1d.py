"""Compatibility wrapper for the canonical DeepConvNet1D definition."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.modeling.architectures.deepconvnet_1d import (  # noqa: E402
    DeepConvNet1D,
    DeepConvNet_1D,
)


__all__ = ["DeepConvNet1D", "DeepConvNet_1D"]
