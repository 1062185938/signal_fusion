"""Compatibility wrapper for the canonical model registry."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.modeling.registry import (  # noqa: E402
    MODEL_REGISTRY,
    build_model,
)


__all__ = ["MODEL_REGISTRY", "build_model"]
