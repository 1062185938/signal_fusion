"""Compatibility wrapper for the canonical LSTMIQ definition."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.modeling.architectures.lstm_iq import LSTMIQ  # noqa: E402


__all__ = ["LSTMIQ"]
