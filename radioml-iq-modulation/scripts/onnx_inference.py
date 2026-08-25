"""Temporary wrapper for the signal_fusion ONNX inference runtime."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.model_inference.cli import (  # noqa: E402
    build_arg_parser,
    run_legacy_cli as main,
)
from signal_fusion.model_inference.legacy import (  # noqa: E402
    _sync_signal_inference,
    load_mod_labels,
    recognize_iq_modulation,
    run_onnx_inference,
)
from signal_fusion.preparation.legacy_burst import (  # noqa: E402, F401
    smart_extract_bursts,
)


__all__ = [
    "_sync_signal_inference",
    "build_arg_parser",
    "load_mod_labels",
    "main",
    "recognize_iq_modulation",
    "run_onnx_inference",
    "smart_extract_bursts",
]


if __name__ == "__main__":
    main()
