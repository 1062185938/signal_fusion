"""Temporary compatibility wrapper for prepared-dataset inspection."""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.preparation.inspection import (  # noqa: E402
    WindowSelection,
    inspect_sigmf_slices,
    main,
)


__all__ = ["WindowSelection", "inspect_sigmf_slices", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
