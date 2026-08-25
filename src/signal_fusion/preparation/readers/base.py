"""Base interface for raw recording readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from signal_fusion.preparation.contracts import RawSignal


class RawSignalReader(ABC):
    """Open a raw capture without imposing a signal-detection strategy."""

    format_name: str

    @abstractmethod
    def open(
        self,
        path: str | Path,
        *,
        source_id: str,
        **kwargs: Any,
    ) -> RawSignal:
        raise NotImplementedError

