"""Reader for continuous IQ recordings stored in MATLAB files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from signal_fusion.preparation.contracts import RawSignal
from signal_fusion.preparation.readers.base import RawSignalReader


_IQ_KEYS = ("iq", "IQ", "signal", "data", "X")
_SAMPLE_RATE_KEYS = ("sample_rate", "Sample_rate", "SampleRate", "fs", "Fs")
_CENTER_FREQUENCY_KEYS = (
    "center_frequency",
    "CenterFrequency",
    "center_freq",
    "fc",
)


def _available_keys(container: dict[str, Any]) -> list[str]:
    return [key for key in container if not key.startswith("__")]


def _select_key(
    container: dict[str, Any],
    explicit_key: str | None,
    candidates: tuple[str, ...],
    value_name: str,
    required: bool,
) -> str | None:
    keys = _available_keys(container)
    if explicit_key is not None:
        if explicit_key not in keys:
            raise KeyError(
                f"Requested {value_name} key {explicit_key!r} not found. "
                f"Available keys: {keys}"
            )
        return explicit_key
    for candidate in candidates:
        if candidate in keys:
            return candidate
    if required:
        raise KeyError(
            f"Cannot find {value_name}. Tried {candidates}; available keys: {keys}"
        )
    return None


def _scalar(container: dict[str, Any], key: str) -> Any:
    value = np.asarray(container[key]).squeeze()
    if value.size != 1:
        raise ValueError(f"MAT metadata field {key!r} must be scalar")
    return value.item()


def _continuous_complex_iq(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if array.ndim == 1 or (array.ndim == 2 and 1 in array.shape):
            return array.reshape(-1).astype(np.complex64, copy=False)
        raise ValueError(
            "Raw MAT IQ must be a continuous complex vector; "
            f"got shape={array.shape}"
        )

    if array.ndim == 2 and array.shape[0] == 2:
        return (array[0] + 1j * array[1]).reshape(-1).astype(np.complex64)
    if array.ndim == 2 and array.shape[1] == 2:
        return (array[:, 0] + 1j * array[:, 1]).reshape(-1).astype(np.complex64)
    raise ValueError(
        "Raw MAT IQ must be complex [N], [N,1], [1,N], or real [2,N]/[N,2]; "
        f"got shape={array.shape}, dtype={array.dtype}"
    )


class MatReader(RawSignalReader):
    format_name = "mat"

    def open(
        self,
        path: str | Path,
        *,
        source_id: str,
        x_key: str | None = None,
        sample_rate: float | None = None,
        sample_rate_key: str | None = None,
        center_frequency: float | None = None,
        center_frequency_key: str | None = None,
    ) -> RawSignal:
        mat_path = Path(path)
        if not mat_path.is_file():
            raise FileNotFoundError(f"MAT file not found: {mat_path}")
        container = sio.loadmat(mat_path)

        selected_x_key = _select_key(
            container, x_key, _IQ_KEYS, "IQ", required=True
        )
        samples = _continuous_complex_iq(container[selected_x_key])

        selected_sample_rate_key = _select_key(
            container,
            sample_rate_key,
            _SAMPLE_RATE_KEYS,
            "sample rate",
            required=sample_rate is None,
        )
        resolved_sample_rate = (
            float(sample_rate)
            if sample_rate is not None
            else float(_scalar(container, selected_sample_rate_key))
        )

        selected_center_key = _select_key(
            container,
            center_frequency_key,
            _CENTER_FREQUENCY_KEYS,
            "center frequency",
            required=False,
        )
        resolved_center_frequency = center_frequency
        if resolved_center_frequency is None and selected_center_key is not None:
            resolved_center_frequency = float(_scalar(container, selected_center_key))

        def read_samples(start: int, count: int) -> np.ndarray:
            return samples[start : start + count]

        return RawSignal(
            source_id=source_id,
            source_path=str(mat_path),
            sample_rate=resolved_sample_rate,
            sample_count=int(samples.size),
            sample_format="complex64",
            center_frequency=resolved_center_frequency,
            _sample_reader=read_samples,
            metadata={
                "data_format": self.format_name,
                "x_key": selected_x_key,
                "sample_rate_key": selected_sample_rate_key,
                "center_frequency_key": selected_center_key,
                "mat_keys": _available_keys(container),
            },
        )


__all__ = ["MatReader"]

