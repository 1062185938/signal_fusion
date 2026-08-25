"""Canonical multi-format IQ loaders.

This module consolidates the two loader copies that originally lived inside the
feature-extraction and modulation-recognition skill directories. Algorithmic
feature extraction, training, inference, and signal slicing intentionally stay
outside this module.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from signal_fusion.contracts import PreparedDataset


SUPPORTED_FORMATS = {
    "auto",
    "pkl",
    "pickle",
    "mat",
    "npz",
    "npy",
    "dat",
    "bin",
    "sigmf",
}


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _resolve_format(path: str | Path, data_format: str) -> str:
    fmt = (data_format or "auto").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported data_format={data_format!r}. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )
    if fmt != "auto":
        return "pkl" if fmt == "pickle" else fmt

    suffix = Path(path).suffix.lower()
    suffix_formats = {
        ".pkl": "pkl",
        ".pickle": "pkl",
        ".mat": "mat",
        ".npz": "npz",
        ".npy": "npy",
        ".dat": "dat",
        ".bin": "bin",
        ".sigmf": "sigmf",
        ".sigmf-data": "sigmf",
    }
    try:
        return suffix_formats[suffix]
    except KeyError as exc:
        raise ValueError(
            f"Cannot infer data format from suffix {suffix!r}: {path}"
        ) from exc


def _validate_seq_len(actual: int, requested: int | None, shape: tuple[int, ...]) -> int:
    actual = int(actual)
    if requested is not None and int(requested) != actual:
        raise ValueError(
            f"Requested seq_len={requested}, but loaded data has seq_len={actual}, "
            f"shape={shape}"
        )
    return actual


def _complex_iq_to_x(
    iq: np.ndarray, seq_len: int | None
) -> tuple[np.ndarray, int, int]:
    if seq_len is None:
        raise ValueError("seq_len is required when loading 1-D complex IQ data")
    if int(seq_len) <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    iq = np.asarray(iq, dtype=np.complex64).reshape(-1)
    total_points = int(iq.size)
    num_samples = total_points // int(seq_len)
    if num_samples == 0:
        raise ValueError(
            f"Not enough IQ points ({total_points}) for one seq_len={seq_len} sample"
        )

    used_points = num_samples * int(seq_len)
    records = iq[:used_points].reshape(num_samples, int(seq_len))
    x = np.empty((num_samples, 2, int(seq_len)), dtype=np.float32)
    x[:, 0, :] = records.real
    x[:, 1, :] = records.imag
    return x, used_points, total_points - used_points


def _standardize_x(x: np.ndarray, seq_len: int | None = None) -> tuple[np.ndarray, int]:
    array = np.asarray(x)

    if np.iscomplexobj(array) and array.ndim == 1:
        standardized, _, _ = _complex_iq_to_x(array, seq_len)
        return standardized, int(standardized.shape[2])

    if array.ndim == 3:
        if array.shape[1] == 2:
            resolved = _validate_seq_len(array.shape[2], seq_len, array.shape)
            return array.astype(np.float32, copy=False), resolved
        if array.shape[2] == 2:
            resolved = _validate_seq_len(array.shape[1], seq_len, array.shape)
            return np.transpose(array, (0, 2, 1)).astype(
                np.float32, copy=False
            ), resolved

    if array.ndim == 2:
        if np.iscomplexobj(array) and 1 in array.shape:
            standardized, _, _ = _complex_iq_to_x(array.reshape(-1), seq_len)
            return standardized, int(standardized.shape[2])
        if array.shape[0] == 2:
            resolved = _validate_seq_len(array.shape[1], seq_len, array.shape)
            return array[np.newaxis, :, :].astype(np.float32, copy=False), resolved
        if array.shape[1] == 2:
            resolved = _validate_seq_len(array.shape[0], seq_len, array.shape)
            return array.T[np.newaxis, :, :].astype(np.float32, copy=False), resolved

    raise ValueError(
        "Unsupported X shape. Expected [N, 2, seq_len], [N, seq_len, 2], "
        "[2, seq_len], [seq_len, 2], 1-D complex IQ data, or a complex "
        f"row/column vector; got shape={array.shape}, dtype={array.dtype}"
    )


def _standardize_y(y: np.ndarray | None) -> np.ndarray | None:
    if y is None:
        return None
    return np.asarray(y).reshape(-1).astype(np.int64, copy=False)


def _container_keys(container: Any) -> list[str]:
    if hasattr(container, "files"):
        return list(container.files)
    return [key for key in container.keys() if not str(key).startswith("__")]


def _pick_key(
    container: Any,
    explicit_key: str | None,
    candidates: tuple[str, ...],
    value_name: str,
    required: bool,
) -> str | None:
    keys = _container_keys(container)
    if explicit_key:
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
            f"Cannot find {value_name}. Tried keys {candidates}. Available keys: {keys}"
        )
    return None


def _load_optional_label(label_path: str | None) -> np.ndarray | None:
    if label_path is None:
        return None
    return np.load(label_path, allow_pickle=False)


def _load_pkl(
    path: str, seq_len: int | None
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    with open(path, "rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    if not isinstance(raw, dict):
        raise ValueError("PKL loader currently expects a RadioML-style dict")

    items: list[tuple[str, int, np.ndarray]] = []
    for key, value in raw.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Unexpected RadioML key: {key!r}")
        modulation, snr = key
        items.append((_to_text(modulation), int(snr), value))

    modulation_names = sorted({modulation for modulation, _, _ in items})
    label_to_index = {
        modulation: index for index, modulation in enumerate(modulation_names)
    }
    snr_values = sorted({snr for _, snr, _ in items})
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    resolved_seq_len = seq_len
    samples_per_key: dict[str, int] = {}

    for modulation, snr, value in sorted(
        items, key=lambda item: (label_to_index[item[0]], item[1])
    ):
        x_chunk, resolved_seq_len = _standardize_x(value, resolved_seq_len)
        x_chunks.append(x_chunk)
        y_chunks.append(
            np.full(x_chunk.shape[0], label_to_index[modulation], dtype=np.int64)
        )
        samples_per_key[f"{modulation}@{snr}"] = int(x_chunk.shape[0])

    if not x_chunks or resolved_seq_len is None:
        raise ValueError("RadioML PKL dataset is empty")
    x = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_chunks, axis=0).astype(np.int64, copy=False)
    return x, y, int(resolved_seq_len), {
        "modulation_names": modulation_names,
        "snr_values": snr_values,
        "samples_per_key": samples_per_key,
    }


def _load_mat(
    path: str,
    seq_len: int | None,
    label_path: str | None,
    x_key: str | None,
    y_key: str | None,
) -> tuple[np.ndarray, np.ndarray | None, int, dict[str, Any]]:
    mat = sio.loadmat(path)
    selected_x_key = _pick_key(mat, x_key, ("X", "data"), "X", required=True)
    selected_y_key = _pick_key(
        mat, y_key, ("y", "label"), "y", required=False
    )
    x, resolved_seq_len = _standardize_x(mat[selected_x_key], seq_len)
    y = (
        mat[selected_y_key]
        if selected_y_key is not None
        else _load_optional_label(label_path)
    )
    return x, _standardize_y(y), int(resolved_seq_len), {
        "x_key": selected_x_key,
        "y_key": selected_y_key,
    }


def _load_npz(
    path: str,
    seq_len: int | None,
    label_path: str | None,
    x_key: str | None,
    y_key: str | None,
) -> tuple[np.ndarray, np.ndarray | None, int, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        selected_x_key = _pick_key(data, x_key, ("X", "data"), "X", required=True)
        selected_y_key = _pick_key(
            data, y_key, ("y", "label"), "y", required=False
        )
        x, resolved_seq_len = _standardize_x(data[selected_x_key], seq_len)
        y = (
            data[selected_y_key]
            if selected_y_key is not None
            else _load_optional_label(label_path)
        )
        meta: dict[str, Any] = {
            "x_key": selected_x_key,
            "y_key": selected_y_key,
        }
        excluded_fields = {selected_x_key, selected_y_key, "source_id"}
        for field_name in data.files:
            if field_name not in excluded_fields:
                meta[field_name] = data[field_name]
        if "source_id" in data:
            meta["embedded_source_id"] = _to_text(data["source_id"].item())
    return x, _standardize_y(y), int(resolved_seq_len), meta


def _load_npy(
    path: str, seq_len: int | None, label_path: str | None
) -> tuple[np.ndarray, np.ndarray | None, int, dict[str, Any]]:
    x, resolved_seq_len = _standardize_x(
        np.load(path, allow_pickle=False), seq_len
    )
    return x, _standardize_y(_load_optional_label(label_path)), int(
        resolved_seq_len
    ), {}


def _load_dat_or_bin(
    path: str,
    seq_len: int | None,
    sample_mode: str | None,
    iq_format: str,
) -> tuple[np.ndarray, None, int, dict[str, Any]]:
    effective_sample_mode = "record_aligned" if sample_mode is None else sample_mode
    if effective_sample_mode != "record_aligned":
        raise ValueError(
            f"Unsupported sample_mode={sample_mode!r}. "
            "Only 'record_aligned' is supported."
        )
    if iq_format == "interleaved_float32":
        raise NotImplementedError(
            "interleaved_float32 IQ loading is reserved for future implementation."
        )
    if iq_format != "complex64":
        raise ValueError("Only iq_format='complex64' is supported at this stage")

    iq = np.fromfile(path, dtype=np.complex64)
    x, used_points, dropped_points = _complex_iq_to_x(iq, seq_len)
    return x, None, int(x.shape[2]), {
        "sample_mode": effective_sample_mode,
        "iq_format": iq_format,
        "used_iq_points": used_points,
        "dropped_iq_points": dropped_points,
    }


def load_prepared_dataset(
    path: str | Path,
    data_format: str = "auto",
    seq_len: int | None = None,
    label_path: str | None = None,
    x_key: str | None = None,
    y_key: str | None = None,
    sample_mode: str | None = None,
    iq_format: str = "complex64",
    source_id: str | None = None,
) -> PreparedDataset:
    """Load IQ data into the canonical :class:`PreparedDataset` contract."""

    path_text = str(path)
    fmt = _resolve_format(path_text, data_format)
    if fmt == "sigmf":
        raise NotImplementedError(
            "Direct SigMF loading is reserved for the future data-preparation API."
        )
    if fmt == "pkl":
        x, y, resolved_seq_len, extra_meta = _load_pkl(path_text, seq_len)
    elif fmt == "mat":
        x, y, resolved_seq_len, extra_meta = _load_mat(
            path_text, seq_len, label_path, x_key, y_key
        )
    elif fmt == "npz":
        x, y, resolved_seq_len, extra_meta = _load_npz(
            path_text, seq_len, label_path, x_key, y_key
        )
    elif fmt == "npy":
        x, y, resolved_seq_len, extra_meta = _load_npy(
            path_text, seq_len, label_path
        )
    elif fmt in {"dat", "bin"}:
        x, y, resolved_seq_len, extra_meta = _load_dat_or_bin(
            path_text, seq_len, sample_mode, iq_format
        )
    else:
        raise ValueError(f"Unsupported resolved data format: {fmt}")

    x = x.astype(np.float32, copy=False)
    y = _standardize_y(y)
    meta: dict[str, Any] = {
        "data_format": fmt,
        "path": path_text,
        "num_samples": int(x.shape[0]),
        "seq_len": int(resolved_seq_len),
        "x_shape": tuple(int(dimension) for dimension in x.shape),
        "y_shape": None if y is None else tuple(int(dimension) for dimension in y.shape),
    }
    meta.update(extra_meta)
    embedded_source_id = meta.pop("embedded_source_id", None)
    return PreparedDataset(
        X=x,
        y=y,
        meta=meta,
        source_id=source_id if source_id is not None else embedded_source_id,
    )


def load_signal_dataset(
    path: str,
    data_format: str = "auto",
    seq_len: int | None = None,
    label_path: str | None = None,
    x_key: str | None = None,
    y_key: str | None = None,
    sample_mode: str | None = None,
    iq_format: str = "complex64",
) -> dict[str, Any]:
    """Compatibility API returning the dictionary consumed by legacy CLIs."""

    return load_prepared_dataset(
        path=path,
        data_format=data_format,
        seq_len=seq_len,
        label_path=label_path,
        x_key=x_key,
        y_key=y_key,
        sample_mode=sample_mode,
        iq_format=iq_format,
    ).to_legacy_dict()


def load_signal_for_inference(
    path: str,
    data_format: str = "auto",
    seq_len: int = 128,
    sample_mode: str | None = None,
    iq_format: str | None = None,
    max_samples: int | None = None,
    x_key: str | None = None,
) -> dict[str, Any]:
    """Compatibility inference loader returning ``X`` and loader metadata."""

    fmt = _resolve_format(path, data_format)
    if fmt in {"dat", "bin"}:
        effective_sample_mode = "record_aligned" if sample_mode is None else sample_mode
        effective_iq_format = "complex64" if iq_format is None else iq_format
    else:
        effective_sample_mode = None
        effective_iq_format = "complex64"

    dataset = load_prepared_dataset(
        path=path,
        data_format=fmt,
        seq_len=seq_len,
        label_path=None,
        x_key=x_key,
        y_key=None,
        sample_mode=effective_sample_mode,
        iq_format=effective_iq_format,
    )
    x = dataset.X
    if max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        x = x[: int(max_samples)]

    meta = dict(dataset.meta)
    meta["inference_loader"] = "load_signal_for_inference"
    meta["max_samples"] = max_samples
    meta["returned_samples"] = int(x.shape[0])
    return {"X": x, "meta": meta}


__all__ = [
    "SUPPORTED_FORMATS",
    "load_prepared_dataset",
    "load_signal_dataset",
    "load_signal_for_inference",
]
