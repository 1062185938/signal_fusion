"""Rebuild complete regions from unnormalized prepared-source windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.evaluation.snr import standardize_iq_windows
from signal_fusion.io import (
    load_prepared_dataset,
    write_dataset_summary,
    write_prepared_dataset,
)


_GROUP_AUDIT_FIELDS = (
    "sample_source_path",
    "sample_location",
    "sample_filename_location",
    "sample_filename_location_mismatch",
    "sample_evaluation_group",
    "sample_class_name",
    "source_sample_rate",
    "center_frequency",
    "gain_db",
    "run",
)


def _sample_vector(dataset: PreparedDataset, name: str) -> np.ndarray:
    if name not in dataset.meta:
        raise ValueError(f"dataset is missing {name} metadata")
    values = np.asarray(dataset.meta[name])
    if values.shape != (dataset.num_samples,):
        raise ValueError(
            f"{name} metadata must contain one value per sample, got {values.shape}"
        )
    return values


def _single_group_value(
    dataset: PreparedDataset,
    name: str,
    indices: np.ndarray,
) -> Any:
    values = np.unique(_sample_vector(dataset, name)[indices])
    if values.size != 1:
        raise ValueError(f"one region must contain exactly one {name} value")
    return values[0]


def _prepared_source_index(
    prepared_root: str | Path,
    source_ids: set[str],
) -> dict[str, Path]:
    root = Path(prepared_root)
    if not root.is_dir():
        raise FileNotFoundError(f"prepared source directory does not exist: {root}")

    paths: dict[str, Path] = {}
    for path in sorted(root.rglob("*.npz")):
        if path.stem not in source_ids:
            continue
        if path.stem in paths:
            raise ValueError(
                f"multiple prepared datasets match source_id={path.stem!r}"
            )
        paths[path.stem] = path
    missing = sorted(source_ids - set(paths))
    if missing:
        raise FileNotFoundError(
            "prepared datasets are missing for source ids: " + ", ".join(missing)
        )
    return paths


def _require_unnormalized_source(dataset: PreparedDataset, path: Path) -> None:
    normalization = str(np.asarray(dataset.meta.get("normalization", "")).item())
    remove_dc = bool(np.asarray(dataset.meta.get("remove_dc", True)).item())
    if normalization != "none" or remove_dc:
        raise ValueError(
            "continuous regions must be rebuilt from unnormalized prepared "
            f"sources (normalization='none', remove_dc=false): {path}"
        )
    schema = str(np.asarray(dataset.meta.get("coordinate_schema", "")).item())
    if schema != "dual_rate_v1":
        raise ValueError(f"prepared source must use dual_rate_v1 coordinates: {path}")


def _rebuild_one_region(
    assembled: PreparedDataset,
    assembled_indices: np.ndarray,
    source_dataset: PreparedDataset,
    source_region_id: int,
) -> np.ndarray:
    source_region_ids = _sample_vector(source_dataset, "region_id").astype(
        np.int64, copy=False
    )
    source_indices = np.flatnonzero(source_region_ids == source_region_id)
    if source_indices.size == 0:
        raise ValueError(
            f"source region_id={source_region_id} does not exist in prepared source"
        )

    source_starts = _sample_vector(
        source_dataset, "target_window_start_sample"
    )[source_indices].astype(np.int64, copy=False)
    source_ends = _sample_vector(
        source_dataset, "target_window_end_sample"
    )[source_indices].astype(np.int64, copy=False)
    source_order = np.argsort(source_starts, kind="stable")
    source_indices = source_indices[source_order]
    source_starts = source_starts[source_order]
    source_ends = source_ends[source_order]

    assembled_starts = _sample_vector(
        assembled, "target_window_start_sample"
    )[assembled_indices].astype(np.int64, copy=False)
    assembled_ends = _sample_vector(
        assembled, "target_window_end_sample"
    )[assembled_indices].astype(np.int64, copy=False)
    assembled_order = np.argsort(assembled_starts, kind="stable")
    assembled_starts = assembled_starts[assembled_order]
    assembled_ends = assembled_ends[assembled_order]

    if not np.array_equal(source_starts, assembled_starts) or not np.array_equal(
        source_ends, assembled_ends
    ):
        raise ValueError(
            "prepared-source windows do not match assembled region coordinates"
        )
    if source_starts[0] >= source_ends[-1] or not np.array_equal(
        source_starts[1:], source_ends[:-1]
    ):
        raise ValueError("prepared-source windows are not contiguous")

    region_start = int(
        _single_group_value(assembled, "target_region_start_sample", assembled_indices)
    )
    region_end = int(
        _single_group_value(assembled, "target_region_end_sample", assembled_indices)
    )
    if source_starts[0] != region_start or source_ends[-1] != region_end:
        raise ValueError("prepared-source windows do not cover the complete region")

    windows = source_dataset.X[source_indices]
    samples = windows[:, 0, :] + 1j * windows[:, 1, :]
    return samples.reshape(-1).astype(np.complex64, copy=False)


def rebuild_continuous_region_dataset(
    dataset_path: str | Path,
    prepared_root: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one globally standardized row for every complete assembled region."""

    output = Path(output_path)
    summary_path = output.with_name(f"{output.stem}_dataset_summary.json")
    existing = [path for path in (output, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "continuous-region output already exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    assembled = load_prepared_dataset(dataset_path)
    group_ids = _sample_vector(assembled, "group_id").astype(np.int64, copy=False)
    source_id_values = _sample_vector(assembled, "sample_source_id").astype(str)
    unique_group_ids = np.unique(group_ids)
    source_paths = _prepared_source_index(
        prepared_root, set(source_id_values.tolist())
    )

    source_cache: dict[str, PreparedDataset] = {}
    raw_regions: list[np.ndarray] = []
    labels: list[int] = []
    metadata: dict[str, list[Any]] = {
        "group_id": [],
        "sample_source_id": [],
        "source_region_id": [],
        "target_region_start_sample": [],
        "target_region_end_sample": [],
        **{name: [] for name in _GROUP_AUDIT_FIELDS if name in assembled.meta},
    }

    for group_id in unique_group_ids:
        indices = np.flatnonzero(group_ids == group_id)
        source_id = str(
            _single_group_value(assembled, "sample_source_id", indices)
        )
        source_region_id = int(
            _single_group_value(assembled, "source_region_id", indices)
        )
        if source_id not in source_cache:
            source_dataset = load_prepared_dataset(source_paths[source_id])
            _require_unnormalized_source(source_dataset, source_paths[source_id])
            if source_dataset.source_id != source_id:
                raise ValueError(
                    "prepared dataset source_id does not match its assembled source: "
                    f"{source_dataset.source_id!r} != {source_id!r}"
                )
            source_cache[source_id] = source_dataset

        raw_regions.append(
            _rebuild_one_region(
                assembled,
                indices,
                source_cache[source_id],
                source_region_id,
            )
        )
        metadata["group_id"].append(int(group_id))
        metadata["sample_source_id"].append(source_id)
        metadata["source_region_id"].append(source_region_id)
        metadata["target_region_start_sample"].append(
            int(
                _single_group_value(
                    assembled, "target_region_start_sample", indices
                )
            )
        )
        metadata["target_region_end_sample"].append(
            int(
                _single_group_value(assembled, "target_region_end_sample", indices)
            )
        )
        for name in _GROUP_AUDIT_FIELDS:
            if name in metadata:
                metadata[name].append(_single_group_value(assembled, name, indices))

        if assembled.y is not None:
            group_labels = np.unique(assembled.y[indices])
            if group_labels.size != 1:
                raise ValueError(f"group_id={group_id} contains multiple labels")
            labels.append(int(group_labels[0]))

    lengths = {int(region.size) for region in raw_regions}
    if len(lengths) != 1:
        raise ValueError(f"all continuous regions must have one length, got {lengths}")
    region_length = lengths.pop()
    raw_x = np.empty((len(raw_regions), 2, region_length), dtype=np.float32)
    for index, region in enumerate(raw_regions):
        raw_x[index, 0, :] = region.real
        raw_x[index, 1, :] = region.imag
    x = standardize_iq_windows(raw_x)

    sample_rates = np.asarray(metadata.pop("source_sample_rate", []))
    target_sample_rates = _sample_vector(assembled, "target_sample_rate")
    unique_target_rates = np.unique(target_sample_rates.astype(np.float64))
    if unique_target_rates.size != 1:
        raise ValueError("all rebuilt regions must share one target sample rate")
    region_meta = {
        name: np.asarray(values) for name, values in metadata.items()
    }
    region_meta.update(
        {
            "sample_rate": float(unique_target_rates[0]),
            "seq_len": region_length,
            "reconstruction": "concatenate_unnormalized_prepared_source_windows",
            "normalization_scope": "complete_region",
            "remove_dc": True,
            "rms_normalize": True,
        }
    )
    if sample_rates.size:
        region_meta["source_sample_rate"] = sample_rates
    region_dataset = PreparedDataset(
        X=x,
        y=(np.asarray(labels, dtype=np.int64) if assembled.y is not None else None),
        meta=region_meta,
        source_id=f"{assembled.source_id or Path(dataset_path).stem}:continuous_regions",
    )
    write_prepared_dataset(region_dataset, output)

    complex_mean = x[:, 0, :].mean(axis=1) + 1j * x[:, 1, :].mean(axis=1)
    rms = np.sqrt(np.mean(np.square(x).sum(axis=1), axis=1))
    report = {
        "schema_version": 1,
        "output_path": str(output),
        "source_dataset_path": str(Path(dataset_path)),
        "prepared_root": str(Path(prepared_root)),
        "shape": list(x.shape),
        "region_count": region_dataset.num_samples,
        "region_sample_count": region_dataset.seq_len,
        "target_sample_rate": float(unique_target_rates[0]),
        "reconstruction": "concatenate_unnormalized_prepared_source_windows",
        "standardization": {
            "scope": "complete_region",
            "remove_dc": True,
            "rms_normalize": True,
            "max_abs_complex_mean": float(np.max(np.abs(complex_mean))),
            "output_rms_min": float(np.min(rms)),
            "output_rms_max": float(np.max(rms)),
        },
    }
    write_dataset_summary(report, output)
    return report


__all__ = ["rebuild_continuous_region_dataset"]
