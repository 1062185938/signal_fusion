"""Deterministic assembly of prepared IQ slices into fixed training splits."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.io import load_prepared_dataset, write_prepared_dataset
from signal_fusion.io.writers import json_safe

from .contracts import (
    AssemblySource,
    SPLIT_NAMES,
    TrainingAssemblyManifest,
    TrainingAssemblyResult,
)


ASSEMBLY_VERSION = "training_assembly_v2"


@dataclass(slots=True)
class _LoadedSource:
    spec: AssemblySource
    dataset: PreparedDataset
    source_id: str
    class_name: str
    region_field: str
    region_ids: np.ndarray
    window_ids: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    region_starts: np.ndarray
    region_ends: np.ndarray
    sample_rates: np.ndarray
    center_frequencies: np.ndarray
    source_window_rms: np.ndarray | None

    @property
    def num_samples(self) -> int:
        return self.dataset.num_samples

    @property
    def seq_len(self) -> int:
        return self.dataset.seq_len

    def grouped_indices(self) -> list[tuple[int, np.ndarray]]:
        groups: list[tuple[int, np.ndarray]] = []
        for region_id in np.unique(self.region_ids):
            indices = np.flatnonzero(self.region_ids == region_id)
            order = np.lexsort(
                (indices, self.window_starts[indices], self.window_ids[indices])
            )
            groups.append((int(region_id), indices[order]))
        return groups


@dataclass(frozen=True, slots=True)
class _RegionGroup:
    source: _LoadedSource
    source_region_id: int
    sample_indices: np.ndarray


def load_assembly_manifest(path: str | Path) -> TrainingAssemblyManifest:
    """Load and strictly validate a JSON assembly manifest."""

    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON assembly manifest: {manifest_path}") from exc
    return TrainingAssemblyManifest.from_dict(raw, manifest_path=manifest_path)


def _scalar(meta: dict[str, Any], field_name: str, *, required: bool) -> Any:
    if field_name not in meta:
        if required:
            raise ValueError(f"source dataset is missing scalar metadata {field_name!r}")
        return None
    value = np.asarray(meta[field_name])
    if value.size != 1:
        raise ValueError(
            f"source metadata {field_name!r} must be scalar, got shape={value.shape}"
        )
    return value.reshape(()).item()


def _integer_sample_field(
    meta: dict[str, Any], field_name: str, num_samples: int
) -> np.ndarray:
    if field_name not in meta:
        raise ValueError(f"source dataset is missing per-sample metadata {field_name!r}")
    value = np.asarray(meta[field_name])
    if value.ndim != 1 or value.shape[0] != num_samples:
        raise ValueError(
            f"source metadata {field_name!r} must have shape [{num_samples}], "
            f"got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"source metadata {field_name!r} must use an integer dtype")
    return value.astype(np.int64, copy=False)


def _numeric_sample_field(
    meta: dict[str, Any],
    field_name: str,
    num_samples: int,
    *,
    required: bool,
    missing_value: float = np.nan,
) -> np.ndarray:
    if field_name not in meta:
        if required:
            raise ValueError(f"source dataset is missing metadata {field_name!r}")
        return np.full(num_samples, missing_value, dtype=np.float64)
    value = np.asarray(meta[field_name])
    if value.size == 1:
        result = np.full(num_samples, value.reshape(()).item(), dtype=np.float64)
    elif value.ndim == 1 and value.shape[0] == num_samples:
        result = value.astype(np.float64, copy=False)
    else:
        raise ValueError(
            f"source metadata {field_name!r} must be scalar or have shape "
            f"[{num_samples}], got {value.shape}"
        )
    if required and not np.all(np.isfinite(result)):
        raise ValueError(f"source metadata {field_name!r} must be finite")
    return result


def _load_source(
    spec: AssemblySource,
    label_map: dict[int, str],
    *,
    require_window_rms: bool,
) -> _LoadedSource:
    if not spec.path.is_file():
        raise FileNotFoundError(f"source dataset does not exist: {spec.path}")
    dataset = load_prepared_dataset(spec.path)
    if dataset.y is None:
        raise ValueError(f"source dataset has no labels: {spec.path}")
    if dataset.num_samples == 0:
        raise ValueError(f"source dataset is empty: {spec.path}")
    if not np.all(np.isfinite(dataset.X)):
        raise ValueError(f"source dataset contains non-finite IQ values: {spec.path}")

    unique_labels = np.unique(dataset.y)
    if unique_labels.tolist() != [spec.label]:
        raise ValueError(
            f"source dataset labels {unique_labels.tolist()} do not match manifest "
            f"label={spec.label}: {spec.path}"
        )
    scalar_label = _scalar(dataset.meta, "label", required=False)
    if scalar_label is not None and int(scalar_label) != spec.label:
        raise ValueError(
            f"source scalar label={scalar_label!r} does not match manifest "
            f"label={spec.label}: {spec.path}"
        )

    if dataset.source_id is None:
        raise ValueError(f"source dataset has no scalar source_id: {spec.path}")
    source_id = dataset.source_id
    if spec.expected_source_id is not None and source_id != spec.expected_source_id:
        raise ValueError(
            f"source_id={source_id!r} does not match manifest source_id="
            f"{spec.expected_source_id!r}: {spec.path}"
        )

    expected_class_name = label_map[spec.label]
    source_class_name = _scalar(dataset.meta, "class_name", required=False)
    if source_class_name is not None and str(source_class_name) != expected_class_name:
        raise ValueError(
            f"class_name={source_class_name!r} does not match label_map name="
            f"{expected_class_name!r}: {spec.path}"
        )

    if "region_id" in dataset.meta:
        region_field = "region_id"
    elif "burst_id" in dataset.meta:
        region_field = "burst_id"
    else:
        raise ValueError(
            f"source dataset needs region_id or burst_id metadata: {spec.path}"
        )
    region_ids = _integer_sample_field(
        dataset.meta, region_field, dataset.num_samples
    )
    window_ids = _integer_sample_field(
        dataset.meta, "window_id", dataset.num_samples
    )
    window_starts = _integer_sample_field(
        dataset.meta, "window_start_sample", dataset.num_samples
    )
    window_ends = _integer_sample_field(
        dataset.meta, "window_end_sample", dataset.num_samples
    )
    if region_field == "region_id":
        region_start_field = "region_start_sample"
        region_end_field = "region_end_sample"
    else:
        region_start_field = "burst_start_sample"
        region_end_field = "burst_end_sample"
    region_starts = _integer_sample_field(
        dataset.meta, region_start_field, dataset.num_samples
    )
    region_ends = _integer_sample_field(
        dataset.meta, region_end_field, dataset.num_samples
    )
    if np.any(window_ends <= window_starts):
        raise ValueError(f"source contains invalid window sample bounds: {spec.path}")
    if np.any(region_ends <= region_starts):
        raise ValueError(f"source contains invalid region sample bounds: {spec.path}")
    if np.any(window_starts < region_starts) or np.any(window_ends > region_ends):
        raise ValueError(f"source windows fall outside their regions: {spec.path}")

    source_window_rms = None
    if require_window_rms:
        source_window_rms = _numeric_sample_field(
            dataset.meta,
            "normalization_scale",
            dataset.num_samples,
            required=True,
        )
        if np.any(source_window_rms <= 0.0):
            raise ValueError(
                "source normalization_scale must contain positive window RMS "
                f"values for top_energy selection: {spec.path}"
            )

    return _LoadedSource(
        spec=spec,
        dataset=dataset,
        source_id=source_id,
        class_name=expected_class_name,
        region_field=region_field,
        region_ids=region_ids,
        window_ids=window_ids,
        window_starts=window_starts,
        window_ends=window_ends,
        region_starts=region_starts,
        region_ends=region_ends,
        sample_rates=_numeric_sample_field(
            dataset.meta, "sample_rate", dataset.num_samples, required=True
        ),
        center_frequencies=_numeric_sample_field(
            dataset.meta,
            "center_frequency",
            dataset.num_samples,
            required=False,
        ),
        source_window_rms=source_window_rms,
    )


def _uniform_positions(total: int, requested: int) -> np.ndarray:
    if requested > total:
        raise ValueError(f"cannot select {requested} items from only {total}")
    if requested == total:
        return np.arange(total, dtype=np.int64)
    positions = np.floor(
        (np.arange(requested, dtype=np.float64) + 0.5) * total / requested
    ).astype(np.int64)
    if np.unique(positions).size != requested:
        raise RuntimeError("uniform selection produced duplicate positions")
    return positions


def _select_window_indices(
    group: _RegionGroup,
    *,
    requested: int,
    strategy: str,
) -> np.ndarray:
    """Select region windows deterministically and return them in time order."""

    available = group.sample_indices
    if requested > available.size:
        raise ValueError(f"cannot select {requested} items from only {available.size}")
    if strategy == "uniform":
        return available[_uniform_positions(available.size, requested)]
    if strategy != "top_energy":
        raise RuntimeError(f"unsupported window selection strategy: {strategy!r}")

    source_window_rms = group.source.source_window_rms
    if source_window_rms is None:
        raise RuntimeError("top_energy selection requires source window RMS metadata")
    rms = source_window_rms[available]
    ranked_positions = np.lexsort(
        (
            group.source.window_ids[available],
            group.source.window_starts[available],
            -rms,
        )
    )
    selected = available[ranked_positions[:requested]]
    time_order = np.lexsort(
        (
            group.source.window_ids[selected],
            group.source.window_starts[selected],
        )
    )
    return selected[time_order]


def _source_report(source: _LoadedSource) -> dict[str, Any]:
    return {
        "path": source.spec.declared_path,
        "resolved_path": str(source.spec.path),
        "source_id": source.source_id,
        "label": source.spec.label,
        "class_name": source.class_name,
        "shape": list(source.dataset.X.shape),
        "region_field": source.region_field,
        "region_count": len(source.grouped_indices()),
        "sample_rate_values": np.unique(source.sample_rates).tolist(),
        "center_frequency_values": np.unique(source.center_frequencies).tolist(),
    }


def _validate_sources(
    manifest: TrainingAssemblyManifest,
    loaded_splits: dict[str, list[_LoadedSource]],
) -> None:
    seq_lengths = {
        source.seq_len
        for sources in loaded_splits.values()
        for source in sources
    }
    if len(seq_lengths) != 1:
        raise ValueError(f"all source datasets must share seq_len, got {seq_lengths}")

    source_owners: dict[str, tuple[str, Path]] = {}
    for split_name, sources in loaded_splits.items():
        for source in sources:
            previous = source_owners.get(source.source_id)
            if previous is not None:
                previous_split, previous_path = previous
                if previous_path != source.spec.path:
                    raise ValueError(
                        f"source_id={source.source_id!r} is used by multiple files: "
                        f"{previous_path} and {source.spec.path}"
                    )
                if (
                    manifest.require_disjoint_sources
                    and previous_split != split_name
                ):
                    raise ValueError(
                        f"source_id={source.source_id!r} appears in both "
                        f"{previous_split} and {split_name}"
                    )
            source_owners[source.source_id] = (split_name, source.spec.path)


def _groups_by_label(
    sources: Iterable[_LoadedSource], labels: Iterable[int]
) -> dict[int, list[_RegionGroup]]:
    grouped: dict[int, list[_RegionGroup]] = {label: [] for label in labels}
    for source in sources:
        for source_region_id, sample_indices in source.grouped_indices():
            grouped[source.spec.label].append(
                _RegionGroup(source, source_region_id, sample_indices)
            )
    for groups in grouped.values():
        groups.sort(key=lambda group: (group.source.source_id, group.source_region_id))
    return grouped


def _standardize_windows(
    x: np.ndarray,
    *,
    remove_dc: bool,
    rms_normalize: bool,
    rms_epsilon: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    working = x.astype(np.float64, copy=True)
    removed_dc_i = working[:, 0, :].mean(axis=1)
    removed_dc_q = working[:, 1, :].mean(axis=1)
    pre_standardization_rms = np.sqrt(np.mean(np.square(working).sum(axis=1), axis=1))

    if remove_dc:
        working[:, 0, :] -= removed_dc_i[:, np.newaxis]
        working[:, 1, :] -= removed_dc_q[:, np.newaxis]
    else:
        removed_dc_i = np.zeros(x.shape[0], dtype=np.float64)
        removed_dc_q = np.zeros(x.shape[0], dtype=np.float64)

    normalization_scale = np.sqrt(
        np.mean(np.square(working).sum(axis=1), axis=1)
    )
    if rms_normalize:
        invalid = np.flatnonzero(normalization_scale <= rms_epsilon)
        if invalid.size:
            raise ValueError(
                "selected windows contain zero or near-zero energy after DC removal; "
                f"first output indices={invalid[:10].tolist()}"
            )
        working /= normalization_scale[:, np.newaxis, np.newaxis]
    else:
        normalization_scale = np.ones(x.shape[0], dtype=np.float64)

    standardized = working.astype(np.float32)
    if not np.all(np.isfinite(standardized)):
        raise ValueError("standardization produced non-finite IQ values")
    return standardized, {
        "removed_dc_i": removed_dc_i,
        "removed_dc_q": removed_dc_q,
        "pre_standardization_rms": pre_standardization_rms,
        "normalization_scale": normalization_scale,
    }


def _assemble_split(
    manifest: TrainingAssemblyManifest,
    split_name: str,
    sources: list[_LoadedSource],
    source_indices: dict[str, int],
    next_group_id: int,
) -> tuple[PreparedDataset, dict[str, Any], int, set[tuple[str, int]]]:
    groups_by_label = _groups_by_label(sources, manifest.label_map)
    available_regions = {
        label: len(groups) for label, groups in groups_by_label.items()
    }
    if any(count == 0 for count in available_regions.values()):
        raise ValueError(
            f"split {split_name!r} has a class without any regions: "
            f"{available_regions}"
        )
    if manifest.regions_per_class is None:
        selected_region_count = min(available_regions.values())
    else:
        selected_region_count = manifest.regions_per_class
        too_small = {
            label: count
            for label, count in available_regions.items()
            if count < selected_region_count
        }
        if too_small:
            raise ValueError(
                f"split {split_name!r} cannot select "
                f"regions_per_class={selected_region_count}; available={too_small}"
            )

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    metadata: dict[str, list[np.ndarray]] = {
        "group_id": [],
        "burst_id": [],
        "source_region_id": [],
        "window_id": [],
        "window_start_sample": [],
        "window_end_sample": [],
        "region_start_sample": [],
        "region_end_sample": [],
        "source_index": [],
        "sample_source_id": [],
        "sample_source_path": [],
        "sample_class_name": [],
        "source_region_field": [],
        "sample_rate": [],
        "center_frequency": [],
    }
    if manifest.window_selection == "top_energy":
        metadata["source_window_rms"] = []
    selected_group_keys: set[tuple[str, int]] = set()
    selected_source_counts = {source.source_id: 0 for source in sources}

    for label in sorted(manifest.label_map):
        available_groups = groups_by_label[label]
        region_positions = _uniform_positions(
            len(available_groups), selected_region_count
        )
        for region_position in region_positions:
            group = available_groups[int(region_position)]
            if group.sample_indices.size < manifest.windows_per_region:
                raise ValueError(
                    f"source_id={group.source.source_id!r}, "
                    f"{group.source.region_field}={group.source_region_id} has only "
                    f"{group.sample_indices.size} windows; "
                    f"{manifest.windows_per_region} required"
                )
            indices = _select_window_indices(
                group,
                requested=manifest.windows_per_region,
                strategy=manifest.window_selection,
            )
            sample_count = int(indices.size)
            group_id = next_group_id
            next_group_id += 1
            group_key = (group.source.source_id, group.source_region_id)
            if group_key in selected_group_keys:
                raise RuntimeError(f"region selected twice: {group_key}")
            selected_group_keys.add(group_key)
            selected_source_counts[group.source.source_id] += sample_count

            x_chunks.append(group.source.dataset.X[indices])
            y_chunks.append(np.full(sample_count, label, dtype=np.int64))
            metadata["group_id"].append(
                np.full(sample_count, group_id, dtype=np.int64)
            )
            metadata["burst_id"].append(
                np.full(sample_count, group_id, dtype=np.int64)
            )
            metadata["source_region_id"].append(
                np.full(sample_count, group.source_region_id, dtype=np.int64)
            )
            metadata["window_id"].append(group.source.window_ids[indices])
            metadata["window_start_sample"].append(group.source.window_starts[indices])
            metadata["window_end_sample"].append(group.source.window_ends[indices])
            metadata["region_start_sample"].append(group.source.region_starts[indices])
            metadata["region_end_sample"].append(group.source.region_ends[indices])
            metadata["source_index"].append(
                np.full(
                    sample_count,
                    source_indices[group.source.source_id],
                    dtype=np.int64,
                )
            )
            metadata["sample_source_id"].append(
                np.full(sample_count, group.source.source_id)
            )
            metadata["sample_source_path"].append(
                np.full(sample_count, group.source.spec.declared_path)
            )
            metadata["sample_class_name"].append(
                np.full(sample_count, group.source.class_name)
            )
            metadata["source_region_field"].append(
                np.full(sample_count, group.source.region_field)
            )
            metadata["sample_rate"].append(group.source.sample_rates[indices])
            metadata["center_frequency"].append(
                group.source.center_frequencies[indices]
            )
            if group.source.source_window_rms is not None:
                metadata["source_window_rms"].append(
                    group.source.source_window_rms[indices]
                )

    selected_x = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
    selected_y = np.concatenate(y_chunks, axis=0)
    standardized_x, standardization_meta = _standardize_windows(
        selected_x,
        remove_dc=manifest.remove_dc,
        rms_normalize=manifest.rms_normalize,
        rms_epsilon=manifest.rms_epsilon,
    )
    flat_meta = {
        field_name: np.concatenate(chunks, axis=0)
        for field_name, chunks in metadata.items()
    }
    flat_meta.update(standardization_meta)
    flat_meta.update(
        {
            "dataset_id": manifest.dataset_id,
            "split": split_name,
            "assembly_version": ASSEMBLY_VERSION,
            "seq_len": standardized_x.shape[2],
            "windows_per_region": manifest.windows_per_region,
            "regions_per_class": selected_region_count,
            "region_selection": manifest.region_selection,
            "window_selection": manifest.window_selection,
            "remove_dc": manifest.remove_dc,
            "rms_normalize": manifest.rms_normalize,
            "rms_epsilon": manifest.rms_epsilon,
            "label_map_json": json.dumps(
                manifest.label_map, ensure_ascii=False, sort_keys=True
            ),
        }
    )
    dataset = PreparedDataset(
        X=standardized_x,
        y=selected_y,
        meta=flat_meta,
        source_id=f"{manifest.dataset_id}:{split_name}",
    )

    complex_mean = standardized_x[:, 0, :].mean(axis=1) + 1j * standardized_x[
        :, 1, :
    ].mean(axis=1)
    output_rms = np.sqrt(
        np.mean(np.square(standardized_x).sum(axis=1), axis=1)
    )
    split_report = {
        "shape": list(standardized_x.shape),
        "class_counts": {
            str(label): int(np.count_nonzero(selected_y == label))
            for label in sorted(manifest.label_map)
        },
        "region_counts": {
            str(label): int(selected_region_count)
            for label in sorted(manifest.label_map)
        },
        "available_region_counts": {
            str(label): count for label, count in available_regions.items()
        },
        "selected_source_sample_counts": selected_source_counts,
        "selected_group_count": len(selected_group_keys),
        "standardization": {
            "max_abs_complex_mean": float(np.max(np.abs(complex_mean))),
            "output_rms_min": float(np.min(output_rms)),
            "output_rms_max": float(np.max(output_rms)),
        },
        "sources": [_source_report(source) for source in sources],
    }
    if manifest.window_selection == "top_energy":
        selected_window_rms = flat_meta["source_window_rms"]
        split_report["selected_window_rms"] = {
            "min": float(np.min(selected_window_rms)),
            "mean": float(np.mean(selected_window_rms)),
            "max": float(np.max(selected_window_rms)),
        }
    return dataset, split_report, next_group_id, selected_group_keys


def assemble_training_dataset(
    manifest: str | Path | TrainingAssemblyManifest,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> TrainingAssemblyResult:
    """Build fixed class-balanced split files and an audit report.

    Source NPZ files are read-only. Split ownership comes exclusively from the
    manifest, so this function never performs a random sample-level split.
    """

    if not isinstance(manifest, TrainingAssemblyManifest):
        manifest = load_assembly_manifest(manifest)
    output_directory = Path(output_dir)
    split_paths = {
        split_name: output_directory / f"{split_name}.npz"
        for split_name in SPLIT_NAMES
    }
    report_path = output_directory / "assembly_report.json"
    planned_outputs = [*split_paths.values(), report_path]
    existing_outputs = [path for path in planned_outputs if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "assembly output already exists; pass overwrite=True to replace it: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    loaded_splits = {
        split_name: [
            _load_source(
                source,
                manifest.label_map,
                require_window_rms=manifest.window_selection == "top_energy",
            )
            for source in manifest.splits[split_name]
        ]
        for split_name in SPLIT_NAMES
    }
    _validate_sources(manifest, loaded_splits)
    all_sources = [
        source
        for split_name in SPLIT_NAMES
        for source in loaded_splits[split_name]
    ]
    source_indices = {
        source_id: index
        for index, source_id in enumerate(
            sorted({source.source_id for source in all_sources})
        )
    }

    assembled: dict[str, PreparedDataset] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    selected_groups: dict[str, set[tuple[str, int]]] = {}
    next_group_id = 0
    for split_name in SPLIT_NAMES:
        dataset, split_report, next_group_id, group_keys = _assemble_split(
            manifest,
            split_name,
            loaded_splits[split_name],
            source_indices,
            next_group_id,
        )
        assembled[split_name] = dataset
        split_reports[split_name] = split_report
        selected_groups[split_name] = group_keys

    overlaps: dict[str, list[list[Any]]] = {}
    for first_index, first_split in enumerate(SPLIT_NAMES):
        for second_split in SPLIT_NAMES[first_index + 1 :]:
            overlap = selected_groups[first_split].intersection(
                selected_groups[second_split]
            )
            key = f"{first_split}__{second_split}"
            overlaps[key] = [list(item) for item in sorted(overlap)]
            if overlap:
                raise RuntimeError(
                    f"region leakage detected between {first_split} and "
                    f"{second_split}: {sorted(overlap)[:10]}"
                )

    output_directory.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        write_prepared_dataset(assembled[split_name], split_paths[split_name])
        split_reports[split_name]["output_path"] = str(split_paths[split_name])

    report = {
        "schema_version": 1,
        "assembly_version": ASSEMBLY_VERSION,
        "dataset_id": manifest.dataset_id,
        "manifest_path": str(manifest.manifest_path),
        "label_map": {str(label): name for label, name in manifest.label_map.items()},
        "assembly": {
            "windows_per_region": manifest.windows_per_region,
            "regions_per_class": (
                "minimum"
                if manifest.regions_per_class is None
                else manifest.regions_per_class
            ),
            "region_selection": manifest.region_selection,
            "window_selection": manifest.window_selection,
            "remove_dc": manifest.remove_dc,
            "rms_normalize": manifest.rms_normalize,
            "rms_epsilon": manifest.rms_epsilon,
            "require_disjoint_sources": manifest.require_disjoint_sources,
        },
        "source_index": {
            str(index): source_id for source_id, index in source_indices.items()
        },
        "leakage_audit": {
            "source_ids_disjoint": manifest.require_disjoint_sources,
            "selected_region_overlaps": overlaps,
            "passed": True,
        },
        "splits": split_reports,
    }
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return TrainingAssemblyResult(
        output_dir=output_directory,
        split_paths=split_paths,
        report_path=report_path,
        report=report,
    )


__all__ = [
    "ASSEMBLY_VERSION",
    "assemble_training_dataset",
    "load_assembly_manifest",
]
