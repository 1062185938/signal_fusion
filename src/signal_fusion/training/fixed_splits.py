"""Loading and validation for preassembled training-data splits."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from signal_fusion.contracts import PreparedDataset
from signal_fusion.io import load_prepared_dataset


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class FixedSplitBundle:
    """Validated fixed splits and their ready-to-use PyTorch DataLoaders."""

    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    seq_len: int
    input_channels: int
    class_num: int
    dataset_id: str
    label_map: dict[str, str]
    split_paths: dict[str, Path]
    split_sizes: dict[str, int]

    def loader_tuple(self):
        return (
            self.train_loader,
            self.validation_loader,
            self.test_loader,
            self.seq_len,
            self.input_channels,
        )


def _metadata_scalar(dataset: PreparedDataset, field_name: str) -> Any:
    if field_name not in dataset.meta:
        raise ValueError(
            f"fixed split {dataset.source_id!r} is missing metadata {field_name!r}"
        )
    value = np.asarray(dataset.meta[field_name])
    if value.size != 1:
        raise ValueError(
            f"fixed split metadata {field_name!r} must be scalar, got {value.shape}"
        )
    return value.reshape(()).item()


def _sample_metadata(
    dataset: PreparedDataset, field_name: str, *, integer: bool = False
) -> np.ndarray:
    if field_name not in dataset.meta:
        raise ValueError(
            f"fixed split {dataset.source_id!r} is missing metadata {field_name!r}"
        )
    value = np.asarray(dataset.meta[field_name])
    if value.ndim != 1 or value.shape[0] != dataset.num_samples:
        raise ValueError(
            f"fixed split metadata {field_name!r} must have shape "
            f"[{dataset.num_samples}], got {value.shape}"
        )
    if integer and not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"fixed split metadata {field_name!r} must be integer")
    return value


def _parse_label_map(dataset: PreparedDataset) -> dict[str, str]:
    raw_json = _metadata_scalar(dataset, "label_map_json")
    try:
        raw_map = json.loads(str(raw_json))
    except json.JSONDecodeError as exc:
        raise ValueError("fixed split label_map_json is invalid JSON") from exc
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("fixed split label_map_json must contain a non-empty object")

    try:
        sorted_items = sorted(raw_map.items(), key=lambda item: int(item[0]))
    except (TypeError, ValueError) as exc:
        raise ValueError("label_map_json keys must be integer indices") from exc

    normalized: dict[str, str] = {}
    for expected_index, (raw_index, raw_name) in enumerate(sorted_items):
        try:
            actual_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("label_map_json keys must be integer indices") from exc
        if actual_index != expected_index:
            raise ValueError(
                "label_map_json indices must be continuous from zero, "
                f"got index={actual_index} at position={expected_index}"
            )
        name = str(raw_name)
        if not name.strip():
            raise ValueError("label_map_json class names must be non-empty")
        normalized[str(actual_index)] = name
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("label_map_json class names must be unique")
    return normalized


def _load_split_directory(
    dataset_dir: str | Path,
) -> tuple[
    dict[str, PreparedDataset],
    dict[str, Path],
    str,
    dict[str, str],
]:
    directory = Path(dataset_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"fixed split directory does not exist: {directory}")

    split_paths = {
        split_name: directory / f"{split_name}.npz"
        for split_name in SPLIT_NAMES
    }
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "fixed split directory is missing required files: " + ", ".join(missing)
        )
    datasets = {
        split_name: load_prepared_dataset(path)
        for split_name, path in split_paths.items()
    }

    reference_dataset_id: str | None = None
    reference_label_map: dict[str, str] | None = None
    reference_seq_len: int | None = None
    for split_name in SPLIT_NAMES:
        dataset = datasets[split_name]
        if dataset.y is None:
            raise ValueError(f"fixed split {split_name!r} has no labels")
        if dataset.num_samples == 0:
            raise ValueError(f"fixed split {split_name!r} is empty")
        if not np.all(np.isfinite(dataset.X)):
            raise ValueError(f"fixed split {split_name!r} contains non-finite IQ")

        embedded_split = str(_metadata_scalar(dataset, "split"))
        if embedded_split != split_name:
            raise ValueError(
                f"file {split_paths[split_name]} declares split={embedded_split!r}, "
                f"expected {split_name!r}"
            )
        dataset_id = str(_metadata_scalar(dataset, "dataset_id"))
        label_map = _parse_label_map(dataset)
        if reference_dataset_id is None:
            reference_dataset_id = dataset_id
            reference_label_map = label_map
            reference_seq_len = dataset.seq_len
        else:
            if dataset_id != reference_dataset_id:
                raise ValueError(
                    "fixed split dataset_id values differ: "
                    f"{reference_dataset_id!r} != {dataset_id!r}"
                )
            if label_map != reference_label_map:
                raise ValueError("fixed split label_map_json values differ")
            if dataset.seq_len != reference_seq_len:
                raise ValueError(
                    "fixed split sequence lengths differ: "
                    f"{reference_seq_len} != {dataset.seq_len}"
                )

    if reference_dataset_id is None or reference_label_map is None:
        raise RuntimeError("fixed split validation did not resolve dataset metadata")
    return (
        datasets,
        split_paths,
        reference_dataset_id,
        reference_label_map,
    )


def _validate_labels(
    datasets: dict[str, PreparedDataset],
    label_map: dict[str, str],
    class_num: int,
) -> None:
    if class_num != len(label_map):
        raise ValueError(
            f"class_num={class_num} does not match label_map size={len(label_map)}"
        )
    expected_labels = list(range(class_num))
    for split_name, dataset in datasets.items():
        actual_labels = np.unique(dataset.y).tolist()
        if actual_labels != expected_labels:
            raise ValueError(
                f"fixed split {split_name!r} must contain labels "
                f"{expected_labels}, got {actual_labels}"
            )


def _validate_group_integrity(datasets: dict[str, PreparedDataset]) -> None:
    source_sets: dict[str, set[str]] = {}
    group_sets: dict[str, set[int]] = {}
    source_region_sets: dict[str, set[tuple[str, int]]] = {}

    for split_name, dataset in datasets.items():
        source_ids = _sample_metadata(dataset, "sample_source_id").astype(str)
        group_ids = _sample_metadata(dataset, "group_id", integer=True).astype(
            np.int64, copy=False
        )
        source_region_ids = _sample_metadata(
            dataset, "source_region_id", integer=True
        ).astype(np.int64, copy=False)
        source_sets[split_name] = set(source_ids.tolist())
        group_sets[split_name] = set(group_ids.tolist())
        source_region_sets[split_name] = set(
            zip(source_ids.tolist(), source_region_ids.tolist())
        )

        for group_id in np.unique(group_ids):
            mask = group_ids == group_id
            if np.unique(dataset.y[mask]).size != 1:
                raise ValueError(
                    f"fixed split {split_name!r} group_id={int(group_id)} "
                    "contains multiple labels"
                )
            if np.unique(source_ids[mask]).size != 1:
                raise ValueError(
                    f"fixed split {split_name!r} group_id={int(group_id)} "
                    "contains multiple source_id values"
                )
            if np.unique(source_region_ids[mask]).size != 1:
                raise ValueError(
                    f"fixed split {split_name!r} group_id={int(group_id)} "
                    "contains multiple source_region_id values"
                )

    for first_index, first_split in enumerate(SPLIT_NAMES):
        for second_split in SPLIT_NAMES[first_index + 1 :]:
            if source_sets[first_split].intersection(source_sets[second_split]):
                raise ValueError(
                    f"source_id leakage between {first_split} and {second_split}"
                )
            if group_sets[first_split].intersection(group_sets[second_split]):
                raise ValueError(
                    f"group_id leakage between {first_split} and {second_split}"
                )
            if source_region_sets[first_split].intersection(
                source_region_sets[second_split]
            ):
                raise ValueError(
                    f"source region leakage between {first_split} and {second_split}"
                )


def load_fixed_split_bundle(
    dataset_dir: str | Path,
    *,
    class_num: int,
    batch_size: int,
    num_workers: int,
) -> FixedSplitBundle:
    """Load an assembled directory without performing another data split."""

    if class_num <= 0:
        raise ValueError("class_num must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")

    datasets, split_paths, dataset_id, label_map = _load_split_directory(
        dataset_dir
    )
    _validate_labels(datasets, label_map, class_num)
    _validate_group_integrity(datasets)

    loader_options: dict[str, Any] = {"num_workers": num_workers}
    if torch.cuda.is_available():
        loader_options["pin_memory"] = True
    tensor_datasets = {
        split_name: TensorDataset(
            torch.from_numpy(dataset.X), torch.from_numpy(dataset.y)
        )
        for split_name, dataset in datasets.items()
    }
    return FixedSplitBundle(
        train_loader=DataLoader(
            tensor_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            **loader_options,
        ),
        validation_loader=DataLoader(
            tensor_datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            **loader_options,
        ),
        test_loader=DataLoader(
            tensor_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            **loader_options,
        ),
        seq_len=datasets["train"].seq_len,
        input_channels=int(datasets["train"].X.shape[1]),
        class_num=class_num,
        dataset_id=dataset_id,
        label_map=label_map,
        split_paths=split_paths,
        split_sizes={
            split_name: dataset.num_samples
            for split_name, dataset in datasets.items()
        },
    )


__all__ = ["FixedSplitBundle", "SPLIT_NAMES", "load_fixed_split_bundle"]
