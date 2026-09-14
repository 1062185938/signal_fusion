"""Build one 62-dimensional feature row per complete continuous region."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.evaluation import add_complex_awgn, standardize_iq_windows
from signal_fusion.feature_extraction import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_ID,
    MAX_SIGNAL_LENGTH,
    FeatureBackend,
    FeatureExtractionService,
    IQFeatureCtypesBackend,
)
from signal_fusion.fusion import load_complete_region
from signal_fusion.io import load_prepared_dataset


REGION_FEATURE_DATASET_VERSION = "region_features_v1"
SPLIT_NAMES = ("train", "validation", "test")


def _scalar(meta: dict[str, Any], name: str) -> Any:
    if name not in meta:
        raise ValueError(f"assembled dataset is missing metadata {name!r}")
    value = np.asarray(meta[name])
    if value.size != 1:
        raise ValueError(f"assembled metadata {name!r} must be scalar")
    return value.reshape(()).item()


def _sample_field(dataset: PreparedDataset, name: str) -> np.ndarray:
    if name not in dataset.meta:
        raise ValueError(f"assembled dataset is missing metadata {name!r}")
    value = np.asarray(dataset.meta[name])
    if value.shape != (dataset.num_samples,):
        raise ValueError(
            f"assembled metadata {name!r} must have shape "
            f"[{dataset.num_samples}], got {value.shape}"
        )
    return value


def _one_group_value(
    dataset: PreparedDataset,
    field_name: str,
    indices: np.ndarray,
) -> Any:
    values = np.unique(_sample_field(dataset, field_name)[indices])
    if values.size != 1:
        raise ValueError(
            f"group contains multiple {field_name!r} values: {values.tolist()}"
        )
    return values[0].item()


def _uniform_positions(total: int, requested: int) -> np.ndarray:
    if requested > total:
        raise ValueError(f"cannot select {requested} regions from only {total}")
    if requested == total:
        return np.arange(total, dtype=np.int64)
    positions = np.floor(
        (np.arange(requested, dtype=np.float64) + 0.5) * total / requested
    ).astype(np.int64)
    if np.unique(positions).size != requested:
        raise RuntimeError("uniform region selection produced duplicates")
    return positions


def _selected_group_ids(
    dataset: PreparedDataset,
    group_ids: np.ndarray,
    regions_per_source: int | None,
) -> np.ndarray:
    unique_groups = np.unique(group_ids)
    if regions_per_source is None:
        return unique_groups
    source_ids = _sample_field(dataset, "sample_source_id").astype(str)
    selected: list[np.ndarray] = []
    for source_id in sorted(set(source_ids.tolist())):
        source_groups = np.unique(group_ids[source_ids == source_id])
        positions = _uniform_positions(source_groups.size, regions_per_source)
        selected.append(source_groups[positions])
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def _region_x(samples: np.ndarray) -> np.ndarray:
    iq = np.asarray(samples, dtype=np.complex64).reshape(-1)
    x = np.empty((1, 2, iq.size), dtype=np.float32)
    x[0, 0] = iq.real
    x[0, 1] = iq.imag
    return x


def _canonical_feature_input(
    samples: np.ndarray,
    *,
    snr_db: float | None,
    noise_seed: int | None,
) -> tuple[np.ndarray, float | None]:
    x = _region_x(samples)
    if snr_db is None:
        canonical = standardize_iq_windows(x)
        achieved_snr = None
    else:
        if noise_seed is None:
            raise ValueError("noise_seed is required when snr_db is set")
        canonical, achieved = add_complex_awgn(
            x,
            snr_db,
            rng=np.random.default_rng(noise_seed),
        )
        achieved_snr = float(achieved[0])
    return canonical[:, :, :MAX_SIGNAL_LENGTH], achieved_snr


def _validate_noise_options(
    train_awgn_copies: int,
    awgn_snr_min: float,
    awgn_snr_max: float,
    test_awgn_snr: float | None,
) -> tuple[int, float, float, float | None]:
    if isinstance(train_awgn_copies, bool) or int(train_awgn_copies) != train_awgn_copies:
        raise TypeError("train_awgn_copies must be an integer")
    train_awgn_copies = int(train_awgn_copies)
    if train_awgn_copies < 0:
        raise ValueError("train_awgn_copies must be non-negative")
    awgn_snr_min = float(awgn_snr_min)
    awgn_snr_max = float(awgn_snr_max)
    if not math.isfinite(awgn_snr_min) or not math.isfinite(awgn_snr_max):
        raise ValueError("AWGN SNR bounds must be finite")
    if awgn_snr_min > awgn_snr_max:
        raise ValueError("awgn_snr_min must not exceed awgn_snr_max")
    if test_awgn_snr is not None:
        test_awgn_snr = float(test_awgn_snr)
        if not math.isfinite(test_awgn_snr):
            raise ValueError("test_awgn_snr must be finite")
    return train_awgn_copies, awgn_snr_min, awgn_snr_max, test_awgn_snr


def _conditions_for_group(
    split_name: str,
    *,
    train_awgn_copies: int,
    awgn_snr_min: float,
    awgn_snr_max: float,
    test_awgn_snr: float | None,
    rng: np.random.Generator,
) -> list[tuple[str, float | None, int | None]]:
    conditions: list[tuple[str, float | None, int | None]] = [
        ("clean", None, None)
    ]
    if split_name == "train":
        for _ in range(train_awgn_copies):
            snr_db = float(rng.uniform(awgn_snr_min, awgn_snr_max))
            noise_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            conditions.append(("train_awgn", snr_db, noise_seed))
    elif split_name == "test" and test_awgn_snr is not None:
        noise_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        conditions.append(("test_awgn", test_awgn_snr, noise_seed))
    return conditions


def _extract_split(
    split_path: Path,
    *,
    split_name: str,
    feature_service: FeatureExtractionService,
    backend: FeatureBackend,
    train_awgn_copies: int,
    awgn_snr_min: float,
    awgn_snr_max: float,
    test_awgn_snr: float | None,
    regions_per_source: int | None,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    dataset = load_prepared_dataset(split_path)
    if dataset.y is None:
        raise ValueError(f"assembled split has no labels: {split_path}")
    group_ids = _sample_field(dataset, "group_id").astype(np.int64, copy=False)

    features: list[np.ndarray] = []
    labels: list[int] = []
    output_group_ids: list[int] = []
    source_region_ids: list[int] = []
    source_ids: list[str] = []
    conditions: list[str] = []
    requested_snr: list[float] = []
    achieved_snr: list[float] = []
    noise_seeds: list[int] = []
    original_counts: list[int] = []
    used_counts: list[int] = []
    sample_rates: list[float] = []

    unique_groups = _selected_group_ids(
        dataset, group_ids, regions_per_source
    )
    for group_position, group_id in enumerate(unique_groups, start=1):
        indices = np.flatnonzero(group_ids == group_id)
        unique_labels = np.unique(dataset.y[indices])
        if unique_labels.size != 1:
            raise ValueError(f"group_id={int(group_id)} contains multiple labels")
        label = int(unique_labels[0])
        source_region_id = int(
            _one_group_value(dataset, "source_region_id", indices)
        )
        direct_fields = {
            "window_start_sample",
            "window_end_sample",
            "region_start_sample",
            "region_end_sample",
            "sample_source_id",
            "sample_rate",
        }
        use_direct_window = indices.size == 1 and direct_fields.issubset(
            dataset.meta
        )
        if use_direct_window:
            index = int(indices[0])
            window_start = int(dataset.meta["window_start_sample"][index])
            window_end = int(dataset.meta["window_end_sample"][index])
            region_start = int(dataset.meta["region_start_sample"][index])
            region_end = int(dataset.meta["region_end_sample"][index])
            use_direct_window = (
                window_start == region_start
                and window_end == region_end
                and region_end - region_start == dataset.seq_len
            )
        if use_direct_window:
            samples = (
                dataset.X[index, 0] + 1j * dataset.X[index, 1]
            ).astype(np.complex64, copy=False)
            region_source_id = str(dataset.meta["sample_source_id"][index])
            region_sample_rate = float(dataset.meta["sample_rate"][index])
        else:
            region = load_complete_region(
                split_path, dataset, group_id=int(group_id)
            )
            samples = region.samples
            region_source_id = region.source_id
            region_sample_rate = region.sample_rate

        for condition, snr_db, noise_seed in _conditions_for_group(
            split_name,
            train_awgn_copies=train_awgn_copies,
            awgn_snr_min=awgn_snr_min,
            awgn_snr_max=awgn_snr_max,
            test_awgn_snr=test_awgn_snr,
            rng=rng,
        ):
            feature_x, actual_snr = _canonical_feature_input(
                samples,
                snr_db=snr_db,
                noise_seed=noise_seed,
            )
            result = feature_service.extract(
                PreparedDataset(X=feature_x, source_id=region_source_id),
                sample_rate=region_sample_rate,
                backend=backend,
                progress_every=0,
            )
            row = result.features[0]
            if not np.all(np.isfinite(row)):
                raise ValueError(
                    f"non-finite feature output for split={split_name}, "
                    f"group_id={int(group_id)}, condition={condition}"
                )
            features.append(row)
            labels.append(label)
            output_group_ids.append(int(group_id))
            source_region_ids.append(source_region_id)
            source_ids.append(region_source_id)
            conditions.append(condition)
            requested_snr.append(np.nan if snr_db is None else snr_db)
            achieved_snr.append(np.nan if actual_snr is None else actual_snr)
            noise_seeds.append(-1 if noise_seed is None else noise_seed)
            original_counts.append(int(samples.size))
            used_counts.append(min(int(samples.size), MAX_SIGNAL_LENGTH))
            sample_rates.append(region_sample_rate)

        if group_position % 25 == 0 or group_position == len(unique_groups):
            print(
                f"[region-feature] {split_name}: "
                f"{group_position}/{len(unique_groups)} regions"
            )

    return {
        "features": np.asarray(features, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "group_id": np.asarray(output_group_ids, dtype=np.int64),
        "source_region_id": np.asarray(source_region_ids, dtype=np.int64),
        "sample_source_id": np.asarray(source_ids),
        "condition": np.asarray(conditions),
        "requested_snr_db": np.asarray(requested_snr, dtype=np.float64),
        "achieved_snr_db": np.asarray(achieved_snr, dtype=np.float64),
        "noise_seed": np.asarray(noise_seeds, dtype=np.int64),
        "original_sample_count": np.asarray(original_counts, dtype=np.int64),
        "used_sample_count": np.asarray(used_counts, dtype=np.int64),
        "sample_rate": np.asarray(sample_rates, dtype=np.float64),
        "feature_names": np.asarray(feature_service.feature_names),
        "feature_schema_id": np.asarray(FEATURE_SCHEMA_ID),
        "feature_dataset_version": np.asarray(REGION_FEATURE_DATASET_VERSION),
        "split": np.asarray(split_name),
    }


def build_region_feature_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    train_awgn_copies: int = 1,
    awgn_snr_min: float = 5.0,
    awgn_snr_max: float = 20.0,
    test_awgn_snr: float | None = 5.0,
    seed: int = 44,
    regions_per_source: int | None = None,
    overwrite: bool = False,
    feature_backend: FeatureBackend | None = None,
) -> dict[str, Any]:
    """Extract fixed region feature splits without changing source IQ datasets."""

    (
        train_awgn_copies,
        awgn_snr_min,
        awgn_snr_max,
        test_awgn_snr,
    ) = _validate_noise_options(
        train_awgn_copies,
        awgn_snr_min,
        awgn_snr_max,
        test_awgn_snr,
    )
    if isinstance(seed, bool) or int(seed) != seed:
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if regions_per_source is not None:
        if (
            isinstance(regions_per_source, bool)
            or int(regions_per_source) != regions_per_source
            or regions_per_source <= 0
        ):
            raise ValueError("regions_per_source must be positive or None")
        regions_per_source = int(regions_per_source)

    source_dir = Path(dataset_dir).resolve()
    output_directory = Path(output_dir)
    source_paths = {
        split: source_dir / f"{split}.npz" for split in SPLIT_NAMES
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing assembled split files: " + ", ".join(missing))
    output_paths = {
        split: output_directory / f"{split}.npz" for split in SPLIT_NAMES
    }
    report_path = output_directory / "feature_extraction_report.json"
    existing = [path for path in (*output_paths.values(), report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "feature dataset output exists; use overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )

    reference = load_prepared_dataset(source_paths["train"])
    source_dataset_id = str(_scalar(reference.meta, "dataset_id"))
    label_map_json = str(_scalar(reference.meta, "label_map_json"))
    label_map = json.loads(label_map_json)
    feature_service = FeatureExtractionService()
    rng = np.random.default_rng(seed)

    def extract_all(backend: FeatureBackend) -> dict[str, dict[str, np.ndarray]]:
        return {
            split: _extract_split(
                source_paths[split],
                split_name=split,
                feature_service=feature_service,
                backend=backend,
                train_awgn_copies=train_awgn_copies,
                awgn_snr_min=awgn_snr_min,
                awgn_snr_max=awgn_snr_max,
                test_awgn_snr=test_awgn_snr,
                regions_per_source=regions_per_source,
                rng=rng,
            )
            for split in SPLIT_NAMES
        }

    if feature_backend is None:
        with IQFeatureCtypesBackend() as backend:
            extracted = extract_all(backend)
    else:
        extracted = extract_all(feature_backend)

    output_directory.mkdir(parents=True, exist_ok=True)
    for split, arrays in extracted.items():
        np.savez_compressed(
            output_paths[split],
            **arrays,
            source_dataset_id=np.asarray(source_dataset_id),
            dataset_id=np.asarray(f"{source_dataset_id}_region_features"),
            label_map_json=np.asarray(label_map_json),
        )

    report = {
        "schema_version": 1,
        "feature_dataset_version": REGION_FEATURE_DATASET_VERSION,
        "dataset_id": f"{source_dataset_id}_region_features",
        "source_dataset_id": source_dataset_id,
        "source_dataset_dir": str(source_dir),
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_count": FEATURE_COUNT,
        "maximum_region_samples": MAX_SIGNAL_LENGTH,
        "label_map": label_map,
        "configuration": {
            "train_awgn_copies": train_awgn_copies,
            "awgn_snr_min": awgn_snr_min,
            "awgn_snr_max": awgn_snr_max,
            "test_awgn_snr": test_awgn_snr,
            "seed": seed,
            "region_preprocessing": "remove_dc_then_complex_rms_normalize",
            "truncation_policy": "keep_first_samples",
            "regions_per_source": regions_per_source,
        },
        "splits": {
            split: {
                "path": str(output_paths[split]),
                "shape": list(arrays["features"].shape),
                "region_count": int(np.unique(arrays["group_id"]).size),
                "condition_counts": {
                    condition: int(np.count_nonzero(arrays["condition"] == condition))
                    for condition in np.unique(arrays["condition"])
                },
                "class_counts": {
                    str(label): int(np.count_nonzero(arrays["y"] == int(label)))
                    for label in sorted(label_map, key=int)
                },
                "truncated_count": int(
                    np.count_nonzero(
                        arrays["original_sample_count"] > MAX_SIGNAL_LENGTH
                    )
                ),
            }
            for split, arrays in extracted.items()
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_path"] = str(report_path)
    return report


def extract_region_feature_split(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    split_name: str,
    regions_per_source: int | None = None,
    seed: int = 44,
    overwrite: bool = False,
    feature_backend: FeatureBackend | None = None,
) -> dict[str, Any]:
    """Extract clean 62-dimensional features from one prepared IQ split."""

    if not isinstance(split_name, str) or not split_name.strip():
        raise ValueError("split_name must be a non-empty string")
    if regions_per_source is not None:
        if (
            isinstance(regions_per_source, bool)
            or int(regions_per_source) != regions_per_source
            or regions_per_source <= 0
        ):
            raise ValueError("regions_per_source must be positive or None")
        regions_per_source = int(regions_per_source)
    if isinstance(seed, bool) or int(seed) != seed:
        raise TypeError("seed must be an integer")
    seed = int(seed)

    source_path = Path(dataset_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"prepared IQ split does not exist: {source_path}")
    output_directory = Path(output_dir)
    output_path = output_directory / f"{split_name}.npz"
    report_path = output_directory / f"{split_name}_feature_extraction_report.json"
    existing = [path for path in (output_path, report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "feature split output exists; use overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )

    source = load_prepared_dataset(source_path)
    source_dataset_id = str(_scalar(source.meta, "dataset_id"))
    label_map_json = str(_scalar(source.meta, "label_map_json"))
    service = FeatureExtractionService()

    def extract(backend: FeatureBackend) -> dict[str, np.ndarray]:
        return _extract_split(
            source_path,
            split_name=split_name,
            feature_service=service,
            backend=backend,
            train_awgn_copies=0,
            awgn_snr_min=5.0,
            awgn_snr_max=20.0,
            test_awgn_snr=None,
            regions_per_source=regions_per_source,
            rng=np.random.default_rng(seed),
        )

    if feature_backend is None:
        with IQFeatureCtypesBackend() as backend:
            arrays = extract(backend)
    else:
        arrays = extract(feature_backend)

    feature_dataset_id = f"{source_dataset_id}_region_features"
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        source_dataset_id=np.asarray(source_dataset_id),
        dataset_id=np.asarray(feature_dataset_id),
        label_map_json=np.asarray(label_map_json),
    )
    report = {
        "schema_version": 1,
        "feature_dataset_version": REGION_FEATURE_DATASET_VERSION,
        "dataset_id": feature_dataset_id,
        "source_dataset_id": source_dataset_id,
        "source_dataset_path": str(source_path),
        "split": split_name,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_count": FEATURE_COUNT,
        "regions_per_source": regions_per_source,
        "source_count": int(np.unique(arrays["sample_source_id"]).size),
        "region_count": int(arrays["features"].shape[0]),
        "shape": list(arrays["features"].shape),
        "class_counts": {
            str(label): int(np.count_nonzero(arrays["y"] == int(label)))
            for label in sorted(json.loads(label_map_json), key=int)
        },
        "output_path": str(output_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_path"] = str(report_path)
    return report


def load_region_feature_split(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate one generated region-feature split."""

    split_path = Path(path)
    with np.load(split_path, allow_pickle=False) as container:
        required = {
            "features",
            "y",
            "group_id",
            "sample_source_id",
            "condition",
            "feature_names",
            "feature_schema_id",
            "dataset_id",
            "label_map_json",
        }
        missing = sorted(required.difference(container.files))
        if missing:
            raise ValueError(f"feature split is missing fields: {missing}")
        result = {name: np.array(container[name], copy=True) for name in container.files}
    features = result["features"]
    labels = result["y"]
    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError(f"features must have shape [N, {FEATURE_COUNT}]")
    if features.dtype != np.float32 or not np.all(np.isfinite(features)):
        raise ValueError("features must be finite float32 values")
    if labels.shape != (features.shape[0],) or labels.dtype != np.int64:
        raise ValueError("y must be int64 with one label per feature row")
    for name in ("group_id", "sample_source_id", "condition"):
        if result[name].shape != (features.shape[0],):
            raise ValueError(f"{name} must contain one value per feature row")
    if str(result["feature_schema_id"].reshape(()).item()) != FEATURE_SCHEMA_ID:
        raise ValueError("feature_schema_id does not match the runtime schema")
    if result["feature_names"].shape != (FEATURE_COUNT,):
        raise ValueError(f"feature_names must contain {FEATURE_COUNT} entries")
    return result


__all__ = [
    "REGION_FEATURE_DATASET_VERSION",
    "build_region_feature_dataset",
    "extract_region_feature_split",
    "load_region_feature_split",
]
