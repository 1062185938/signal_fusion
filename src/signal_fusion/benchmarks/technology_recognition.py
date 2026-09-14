"""Public LTE/WiFi/DVB-T benchmark inventory and Phase 1 preparation."""

from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.evaluation.snr import standardize_iq_windows
from signal_fusion.io import load_prepared_dataset, write_prepared_dataset
from signal_fusion.io.writers import json_safe
from signal_fusion.preparation import (
    FixedBlockDetector,
    PreparationConfig,
    build_prepared_dataset,
)


DATASET_ID = "technology_recognition_lte_wifi_dvbt_v1_1msps"
ALL_REGIONS_DATASET_ID = (
    "technology_recognition_lte_wifi_dvbt_v2_all_regions_1msps"
)
EXTERNAL_1MSPS_DATASET_ID = (
    "technology_recognition_lte_wifi_dvbt_external_1msps_4096"
)
V1_LOCATIONS = ("gentbrugge", "merelbeke", "rabot", "reep")
LABELS = {"LTE": 0, "WiFi": 1, "DVB-T": 2}
REGION_SIZE = 4096
REGIONS_PER_SOURCE = 32
WINDOW_SIZE = 128
WINDOW_SIZES = (128, 512, 1024, 2048, 4096)
WINDOWS_PER_REGION = REGION_SIZE // WINDOW_SIZE
EXPECTED_COMPLEX_SAMPLES = 1_100_000
LOCATION_FOLDS = {
    "fold1": {
        "train": ("gentbrugge", "merelbeke"),
        "validation": ("rabot",),
        "test": ("reep",),
    },
    "fold2": {
        "train": ("gentbrugge", "reep"),
        "validation": ("merelbeke",),
        "test": ("rabot",),
    },
    "fold3": {
        "train": ("rabot", "reep"),
        "validation": ("gentbrugge",),
        "test": ("merelbeke",),
    },
    "fold4": {
        "train": ("merelbeke", "rabot"),
        "validation": ("reep",),
        "test": ("gentbrugge",),
    },
}
ALL_LOCATION_RUN_SPLITS = {
    "train": tuple(range(1, 9)),
    "validation": (9,),
    "test": (10,),
}

_FILENAME_PATTERN = re.compile(
    r"^(?P<technology>dvbt|lte|wf)(?P<rate>10Msps)?_g(?P<gain>\d+)_"
    r"(?P<location>[^_]+)_f(?P<center_frequency>\d+)MHz_"
    r"r(?P<run>\d+)\.bin$",
    re.IGNORECASE,
)
_TECHNOLOGY_NAMES = {"dvbt": "DVB-T", "lte": "LTE", "wf": "WiFi"}
_TECHNOLOGY_SLUGS = {"DVB-T": "dvbt", "LTE": "lte", "WiFi": "wifi"}


def parse_recording_filename(path: str | Path) -> dict[str, Any]:
    """Parse the naming convention published with the benchmark."""

    file_path = Path(path)
    match = _FILENAME_PATTERN.fullmatch(file_path.name)
    if match is None:
        raise ValueError(
            f"unrecognized technology-recognition filename: {file_path.name}"
        )
    raw = match.groupdict()
    technology = _TECHNOLOGY_NAMES[raw["technology"].lower()]
    sample_rate = 10_000_000 if raw["rate"] else 1_000_000
    return {
        "technology": technology,
        "sample_rate": sample_rate,
        "gain_db": int(raw["gain"]),
        "filename_location": raw["location"].lower(),
        "center_frequency_hz": int(raw["center_frequency"]) * 1_000_000,
        "run": int(raw["run"]),
    }


def profile_complex64_recording(path: Path) -> dict[str, Any]:
    samples = np.memmap(path, dtype=np.dtype("<c8"), mode="r")
    finite = True
    energy_sum = 0.0
    real_sum = 0.0
    imag_sum = 0.0
    for start in range(0, samples.size, 262_144):
        chunk = np.asarray(samples[start : start + 262_144])
        chunk_finite = np.isfinite(chunk)
        if not bool(np.all(chunk_finite)):
            finite = False
            continue
        real = chunk.real.astype(np.float64)
        imag = chunk.imag.astype(np.float64)
        energy_sum += float(np.dot(real, real) + np.dot(imag, imag))
        real_sum += float(real.sum())
        imag_sum += float(imag.sum())
    if not finite or samples.size == 0:
        return {"finite": finite, "rms": None, "dc_to_rms_ratio": None}
    rms = math.sqrt(energy_sum / samples.size)
    mean = complex(real_sum / samples.size, imag_sum / samples.size)
    return {
        "finite": True,
        "rms": rms,
        "dc_to_rms_ratio": abs(mean) / rms if rms > 0.0 else None,
    }


def _selection_reason(entry: dict[str, Any]) -> str | None:
    if entry["filename_location"] != entry["location"]:
        return "filename_location_mismatch"
    if entry["sample_rate"] != 1_000_000:
        return "non_v1_sample_rate"
    if entry["location"] not in V1_LOCATIONS:
        return "location_not_in_v1"
    if not 1 <= entry["run"] <= 10:
        return "run_not_in_1_to_10"
    if (
        entry["location"] == "merelbeke"
        and entry["technology"] == "WiFi"
        and entry["center_frequency_hz"] != 2_412_000_000
    ):
        return "extra_merelbeke_wifi_frequency"
    if not entry["file_size_aligned"]:
        return "invalid_complex64_file_size"
    return None


def build_v1_inventory(dataset_root: str | Path) -> dict[str, Any]:
    """Inspect all recordings and freeze the balanced four-location V1 subset."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"technology-recognition dataset not found: {root}")

    recordings: list[dict[str, Any]] = []
    parse_failures: list[str] = []
    for path in sorted(root.glob("*/*.bin")):
        try:
            parsed = parse_recording_filename(path)
        except ValueError:
            parse_failures.append(str(path.relative_to(root)))
            continue
        file_size = path.stat().st_size
        entry = {
            "relative_path": str(path.relative_to(root)),
            "location": path.parent.name.lower(),
            **parsed,
            "file_size_bytes": file_size,
            "file_size_aligned": file_size % 8 == 0,
            "complex_sample_count": file_size // 8,
        }
        reason = _selection_reason(entry)
        entry["selected_for_v1"] = reason is None
        entry["exclusion_reason"] = reason
        if reason is None:
            slug = _TECHNOLOGY_SLUGS[entry["technology"]]
            entry["source_id"] = (
                f"tr_{entry['location']}_{slug}_r{entry['run']:02d}"
            )
            entry.update(profile_complex64_recording(path))
        recordings.append(entry)

    selected = [entry for entry in recordings if entry["selected_for_v1"]]
    group_counts = Counter(
        (entry["location"], entry["technology"]) for entry in selected
    )
    expected_groups = {
        (location, technology): 10
        for location in V1_LOCATIONS
        for technology in LABELS
    }
    if dict(group_counts) != expected_groups:
        raise ValueError(
            "balanced V1 selection does not contain exactly 10 recordings per "
            f"location and technology: {dict(group_counts)}"
        )
    if parse_failures:
        raise ValueError(f"unrecognized .bin filenames: {parse_failures}")
    if any(not entry["finite"] for entry in selected):
        raise ValueError("selected V1 recordings contain non-finite IQ values")
    invalid_sizes = [
        entry["relative_path"]
        for entry in selected
        if entry["complex_sample_count"] != EXPECTED_COMPLEX_SAMPLES
    ]
    if invalid_sizes:
        raise ValueError(
            "selected V1 recordings do not contain 1,100,000 complex samples: "
            f"{invalid_sizes}"
        )
    source_ids = [entry["source_id"] for entry in selected]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("selected V1 source_id values are not unique")

    exclusion_counts = Counter(
        entry["exclusion_reason"]
        for entry in recordings
        if not entry["selected_for_v1"]
    )
    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_root": str(root),
        "iq_format": "little-endian interleaved float32 I/Q (complex64 layout)",
        "label_map": {str(label): name for name, label in LABELS.items()},
        "official_example_audit": {
            "discard_initial_samples": 0,
            "discard_first_100k_found": False,
            "tail_policy": (
                "drop samples after the final complete 4096-sample region"
            ),
            "evidence": [
                "script_for_accessing_bin_files.m reads from the first sample",
                "Processing .bin files and add SNR/inp_bin_ext_snr.m only drops the tail remainder",
            ],
        },
        "v1_selection": {
            "locations": list(V1_LOCATIONS),
            "sample_rate": 1_000_000,
            "recordings_per_location_and_class": 10,
            "selected_recordings": len(selected),
            "excluded_recordings": len(recordings) - len(selected),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
        },
        "preparation": {
            "region_size": REGION_SIZE,
            "regions_per_source": REGIONS_PER_SOURCE,
            "region_selection": "uniform_complete_blocks",
            "window_size": WINDOW_SIZE,
            "window_hop": WINDOW_SIZE,
            "windows_per_region": WINDOWS_PER_REGION,
            "window_selection": "all_non_overlapping",
            "normalization": "none",
            "remove_dc": False,
        },
        "recordings": recordings,
    }


def select_external_1msps_recordings(
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Select the unused native-1 MS/s recordings for external evaluation."""

    root = Path(str(inventory["dataset_root"]))
    recordings = [dict(entry) for entry in inventory["recordings"]]
    selected: list[dict[str, Any]] = []
    for entry in recordings:
        is_unseen_location = (
            entry["sample_rate"] == 1_000_000
            and entry["location"] in {"uz", "igent"}
        )
        is_unseen_frequency = (
            entry["sample_rate"] == 1_000_000
            and entry["location"] == "merelbeke"
            and entry["technology"] == "WiFi"
            and entry["center_frequency_hz"] == 5_180_000_000
        )
        entry["selected_for_v1"] = is_unseen_location or is_unseen_frequency
        if not entry["selected_for_v1"]:
            continue
        entry["evaluation_group"] = (
            "unseen_location" if is_unseen_location else "unseen_frequency"
        )
        slug = _TECHNOLOGY_SLUGS[entry["technology"]]
        frequency_mhz = entry["center_frequency_hz"] // 1_000_000
        entry["source_id"] = (
            f"tr_external_{entry['location']}_{slug}_f{frequency_mhz}_"
            f"r{entry['run']:02d}"
        )
        entry.update(
            profile_complex64_recording(root / entry["relative_path"])
        )
        selected.append(entry)

    expected_file_counts = {"LTE": 20, "WiFi": 4, "DVB-T": 20}
    actual_file_counts = Counter(entry["technology"] for entry in selected)
    if dict(actual_file_counts) != expected_file_counts:
        raise ValueError(
            "unexpected external 1 MS/s file counts: "
            f"expected={expected_file_counts}, actual={dict(actual_file_counts)}"
        )
    if any(not entry["finite"] for entry in selected):
        raise ValueError("external 1 MS/s recordings contain non-finite IQ values")
    source_ids = [entry["source_id"] for entry in selected]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("external 1 MS/s source_id values are not unique")

    return {
        **inventory,
        "schema_version": 1,
        "dataset_id": EXTERNAL_1MSPS_DATASET_ID,
        "recording_set": "external_1msps",
        "recording_selection": {
            "sample_rate": 1_000_000,
            "selected_recordings": len(selected),
            "file_counts_by_class": dict(actual_file_counts),
            "selection": [
                "all native-1 MS/s recordings from UZ and iGent",
                "native-1 MS/s Merelbeke WiFi recordings at 5180 MHz",
            ],
        },
        "preparation": {
            "region_size": REGION_SIZE,
            "regions_per_source": None,
            "region_selection": "all_complete_blocks",
            "window_size": REGION_SIZE,
            "window_hop": REGION_SIZE,
            "windows_per_region": 1,
            "window_selection": "all_non_overlapping",
            "normalization": "none",
            "remove_dc": False,
        },
        "recordings": recordings,
    }


def write_inventory(inventory: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(inventory), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def prepare_v1_sources(
    inventory: dict[str, Any],
    output_dir: str | Path,
    *,
    window_size: int = WINDOW_SIZE,
    region_count_per_source: int | None = REGIONS_PER_SOURCE,
) -> dict[str, Any]:
    """Prepare selected recordings with fixed 4096-point regions."""

    window_size = _validate_window_size(window_size)
    if region_count_per_source is not None:
        if (
            isinstance(region_count_per_source, bool)
            or int(region_count_per_source) != region_count_per_source
            or region_count_per_source <= 0
        ):
            raise ValueError("region_count_per_source must be positive or None")
        region_count_per_source = int(region_count_per_source)
    windows_per_region = REGION_SIZE // window_size
    root = Path(str(inventory["dataset_root"]))
    output = Path(output_dir)
    prepared_root = output / "prepared_sources"
    if prepared_root.exists() and any(prepared_root.rglob("*.npz")):
        raise FileExistsError(
            f"prepared output already exists; choose a new directory: {prepared_root}"
        )

    results: list[dict[str, Any]] = []
    selected = [
        entry for entry in inventory["recordings"] if entry["selected_for_v1"]
    ]
    for index, entry in enumerate(selected, start=1):
        source_path = root / entry["relative_path"]
        output_path = prepared_root / entry["location"] / f"{entry['source_id']}.npz"
        result = build_prepared_dataset(
            source_path,
            output_path,
            source_id=entry["source_id"],
            data_format="bin",
            reader_options={
                "sample_rate": entry["sample_rate"],
                "center_frequency": entry["center_frequency_hz"],
                "iq_format": "complex64",
            },
            detector=FixedBlockDetector(
                block_size_samples=REGION_SIZE,
                block_count=region_count_per_source,
            ),
            config=PreparationConfig(
                seq_len=window_size,
                hop_len=window_size,
                remainder="drop",
                normalization="none",
                remove_dc=False,
                label=LABELS[entry["technology"]],
                class_name=entry["technology"],
            ),
        )
        expected_region_count = (
            region_count_per_source
            if region_count_per_source is not None
            else entry["complex_sample_count"] // REGION_SIZE
        )
        if result["num_regions"] != expected_region_count:
            raise RuntimeError(
                f"unexpected region count for {entry['source_id']}: {result}"
            )
        expected_windows = expected_region_count * windows_per_region
        if result["num_samples"] != expected_windows:
            raise RuntimeError(
                f"unexpected window count for {entry['source_id']}: {result}"
            )
        results.append(
            {
                "index": index,
                "source_id": entry["source_id"],
                "location": entry["location"],
                "technology": entry["technology"],
                **result,
            }
        )

    region_counts = [result["num_regions"] for result in results]
    if not region_counts:
        raise ValueError("inventory does not select any recordings")
    unique_region_counts = set(region_counts)
    resolved_region_count = (
        region_counts[0] if len(unique_region_counts) == 1 else None
    )
    report = {
        "schema_version": 1,
        "dataset_id": inventory["dataset_id"],
        "prepared_source_count": len(results),
        "window_size": window_size,
        "regions_per_source": resolved_region_count,
        "regions_per_source_min": min(region_counts),
        "regions_per_source_max": max(region_counts),
        "region_selection": (
            "all_complete_blocks"
            if region_count_per_source is None
            else "uniform_complete_blocks"
        ),
        "windows_per_region": windows_per_region,
        "total_regions": sum(result["num_regions"] for result in results),
        "total_windows": sum(result["num_samples"] for result in results),
        "sources": results,
    }
    report_path = output / "preparation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def assemble_external_test_dataset(
    inventory: dict[str, Any],
    prepared_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Combine prepared 4096-point sources into one standardized test set."""

    if inventory.get("recording_set") != "external_1msps":
        raise ValueError("inventory is not the external_1msps recording set")
    output = Path(output_path)
    report_path = output.with_name("test_assembly_report.json")
    existing = [path for path in (output, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "external test outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )

    base = Path(prepared_root)
    selected = [
        entry for entry in inventory["recordings"] if entry["selected_for_v1"]
    ]
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    metadata_chunks: dict[str, list[np.ndarray]] = {
        "group_id": [],
        "source_region_id": [],
        "window_id": [],
        "window_start_sample": [],
        "window_end_sample": [],
        "region_start_sample": [],
        "region_end_sample": [],
        "source_index": [],
        "sample_source_id": [],
        "sample_source_path": [],
        "sample_location": [],
        "sample_evaluation_group": [],
        "sample_class_name": [],
        "sample_rate": [],
        "center_frequency": [],
    }
    source_reports: list[dict[str, Any]] = []
    next_group_id = 0
    for source_index, entry in enumerate(selected):
        source_path = (
            base / entry["location"] / f"{entry['source_id']}.npz"
        )
        dataset = load_prepared_dataset(source_path)
        if dataset.seq_len != REGION_SIZE or dataset.y is None:
            raise ValueError(
                f"external source must be labeled 4096-point IQ: {source_path}"
            )
        if np.unique(dataset.y).tolist() != [LABELS[entry["technology"]]]:
            raise ValueError(f"unexpected source label: {source_path}")
        region_ids = np.asarray(dataset.meta["region_id"], dtype=np.int64)
        if region_ids.shape != (dataset.num_samples,):
            raise ValueError(f"invalid region_id metadata: {source_path}")
        if np.unique(region_ids).size != dataset.num_samples:
            raise ValueError(
                f"4096-point external sources require one window per region: {source_path}"
            )

        count = dataset.num_samples
        x_chunks.append(dataset.X)
        y_chunks.append(dataset.y)
        metadata_chunks["group_id"].append(
            np.arange(next_group_id, next_group_id + count, dtype=np.int64)
        )
        next_group_id += count
        metadata_chunks["source_region_id"].append(region_ids)
        for field_name in (
            "window_id",
            "window_start_sample",
            "window_end_sample",
            "region_start_sample",
            "region_end_sample",
        ):
            metadata_chunks[field_name].append(
                np.asarray(dataset.meta[field_name], dtype=np.int64)
            )
        metadata_chunks["source_index"].append(
            np.full(count, source_index, dtype=np.int64)
        )
        metadata_chunks["sample_source_id"].append(
            np.full(count, entry["source_id"])
        )
        metadata_chunks["sample_source_path"].append(
            np.full(count, entry["relative_path"])
        )
        metadata_chunks["sample_location"].append(
            np.full(count, entry["location"])
        )
        metadata_chunks["sample_evaluation_group"].append(
            np.full(count, entry["evaluation_group"])
        )
        metadata_chunks["sample_class_name"].append(
            np.full(count, entry["technology"])
        )
        metadata_chunks["sample_rate"].append(
            np.full(count, entry["sample_rate"], dtype=np.float64)
        )
        metadata_chunks["center_frequency"].append(
            np.full(count, entry["center_frequency_hz"], dtype=np.float64)
        )
        source_reports.append(
            {
                "source_id": entry["source_id"],
                "relative_path": entry["relative_path"],
                "location": entry["location"],
                "evaluation_group": entry["evaluation_group"],
                "technology": entry["technology"],
                "center_frequency_hz": entry["center_frequency_hz"],
                "region_count": count,
            }
        )

    x = standardize_iq_windows(np.concatenate(x_chunks, axis=0))
    y = np.concatenate(y_chunks, axis=0).astype(np.int64, copy=False)
    meta = {
        field_name: np.concatenate(chunks, axis=0)
        for field_name, chunks in metadata_chunks.items()
    }
    meta.update(
        {
            "dataset_id": inventory["dataset_id"],
            "split": "external_test",
            "seq_len": REGION_SIZE,
            "windows_per_region": 1,
            "region_selection": "all_complete_blocks",
            "window_selection": "all_non_overlapping",
            "remove_dc": True,
            "rms_normalize": True,
            "rms_epsilon": 1e-12,
            "label_map_json": json.dumps(
                {str(label): name for name, label in LABELS.items()},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
    )
    test_dataset = PreparedDataset(
        X=x,
        y=y,
        meta=meta,
        source_id=f"{inventory['dataset_id']}:external_test",
    )
    write_prepared_dataset(test_dataset, output)

    class_counts = {
        str(label): int(np.count_nonzero(y == label))
        for label in sorted(LABELS.values())
    }
    complex_mean = x[:, 0, :].mean(axis=1) + 1j * x[:, 1, :].mean(axis=1)
    rms = np.sqrt(np.mean(np.square(x).sum(axis=1), axis=1))
    report = {
        "schema_version": 1,
        "dataset_id": inventory["dataset_id"],
        "output_path": str(output),
        "shape": list(x.shape),
        "source_count": len(selected),
        "region_count": int(x.shape[0]),
        "class_counts": class_counts,
        "standardization": {
            "remove_dc": True,
            "rms_normalize": True,
            "max_abs_complex_mean": float(np.max(np.abs(complex_mean))),
            "output_rms_min": float(np.min(rms)),
            "output_rms_max": float(np.max(rms)),
        },
        "sources": source_reports,
    }
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _validate_window_size(window_size: int) -> int:
    if isinstance(window_size, bool) or int(window_size) != window_size:
        raise ValueError("window_size must be an integer")
    resolved = int(window_size)
    if resolved not in WINDOW_SIZES:
        raise ValueError(f"window_size must be one of {WINDOW_SIZES}")
    return resolved


def profile_directory(output_dir: str | Path, window_size: int) -> Path:
    """Return the canonical prepared-data directory for one window length."""

    resolved = _validate_window_size(window_size)
    return Path(output_dir) / "profiles" / f"window_{resolved}"


def write_fold_manifest(
    inventory: dict[str, Any],
    prepared_root: str | Path,
    manifest_path: str | Path,
    *,
    fold_name: str,
    window_size: int,
) -> Path:
    """Write one location-disjoint fixed-split assembly manifest."""

    if fold_name not in LOCATION_FOLDS:
        raise ValueError(f"unknown fold_name={fold_name!r}")
    window_size = _validate_window_size(window_size)
    windows_per_region = REGION_SIZE // window_size
    prepared = Path(prepared_root).absolute()
    output = Path(manifest_path).resolve()
    if output.exists():
        raise FileExistsError(f"fold manifest already exists: {output}")

    split_for_location = {
        location: split
        for split, locations in LOCATION_FOLDS[fold_name].items()
        for location in locations
    }
    splits: dict[str, list[dict[str, Any]]] = {
        split: [] for split in LOCATION_FOLDS[fold_name]
    }
    for entry in inventory["recordings"]:
        if not entry["selected_for_v1"]:
            continue
        location = entry["location"]
        try:
            split = split_for_location[location]
        except KeyError as exc:
            raise ValueError(
                f"selected location has no fold assignment: {location}"
            ) from exc
        source_path = prepared / location / f"{entry['source_id']}.npz"
        if not source_path.is_file():
            raise FileNotFoundError(f"prepared source does not exist: {source_path}")
        splits[split].append(
            {
                "path": os.path.relpath(source_path, output.parent),
                "label": LABELS[entry["technology"]],
                "source_id": entry["source_id"],
            }
        )

    expected_source_counts = {"train": 60, "validation": 30, "test": 30}
    actual_source_counts = {split: len(sources) for split, sources in splits.items()}
    if actual_source_counts != expected_source_counts:
        raise ValueError(
            "unexpected fold source counts: "
            f"expected={expected_source_counts}, actual={actual_source_counts}"
        )

    manifest = {
        "schema_version": 1,
        "dataset_id": (
            f"{inventory['dataset_id']}_{fold_name}_clean_{window_size}"
        ),
        "label_map": {str(label): name for name, label in LABELS.items()},
        "assembly": {
            "windows_per_region": windows_per_region,
            "regions_per_class": "minimum",
            "region_selection": "uniform",
            "window_selection": "uniform",
            "remove_dc": True,
            "rms_normalize": True,
            "rms_epsilon": 1e-12,
            "require_disjoint_sources": True,
        },
        "splits": splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def write_all_location_manifest(
    inventory: dict[str, Any],
    prepared_root: str | Path,
    manifest_path: str | Path,
    *,
    window_size: int,
) -> Path:
    """Write a source-disjoint run split containing every core location."""

    window_size = _validate_window_size(window_size)
    windows_per_region = REGION_SIZE // window_size
    prepared = Path(prepared_root).absolute()
    output = Path(manifest_path).resolve()
    if output.exists():
        raise FileExistsError(f"all-location manifest already exists: {output}")

    split_for_run = {
        run: split
        for split, runs in ALL_LOCATION_RUN_SPLITS.items()
        for run in runs
    }
    splits: dict[str, list[dict[str, Any]]] = {
        split: [] for split in ALL_LOCATION_RUN_SPLITS
    }
    selected_locations: set[str] = set()
    for entry in inventory["recordings"]:
        if not entry["selected_for_v1"]:
            continue
        selected_locations.add(entry["location"])
        try:
            split = split_for_run[entry["run"]]
        except KeyError as exc:
            raise ValueError(
                f"selected recording has no run split: {entry['source_id']}"
            ) from exc
        source_path = (
            prepared / entry["location"] / f"{entry['source_id']}.npz"
        )
        if not source_path.is_file():
            raise FileNotFoundError(
                f"prepared source does not exist: {source_path}"
            )
        splits[split].append(
            {
                "path": os.path.relpath(source_path, output.parent),
                "label": LABELS[entry["technology"]],
                "source_id": entry["source_id"],
            }
        )

    if selected_locations != set(V1_LOCATIONS):
        raise ValueError(
            "all-location manifest requires the four core locations: "
            f"actual={sorted(selected_locations)}"
        )
    expected_source_counts = {"train": 96, "validation": 12, "test": 12}
    actual_source_counts = {
        split: len(sources) for split, sources in splits.items()
    }
    if actual_source_counts != expected_source_counts:
        raise ValueError(
            "unexpected all-location source counts: "
            f"expected={expected_source_counts}, actual={actual_source_counts}"
        )

    manifest = {
        "schema_version": 1,
        "dataset_id": (
            f"{inventory['dataset_id']}_all_locations_clean_{window_size}"
        ),
        "label_map": {str(label): name for name, label in LABELS.items()},
        "assembly": {
            "windows_per_region": windows_per_region,
            "regions_per_class": "minimum",
            "region_selection": "uniform",
            "window_selection": "uniform",
            "remove_dc": True,
            "rms_normalize": True,
            "rms_epsilon": 1e-12,
            "require_disjoint_sources": True,
        },
        "splits": splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def write_ablation_manifests(
    inventory: dict[str, Any],
    profiles_root: str | Path,
    manifests_dir: str | Path,
    *,
    window_sizes: tuple[int, ...] = WINDOW_SIZES,
) -> list[Path]:
    """Write every location-fold manifest for the requested window lengths."""

    resolved_sizes = tuple(_validate_window_size(size) for size in window_sizes)
    if len(set(resolved_sizes)) != len(resolved_sizes):
        raise ValueError("window_sizes must not contain duplicates")
    manifest_root = Path(manifests_dir)
    planned = [
        manifest_root / f"window_{size}_{fold_name}.json"
        for size in resolved_sizes
        for fold_name in LOCATION_FOLDS
    ]
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(
            "ablation manifests already exist: "
            + ", ".join(str(path) for path in existing)
        )

    outputs: list[Path] = []
    for size in resolved_sizes:
        prepared_root = (
            Path(profiles_root) / f"window_{size}" / "prepared_sources"
        )
        for fold_name in LOCATION_FOLDS:
            outputs.append(
                write_fold_manifest(
                    inventory,
                    prepared_root,
                    manifest_root / f"window_{size}_{fold_name}.json",
                    fold_name=fold_name,
                    window_size=size,
                )
            )
    return outputs


__all__ = [
    "ALL_REGIONS_DATASET_ID",
    "ALL_LOCATION_RUN_SPLITS",
    "DATASET_ID",
    "EXTERNAL_1MSPS_DATASET_ID",
    "EXPECTED_COMPLEX_SAMPLES",
    "LABELS",
    "LOCATION_FOLDS",
    "REGIONS_PER_SOURCE",
    "REGION_SIZE",
    "V1_LOCATIONS",
    "WINDOWS_PER_REGION",
    "WINDOW_SIZE",
    "WINDOW_SIZES",
    "assemble_external_test_dataset",
    "build_v1_inventory",
    "parse_recording_filename",
    "prepare_v1_sources",
    "profile_complex64_recording",
    "profile_directory",
    "select_external_1msps_recordings",
    "write_ablation_manifests",
    "write_all_location_manifest",
    "write_fold_manifest",
    "write_inventory",
]
