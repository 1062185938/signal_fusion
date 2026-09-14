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


def _profile_complex64(path: Path) -> dict[str, Any]:
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
            entry.update(_profile_complex64(path))
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

    region_counts = {result["num_regions"] for result in results}
    if len(region_counts) != 1:
        raise RuntimeError(
            f"prepared sources have inconsistent region counts: {region_counts}"
        )
    resolved_region_count = region_counts.pop()
    report = {
        "schema_version": 1,
        "dataset_id": inventory["dataset_id"],
        "prepared_source_count": len(results),
        "window_size": window_size,
        "regions_per_source": resolved_region_count,
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
    "DATASET_ID",
    "EXPECTED_COMPLEX_SAMPLES",
    "LABELS",
    "LOCATION_FOLDS",
    "REGIONS_PER_SOURCE",
    "REGION_SIZE",
    "V1_LOCATIONS",
    "WINDOWS_PER_REGION",
    "WINDOW_SIZE",
    "WINDOW_SIZES",
    "build_v1_inventory",
    "parse_recording_filename",
    "prepare_v1_sources",
    "profile_directory",
    "write_ablation_manifests",
    "write_fold_manifest",
    "write_inventory",
]
