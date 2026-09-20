"""Native-10 MS/s cross-rate preparation for the public benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.contracts import PreparedDataset
from signal_fusion.evaluation.snr import standardize_iq_windows
from signal_fusion.io import load_prepared_dataset, write_prepared_dataset
from signal_fusion.io.writers import json_safe
from signal_fusion.preparation import (
    FixedBlockDetector,
    PreparationConfig,
    ResamplingConfig,
    build_prepared_dataset,
)

from .technology_recognition import LABELS, profile_complex64_recording

CROSS_RATE_DATASET_ID = (
    "technology_recognition_lte_wifi_dvbt_external_10msps_to_1msps_2048"
)
SOURCE_SAMPLE_RATE = 10_000_000
TARGET_SAMPLE_RATE = 1_000_000
SOURCE_REGION_SIZE = 40_960
TARGET_REGION_SIZE = 4_096
CROSS_RATE_WINDOW_SIZE = 2_048
WINDOWS_PER_REGION = TARGET_REGION_SIZE // CROSS_RATE_WINDOW_SIZE
EXPECTED_FILE_COUNTS = {"LTE": 12, "WiFi": 9, "DVB-T": 5}


def _technology_slug(technology: str) -> str:
    return technology.lower().replace("-", "")


def _selected_recordings(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in inventory["recordings"]
        if entry.get("selected_for_v1", False)
    ]


def select_external_10msps_recordings(
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Select all native-10 MS/s recordings without hiding metadata mismatches."""

    root = Path(str(inventory["dataset_root"]))
    recordings = [dict(entry) for entry in inventory["recordings"]]
    selected: list[dict[str, Any]] = []
    for entry in recordings:
        is_selected = entry["sample_rate"] == SOURCE_SAMPLE_RATE
        entry["selected_for_v1"] = is_selected
        if not is_selected:
            continue
        if not entry["file_size_aligned"]:
            raise ValueError(
                "native-10 MS/s recording is not aligned complex64: "
                f"{entry['relative_path']}"
            )

        frequency_mhz = entry["center_frequency_hz"] // 1_000_000
        slug = _technology_slug(entry["technology"])
        entry["evaluation_group"] = "native_10msps_resampled_to_1msps"
        entry["filename_location_mismatch"] = (
            entry["filename_location"] != entry["location"]
        )
        entry["source_id"] = (
            f"tr_external10_{entry['location']}_{slug}_f{frequency_mhz}_"
            f"g{entry['gain_db']}_r{entry['run']:02d}"
        )
        entry.update(profile_complex64_recording(root / entry["relative_path"]))
        selected.append(entry)

    actual_counts = Counter(entry["technology"] for entry in selected)
    if dict(actual_counts) != EXPECTED_FILE_COUNTS:
        raise ValueError(
            "unexpected external 10 MS/s file counts: "
            f"expected={EXPECTED_FILE_COUNTS}, actual={dict(actual_counts)}"
        )
    if any(not entry["finite"] for entry in selected):
        raise ValueError("external 10 MS/s recordings contain non-finite IQ values")
    source_ids = [entry["source_id"] for entry in selected]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("external 10 MS/s source_id values are not unique")

    mismatch_count = sum(
        bool(entry["filename_location_mismatch"]) for entry in selected
    )
    return {
        **inventory,
        "schema_version": 1,
        "dataset_id": CROSS_RATE_DATASET_ID,
        "recording_set": "external_10msps",
        "recording_selection": {
            "source_sample_rate": SOURCE_SAMPLE_RATE,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "selected_recordings": len(selected),
            "file_counts_by_class": dict(actual_counts),
            "filename_location_mismatch_count": mismatch_count,
            "selection": "all native-10 MS/s recordings",
        },
        "preparation": {
            "source_region_size": SOURCE_REGION_SIZE,
            "target_region_size": TARGET_REGION_SIZE,
            "regions_per_source": None,
            "region_selection": "all_complete_source_blocks",
            "window_size": CROSS_RATE_WINDOW_SIZE,
            "window_hop": CROSS_RATE_WINDOW_SIZE,
            "windows_per_region": WINDOWS_PER_REGION,
            "window_selection": "all_non_overlapping",
            "resampling": "polyphase_kaiser5_v1",
            "normalization": "none",
            "remove_dc": False,
        },
        "recordings": recordings,
    }


def prepare_cross_rate_sources(
    inventory: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Resample complete 40,960-point native regions and make 2048 windows."""

    if inventory.get("recording_set") != "external_10msps":
        raise ValueError("inventory is not the external_10msps recording set")

    root = Path(str(inventory["dataset_root"]))
    output = Path(output_dir)
    prepared_root = output / "prepared_sources"
    if prepared_root.exists() and any(prepared_root.rglob("*.npz")):
        raise FileExistsError(
            f"prepared output already exists; choose a new directory: {prepared_root}"
        )

    selected = _selected_recordings(inventory)
    if not selected:
        raise ValueError("inventory does not select any recordings")
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(selected, start=1):
        if entry["sample_rate"] != SOURCE_SAMPLE_RATE:
            raise ValueError(
                f"cross-rate source is not 10 MS/s: {entry['source_id']}"
            )
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
                block_size_samples=SOURCE_REGION_SIZE,
                block_count=None,
            ),
            config=PreparationConfig(
                seq_len=CROSS_RATE_WINDOW_SIZE,
                hop_len=CROSS_RATE_WINDOW_SIZE,
                remainder="drop",
                normalization="none",
                remove_dc=False,
                label=LABELS[entry["technology"]],
                class_name=entry["technology"],
            ),
            resampling=ResamplingConfig(target_sample_rate=TARGET_SAMPLE_RATE),
        )
        expected_regions = entry["complex_sample_count"] // SOURCE_REGION_SIZE
        expected_windows = expected_regions * WINDOWS_PER_REGION
        if result["num_regions"] != expected_regions:
            raise RuntimeError(
                f"unexpected region count for {entry['source_id']}: {result}"
            )
        if result["num_samples"] != expected_windows:
            raise RuntimeError(
                f"unexpected window count for {entry['source_id']}: {result}"
            )
        results.append(
            {
                "index": index,
                "source_id": entry["source_id"],
                "location": entry["location"],
                "filename_location": entry["filename_location"],
                "filename_location_mismatch": entry[
                    "filename_location_mismatch"
                ],
                "technology": entry["technology"],
                **result,
            }
        )

    region_counts = [result["num_regions"] for result in results]
    report = {
        "schema_version": 1,
        "dataset_id": inventory["dataset_id"],
        "prepared_source_count": len(results),
        "source_sample_rate": SOURCE_SAMPLE_RATE,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "source_region_size": SOURCE_REGION_SIZE,
        "target_region_size": TARGET_REGION_SIZE,
        "window_size": CROSS_RATE_WINDOW_SIZE,
        "windows_per_region": WINDOWS_PER_REGION,
        "regions_per_source": (
            region_counts[0] if len(set(region_counts)) == 1 else None
        ),
        "regions_per_source_min": min(region_counts),
        "regions_per_source_max": max(region_counts),
        "region_selection": "all_complete_source_blocks",
        "total_regions": sum(region_counts),
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


_COORDINATE_FIELDS = (
    "window_id",
    "window_start_sample",
    "window_end_sample",
    "region_start_sample",
    "region_end_sample",
    "source_window_start_sample",
    "source_window_end_sample",
    "source_region_start_sample",
    "source_region_end_sample",
    "source_read_start_sample",
    "source_read_end_sample",
    "target_window_start_sample",
    "target_window_end_sample",
    "target_region_start_sample",
    "target_region_end_sample",
    "target_read_start_sample",
    "target_read_end_sample",
)


def assemble_cross_rate_test_dataset(
    inventory: dict[str, Any],
    prepared_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Combine resampled sources into the frozen 2048-point cross-rate test set."""

    if inventory.get("recording_set") != "external_10msps":
        raise ValueError("inventory is not the external_10msps recording set")
    output = Path(output_path)
    report_path = output.with_name("test_assembly_report.json")
    existing = [path for path in (output, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "cross-rate test outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )

    selected = _selected_recordings(inventory)
    if not selected:
        raise ValueError("inventory does not select any recordings")
    base = Path(prepared_root)
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    metadata_chunks: dict[str, list[np.ndarray]] = {
        "group_id": [],
        "source_region_id": [],
        **{field: [] for field in _COORDINATE_FIELDS},
        "source_index": [],
        "sample_source_id": [],
        "sample_source_path": [],
        "sample_location": [],
        "sample_filename_location": [],
        "sample_filename_location_mismatch": [],
        "sample_evaluation_group": [],
        "sample_class_name": [],
        "source_sample_rate": [],
        "target_sample_rate": [],
        "sample_rate": [],
        "center_frequency": [],
        "gain_db": [],
        "run": [],
    }
    source_reports: list[dict[str, Any]] = []
    next_group_id = 0
    for source_index, entry in enumerate(selected):
        source_path = base / entry["location"] / f"{entry['source_id']}.npz"
        dataset = load_prepared_dataset(source_path)
        if dataset.seq_len != CROSS_RATE_WINDOW_SIZE or dataset.y is None:
            raise ValueError(
                f"cross-rate source must be labeled 2048-point IQ: {source_path}"
            )
        if np.unique(dataset.y).tolist() != [LABELS[entry["technology"]]]:
            raise ValueError(f"unexpected source label: {source_path}")
        if str(np.asarray(dataset.meta["coordinate_schema"]).item()) != "dual_rate_v1":
            raise ValueError(f"missing dual-rate coordinates: {source_path}")
        if (
            float(np.asarray(dataset.meta["source_sample_rate"]).item())
            != SOURCE_SAMPLE_RATE
        ):
            raise ValueError(f"unexpected native sample rate: {source_path}")
        if (
            float(np.asarray(dataset.meta["target_sample_rate"]).item())
            != TARGET_SAMPLE_RATE
        ):
            raise ValueError(f"unexpected target sample rate: {source_path}")

        region_ids = np.asarray(dataset.meta["region_id"], dtype=np.int64)
        unique_regions, inverse, counts = np.unique(
            region_ids, return_inverse=True, return_counts=True
        )
        if not np.all(counts == WINDOWS_PER_REGION):
            raise ValueError(
                f"cross-rate regions require two windows each: {source_path}"
            )
        source_region_starts = np.asarray(
            dataset.meta["source_region_start_sample"], dtype=np.int64
        )
        source_region_ends = np.asarray(
            dataset.meta["source_region_end_sample"], dtype=np.int64
        )
        target_region_starts = np.asarray(
            dataset.meta["target_region_start_sample"], dtype=np.int64
        )
        target_region_ends = np.asarray(
            dataset.meta["target_region_end_sample"], dtype=np.int64
        )
        if not np.all(
            source_region_ends - source_region_starts == SOURCE_REGION_SIZE
        ):
            raise ValueError(f"unexpected native region size: {source_path}")
        if not np.all(
            target_region_ends - target_region_starts == TARGET_REGION_SIZE
        ):
            raise ValueError(f"unexpected target region size: {source_path}")

        count = dataset.num_samples
        x_chunks.append(dataset.X)
        y_chunks.append(dataset.y)
        metadata_chunks["group_id"].append(next_group_id + inverse)
        next_group_id += unique_regions.size
        metadata_chunks["source_region_id"].append(region_ids)
        for field_name in _COORDINATE_FIELDS:
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
        metadata_chunks["sample_filename_location"].append(
            np.full(count, entry["filename_location"])
        )
        metadata_chunks["sample_filename_location_mismatch"].append(
            np.full(count, entry["filename_location_mismatch"], dtype=np.bool_)
        )
        metadata_chunks["sample_evaluation_group"].append(
            np.full(count, entry["evaluation_group"])
        )
        metadata_chunks["sample_class_name"].append(
            np.full(count, entry["technology"])
        )
        metadata_chunks["source_sample_rate"].append(
            np.full(count, SOURCE_SAMPLE_RATE, dtype=np.float64)
        )
        metadata_chunks["target_sample_rate"].append(
            np.full(count, TARGET_SAMPLE_RATE, dtype=np.float64)
        )
        metadata_chunks["sample_rate"].append(
            np.full(count, TARGET_SAMPLE_RATE, dtype=np.float64)
        )
        metadata_chunks["center_frequency"].append(
            np.full(count, entry["center_frequency_hz"], dtype=np.float64)
        )
        metadata_chunks["gain_db"].append(
            np.full(count, entry["gain_db"], dtype=np.int64)
        )
        metadata_chunks["run"].append(
            np.full(count, entry["run"], dtype=np.int64)
        )
        source_reports.append(
            {
                "source_id": entry["source_id"],
                "relative_path": entry["relative_path"],
                "location": entry["location"],
                "filename_location": entry["filename_location"],
                "filename_location_mismatch": entry[
                    "filename_location_mismatch"
                ],
                "technology": entry["technology"],
                "center_frequency_hz": entry["center_frequency_hz"],
                "gain_db": entry["gain_db"],
                "run": entry["run"],
                "region_count": int(unique_regions.size),
                "window_count": count,
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
            "split": "external_cross_rate_test",
            "seq_len": CROSS_RATE_WINDOW_SIZE,
            "windows_per_region": WINDOWS_PER_REGION,
            "source_region_size": SOURCE_REGION_SIZE,
            "target_region_size": TARGET_REGION_SIZE,
            "region_selection": "all_complete_source_blocks",
            "window_selection": "all_non_overlapping",
            "coordinate_schema": "dual_rate_v1",
            "detector_coordinate_system": "source",
            "segmentation_coordinate_system": "source",
            "region_coordinate_system": "target",
            "window_coordinate_system": "target",
            "resampling_profile": "polyphase_kaiser5_v1",
            "resample_up": 1,
            "resample_down": 10,
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
        source_id=f"{inventory['dataset_id']}:external_cross_rate_test",
    )
    write_prepared_dataset(test_dataset, output)

    group_ids = np.asarray(meta["group_id"], dtype=np.int64)
    region_labels = y[np.unique(group_ids, return_index=True)[1]]
    class_window_counts = {
        str(label): int(np.count_nonzero(y == label))
        for label in sorted(LABELS.values())
    }
    class_region_counts = {
        str(label): int(np.count_nonzero(region_labels == label))
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
        "region_count": int(np.unique(group_ids).size),
        "window_count": int(x.shape[0]),
        "class_window_counts": class_window_counts,
        "class_region_counts": class_region_counts,
        "source_sample_rate": SOURCE_SAMPLE_RATE,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "source_region_size": SOURCE_REGION_SIZE,
        "target_region_size": TARGET_REGION_SIZE,
        "window_size": CROSS_RATE_WINDOW_SIZE,
        "windows_per_region": WINDOWS_PER_REGION,
        "standardization": {
            "remove_dc": True,
            "rms_normalize": True,
            "max_abs_complex_mean": float(np.max(np.abs(complex_mean))),
            "output_rms_min": float(np.min(rms)),
            "output_rms_max": float(np.max(rms)),
        },
        "sources": source_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


__all__ = [
    "CROSS_RATE_DATASET_ID",
    "CROSS_RATE_WINDOW_SIZE",
    "SOURCE_REGION_SIZE",
    "SOURCE_SAMPLE_RATE",
    "TARGET_REGION_SIZE",
    "TARGET_SAMPLE_RATE",
    "WINDOWS_PER_REGION",
    "assemble_cross_rate_test_dataset",
    "prepare_cross_rate_sources",
    "select_external_10msps_recordings",
]
