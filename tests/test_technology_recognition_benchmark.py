import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from signal_fusion.benchmarks.technology_recognition import (
    ALL_REGIONS_DATASET_ID,
    DATASET_ID,
    LABELS,
    LOCATION_FOLDS,
    V1_LOCATIONS,
    WINDOW_SIZES,
    parse_recording_filename,
    prepare_v1_sources,
    select_external_1msps_recordings,
    write_ablation_manifests,
    write_all_location_manifest,
    write_fold_manifest,
)
from signal_fusion.benchmarks.technology_recognition_ablation import (
    summarize_ablation_runs,
)
from signal_fusion.benchmarks.technology_recognition_awgn import (
    AWGN_SNR_DB_VALUES,
    AWGN_WINDOW_SIZES,
    summarize_awgn_runs,
)
from signal_fusion.benchmarks.technology_recognition_cross_rate import (
    CROSS_RATE_DATASET_ID,
    SOURCE_REGION_SIZE,
    assemble_cross_rate_test_dataset,
    prepare_cross_rate_sources,
    select_external_10msps_recordings,
)


class TechnologyRecognitionBenchmarkTests(unittest.TestCase):
    def test_selects_all_native_10msps_recordings_with_unique_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recordings = []
            class_counts = {"LTE": 12, "WiFi": 9, "DVB-T": 5}
            frequencies = {
                "LTE": 806_000_000,
                "WiFi": 2_412_000_000,
                "DVB-T": 482_000_000,
            }
            for technology, count in class_counts.items():
                for run in range(1, count + 1):
                    location = "gentbrugge"
                    path = root / location / f"{technology}_{run}.bin"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.ones(8, dtype=np.complex64).tofile(path)
                    recordings.append(
                        {
                            "relative_path": str(path.relative_to(root)),
                            "location": location,
                            "technology": technology,
                            "sample_rate": 10_000_000,
                            "gain_db": 20,
                            "filename_location": (
                                "uz" if technology == "LTE" and run == 1 else location
                            ),
                            "center_frequency_hz": frequencies[technology],
                            "run": run,
                            "file_size_aligned": True,
                            "complex_sample_count": 8,
                            "selected_for_v1": False,
                        }
                    )

            selected_inventory = select_external_10msps_recordings(
                {"dataset_root": str(root), "recordings": recordings}
            )
            selected = [
                entry
                for entry in selected_inventory["recordings"]
                if entry["selected_for_v1"]
            ]

        self.assertEqual(len(selected), 26)
        self.assertEqual(len({entry["source_id"] for entry in selected}), 26)
        self.assertEqual(
            selected_inventory["recording_selection"]["file_counts_by_class"],
            class_counts,
        )
        self.assertEqual(
            selected_inventory["recording_selection"][
                "filename_location_mismatch_count"
            ],
            1,
        )

    def test_prepares_and_assembles_dual_rate_test_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gentbrugge" / "lte10.bin"
            source.parent.mkdir(parents=True, exist_ok=True)
            sample_count = SOURCE_REGION_SIZE * 2
            phase = (
                np.arange(sample_count, dtype=np.float64)
                * (2.0 * np.pi * 100_000.0 / 10_000_000.0)
            )
            np.exp(1j * phase).astype(np.complex64).tofile(source)
            source_id = "tr_external10_gentbrugge_lte_f806_g20_r01"
            inventory = {
                "dataset_id": CROSS_RATE_DATASET_ID,
                "dataset_root": str(root),
                "recording_set": "external_10msps",
                "recordings": [
                    {
                        "relative_path": str(source.relative_to(root)),
                        "location": "gentbrugge",
                        "filename_location": "uz",
                        "filename_location_mismatch": True,
                        "evaluation_group": "native_10msps_resampled_to_1msps",
                        "technology": "LTE",
                        "sample_rate": 10_000_000,
                        "center_frequency_hz": 806_000_000,
                        "gain_db": 20,
                        "run": 1,
                        "complex_sample_count": sample_count,
                        "source_id": source_id,
                        "selected_for_v1": True,
                    }
                ],
            }
            profile = root / "profile"
            preparation = prepare_cross_rate_sources(inventory, profile)
            output = root / "test.npz"
            assembly = assemble_cross_rate_test_dataset(
                inventory,
                profile / "prepared_sources",
                output,
            )

            with np.load(output, allow_pickle=False) as dataset:
                self.assertEqual(dataset["X"].shape, (4, 2, 2048))
                np.testing.assert_array_equal(dataset["group_id"], [0, 0, 1, 1])
                np.testing.assert_array_equal(
                    dataset["source_window_start_sample"],
                    [0, 20480, 40960, 61440],
                )
                np.testing.assert_array_equal(
                    dataset["target_window_start_sample"],
                    [0, 2048, 4096, 6144],
                )
                np.testing.assert_array_equal(
                    dataset["source_sample_rate"],
                    np.full(4, 10_000_000),
                )
                np.testing.assert_array_equal(
                    dataset["target_sample_rate"],
                    np.full(4, 1_000_000),
                )
                self.assertTrue(
                    bool(dataset["sample_filename_location_mismatch"][0])
                )
                self.assertEqual(dataset["coordinate_schema"].item(), "dual_rate_v1")

        self.assertEqual(preparation["total_regions"], 2)
        self.assertEqual(preparation["total_windows"], 4)
        self.assertEqual(assembly["region_count"], 2)
        self.assertEqual(assembly["window_count"], 4)

    def test_writes_all_location_run_split_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            recordings = []
            for location in V1_LOCATIONS:
                for class_name, label in LABELS.items():
                    slug = class_name.lower().replace("-", "")
                    for run in range(1, 11):
                        source_id = f"{location}_{slug}_{run}"
                        path = prepared / location / f"{source_id}.npz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
                        recordings.append(
                            {
                                "location": location,
                                "technology": class_name,
                                "run": run,
                                "source_id": source_id,
                                "selected_for_v1": True,
                                "label": label,
                            }
                        )

            path = write_all_location_manifest(
                {"dataset_id": ALL_REGIONS_DATASET_ID, "recordings": recordings},
                prepared,
                root / "manifest.json",
                window_size=4096,
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(
                {name: len(items) for name, items in manifest["splits"].items()},
                {"train": 96, "validation": 12, "test": 12},
            )
            for split_name, expected_runs in {
                "train": set(range(1, 9)),
                "validation": {9},
                "test": {10},
            }.items():
                actual_runs = {
                    int(source["source_id"].rsplit("_", 1)[1])
                    for source in manifest["splits"][split_name]
                }
                self.assertEqual(actual_runs, expected_runs)

    def test_selects_unused_native_rate_external_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recordings = []

            def add(location, technology, run, frequency_hz):
                path = root / location / f"{technology}_{run}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.ones(8, dtype=np.complex64).tofile(path)
                recordings.append(
                    {
                        "relative_path": str(path.relative_to(root)),
                        "location": location,
                        "technology": technology,
                        "sample_rate": 1_000_000,
                        "center_frequency_hz": frequency_hz,
                        "run": run,
                        "selected_for_v1": False,
                    }
                )

            for location in ("uz", "igent"):
                for run in range(1, 11):
                    add(location, "LTE", run, 806_000_000)
                    add(location, "DVB-T", run, 482_000_000)
            add("igent", "WiFi", 1, 2_412_000_000)
            for run in range(1, 4):
                add("merelbeke", "WiFi", run, 5_180_000_000)

            selected = select_external_1msps_recordings(
                {
                    "dataset_root": str(root),
                    "recordings": recordings,
                }
            )

            entries = [
                entry
                for entry in selected["recordings"]
                if entry["selected_for_v1"]
            ]
            self.assertEqual(len(entries), 44)
            self.assertEqual(
                {entry["evaluation_group"] for entry in entries},
                {"unseen_location", "unseen_frequency"},
            )
            self.assertEqual(len({entry["source_id"] for entry in entries}), 44)

    def test_summarizes_selected_window_awgn_runs_across_folds(self):
        runs = []
        for size in AWGN_WINDOW_SIZES:
            for fold_index, fold_name in enumerate(LOCATION_FOLDS):
                conditions = []
                for condition_index, snr_db in enumerate(
                    (None, *AWGN_SNR_DB_VALUES)
                ):
                    accuracy = 99.0 - condition_index - fold_index
                    per_class = {
                        str(label): {
                            "class_name": name,
                            "mean": accuracy - label,
                            "std": 0.1,
                        }
                        for name, label in LABELS.items()
                    }
                    conditions.append(
                        {
                            "snr_db": snr_db,
                            "aggregate": {
                                "trial_count": 1 if snr_db is None else 5,
                                "window_accuracy_percent": {
                                    "mean": accuracy,
                                    "std": 0.2,
                                },
                                "group_accuracy_percent": {
                                    "mean": accuracy + 0.5,
                                    "std": 0.1,
                                },
                                "per_class_window_accuracy_percent": per_class,
                                "per_class_group_accuracy_percent": per_class,
                            },
                        }
                    )
                runs.append(
                    {
                        "window_size": size,
                        "fold": fold_name,
                        "conditions": conditions,
                    }
                )

        summaries = summarize_awgn_runs(runs)

        self.assertEqual(set(summaries), {"512", "4096"})
        self.assertEqual(
            summaries["512"]["conditions"]["clean"][
                "window_accuracy_percent"
            ]["mean"],
            97.5,
        )
        self.assertEqual(
            summaries["512"]["conditions"]["5_db"][
                "region_delta_from_clean_percent_points"
            ],
            -3.0,
        )
        self.assertEqual(
            summaries["4096"]["conditions"]["10_db"][
                "per_class_window_accuracy_percent"
            ]["1"]["class_name"],
            "WiFi",
        )
        self.assertEqual(
            summaries["512"]["conditions"]["5_db"][
                "per_class_region_accuracy_percent"
            ]["2"]["mean"],
            92.5,
        )

        selected = summarize_awgn_runs(runs, window_sizes=(4096,))
        self.assertEqual(set(selected), {"4096"})

    def test_summarizes_all_folds_by_window_size(self):
        runs = []
        for size in WINDOW_SIZES:
            for index, fold_name in enumerate(LOCATION_FOLDS):
                per_class = {
                    str(label): {"accuracy_percent": 80.0 + index}
                    for label in LABELS.values()
                }
                runs.append(
                    {
                        "window_size": size,
                        "fold": fold_name,
                        "window": {
                            "accuracy_percent": 80.0 + index,
                            "mean_predicted_confidence": 0.8,
                            "per_class": per_class,
                        },
                        "region": {"accuracy_percent": 90.0 + index},
                    }
                )

        summaries = summarize_ablation_runs(runs)

        self.assertEqual(set(summaries), {str(size) for size in WINDOW_SIZES})
        self.assertEqual(
            summaries["128"]["window_accuracy_percent"]["mean"], 81.5
        )
        self.assertEqual(
            summaries["4096"]["region_accuracy_percent"]["mean"], 91.5
        )

    def test_summarizes_only_requested_window_sizes(self):
        runs = []
        for size in (512, 4096):
            for index, fold_name in enumerate(LOCATION_FOLDS):
                per_class = {
                    str(label): {"accuracy_percent": 90.0 + index}
                    for label in LABELS.values()
                }
                runs.append(
                    {
                        "window_size": size,
                        "fold": fold_name,
                        "window": {
                            "accuracy_percent": 90.0 + index,
                            "mean_predicted_confidence": 0.9,
                            "per_class": per_class,
                        },
                        "region": {"accuracy_percent": 95.0 + index},
                    }
                )

        summaries = summarize_ablation_runs(
            runs, window_sizes=(512, 4096)
        )

        self.assertEqual(set(summaries), {"512", "4096"})

    def test_writes_location_disjoint_fold_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_root = root / "prepared"
            recordings = []
            for location in V1_LOCATIONS:
                for technology in LABELS:
                    for run in range(1, 11):
                        slug = technology.lower().replace("-", "")
                        source_id = f"tr_{location}_{slug}_r{run:02d}"
                        source_path = prepared_root / location / f"{source_id}.npz"
                        source_path.parent.mkdir(parents=True, exist_ok=True)
                        source_path.touch()
                        recordings.append(
                            {
                                "selected_for_v1": True,
                                "location": location,
                                "technology": technology,
                                "source_id": source_id,
                            }
                        )
            inventory = {"dataset_id": DATASET_ID, "recordings": recordings}
            output = root / "configs" / "fold1.json"

            actual_path = write_fold_manifest(
                inventory,
                prepared_root,
                output,
                fold_name="fold1",
                window_size=512,
            )

            manifest = json.loads(actual_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {split: len(sources) for split, sources in manifest["splits"].items()},
                {"train": 60, "validation": 30, "test": 30},
            )
            self.assertEqual(manifest["assembly"]["windows_per_region"], 8)
            self.assertEqual(manifest["assembly"]["window_selection"], "uniform")
            self.assertEqual(
                manifest["dataset_id"], f"{DATASET_ID}_fold1_clean_512"
            )

    def test_location_folds_use_every_location_once_for_validation_and_test(self):
        validation_locations = [
            fold["validation"][0] for fold in LOCATION_FOLDS.values()
        ]
        test_locations = [fold["test"][0] for fold in LOCATION_FOLDS.values()]

        self.assertCountEqual(validation_locations, V1_LOCATIONS)
        self.assertCountEqual(test_locations, V1_LOCATIONS)
        for fold in LOCATION_FOLDS.values():
            assigned = set().union(*map(set, fold.values()))
            self.assertEqual(assigned, set(V1_LOCATIONS))

    def test_writes_complete_ablation_manifest_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles_root = root / "profiles"
            recordings = []
            for location in ("gentbrugge", "merelbeke", "rabot", "reep"):
                for technology in LABELS:
                    for run in range(1, 11):
                        slug = technology.lower().replace("-", "")
                        source_id = f"tr_{location}_{slug}_r{run:02d}"
                        recordings.append(
                            {
                                "selected_for_v1": True,
                                "location": location,
                                "technology": technology,
                                "source_id": source_id,
                            }
                        )
                        for size in WINDOW_SIZES:
                            path = (
                                profiles_root
                                / f"window_{size}"
                                / "prepared_sources"
                                / location
                                / f"{source_id}.npz"
                            )
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.touch()

            outputs = write_ablation_manifests(
                {"dataset_id": DATASET_ID, "recordings": recordings},
                profiles_root,
                root / "manifests",
            )

            self.assertEqual(len(outputs), 20)
            self.assertTrue(all(path.is_file() for path in outputs))

    def test_parses_published_filename_convention(self):
        parsed = parse_recording_filename(
            "wf10Msps_g76_rabot_f5240MHz_r1.bin"
        )

        self.assertEqual(parsed["technology"], "WiFi")
        self.assertEqual(parsed["sample_rate"], 10_000_000)
        self.assertEqual(parsed["gain_db"], 76)
        self.assertEqual(parsed["filename_location"], "rabot")
        self.assertEqual(parsed["center_frequency_hz"], 5_240_000_000)
        self.assertEqual(parsed["run"], 1)

    def test_prepares_32_regions_with_all_non_overlapping_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "capture.bin"
            phase = np.arange(40 * 4096, dtype=np.float32) * np.float32(0.01)
            iq = np.exp(1j * phase).astype(np.complex64)
            iq.tofile(source)
            output = root / "output"
            inventory = {
                "dataset_id": DATASET_ID,
                "dataset_root": str(root),
                "recordings": [
                    {
                        "relative_path": source.name,
                        "location": "rabot",
                        "technology": "LTE",
                        "sample_rate": 1_000_000,
                        "center_frequency_hz": 806_000_000,
                        "source_id": "tr_rabot_lte_r01",
                        "selected_for_v1": True,
                    }
                ],
            }

            report = prepare_v1_sources(inventory, output, window_size=512)

            prepared_path = (
                output
                / "prepared_sources"
                / "rabot"
                / "tr_rabot_lte_r01.npz"
            )
            with np.load(prepared_path, allow_pickle=False) as prepared:
                self.assertEqual(prepared["X"].shape, (256, 2, 512))
                self.assertEqual(np.unique(prepared["region_id"]).size, 32)
                np.testing.assert_array_equal(
                    np.unique(prepared["region_id"], return_counts=True)[1],
                    np.full(32, 8),
                )
                self.assertEqual(prepared["normalization"].item(), "none")
                self.assertFalse(bool(prepared["remove_dc"].item()))
                self.assertEqual(prepared["label"].item(), LABELS["LTE"])

        self.assertEqual(report["prepared_source_count"], 1)
        self.assertEqual(report["total_regions"], 32)
        self.assertEqual(report["total_windows"], 256)

    def test_prepares_all_complete_regions_when_region_count_is_none(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "capture.bin"
            block_count = 40
            phase = np.arange(block_count * 4096, dtype=np.float32) * np.float32(
                0.01
            )
            np.exp(1j * phase).astype(np.complex64).tofile(source)
            inventory = {
                "dataset_id": ALL_REGIONS_DATASET_ID,
                "dataset_root": str(root),
                "recordings": [
                    {
                        "relative_path": source.name,
                        "location": "rabot",
                        "technology": "LTE",
                        "sample_rate": 1_000_000,
                        "center_frequency_hz": 806_000_000,
                        "complex_sample_count": block_count * 4096,
                        "source_id": "tr_rabot_lte_r01",
                        "selected_for_v1": True,
                    }
                ],
            }

            report = prepare_v1_sources(
                inventory,
                root / "output",
                window_size=512,
                region_count_per_source=None,
            )

            self.assertEqual(report["regions_per_source"], block_count)
            self.assertEqual(report["region_selection"], "all_complete_blocks")
            self.assertEqual(report["total_regions"], block_count)
            self.assertEqual(report["total_windows"], block_count * 8)


if __name__ == "__main__":
    unittest.main()
