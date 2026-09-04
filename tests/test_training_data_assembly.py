import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion.training_data import (
    assemble_training_dataset,
    load_assembly_manifest,
)
from signal_fusion.training_data.cli import main as assembly_cli_main


LABEL_MAP = {0: "ClassA", 1: "ClassB"}


def _write_source(
    path: Path,
    *,
    source_id: str,
    label: int,
    region_count: int,
    windows_per_region: int = 4,
    seq_len: int = 8,
    normalization_scale: np.ndarray | None = None,
) -> None:
    sample_count = region_count * windows_per_region
    region_id = np.repeat(np.arange(region_count), windows_per_region)
    window_id = np.tile(np.arange(windows_per_region), region_count)
    window_start = region_id * 1000 + window_id * seq_len
    window_end = window_start + seq_len
    region_start = region_id * 1000
    region_end = region_start + windows_per_region * seq_len

    phase = np.linspace(0.0, 2.0 * np.pi, seq_len, endpoint=False)
    X = np.empty((sample_count, 2, seq_len), dtype=np.float32)
    for index in range(sample_count):
        X[index, 0] = np.cos(phase + index * 0.1) + 2.0 + label
        X[index, 1] = np.sin(phase + index * 0.1) - 3.0 - label
    if normalization_scale is None:
        normalization_scale = np.ones(sample_count, dtype=np.float64)

    np.savez_compressed(
        path,
        X=X,
        y=np.full(sample_count, label, dtype=np.int64),
        region_id=region_id.astype(np.int64),
        window_id=window_id.astype(np.int64),
        window_start_sample=window_start.astype(np.int64),
        window_end_sample=window_end.astype(np.int64),
        region_start_sample=region_start.astype(np.int64),
        region_end_sample=region_end.astype(np.int64),
        sample_rate=np.asarray(4_000_000.0),
        center_frequency=np.asarray(1_000_000_000.0 + label),
        normalization_scale=np.asarray(normalization_scale, dtype=np.float64),
        seq_len=np.asarray(seq_len, dtype=np.int64),
        label=np.asarray(label, dtype=np.int64),
        class_name=np.asarray(LABEL_MAP[label]),
        source_id=np.asarray(source_id),
    )


def _write_manifest(
    path: Path,
    split_sources: dict[str, list[dict[str, object]]],
    *,
    windows_per_region: int = 2,
    window_selection: str = "uniform",
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "synthetic_v1",
                "label_map": {"0": "ClassA", "1": "ClassB"},
                "assembly": {
                    "windows_per_region": windows_per_region,
                    "regions_per_class": "minimum",
                    "region_selection": "uniform",
                    "window_selection": window_selection,
                    "remove_dc": True,
                    "rms_normalize": True,
                    "require_disjoint_sources": True,
                },
                "splits": split_sources,
            }
        ),
        encoding="utf-8",
    )


class TrainingDataAssemblyTests(unittest.TestCase):
    def _build_fixture(self, directory: str):
        root = Path(directory)
        split_sources: dict[str, list[dict[str, object]]] = {}
        for split_name in ("train", "validation", "test"):
            split_sources[split_name] = []
            for label in LABEL_MAP:
                source_id = f"{split_name}_class_{label}"
                source_path = root / f"{source_id}.npz"
                region_count = 3 if split_name == "train" and label == 0 else 2
                if split_name != "train":
                    region_count = 1
                _write_source(
                    source_path,
                    source_id=source_id,
                    label=label,
                    region_count=region_count,
                )
                split_sources[split_name].append(
                    {
                        "path": source_path.name,
                        "label": label,
                        "source_id": source_id,
                    }
                )
        manifest_path = root / "manifest.json"
        _write_manifest(manifest_path, split_sources)
        return manifest_path

    def test_manifest_resolves_paths_relative_to_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._build_fixture(directory)
            manifest = load_assembly_manifest(manifest_path)

        self.assertEqual(manifest.dataset_id, "synthetic_v1")
        self.assertEqual(manifest.windows_per_region, 2)
        self.assertEqual(manifest.label_map, LABEL_MAP)
        self.assertTrue(all(source.path.is_absolute() for source in manifest.splits["train"]))

    def test_assembly_balances_regions_and_standardizes_every_window(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._build_fixture(directory)
            output_dir = Path(directory) / "assembled"
            result = assemble_training_dataset(manifest_path, output_dir)

            expected_shapes = {
                "train": (8, 2, 8),
                "validation": (4, 2, 8),
                "test": (4, 2, 8),
            }
            split_source_sets: dict[str, set[str]] = {}
            all_group_ids: list[np.ndarray] = []
            for split_name, expected_shape in expected_shapes.items():
                with np.load(result.split_paths[split_name], allow_pickle=False) as data:
                    self.assertEqual(data["X"].shape, expected_shape)
                    self.assertEqual(data["X"].dtype, np.float32)
                    self.assertEqual(data["y"].dtype, np.int64)
                    self.assertEqual(
                        np.bincount(data["y"], minlength=2).tolist(),
                        [expected_shape[0] // 2, expected_shape[0] // 2],
                    )
                    complex_mean = data["X"][:, 0].mean(axis=1) + 1j * data[
                        "X"
                    ][:, 1].mean(axis=1)
                    rms = np.sqrt(np.mean(np.square(data["X"]).sum(axis=1), axis=1))
                    np.testing.assert_allclose(complex_mean, 0.0, atol=1e-6)
                    np.testing.assert_allclose(rms, 1.0, atol=1e-6)
                    unique_groups, group_counts = np.unique(
                        data["group_id"], return_counts=True
                    )
                    self.assertTrue(np.all(group_counts == 2))
                    np.testing.assert_array_equal(data["group_id"], data["burst_id"])
                    self.assertEqual(data["source_id"].item(), f"synthetic_v1:{split_name}")
                    split_source_sets[split_name] = set(
                        data["sample_source_id"].tolist()
                    )
                    all_group_ids.append(unique_groups)

            self.assertFalse(split_source_sets["train"] & split_source_sets["validation"])
            self.assertFalse(split_source_sets["train"] & split_source_sets["test"])
            self.assertFalse(split_source_sets["validation"] & split_source_sets["test"])
            concatenated_groups = np.concatenate(all_group_ids)
            self.assertEqual(
                np.unique(concatenated_groups).size, concatenated_groups.size
            )
            self.assertTrue(result.report["leakage_audit"]["passed"])
            self.assertTrue(result.report_path.is_file())

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._build_fixture(directory)
            output_dir = Path(directory) / "assembled"
            assemble_training_dataset(manifest_path, output_dir)
            with self.assertRaisesRegex(FileExistsError, "overwrite=True"):
                assemble_training_dataset(manifest_path, output_dir)
            result = assemble_training_dataset(
                manifest_path, output_dir, overwrite=True
            )

        self.assertEqual(result.report["dataset_id"], "synthetic_v1")

    def test_cli_builds_the_same_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._build_fixture(directory)
            output_dir = Path(directory) / "cli_output"
            result = assembly_cli_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

        self.assertEqual(set(result.split_paths), {"train", "validation", "test"})

    def test_source_with_too_few_windows_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._build_fixture(directory)
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw["assembly"]["windows_per_region"] = 5
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            output_dir = Path(directory) / "assembled"

            with self.assertRaisesRegex(ValueError, "has only 4 windows"):
                assemble_training_dataset(manifest_path, output_dir)

            self.assertFalse(output_dir.exists())

    def test_top_energy_selects_highest_rms_windows_in_time_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_sources: dict[str, list[dict[str, object]]] = {}
            expected_rms = np.asarray([1.0, 4.0, 3.0, 2.0])
            for split_name in ("train", "validation", "test"):
                split_sources[split_name] = []
                for label in LABEL_MAP:
                    source_id = f"{split_name}_class_{label}"
                    source_path = root / f"{source_id}.npz"
                    _write_source(
                        source_path,
                        source_id=source_id,
                        label=label,
                        region_count=1,
                        normalization_scale=expected_rms,
                    )
                    split_sources[split_name].append(
                        {
                            "path": source_path.name,
                            "label": label,
                            "source_id": source_id,
                        }
                    )

            manifest_path = root / "manifest.json"
            _write_manifest(
                manifest_path,
                split_sources,
                window_selection="top_energy",
            )
            result = assemble_training_dataset(manifest_path, root / "assembled")

            with np.load(result.split_paths["train"], allow_pickle=False) as data:
                for group_id in np.unique(data["group_id"]):
                    indices = np.flatnonzero(data["group_id"] == group_id)
                    np.testing.assert_array_equal(data["window_id"][indices], [1, 2])
                    np.testing.assert_allclose(
                        data["source_window_rms"][indices], [4.0, 3.0]
                    )
                self.assertEqual(data["window_selection"].item(), "top_energy")

            self.assertEqual(
                result.report["splits"]["train"]["selected_window_rms"],
                {"min": 3.0, "mean": 3.5, "max": 4.0},
            )


if __name__ == "__main__":
    unittest.main()
