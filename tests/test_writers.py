import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion import PreparedDataset
from signal_fusion.io import (
    load_prepared_dataset,
    write_dataset_summary,
    write_prepared_dataset,
)
from signal_fusion.preparation.detectors import (
    BlePacketDetectorV1,
    EnergyDetectorV1,
    FixedBlockDetector,
    FullSignalDetector,
    available_detectors,
    build_detector,
)


class PreparedDatasetWriterTests(unittest.TestCase):
    def test_writer_is_pickle_free_and_round_trips_core_contract(self):
        dataset = PreparedDataset(
            X=np.arange(32, dtype=np.float32).reshape(2, 2, 8),
            y=np.asarray([3, 3], dtype=np.int64),
            source_id="writer_fixture",
            meta={
                "seq_len": 8,
                "sample_rate": 1_000_000.0,
                "region_id": np.asarray([0, 0], dtype=np.int64),
                "window_id": np.asarray([0, 1], dtype=np.int64),
                "source_metadata": {"nested": "summary-only"},
                "center_frequency": None,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.npz"
            write_prepared_dataset(dataset, path)
            with np.load(path, allow_pickle=False) as saved:
                self.assertNotIn("source_metadata", saved.files)
                self.assertNotIn("center_frequency", saved.files)
                self.assertEqual(saved["source_id"].item(), "writer_fixture")
                self.assertTrue(all(not saved[key].dtype.hasobject for key in saved.files))
            loaded = load_prepared_dataset(path)

        self.assertEqual(loaded.source_id, "writer_fixture")
        np.testing.assert_array_equal(loaded.X, dataset.X)
        np.testing.assert_array_equal(loaded.y, dataset.y)
        np.testing.assert_array_equal(loaded.meta["region_id"], [0, 0])
        self.assertEqual(float(loaded.meta["sample_rate"]), 1_000_000.0)

    def test_summary_writer_serializes_nested_numpy_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared.npz"
            path = write_dataset_summary(
                {"shape": np.asarray([2, 2, 8]), "nested": {"count": np.int64(2)}},
                output,
            )
            actual = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(actual, {"shape": [2, 2, 8], "nested": {"count": 2}})


class DetectorRegistryTests(unittest.TestCase):
    def test_registry_builds_format_independent_strategies(self):
        self.assertEqual(
            available_detectors(),
            ("full_signal", "energy_v1", "ble_packet_v1", "fixed_blocks"),
        )
        self.assertIsInstance(
            build_detector("full_signal", {"start_sample": 2, "end_sample": 10}),
            FullSignalDetector,
        )
        self.assertIsInstance(
            build_detector("energy_v1", {"window_ms": 0.5}),
            EnergyDetectorV1,
        )
        self.assertIsInstance(
            build_detector("ble_packet_v1", {"channel": 37}),
            BlePacketDetectorV1,
        )
        self.assertIsInstance(
            build_detector(
                "fixed_blocks",
                {"block_size_samples": 4096, "block_count": 32},
            ),
            FixedBlockDetector,
        )

    def test_registry_rejects_unknown_detector(self):
        with self.assertRaisesRegex(ValueError, "Unknown detector"):
            build_detector("ble_future")


if __name__ == "__main__":
    unittest.main()
