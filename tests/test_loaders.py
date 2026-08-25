import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io as sio

from signal_fusion import PreparedDataset
from signal_fusion.io import (
    load_prepared_dataset,
    load_signal_dataset,
    load_signal_for_inference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import compatibility wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoaderTests(unittest.TestCase):
    def test_npz_loads_to_prepared_dataset_and_preserves_sample_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            source_x = np.arange(3 * 8 * 2, dtype=np.float64).reshape(3, 8, 2)
            np.savez(
                path,
                X=source_x,
                y=np.array([[0], [1], [1]], dtype=np.int32),
                burst_id=np.array([4, 4, 5], dtype=np.int64),
            )

            dataset = load_prepared_dataset(
                path,
                data_format="npz",
                seq_len=8,
                source_id="synthetic_npz",
            )

        self.assertIsInstance(dataset, PreparedDataset)
        self.assertEqual(dataset.X.shape, (3, 2, 8))
        self.assertEqual(dataset.X.dtype, np.float32)
        self.assertEqual(dataset.y.dtype, np.int64)
        np.testing.assert_array_equal(dataset.meta["burst_id"], [4, 4, 5])
        self.assertEqual(dataset.source_id, "synthetic_npz")

    def test_mat_complex_column_uses_feature_loader_superset_behavior(self):
        iq = (
            np.arange(8, dtype=np.float32)
            + 1j * np.arange(10, 18, dtype=np.float32)
        ).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "column_iq.mat"
            sio.savemat(path, {"iq": iq.reshape(-1, 1)})
            dataset = load_prepared_dataset(
                path, data_format="mat", x_key="iq", seq_len=4
            )

        self.assertEqual(dataset.X.shape, (2, 2, 4))
        np.testing.assert_array_equal(dataset.X[0, 0], iq.real[:4])
        np.testing.assert_array_equal(dataset.X[0, 1], iq.imag[:4])

    def test_dat_loader_is_record_aligned_and_reports_dropped_points(self):
        iq = (
            np.arange(10, dtype=np.float32)
            + 1j * np.arange(20, 30, dtype=np.float32)
        ).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.dat"
            iq.tofile(path)
            loaded = load_signal_dataset(path, data_format="dat", seq_len=4)

        self.assertEqual(loaded["X"].shape, (2, 2, 4))
        self.assertIsNone(loaded["y"])
        self.assertEqual(loaded["meta"]["used_iq_points"], 8)
        self.assertEqual(loaded["meta"]["dropped_iq_points"], 2)

    def test_inference_loader_limits_samples_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            np.savez(
                path,
                X=np.zeros((4, 2, 16), dtype=np.float32),
                y=np.arange(4, dtype=np.int64),
            )
            loaded = load_signal_for_inference(
                path,
                data_format="npz",
                seq_len=16,
                max_samples=2,
            )

        self.assertEqual(set(loaded), {"X", "meta"})
        self.assertEqual(loaded["X"].shape, (2, 2, 16))
        self.assertEqual(loaded["meta"]["returned_samples"], 2)

    def test_phase0_lora_fixture_keeps_schema(self):
        path = PROJECT_ROOT / "data/processed/slices/sigmf_lora_dataset_128.npz"
        dataset = load_prepared_dataset(
            path,
            data_format="npz",
            seq_len=128,
            source_id="lora_slices_128_example",
        )

        self.assertEqual(dataset.X.shape, (4804, 2, 128))
        self.assertEqual(dataset.X.dtype, np.float32)
        self.assertEqual(dataset.y.shape, (4804,))
        self.assertEqual(dataset.meta["x_shape"], (4804, 2, 128))
        self.assertEqual(dataset.meta["seq_len"], 128)

    def test_legacy_loader_paths_reexport_shared_implementation(self):
        feature_wrapper = _load_module(
            "legacy_feature_data_loaders",
            PROJECT_ROOT / "iq_feature_extraction/scripts/data_loaders.py",
        )
        model_wrapper = _load_module(
            "legacy_model_data_loaders",
            PROJECT_ROOT / "radioml-iq-modulation/scripts/data_loaders.py",
        )

        self.assertIs(feature_wrapper.load_signal_dataset, load_signal_dataset)
        self.assertIs(model_wrapper.load_signal_dataset, load_signal_dataset)
        self.assertIs(
            model_wrapper.load_signal_for_inference, load_signal_for_inference
        )

    def test_direct_sigmf_loading_remains_outside_main_analysis_loader(self):
        with self.assertRaises(NotImplementedError):
            load_prepared_dataset("capture.sigmf-data", data_format="sigmf")


if __name__ == "__main__":
    unittest.main()

