import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io as sio

from signal_fusion.preparation import open_raw_signal
from signal_fusion.preparation.readers import ComplexDatReader, MatReader, SigMFReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SigMFReaderTests(unittest.TestCase):
    def test_reads_cf32_little_endian_with_metadata(self):
        iq = (
            np.arange(10, dtype=np.float32)
            + 1j * np.arange(20, 30, dtype=np.float32)
        ).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "capture.sigmf-data"
            meta_path = Path(directory) / "capture.sigmf-meta"
            iq.tofile(data_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "global": {
                            "core:datatype": "cf32_le",
                            "core:sample_rate": 1_000_000,
                            "core:description": "synthetic",
                        },
                        "captures": [{"core:frequency": 915_000_000}],
                    }
                ),
                encoding="utf-8",
            )
            signal = SigMFReader().open(data_path, source_id="synthetic_sigmf")
            actual = signal.read_samples(3, 4)

        np.testing.assert_array_equal(actual, iq[3:7])
        self.assertEqual(signal.sample_count, 10)
        self.assertEqual(signal.sample_rate, 1_000_000)
        self.assertEqual(signal.center_frequency, 915_000_000)
        self.assertEqual(signal.metadata["description"], "synthetic")

    def test_scales_ci16_without_adaptive_normalization(self):
        raw_dtype = np.dtype([("i", "<i2"), ("q", "<i2")])
        raw = np.zeros(2, dtype=raw_dtype)
        raw["i"] = [16384, -32768]
        raw["q"] = [8192, 0]
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "capture.sigmf-data"
            meta_path = Path(directory) / "capture.sigmf-meta"
            raw.tofile(data_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "global": {
                            "core:datatype": "ci16_le",
                            "core:sample_rate": 2_000_000,
                        }
                    }
                ),
                encoding="utf-8",
            )
            signal = SigMFReader().open(data_path, source_id="integer_sigmf")
            actual = signal.read_samples()

        np.testing.assert_allclose(actual, [0.5 + 0.25j, -1.0 + 0j])


class MatReaderTests(unittest.TestCase):
    def test_reads_complex_column_and_infers_capture_metadata(self):
        iq = (
            np.arange(8, dtype=np.float32)
            + 1j * np.arange(10, 18, dtype=np.float32)
        ).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.mat"
            sio.savemat(
                path,
                {
                    "iq": iq.reshape(-1, 1),
                    "Sample_rate": np.array([[100_000_000]]),
                    "CenterFrequency": np.array([[5_230_000_000.0]]),
                },
            )
            signal = MatReader().open(path, source_id="synthetic_mat")
            actual = signal.read_samples(2, 3)

        np.testing.assert_array_equal(actual, iq[2:5])
        self.assertEqual(signal.sample_count, 8)
        self.assertEqual(signal.sample_rate, 100_000_000)
        self.assertEqual(signal.center_frequency, 5_230_000_000)


class DatReaderTests(unittest.TestCase):
    def test_reads_only_requested_complex64_interval(self):
        iq = (
            np.arange(9, dtype=np.float32)
            + 1j * np.arange(30, 39, dtype=np.float32)
        ).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.dat"
            iq.tofile(path)
            signal = ComplexDatReader().open(
                path,
                source_id="synthetic_dat",
                sample_rate=4_000_000,
            )
            actual = signal.read_samples(4, 3)

        np.testing.assert_array_equal(actual, iq[4:7])
        self.assertEqual(signal.sample_count, 9)
        self.assertEqual(signal.metadata["bytes_per_sample"], 8)


class RealFixtureReaderTests(unittest.TestCase):
    def test_opens_phase0_sigmf_mat_and_ble_dat_sources(self):
        sigmf = open_raw_signal(
            PROJECT_ROOT / "data/raw/lora/sigmf_lora.sigmf-data",
            source_id="lora_raw_sigmf",
        )
        self.assertEqual(sigmf.sample_count, 19_965_036)
        self.assertEqual(sigmf.sample_rate, 1_000_000)
        self.assertTrue(np.isfinite(sigmf.read_samples(0, 32)).all())

        mat = open_raw_signal(
            PROJECT_ROOT
            / "data/raw/wifi/WIFI_5_batch;Freq=5230 MHz;Span=80 MHz;Rate=100.0 MHz;0005.mat",
            source_id="wifi5_raw_mat_0005",
        )
        self.assertEqual(mat.sample_count, 1_000_000)
        self.assertEqual(mat.sample_rate, 100_000_000)
        self.assertTrue(np.isfinite(mat.read_samples(0, 32)).all())

        ble = open_raw_signal(
            PROJECT_ROOT / "data/raw/BLE/BLE_1.7GHz_r09_rx_iq.dat",
            source_id="ble_raw_dat_r09_1p7ghz",
            sample_rate=4_000_000,
        )
        self.assertEqual(ble.sample_count, 11_600_000)
        self.assertTrue(np.isfinite(ble.read_samples(0, 4096)).all())


if __name__ == "__main__":
    unittest.main()

