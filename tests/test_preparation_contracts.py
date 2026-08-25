import unittest

import numpy as np

from signal_fusion.preparation import PreparationConfig, RawSignal, SignalRegion


class RawSignalTests(unittest.TestCase):
    def test_lazy_reader_enforces_bounds_and_complex64_output(self):
        source = np.arange(12, dtype=np.float32).astype(np.complex64)
        signal = RawSignal(
            source_id="synthetic_raw",
            source_path="synthetic.dat",
            sample_rate=4_000_000,
            sample_count=source.size,
            sample_format="complex64",
            _sample_reader=lambda start, count: source[start : start + count],
        )

        self.assertEqual(signal.duration_seconds, 3e-6)
        np.testing.assert_array_equal(signal.read_samples(3, 4), source[3:7])
        self.assertEqual(signal.read_samples(10, 100).shape, (2,))
        self.assertEqual(signal.read_samples(12).dtype, np.complex64)
        with self.assertRaises(ValueError):
            signal.read_samples(-1, 2)


class SignalRegionTests(unittest.TestCase):
    def test_half_open_region_reports_sample_count(self):
        region = SignalRegion(10, 25, detector="test", score=0.8)
        self.assertEqual(region.sample_count, 15)

    def test_rejects_invalid_region(self):
        with self.assertRaises(ValueError):
            SignalRegion(4, 4, detector="test")


class PreparationConfigTests(unittest.TestCase):
    def test_defaults_hop_to_sequence_length(self):
        config = PreparationConfig(seq_len=128)
        self.assertEqual(config.hop_len, 128)

    def test_rejects_unknown_policy(self):
        with self.assertRaises(ValueError):
            PreparationConfig(seq_len=128, remainder="keep")
        with self.assertRaises(ValueError):
            PreparationConfig(seq_len=128, normalization="zscore")


if __name__ == "__main__":
    unittest.main()

