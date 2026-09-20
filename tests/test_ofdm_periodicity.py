import json
import unittest

import numpy as np

from signal_fusion.feature_extraction import (
    PERIODICITY_SCHEMA_ID,
    PeriodicityCandidate,
    measure_periodicity,
    measure_periodicity_batch,
)


class OfdmPeriodicityTests(unittest.TestCase):
    def test_finds_a_caller_provided_repetition_period(self):
        rng = np.random.default_rng(44)
        block = (
            rng.standard_normal(64) + 1j * rng.standard_normal(64)
        ).astype(np.complex64)
        signal = np.tile(block, 32)
        candidate = PeriodicityCandidate("period_64us", 64e-6)

        result = measure_periodicity(signal, 1_000_000.0, [candidate])

        self.assertEqual(result.peak_lags[0, 0], 64)
        self.assertGreater(result.normalized_correlations[0, 0], 0.999)
        self.assertTrue(result.resolvable[0])

    def test_normalized_measurement_is_scale_and_phase_invariant(self):
        rng = np.random.default_rng(45)
        block = (
            rng.standard_normal(67) + 1j * rng.standard_normal(67)
        ).astype(np.complex64)
        signal = np.tile(block, 32)[:2048]
        candidate = PeriodicityCandidate("period_66p67us", 1.0 / 15_000.0)

        original = measure_periodicity(signal, 1_000_000.0, [candidate])
        transformed = measure_periodicity(
            signal * np.complex64(3.5 * np.exp(1j * 0.73)),
            1_000_000.0,
            [candidate],
        )

        np.testing.assert_allclose(
            original.normalized_correlations,
            transformed.normalized_correlations,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(original.peak_lags, transformed.peak_lags)

    def test_marks_a_period_unresolved_when_search_window_does_not_fit(self):
        signal = np.ones(128, dtype=np.complex64)
        candidate = PeriodicityCandidate("period_896us", 896e-6)

        result = measure_periodicity(signal, 1_000_000.0, [candidate])
        evidence = result.evidence_for_sample(0)

        self.assertFalse(result.resolvable[0])
        self.assertEqual(result.normalized_correlations[0, 0], 0.0)
        self.assertIsNone(evidence["candidates"][0]["peak_lag_samples"])
        self.assertIsNone(evidence["candidates"][0]["normalized_correlation"])
        self.assertEqual(evidence["schema_id"], PERIODICITY_SCHEMA_ID)
        json.dumps(evidence, allow_nan=False)

    def test_batch_output_uses_one_row_per_region(self):
        time = np.arange(1024, dtype=np.float32)
        signals = np.stack(
            (
                np.exp(1j * 2.0 * np.pi * time / 67.0),
                np.exp(1j * 2.0 * np.pi * time / 224.0),
            )
        ).astype(np.complex64)
        candidates = (
            PeriodicityCandidate("period_66p67us", 1.0 / 15_000.0),
            PeriodicityCandidate("period_224us", 224e-6),
        )

        result = measure_periodicity_batch(
            signals, 1_000_000.0, candidates, batch_size=1
        )

        self.assertEqual(result.normalized_correlations.shape, (2, 2))
        self.assertEqual(result.peak_lags.shape, (2, 2))
        self.assertEqual(result.normalized_correlations.dtype, np.float32)
        self.assertEqual(result.peak_lags.dtype, np.int64)

    def test_rejects_invalid_inputs_and_candidates(self):
        candidate = PeriodicityCandidate("period", 64e-6)
        with self.assertRaisesRegex(TypeError, "complex"):
            measure_periodicity_batch(
                np.ones((1, 128), dtype=np.float32),
                1_000_000.0,
                [candidate],
            )
        invalid = np.ones((1, 128), dtype=np.complex64)
        invalid[0, 3] = np.nan + 1j
        with self.assertRaisesRegex(ValueError, "NaN"):
            measure_periodicity_batch(invalid, 1_000_000.0, [candidate])
        with self.assertRaisesRegex(ValueError, "positive"):
            measure_periodicity_batch(
                np.ones((1, 128), dtype=np.complex64),
                0.0,
                [candidate],
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            measure_periodicity_batch(
                np.ones((1, 128), dtype=np.complex64),
                1_000_000.0,
                [candidate, candidate],
            )


if __name__ == "__main__":
    unittest.main()
