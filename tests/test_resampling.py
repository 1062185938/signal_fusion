import math
from dataclasses import replace
import unittest

import numpy as np

from signal_fusion.preparation import (
    RESAMPLING_PROFILE_V1,
    RawSignal,
    ResamplingConfig,
    ResamplingPlan,
    SignalRegion,
    build_resampling_plan,
    resample_iq,
    resample_region,
)


class RecordingReader:
    def __init__(self, samples: np.ndarray):
        self.samples = np.asarray(samples, dtype=np.complex64)
        self.calls: list[tuple[int, int]] = []

    def __call__(self, start: int, count: int) -> np.ndarray:
        self.calls.append((start, count))
        return self.samples[start : start + count]


def _signal(samples: np.ndarray, sample_rate: float) -> tuple[RawSignal, RecordingReader]:
    reader = RecordingReader(samples)
    signal = RawSignal(
        source_id="resampling_test",
        source_path="synthetic.dat",
        sample_rate=sample_rate,
        sample_count=reader.samples.size,
        sample_format="complex64",
        _sample_reader=reader,
    )
    return signal, reader


class ResamplingConfigTests(unittest.TestCase):
    def test_defaults_define_the_v2_target(self):
        config = ResamplingConfig()
        self.assertEqual(config.target_sample_rate, 4_000_000.0)
        self.assertEqual(config.profile, RESAMPLING_PROFILE_V1)

    def test_rejects_invalid_values(self):
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(target_sample_rate=value):
                with self.assertRaises(ValueError):
                    ResamplingConfig(target_sample_rate=value)
        with self.assertRaises(ValueError):
            ResamplingConfig(profile="unversioned")
        with self.assertRaises(ValueError):
            ResamplingConfig(max_denominator=0)
        with self.assertRaises(ValueError):
            ResamplingConfig(rate_tolerance_ppm=-1)
        with self.assertRaises(ValueError):
            ResamplingConfig(max_output_samples=0)
        with self.assertRaises(TypeError):
            ResamplingConfig(max_denominator=1.5)
        with self.assertRaises(TypeError):
            ResamplingConfig(max_output_samples=True)
        with self.assertRaises(ValueError):
            ResamplingConfig(max_input_samples=0)
        with self.assertRaises(ValueError):
            ResamplingConfig(max_resampling_factor=4_097)


class ResamplingPlanTests(unittest.TestCase):
    def test_resolves_current_dataset_rates(self):
        upsample = build_resampling_plan(1_000_000, ResamplingConfig())
        identity = build_resampling_plan(4_000_000, ResamplingConfig())

        self.assertEqual((upsample.up, upsample.down), (4, 1))
        self.assertEqual(upsample.output_length(128), 512)
        self.assertFalse(upsample.is_identity)
        self.assertEqual((identity.up, identity.down), (1, 1))
        self.assertTrue(identity.is_identity)

    def test_resolves_common_fraction_and_integer_coordinates(self):
        plan = build_resampling_plan(
            44_100,
            ResamplingConfig(target_sample_rate=48_000),
        )
        self.assertEqual((plan.up, plan.down), (160, 147))
        self.assertEqual(plan.source_boundary_to_target(147), 160)
        self.assertEqual(plan.target_interval_to_source(160, 320), (147, 294))
        self.assertEqual(plan.target_interval_to_source(161, 161), (147, 147))

        large_boundary = 2**60 + 1
        self.assertEqual(
            plan.source_boundary_to_target(large_boundary),
            (large_boundary * 160 + 146) // 147,
        )

    def test_metadata_is_explicit_for_identity_and_polyphase(self):
        identity = build_resampling_plan(4_000_000, ResamplingConfig())
        converted = build_resampling_plan(1_000_000, ResamplingConfig())

        self.assertEqual(identity.to_metadata()["resampling_method"], "identity")
        self.assertFalse(identity.to_metadata()["resampling_applied"])
        self.assertEqual(converted.to_metadata()["resample_up"], 4)
        self.assertEqual(converted.to_metadata()["resampling_method"], "polyphase")

    def test_public_plan_rejects_unsafe_filter_factor(self):
        with self.assertRaisesRegex(ValueError, "safety limit"):
            ResamplingPlan(
                source_sample_rate=1,
                target_sample_rate=4_097,
                effective_sample_rate=4_097,
                up=4_097,
                down=1,
                rate_error_ppm=0,
            )

    def test_coordinates_reject_fractional_and_boolean_values(self):
        plan = build_resampling_plan(1_000_000, ResamplingConfig())
        with self.assertRaises(TypeError):
            plan.source_boundary_to_target(1.5)
        with self.assertRaises(TypeError):
            plan.output_length(True)


class IQResamplingTests(unittest.TestCase):
    def test_identity_is_bitwise_and_complex64(self):
        samples = np.array([1 + 2j, -3 + 4j, 5 - 6j], dtype=np.complex64)
        plan = build_resampling_plan(4_000_000, ResamplingConfig())
        actual = resample_iq(samples, plan=plan)

        np.testing.assert_array_equal(actual, samples)
        self.assertEqual(actual.dtype, np.complex64)
        self.assertIsNot(actual, samples)

    def test_rejects_non_iq_or_wrong_shape(self):
        plan = build_resampling_plan(4_000_000, ResamplingConfig())
        with self.assertRaises(TypeError):
            resample_iq(np.ones(8, dtype=np.float32), plan=plan)
        with self.assertRaises(ValueError):
            resample_iq(np.ones((2, 8), dtype=np.complex64), plan=plan)

    def test_empty_iq_preserves_contract(self):
        plan = build_resampling_plan(1_000_000, ResamplingConfig())
        actual = resample_iq(np.empty(0, dtype=np.complex64), plan=plan)
        self.assertEqual(actual.shape, (0,))
        self.assertEqual(actual.dtype, np.complex64)

    def test_polyphase_direct_output_is_complex64(self):
        plan = build_resampling_plan(1_000_000, ResamplingConfig())
        samples = np.exp(1j * np.arange(32)).astype(np.complex128)
        actual = resample_iq(samples, plan=plan)
        self.assertEqual(actual.shape, (128,))
        self.assertEqual(actual.dtype, np.complex64)

    def test_output_limit_is_inclusive(self):
        plan = build_resampling_plan(
            1_000_000,
            ResamplingConfig(max_output_samples=16),
        )
        actual = resample_iq(np.ones(4, dtype=np.complex64), plan=plan)
        self.assertEqual(actual.shape, (16,))

    def test_enforces_output_limit_before_resampling(self):
        plan = build_resampling_plan(
            1_000_000,
            ResamplingConfig(max_output_samples=15),
        )
        with self.assertRaisesRegex(ValueError, "max_output_samples"):
            resample_iq(np.ones(4, dtype=np.complex64), plan=plan)

    def test_enforces_input_limit_before_resampling(self):
        plan = build_resampling_plan(
            4_000_000,
            ResamplingConfig(max_input_samples=3),
        )
        with self.assertRaisesRegex(ValueError, "max_input_samples"):
            resample_iq(np.ones(4, dtype=np.complex64), plan=plan)

    def test_one_to_four_preserves_complex_tone(self):
        source_rate = 1_000_000
        target_rate = 4_000_000
        tone_frequency = 125_000
        indices = np.arange(4_096)
        samples = np.exp(
            2j * np.pi * tone_frequency * indices / source_rate
        ).astype(np.complex64)
        signal, _ = _signal(samples, source_rate)
        plan = build_resampling_plan(source_rate, ResamplingConfig())

        actual = resample_region(
            signal,
            SignalRegion(512, 1_536, "test"),
            plan=plan,
        )

        target_indices = np.arange(512 * 4, 1_536 * 4)
        expected = np.exp(
            2j * np.pi * tone_frequency * target_indices / target_rate
        )
        self.assertEqual(actual.samples.shape, (4_096,))
        self.assertEqual(actual.samples.dtype, np.complex64)
        np.testing.assert_allclose(actual.samples, expected, atol=1.5e-3, rtol=0)


class RegionResamplingTests(unittest.TestCase):
    def test_identity_reads_only_the_requested_region(self):
        samples = (
            np.arange(64, dtype=np.float32)
            + 1j * np.arange(64, dtype=np.float32)[::-1]
        ).astype(np.complex64)
        signal, reader = _signal(samples, 4_000_000)
        plan = build_resampling_plan(4_000_000, ResamplingConfig())

        actual = resample_region(
            signal,
            SignalRegion(7, 29, "test"),
            plan=plan,
        )

        np.testing.assert_array_equal(actual.samples, samples[7:29])
        self.assertEqual(reader.calls, [(7, 22)])
        self.assertEqual(
            (actual.target_region_start_sample, actual.target_region_end_sample),
            (7, 29),
        )

    def test_guarded_region_matches_whole_recording_resample(self):
        rng = np.random.default_rng(42)
        samples = (
            rng.normal(size=2_048) + 1j * rng.normal(size=2_048)
        ).astype(np.complex64)
        signal, reader = _signal(samples, 1_000_000)
        plan = build_resampling_plan(1_000_000, ResamplingConfig())
        region = SignalRegion(503, 1_307, "test")

        actual = resample_region(signal, region, plan=plan)
        whole = resample_iq(samples, plan=plan)
        start = plan.source_boundary_to_target(region.start_sample)
        end = plan.source_boundary_to_target(region.end_sample)

        np.testing.assert_array_equal(actual.samples, whole[start:end])
        self.assertEqual(len(reader.calls), 1)
        read_start, read_count = reader.calls[0]
        self.assertEqual((read_start, read_count), (493, 824))
        self.assertEqual(actual.source_read_start_sample, 493)
        self.assertEqual(actual.source_read_end_sample, 1_317)

    def test_downsampling_uses_global_grid_and_aligned_read_start(self):
        rng = np.random.default_rng(7)
        samples = (
            rng.normal(size=1_024) + 1j * rng.normal(size=1_024)
        ).astype(np.complex64)
        signal, reader = _signal(samples, 6_000_000)
        plan = build_resampling_plan(
            6_000_000,
            ResamplingConfig(target_sample_rate=4_000_000),
        )
        region = SignalRegion(101, 161, "test")

        actual = resample_region(signal, region, plan=plan)
        whole = resample_iq(samples, plan=plan)

        self.assertEqual((plan.up, plan.down), (2, 3))
        self.assertEqual(
            (actual.target_region_start_sample, actual.target_region_end_sample),
            (math.ceil(101 * 2 / 3), math.ceil(161 * 2 / 3)),
        )
        self.assertEqual(actual.sample_count, 40)
        self.assertEqual(reader.calls, [(84, 92)])
        self.assertEqual(actual.source_read_start_sample % plan.down, 0)
        np.testing.assert_array_equal(actual.samples, whole[68:108])

    def test_guard_buffer_limit_fails_before_reading(self):
        samples = np.ones(128, dtype=np.complex64)
        signal, reader = _signal(samples, 1_000_000)
        plan = build_resampling_plan(
            1_000_000,
            ResamplingConfig(max_output_samples=15),
        )

        with self.assertRaisesRegex(ValueError, "guarded resampling buffer"):
            resample_region(
                signal,
                SignalRegion(50, 51, "test"),
                plan=plan,
            )
        self.assertEqual(reader.calls, [])

    def test_native_buffer_limit_fails_before_reading(self):
        samples = np.ones(128, dtype=np.complex64)
        signal, reader = _signal(samples, 4_000_000)
        plan = build_resampling_plan(
            4_000_000,
            ResamplingConfig(
                target_sample_rate=1_000_000,
                max_input_samples=20,
                max_output_samples=100,
            ),
        )

        with self.assertRaisesRegex(ValueError, "guarded native buffer"):
            resample_region(
                signal,
                SignalRegion(50, 51, "test"),
                plan=plan,
            )
        self.assertEqual(reader.calls, [])

    def test_short_downsampled_region_may_have_empty_output(self):
        samples = np.ones(100, dtype=np.complex64)
        signal, _ = _signal(samples, 4_000_000)
        plan = build_resampling_plan(
            4_000_000,
            ResamplingConfig(target_sample_rate=1_000_000),
        )

        actual = resample_region(
            signal,
            SignalRegion(1, 2, "test"),
            plan=plan,
        )
        self.assertEqual(actual.samples.shape, (0,))
        self.assertEqual(
            (actual.target_region_start_sample, actual.target_region_end_sample),
            (1, 1),
        )

    def test_resampled_region_rejects_inconsistent_target_coordinates(self):
        samples = np.ones(128, dtype=np.complex64)
        signal, _ = _signal(samples, 1_000_000)
        plan = build_resampling_plan(1_000_000, ResamplingConfig())
        actual = resample_region(
            signal,
            SignalRegion(50, 60, "test"),
            plan=plan,
        )

        with self.assertRaisesRegex(ValueError, "target region coordinates"):
            replace(
                actual,
                target_region_start_sample=actual.target_region_start_sample + 1,
                target_region_end_sample=actual.target_region_end_sample + 1,
            )

    def test_file_boundaries_match_whole_recording_behavior(self):
        rng = np.random.default_rng(99)
        samples = (
            rng.normal(size=100) + 1j * rng.normal(size=100)
        ).astype(np.complex64)
        plan = build_resampling_plan(1_000_000, ResamplingConfig())
        whole = resample_iq(samples, plan=plan)

        for region in (
            SignalRegion(0, 33, "test"),
            SignalRegion(69, 100, "test"),
        ):
            with self.subTest(region=(region.start_sample, region.end_sample)):
                signal, reader = _signal(samples, 1_000_000)
                actual = resample_region(signal, region, plan=plan)
                start = plan.source_boundary_to_target(region.start_sample)
                end = plan.source_boundary_to_target(region.end_sample)
                np.testing.assert_array_equal(actual.samples, whole[start:end])
                read_start, read_count = reader.calls[0]
                self.assertGreaterEqual(read_start, 0)
                self.assertLessEqual(read_start + read_count, samples.size)

    def test_rejects_out_of_bounds_region_and_rate_mismatch(self):
        signal, _ = _signal(np.ones(32, dtype=np.complex64), 1_000_000)
        plan = build_resampling_plan(1_000_000, ResamplingConfig())
        with self.assertRaisesRegex(ValueError, "recording sample count"):
            resample_region(
                signal,
                SignalRegion(16, 33, "test"),
                plan=plan,
            )

        wrong_plan = build_resampling_plan(2_000_000, ResamplingConfig())
        with self.assertRaisesRegex(ValueError, "does not match"):
            resample_region(
                signal,
                SignalRegion(0, 16, "test"),
                plan=wrong_plan,
            )


if __name__ == "__main__":
    unittest.main()
