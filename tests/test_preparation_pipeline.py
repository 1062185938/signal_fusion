import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from signal_fusion import PreparedDataset
from signal_fusion.preparation import (
    FixedBlockDetector,
    FullSignalDetector,
    PreparationConfig,
    RawSignal,
    ResamplingConfig,
    SignalDetector,
    SignalRegion,
    build_prepared_dataset,
    normalize_iq,
    prepare_file,
    prepare_signal,
    segment_regions,
    window_spans,
)


def _raw_signal(samples: np.ndarray) -> RawSignal:
    iq = np.asarray(samples, dtype=np.complex64).reshape(-1)
    return RawSignal(
        source_id="synthetic_pipeline",
        source_path="synthetic.dat",
        sample_rate=1_000_000,
        sample_count=iq.size,
        sample_format="complex64",
        _sample_reader=lambda start, count: iq[start : start + count],
    )


class FixedDetector(SignalDetector):
    name = "fixed_test"

    def __init__(self, regions):
        self._regions = regions

    def detect(self, signal: RawSignal) -> list[SignalRegion]:
        return list(self._regions)


class RecordingDetector(SignalDetector):
    name = "recording_test"

    def __init__(self, regions):
        self._regions = regions
        self.observed_sample_rate = None
        self.observed_sample_count = None

    def detect(self, signal: RawSignal) -> list[SignalRegion]:
        self.observed_sample_rate = signal.sample_rate
        self.observed_sample_count = signal.sample_count
        return list(self._regions)


class SegmentationTests(unittest.TestCase):
    def test_filters_pads_and_merges_regions(self):
        regions = [
            SignalRegion(10, 20, "fixed", score=0.4),
            SignalRegion(22, 30, "fixed", score=0.8),
            SignalRegion(40, 42, "fixed", score=0.2),
        ]
        config = PreparationConfig(
            seq_len=8,
            min_region_samples=4,
            merge_gap_samples=0,
            pad_before_samples=1,
            pad_after_samples=1,
        )

        actual = segment_regions(
            regions, recording_sample_count=100, config=config
        )

        self.assertEqual(len(actual), 1)
        self.assertEqual((actual[0].start_sample, actual[0].end_sample), (9, 31))
        self.assertEqual(actual[0].score, 0.8)
        self.assertEqual(actual[0].metadata["merged_region_count"], 2)


class WindowingTests(unittest.TestCase):
    def test_drop_and_pad_have_explicit_tail_behavior(self):
        region = SignalRegion(10, 20, "fixed")
        dropped = window_spans(
            region, PreparationConfig(seq_len=4, hop_len=4, remainder="drop")
        )
        padded = window_spans(
            region, PreparationConfig(seq_len=4, hop_len=4, remainder="pad")
        )

        self.assertEqual(
            [(span.start_sample, span.end_sample) for span in dropped],
            [(10, 14), (14, 18)],
        )
        self.assertEqual(
            [span.valid_samples for span in padded],
            [4, 4, 2],
        )


class NormalizationTests(unittest.TestCase):
    def test_remove_dc_and_rms_are_reported(self):
        iq = np.array([2 + 1j, 4 + 1j], dtype=np.complex64)
        normalized = normalize_iq(iq, mode="rms", remove_dc=True)

        self.assertAlmostEqual(normalized.dc_offset.real, 3.0)
        self.assertAlmostEqual(normalized.dc_offset.imag, 1.0)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.abs(normalized.samples) ** 2))), 1.0
        )


class PreparationPipelineTests(unittest.TestCase):
    def test_fixed_blocks_keep_all_non_overlapping_windows(self):
        iq = np.arange(64, dtype=np.float32).astype(np.complex64)
        dataset = prepare_signal(
            _raw_signal(iq),
            detector=FixedBlockDetector(block_size_samples=16, block_count=4),
            config=PreparationConfig(
                seq_len=4,
                hop_len=4,
                normalization="none",
                remove_dc=False,
            ),
        )

        self.assertEqual(dataset.X.shape, (16, 2, 4))
        np.testing.assert_array_equal(
            np.unique(dataset.meta["region_id"], return_counts=True)[1],
            [4, 4, 4, 4],
        )
        np.testing.assert_array_equal(
            dataset.meta["region_start_sample"],
            np.repeat([0, 16, 32, 48], 4),
        )

    def test_fixed_blocks_select_regions_uniformly(self):
        detector = FixedBlockDetector(block_size_samples=10, block_count=3)
        regions = detector.detect(_raw_signal(np.arange(100, dtype=np.float32)))

        self.assertEqual(
            [(region.start_sample, region.end_sample) for region in regions],
            [(10, 20), (50, 60), (80, 90)],
        )

    def test_pipeline_returns_prepared_dataset_with_absolute_coordinates(self):
        iq = (
            np.arange(10, dtype=np.float32)
            + 1j * np.arange(20, 30, dtype=np.float32)
        ).astype(np.complex64)
        config = PreparationConfig(
            seq_len=4,
            hop_len=4,
            remainder="pad",
            label=7,
            class_name="synthetic",
        )

        dataset = prepare_signal(
            _raw_signal(iq),
            detector=FullSignalDetector(),
            config=config,
        )

        self.assertIsInstance(dataset, PreparedDataset)
        self.assertEqual(dataset.X.shape, (3, 2, 4))
        np.testing.assert_array_equal(dataset.y, [7, 7, 7])
        np.testing.assert_array_equal(
            dataset.meta["window_start_sample"], [0, 4, 8]
        )
        np.testing.assert_array_equal(dataset.meta["valid_sample_count"], [4, 4, 2])
        np.testing.assert_array_equal(dataset.X[-1, 0], [8, 9, 0, 0])
        np.testing.assert_array_equal(dataset.X[-1, 1], [28, 29, 0, 0])
        self.assertNotIn("resampling_enabled", dataset.meta)
        self.assertNotIn("source_sample_rate", dataset.meta)

    def test_optional_resampling_uses_native_detection_and_target_windowing(self):
        indices = np.arange(16, dtype=np.float32)
        iq = (2.0 + np.exp(1j * indices * 0.4)).astype(np.complex64)
        detector = RecordingDetector([SignalRegion(2, 6, "recording_test")])

        dataset = prepare_signal(
            _raw_signal(iq),
            detector=detector,
            config=PreparationConfig(
                seq_len=4,
                hop_len=4,
                normalization="rms",
                remove_dc=True,
                label=2,
                class_name="synthetic",
            ),
            resampling=ResamplingConfig(target_sample_rate=4_000_000),
        )

        self.assertEqual(detector.observed_sample_rate, 1_000_000)
        self.assertEqual(detector.observed_sample_count, 16)
        self.assertEqual(dataset.X.shape, (4, 2, 4))
        self.assertEqual(float(dataset.meta["sample_rate"]), 4_000_000)
        self.assertEqual(float(dataset.meta["source_sample_rate"]), 1_000_000)
        self.assertEqual(dataset.meta["coordinate_schema"], "dual_rate_v1")
        self.assertEqual(dataset.meta["detector_coordinate_system"], "source")
        self.assertEqual(dataset.meta["segmentation_coordinate_system"], "source")
        self.assertEqual(dataset.meta["recording_sample_count"], 64)
        self.assertEqual(dataset.meta["source_recording_sample_count"], 16)
        np.testing.assert_array_equal(
            dataset.meta["window_start_sample"], [8, 12, 16, 20]
        )
        np.testing.assert_array_equal(
            dataset.meta["target_window_start_sample"], [8, 12, 16, 20]
        )
        np.testing.assert_array_equal(
            dataset.meta["source_window_start_sample"], [2, 3, 4, 5]
        )
        np.testing.assert_array_equal(
            dataset.meta["source_window_end_sample"], [3, 4, 5, 6]
        )
        np.testing.assert_array_equal(
            dataset.meta["region_start_sample"], [8, 8, 8, 8]
        )
        np.testing.assert_array_equal(
            dataset.meta["source_region_start_sample"], [2, 2, 2, 2]
        )

        complex_windows = dataset.X[:, 0, :] + 1j * dataset.X[:, 1, :]
        np.testing.assert_allclose(
            np.mean(complex_windows, axis=1), 0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            np.sqrt(np.mean(np.abs(complex_windows) ** 2, axis=1)),
            1.0,
            atol=1e-6,
        )

    def test_identity_resampling_preserves_legacy_values(self):
        iq = (
            np.arange(12, dtype=np.float32)
            + 1j * np.arange(20, 32, dtype=np.float32)
        ).astype(np.complex64)
        signal = _raw_signal(iq)
        config = PreparationConfig(seq_len=4, hop_len=4, remainder="pad")

        legacy = prepare_signal(
            signal,
            detector=FullSignalDetector(),
            config=config,
        )
        identity = prepare_signal(
            signal,
            detector=FullSignalDetector(),
            config=config,
            resampling=ResamplingConfig(target_sample_rate=1_000_000),
        )

        np.testing.assert_array_equal(identity.X, legacy.X)
        np.testing.assert_array_equal(
            identity.meta["window_start_sample"],
            legacy.meta["window_start_sample"],
        )
        self.assertEqual(identity.meta["resampling_method"], "identity")
        self.assertFalse(identity.meta["resampling_applied"])

    def test_prepare_file_uses_reader_registry(self):
        iq = np.arange(8, dtype=np.float32).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.dat"
            iq.tofile(path)
            dataset = prepare_file(
                path,
                source_id="registry_dat",
                data_format="dat",
                reader_options={"sample_rate": 2_000_000},
                detector=FullSignalDetector(start_sample=2, end_sample=6),
                config=PreparationConfig(seq_len=4),
            )

        self.assertEqual(dataset.X.shape, (1, 2, 4))
        self.assertEqual(dataset.meta["window_start_sample"].item(), 2)
        self.assertEqual(dataset.source_id, "registry_dat")

    def test_prepare_file_forwards_optional_resampling(self):
        iq = np.arange(8, dtype=np.float32).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.dat"
            iq.tofile(path)
            dataset = prepare_file(
                path,
                source_id="resampled_dat",
                data_format="dat",
                reader_options={"sample_rate": 2_000_000},
                detector=FullSignalDetector(start_sample=2, end_sample=6),
                config=PreparationConfig(seq_len=4),
                resampling=ResamplingConfig(target_sample_rate=4_000_000),
            )

        self.assertEqual(dataset.X.shape, (2, 2, 4))
        self.assertEqual(float(dataset.meta["sample_rate"]), 4_000_000)
        self.assertEqual(float(dataset.meta["source_sample_rate"]), 2_000_000)

    def test_file_builder_writes_resampling_contract_and_summary(self):
        iq = np.arange(8, dtype=np.float32).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = base / "capture.dat"
            output_path = base / "prepared.npz"
            iq.tofile(input_path)

            result = build_prepared_dataset(
                input_path,
                output_path,
                source_id="resampled_builder",
                data_format="dat",
                reader_options={"sample_rate": 2_000_000},
                detector=FullSignalDetector(start_sample=2, end_sample=6),
                config=PreparationConfig(seq_len=4),
                resampling=ResamplingConfig(target_sample_rate=4_000_000),
            )

            with np.load(output_path, allow_pickle=False) as prepared:
                self.assertEqual(float(prepared["sample_rate"]), 4_000_000)
                self.assertEqual(
                    float(prepared["source_sample_rate"]), 2_000_000
                )
                self.assertEqual(prepared["coordinate_schema"].item(), "dual_rate_v1")
                np.testing.assert_array_equal(
                    prepared["target_window_start_sample"], [4, 8]
                )
                np.testing.assert_array_equal(
                    prepared["source_window_start_sample"], [2, 4]
                )

            summary = json.loads(
                Path(result["summary_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(result["sample_rate"], 4_000_000)
        self.assertEqual(summary["source_sample_rate"], 2_000_000)
        self.assertEqual(summary["resampling"]["resample_up"], 2)
        self.assertEqual(summary["resampling"]["resample_down"], 1)

    def test_detector_is_independent_from_source_format(self):
        detector = FixedDetector([SignalRegion(2, 6, "fixed_test")])
        dataset = prepare_signal(
            _raw_signal(np.arange(8, dtype=np.float32)),
            detector=detector,
            config=PreparationConfig(seq_len=4),
        )
        self.assertEqual(dataset.X.shape, (1, 2, 4))
        self.assertEqual(dataset.meta["detector"], "fixed_test")


if __name__ == "__main__":
    unittest.main()
