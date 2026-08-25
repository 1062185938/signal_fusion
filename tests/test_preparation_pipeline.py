import tempfile
from pathlib import Path
import unittest

import numpy as np

from signal_fusion import PreparedDataset
from signal_fusion.preparation import (
    FullSignalDetector,
    PreparationConfig,
    RawSignal,
    SignalDetector,
    SignalRegion,
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

