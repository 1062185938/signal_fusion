import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from signal_fusion.preparation import open_raw_signal
from signal_fusion.preparation.detectors import EnergyDetectorV1, EnergyDetectorV1Config
from signal_fusion.preparation.energy_v1_dataset import prepare_energy_v1


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _lora_config() -> EnergyDetectorV1Config:
    return EnergyDetectorV1Config(
        chunk_size=1_000_000,
        window_ms=1.0,
        start_threshold_db=6.0,
        end_threshold_db=5.0,
        min_signal_ms=8.0,
        min_gap_ms=2.0,
        pad_before_ms=0.0,
        pad_after_ms=0.0,
        release_windows=2,
        noise_percentile=20.0,
        noise_probe_count=8,
        ignore_initial_ms=30.0,
        window_power_ratio=0.05,
    )


class EnergyDetectorV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signal = open_raw_signal(
            PROJECT_ROOT / "data/raw/lora/sigmf_lora.sigmf-data",
            source_id="lora_raw_sigmf",
        )
        cls.detector = EnergyDetectorV1(_lora_config())
        cls.dataset = prepare_energy_v1(
            cls.signal,
            detector=cls.detector,
            seq_len=128,
            hop_len=64,
            remove_dc=True,
            normalize="rms",
            remainder="drop",
            label=0,
            class_name="LoRa",
        )

    def test_regions_match_phase0_energy_v1_summary(self):
        expected = json.loads(
            (
                PROJECT_ROOT
                / "data/processed/slices/sigmf_lora_dataset_128_dataset_summary.json"
            ).read_text(encoding="utf-8")
        )
        report = self.detector.last_report

        self.assertEqual(report["burst_regions"], expected["burst_regions"])
        self.assertEqual(len(report["coarse_candidates"]), 26)
        self.assertEqual(len(report["merged_candidates"]), 26)
        self.assertEqual(report["skipped_short"], 6)
        self.assertAlmostEqual(
            report["noise_floor_db"], expected["noise_floor_db"], places=12
        )

    def test_prepared_windows_preserve_phase0_schema_and_values(self):
        fixture_path = (
            PROJECT_ROOT / "data/processed/slices/sigmf_lora_dataset_128.npz"
        )
        with np.load(fixture_path, allow_pickle=False) as expected:
            self.assertEqual(self.dataset.X.shape, (4804, 2, 128))
            self.assertEqual(self.dataset.X.dtype, np.float32)
            self.assertTrue(
                np.allclose(
                    self.dataset.X, expected["X"], rtol=1e-6, atol=1e-7
                )
            )
            for field in (
                "burst_id",
                "window_id",
                "window_start_sample",
                "window_end_sample",
                "burst_start_sample",
                "burst_end_sample",
                "raw_burst_start_sample",
                "raw_burst_end_sample",
            ):
                np.testing.assert_array_equal(self.dataset.meta[field], expected[field])
            for field in (
                "burst_rms",
                "burst_mean_power",
                "burst_peak_power",
                "normalization_scale",
            ):
                np.testing.assert_allclose(
                    self.dataset.meta[field], expected[field], rtol=1e-6, atol=1e-12
                )

    def test_legacy_script_reexports_core_builder(self):
        path = (
            PROJECT_ROOT
            / "radioml-iq-modulation/scripts/sigmf_dataset_builder.py"
        )
        spec = importlib.util.spec_from_file_location("legacy_sigmf_wrapper", path)
        if spec is None or spec.loader is None:
            self.fail("Cannot import legacy SigMF wrapper")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(
            module.build_sigmf_dataset.__module__,
            "signal_fusion.preparation.energy_v1_dataset",
        )


if __name__ == "__main__":
    unittest.main()

