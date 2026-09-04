from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io as sio

from signal_fusion.preparation.cli import main


def _synthetic_capture() -> np.ndarray:
    rng = np.random.default_rng(12)

    def noise(count):
        return (
            rng.normal(0.0, 0.01, count)
            + 1j * rng.normal(0.0, 0.01, count)
        ).astype(np.complex64)

    def burst(count):
        phase = np.arange(count, dtype=np.float32) * np.float32(0.2)
        return (0.6 * np.exp(1j * phase) + noise(count)).astype(np.complex64)

    return np.concatenate([noise(250), burst(300), noise(250)]).astype(np.complex64)


class SignalPrepareCliTests(unittest.TestCase):
    def test_unified_sigmf_cli_records_source_and_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data_path = base / "capture.sigmf-data"
            meta_path = base / "capture.sigmf-meta"
            output_path = base / "prepared.npz"
            _synthetic_capture().tofile(data_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "global": {
                            "core:datatype": "cf32_le",
                            "core:sample_rate": 10_000,
                        }
                    }
                ),
                encoding="utf-8",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--input_path",
                        str(data_path),
                        "--meta_path",
                        str(meta_path),
                        "--source_id",
                        "synthetic_cli_sigmf",
                        "--output_path",
                        str(output_path),
                        "--seq_len",
                        "128",
                        "--hop_len",
                        "64",
                        "--chunk_size",
                        "300",
                        "--window_ms",
                        "2",
                        "--start_threshold_db",
                        "10",
                        "--end_threshold_db",
                        "6",
                        "--min_signal_ms",
                        "10",
                        "--min_gap_ms",
                        "5",
                        "--pad_before_ms",
                        "0",
                        "--pad_after_ms",
                        "0",
                        "--noise_probe_count",
                        "4",
                    ]
                )

            self.assertEqual(exit_code, 0)
            with np.load(output_path, allow_pickle=False) as prepared:
                self.assertEqual(prepared["source_id"].item(), "synthetic_cli_sigmf")
                self.assertEqual(prepared["X"].shape[1:], (2, 128))
                self.assertGreater(prepared["X"].shape[0], 0)
                self.assertIn("burst_id", prepared.files)
                self.assertNotIn("coordinate_schema", prepared.files)
            summary = json.loads(
                (base / "prepared_dataset_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["source_id"], "synthetic_cli_sigmf")
            self.assertEqual(summary["detector"], "energy_v1")

    def test_full_signal_cli_can_resample_before_target_windowing(self):
        iq = np.arange(16, dtype=np.float32).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = base / "capture.dat"
            output_path = base / "prepared.npz"
            iq.tofile(input_path)

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--input_path",
                        str(input_path),
                        "--output_path",
                        str(output_path),
                        "--source_id",
                        "cli_resampled_dat",
                        "--data_format",
                        "dat",
                        "--sample_rate",
                        "1000000",
                        "--detector",
                        "full_signal",
                        "--start_sample",
                        "2",
                        "--end_sample",
                        "6",
                        "--target-sample-rate",
                        "4000000",
                        "--seq_len",
                        "4",
                        "--hop_len",
                        "4",
                        "--normalize",
                        "none",
                        "--no_remove_dc",
                    ]
                )

            self.assertEqual(exit_code, 0)
            with np.load(output_path, allow_pickle=False) as prepared:
                self.assertEqual(prepared["X"].shape, (4, 2, 4))
                self.assertEqual(float(prepared["sample_rate"]), 4_000_000)
                self.assertEqual(
                    float(prepared["source_sample_rate"]), 1_000_000
                )
                self.assertEqual(
                    prepared["coordinate_schema"].item(), "dual_rate_v1"
                )
                np.testing.assert_array_equal(
                    prepared["window_start_sample"], [8, 12, 16, 20]
                )
                np.testing.assert_array_equal(
                    prepared["source_window_start_sample"], [2, 3, 4, 5]
                )

    def test_energy_cli_uses_generic_pipeline_when_resampling_is_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = base / "capture.dat"
            output_path = base / "prepared.npz"
            _synthetic_capture().tofile(input_path)

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--input_path",
                        str(input_path),
                        "--output_path",
                        str(output_path),
                        "--source_id",
                        "energy_resampled",
                        "--data_format",
                        "dat",
                        "--sample_rate",
                        "10000",
                        "--detector",
                        "energy_v1",
                        "--target_sample_rate",
                        "40000",
                        "--seq_len",
                        "128",
                        "--hop_len",
                        "128",
                        "--chunk_size",
                        "300",
                        "--window_ms",
                        "2",
                        "--start_threshold_db",
                        "10",
                        "--end_threshold_db",
                        "6",
                        "--min_signal_ms",
                        "10",
                        "--min_gap_ms",
                        "5",
                        "--pad_before_ms",
                        "0",
                        "--pad_after_ms",
                        "0",
                        "--noise_probe_count",
                        "4",
                    ]
                )

            self.assertEqual(exit_code, 0)
            with np.load(output_path, allow_pickle=False) as prepared:
                self.assertGreater(prepared["X"].shape[0], 0)
                self.assertIn("region_id", prepared.files)
                self.assertNotIn("burst_id", prepared.files)
                self.assertEqual(float(prepared["sample_rate"]), 40_000)
                self.assertEqual(float(prepared["source_sample_rate"]), 10_000)
                self.assertEqual(
                    prepared["resampling_method"].item(), "polyphase"
                )

    def test_cli_rejects_invalid_target_sample_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.dat"
            np.zeros(16, dtype=np.complex64).tofile(path)
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(
                    [
                        "--input_path",
                        str(path),
                        "--source_id",
                        "invalid_target_rate",
                        "--output_path",
                        str(Path(directory) / "prepared.npz"),
                        "--data_format",
                        "dat",
                        "--sample_rate",
                        "1000000",
                        "--detector",
                        "full_signal",
                        "--target_sample_rate",
                        "0",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("target_sample_rate must be finite and positive", errors.getvalue())

    def test_dat_cli_requires_sample_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.dat"
            np.zeros(128, dtype=np.complex64).tofile(path)
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(
                    [
                        "--input_path",
                        str(path),
                        "--source_id",
                        "missing_rate",
                        "--output_path",
                        str(Path(directory) / "prepared.npz"),
                    ]
                )
        self.assertEqual(exit_code, 1)
        self.assertIn("--sample_rate is required", errors.getvalue())

    def test_full_signal_cli_supports_mat_and_dat(self):
        capture = _synthetic_capture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mat_path = base / "capture.mat"
            dat_path = base / "capture.dat"
            sio.savemat(mat_path, {"iq": capture, "sample_rate": 10_000.0})
            capture.tofile(dat_path)

            cases = (
                (mat_path, []),
                (dat_path, ["--sample_rate", "10000"]),
            )
            for input_path, reader_args in cases:
                with self.subTest(suffix=input_path.suffix):
                    output_path = base / f"full_{input_path.suffix[1:]}.npz"
                    with redirect_stdout(io.StringIO()):
                        exit_code = main(
                            [
                            "--input_path",
                            str(input_path),
                            "--output_path",
                            str(output_path),
                            "--source_id",
                            f"full_{input_path.suffix[1:]}",
                            "--detector",
                            "full_signal",
                            "--start_sample",
                            "100",
                            "--end_sample",
                            "500",
                            "--seq_len",
                            "128",
                            "--hop_len",
                            "128",
                            "--remainder",
                            "pad",
                            "--normalize",
                            "none",
                            "--no_remove_dc",
                                *reader_args,
                            ]
                        )
                    self.assertEqual(exit_code, 0)
                    with np.load(output_path, allow_pickle=False) as prepared:
                        self.assertEqual(prepared["X"].shape, (4, 2, 128))
                        self.assertEqual(prepared["detector"].item(), "full_signal")
                        np.testing.assert_array_equal(
                            prepared["valid_sample_count"], [128, 128, 128, 16]
                        )

    def test_energy_v1_cli_supports_mat_and_dat(self):
        capture = _synthetic_capture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mat_path = base / "capture.mat"
            dat_path = base / "capture.dat"
            sio.savemat(mat_path, {"iq": capture, "sample_rate": 10_000.0})
            capture.tofile(dat_path)

            for input_path, reader_args in (
                (mat_path, []),
                (dat_path, ["--sample_rate", "10000"]),
            ):
                with self.subTest(suffix=input_path.suffix):
                    output_path = base / f"energy_{input_path.suffix[1:]}.npz"
                    with redirect_stdout(io.StringIO()):
                        exit_code = main(
                            [
                            "--input_path",
                            str(input_path),
                            "--output_path",
                            str(output_path),
                            "--source_id",
                            f"energy_{input_path.suffix[1:]}",
                            "--detector",
                            "energy_v1",
                            "--seq_len",
                            "128",
                            "--hop_len",
                            "64",
                            "--chunk_size",
                            "300",
                            "--window_ms",
                            "2",
                            "--start_threshold_db",
                            "10",
                            "--end_threshold_db",
                            "6",
                            "--min_signal_ms",
                            "10",
                            "--min_gap_ms",
                            "5",
                            "--pad_before_ms",
                            "0",
                            "--pad_after_ms",
                            "0",
                            "--noise_probe_count",
                            "4",
                                *reader_args,
                            ]
                        )
                    self.assertEqual(exit_code, 0)
                    with np.load(output_path, allow_pickle=False) as prepared:
                        self.assertGreater(prepared["X"].shape[0], 0)
                        self.assertEqual(
                            prepared["source_id"].item(),
                            f"energy_{input_path.suffix[1:]}",
                        )


if __name__ == "__main__":
    unittest.main()
