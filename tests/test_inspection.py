import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from signal_fusion import PreparedDataset
from signal_fusion.io import write_prepared_dataset
from signal_fusion.preparation.inspection import inspect_prepared_slices


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PreparedDatasetInspectionTests(unittest.TestCase):
    def test_generic_region_schema_is_inspected_without_sigmf_dependency(self):
        dataset = PreparedDataset(
            X=np.zeros((2, 2, 8), dtype=np.float32),
            y=np.zeros(2, dtype=np.int64),
            source_id="inspection_fixture",
            meta={
                "sample_rate": 1_000.0,
                "seq_len": 8,
                "hop_len": 8,
                "region_id": np.asarray([0, 0], dtype=np.int64),
                "window_id": np.asarray([0, 1], dtype=np.int64),
                "window_start_sample": np.asarray([10, 18], dtype=np.int64),
                "window_end_sample": np.asarray([18, 26], dtype=np.int64),
                "region_start_sample": np.asarray([10, 10], dtype=np.int64),
                "region_end_sample": np.asarray([26, 26], dtype=np.int64),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            npz_path = base / "prepared.npz"
            write_prepared_dataset(dataset, npz_path)
            with redirect_stdout(io.StringIO()):
                with patch(
                    "signal_fusion.preparation.inspection._plot_window",
                    return_value=str(base / "window.png"),
                ):
                    result = inspect_prepared_slices(
                        str(npz_path),
                        str(base / "plots"),
                        max_regions=1,
                        windows_per_region=1,
                    )

        self.assertEqual(result["num_regions"], 1)
        self.assertEqual(result["plotted_windows"], 1)
        self.assertEqual(result["warning_count"], 0)

    def test_dual_rate_inspection_opens_raw_data_with_source_coordinates(self):
        raw_iq = (
            np.arange(64, dtype=np.float32)
            + 1j * np.arange(100, 164, dtype=np.float32)
        ).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_path = base / "capture.dat"
            npz_path = base / "prepared.npz"
            raw_iq.tofile(raw_path)
            dataset = PreparedDataset(
                X=np.zeros((2, 2, 8), dtype=np.float32),
                y=np.zeros(2, dtype=np.int64),
                source_id="dual_rate_fixture",
                meta={
                    "source_path": str(raw_path),
                    "sample_rate": 4_000.0,
                    "source_sample_rate": 1_000.0,
                    "coordinate_schema": "dual_rate_v1",
                    "seq_len": 8,
                    "hop_len": 8,
                    "region_id": np.asarray([0, 0], dtype=np.int64),
                    "window_id": np.asarray([0, 1], dtype=np.int64),
                    "window_start_sample": np.asarray([40, 48], dtype=np.int64),
                    "window_end_sample": np.asarray([48, 56], dtype=np.int64),
                    "region_start_sample": np.asarray([40, 40], dtype=np.int64),
                    "region_end_sample": np.asarray([56, 56], dtype=np.int64),
                    "target_window_start_sample": np.asarray(
                        [40, 48], dtype=np.int64
                    ),
                    "target_window_end_sample": np.asarray(
                        [48, 56], dtype=np.int64
                    ),
                    "target_region_start_sample": np.asarray(
                        [40, 40], dtype=np.int64
                    ),
                    "target_region_end_sample": np.asarray(
                        [56, 56], dtype=np.int64
                    ),
                    "source_window_start_sample": np.asarray(
                        [10, 12], dtype=np.int64
                    ),
                    "source_window_end_sample": np.asarray(
                        [12, 14], dtype=np.int64
                    ),
                    "source_region_start_sample": np.asarray(
                        [10, 10], dtype=np.int64
                    ),
                    "source_region_end_sample": np.asarray(
                        [14, 14], dtype=np.int64
                    ),
                },
            )
            write_prepared_dataset(dataset, npz_path)

            def inspect_overview(signal, data, schema, *args, **kwargs):
                self.assertEqual(signal.sample_rate, 1_000.0)
                self.assertEqual(schema.source_region_start, "source_region_start_sample")
                self.assertEqual(schema.source_window_start, "source_window_start_sample")
                np.testing.assert_array_equal(signal.read_samples(10, 4), raw_iq[10:14])
                return str(base / "overview.png")

            with patch(
                "signal_fusion.preparation.inspection._plot_region_overview",
                side_effect=inspect_overview,
            ), patch(
                "signal_fusion.preparation.inspection._plot_window",
                return_value=str(base / "window.png"),
            ):
                result = inspect_prepared_slices(
                    str(npz_path),
                    str(base / "plots"),
                    data_format="dat",
                    max_regions=1,
                    windows_per_region=1,
                )

        self.assertEqual(result["coordinate_schema"], "dual_rate_v1")
        self.assertEqual(result["source_sample_rate"], 1_000.0)
        self.assertEqual(result["sample_rate"], 4_000.0)
        self.assertEqual(result["warning_count"], 0)

    def test_dual_rate_inspection_rejects_wrong_raw_sample_rate(self):
        raw_iq = np.zeros(32, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_path = base / "capture.dat"
            npz_path = base / "prepared.npz"
            raw_iq.tofile(raw_path)
            dataset = PreparedDataset(
                X=np.zeros((1, 2, 8), dtype=np.float32),
                source_id="dual_rate_mismatch",
                meta={
                    "source_path": str(raw_path),
                    "sample_rate": 4_000.0,
                    "source_sample_rate": 1_000.0,
                    "coordinate_schema": "dual_rate_v1",
                    "seq_len": 8,
                    "hop_len": 8,
                    "region_id": np.asarray([0], dtype=np.int64),
                    "window_id": np.asarray([0], dtype=np.int64),
                    "window_start_sample": np.asarray([40], dtype=np.int64),
                    "window_end_sample": np.asarray([48], dtype=np.int64),
                    "region_start_sample": np.asarray([40], dtype=np.int64),
                    "region_end_sample": np.asarray([48], dtype=np.int64),
                    "source_window_start_sample": np.asarray([10], dtype=np.int64),
                    "source_window_end_sample": np.asarray([12], dtype=np.int64),
                    "source_region_start_sample": np.asarray([10], dtype=np.int64),
                    "source_region_end_sample": np.asarray([12], dtype=np.int64),
                },
            )
            write_prepared_dataset(dataset, npz_path)

            with self.assertRaisesRegex(ValueError, "does not match"):
                inspect_prepared_slices(
                    str(npz_path),
                    str(base / "plots"),
                    raw_path=str(raw_path),
                    data_format="dat",
                    sample_rate=2_000,
                )

    def test_dual_rate_sigmf_inspection_can_infer_metadata_path(self):
        raw_iq = np.zeros(32, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_path = base / "capture.sigmf-data"
            meta_path = base / "capture.sigmf-meta"
            npz_path = base / "prepared.npz"
            raw_iq.tofile(raw_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "global": {
                            "core:datatype": "cf32_le",
                            "core:sample_rate": 1_000.0,
                            "core:version": "1.0.0",
                        },
                        "captures": [{"core:sample_start": 0}],
                        "annotations": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset = PreparedDataset(
                X=np.zeros((1, 2, 8), dtype=np.float32),
                source_id="dual_rate_sigmf",
                meta={
                    "source_path": str(raw_path),
                    "sample_rate": 4_000.0,
                    "source_sample_rate": 1_000.0,
                    "coordinate_schema": "dual_rate_v1",
                    "seq_len": 8,
                    "hop_len": 8,
                    "region_id": np.asarray([0], dtype=np.int64),
                    "window_id": np.asarray([0], dtype=np.int64),
                    "window_start_sample": np.asarray([40], dtype=np.int64),
                    "window_end_sample": np.asarray([48], dtype=np.int64),
                    "region_start_sample": np.asarray([40], dtype=np.int64),
                    "region_end_sample": np.asarray([48], dtype=np.int64),
                    "source_window_start_sample": np.asarray([10], dtype=np.int64),
                    "source_window_end_sample": np.asarray([12], dtype=np.int64),
                    "source_region_start_sample": np.asarray([10], dtype=np.int64),
                    "source_region_end_sample": np.asarray([12], dtype=np.int64),
                },
            )
            write_prepared_dataset(dataset, npz_path)

            with patch(
                "signal_fusion.preparation.inspection._plot_region_overview",
                return_value=str(base / "overview.png"),
            ), patch(
                "signal_fusion.preparation.inspection._plot_window",
                return_value=str(base / "window.png"),
            ):
                result = inspect_prepared_slices(
                    str(npz_path),
                    str(base / "plots"),
                    raw_path=str(raw_path),
                    data_format="sigmf",
                    max_regions=1,
                    windows_per_region=1,
                )

        self.assertEqual(result["source_sample_rate"], 1_000.0)
        self.assertEqual(result["warning_count"], 0)

    def test_legacy_script_is_a_thin_core_wrapper(self):
        path = PROJECT_ROOT / "radioml-iq-modulation/scripts/inspect_sigmf_slices.py"
        spec = importlib.util.spec_from_file_location("legacy_inspection_wrapper", path)
        if spec is None or spec.loader is None:
            self.fail("Cannot import legacy inspection wrapper")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(
            module.inspect_sigmf_slices.__module__,
            "signal_fusion.preparation.inspection",
        )


if __name__ == "__main__":
    unittest.main()
