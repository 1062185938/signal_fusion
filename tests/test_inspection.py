import importlib.util
from contextlib import redirect_stdout
import io
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
