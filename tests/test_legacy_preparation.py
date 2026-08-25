import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion.preparation.legacy_burst import smart_extract_bursts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LegacyBurstCompatibilityTests(unittest.TestCase):
    def test_historical_helper_remains_callable_from_preparation(self):
        noise = np.zeros(256, dtype=np.complex64)
        burst = np.ones(160, dtype=np.complex64)
        capture = np.concatenate((noise, burst, noise))
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "capture.dat"
            output_path = Path(directory) / "bursts.dat"
            capture.tofile(input_path)
            messages: list[str] = []
            result = smart_extract_bursts(
                str(input_path), str(output_path), 128, messages
            )
            output = np.fromfile(output_path, dtype=np.complex64)

        self.assertTrue(result)
        self.assertGreaterEqual(output.size, 128)
        self.assertTrue(any("提取完成" in message for message in messages))

    def test_onnx_module_no_longer_implements_raw_detection(self):
        path = PROJECT_ROOT / "radioml-iq-modulation/scripts/onnx_inference.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }

        self.assertNotIn("smart_extract_bursts", functions)
        self.assertNotIn("_sync_random_sample_and_inference", functions)
        self.assertIn("signal_fusion.preparation.legacy_burst", imports)


if __name__ == "__main__":
    unittest.main()
