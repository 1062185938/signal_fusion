import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODELS_DIR = PROJECT_ROOT / "radioml-iq-modulation/scripts/models"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from signal_fusion.modeling import (
        DeepConvNet1D,
        DeepConvNet_1D,
        LSTMIQ,
        MODEL_REGISTRY,
        build_model,
    )


class ModelingWrapperStructureTests(unittest.TestCase):
    def test_legacy_model_files_only_wrap_canonical_modeling(self):
        for filename in (
            "__init__.py",
            "deepconvnet_1d.py",
            "lstm_iq.py",
            "model_factory.py",
        ):
            path = LEGACY_MODELS_DIR / filename
            tree = ast.parse(path.read_text(encoding="utf-8"))
            class_names = [
                node.name for node in tree.body if isinstance(node, ast.ClassDef)
            ]
            function_names = [
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            ]
            imported_modules = {
                node.module
                for node in tree.body
                if isinstance(node, ast.ImportFrom) and node.module is not None
            }

            self.assertEqual(class_names, [])
            self.assertNotIn("build_model", function_names)
            self.assertTrue(
                any(
                    module.startswith("signal_fusion.modeling")
                    for module in imported_modules
                )
            )


@unittest.skipUnless(TORCH_AVAILABLE, "torch optional dependency is not installed")
class ModelingRuntimeTests(unittest.TestCase):
    def test_registry_preserves_names_alias_and_factory_errors(self):
        self.assertEqual(
            set(MODEL_REGISTRY),
            {"deepconvnet_1d", "lstm_iq"},
        )
        self.assertIs(DeepConvNet_1D, DeepConvNet1D)
        self.assertIsInstance(build_model("DEEPCONVNET_1D", 3), DeepConvNet1D)
        self.assertIsInstance(build_model("lstm_iq", 3), LSTMIQ)
        with self.assertRaisesRegex(
            ValueError,
            "Available models: deepconvnet_1d, lstm_iq",
        ):
            build_model("unknown", 3)

    def test_deepconvnet_state_keys_and_forward_shapes_are_stable(self):
        model = build_model("deepconvnet_1d", 3, input_channels=2, seq_len=128)
        keys = tuple(model.state_dict())
        model.eval()
        with torch.no_grad():
            logits, features = model(torch.zeros((2, 2, 128), dtype=torch.float32))

        self.assertEqual(len(keys), 62)
        self.assertEqual(keys[0], "features.0.weight")
        self.assertEqual(keys[-1], "fc1.1.bias")
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(features.shape), (2, 300))

    def test_lstm_state_keys_and_forward_shapes_are_stable(self):
        model = build_model("lstm_iq", 3, input_channels=2, seq_len=128)
        keys = tuple(model.state_dict())
        model.eval()
        with torch.no_grad():
            logits, features = model(torch.zeros((2, 2, 128), dtype=torch.float32))

        self.assertEqual(len(keys), 15)
        self.assertEqual(keys[0], "input_norm.weight")
        self.assertEqual(keys[-1], "fc.bias")
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(features.shape), (2, 128))

    def test_historical_pth_strictly_loads_and_matches_onnx_fixture(self):
        state = torch.load(
            PROJECT_ROOT
            / "radioml-iq-modulation/training/sigmf_lora/best_model.pth",
            map_location="cpu",
            weights_only=True,
        )
        excluded_keys = {
            key
            for key in state
            if key.endswith("total_ops") or key.endswith("total_params")
        }
        clean_state = {
            key: value for key, value in state.items() if key not in excluded_keys
        }
        model = build_model("deepconvnet_1d", 1, input_channels=2, seq_len=128)
        incompatible = model.load_state_dict(clean_state, strict=True)
        model.eval()

        with np.load(
            PROJECT_ROOT / "tests/fixtures/lora_single_class_onnx_reference.npz",
            allow_pickle=False,
        ) as fixture:
            inputs = torch.from_numpy(
                np.ascontiguousarray(fixture["X"], dtype=np.float32)
            )
            with torch.no_grad():
                logits, features = model(inputs)
            np.testing.assert_allclose(
                logits.numpy(), fixture["logits"], rtol=1e-4, atol=1e-5
            )
            np.testing.assert_allclose(
                features.numpy(), fixture["features"], rtol=1e-4, atol=1e-5
            )

        self.assertEqual(excluded_keys, {"total_ops", "total_params"})
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_legacy_package_reexports_canonical_objects(self):
        scripts_dir = str(LEGACY_MODELS_DIR.parent)
        sys.path.insert(0, scripts_dir)
        try:
            sys.modules.pop("models", None)
            import models

            self.assertIs(models.build_model, build_model)
            self.assertIs(models.DeepConvNet1D, DeepConvNet1D)
            self.assertIs(models.LSTMIQ, LSTMIQ)
        finally:
            sys.modules.pop("models", None)
            if sys.path[0] == scripts_dir:
                sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
