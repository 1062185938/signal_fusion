import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion import Evidence, ModelManifest, PreparedDataset
from signal_fusion.model_inference import (
    ModelInferenceResult,
    ModelInferenceService,
    RankedPrediction,
    load_label_map,
    stable_softmax,
)
from signal_fusion.model_inference.cli import run_legacy_cli
from signal_fusion.model_inference.legacy import (
    _sync_signal_inference,
    load_mod_labels,
    recognize_iq_modulation,
    run_onnx_inference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import compatibility wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModelBackend:
    input_name = "input"
    input_shape = ("batch_size", 2, 4)
    input_dtype = "float32"
    output_names = ("output", "feature")
    provider = "CPU"

    def run(self, batch):
        score = batch[:, 0, :].mean(axis=1)
        logits = np.stack((score, np.zeros_like(score), -score), axis=1).astype(
            np.float32
        )
        features = np.stack(
            (batch[:, 0, :].sum(axis=1), batch[:, 1, :].sum(axis=1)),
            axis=1,
        ).astype(np.float32)
        return {"output": logits, "feature": features}


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_id="fake_modulation_model",
        model_format="onnx",
        model_path="unused-in-injected-backend.onnx",
        input_name="input",
        input_shape=("batch_size", 2, 4),
        outputs={
            "output": ("batch_size", 3),
            "feature": ("batch_size", 2),
        },
        labels=("POSITIVE", "NEUTRAL", "NEGATIVE"),
        metadata={"logits_output": "output"},
    )


class LabelMapTests(unittest.TestCase):
    def test_loads_numeric_keys_in_index_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(
                json.dumps({"2": "C", "0": "A", "1": "B"}),
                encoding="utf-8",
            )
            self.assertEqual(load_label_map(path), ("A", "B", "C"))

    def test_rejects_non_contiguous_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({"0": "A", "2": "C"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "continuous"):
                load_label_map(path)


class ModelInferenceServiceTests(unittest.TestCase):
    def test_predicts_selected_samples_and_preserves_auxiliary_output(self):
        x = np.zeros((3, 2, 4), dtype=np.float32)
        x[0, 0, :] = 2.0
        x[1, 0, :] = -2.0
        x[2, 0, :] = 1.0
        dataset = PreparedDataset(X=x, source_id="prepared_fixture")

        result = ModelInferenceService(
            _manifest(),
            backend=FakeModelBackend(),
        ).predict(
            dataset,
            batch_size=1,
            sample_indices=np.asarray([2, 0], dtype=np.int64),
        )

        self.assertIsInstance(result, ModelInferenceResult)
        self.assertEqual(result.logits.shape, (2, 3))
        self.assertEqual(result.probabilities.shape, (2, 3))
        self.assertEqual(result.auxiliary_outputs["feature"].shape, (2, 2))
        np.testing.assert_array_equal(result.sample_indices, [2, 0])
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)

    def test_top5_is_limited_by_class_count_and_projects_to_evidence(self):
        x = np.zeros((1, 2, 4), dtype=np.float32)
        x[0, 0, :] = 2.0
        result = ModelInferenceService(
            _manifest(),
            backend=FakeModelBackend(),
        ).predict(PreparedDataset(X=x, source_id="topk_fixture"))

        predictions = result.top_k_for_sample(0, top_k=5)
        evidence = result.evidence_for_sample(0, top_k=5)

        self.assertEqual(len(predictions), 3)
        self.assertIsInstance(predictions[0], RankedPrediction)
        self.assertEqual(predictions[0].label, "POSITIVE")
        self.assertIsInstance(evidence, Evidence)
        self.assertEqual(evidence.kind, "modulation_prediction")
        self.assertEqual(evidence.producer, "fake_modulation_model")
        self.assertEqual(evidence.payload["sample_index"], 0)
        self.assertEqual(len(evidence.payload["top_k"]), 3)

    def test_rejects_manifest_input_shape_mismatch(self):
        dataset = PreparedDataset(
            X=np.zeros((1, 2, 8), dtype=np.float32),
            source_id="wrong_seq_len",
        )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            ModelInferenceService(
                _manifest(),
                backend=FakeModelBackend(),
            ).predict(dataset)

    def test_softmax_is_stable_for_large_logits(self):
        probabilities = stable_softmax(
            np.asarray([[10_000.0, 9_999.0, -10_000.0]], dtype=np.float32)
        )
        self.assertTrue(np.all(np.isfinite(probabilities)))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


class ModelInferenceCompatibilityTests(unittest.TestCase):
    def test_legacy_module_is_a_thin_core_wrapper(self):
        wrapper = _load_module(
            "legacy_onnx_inference",
            PROJECT_ROOT / "radioml-iq-modulation/scripts/onnx_inference.py",
        )

        self.assertIs(wrapper._sync_signal_inference, _sync_signal_inference)
        self.assertIs(wrapper.load_mod_labels, load_mod_labels)
        self.assertIs(wrapper.recognize_iq_modulation, recognize_iq_modulation)
        self.assertIs(wrapper.run_onnx_inference, run_onnx_inference)
        self.assertIs(wrapper.main, run_legacy_cli)


if __name__ == "__main__":
    unittest.main()
