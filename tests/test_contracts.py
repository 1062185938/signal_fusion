import unittest

import numpy as np

from signal_fusion import Evidence, ModelManifest, PreparedDataset


class PreparedDatasetTests(unittest.TestCase):
    def test_valid_dataset_exposes_shape_properties_and_legacy_view(self):
        x = np.zeros((3, 2, 128), dtype=np.float32)
        y = np.array([0, 1, 1], dtype=np.int64)
        dataset = PreparedDataset(
            X=x,
            y=y,
            meta={"sample_rate": 1_000_000.0},
            source_id="synthetic_iq",
        )

        self.assertEqual(dataset.num_samples, 3)
        self.assertEqual(dataset.seq_len, 128)
        legacy = dataset.to_legacy_dict()
        self.assertIs(legacy["X"], x)
        self.assertIs(legacy["y"], y)
        self.assertEqual(legacy["meta"]["source_id"], "synthetic_iq")

    def test_rejects_noncanonical_x(self):
        with self.assertRaises(TypeError):
            PreparedDataset(X=np.zeros((1, 2, 8), dtype=np.float64))
        with self.assertRaises(ValueError):
            PreparedDataset(X=np.zeros((1, 8, 2), dtype=np.float32))

    def test_rejects_label_count_mismatch(self):
        with self.assertRaises(ValueError):
            PreparedDataset(
                X=np.zeros((2, 2, 8), dtype=np.float32),
                y=np.array([0], dtype=np.int64),
            )


class EvidenceTests(unittest.TestCase):
    def test_serializable_view_preserves_explainable_fields(self):
        evidence = Evidence(
            source_id="lora_slices_128_example",
            kind="modulation_prediction",
            producer="lora_deep_iq_cnn_single_class",
            payload={"top1": "LORA"},
            confidence=np.float32(0.75),
        )

        self.assertEqual(evidence.to_dict()["confidence"], 0.75)
        self.assertEqual(evidence.to_dict()["payload"]["top1"], "LORA")

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            Evidence(
                source_id="source",
                kind="prediction",
                producer="model",
                payload={},
                confidence=1.1,
            )


class ModelManifestTests(unittest.TestCase):
    def test_describes_phase0_onnx_contract(self):
        manifest = ModelManifest(
            model_id="lora_deep_iq_cnn_single_class",
            model_format="ONNX",
            model_path="radioml-iq-modulation/training/sigmf_lora/deep_iq_cnn.onnx",
            input_name="input",
            input_shape=("batch_size", 2, 128),
            outputs={
                "output": ("batch_size", 1),
                "feature": ("batch_size", 300),
            },
            labels=("LORA",),
        )

        payload = manifest.to_dict()
        self.assertEqual(payload["model_format"], "onnx")
        self.assertEqual(payload["input"]["shape"], ["batch_size", 2, 128])
        self.assertEqual(payload["outputs"]["feature"], ["batch_size", 300])

    def test_rejects_duplicate_labels(self):
        with self.assertRaises(ValueError):
            ModelManifest(
                model_id="model",
                model_format="onnx",
                model_path="model.onnx",
                input_name="input",
                input_shape=(None, 2, 128),
                outputs={"output": (None, 2)},
                labels=("A", "A"),
            )


if __name__ == "__main__":
    unittest.main()

