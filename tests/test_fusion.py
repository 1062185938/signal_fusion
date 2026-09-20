import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion import ModelManifest, PreparedDataset
from signal_fusion.feature_extraction import FEATURE_COUNT, FeatureExtractionService
from signal_fusion.fusion import (
    CompleteRegion,
    FusionManifest,
    analyze_group,
    build_hermes_input,
    load_complete_region,
)
from signal_fusion.model_inference import ModelInferenceService


class _FeatureBackend:
    def __init__(self):
        self.input_lengths = []

    def extract_features(self, i_data, q_data, sample_rate):
        self.input_lengths.append(len(i_data))
        return np.arange(FEATURE_COUNT, dtype=np.float32) + float(np.mean(i_data))


class _ModelBackend:
    input_name = "input"
    input_shape = (None, 2, 128)
    input_dtype = "float32"
    output_names = ("output", "feature")
    provider = "CPU"

    def run(self, batch):
        logits = np.tile(
            np.asarray([0.0, 1.0, 3.0], dtype=np.float32),
            (len(batch), 1),
        )
        return {
            "output": logits,
            "feature": np.zeros((len(batch), 4), dtype=np.float32),
        }


class _FeatureClassifierResult:
    def __init__(self, labels, probabilities):
        self.labels = labels
        self.probabilities = np.asarray([probabilities], dtype=np.float64)
        self.model_id = "feature_model_with_private_training_details"
        self.provider = "CPUExecutionProvider"


class _FeatureClassifierService:
    def __init__(
        self,
        *,
        labels=("LoRa", "Zigbee", "BLE"),
        feature_names=None,
    ):
        self.labels = tuple(labels)
        self.feature_names = tuple(
            feature_names or FeatureExtractionService().feature_names
        )
        self.inputs = []

    def predict(self, features):
        self.inputs.append(np.asarray(features).copy())
        return _FeatureClassifierResult(
            self.labels,
            [0.8, 0.15, 0.05],
        )


def _manifest():
    return ModelManifest(
        model_id="demo_model",
        model_format="onnx",
        model_path="unused.onnx",
        input_name="input",
        input_shape=(None, 2, 128),
        outputs={"output": (None, 3), "feature": (None, 4)},
        labels=("LoRa", "Zigbee", "BLE"),
        metadata={"logits_output": "output"},
    )


def _fusion_manifest():
    return FusionManifest(
        labels=("LoRa", "Zigbee", "BLE"),
        iq_model_weight=0.5,
        feature_classifier_weight=0.5,
    )


def _dataset(region_length=256):
    return PreparedDataset(
        X=np.zeros((2, 2, 128), dtype=np.float32),
        y=np.asarray([2, 2], dtype=np.int64),
        source_id="demo:test",
        meta={
            "group_id": np.asarray([11, 11]),
            "source_region_id": np.asarray([8, 8]),
            "source_region_field": np.asarray(["region_id", "region_id"]),
            "window_id": np.asarray([0, 1]),
            "window_start_sample": np.asarray([500, 628]),
            "window_end_sample": np.asarray([628, 756]),
            "sample_source_id": np.asarray(["capture_b", "capture_b"]),
            "sample_rate": np.full(2, 4_000_000.0),
            "region_start_sample": np.asarray([500, 500]),
            "region_end_sample": np.asarray(
                [500 + region_length, 500 + region_length]
            ),
            "remove_dc": np.asarray(True),
            "rms_normalize": np.asarray(True),
            "rms_epsilon": np.asarray(1e-12),
        },
    )


def _region(sample_count=256):
    rng = np.random.default_rng(7)
    samples = (
        rng.standard_normal(sample_count)
        + 1j * rng.standard_normal(sample_count)
    ).astype(np.complex64)
    return CompleteRegion(
        samples=samples,
        sample_rate=4_000_000.0,
        start_sample=500,
        end_sample=500 + sample_count,
        source_id="capture_b",
        source_region_id=8,
        raw_source_path="raw.dat",
        source_dataset_path="slices.npz",
    )


class FusionTests(unittest.TestCase):
    def test_group_analysis_uses_one_region_feature_vector(self):
        feature_backend = _FeatureBackend()
        feature_classifier = _FeatureClassifierService()
        bundle = analyze_group(
            _dataset(),
            _region(),
            group_id=11,
            model_service=ModelInferenceService(
                _manifest(), backend=_ModelBackend()
            ),
            feature_classifier_service=feature_classifier,
            fusion_manifest=_fusion_manifest(),
            feature_service=FeatureExtractionService(),
            feature_backend=feature_backend,
        )

        self.assertEqual(bundle["schema_version"], 4)
        self.assertEqual(bundle["ground_truth"], {"class_index": 2, "label": "BLE"})
        self.assertEqual(bundle["region"]["sample_count"], 256)
        self.assertEqual(feature_backend.input_lengths, [256])
        model = bundle["iq_model_evidence"]
        self.assertEqual(model["region_top3"][0]["label"], "BLE")
        self.assertEqual(model["window_agreement"]["ratio"], 1.0)
        self.assertEqual(
            set(model["window_predictions"][0]), {"label", "confidence"}
        )
        self.assertEqual(
            bundle["feature_model_evidence"]["region_top3"][0]["label"],
            "LoRa",
        )
        self.assertEqual(len(feature_classifier.inputs), 1)
        self.assertEqual(feature_classifier.inputs[0].shape, (1, FEATURE_COUNT))
        self.assertEqual(bundle["fusion_result"]["final_label"], "BLE")
        self.assertEqual(bundle["fusion_result"]["weights"]["iq_model"], 0.5)

        features = bundle["feature_evidence"]
        self.assertEqual(features["scope"], "complete_continuous_region")
        self.assertEqual(features["feature_count"], FEATURE_COUNT)
        self.assertEqual(features["extraction_count"], 1)
        self.assertEqual(features["input"]["used_sample_count"], 256)
        self.assertFalse(features["input"]["truncated"])
        self.assertEqual(len(features["groups"]["time_domain"]), 15)
        first_feature = features["groups"]["time_domain"][0]
        self.assertEqual(
            set(first_feature),
            {
                "index",
                "code_name",
                "display_name",
                "display_name_zh",
                "value",
                "reference",
            },
        )
        self.assertNotIn("median", first_feature)
        self.assertNotIn("mean", first_feature)

        hermes_input = build_hermes_input(bundle, case_id=7)
        self.assertEqual(hermes_input["schema_version"], 5)
        self.assertEqual(hermes_input["analysis_id"], "case_0007")
        serialized = json.dumps(hermes_input).lower()
        self.assertNotIn("capture_b", serialized)
        self.assertNotIn("ground_truth", serialized)
        self.assertNotIn("class_feature_reference", serialized)
        self.assertNotIn('"noise"', serialized)
        self.assertNotIn("snr", serialized)
        self.assertNotIn("awgn", serialized)
        self.assertNotIn("raw.dat", serialized)
        self.assertNotIn("slices.npz", serialized)
        self.assertNotIn("producer", hermes_input["iq_model_evidence"])
        self.assertNotIn("provider", hermes_input["iq_model_evidence"])
        self.assertNotIn("producer", hermes_input["feature_model_evidence"])
        self.assertNotIn("provider", hermes_input["feature_model_evidence"])
        self.assertEqual(hermes_input["fusion_result"]["final_label"], "BLE")

    def test_blind_input_hides_added_noise_condition(self):
        bundle = analyze_group(
            _dataset(),
            _region(),
            group_id=11,
            model_service=ModelInferenceService(
                _manifest(), backend=_ModelBackend()
            ),
            feature_classifier_service=_FeatureClassifierService(),
            fusion_manifest=_fusion_manifest(),
            feature_backend=_FeatureBackend(),
            snr_db=5.0,
            noise_seed=9,
        )

        hermes_input = build_hermes_input(bundle, case_id=12)
        serialized = json.dumps(hermes_input).lower()
        self.assertEqual(hermes_input["analysis_id"], "case_0012")
        self.assertNotIn('"noise"', serialized)
        self.assertNotIn("snr", serialized)
        self.assertNotIn("awgn", serialized)
        self.assertNotIn("seed", serialized)

    def test_blind_case_id_must_be_positive_integer(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            build_hermes_input({}, case_id=0)
        with self.assertRaisesRegex(TypeError, "integer"):
            build_hermes_input({}, case_id=True)

    def test_noise_is_added_to_complete_region_before_both_branches(self):
        original = _region()
        original_samples = original.samples.copy()
        bundle = analyze_group(
            _dataset(),
            original,
            group_id=11,
            model_service=ModelInferenceService(
                _manifest(), backend=_ModelBackend()
            ),
            feature_classifier_service=_FeatureClassifierService(),
            fusion_manifest=_fusion_manifest(),
            feature_backend=_FeatureBackend(),
            snr_db=5.0,
            noise_seed=9,
        )

        np.testing.assert_array_equal(original.samples, original_samples)
        self.assertTrue(bundle["noise"]["applied"])
        self.assertEqual(
            bundle["noise"]["scope"], "complete_region_before_windowing"
        )
        self.assertAlmostEqual(bundle["noise"]["achieved_snr_db"], 5.0)

    def test_feature_input_is_truncated_to_16384_samples(self):
        feature_backend = _FeatureBackend()
        bundle = analyze_group(
            _dataset(region_length=20_000),
            _region(sample_count=20_000),
            group_id=11,
            model_service=ModelInferenceService(
                _manifest(), backend=_ModelBackend()
            ),
            feature_classifier_service=_FeatureClassifierService(),
            fusion_manifest=_fusion_manifest(),
            feature_backend=feature_backend,
        )

        self.assertEqual(feature_backend.input_lengths, [16_384])
        feature_input = bundle["feature_evidence"]["input"]
        self.assertEqual(feature_input["original_sample_count"], 20_000)
        self.assertEqual(feature_input["used_sample_count"], 16_384)
        self.assertTrue(feature_input["truncated"])
        self.assertEqual(feature_input["truncation_policy"], "keep_first_samples")

    def test_complete_region_loader_reads_recorded_dat_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
            raw_path = root / "data" / "raw" / "capture.dat"
            raw_path.parent.mkdir(parents=True)
            raw = (np.arange(32) + 1j * np.arange(32)[::-1]).astype(np.complex64)
            raw.tofile(raw_path)

            source_dataset_path = root / "data" / "processed" / "slice.npz"
            source_dataset_path.parent.mkdir(parents=True)
            np.savez(
                source_dataset_path,
                X=np.zeros((2, 2, 128), dtype=np.float32),
                source_path=np.asarray("data/raw/capture.dat"),
                sample_rate=np.asarray(4_000_000.0),
                center_frequency=np.asarray(2_400_000_000.0),
                region_id=np.asarray([4, 4]),
            )

            assembled_path = root / "assembled" / "test.npz"
            assembled_path.parent.mkdir()
            report = {
                "splits": {
                    "test": {
                        "sources": [
                            {
                                "source_id": "capture",
                                "resolved_path": str(source_dataset_path),
                            }
                        ]
                    }
                }
            }
            (assembled_path.parent / "assembly_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            dataset = PreparedDataset(
                X=np.zeros((2, 2, 128), dtype=np.float32),
                source_id="fixture:test",
                meta={
                    "split": np.asarray("test"),
                    "group_id": np.asarray([9, 9]),
                    "sample_source_id": np.asarray(["capture", "capture"]),
                    "source_region_id": np.asarray([4, 4]),
                    "source_region_field": np.asarray(["region_id", "region_id"]),
                    "region_start_sample": np.asarray([5, 5]),
                    "region_end_sample": np.asarray([15, 15]),
                    "sample_rate": np.full(2, 4_000_000.0),
                },
            )

            loaded = load_complete_region(assembled_path, dataset, group_id=9)

        np.testing.assert_array_equal(loaded.samples, raw[5:15])
        self.assertEqual(loaded.start_sample, 5)
        self.assertEqual(loaded.end_sample, 15)

    def test_missing_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            analyze_group(
                _dataset(),
                _region(),
                group_id=99,
                model_service=ModelInferenceService(
                    _manifest(), backend=_ModelBackend()
                ),
                feature_classifier_service=_FeatureClassifierService(),
                fusion_manifest=_fusion_manifest(),
                feature_backend=_FeatureBackend(),
            )

    def test_classifier_label_order_must_match(self):
        with self.assertRaisesRegex(ValueError, "labels must match"):
            analyze_group(
                _dataset(),
                _region(),
                group_id=11,
                model_service=ModelInferenceService(
                    _manifest(), backend=_ModelBackend()
                ),
                feature_classifier_service=_FeatureClassifierService(
                    labels=("BLE", "Zigbee", "LoRa")
                ),
                fusion_manifest=_fusion_manifest(),
                feature_backend=_FeatureBackend(),
            )

    def test_classifier_feature_order_must_match(self):
        names = list(FeatureExtractionService().feature_names)
        names[0], names[1] = names[1], names[0]
        with self.assertRaisesRegex(ValueError, "feature order"):
            analyze_group(
                _dataset(),
                _region(),
                group_id=11,
                model_service=ModelInferenceService(
                    _manifest(), backend=_ModelBackend()
                ),
                feature_classifier_service=_FeatureClassifierService(
                    feature_names=names
                ),
                fusion_manifest=_fusion_manifest(),
                feature_backend=_FeatureBackend(),
            )


if __name__ == "__main__":
    unittest.main()
