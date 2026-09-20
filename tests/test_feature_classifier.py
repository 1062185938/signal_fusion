import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from signal_fusion.feature_classifier import (
    FeatureClassifierService,
    build_region_feature_dataset,
    evaluate_feature_classifier,
    extract_region_feature_split,
    load_region_feature_split,
)
from signal_fusion.feature_classifier.trainer import train_feature_classifier
from signal_fusion.feature_extraction import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_ID,
    feature_code_names,
    load_feature_map,
)
from signal_fusion.fusion import CompleteRegion


HAS_TRAINING_RUNTIME = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "onnx", "onnxruntime")
)


class _FakeFeatureBackend:
    def extract_features(self, i_data, q_data, sample_rate):
        values = np.asarray(
            [
                np.mean(i_data),
                np.mean(q_data),
                np.std(i_data),
                np.std(q_data),
                sample_rate / 1e6,
            ],
            dtype=np.float32,
        )
        return np.resize(values, FEATURE_COUNT).astype(np.float32)


def _write_assembled_split(path: Path, split: str) -> None:
    x = np.ones((2, 2, 32), dtype=np.float32)
    np.savez_compressed(
        path,
        X=x,
        y=np.asarray([0, 1], dtype=np.int64),
        group_id=np.asarray([0, 1], dtype=np.int64),
        source_region_id=np.asarray([10, 11], dtype=np.int64),
        dataset_id=np.asarray("synthetic_iq"),
        split=np.asarray(split),
        label_map_json=np.asarray(json.dumps({"0": "A", "1": "B"})),
        source_id=np.asarray(f"synthetic_iq:{split}"),
    )


def _fake_region_loader(dataset_path, dataset, *, group_id):
    sample_count = 64 + int(group_id)
    phase = np.linspace(0.0, 4.0 * np.pi, sample_count, endpoint=False)
    samples = np.exp(1j * phase).astype(np.complex64)
    return CompleteRegion(
        samples=samples,
        sample_rate=4_000_000.0,
        start_sample=1000 * int(group_id),
        end_sample=1000 * int(group_id) + sample_count,
        source_id=f"source_{int(group_id)}",
        source_region_id=10 + int(group_id),
        raw_source_path="raw.dat",
        source_dataset_path="slices.npz",
    )


def _write_feature_split(
    path: Path,
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    feature_names = np.asarray(feature_code_names(load_feature_map()))
    count = len(labels)
    np.savez_compressed(
        path,
        features=features.astype(np.float32),
        y=labels.astype(np.int64),
        group_id=np.arange(count, dtype=np.int64),
        sample_source_id=np.asarray([f"{split}_{index}" for index in range(count)]),
        condition=np.full(count, "clean"),
        feature_names=feature_names,
        feature_schema_id=np.asarray(FEATURE_SCHEMA_ID),
        dataset_id=np.asarray("synthetic_region_features"),
        label_map_json=np.asarray(
            json.dumps({"0": "LoRa", "1": "Zigbee", "2": "BLE"})
        ),
    )


def _write_direct_region_split(path: Path) -> None:
    count = 8
    length = 32
    phase = np.linspace(0.0, 4.0 * np.pi, length, endpoint=False)
    x = np.empty((count, 2, length), dtype=np.float32)
    for index in range(count):
        x[index, 0] = np.cos(phase + index * 0.1)
        x[index, 1] = np.sin(phase + index * 0.1)
    starts = np.arange(count, dtype=np.int64) * length
    np.savez_compressed(
        path,
        X=x,
        y=np.repeat(np.arange(2, dtype=np.int64), 4),
        group_id=np.arange(count, dtype=np.int64),
        source_region_id=np.arange(count, dtype=np.int64),
        window_start_sample=starts,
        window_end_sample=starts + length,
        region_start_sample=starts,
        region_end_sample=starts + length,
        sample_source_id=np.repeat(np.asarray(["source_a", "source_b"]), 4),
        sample_rate=np.full(count, 1_000_000.0),
        dataset_id=np.asarray("direct_regions"),
        split=np.asarray("test"),
        label_map_json=np.asarray(json.dumps({"0": "A", "1": "B"})),
        source_id=np.asarray("direct_regions:test"),
    )


class RegionFeatureDatasetTests(unittest.TestCase):
    def test_builds_one_clean_and_configured_noisy_row_per_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "assembled"
            source.mkdir()
            for split in ("train", "validation", "test"):
                _write_assembled_split(source / f"{split}.npz", split)

            with patch(
                "signal_fusion.feature_classifier.dataset.load_complete_region",
                side_effect=_fake_region_loader,
            ):
                report = build_region_feature_dataset(
                    source,
                    root / "features",
                    train_awgn_copies=1,
                    test_awgn_snr=5.0,
                    feature_backend=_FakeFeatureBackend(),
                )

            train = load_region_feature_split(root / "features" / "train.npz")
            validation = load_region_feature_split(
                root / "features" / "validation.npz"
            )
            test = load_region_feature_split(root / "features" / "test.npz")
            self.assertEqual(train["features"].shape, (4, FEATURE_COUNT))
            self.assertEqual(validation["features"].shape, (2, FEATURE_COUNT))
            self.assertEqual(test["features"].shape, (4, FEATURE_COUNT))
            self.assertEqual(
                dict(zip(*np.unique(train["condition"], return_counts=True))),
                {"clean": 2, "train_awgn": 2},
            )
            self.assertEqual(report["splits"]["test"]["region_count"], 2)

    def test_extracts_uniform_regions_directly_from_single_window_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "test.npz"
            _write_direct_region_split(source_path)

            report = extract_region_feature_split(
                source_path,
                root / "features",
                split_name="test",
                regions_per_source=2,
                feature_backend=_FakeFeatureBackend(),
            )

            result = load_region_feature_split(root / "features" / "test.npz")
            self.assertEqual(result["features"].shape, (4, FEATURE_COUNT))
            np.testing.assert_array_equal(result["group_id"], [1, 3, 5, 7])
            self.assertEqual(
                dict(zip(*np.unique(result["sample_source_id"], return_counts=True))),
                {"source_a": 2, "source_b": 2},
            )
            self.assertEqual(report["source_count"], 2)
            self.assertEqual(report["region_count"], 4)


@unittest.skipUnless(HAS_TRAINING_RUNTIME, "training runtime is unavailable")
class FeatureClassifierTrainingTests(unittest.TestCase):
    def test_trained_onnx_service_reproduces_separable_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "features"
            dataset_dir.mkdir()
            for split, per_class in (("train", 12), ("validation", 4), ("test", 4)):
                labels = np.repeat(np.arange(3), per_class)
                features = np.zeros((labels.size, FEATURE_COUNT), dtype=np.float32)
                features[np.arange(labels.size), labels] = 5.0
                _write_feature_split(
                    dataset_dir / f"{split}.npz", split, features, labels
                )

            output_dir = root / "model"
            result = train_feature_classifier(
                dataset_dir,
                output_dir,
                device="cpu",
                learning_rate=0.05,
                max_epochs=80,
                patience=10,
            )
            test = load_region_feature_split(dataset_dir / "test.npz")
            service = FeatureClassifierService(
                output_dir / "feature_classifier_manifest.json"
            )
            prediction = service.predict(test["features"])

            np.testing.assert_array_equal(
                prediction.probabilities.argmax(axis=1), test["y"]
            )
            self.assertEqual(result["metrics"]["test"]["accuracy_percent"], 100.0)
            self.assertEqual(len(prediction.top_k()), 3)

            evaluation = evaluate_feature_classifier(
                dataset_dir / "test.npz",
                output_dir / "feature_classifier_manifest.json",
                root / "evaluation.json",
            )
            self.assertEqual(
                evaluation["metrics"]["window"]["accuracy_percent"], 100.0
            )
            self.assertTrue((root / "evaluation.json").is_file())


if __name__ == "__main__":
    unittest.main()
