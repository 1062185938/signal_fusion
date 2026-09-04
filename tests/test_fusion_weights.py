import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion.fusion.selection import select_fusion_weight
from signal_fusion.fusion.weights import FusionManifest


class FusionWeightTests(unittest.TestCase):
    def test_manifest_loads_and_combines_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "method": "weighted_probability_average",
                        "labels": ["A", "B"],
                        "weights": {
                            "iq_model": 0.6,
                            "feature_classifier": 0.4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = FusionManifest.load(path)

        combined = manifest.combine(
            np.asarray([0.8, 0.2]),
            np.asarray([0.3, 0.7]),
        )
        np.testing.assert_allclose(combined, [[0.6, 0.4]])

    def test_invalid_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            FusionManifest(
                labels=("A", "B"),
                iq_model_weight=0.7,
                feature_classifier_weight=0.7,
            )

    def test_probability_shapes_must_match(self):
        manifest = FusionManifest(
            labels=("A", "B"),
            iq_model_weight=0.5,
            feature_classifier_weight=0.5,
        )
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            manifest.combine(
                np.asarray([[0.8, 0.2], [0.4, 0.6]]),
                np.asarray([[0.3, 0.7]]),
            )

    def test_selection_uses_validation_accuracy_then_nearest_equal_weight(self):
        targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
        iq = np.asarray(
            [
                [0.01, 0.99],
                [0.1, 0.9],
                [0.01, 0.99],
                [0.1, 0.9],
            ]
        )
        feature = np.asarray(
            [
                [0.7, 0.3],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.2, 0.8],
            ]
        )

        selected, candidates = select_fusion_weight(
            iq,
            feature,
            targets,
            ("A", "B"),
            weight_step=0.1,
        )

        self.assertEqual(len(candidates), 11)
        self.assertAlmostEqual(selected, 0.2)

    def test_equal_predictions_select_equal_weights(self):
        probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9]])
        selected, _ = select_fusion_weight(
            probabilities,
            probabilities,
            np.asarray([0, 1]),
            ("A", "B"),
        )
        self.assertEqual(selected, 0.5)


if __name__ == "__main__":
    unittest.main()
