import unittest

import numpy as np

from signal_fusion.benchmarks.technology_recognition_feature_complementarity import (
    select_matched_correct_controls,
    summarize_fold_complementarity,
    uniform_indices_per_source,
)


class TechnologyRecognitionFeatureComplementarityTests(unittest.TestCase):
    def test_uniform_selection_is_balanced_by_source(self):
        sources = np.asarray(["a"] * 10 + ["b"] * 10)
        selected = uniform_indices_per_source(sources, 3)

        self.assertEqual(selected.tolist(), [1, 5, 8, 11, 15, 18])
        self.assertEqual(np.count_nonzero(sources[selected] == "a"), 3)
        self.assertEqual(np.count_nonzero(sources[selected] == "b"), 3)

    def test_controls_match_error_source_and_count(self):
        labels = np.asarray([2] * 6 + [1] * 6)
        sources = np.asarray(["dvbt"] * 6 + ["wifi"] * 6)
        predictions = labels.copy()
        predictions[[0, 2, 7]] = [1, 1, 2]

        controls = select_matched_correct_controls(labels, sources, predictions)

        self.assertEqual(controls.size, 3)
        self.assertEqual(np.count_nonzero(sources[controls] == "dvbt"), 2)
        self.assertEqual(np.count_nonzero(sources[controls] == "wifi"), 1)
        np.testing.assert_array_equal(predictions[controls], labels[controls])

    def test_summary_separates_rescue_from_control_harm(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        iq = np.asarray([0, 2, 1, 0, 1, 2])
        feature = np.asarray([0, 1, 2, 2, 1, 0])
        uniform = np.ones(6, dtype=bool)
        errors = np.asarray([False, True, True, False, False, False])
        controls = np.asarray([True, False, False, True, False, False])

        result = summarize_fold_complementarity(
            labels, iq, feature, uniform, errors, controls
        )

        self.assertEqual(
            result["iq_error_diagnostic"]["feature_rescued_count"], 2
        )
        self.assertEqual(
            result["matched_correct_control"]["feature_wrong_count"], 1
        )
        self.assertAlmostEqual(
            result["uniform_cross_location"]["feature_probe"][
                "accuracy_percent"
            ],
            4 / 6 * 100.0,
        )


if __name__ == "__main__":
    unittest.main()
