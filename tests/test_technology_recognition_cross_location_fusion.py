import unittest

import numpy as np

from signal_fusion.benchmarks.technology_recognition_cross_location_fusion import (
    apply_selective_gate,
    disagreement_router_metrics,
    select_safe_fusion_weight,
    select_safe_gate_threshold,
)


class TechnologyRecognitionCrossLocationFusionTests(unittest.TestCase):
    def test_weight_tie_keeps_iq_only(self):
        iq = np.asarray(
            [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        )
        feature = iq.copy()

        selected, _ = select_safe_fusion_weight(
            iq, feature, np.asarray([0, 1, 2])
        )

        self.assertEqual(selected, 1.0)

    def test_gate_tie_avoids_unnecessary_override(self):
        iq = np.asarray([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1]])
        feature = np.asarray([[0.2, 0.8, 0.0], [0.8, 0.1, 0.1]])
        targets = np.asarray([0, 1])

        selected, _ = select_safe_gate_threshold(iq, feature, targets)
        fused = apply_selective_gate(iq, feature, selected)

        self.assertIsNone(selected)
        np.testing.assert_array_equal(fused, iq)

    def test_disagreement_router_counts_rescue_and_harm_opportunities(self):
        iq = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
            ]
        )
        feature = np.asarray(
            [
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.1, 0.2, 0.7],
                [0.1, 0.2, 0.7],
            ]
        )
        targets = np.asarray([1, 1, 2, 1])

        metrics = disagreement_router_metrics(iq, feature, targets)

        self.assertEqual(metrics["review_count"], 3)
        self.assertEqual(metrics["reviewed_iq_error_count"], 2)
        self.assertEqual(metrics["iq_wrong_feature_correct_count"], 1)
        self.assertEqual(metrics["iq_correct_feature_wrong_count"], 1)
        self.assertEqual(metrics["both_wrong_count"], 1)


if __name__ == "__main__":
    unittest.main()
