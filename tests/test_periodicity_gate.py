import unittest

import numpy as np

from signal_fusion import PreparedDataset
from signal_fusion.fusion.periodicity_gate import (
    apply_periodicity_gate,
    lte_dvbt_disagreement_mask,
    select_periodicity_gate_margin,
    score_technology_periodicity,
    summarize_periodicity_gate,
)


class PeriodicityGateTests(unittest.TestCase):
    def test_production_scorer_does_not_require_ground_truth(self):
        rng = np.random.default_rng(2)
        block = (
            rng.standard_normal(67) + 1j * rng.standard_normal(67)
        ).astype(np.complex64)
        signal = np.tile(block, 62)[:4096]
        dataset = PreparedDataset(
            X=np.stack((signal.real, signal.imag), axis=0)[np.newaxis].astype(
                np.float32
            ),
            source_id="unlabeled",
            meta={"sample_rate": np.asarray([1_000_000.0])},
        )

        scores = score_technology_periodicity(dataset, batch_size=1)

        self.assertEqual(scores["prediction"].shape, (1,))
        self.assertEqual(scores["margin"].shape, (1,))
        self.assertGreater(float(scores["lte_score"][0]), 0.9)

    def test_only_lte_dvbt_member_conflicts_are_eligible(self):
        members = np.asarray(
            [
                [0, 0, 0],
                [0, 2, 2],
                [0, 1, 2],
                [0, 2, 2],
            ]
        )
        ensemble = np.asarray([0, 2, 1, 1])

        mask = lte_dvbt_disagreement_mask(members, ensemble)

        np.testing.assert_array_equal(mask, [False, True, False, False])

    def test_validation_threshold_rejects_eligible_periodicity_errors(self):
        truth = np.asarray([2, 2, 0, 1])
        members = np.asarray(
            [
                [0, 0, 2],
                [0, 2, 2],
                [0, 2, 2],
                [0, 1, 2],
            ]
        )
        ensemble = np.asarray([0, 2, 2, 1])
        periodicity = np.asarray([2, 0, 0, 0])
        margins = np.asarray([0.5, 0.2, 0.4, 0.9])

        threshold = select_periodicity_gate_margin(
            truth,
            periodicity,
            margins,
        )
        gated = apply_periodicity_gate(
            members,
            ensemble,
            periodicity,
            margins,
            threshold,
        )

        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.4)
        np.testing.assert_array_equal(gated["resolved"], [True, False, True, False])
        np.testing.assert_array_equal(gated["prediction"], [2, 2, 0, 1])

    def test_gate_reports_net_corrections_and_protects_wifi(self):
        truth = np.asarray([0, 2, 0, 1])
        members = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 2],
                [0, 2, 2],
                [0, 1, 2],
            ]
        )
        ensemble = np.asarray([0, 0, 2, 1])
        periodicity = np.asarray([2, 2, 0, 0])
        margins = np.asarray([0.9, 0.6, 0.7, 0.9])

        gated = apply_periodicity_gate(
            members,
            ensemble,
            periodicity,
            margins,
            reliability_margin=0.5,
        )
        summary = summarize_periodicity_gate(
            truth,
            members,
            ensemble,
            gated,
        )

        np.testing.assert_array_equal(gated["prediction"], [0, 2, 0, 1])
        self.assertEqual(summary["corrected_error_count"], 2)
        self.assertEqual(summary["introduced_error_count"], 0)
        self.assertEqual(summary["net_correction_count"], 2)
        self.assertEqual(summary["fused"]["accuracy_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
