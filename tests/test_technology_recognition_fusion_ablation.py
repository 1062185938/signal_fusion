import unittest

import numpy as np

from signal_fusion.benchmarks.technology_recognition_fusion_ablation import (
    summarize_fusion_ablation,
)


def _clean_fixture():
    return {
        "y": np.asarray([0, 0, 2, 2]),
        "member_predictions": np.asarray(
            [[0, 0, 0], [2, 0, 2], [0, 2, 2], [2, 2, 2]]
        ),
        "ensemble_prediction": np.asarray([0, 2, 0, 2]),
        "gate_eligible": np.asarray([False, True, True, False]),
        "gate_resolved": np.asarray([False, True, False, False]),
        "label_changed": np.asarray([False, True, False, False]),
        "fused_prediction": np.asarray([0, 0, 0, 2]),
    }


class FusionAblationTests(unittest.TestCase):
    def test_separates_full_accuracy_from_reject_coverage(self):
        clean = _clean_fixture()
        awgn = {
            name: np.concatenate((values, values), axis=0)
            for name, values in clean.items()
        }
        awgn["snr_db"] = np.full(8, 5.0)
        awgn["noise_seed"] = np.repeat([44, 45], 4)

        summary = summarize_fusion_ablation(clean, awgn)

        self.assertEqual(
            [item["condition_id"] for item in summary["conditions"]],
            ["clean", "awgn_5_db"],
        )
        clean_summary = summary["conditions"][0]["aggregate"]
        self.assertEqual(
            clean_summary["stages"]["iq_ensemble"][
                "full_coverage_accuracy_percent"
            ]["mean"],
            50.0,
        )
        self.assertEqual(
            clean_summary["stages"]["iq_ensemble_plus_periodicity"][
                "full_coverage_accuracy_percent"
            ]["mean"],
            75.0,
        )
        rejected = clean_summary["stages"][
            "iq_ensemble_plus_periodicity_with_reject"
        ]
        self.assertIsNone(rejected["full_coverage_accuracy_percent"])
        self.assertEqual(rejected["coverage_percent"]["mean"], 75.0)
        self.assertEqual(rejected["accepted_accuracy_percent"]["mean"], 100.0)
        self.assertEqual(
            clean_summary["gate_activity"]["corrected_error_count"]["mean"],
            1.0,
        )
        self.assertEqual(
            clean_summary["gate_activity"]["review_required_count"]["mean"],
            1.0,
        )

    def test_rejects_inconsistent_changed_flag(self):
        clean = _clean_fixture()
        clean["label_changed"] = np.zeros(4, dtype=bool)
        awgn = {
            name: np.asarray(values).copy() for name, values in _clean_fixture().items()
        }
        awgn["snr_db"] = np.full(4, 5.0)
        awgn["noise_seed"] = np.full(4, 44)

        with self.assertRaisesRegex(ValueError, "label_changed"):
            summarize_fusion_ablation(clean, awgn)


if __name__ == "__main__":
    unittest.main()
