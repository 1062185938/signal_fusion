import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from signal_fusion.benchmarks.technology_recognition_hermes_review_runner import (
    load_submission_cases,
)
from signal_fusion.benchmarks.technology_recognition_llm_review import (
    build_adjudication_case,
    build_review_experiment_files,
    select_adjudication_rows,
)
from signal_fusion.feature_extraction import feature_code_names, load_feature_map


def _probabilities():
    iq = np.asarray(
        [
            [0.60, 0.40, 0.00],
            [0.80, 0.20, 0.00],
        ],
        dtype=np.float64,
    )
    feature = np.asarray(
        [
            [0.20, 0.80, 0.00],
            [0.30, 0.70, 0.00],
        ],
        dtype=np.float64,
    )
    return iq, feature


class LlmReviewCaseBuilderTests(unittest.TestCase):
    def test_selection_uses_only_predictions_and_balances_roles(self):
        iq, feature = _probabilities()
        selected = select_adjudication_rows(
            iq,
            feature,
            np.asarray(["fold1", "fold1"]),
            random_seed=44,
        )

        np.testing.assert_array_equal(selected["proposal_indices"], [0])
        np.testing.assert_array_equal(selected["shadow_indices"], [1])
        self.assertEqual(selected["same_fold"].tolist(), [True])

    def test_public_variants_have_no_private_metadata(self):
        iq, feature = _probabilities()
        names = feature_code_names(load_feature_map())
        set_a = build_adjudication_case(
            "review_0001", iq[0], feature[0], include_physical_features=False
        )
        set_b = build_adjudication_case(
            "review_0002",
            iq[0],
            feature[0],
            include_physical_features=True,
            physical_features=np.arange(64, dtype=np.float32),
            feature_names=names,
        )

        forbidden = {
            "true_label",
            "selection_role",
            "fold",
            "location",
            "sample_source_id",
            "source_region_id",
            "global_group_id",
            "equal_fusion_label",
        }
        self.assertFalse(forbidden & set_a.keys())
        self.assertFalse(forbidden & set_b.keys())
        self.assertNotIn("physical_feature_evidence", set_a)
        self.assertEqual(set_b["physical_feature_evidence"]["feature_count"], 64)
        self.assertEqual(set_a["output_language"], "zh-CN")
        self.assertEqual(set_a["iq_branch"]["branch_name"], "iq_model")

    def test_writes_paired_cases_manifest_and_private_audit(self):
        iq, feature = _probabilities()
        names = np.asarray(feature_code_names(load_feature_map()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "predictions.npz"
            np.savez_compressed(
                prediction_path,
                fold=np.asarray(["fold1", "fold1"]),
                location=np.asarray(["UZ", "UZ"]),
                sample_source_id=np.asarray(["source-a", "source-b"]),
                source_region_id=np.asarray([10, 20]),
                global_group_id=np.asarray([0, 1]),
                y=np.asarray([1, 0]),
                iq_model_probabilities=iq,
                feature_probe_probabilities=feature,
                equal_weight_fusion_probabilities=(iq + feature) * 0.5,
            )
            feature_bank = {
                "features": np.arange(128, dtype=np.float32).reshape(2, 64),
                "feature_names": names,
                "sample_rate": np.asarray([1_000_000.0, 1_000_000.0]),
                "row_lookup": {("source-a", 10): 0, ("source-b", 20): 1},
            }
            public = root / "public"
            audit_path = root / "private" / "audit.json"
            with patch(
                "signal_fusion.benchmarks.technology_recognition_llm_review."
                "load_full_feature_bank",
                return_value=feature_bank,
            ):
                result = build_review_experiment_files(
                    prediction_path,
                    root / "datasets",
                    root / "features",
                    public,
                    audit_path,
                )

            self.assertEqual(result["signal_case_count"], 2)
            self.assertEqual(result["public_case_count"], 4)
            cases = load_submission_cases(public / "submission_order.json")
            self.assertEqual(len(cases), 4)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {case["selection_role"] for case in audit["cases"]},
                {"candidate", "shadow"},
            )
            self.assertTrue(audit["selection"]["selection_uses_ground_truth"] is False)
            for _, public_case in cases:
                self.assertNotIn("true_label", public_case)
                self.assertNotIn("selection_role", public_case)


if __name__ == "__main__":
    unittest.main()
