import json
import tempfile
import unittest
from pathlib import Path

from signal_fusion.benchmarks.technology_recognition_llm_review_evaluation import (
    evaluate_review_responses,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _response(analysis_id, status, label, *, feature=False, chinese=True):
    return {
        "analysis_id": analysis_id,
        "recommendation_status": status,
        "recommended_label": label,
        "confidence_level": "medium" if status == "recommend" else "low",
        "summary": "盲审结果" if chinese else "Blind review result",
        "model_evidence": ["模型证据" if chinese else "Model evidence"],
        "periodicity_evidence": [],
        "feature_evidence": (
            ["特征证据" if chinese else "Feature evidence"] if feature else []
        ),
        "conflicting_evidence": [],
        "limitations": [],
    }


class LlmReviewEvaluationTests(unittest.TestCase):
    def test_scores_paired_recommendations_and_language(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "responses"
            public = root / "public"
            responses.mkdir()
            audit = {
                "audit_type": "private_p6b_clean_review_pairing_audit",
                "pairs": [
                    {
                        "pair_index": 1,
                        "source_analysis_id": "case_0001",
                        "set_1_analysis_id": "review_0001",
                        "set_2_analysis_id": "review_0002",
                        "private_source_audit": {
                            "true_label": "LTE",
                            "prediction_label": "DVB-T",
                        },
                    }
                ],
            }
            audit_path = root / "audit.json"
            _write(audit_path, audit)
            _write(
                public / "set_1/review_0001.json",
                {"analysis_id": "review_0001"},
            )
            _write(
                public / "set_2/review_0002.json",
                {
                    "analysis_id": "review_0002",
                    "global_feature_evidence": {},
                },
            )
            _write(
                responses / "review_0001.json",
                _response("review_0001", "abstain", None),
            )
            _write(
                responses / "review_0002.json",
                _response(
                    "review_0002",
                    "recommend",
                    "LTE",
                    feature=True,
                    chinese=False,
                ),
            )

            report = evaluate_review_responses(responses, public, audit_path)

            a = report["variants"]["A_iq_and_periodicity"]
            b = report["variants"]["B_iq_periodicity_and_global_features"]
            self.assertEqual(a["recommendation"]["abstain_count"], 1)
            self.assertEqual(
                b["error_effect"]["corrected_provisional_error_count"], 1
            )
            self.assertEqual(report["language"]["non_chinese_narrative_count"], 1)
            self.assertEqual(
                report["paired_comparison"]["outcome_counts"],
                {"a_abstain__b_correct": 1},
            )

    def test_rejects_feature_claim_from_feature_absent_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "audit.json"
            _write(
                audit_path,
                {
                    "audit_type": "private_p6b_clean_review_pairing_audit",
                    "pairs": [
                        {
                            "pair_index": 1,
                            "source_analysis_id": "case_0001",
                            "set_1_analysis_id": "review_0001",
                            "set_2_analysis_id": "review_0002",
                            "private_source_audit": {
                                "true_label": "LTE",
                                "prediction_label": "LTE",
                            },
                        }
                    ],
                },
            )
            _write(
                root / "public/set_1/review_0001.json",
                {"analysis_id": "review_0001"},
            )
            _write(
                root / "public/set_2/review_0002.json",
                {
                    "analysis_id": "review_0002",
                    "global_feature_evidence": {},
                },
            )
            _write(
                root / "responses/review_0001.json",
                _response("review_0001", "recommend", "LTE", feature=True),
            )
            _write(
                root / "responses/review_0002.json",
                _response("review_0002", "recommend", "LTE", feature=True),
            )

            with self.assertRaisesRegex(ValueError, "without global features"):
                evaluate_review_responses(
                    root / "responses", root / "public", audit_path
                )


if __name__ == "__main__":
    unittest.main()
