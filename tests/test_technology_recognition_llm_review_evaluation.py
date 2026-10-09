import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from signal_fusion.benchmarks.technology_recognition_llm_review_evaluation import (
    _variant_metrics,
    evaluate_review_responses,
    write_review_evaluation,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _response(analysis_id, decision, label):
    return {
        "analysis_id": analysis_id,
        "decision": decision,
        "recommended_label": label,
        "confidence_level": "low" if decision == "abstain" else "medium",
        "summary": "盲测判决",
        "iq_evidence": [],
        "feature_probe_evidence": [],
        "physical_feature_evidence": [],
        "conflicting_evidence": [],
        "limitations": [],
    }


def _audit():
    return {
        "schema_version": 2,
        "audit_type": "private_cross_location_llm_adjudication_audit",
        "population_summary": {
            "region_count": 100,
            "iq_correct_count": 90,
            "iq_error_count": 10,
            "candidate_count": 1,
            "shadow_count": 1,
        },
        "cases": [
            {
                "pair_index": 1,
                "signal_case_id": "signal_0001",
                "selection_role": "candidate",
                "set_a_analysis_id": "review_0001",
                "set_b_analysis_id": "review_0002",
                "true_label": "LTE",
                "iq_label": "DVB-T",
                "feature_label": "LTE",
                "equal_fusion_label": "LTE",
            },
            {
                "pair_index": 2,
                "signal_case_id": "signal_0002",
                "selection_role": "shadow",
                "set_a_analysis_id": "review_0003",
                "set_b_analysis_id": "review_0004",
                "true_label": "WiFi",
                "iq_label": "WiFi",
                "feature_label": "DVB-T",
                "equal_fusion_label": "WiFi",
            },
        ],
    }


class LlmReviewEvaluationTests(unittest.TestCase):
    def test_selective_accuracy_is_none_when_every_case_abstains(self):
        metrics = _variant_metrics(
            [
                {
                    "decision": "abstain",
                    "recommended_label": None,
                    "true_label": "LTE",
                    "iq_label": "DVB-T",
                    "feature_label": "LTE",
                    "equal_fusion_label": "LTE",
                    "operational_label": "DVB-T",
                }
            ]
        )

        selective = metrics["selective_adjudication"]
        self.assertEqual(selective["covered_count"], 0)
        self.assertEqual(selective["coverage_percent"], 0.0)
        self.assertIsNone(selective["accuracy_percent"])

    def test_scores_branches_effects_roles_projection_and_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "audit.json"
            _write(audit_path, _audit())
            for variant, identifiers in {
                "set_a": ("review_0001", "review_0003"),
                "set_b": ("review_0002", "review_0004"),
            }.items():
                for analysis_id in identifiers:
                    _write(
                        root / "public" / variant / f"{analysis_id}.json",
                        {"analysis_id": analysis_id},
                    )
            responses = {
                "review_0001": _response("review_0001", "change", "LTE"),
                "review_0002": _response("review_0002", "abstain", None),
                "review_0003": _response("review_0003", "keep", "WiFi"),
                "review_0004": _response("review_0004", "change", "DVB-T"),
            }
            for analysis_id, response in responses.items():
                _write(root / "responses" / f"{analysis_id}.json", response)

            with patch(
                "signal_fusion.benchmarks."
                "technology_recognition_llm_review_evaluation.validate_response",
                side_effect=lambda response, *, case: dict(response),
            ):
                report = evaluate_review_responses(
                    root / "responses", root / "public", audit_path
                )

            a = report["variants"]["A_branch_outputs_only"]
            b = report["variants"]["B_branch_outputs_and_64_features"]
            self.assertEqual(
                a["branch_accuracy"]["llm_operational_with_abstain_fallback"][
                    "correct_count"
                ],
                2,
            )
            self.assertEqual(a["effect_vs_iq"]["corrected_iq_error_count"], 1)
            self.assertEqual(a["effect_vs_iq"]["harmed_iq_correct_count"], 0)
            self.assertEqual(a["change_quality"]["change_precision_percent"], 100.0)
            self.assertEqual(a["selective_adjudication"]["covered_count"], 2)
            self.assertEqual(a["selective_adjudication"]["accuracy_percent"], 100.0)
            self.assertEqual(
                a["candidate_population_projection"]["projected_correct_count"],
                91,
            )
            self.assertEqual(
                a["candidate_population_projection"]["projected_accuracy_percent"],
                91.0,
            )
            self.assertEqual(b["decisions"]["abstain_count"], 1)
            self.assertEqual(b["selective_adjudication"]["covered_count"], 1)
            self.assertEqual(b["selective_adjudication"]["coverage_percent"], 50.0)
            self.assertEqual(b["selective_adjudication"]["accuracy_percent"], 0.0)
            self.assertEqual(b["effect_vs_iq"]["harmed_iq_correct_count"], 1)
            self.assertEqual(
                b["by_selection_role"]["shadow"]["effect_vs_iq"][
                    "harmed_iq_correct_count"
                ],
                1,
            )
            self.assertEqual(
                report["paired_comparison"]["operational_outcome_counts"],
                {"a_correct__b_wrong": 2},
            )

    def test_writes_only_json_and_csv(self):
        report = {
            "variants": {
                name: {
                    "case_count": 0,
                    "branch_accuracy": {
                        key: {"accuracy_percent": None}
                        for key in (
                            "iq_baseline",
                            "feature_probe",
                            "equal_fusion",
                            "llm_operational_with_abstain_fallback",
                        )
                    },
                    "decisions": {
                        "keep_count": 0,
                        "change_count": 0,
                        "abstain_count": 0,
                    },
                    "effect_vs_iq": {
                        "corrected_iq_error_count": 0,
                        "harmed_iq_correct_count": 0,
                        "net_correction_count": 0,
                    },
                    "change_quality": {"change_precision_percent": None},
                    "by_selection_role": {},
                }
                for name in (
                    "A_branch_outputs_only",
                    "B_branch_outputs_and_64_features",
                )
            }
        }
        for metrics in report["variants"].values():
            metrics["by_selection_role"] = {
                role: {
                    "case_count": 0,
                    "branch_accuracy": metrics["branch_accuracy"],
                    "decisions": metrics["decisions"],
                    "effect_vs_iq": metrics["effect_vs_iq"],
                    "change_quality": metrics["change_quality"],
                }
                for role in ("candidate", "shadow")
            }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = write_review_evaluation(report, output)

            self.assertEqual(set(paths), {"json_path", "csv_path"})
            self.assertFalse((output / "llm_review_evaluation.md").exists())
            with Path(paths["csv_path"]).open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)

    def test_rejects_audit_count_mismatch(self):
        audit = _audit()
        audit["population_summary"]["candidate_count"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "audit.json", audit)
            with self.assertRaisesRegex(ValueError, "candidate_count"):
                evaluate_review_responses(
                    root / "responses", root / "public", root / "audit.json"
                )


if __name__ == "__main__":
    unittest.main()
