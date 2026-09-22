import unittest

from signal_fusion.benchmarks.technology_recognition_llm_review import (
    FEATURE_REFERENCE_DOCUMENTS,
    PERIODICITY_REFERENCE_DOCUMENT,
    TECHNOLOGY_REFERENCE_DOCUMENT,
    build_review_experiment_case,
)
from signal_fusion.feature_extraction import asset_path
from signal_fusion.fusion.references import (
    periodicity_reference_path,
    technology_reference_path,
)


def _source_case():
    feature_paths = {
        name: str(asset_path(name).resolve())
        for name in FEATURE_REFERENCE_DOCUMENTS
    }
    feature_paths[TECHNOLOGY_REFERENCE_DOCUMENT] = str(
        technology_reference_path().resolve()
    )
    return {
        "schema_version": 3,
        "bundle_type": "hermes_signal_fusion_input",
        "signal_context": {"effective_sample_rate_hz": 1_000_000.0},
        "iq_ensemble_evidence": {
            "risk_gate": {"unanimous": False},
            "members": [{}, {}, {}],
        },
        "periodicity_evidence": {
            "frozen_gate": {"resolved": False, "changed": False},
            "reference_document": PERIODICITY_REFERENCE_DOCUMENT,
            "reference_document_path": str(periodicity_reference_path().resolve()),
        },
        "deterministic_fusion_result": {
            "decision_status": "review_required",
            "resolution": "unresolved_review",
            "final_label": None,
            "provisional_label": "LTE",
            "review_reason": "periodicity_below_threshold",
        },
        "global_feature_evidence": {
            "feature_count": 64,
            "values": {f"feature_{index}": float(index) for index in range(64)},
            "reference_documents": [
                *FEATURE_REFERENCE_DOCUMENTS,
                TECHNOLOGY_REFERENCE_DOCUMENT,
            ],
            "reference_document_paths": feature_paths,
        },
    }


class LlmReviewExperimentTests(unittest.TestCase):
    def test_paired_cases_differ_only_by_optional_feature_evidence(self):
        source = _source_case()

        without_features = build_review_experiment_case(
            source,
            "review_0001",
            include_global_features=False,
        )
        with_features = build_review_experiment_case(
            source,
            "review_0002",
            include_global_features=True,
        )

        self.assertNotIn("global_feature_evidence", without_features)
        self.assertIn("global_feature_evidence", with_features)
        self.assertEqual(
            set(without_features["reference_document_paths"]),
            {TECHNOLOGY_REFERENCE_DOCUMENT, PERIODICITY_REFERENCE_DOCUMENT},
        )
        self.assertEqual(len(with_features["reference_document_paths"]), 5)
        self.assertNotIn("deterministic_fusion_result", without_features)
        self.assertNotIn("final_label", str(without_features))
        self.assertEqual(
            without_features["review_context"]["provisional_label"], "LTE"
        )
        left = dict(without_features)
        right = dict(with_features)
        left.pop("analysis_id")
        right.pop("analysis_id")
        right.pop("global_feature_evidence")
        left["reference_document_paths"] = {
            name: path
            for name, path in left["reference_document_paths"].items()
        }
        right["reference_document_paths"] = {
            name: path
            for name, path in right["reference_document_paths"].items()
            if name in left["reference_document_paths"]
        }
        self.assertEqual(left, right)

    def test_rejects_a_resolved_source_case(self):
        source = _source_case()
        source["periodicity_evidence"]["frozen_gate"]["resolved"] = True

        with self.assertRaisesRegex(ValueError, "unresolved"):
            build_review_experiment_case(
                source,
                "review_0001",
                include_global_features=False,
            )


if __name__ == "__main__":
    unittest.main()
