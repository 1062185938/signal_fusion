import json
import tempfile
import unittest
from pathlib import Path

from signal_fusion.benchmarks.technology_recognition_hermes_review_runner import (
    build_review_prompt,
    extract_response_object,
    load_submission_cases,
    run_review_manifest,
    validate_response,
)
from signal_fusion.feature_extraction import feature_code_names, load_feature_map


LABELS = ("LTE", "WiFi", "DVB-T")
REQUIRED_OUTPUT = {
    "analysis_id": "exactly this input analysis_id",
    "decision": "keep, change, or abstain",
    "recommended_label": "LTE, WiFi, DVB-T, or null",
    "confidence_level": "medium or low",
    "summary": "concise Chinese blind adjudication summary",
    "iq_evidence": "list of IQ-branch observations",
    "feature_probe_evidence": "list of feature-probe observations",
    "physical_feature_evidence": (
        "list of raw-feature observations; empty when raw features are absent"
    ),
    "conflicting_evidence": "list of material conflicts",
    "limitations": "list of limitations",
}


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _branch(name, probabilities):
    order = sorted(LABELS, key=lambda label: probabilities[label], reverse=True)
    return {
        "branch_name": name,
        "probabilities": probabilities,
        "top3": [
            {"label": label, "probability": probabilities[label]}
            for label in order
        ],
        "top1": {
            "label": order[0],
            "probability": probabilities[order[0]],
        },
        "top1_top2_margin": probabilities[order[0]] - probabilities[order[1]],
        "normalized_entropy": 0.5,
    }


def _case(analysis_id, references, *, with_physical_features=False):
    value = {
        "schema_version": 2,
        "bundle_type": "signal_fusion_cross_location_review_input",
        "evidence_type": "blind_iq_feature_disagreement_adjudication",
        "analysis_id": analysis_id,
        "output_language": "zh-CN",
        "signal_context": {
            "effective_sample_rate_hz": 1_000_000.0,
            "sample_count": 4096,
            "duration_ms": 4.096,
        },
        "iq_branch": _branch(
            "iq_ensemble", {"LTE": 0.7, "WiFi": 0.1, "DVB-T": 0.2}
        ),
        "feature_probe_branch": _branch(
            "feature_probe", {"LTE": 0.2, "WiFi": 0.1, "DVB-T": 0.7}
        ),
        "reference_document_paths": {
            reference.name: str(reference) for reference in references
        },
        "required_output": REQUIRED_OUTPUT,
    }
    if with_physical_features:
        value["physical_feature_evidence"] = {
            "feature_schema_id": "matlab_iq_features_64_v2",
            "feature_count": 64,
            "scope": "complete_4096_sample_region",
            "values": {
                name: float(index)
                for index, name in enumerate(
                    feature_code_names(load_feature_map())
                )
            },
        }
    return value


def _response(analysis_id, *, decision="keep", label="LTE", physical=False):
    return {
        "analysis_id": analysis_id,
        "decision": decision,
        "recommended_label": label,
        "confidence_level": "medium" if decision != "abstain" else "low",
        "summary": "两条模型分支存在分歧，当前证据支持这一裁决。",
        "iq_evidence": ["IQ 分支支持 LTE。"],
        "feature_probe_evidence": ["特征探针分支支持 DVB-T。"],
        "physical_feature_evidence": ["物理特征支持改判。"] if physical else [],
        "conflicting_evidence": ["两个分支的标签不一致。"],
        "limitations": ["两个分支的置信度没有校准。"],
    }


def _make_references(root, count):
    root.mkdir(parents=True, exist_ok=True)
    names = [
        "technology_reference_lte_wifi_dvbt.md",
        "time_domain_iq_features.md",
        "frequency_domain_iq_features.md",
        "time_frequency_iq_features.md",
    ]
    references = []
    for name in names[:count]:
        path = root / name
        path.write_text("reference", encoding="utf-8")
        references.append(path)
    return references


def _write_paired_submission(root):
    references_a = _make_references(root / "references_a", 1)
    references_b = _make_references(root / "references_b", 4)
    first = root / "cases/set_a/review_0001.json"
    second = root / "cases/set_b/review_0002.json"
    _write_json(first, _case("review_0001", references_a))
    _write_json(
        second,
        _case("review_0002", references_b, with_physical_features=True),
    )
    manifest = root / "cases/submission_order.json"
    _write_json(
        manifest,
        {
            "schema_version": 2,
            "manifest_type": "blind_fusion_adjudication_submission",
            "case_count": 2,
            "signal_case_count": 1,
            "session_mode": "shared_resumable",
            "cases": [
                "set_a/review_0001.json",
                "set_b/review_0002.json",
            ],
        },
    )
    return manifest, first, second


class HermesReviewRunnerTests(unittest.TestCase):
    def test_extracts_fenced_json_before_session_id(self):
        response = _response("review_0001")
        stdout = (
            "```json\n"
            + json.dumps(response, ensure_ascii=False)
            + "\n```\n\nsession_id: session-123\n"
        )

        self.assertEqual(
            extract_response_object(stdout, "review_0001"), response
        )

    def test_response_decisions_follow_iq_label_and_variant_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_a = _case("review_0001", _make_references(root, 1))
            case_b = _case(
                "review_0002",
                _make_references(root / "b", 4),
                with_physical_features=True,
            )

            keep = validate_response(_response("review_0001"), case=case_a)
            self.assertEqual(keep["recommended_label"], "LTE")
            change = validate_response(
                _response(
                    "review_0002",
                    decision="change",
                    label="DVB-T",
                    physical=True,
                ),
                case=case_b,
            )
            self.assertEqual(change["recommended_label"], "DVB-T")
            abstain = _response(
                "review_0001", decision="abstain", label=None
            )
            self.assertIsNone(
                validate_response(abstain, case=case_a)["recommended_label"]
            )

            invalid_keep = _response("review_0001", label="DVB-T")
            with self.assertRaisesRegex(ValueError, "IQ top1"):
                validate_response(invalid_keep, case=case_a)
            invalid_high = _response("review_0001")
            invalid_high["confidence_level"] = "high"
            with self.assertRaisesRegex(ValueError, "medium or low"):
                validate_response(invalid_high, case=case_a)
            invalid_a = _response("review_0001", physical=True)
            with self.assertRaisesRegex(ValueError, "A-variant"):
                validate_response(invalid_a, case=case_a)

    def test_rejects_english_narrative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case("review_0001", _make_references(root, 1))
            response = _response("review_0001")
            response["summary"] = "Keep LTE."

            with self.assertRaisesRegex(ValueError, "Simplified Chinese"):
                validate_response(response, case=case)

    def test_validates_manifest_and_rejects_invalid_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, first_path, _ = _write_paired_submission(root)

            cases = load_submission_cases(manifest)
            self.assertEqual(
                [case["analysis_id"] for _, case in cases],
                ["review_0001", "review_0002"],
            )

            first = json.loads(first_path.read_text(encoding="utf-8"))
            first["true_label"] = "LTE"
            _write_json(first_path, first)
            with self.assertRaisesRegex(ValueError, "field mismatch"):
                load_submission_cases(manifest)

            first.pop("true_label")
            first["feature_probe_branch"] = first["iq_branch"]
            _write_json(first_path, first)
            with self.assertRaisesRegex(ValueError, "must disagree"):
                load_submission_cases(manifest)

    def test_rejects_manifest_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, _ = _write_paired_submission(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][0] = "../outside.json"
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "escapes"):
                load_submission_cases(manifest_path)

    def test_prompt_forbids_numeric_weighting_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            skill = project / "skills/signal-fusion-review-experiment/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("skill", encoding="utf-8")
            case = root / "review_0001.json"
            case.write_text("{}", encoding="utf-8")

            prompt = build_review_prompt(case, project)

            self.assertIn("不得对两分支概率做平均", prompt)
            self.assertIn("不是第三个独立投票", prompt)
            self.assertIn("不要沿用前一个 case", prompt)
            self.assertIn("keep、change 或 abstain", prompt)

    def test_runs_fake_hermes_in_one_session_and_resumes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            skill = project / "skills/signal-fusion-review-experiment/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("skill", encoding="utf-8")
            manifest_path, first_path, _ = _write_paired_submission(root)
            output = root / "responses"
            seen_sessions = []

            def fake_review(prompt, session_id):
                seen_sessions.append(session_id)
                analysis_id = (
                    "review_0001" if str(first_path) in prompt else "review_0002"
                )
                response = _response(analysis_id)
                stdout = (
                    json.dumps(response, ensure_ascii=False)
                    + "\n\nsession_id: shared-session\n"
                )
                return stdout, "shared-session", ""

            first = run_review_manifest(
                manifest_path,
                output,
                project_root=project,
                retries=0,
                review_callable=fake_review,
            )
            second = run_review_manifest(
                manifest_path,
                output,
                project_root=project,
                retries=0,
                review_callable=fake_review,
            )

            self.assertEqual(
                first["completed"], ["review_0001", "review_0002"]
            )
            self.assertEqual(seen_sessions, [None, "shared-session"])
            self.assertEqual(
                second["skipped"], ["review_0001", "review_0002"]
            )
            self.assertEqual(second["session_id"], "shared-session")
            saved = json.loads(
                (output / "review_0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved, _response("review_0001"))

    def test_cli_process_resumes_session_without_provider_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            skill = project / "skills/signal-fusion-review-experiment/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("skill", encoding="utf-8")
            manifest, _, _ = _write_paired_submission(root)
            invocation_log = root / "invocations.jsonl"
            fake_hermes = root / "hermes"
            fake_hermes.write_text(
                "#!/usr/bin/env python3\n"
                "import json, re, sys\n"
                f"log_path = {str(invocation_log)!r}\n"
                "args = sys.argv[1:]\n"
                "with open(log_path, 'a', encoding='utf-8') as stream:\n"
                "    stream.write(json.dumps(args) + '\\n')\n"
                "prompt = args[args.index('--query') + 1]\n"
                "analysis_id = re.search(r'review_\\d+', prompt).group(0)\n"
                "response = {\n"
                "  'analysis_id': analysis_id,\n"
                "  'decision': 'keep',\n"
                "  'recommended_label': 'LTE',\n"
                "  'confidence_level': 'medium',\n"
                "  'summary': '两条分支存在分歧，当前保留 IQ 标签。',\n"
                "  'iq_evidence': ['IQ 分支支持 LTE。'],\n"
                "  'feature_probe_evidence': ['特征探针支持 DVB-T。'],\n"
                "  'physical_feature_evidence': [],\n"
                "  'conflicting_evidence': ['两个分支的标签不一致。'],\n"
                "  'limitations': ['分支置信度没有校准。'],\n"
                "}\n"
                "print(json.dumps(response, ensure_ascii=False))\n"
                "print('\\nsession_id: shared-session')\n",
                encoding="utf-8",
            )
            fake_hermes.chmod(0o755)

            report = run_review_manifest(
                manifest,
                root / "responses",
                project_root=project,
                hermes_executable=fake_hermes,
                retries=0,
            )

            invocations = [
                json.loads(line)
                for line in invocation_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertNotIn("--resume", invocations[0])
            self.assertNotIn("--provider", invocations[0])
            self.assertIn("--pass-session-id", invocations[0])
            self.assertEqual(
                invocations[1][invocations[1].index("--resume") + 1],
                "shared-session",
            )
            self.assertEqual(report["session_id"], "shared-session")


if __name__ == "__main__":
    unittest.main()
