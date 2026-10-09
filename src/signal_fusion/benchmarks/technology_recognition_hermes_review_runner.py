"""Run blind cross-location fusion adjudication cases through Hermes."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from signal_fusion.feature_extraction import feature_code_names, load_feature_map


LABELS = ("LTE", "WiFi", "DVB-T")
LIST_FIELDS = (
    "iq_evidence",
    "feature_probe_evidence",
    "physical_feature_evidence",
    "conflicting_evidence",
    "limitations",
)
RESPONSE_FIELDS = {
    "analysis_id",
    "decision",
    "recommended_label",
    "confidence_level",
    "summary",
    *LIST_FIELDS,
}
BUNDLE_TYPE = "signal_fusion_cross_location_review_input"
EVIDENCE_TYPE = "blind_iq_feature_disagreement_adjudication"
MANIFEST_TYPE = "blind_fusion_adjudication_submission"
FEATURE_SCHEMA_ID = "matlab_iq_features_64_v2"
FEATURE_COUNT = 64
REFERENCE_NAMES_A = {"technology_reference_lte_wifi_dvbt.md"}
REFERENCE_NAMES_B = REFERENCE_NAMES_A | {
    "time_domain_iq_features.md",
    "frequency_domain_iq_features.md",
    "time_frequency_iq_features.md",
}
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
_ANALYSIS_ID_RE = re.compile(r"review_\d+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_SESSION_ID_RE = re.compile(r"^session_id:\s*(\S+)\s*$", re.MULTILINE)


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _finite_probability(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be finite and between 0 and 1")
    return number


def _validate_branch(branch: object, field: str) -> dict[str, Any]:
    if not isinstance(branch, dict):
        raise ValueError(f"{field} must be an object")
    expected_fields = {
        "branch_name",
        "probabilities",
        "top3",
        "top1",
        "top1_top2_margin",
        "normalized_entropy",
    }
    if set(branch) != expected_fields:
        raise ValueError(f"{field} fields do not match schema_version 2")
    if not isinstance(branch["branch_name"], str) or not branch["branch_name"]:
        raise ValueError(f"{field}.branch_name must be a non-empty string")

    probabilities = branch["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
        raise ValueError(f"{field}.probabilities must contain exactly {LABELS}")
    normalized_probabilities = {
        label: _finite_probability(
            probabilities[label], f"{field}.probabilities.{label}"
        )
        for label in LABELS
    }
    if not math.isclose(
        sum(normalized_probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError(f"{field}.probabilities must sum to 1")

    expected_order = sorted(
        LABELS, key=lambda label: normalized_probabilities[label], reverse=True
    )
    top3 = branch["top3"]
    if not isinstance(top3, list) or len(top3) != len(LABELS):
        raise ValueError(f"{field}.top3 must contain all three labels")
    for index, (entry, expected_label) in enumerate(zip(top3, expected_order)):
        if not isinstance(entry, dict) or set(entry) != {"label", "probability"}:
            raise ValueError(f"{field}.top3[{index}] has invalid fields")
        if entry["label"] != expected_label:
            raise ValueError(f"{field}.top3 must be sorted by probability")
        probability = _finite_probability(
            entry["probability"], f"{field}.top3[{index}].probability"
        )
        if not math.isclose(
            probability,
            normalized_probabilities[expected_label],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{field}.top3 conflicts with probabilities")

    top1 = branch["top1"]
    if not isinstance(top1, dict) or set(top1) != {"label", "probability"}:
        raise ValueError(f"{field}.top1 has invalid fields")
    if top1["label"] != expected_order[0]:
        raise ValueError(f"{field}.top1 conflicts with probabilities")
    top1_probability = _finite_probability(
        top1["probability"], f"{field}.top1.probability"
    )
    if not math.isclose(
        top1_probability,
        normalized_probabilities[expected_order[0]],
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{field}.top1 probability conflicts with probabilities")

    margin = _finite_probability(
        branch["top1_top2_margin"], f"{field}.top1_top2_margin"
    )
    expected_margin = (
        normalized_probabilities[expected_order[0]]
        - normalized_probabilities[expected_order[1]]
    )
    if not math.isclose(margin, expected_margin, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{field}.top1_top2_margin is inconsistent")
    _finite_probability(branch["normalized_entropy"], f"{field}.normalized_entropy")
    return branch


def _validate_physical_features(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("physical_feature_evidence must be an object")
    expected_fields = {
        "feature_schema_id",
        "feature_count",
        "scope",
        "values",
    }
    if set(value) != expected_fields:
        raise ValueError("physical_feature_evidence fields do not match schema")
    if value["feature_schema_id"] != FEATURE_SCHEMA_ID:
        raise ValueError("unexpected physical feature schema")
    if value["feature_count"] != FEATURE_COUNT:
        raise ValueError(f"physical feature_count must be {FEATURE_COUNT}")
    if not isinstance(value["scope"], str) or not value["scope"]:
        raise ValueError("physical feature scope must be a non-empty string")
    features = value["values"]
    if not isinstance(features, dict) or len(features) != FEATURE_COUNT:
        raise ValueError(f"physical feature values must contain {FEATURE_COUNT} entries")
    if set(features) != set(feature_code_names(load_feature_map())):
        raise ValueError("physical feature names do not match the 64-feature schema")
    for name, feature_value in features.items():
        if not isinstance(name, str) or not name:
            raise ValueError("physical feature names must be non-empty strings")
        if (
            not isinstance(feature_value, (int, float))
            or isinstance(feature_value, bool)
            or not math.isfinite(float(feature_value))
        ):
            raise ValueError(f"physical feature {name} must be finite numeric")
    return value


def _validate_case(path: Path) -> dict[str, Any]:
    case = _load_json_object(path)
    common_fields = {
        "schema_version",
        "bundle_type",
        "evidence_type",
        "analysis_id",
        "output_language",
        "signal_context",
        "iq_branch",
        "feature_probe_branch",
        "reference_document_paths",
        "required_output",
    }
    has_physical_features = "physical_feature_evidence" in case
    expected_fields = common_fields | (
        {"physical_feature_evidence"} if has_physical_features else set()
    )
    if set(case) != expected_fields:
        missing = sorted(expected_fields - set(case))
        extra = sorted(set(case) - expected_fields)
        raise ValueError(
            f"case field mismatch in {path}; missing={missing}, extra={extra}"
        )
    analysis_id = case.get("analysis_id")
    if not isinstance(analysis_id, str) or not _ANALYSIS_ID_RE.fullmatch(
        analysis_id
    ):
        raise ValueError(f"invalid analysis_id in {path}")
    if analysis_id != path.stem:
        raise ValueError(f"analysis_id does not match filename: {path}")
    if case.get("schema_version") != 2:
        raise ValueError(f"case must use schema_version 2: {path}")
    if case.get("bundle_type") != BUNDLE_TYPE:
        raise ValueError(f"unexpected bundle_type in {path}")
    if case.get("evidence_type") != EVIDENCE_TYPE:
        raise ValueError(f"unexpected evidence_type in {path}")
    if case.get("output_language") != "zh-CN":
        raise ValueError(f"case does not require zh-CN output: {path}")
    signal_context = case.get("signal_context")
    if not isinstance(signal_context, dict) or set(signal_context) != {
        "effective_sample_rate_hz",
        "sample_count",
        "duration_ms",
    }:
        raise ValueError(f"invalid signal_context in {path}")
    sample_rate = signal_context["effective_sample_rate_hz"]
    duration_ms = signal_context["duration_ms"]
    sample_count = signal_context["sample_count"]
    if (
        not isinstance(sample_rate, (int, float))
        or isinstance(sample_rate, bool)
        or not math.isfinite(float(sample_rate))
        or float(sample_rate) <= 0.0
        or not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
        or not isinstance(duration_ms, (int, float))
        or isinstance(duration_ms, bool)
        or not math.isfinite(float(duration_ms))
        or not math.isclose(
            float(duration_ms),
            sample_count / float(sample_rate) * 1000.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(f"unexpected signal_context values in {path}")
    if case.get("required_output") != REQUIRED_OUTPUT:
        raise ValueError(f"unexpected required_output in {path}")

    iq_branch = _validate_branch(case.get("iq_branch"), "iq_branch")
    feature_branch = _validate_branch(
        case.get("feature_probe_branch"), "feature_probe_branch"
    )
    if iq_branch["top1"]["label"] == feature_branch["top1"]["label"]:
        raise ValueError(f"case branches must disagree on top1: {path}")

    if has_physical_features:
        _validate_physical_features(case["physical_feature_evidence"])
    expected_reference_names = (
        REFERENCE_NAMES_B if has_physical_features else REFERENCE_NAMES_A
    )
    references = case.get("reference_document_paths")
    if not isinstance(references, dict) or set(references) != expected_reference_names:
        raise ValueError(
            f"case has unexpected reference documents: {path}"
        )
    for reference in references.values():
        if not isinstance(reference, str) or not Path(reference).is_file():
            raise ValueError(f"case has an invalid reference path: {path}")
    return case


def load_submission_cases(
    manifest_path: str | Path,
) -> list[tuple[Path, dict[str, Any]]]:
    """Load and validate a blind-adjudication submission manifest."""

    path = Path(manifest_path).resolve()
    manifest = _load_json_object(path)
    if manifest.get("schema_version") != 2:
        raise ValueError("submission manifest must use schema_version 2")
    if manifest.get("manifest_type") != MANIFEST_TYPE:
        raise ValueError("unexpected submission manifest_type")
    if manifest.get("session_mode") != "shared_resumable":
        raise ValueError("submission manifest must use shared_resumable mode")
    relative_paths = manifest.get("cases")
    if not isinstance(relative_paths, list) or not relative_paths:
        raise ValueError("submission manifest contains no cases")
    if manifest.get("case_count") != len(relative_paths):
        raise ValueError("submission manifest case_count does not match cases")
    signal_case_count = manifest.get("signal_case_count")
    if (
        not isinstance(signal_case_count, int)
        or isinstance(signal_case_count, bool)
        or signal_case_count <= 0
        or len(relative_paths) != 2 * signal_case_count
    ):
        raise ValueError("submission manifest must contain paired A/B cases")
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("submission manifest contains duplicate case paths")

    root = path.parent
    cases: list[tuple[Path, dict[str, Any]]] = []
    analysis_ids: set[str] = set()
    for relative_path in relative_paths:
        if not isinstance(relative_path, str):
            raise ValueError("submission case paths must be strings")
        candidate = (root / relative_path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(
                f"submission case escapes manifest directory: {relative_path}"
            )
        if not candidate.is_file():
            raise FileNotFoundError(f"submission case does not exist: {candidate}")
        case = _validate_case(candidate)
        analysis_id = str(case["analysis_id"])
        if analysis_id in analysis_ids:
            raise ValueError(f"duplicate analysis_id: {analysis_id}")
        analysis_ids.add(analysis_id)
        cases.append((candidate, case))
    return cases


def _review_skill_path(project_root: Path) -> Path:
    path = (
        project_root
        / "skills"
        / "signal-fusion-review-experiment"
        / "SKILL.md"
    )
    if not path.is_file():
        raise FileNotFoundError(f"project review skill does not exist: {path}")
    return path


def build_review_prompt(
    case_path: str | Path,
    project_root: str | Path,
) -> str:
    """Build the fixed blind-adjudication prompt for one case."""

    case = Path(case_path).resolve()
    skill = _review_skill_path(Path(project_root).resolve())
    return (
        "请使用项目中的 signal-fusion-review-experiment 规则，独立处理当前盲裁决 case。\n"
        f"规则文件：{skill}\n"
        f"当前 case：{case}\n"
        "只读取当前 case 及 reference_document_paths 明确列出的参考文档；"
        "不要读取真值、审计文件、同目录其他 case、历史响应或文件路径以外的信息，"
        "也不要沿用前一个 case 的证据或结论。\n"
        "IQ 分支与 64 维 feature-probe 分支的置信度未校准，不得直接比较数值大小，"
        "不得对两分支概率做平均、发明权重或用固定 if/else 阈值裁决。"
        "64 维物理特征正是 feature-probe 的输入，不是第三个独立投票；"
        "OFDM 三类共享的宽带、类噪声、高 PAPR 等现象不能单独作为改判依据。\n"
        "请综合证据选择 keep、change 或 abstain。仅在多项相干且具有类别区分力的"
        "证据支持时 change；证据不足或冲突无法消解时 abstain。\n"
        "严格按照 case.required_output 只返回一个 JSON 对象，不要输出 Markdown "
        "代码块或 JSON 之外的文字。所有叙述字段必须使用简体中文。"
    )


def extract_response_object(stdout: str, analysis_id: str) -> dict[str, Any]:
    """Extract the matching response object from quiet Hermes output."""

    decoder = json.JSONDecoder()
    matches: list[dict[str, Any]] = []
    for index, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("analysis_id") == analysis_id:
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one JSON object for {analysis_id}, found {len(matches)}"
        )
    return matches[0]


def _extract_session_id(stdout: str) -> str:
    matches = _SESSION_ID_RE.findall(stdout)
    if len(matches) != 1:
        raise ValueError(f"expected one Hermes session_id, found {len(matches)}")
    return matches[0]


def validate_response(
    response: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one Hermes adjudication response."""

    actual_fields = set(response)
    if actual_fields != RESPONSE_FIELDS:
        missing = sorted(RESPONSE_FIELDS - actual_fields)
        extra = sorted(actual_fields - RESPONSE_FIELDS)
        raise ValueError(f"response field mismatch; missing={missing}, extra={extra}")
    analysis_id = case.get("analysis_id")
    if response["analysis_id"] != analysis_id:
        raise ValueError("response analysis_id does not match the case")
    iq_branch = _validate_branch(case.get("iq_branch"), "iq_branch")
    iq_label = iq_branch["top1"]["label"]

    decision = response["decision"]
    label = response["recommended_label"]
    confidence = response["confidence_level"]
    if decision not in {"keep", "change", "abstain"}:
        raise ValueError("invalid decision")
    if confidence not in {"medium", "low"}:
        raise ValueError("confidence_level must be medium or low")
    if decision == "keep" and label != iq_label:
        raise ValueError("keep requires the IQ top1 label")
    if decision == "change" and (label not in LABELS or label == iq_label):
        raise ValueError("change requires an allowed label different from IQ top1")
    if decision == "abstain" and (label is not None or confidence != "low"):
        raise ValueError("abstain requires a null label and low confidence")

    summary = response["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be a non-empty string")
    if not _CJK_RE.search(summary):
        raise ValueError("summary must use Simplified Chinese")
    for field in LIST_FIELDS:
        values = response[field]
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError(f"{field} must be a list of non-empty strings")
        if any(not _CJK_RE.search(value) for value in values):
            raise ValueError(f"every {field} item must use Simplified Chinese")
    if (
        "physical_feature_evidence" not in case
        and response["physical_feature_evidence"]
    ):
        raise ValueError(
            "physical_feature_evidence must be empty for an A-variant case"
        )
    return dict(response)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_attempt_log(
    log_dir: Path,
    analysis_id: str,
    attempt: int,
    *,
    stdout: str,
    stderr: str,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{analysis_id}.attempt_{attempt}"
    (log_dir / f"{stem}.stdout.txt").write_text(stdout, encoding="utf-8")
    (log_dir / f"{stem}.stderr.txt").write_text(stderr, encoding="utf-8")


ReviewCallable = Callable[[str, str | None], tuple[str, str, str]]


def _build_cli_reviewer(
    *,
    hermes_executable: str | Path,
    project_root: Path,
    model: str | None,
    max_turns: int,
    timeout_seconds: int,
) -> ReviewCallable:
    executable = str(hermes_executable)

    def review(prompt: str, session_id: str | None) -> tuple[str, str, str]:
        command = [
            executable,
            "chat",
            "--quiet",
            "--source",
            "tool",
            "--toolsets",
            "file",
            "--max-turns",
            str(max_turns),
            "--pass-session-id",
        ]
        if model is not None:
            command.extend(("--model", model))
        if session_id is not None:
            command.extend(("--resume", session_id))
        command.extend(("--query", prompt))
        environment = os.environ.copy()
        environment.update({"NO_COLOR": "1", "TERM": "dumb"})
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Hermes exceeded the {timeout_seconds}-second timeout"
            ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-1000:]
            raise RuntimeError(
                f"Hermes exited with status {completed.returncode}"
                + (f": {detail}" if detail else "")
            )
        returned_session_id = _extract_session_id(completed.stdout)
        if session_id is not None and returned_session_id != session_id:
            raise RuntimeError("Hermes returned a different resumed session_id")
        return completed.stdout, returned_session_id, completed.stderr

    return review


def _resume_session_id(report_path: Path, manifest_path: Path) -> str | None:
    if not report_path.exists():
        return None
    previous = _load_json_object(report_path)
    if previous.get("manifest_path") != str(manifest_path):
        raise ValueError("existing run report belongs to a different manifest")
    session_id = previous.get("session_id")
    if session_id is None:
        return None
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("existing run report has an invalid session_id")
    return session_id


def run_review_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    hermes_executable: str | Path = "hermes",
    model: str | None = None,
    max_turns: int = 30,
    timeout_seconds: int = 900,
    retries: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    review_callable: ReviewCallable | None = None,
) -> dict[str, Any]:
    """Run manifest cases sequentially in one resumable Hermes session."""

    if max_turns <= 0 or timeout_seconds <= 0 or retries < 0:
        raise ValueError(
            "max_turns and timeout_seconds must be positive; retries cannot be negative"
        )
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    resolved_manifest = Path(manifest_path).resolve()
    cases = load_submission_cases(resolved_manifest)
    if limit is not None:
        cases = cases[:limit]
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    _review_skill_path(root)
    destination = Path(output_dir).resolve()
    runner_dir = destination / "_runner"
    report_path = runner_dir / "run_report.json"
    session_id = _resume_session_id(report_path, resolved_manifest)
    report: dict[str, Any] = {
        "schema_version": 2,
        "runner_type": "shared_session_blind_fusion_adjudication",
        "manifest_path": str(resolved_manifest),
        "output_dir": str(destination),
        "requested_case_count": len(cases),
        "model": model or "hermes_configured_model",
        "shared_session": True,
        "session_id": session_id,
        "completed": [],
        "skipped": [],
        "failed": [],
    }
    if dry_run:
        report["validated_case_count"] = len(cases)
        return report

    reviewer = review_callable or _build_cli_reviewer(
        hermes_executable=hermes_executable,
        project_root=root,
        model=model,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
    )
    log_dir = runner_dir / "raw"
    for position, (case_path, case) in enumerate(cases, start=1):
        analysis_id = str(case["analysis_id"])
        response_path = destination / f"{analysis_id}.json"
        if response_path.exists():
            existing = _load_json_object(response_path)
            validate_response(existing, case=case)
            report["skipped"].append(analysis_id)
            print(
                f"[hermes-review] {position}/{len(cases)} {analysis_id}: skipped",
                flush=True,
            )
            continue

        print(
            f"[hermes-review] {position}/{len(cases)} {analysis_id}: running",
            flush=True,
        )
        prompt = build_review_prompt(case_path, root)
        errors: list[str] = []
        for attempt in range(1, retries + 2):
            try:
                stdout, returned_session_id, stderr = reviewer(prompt, session_id)
                session_id = returned_session_id
                report["session_id"] = session_id
                _write_json_atomic(report_path, report)
                _write_attempt_log(
                    log_dir,
                    analysis_id,
                    attempt,
                    stdout=stdout,
                    stderr=stderr,
                )
                response = extract_response_object(stdout, analysis_id)
                normalized = validate_response(response, case=case)
                _write_json_atomic(response_path, normalized)
                report["completed"].append(analysis_id)
                print(
                    f"[hermes-review] {position}/{len(cases)} "
                    f"{analysis_id}: saved",
                    flush=True,
                )
                break
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(f"attempt {attempt}: {error}")
        else:
            report["failed"].append(
                {"analysis_id": analysis_id, "errors": errors}
            )
            print(
                f"[hermes-review] {position}/{len(cases)} {analysis_id}: failed",
                flush=True,
            )
        _write_json_atomic(report_path, report)

    report["completed_count"] = len(report["completed"])
    report["skipped_count"] = len(report["skipped"])
    report["failed_count"] = len(report["failed"])
    _write_json_atomic(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run blind fusion adjudication through one Hermes session."
    )
    parser.add_argument("manifest_path")
    parser.add_argument("output_dir")
    parser.add_argument("--project-root")
    parser.add_argument("--hermes", default="hermes")
    parser.add_argument("--model")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_review_manifest(
        args.manifest_path,
        args.output_dir,
        project_root=args.project_root,
        hermes_executable=args.hermes,
        model=args.model,
        max_turns=args.max_turns,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_review_prompt",
    "extract_response_object",
    "load_submission_cases",
    "run_review_manifest",
    "validate_response",
]
