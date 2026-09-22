"""Evaluate paired P6-B blind LLM review responses."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from signal_fusion.io.writers import json_safe


LABELS = ("LTE", "WiFi", "DVB-T")
RESPONSE_LIST_FIELDS = (
    "model_evidence",
    "periodicity_evidence",
    "feature_evidence",
    "conflicting_evidence",
    "limitations",
)
VARIANTS = {
    "set_1": "A_iq_and_periodicity",
    "set_2": "B_iq_periodicity_and_global_features",
}
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _validate_response(response: dict[str, Any], analysis_id: str) -> list[str]:
    core_required = {
        "analysis_id",
        "recommendation_status",
        "recommended_label",
        "confidence_level",
        "summary",
    }
    missing = sorted(core_required - set(response))
    if missing:
        raise ValueError(f"{analysis_id} is missing response fields: {missing}")
    violations = []
    for name in RESPONSE_LIST_FIELDS:
        if name not in response:
            response[name] = []
            violations.append(f"missing_{name}_normalized_to_empty_list")
    if response["analysis_id"] != analysis_id:
        raise ValueError(f"response ID does not match its filename: {analysis_id}")
    status = response["recommendation_status"]
    label = response["recommended_label"]
    confidence = response["confidence_level"]
    if status not in {"recommend", "abstain"}:
        raise ValueError(f"{analysis_id} has an invalid recommendation_status")
    if confidence not in {"medium", "low"}:
        raise ValueError(f"{analysis_id} has an invalid confidence_level")
    if status == "recommend" and label not in LABELS:
        raise ValueError(f"{analysis_id} has an invalid recommended_label")
    if status == "abstain" and (label is not None or confidence != "low"):
        raise ValueError(f"{analysis_id} violates the abstention contract")
    if not isinstance(response["summary"], str):
        raise ValueError(f"{analysis_id} summary must be a string")
    for name in RESPONSE_LIST_FIELDS:
        values = response[name]
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"{analysis_id} {name} must be a list of strings")
    return violations


def _uses_chinese(response: Mapping[str, Any]) -> bool:
    narrative = [str(response["summary"])]
    for field in RESPONSE_LIST_FIELDS:
        narrative.extend(str(value) for value in response[field])
    return _CJK_RE.search(" ".join(narrative)) is not None


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def _variant_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    recommended = [row for row in records if row["recommendation_status"] == "recommend"]
    baseline_errors = [row for row in records if not row["provisional_correct"]]
    baseline_correct = count - len(baseline_errors)
    recommended_correct = sum(bool(row["recommendation_correct"]) for row in recommended)
    corrected = sum(
        not row["provisional_correct"]
        and row["recommendation_status"] == "recommend"
        and row["recommendation_correct"] is True
        for row in records
    )
    introduced = sum(
        row["provisional_correct"]
        and row["recommendation_status"] == "recommend"
        and row["recommendation_correct"] is False
        for row in records
    )
    changed = sum(
        row["recommendation_status"] == "recommend"
        and row["recommended_label"] != row["provisional_label"]
        for row in records
    )
    chinese = sum(bool(row["contains_chinese_narrative"]) for row in records)
    return {
        "case_count": count,
        "provisional_baseline": {
            "correct_count": baseline_correct,
            "error_count": len(baseline_errors),
            "accuracy_percent": _percent(baseline_correct, count),
        },
        "recommendation": {
            "recommend_count": len(recommended),
            "abstain_count": count - len(recommended),
            "coverage_percent": _percent(len(recommended), count),
            "correct_count": recommended_correct,
            "error_count": len(recommended) - recommended_correct,
            "accuracy_when_recommended_percent": _percent(
                recommended_correct, len(recommended)
            ),
            "same_as_provisional_count": len(recommended) - changed,
            "changed_from_provisional_count": changed,
            "medium_count": sum(
                row["confidence_level"] == "medium" for row in records
            ),
            "low_count": sum(row["confidence_level"] == "low" for row in records),
        },
        "error_effect": {
            "corrected_provisional_error_count": corrected,
            "uncorrected_provisional_error_count": len(baseline_errors) - corrected,
            "introduced_error_count": introduced,
            "net_correction_count": corrected - introduced,
        },
        "feature_evidence_nonempty_count": sum(
            bool(row["feature_evidence_nonempty"]) for row in records
        ),
        "language": {
            "chinese_narrative_count": chinese,
            "non_chinese_narrative_count": count - chinese,
            "chinese_narrative_percent": _percent(chinese, count),
        },
    }


def _paired_outcome(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    def result(row: Mapping[str, Any]) -> str:
        if row["recommendation_status"] == "abstain":
            return "abstain"
        return "correct" if row["recommendation_correct"] else "wrong"

    return f"a_{result(a)}__b_{result(b)}"


def evaluate_review_responses(
    response_dir: str | Path,
    public_case_dir: str | Path,
    private_audit_path: str | Path,
) -> dict[str, Any]:
    """Validate and score the paired blind responses against private audit data."""

    response_root = Path(response_dir)
    public_root = Path(public_case_dir)
    audit = _load_json(Path(private_audit_path))
    if audit.get("audit_type") != "private_p6b_clean_review_pairing_audit":
        raise ValueError("private audit has an unexpected audit_type")
    pairs = audit.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("private audit contains no pairs")

    expected: dict[str, tuple[str, Mapping[str, Any], int]] = {}
    for pair in pairs:
        for set_name in VARIANTS:
            analysis_id = str(pair[f"{set_name}_analysis_id"])
            if analysis_id in expected:
                raise ValueError(f"duplicate analysis ID in audit: {analysis_id}")
            expected[analysis_id] = (
                set_name,
                pair["private_source_audit"],
                int(pair["pair_index"]),
            )

    response_paths = sorted(response_root.glob("*.json"))
    actual_ids = {path.stem for path in response_paths}
    missing = sorted(set(expected) - actual_ids)
    extra = sorted(actual_ids - set(expected))
    if missing or extra:
        raise ValueError(f"response set mismatch; missing={missing}, extra={extra}")

    records: dict[str, dict[str, Any]] = {}
    for path in response_paths:
        analysis_id = path.stem
        response = _load_json(path)
        contract_violations = _validate_response(response, analysis_id)
        set_name, source_audit, pair_index = expected[analysis_id]
        public_case_path = public_root / set_name / f"{analysis_id}.json"
        public_case = _load_json(public_case_path)
        if public_case.get("analysis_id") != analysis_id:
            raise ValueError(f"public case ID mismatch: {public_case_path}")
        has_global_features = "global_feature_evidence" in public_case
        expected_features = set_name == "set_2"
        if has_global_features != expected_features:
            raise ValueError(f"unexpected public evidence variant: {analysis_id}")
        feature_evidence_nonempty = bool(response["feature_evidence"])
        if not expected_features and feature_evidence_nonempty:
            raise ValueError(
                f"{analysis_id} supplied feature evidence without global features"
            )

        status = response["recommendation_status"]
        recommended_label = response["recommended_label"]
        true_label = source_audit["true_label"]
        provisional_label = source_audit["prediction_label"]
        records[analysis_id] = {
            "analysis_id": analysis_id,
            "pair_index": pair_index,
            "variant": VARIANTS[set_name],
            "true_label": true_label,
            "provisional_label": provisional_label,
            "provisional_correct": provisional_label == true_label,
            "recommendation_status": status,
            "recommended_label": recommended_label,
            "recommendation_correct": (
                recommended_label == true_label if status == "recommend" else None
            ),
            "confidence_level": response["confidence_level"],
            "contains_chinese_narrative": _uses_chinese(response),
            "feature_evidence_nonempty": feature_evidence_nonempty,
            "contract_violations": contract_violations,
        }

    by_set: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    for analysis_id, (set_name, _, _) in expected.items():
        by_set[set_name].append(records[analysis_id])
    for values in by_set.values():
        values.sort(key=lambda row: int(row["pair_index"]))

    pair_records = []
    transition_counts: dict[str, int] = {}
    for pair in pairs:
        a = records[str(pair["set_1_analysis_id"])]
        b = records[str(pair["set_2_analysis_id"])]
        outcome = _paired_outcome(a, b)
        transition_counts[outcome] = transition_counts.get(outcome, 0) + 1
        pair_records.append(
            {
                "pair_index": int(pair["pair_index"]),
                "source_analysis_id": pair["source_analysis_id"],
                "true_label": a["true_label"],
                "provisional_label": a["provisional_label"],
                "set_1": {
                    key: a[key]
                    for key in (
                        "analysis_id",
                        "recommendation_status",
                        "recommended_label",
                        "recommendation_correct",
                        "confidence_level",
                        "contains_chinese_narrative",
                    )
                },
                "set_2": {
                    key: b[key]
                    for key in (
                        "analysis_id",
                        "recommendation_status",
                        "recommended_label",
                        "recommendation_correct",
                        "confidence_level",
                        "contains_chinese_narrative",
                    )
                },
                "paired_outcome": outcome,
            }
        )

    variant_metrics = {
        VARIANTS[set_name]: _variant_metrics(values)
        for set_name, values in by_set.items()
    }
    a_metrics = variant_metrics[VARIANTS["set_1"]]
    b_metrics = variant_metrics[VARIANTS["set_2"]]
    total_chinese = sum(
        row["contains_chinese_narrative"] for row in records.values()
    )
    invalid_records = [
        row for row in records.values() if row["contract_violations"]
    ]
    true_label_counts = Counter(
        pair["private_source_audit"]["true_label"] for pair in pairs
    )
    baseline_error_sources = {
        pair["private_source_audit"].get("source_id")
        for pair in pairs
        if pair["private_source_audit"]["prediction_label"]
        != pair["private_source_audit"]["true_label"]
    }
    baseline_error_sources.discard(None)
    return {
        "schema_version": 1,
        "evaluation_type": "private_p6b_paired_llm_review_evaluation",
        "pair_count": len(pairs),
        "response_count": len(records),
        "contract_validation": {
            "strictly_valid_response_count": len(records) - len(invalid_records),
            "recoverable_format_issue_count": len(invalid_records),
            "evaluated_response_count": len(records),
            "missing_response_count": 0,
            "extra_response_count": 0,
            "issues": [
                {
                    "analysis_id": row["analysis_id"],
                    "violations": row["contract_violations"],
                }
                for row in sorted(
                    invalid_records, key=lambda value: value["analysis_id"]
                )
            ],
        },
        "selection_profile": {
            "true_label_counts": {
                label: int(true_label_counts.get(label, 0)) for label in LABELS
            },
            "provisional_error_count": a_metrics["provisional_baseline"][
                "error_count"
            ],
            "provisional_error_unique_source_count": len(baseline_error_sources),
        },
        "variants": variant_metrics,
        "paired_comparison": {
            "outcome_counts": dict(sorted(transition_counts.items())),
            "set_2_minus_set_1": {
                "recommendation_coverage_percentage_points": (
                    b_metrics["recommendation"]["coverage_percent"]
                    - a_metrics["recommendation"]["coverage_percent"]
                ),
                "corrected_provisional_error_count": (
                    b_metrics["error_effect"]["corrected_provisional_error_count"]
                    - a_metrics["error_effect"]["corrected_provisional_error_count"]
                ),
                "introduced_error_count": (
                    b_metrics["error_effect"]["introduced_error_count"]
                    - a_metrics["error_effect"]["introduced_error_count"]
                ),
                "net_correction_count": (
                    b_metrics["error_effect"]["net_correction_count"]
                    - a_metrics["error_effect"]["net_correction_count"]
                ),
            },
        },
        "language": {
            "expected_narrative_language": "Chinese",
            "chinese_narrative_count": int(total_chinese),
            "non_chinese_narrative_count": len(records) - int(total_chinese),
            "chinese_narrative_percent": _percent(int(total_chinese), len(records)),
            "non_chinese_analysis_ids": sorted(
                analysis_id
                for analysis_id, row in records.items()
                if not row["contains_chinese_narrative"]
            ),
            "classification_metrics_include_all_valid_responses": True,
        },
        "interpretation": {
            "baseline_error_count": a_metrics["provisional_baseline"]["error_count"],
            "baseline_errors_corrected_by_set_1": a_metrics["error_effect"][
                "corrected_provisional_error_count"
            ],
            "baseline_errors_corrected_by_set_2": b_metrics["error_effect"][
                "corrected_provisional_error_count"
            ],
            "feature_branch_demonstrated_correction_value": False,
            "feature_branch_demonstrated_safe_coverage_increase": (
                b_metrics["recommendation"]["coverage_percent"]
                > a_metrics["recommendation"]["coverage_percent"]
                and b_metrics["recommendation"]["error_count"] == 0
            ),
            "conclusion": (
                "The global-feature variant increased recommendation coverage "
                "without an observed error, but neither variant corrected either "
                "provisional error. This pilot does not establish LLM correction value."
            ),
        },
        "private_pair_results": pair_records,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    fields = (
        "variant",
        "case_count",
        "baseline_error_count",
        "recommend_count",
        "abstain_count",
        "coverage_percent",
        "recommendation_correct_count",
        "recommendation_error_count",
        "accuracy_when_recommended_percent",
        "corrected_provisional_error_count",
        "introduced_error_count",
        "net_correction_count",
        "chinese_narrative_count",
        "non_chinese_narrative_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, metrics in report["variants"].items():
            writer.writerow(
                {
                    "variant": variant,
                    "case_count": metrics["case_count"],
                    "baseline_error_count": metrics["provisional_baseline"][
                        "error_count"
                    ],
                    "recommend_count": metrics["recommendation"]["recommend_count"],
                    "abstain_count": metrics["recommendation"]["abstain_count"],
                    "coverage_percent": metrics["recommendation"]["coverage_percent"],
                    "recommendation_correct_count": metrics["recommendation"][
                        "correct_count"
                    ],
                    "recommendation_error_count": metrics["recommendation"][
                        "error_count"
                    ],
                    "accuracy_when_recommended_percent": metrics[
                        "recommendation"
                    ]["accuracy_when_recommended_percent"],
                    "corrected_provisional_error_count": metrics["error_effect"][
                        "corrected_provisional_error_count"
                    ],
                    "introduced_error_count": metrics["error_effect"][
                        "introduced_error_count"
                    ],
                    "net_correction_count": metrics["error_effect"][
                        "net_correction_count"
                    ],
                    "chinese_narrative_count": metrics["language"][
                        "chinese_narrative_count"
                    ],
                    "non_chinese_narrative_count": metrics["language"][
                        "non_chinese_narrative_count"
                    ],
                }
            )


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    a = report["variants"][VARIANTS["set_1"]]
    b = report["variants"][VARIANTS["set_2"]]
    language = report["language"]
    selection = report["selection_profile"]
    lines = [
        "# P6-B.2 LLM 盲审 A/B 评估",
        "",
        "## 结果",
        "",
        "| 组别 | 证据 | 推荐/总数 | 推荐时正确率 | 纠正旧错误 | 引入新错误 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        (
            f"| A | IQ ensemble + 周期证据 | "
            f"{a['recommendation']['recommend_count']}/{a['case_count']} | "
            f"{a['recommendation']['accuracy_when_recommended_percent']:.2f}% | "
            f"{a['error_effect']['corrected_provisional_error_count']} | "
            f"{a['error_effect']['introduced_error_count']} |"
        ),
        (
            f"| B | IQ ensemble + 周期证据 + 64维全局特征 | "
            f"{b['recommendation']['recommend_count']}/{b['case_count']} | "
            f"{b['recommendation']['accuracy_when_recommended_percent']:.2f}% | "
            f"{b['error_effect']['corrected_provisional_error_count']} | "
            f"{b['error_effect']['introduced_error_count']} |"
        ),
        "",
        "- 两组面对的是同一批 19 个 review case；冻结 provisional 基线为 "
        f"{a['provisional_baseline']['correct_count']}/{a['case_count']} 正确。",
        "- 真值构成为 LTE "
        f"{selection['true_label_counts']['LTE']}、WiFi "
        f"{selection['true_label_counts']['WiFi']}、DVB-T "
        f"{selection['true_label_counts']['DVB-T']}；两个 provisional 错误来自 "
        f"{selection['provisional_error_unique_source_count']} 个原始来源。",
        "- A 组与 B 组对两个 provisional 错误都选择弃权，没有完成纠错。",
        "- B 组比 A 组多推荐 2 个案例，且这些新增推荐没有产生错误；这只说明本批样本上的安全覆盖率有所增加，不能证明 64 维特征具有纠错能力。",
        "- 两组均没有把原本正确的 provisional 标签改错。",
        "",
        "## 语言与格式",
        "",
        "- 38 份响应全部可解析且可参与分类评估；其中 37 份严格满足字段契约。",
        "- `review_0014` 缺少应为空数组的 `feature_evidence`，评估时按空数组归一化并记为可恢复的格式问题。",
        f"- 中文叙述 {language['chinese_narrative_count']}/38；全英文叙述 "
        f"{language['non_chinese_narrative_count']}/38："
        f"{', '.join(language['non_chinese_analysis_ids'])}。",
        "- 英文输出只计为语言遵循问题，未从分类指标中删除。",
        "",
        "## 结论",
        "",
        "当前 19 对样本不足以证明 LLM 或 64 维特征能够纠正 IQ/周期分支的错误。B 组表现出较高的推荐覆盖率，但真正的两个错误样本均未被纠正；下一步应扩充包含更多真实 provisional 错误的 review 集，而不是据此把 LLM 接入自动改判路径。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_evaluation(
    report: Mapping[str, Any], output_dir: str | Path
) -> dict[str, str]:
    """Write compact private JSON, CSV, and Markdown evaluation artifacts."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "llm_review_evaluation_private.json"
    csv_path = output_root / "llm_review_summary.csv"
    markdown_path = output_root / "llm_review_evaluation.md"
    _write_json(json_path, report)
    _write_csv(csv_path, report)
    _write_markdown(markdown_path, report)
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate paired P6-B blind LLM review responses."
    )
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--public-case-dir", required=True)
    parser.add_argument("--private-audit-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = evaluate_review_responses(
        args.response_dir,
        args.public_case_dir,
        args.private_audit_path,
    )
    paths = write_review_evaluation(report, args.output_dir)
    summary = {
        **paths,
        "pair_count": report["pair_count"],
        "response_count": report["response_count"],
        "interpretation": report["interpretation"],
    }
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_arg_parser",
    "evaluate_review_responses",
    "main",
    "write_review_evaluation",
]
