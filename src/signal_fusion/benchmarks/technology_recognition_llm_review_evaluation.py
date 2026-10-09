"""Evaluate paired blind IQ/feature disagreement adjudication responses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from signal_fusion.benchmarks.technology_recognition_hermes_review_runner import (
    validate_response,
)
from signal_fusion.io.writers import json_safe


LABELS = ("LTE", "WiFi", "DVB-T")
VARIANTS = {
    "set_a": "A_branch_outputs_only",
    "set_b": "B_branch_outputs_and_64_features",
}
SELECTION_ROLES = ("candidate", "shadow")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def _accuracy(correct_count: int, case_count: int) -> dict[str, Any]:
    return {
        "correct_count": int(correct_count),
        "error_count": int(case_count - correct_count),
        "accuracy_percent": _percent(correct_count, case_count),
    }


def _validate_audit(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    if audit.get("schema_version") != 2:
        raise ValueError("private audit must use schema_version 2")
    if audit.get("audit_type") != "private_cross_location_llm_adjudication_audit":
        raise ValueError("private audit has an unexpected audit_type")

    population = audit.get("population_summary")
    if not isinstance(population, dict):
        raise ValueError("private audit has no population_summary")
    required_population = {
        "region_count",
        "iq_correct_count",
        "iq_error_count",
        "candidate_count",
        "shadow_count",
    }
    if set(population) != required_population:
        missing = sorted(required_population - set(population))
        extra = sorted(set(population) - required_population)
        raise ValueError(
            f"population_summary field mismatch; missing={missing}, extra={extra}"
        )
    for name in required_population:
        value = population[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"population_summary.{name} must be a non-negative int")
    if population["iq_correct_count"] + population["iq_error_count"] != population[
        "region_count"
    ]:
        raise ValueError("population IQ counts do not sum to region_count")

    cases = audit.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("private audit contains no cases")
    required_case = {
        "pair_index",
        "signal_case_id",
        "selection_role",
        "set_a_analysis_id",
        "set_b_analysis_id",
        "true_label",
        "iq_label",
        "feature_label",
        "equal_fusion_label",
    }
    analysis_ids: set[str] = set()
    signal_ids: set[str] = set()
    pair_indices: set[int] = set()
    role_counts: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            raise ValueError("every private audit case must be an object")
        missing = sorted(required_case - set(raw_case))
        if missing:
            raise ValueError(f"private audit case is missing fields: {missing}")
        pair_index = raw_case["pair_index"]
        if not isinstance(pair_index, int) or isinstance(pair_index, bool) or pair_index < 1:
            raise ValueError("pair_index must be a positive integer")
        if pair_index in pair_indices:
            raise ValueError(f"duplicate pair_index: {pair_index}")
        pair_indices.add(pair_index)
        signal_case_id = raw_case["signal_case_id"]
        if not isinstance(signal_case_id, str) or not signal_case_id:
            raise ValueError("signal_case_id must be a non-empty string")
        if signal_case_id in signal_ids:
            raise ValueError(f"duplicate signal_case_id: {signal_case_id}")
        signal_ids.add(signal_case_id)
        role = raw_case["selection_role"]
        if role not in SELECTION_ROLES:
            raise ValueError(f"invalid selection_role: {role}")
        role_counts[role] += 1
        for label_field in (
            "true_label",
            "iq_label",
            "feature_label",
            "equal_fusion_label",
        ):
            if raw_case[label_field] not in LABELS:
                raise ValueError(f"invalid {label_field}: {raw_case[label_field]}")
        for variant in VARIANTS:
            analysis_id = raw_case[f"{variant}_analysis_id"]
            if not isinstance(analysis_id, str) or not analysis_id:
                raise ValueError(f"{variant}_analysis_id must be a non-empty string")
            if analysis_id in analysis_ids:
                raise ValueError(f"duplicate analysis_id: {analysis_id}")
            analysis_ids.add(analysis_id)
        normalized.append(dict(raw_case))

    if role_counts["candidate"] != population["candidate_count"]:
        raise ValueError("candidate_count does not match private audit cases")
    if role_counts["shadow"] != population["shadow_count"]:
        raise ValueError("shadow_count does not match private audit cases")
    normalized.sort(key=lambda row: int(row["pair_index"]))
    return normalized


def _operational_label(
    decision: str,
    recommended_label: str | None,
    iq_label: str,
) -> str:
    if decision in {"keep", "abstain"}:
        return iq_label
    if recommended_label is None:
        raise ValueError("change decision has no recommended label")
    return recommended_label


def _branch_accuracy(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    correct = sum(row[field] == row["true_label"] for row in records)
    return _accuracy(correct, len(records))


def _variant_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    decisions = Counter(str(row["decision"]) for row in records)
    iq_correct = sum(row["iq_label"] == row["true_label"] for row in records)
    operational_correct = sum(
        row["operational_label"] == row["true_label"] for row in records
    )
    corrected = sum(
        row["iq_label"] != row["true_label"]
        and row["operational_label"] == row["true_label"]
        for row in records
    )
    harmed = sum(
        row["iq_label"] == row["true_label"]
        and row["operational_label"] != row["true_label"]
        for row in records
    )
    changed = [row for row in records if row["decision"] == "change"]
    correct_changes = sum(
        row["operational_label"] == row["true_label"] for row in changed
    )
    abstained = [row for row in records if row["decision"] == "abstain"]
    adjudicated = [row for row in records if row["decision"] != "abstain"]
    adjudicated_correct = sum(
        row["operational_label"] == row["true_label"] for row in adjudicated
    )
    return {
        "case_count": count,
        "branch_accuracy": {
            "iq_baseline": _accuracy(iq_correct, count),
            "feature_probe": _branch_accuracy(records, "feature_label"),
            "equal_fusion": _branch_accuracy(records, "equal_fusion_label"),
            "llm_operational_with_abstain_fallback": _accuracy(
                operational_correct, count
            ),
        },
        "decisions": {
            "keep_count": int(decisions["keep"]),
            "change_count": int(decisions["change"]),
            "abstain_count": int(decisions["abstain"]),
            "keep_percent": _percent(decisions["keep"], count),
            "change_percent": _percent(decisions["change"], count),
            "abstain_percent": _percent(decisions["abstain"], count),
        },
        "effect_vs_iq": {
            "corrected_iq_error_count": int(corrected),
            "harmed_iq_correct_count": int(harmed),
            "net_correction_count": int(corrected - harmed),
            "accuracy_delta_percentage_points": (
                None if count == 0 else 100.0 * (operational_correct - iq_correct) / count
            ),
        },
        "change_quality": {
            "change_count": len(changed),
            "correct_change_count": int(correct_changes),
            "incorrect_change_count": int(len(changed) - correct_changes),
            "change_precision_percent": _percent(correct_changes, len(changed)),
        },
        "abstention": {
            "abstain_count": len(abstained),
            "abstained_iq_error_count": sum(
                row["iq_label"] != row["true_label"] for row in abstained
            ),
            "fallback_policy": "use_iq_label",
        },
        "selective_adjudication": {
            "covered_count": len(adjudicated),
            "coverage_percent": _percent(len(adjudicated), count),
            "correct_count": int(adjudicated_correct),
            "error_count": int(len(adjudicated) - adjudicated_correct),
            "accuracy_percent": _percent(adjudicated_correct, len(adjudicated)),
        },
    }


def _candidate_projection(
    records: Sequence[Mapping[str, Any]], population: Mapping[str, int]
) -> dict[str, Any]:
    candidate_records = [
        row for row in records if row["selection_role"] == "candidate"
    ]
    candidate_metrics = _variant_metrics(candidate_records)
    effect = candidate_metrics["effect_vs_iq"]
    projected_correct = (
        population["iq_correct_count"]
        + effect["corrected_iq_error_count"]
        - effect["harmed_iq_correct_count"]
    )
    return {
        "population_region_count": population["region_count"],
        "population_iq_correct_count": population["iq_correct_count"],
        "evaluated_candidate_count": len(candidate_records),
        "noncandidate_policy": "retain_iq_label",
        "shadow_cases_included": False,
        "projected_correct_count": int(projected_correct),
        "projected_error_count": int(population["region_count"] - projected_correct),
        "projected_accuracy_percent": _percent(
            projected_correct, population["region_count"]
        ),
        "projected_delta_percentage_points_vs_iq": (
            100.0
            * (
                effect["corrected_iq_error_count"]
                - effect["harmed_iq_correct_count"]
            )
            / population["region_count"]
        ),
    }


def _paired_result_name(row: Mapping[str, Any]) -> str:
    return "correct" if row["operational_label"] == row["true_label"] else "wrong"


def evaluate_review_responses(
    response_dir: str | Path,
    public_case_dir: str | Path,
    private_audit_path: str | Path,
) -> dict[str, Any]:
    """Validate and score paired blind adjudication responses."""

    response_root = Path(response_dir)
    public_root = Path(public_case_dir)
    audit = _load_json(Path(private_audit_path))
    audit_cases = _validate_audit(audit)
    population = audit["population_summary"]

    expected: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for audit_case in audit_cases:
        for variant in VARIANTS:
            analysis_id = str(audit_case[f"{variant}_analysis_id"])
            expected[analysis_id] = (variant, audit_case)

    response_paths = sorted(response_root.glob("*.json"))
    actual_ids = {path.stem for path in response_paths}
    missing = sorted(set(expected) - actual_ids)
    extra = sorted(actual_ids - set(expected))
    if missing or extra:
        raise ValueError(f"response set mismatch; missing={missing}, extra={extra}")

    records: dict[str, dict[str, Any]] = {}
    for response_path in response_paths:
        analysis_id = response_path.stem
        variant, audit_case = expected[analysis_id]
        public_case_path = public_root / variant / f"{analysis_id}.json"
        public_case = _load_json(public_case_path)
        response = validate_response(
            _load_json(response_path),
            case=public_case,
        )
        decision = str(response["decision"])
        recommended_label = response["recommended_label"]
        iq_label = str(audit_case["iq_label"])
        operational_label = _operational_label(
            decision, recommended_label, iq_label
        )
        records[analysis_id] = {
            "analysis_id": analysis_id,
            "pair_index": int(audit_case["pair_index"]),
            "signal_case_id": str(audit_case["signal_case_id"]),
            "variant": VARIANTS[variant],
            "selection_role": str(audit_case["selection_role"]),
            "true_label": str(audit_case["true_label"]),
            "iq_label": iq_label,
            "feature_label": str(audit_case["feature_label"]),
            "equal_fusion_label": str(audit_case["equal_fusion_label"]),
            "decision": decision,
            "recommended_label": recommended_label,
            "operational_label": operational_label,
            "confidence_level": str(response["confidence_level"]),
        }

    by_variant: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in VARIANTS
    }
    for analysis_id, (variant, _) in expected.items():
        by_variant[variant].append(records[analysis_id])
    for variant_records in by_variant.values():
        variant_records.sort(key=lambda row: int(row["pair_index"]))

    variant_reports: dict[str, Any] = {}
    for variant, variant_records in by_variant.items():
        variant_reports[VARIANTS[variant]] = {
            **_variant_metrics(variant_records),
            "by_selection_role": {
                role: _variant_metrics(
                    [
                        row
                        for row in variant_records
                        if row["selection_role"] == role
                    ]
                )
                for role in SELECTION_ROLES
            },
            "candidate_population_projection": _candidate_projection(
                variant_records, population
            ),
        }

    paired_records: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    decision_transition_counts: Counter[str] = Counter()
    for audit_case in audit_cases:
        a = records[str(audit_case["set_a_analysis_id"])]
        b = records[str(audit_case["set_b_analysis_id"])]
        outcome = f"a_{_paired_result_name(a)}__b_{_paired_result_name(b)}"
        transition = f"a_{a['decision']}__b_{b['decision']}"
        outcome_counts[outcome] += 1
        decision_transition_counts[transition] += 1
        paired_records.append(
            {
                "pair_index": int(audit_case["pair_index"]),
                "signal_case_id": str(audit_case["signal_case_id"]),
                "selection_role": str(audit_case["selection_role"]),
                "true_label": str(audit_case["true_label"]),
                "iq_label": str(audit_case["iq_label"]),
                "set_a": {
                    key: a[key]
                    for key in (
                        "analysis_id",
                        "decision",
                        "recommended_label",
                        "operational_label",
                        "confidence_level",
                    )
                },
                "set_b": {
                    key: b[key]
                    for key in (
                        "analysis_id",
                        "decision",
                        "recommended_label",
                        "operational_label",
                        "confidence_level",
                    )
                },
                "paired_operational_outcome": outcome,
                "decision_transition": transition,
            }
        )

    a_name = VARIANTS["set_a"]
    b_name = VARIANTS["set_b"]
    a_metrics = variant_reports[a_name]
    b_metrics = variant_reports[b_name]
    a_operational = a_metrics["branch_accuracy"][
        "llm_operational_with_abstain_fallback"
    ]
    b_operational = b_metrics["branch_accuracy"][
        "llm_operational_with_abstain_fallback"
    ]
    return {
        "schema_version": 2,
        "evaluation_type": "private_cross_location_llm_adjudication_evaluation",
        "signal_case_count": len(audit_cases),
        "response_count": len(records),
        "population_summary": dict(population),
        "selection_profile": {
            "role_counts": {
                role: sum(
                    case["selection_role"] == role for case in audit_cases
                )
                for role in SELECTION_ROLES
            },
            "true_label_counts": {
                label: sum(case["true_label"] == label for case in audit_cases)
                for label in LABELS
            },
            "sample_is_case_control_not_natural_prevalence": True,
        },
        "variants": variant_reports,
        "paired_comparison": {
            "operational_outcome_counts": dict(sorted(outcome_counts.items())),
            "decision_transition_counts": dict(
                sorted(decision_transition_counts.items())
            ),
            "set_b_minus_set_a": {
                "sample_accuracy_percentage_points": (
                    b_operational["accuracy_percent"]
                    - a_operational["accuracy_percent"]
                ),
                "corrected_iq_error_count": (
                    b_metrics["effect_vs_iq"]["corrected_iq_error_count"]
                    - a_metrics["effect_vs_iq"]["corrected_iq_error_count"]
                ),
                "harmed_iq_correct_count": (
                    b_metrics["effect_vs_iq"]["harmed_iq_correct_count"]
                    - a_metrics["effect_vs_iq"]["harmed_iq_correct_count"]
                ),
                "net_correction_count": (
                    b_metrics["effect_vs_iq"]["net_correction_count"]
                    - a_metrics["effect_vs_iq"]["net_correction_count"]
                ),
                "candidate_projected_accuracy_percentage_points": (
                    b_metrics["candidate_population_projection"][
                        "projected_accuracy_percent"
                    ]
                    - a_metrics["candidate_population_projection"][
                        "projected_accuracy_percent"
                    ]
                ),
            },
        },
        "private_pair_results": paired_records,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    fields = (
        "variant",
        "selection_role",
        "case_count",
        "iq_accuracy_percent",
        "feature_probe_accuracy_percent",
        "equal_fusion_accuracy_percent",
        "llm_operational_accuracy_percent",
        "keep_count",
        "change_count",
        "abstain_count",
        "corrected_iq_error_count",
        "harmed_iq_correct_count",
        "net_correction_count",
        "change_precision_percent",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, metrics in report["variants"].items():
            metric_rows = {"all": metrics, **metrics["by_selection_role"]}
            for role, role_metrics in metric_rows.items():
                branches = role_metrics["branch_accuracy"]
                decisions = role_metrics["decisions"]
                effect = role_metrics["effect_vs_iq"]
                writer.writerow(
                    {
                        "variant": variant,
                        "selection_role": role,
                        "case_count": role_metrics["case_count"],
                        "iq_accuracy_percent": branches["iq_baseline"][
                            "accuracy_percent"
                        ],
                        "feature_probe_accuracy_percent": branches[
                            "feature_probe"
                        ]["accuracy_percent"],
                        "equal_fusion_accuracy_percent": branches["equal_fusion"][
                            "accuracy_percent"
                        ],
                        "llm_operational_accuracy_percent": branches[
                            "llm_operational_with_abstain_fallback"
                        ]["accuracy_percent"],
                        "keep_count": decisions["keep_count"],
                        "change_count": decisions["change_count"],
                        "abstain_count": decisions["abstain_count"],
                        "corrected_iq_error_count": effect[
                            "corrected_iq_error_count"
                        ],
                        "harmed_iq_correct_count": effect[
                            "harmed_iq_correct_count"
                        ],
                        "net_correction_count": effect["net_correction_count"],
                        "change_precision_percent": role_metrics["change_quality"][
                            "change_precision_percent"
                        ],
                    }
                )


def write_review_evaluation(
    report: Mapping[str, Any], output_dir: str | Path
) -> dict[str, str]:
    """Write the private JSON report and compact CSV summary."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "llm_adjudication_evaluation_private.json"
    csv_path = output_root / "llm_adjudication_summary.csv"
    _write_json(json_path, report)
    _write_csv(csv_path, report)
    return {"json_path": str(json_path), "csv_path": str(csv_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate paired blind IQ/feature adjudication responses."
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
    print(
        json.dumps(
            json_safe(
                {
                    **paths,
                    "signal_case_count": report["signal_case_count"],
                    "response_count": report["response_count"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_arg_parser",
    "evaluate_review_responses",
    "main",
    "write_review_evaluation",
]
