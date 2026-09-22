"""P6-A contribution analysis for IQ models and the frozen periodicity gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.io.writers import json_safe


STAGE_NAMES = (
    "iq_member_1",
    "iq_member_2",
    "iq_member_3",
    "iq_ensemble",
    "iq_ensemble_plus_periodicity",
    "iq_ensemble_plus_periodicity_with_reject",
)

_BASE_FIELDS = (
    "y",
    "member_predictions",
    "ensemble_prediction",
    "gate_eligible",
    "gate_resolved",
    "label_changed",
    "fused_prediction",
)


def _as_arrays(values: Mapping[str, Any], *, awgn: bool) -> dict[str, np.ndarray]:
    fields = (*_BASE_FIELDS, "snr_db", "noise_seed") if awgn else _BASE_FIELDS
    missing = [name for name in fields if name not in values]
    if missing:
        raise ValueError("prediction artifact is missing fields: " + ", ".join(missing))
    arrays = {name: np.asarray(values[name]) for name in fields}
    count = arrays["y"].size
    if arrays["y"].shape != (count,):
        raise ValueError("y must have shape [N]")
    if arrays["member_predictions"].shape != (count, 3):
        raise ValueError("member_predictions must have shape [N, 3]")
    for name in fields:
        if name == "member_predictions":
            continue
        if arrays[name].shape != (count,):
            raise ValueError(f"{name} must have shape [N]")
    for name in ("y", "member_predictions", "ensemble_prediction", "fused_prediction"):
        if not np.all(np.isin(arrays[name], (0, 1, 2))):
            raise ValueError(f"{name} contains an invalid class index")
    for name in ("gate_eligible", "gate_resolved", "label_changed"):
        arrays[name] = arrays[name].astype(bool, copy=False)
    if np.any(arrays["gate_resolved"] & ~arrays["gate_eligible"]):
        raise ValueError("gate_resolved contains an ineligible region")
    actual_changed = arrays["fused_prediction"] != arrays["ensemble_prediction"]
    if not np.array_equal(arrays["label_changed"], actual_changed):
        raise ValueError("label_changed does not match the prediction arrays")
    if np.any(arrays["label_changed"] & ~arrays["gate_resolved"]):
        raise ValueError("the gate changed an unresolved region")
    return arrays


def _stage_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, Any]:
    count = int(truth.size)
    accepted_count = int(np.sum(accepted))
    accepted_correct = int(np.sum((prediction == truth) & accepted))
    covers_all = accepted_count == count
    return {
        "region_count": count,
        "coverage_percent": 100.0 * accepted_count / count,
        "accepted_count": accepted_count,
        "accepted_correct_count": accepted_correct,
        "accepted_error_count": accepted_count - accepted_correct,
        "accepted_accuracy_percent": (
            100.0 * accepted_correct / accepted_count
            if accepted_count
            else None
        ),
        "full_coverage_accuracy_percent": (
            100.0 * accepted_correct / count if covers_all else None
        ),
    }


def _trial_summary(
    arrays: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    noise_seed: int | None,
) -> dict[str, Any]:
    truth = np.asarray(arrays["y"])[mask]
    members = np.asarray(arrays["member_predictions"])[mask]
    ensemble = np.asarray(arrays["ensemble_prediction"])[mask]
    fused = np.asarray(arrays["fused_prediction"])[mask]
    eligible = np.asarray(arrays["gate_eligible"])[mask]
    resolved = np.asarray(arrays["gate_resolved"])[mask]
    changed = np.asarray(arrays["label_changed"])[mask]
    disagreement = np.any(members != members[:, :1], axis=1)
    review = disagreement & ~resolved
    all_regions = np.ones(truth.shape, dtype=bool)
    accepted = ~review

    stages = {
        f"iq_member_{index + 1}": _stage_metrics(
            truth, members[:, index], all_regions
        )
        for index in range(3)
    }
    stages["iq_ensemble"] = _stage_metrics(truth, ensemble, all_regions)
    stages["iq_ensemble_plus_periodicity"] = _stage_metrics(
        truth, fused, all_regions
    )
    stages["iq_ensemble_plus_periodicity_with_reject"] = _stage_metrics(
        truth, fused, accepted
    )

    member_accuracies = [
        stages[f"iq_member_{index + 1}"]["full_coverage_accuracy_percent"]
        for index in range(3)
    ]
    ensemble_accuracy = stages["iq_ensemble"]["full_coverage_accuracy_percent"]
    fused_accuracy = stages["iq_ensemble_plus_periodicity"][
        "full_coverage_accuracy_percent"
    ]
    return {
        "noise_seed": noise_seed,
        "region_count": int(truth.size),
        "stages": stages,
        "gate_activity": {
            "member_disagreement_count": int(np.sum(disagreement)),
            "eligible_count": int(np.sum(eligible)),
            "resolved_count": int(np.sum(resolved)),
            "changed_label_count": int(np.sum(changed)),
            "corrected_error_count": int(
                np.sum((ensemble != truth) & (fused == truth))
            ),
            "introduced_error_count": int(
                np.sum((ensemble == truth) & (fused != truth))
            ),
            "review_required_count": int(np.sum(review)),
        },
        "accuracy_deltas_percent_points": {
            "ensemble_minus_best_member": float(
                ensemble_accuracy - max(member_accuracies)
            ),
            "periodicity_minus_ensemble": float(
                fused_accuracy - ensemble_accuracy
            ),
        },
    }


def _stats(values: Sequence[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _optional_stats(values: Sequence[float | None]) -> dict[str, float] | None:
    if any(value is None for value in values):
        return None
    return _stats([float(value) for value in values])


def _aggregate_trials(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stage_fields = (
        "coverage_percent",
        "accepted_count",
        "accepted_error_count",
        "accepted_accuracy_percent",
        "full_coverage_accuracy_percent",
    )
    stages = {
        stage_name: {
            field: _optional_stats(
                [trial["stages"][stage_name][field] for trial in trials]
            )
            for field in stage_fields
        }
        for stage_name in STAGE_NAMES
    }
    gate_fields = tuple(trials[0]["gate_activity"])
    delta_fields = tuple(trials[0]["accuracy_deltas_percent_points"])
    return {
        "trial_count": len(trials),
        "region_count_per_trial": trials[0]["region_count"],
        "stages": stages,
        "gate_activity": {
            field: _stats([trial["gate_activity"][field] for trial in trials])
            for field in gate_fields
        },
        "accuracy_deltas_percent_points": {
            field: _stats(
                [trial["accuracy_deltas_percent_points"][field] for trial in trials]
            )
            for field in delta_fields
        },
    }


def summarize_fusion_ablation(
    clean_values: Mapping[str, Any],
    awgn_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize clean and AWGN contribution stages from saved predictions."""

    clean = _as_arrays(clean_values, awgn=False)
    awgn = _as_arrays(awgn_values, awgn=True)
    clean_mask = np.ones(clean["y"].shape, dtype=bool)
    conditions: list[dict[str, Any]] = []
    clean_trials = [_trial_summary(clean, clean_mask, noise_seed=None)]
    conditions.append(
        {
            "condition_id": "clean",
            "snr_db": None,
            "trials": clean_trials,
            "aggregate": _aggregate_trials(clean_trials),
        }
    )

    snr_values = sorted(np.unique(awgn["snr_db"]), reverse=True)
    for snr_db in snr_values:
        trials = []
        snr_mask = np.isclose(awgn["snr_db"], snr_db)
        for seed in sorted(np.unique(awgn["noise_seed"][snr_mask])):
            mask = snr_mask & (awgn["noise_seed"] == seed)
            trials.append(_trial_summary(awgn, mask, noise_seed=int(seed)))
        conditions.append(
            {
                "condition_id": f"awgn_{snr_db:g}_db",
                "snr_db": float(snr_db),
                "trials": trials,
                "aggregate": _aggregate_trials(trials),
            }
        )
    return {
        "schema_version": 1,
        "result_type": "technology_recognition_fusion_ablation",
        "stage_definitions": {
            "iq_member_1..3": "each independently trained IQ model",
            "iq_ensemble": "mean region probability across the three IQ models",
            "iq_ensemble_plus_periodicity": (
                "ensemble prediction after the frozen LTE/DVB-T periodicity gate; "
                "unresolved reviews retain a provisional prediction for scoring"
            ),
            "iq_ensemble_plus_periodicity_with_reject": (
                "same gated prediction, but unresolved member disagreements are "
                "rejected and have no final label"
            ),
        },
        "conditions": conditions,
    }


def _write_csv(path: Path, summary: Mapping[str, Any]) -> None:
    fields = (
        "condition_id",
        "snr_db",
        "trial_count",
        "stage",
        "full_accuracy_mean_percent",
        "full_accuracy_std_percent",
        "coverage_mean_percent",
        "accepted_accuracy_mean_percent",
        "accepted_error_count_mean",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in summary["conditions"]:
            aggregate = condition["aggregate"]
            for stage_name in STAGE_NAMES:
                stage = aggregate["stages"][stage_name]
                full_accuracy = stage["full_coverage_accuracy_percent"]
                accepted_accuracy = stage["accepted_accuracy_percent"]
                writer.writerow(
                    {
                        "condition_id": condition["condition_id"],
                        "snr_db": condition["snr_db"],
                        "trial_count": aggregate["trial_count"],
                        "stage": stage_name,
                        "full_accuracy_mean_percent": (
                            None if full_accuracy is None else full_accuracy["mean"]
                        ),
                        "full_accuracy_std_percent": (
                            None if full_accuracy is None else full_accuracy["std"]
                        ),
                        "coverage_mean_percent": stage["coverage_percent"]["mean"],
                        "accepted_accuracy_mean_percent": stage[
                            "accepted_accuracy_percent"
                        ]["mean"] if accepted_accuracy is not None else None,
                        "accepted_error_count_mean": stage[
                            "accepted_error_count"
                        ]["mean"],
                    }
                )


def _format_mean(
    stats: Mapping[str, float] | None,
    digits: int = 4,
) -> str:
    if stats is None:
        return "-"
    return f"{stats['mean']:.{digits}f}"


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# P6-A：IQ 模型与周期门控消融",
        "",
        "实验对象是 1 MS/s、4096 点连续 region。每个 IQ 成员先对两个非重叠的 "
        "2048 点窗口取平均概率，再进行三成员 ensemble。",
        "",
        "AWGN 条件为同一批 region 的 5 个噪声种子均值；它们是压力测试，"
        "不是新增的独立采集样本。",
        "",
        "## 阶段结果",
        "",
        "| 条件 | IQ成员1 | IQ成员2 | IQ成员3 | IQ ensemble | +周期门控 | 门控增益(pp) | +拒识后已接受准确率 | 覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in summary["conditions"]:
        aggregate = condition["aggregate"]
        stages = aggregate["stages"]
        label = "clean" if condition["snr_db"] is None else f"{condition['snr_db']:g} dB"
        values = [
            _format_mean(stages[f"iq_member_{index}"]["full_coverage_accuracy_percent"])
            for index in range(1, 4)
        ]
        ensemble = _format_mean(
            stages["iq_ensemble"]["full_coverage_accuracy_percent"]
        )
        fused = _format_mean(
            stages["iq_ensemble_plus_periodicity"][
                "full_coverage_accuracy_percent"
            ]
        )
        gain = _format_mean(
            aggregate["accuracy_deltas_percent_points"][
                "periodicity_minus_ensemble"
            ]
        )
        rejected = stages["iq_ensemble_plus_periodicity_with_reject"]
        lines.append(
            f"| {label} | {' | '.join(values)} | {ensemble} | {fused} | "
            f"{gain} | {_format_mean(rejected['accepted_accuracy_percent'])} | "
            f"{_format_mean(rejected['coverage_percent'])} |"
        )

    lines.extend(
        [
            "",
            "## 周期门控活动",
            "",
            "| 条件 | 成员分歧 | 可进入门控 | 已解决 | 改变标签 | 纠正 | 引入错误 | review |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in summary["conditions"]:
        aggregate = condition["aggregate"]
        gate = aggregate["gate_activity"]
        label = "clean" if condition["snr_db"] is None else f"{condition['snr_db']:g} dB"
        lines.append(
            f"| {label} | {_format_mean(gate['member_disagreement_count'], 1)} | "
            f"{_format_mean(gate['eligible_count'], 1)} | "
            f"{_format_mean(gate['resolved_count'], 1)} | "
            f"{_format_mean(gate['changed_label_count'], 1)} | "
            f"{_format_mean(gate['corrected_error_count'], 1)} | "
            f"{_format_mean(gate['introduced_error_count'], 1)} | "
            f"{_format_mean(gate['review_required_count'], 1)} |"
        )

    clean = summary["conditions"][0]["aggregate"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- IQ 模型是主识别路径：clean 条件下，ensemble 在周期门控前已经达到 "
            f"{_format_mean(clean['stages']['iq_ensemble']['full_coverage_accuracy_percent'])}%。",
            "- 周期门控是窄范围纠错器：clean 条件只改变 "
            f"{_format_mean(clean['gate_activity']['changed_label_count'], 1)} 个 region，"
            f"纠正 {_format_mean(clean['gate_activity']['corrected_error_count'], 1)} 个，"
            f"引入 {_format_mean(clean['gate_activity']['introduced_error_count'], 1)} 个错误。",
            "- 拒识不是新的分类器：它通过不给 unresolved case 最终标签来提高已接受结果可靠性，"
            "必须同时报告准确率和覆盖率。",
            "- 噪声下 ensemble 不一定优于每个单模型；均值融合可能被较弱成员拖累，"
            "因此不能把三模型 ensemble 的收益视为恒定。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_fusion_ablation_report(
    clean_predictions_path: str | Path,
    awgn_predictions_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Read saved P4/P5 predictions and write compact P6-A artifacts."""

    clean_path = Path(clean_predictions_path)
    awgn_path = Path(awgn_predictions_path)
    output = Path(output_dir)
    json_path = output / "fusion_ablation.json"
    csv_path = output / "fusion_ablation.csv"
    markdown_path = output / "fusion_ablation.md"
    existing = [path for path in (json_path, csv_path, markdown_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "fusion ablation outputs already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    for path in (clean_path, awgn_path):
        if not path.is_file():
            raise FileNotFoundError(f"prediction artifact not found: {path}")

    with np.load(clean_path, allow_pickle=False) as clean_values:
        with np.load(awgn_path, allow_pickle=False) as awgn_values:
            summary = summarize_fusion_ablation(clean_values, awgn_values)
    summary["input"] = {
        "clean_predictions_path": str(clean_path.resolve()),
        "awgn_predictions_path": str(awgn_path.resolve()),
    }

    output.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, summary)
    _write_markdown(markdown_path, summary)
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
        "summary": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize IQ-member, ensemble, periodicity, and reject stages."
    )
    parser.add_argument("--clean-predictions", required=True)
    parser.add_argument("--awgn-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_fusion_ablation_report(
        args.clean_predictions,
        args.awgn_predictions,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {name: value for name, value in report.items() if name != "summary"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STAGE_NAMES",
    "build_arg_parser",
    "build_fusion_ablation_report",
    "main",
    "summarize_fusion_ablation",
]
