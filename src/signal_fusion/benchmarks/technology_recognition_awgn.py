"""Cross-fold summary for the public technology-recognition AWGN runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .technology_recognition import LABELS, LOCATION_FOLDS


AWGN_WINDOW_SIZES = (512, 4096)
AWGN_SNR_DB_VALUES = (10.0, 7.5, 5.0)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _condition_key(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g}_db"


def _condition_label(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g} dB"


def _validate_source_result(
    result: dict[str, Any], *, window_size: int, fold: str
) -> None:
    expected_label_map = {
        str(label): name for name, label in LABELS.items()
    }
    if result.get("evaluation_type") != "incremental_complex_awgn_snr_sweep":
        raise ValueError(f"unexpected evaluation type for {window_size}/{fold}")
    if result.get("device") != "cuda":
        raise ValueError(f"AWGN evaluation did not use CUDA for {window_size}/{fold}")
    if result.get("seq_len") != window_size:
        raise ValueError(f"unexpected seq_len for {window_size}/{fold}")
    if result.get("label_map") != expected_label_map:
        raise ValueError(f"unexpected label map for {window_size}/{fold}")
    expected_windows_per_region = 4096 // window_size
    expected_sample_count = (
        result.get("group_count") * expected_windows_per_region
    )
    if result.get("sample_count") != expected_sample_count:
        raise ValueError(f"unexpected sample/group ratio for {window_size}/{fold}")

    configuration = result.get("configuration", {})
    if configuration.get("snr_db_values") != list(AWGN_SNR_DB_VALUES):
        raise ValueError(f"unexpected SNR sweep for {window_size}/{fold}")
    if configuration.get("trials_per_snr") != 5:
        raise ValueError(f"unexpected trial count for {window_size}/{fold}")
    if configuration.get("base_seed") != 44:
        raise ValueError(f"unexpected base seed for {window_size}/{fold}")
    if configuration.get("noise") != "circular_complex_gaussian_exact_per_window_power":
        raise ValueError(f"unexpected noise definition for {window_size}/{fold}")

    conditions = result.get("conditions", [])
    expected_snr = [None, *AWGN_SNR_DB_VALUES]
    if [condition.get("snr_db") for condition in conditions] != expected_snr:
        raise ValueError(f"unexpected condition order for {window_size}/{fold}")
    for condition in conditions:
        snr_db = condition["snr_db"]
        trials = condition.get("trials", [])
        expected_seeds = [None] if snr_db is None else list(range(44, 49))
        if [trial.get("seed") for trial in trials] != expected_seeds:
            raise ValueError(f"unexpected seeds for {window_size}/{fold}/{snr_db}")
        if condition.get("aggregate", {}).get("trial_count") != len(trials):
            raise ValueError(
                f"trial aggregate mismatch for {window_size}/{fold}/{snr_db}"
            )
        for trial in trials:
            if snr_db is not None:
                achieved = float(trial["achieved_snr_db_mean"])
                if not np.isclose(achieved, snr_db, atol=1e-5):
                    raise ValueError(
                        f"achieved SNR mismatch for {window_size}/{fold}/{snr_db}"
                    )
            metrics = trial["metrics"]
            if metrics["window"]["sample_count"] != result["sample_count"]:
                raise ValueError(f"window count mismatch for {window_size}/{fold}")
            if metrics["group"]["sample_count"] != result["group_count"]:
                raise ValueError(f"group count mismatch for {window_size}/{fold}")
            for granularity in ("window", "group"):
                confusion = np.asarray(
                    metrics[granularity]["confusion_matrix"], dtype=np.int64
                )
                if int(confusion.sum()) != metrics[granularity]["sample_count"]:
                    raise ValueError(
                        "confusion total mismatch for "
                        f"{window_size}/{fold}/{granularity}"
                    )


def _group_per_class_aggregate(condition: dict[str, Any]) -> dict[str, Any]:
    trials = condition["trials"]
    return {
        str(label): {
            "class_name": name,
            **_stats(
                [
                    trial["metrics"]["group"]["per_class"][str(label)][
                        "accuracy_percent"
                    ]
                    for trial in trials
                ]
            ),
        }
        for name, label in LABELS.items()
    }


def load_awgn_runs(
    model_root: str | Path,
    *,
    window_sizes: tuple[int, ...] = AWGN_WINDOW_SIZES,
) -> list[dict[str, Any]]:
    """Load and validate selected-window AWGN evaluations."""

    root = Path(model_root)
    runs: list[dict[str, Any]] = []
    for window_size in window_sizes:
        for fold, locations in LOCATION_FOLDS.items():
            source_path = (
                root
                / f"window_{window_size}"
                / fold
                / "awgn_evaluation"
                / "snr_evaluation.json"
            )
            if not source_path.is_file():
                raise FileNotFoundError(f"AWGN result does not exist: {source_path}")
            result = json.loads(source_path.read_text(encoding="utf-8"))
            _validate_source_result(result, window_size=window_size, fold=fold)
            runs.append(
                {
                    "window_size": window_size,
                    "windows_per_region": 4096 // window_size,
                    "fold": fold,
                    "test_location": locations["test"][0],
                    "source_path": str(source_path),
                    "dataset_id": result["dataset_id"],
                    "sample_count": result["sample_count"],
                    "group_count": result["group_count"],
                    "conditions": [
                        {
                            "condition": condition["condition"],
                            "snr_db": condition["snr_db"],
                            "aggregate": {
                                **condition["aggregate"],
                                "per_class_group_accuracy_percent": (
                                    _group_per_class_aggregate(condition)
                                ),
                            },
                        }
                        for condition in result["conditions"]
                    ],
                }
            )
    return runs


def summarize_awgn_runs(
    runs: list[dict[str, Any]],
    *,
    window_sizes: tuple[int, ...] = AWGN_WINDOW_SIZES,
) -> dict[str, Any]:
    """Aggregate fold means while retaining within-fold noise-seed variation."""

    summaries: dict[str, Any] = {}
    for window_size in window_sizes:
        selected = [run for run in runs if run["window_size"] == window_size]
        if len(selected) != len(LOCATION_FOLDS):
            raise ValueError(
                f"window_size={window_size} must contain {len(LOCATION_FOLDS)} folds"
            )
        clean_window = np.mean(
            [
                run["conditions"][0]["aggregate"][
                    "window_accuracy_percent"
                ]["mean"]
                for run in selected
            ]
        )
        clean_region = np.mean(
            [
                run["conditions"][0]["aggregate"][
                    "group_accuracy_percent"
                ]["mean"]
                for run in selected
            ]
        )
        condition_summaries: dict[str, Any] = {}
        for condition_index, snr_db in enumerate((None, *AWGN_SNR_DB_VALUES)):
            conditions = [run["conditions"][condition_index] for run in selected]
            window_values = [
                condition["aggregate"]["window_accuracy_percent"]["mean"]
                for condition in conditions
            ]
            region_values = [
                condition["aggregate"]["group_accuracy_percent"]["mean"]
                for condition in conditions
            ]
            condition_summaries[_condition_key(snr_db)] = {
                "label": _condition_label(snr_db),
                "snr_db": snr_db,
                "fold_count": len(selected),
                "trials_per_fold": conditions[0]["aggregate"]["trial_count"],
                "window_accuracy_percent": _stats(window_values),
                "region_accuracy_percent": _stats(region_values),
                "window_delta_from_clean_percent_points": float(
                    np.mean(window_values) - clean_window
                ),
                "region_delta_from_clean_percent_points": float(
                    np.mean(region_values) - clean_region
                ),
                "within_fold_seed_std_percent": {
                    "window": _stats(
                        [
                            condition["aggregate"]["window_accuracy_percent"]["std"]
                            for condition in conditions
                        ]
                    ),
                    "region": _stats(
                        [
                            condition["aggregate"]["group_accuracy_percent"]["std"]
                            for condition in conditions
                        ]
                    ),
                },
                "per_class_window_accuracy_percent": {
                    str(label): {
                        "class_name": name,
                        **_stats(
                            [
                                condition["aggregate"][
                                    "per_class_window_accuracy_percent"
                                ][str(label)]["mean"]
                                for condition in conditions
                            ]
                        ),
                    }
                    for name, label in LABELS.items()
                },
                "per_class_region_accuracy_percent": {
                    str(label): {
                        "class_name": name,
                        **_stats(
                            [
                                condition["aggregate"][
                                    "per_class_group_accuracy_percent"
                                ][str(label)]["mean"]
                                for condition in conditions
                            ]
                        ),
                    }
                    for name, label in LABELS.items()
                },
            }
        summaries[str(window_size)] = {
            "windows_per_region": 4096 // window_size,
            "conditions": condition_summaries,
        }
    return summaries


def _write_fold_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "window_size",
        "windows_per_region",
        "fold",
        "test_location",
        "condition",
        "snr_db",
        "trials",
        "window_accuracy_percent",
        "window_seed_std_percent",
        "region_accuracy_percent",
        "region_seed_std_percent",
        "window_accuracy_LTE_percent",
        "window_accuracy_WiFi_percent",
        "window_accuracy_DVB-T_percent",
        "region_accuracy_LTE_percent",
        "region_accuracy_WiFi_percent",
        "region_accuracy_DVB-T_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for condition in run["conditions"]:
                aggregate = condition["aggregate"]
                row = {
                    "window_size": run["window_size"],
                    "windows_per_region": run["windows_per_region"],
                    "fold": run["fold"],
                    "test_location": run["test_location"],
                    "condition": _condition_label(condition["snr_db"]),
                    "snr_db": condition["snr_db"],
                    "trials": aggregate["trial_count"],
                    "window_accuracy_percent": aggregate[
                        "window_accuracy_percent"
                    ]["mean"],
                    "window_seed_std_percent": aggregate[
                        "window_accuracy_percent"
                    ]["std"],
                    "region_accuracy_percent": aggregate[
                        "group_accuracy_percent"
                    ]["mean"],
                    "region_seed_std_percent": aggregate[
                        "group_accuracy_percent"
                    ]["std"],
                }
                for name, label in LABELS.items():
                    row[f"window_accuracy_{name}_percent"] = aggregate[
                        "per_class_window_accuracy_percent"
                    ][str(label)]["mean"]
                    row[f"region_accuracy_{name}_percent"] = aggregate[
                        "per_class_group_accuracy_percent"
                    ][str(label)]["mean"]
                writer.writerow(row)


def _write_summary_csv(
    path: Path,
    summaries: dict[str, Any],
    window_sizes: tuple[int, ...],
) -> None:
    fields = [
        "window_size",
        "windows_per_region",
        "condition",
        "snr_db",
        "folds",
        "trials_per_fold",
        "window_accuracy_percent",
        "window_fold_std_percent",
        "region_accuracy_percent",
        "region_fold_std_percent",
        "window_delta_from_clean_percent_points",
        "region_delta_from_clean_percent_points",
        "window_accuracy_LTE_percent",
        "window_accuracy_WiFi_percent",
        "window_accuracy_DVB-T_percent",
        "region_accuracy_LTE_percent",
        "region_accuracy_WiFi_percent",
        "region_accuracy_DVB-T_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for window_size in window_sizes:
            window_summary = summaries[str(window_size)]
            for condition in window_summary["conditions"].values():
                row = {
                    "window_size": window_size,
                    "windows_per_region": window_summary["windows_per_region"],
                    "condition": condition["label"],
                    "snr_db": condition["snr_db"],
                    "folds": condition["fold_count"],
                    "trials_per_fold": condition["trials_per_fold"],
                    "window_accuracy_percent": condition[
                        "window_accuracy_percent"
                    ]["mean"],
                    "window_fold_std_percent": condition[
                        "window_accuracy_percent"
                    ]["std"],
                    "region_accuracy_percent": condition[
                        "region_accuracy_percent"
                    ]["mean"],
                    "region_fold_std_percent": condition[
                        "region_accuracy_percent"
                    ]["std"],
                    "window_delta_from_clean_percent_points": condition[
                        "window_delta_from_clean_percent_points"
                    ],
                    "region_delta_from_clean_percent_points": condition[
                        "region_delta_from_clean_percent_points"
                    ],
                }
                for name, label in LABELS.items():
                    row[f"window_accuracy_{name}_percent"] = condition[
                        "per_class_window_accuracy_percent"
                    ][str(label)]["mean"]
                    row[f"region_accuracy_{name}_percent"] = condition[
                        "per_class_region_accuracy_percent"
                    ][str(label)]["mean"]
                writer.writerow(row)


def _write_plot(
    path: Path,
    summaries: dict[str, Any],
    window_sizes: tuple[int, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("clean", "10_db", "7.5_db", "5_db")
    labels = ["clean", "10 dB", "7.5 dB", "5 dB"]
    series = []
    for window_size in window_sizes:
        key = str(window_size)
        series.append(
            (f"{window_size} window", key, "window_accuracy_percent", "o")
        )
        if window_size < 4096:
            series.append(
                (
                    f"{window_size} region mean",
                    key,
                    "region_accuracy_percent",
                    "s",
                )
            )
    x = np.arange(len(keys), dtype=np.float64)
    offsets = (
        np.zeros(1, dtype=np.float64)
        if len(series) == 1
        else np.linspace(-0.16, 0.16, len(series))
    )
    fig, axis = plt.subplots(figsize=(9.2, 5.5))
    for offset, (label, size, metric, marker) in zip(offsets, series):
        values = [
            summaries[size]["conditions"][key][metric]["mean"] for key in keys
        ]
        errors = [
            summaries[size]["conditions"][key][metric]["std"] for key in keys
        ]
        axis.errorbar(
            x + offset,
            values,
            yerr=errors,
            marker=marker,
            linestyle="none",
            capsize=4,
            label=label,
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(74.0, 101.0)
    axis.set_xlabel("Evaluation condition")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Technology Recognition: incremental AWGN robustness")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_awgn_summary(
    *,
    model_root: str | Path,
    output_dir: str | Path,
    window_sizes: tuple[int, ...] = AWGN_WINDOW_SIZES,
    plot: bool = True,
) -> dict[str, Any]:
    """Validate source evaluations and write one cross-fold result bundle."""

    output = Path(output_dir)
    json_path = output / "awgn_summary.json"
    summary_csv_path = output / "awgn_summary.csv"
    fold_csv_path = output / "awgn_fold_conditions.csv"
    plot_path = output / "awgn_accuracy.png" if plot else None
    existing = [
        path
        for path in (json_path, summary_csv_path, fold_csv_path, plot_path)
        if path is not None and path.exists()
    ]
    if existing:
        raise FileExistsError(
            "AWGN summary outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )

    if not window_sizes:
        raise ValueError("window_sizes must not be empty")
    if len(set(window_sizes)) != len(window_sizes):
        raise ValueError("window_sizes must not contain duplicates")
    unsupported = [
        size for size in window_sizes if size not in AWGN_WINDOW_SIZES
    ]
    if unsupported:
        raise ValueError(f"unsupported AWGN window sizes: {unsupported}")

    runs = load_awgn_runs(model_root, window_sizes=window_sizes)
    summaries = summarize_awgn_runs(runs, window_sizes=window_sizes)
    result = {
        "schema_version": 1,
        "evaluation_type": "location_rotated_incremental_awgn_summary",
        "model_name": "deepconvnet_1d",
        "region_size": 4096,
        "window_sizes": list(window_sizes),
        "conditions": ["clean", "10 dB", "7.5 dB", "5 dB"],
        "training_artificial_awgn": False,
        "noise": "circular_complex_gaussian_exact_per_window_power",
        "snr_interpretation": "input_window_power_to_newly_added_noise_power",
        "fold_aggregation": (
            "unweighted mean and population standard deviation of the four "
            "held-out-location fold means"
        ),
        "noisy_trial_aggregation": (
            "each fold mean is computed from five noise seeds 44 through 48"
        ),
        "summaries": summaries,
        "runs": runs,
        "quality_checks": {
            "source_result_count": len(runs),
            "expected_source_result_count": len(window_sizes)
            * len(LOCATION_FOLDS),
            "all_sources_validated": True,
            "all_evaluations_used_cuda": True,
            "all_noisy_conditions_have_seeds_44_through_48": True,
            "all_achieved_snr_means_match_targets": True,
            "all_confusion_matrix_totals_match_sample_counts": True,
        },
        "outputs": {
            "json": str(json_path),
            "summary_csv": str(summary_csv_path),
            "fold_csv": str(fold_csv_path),
            "plot": str(plot_path) if plot_path is not None else None,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_summary_csv(summary_csv_path, summaries, window_sizes)
    _write_fold_csv(fold_csv_path, runs)
    if plot_path is not None:
        _write_plot(plot_path, summaries, window_sizes)
    return result


__all__ = [
    "AWGN_SNR_DB_VALUES",
    "AWGN_WINDOW_SIZES",
    "load_awgn_runs",
    "summarize_awgn_runs",
    "write_awgn_summary",
]
