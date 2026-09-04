"""Visual and coordinate inspection for prepared IQ datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from signal_fusion.preparation.contracts import RawSignal
from signal_fusion.preparation.readers import open_raw_signal, resolve_raw_format


_REQUIRED_FIELDS = (
    "X",
    "window_id",
    "window_start_sample",
    "window_end_sample",
    "sample_rate",
    "seq_len",
    "hop_len",
)


@dataclass(frozen=True)
class WindowSelection:
    sample_index: int
    burst_id: int
    window_id: int


@dataclass(frozen=True, slots=True)
class _CoordinateSchema:
    region_id: str
    region_start: str
    region_end: str
    source_region_start: str
    source_region_end: str
    source_window_start: str
    source_window_end: str
    raw_region_start: str
    raw_region_end: str
    display_name: str
    is_dual_rate: bool = False


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Slice plotting requires matplotlib. Install signal-fusion[inspection]."
        ) from exc
    return plt


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else value


def _text(value: Any) -> str | None:
    if value is None:
        return None
    scalar = _scalar(value)
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    result = str(scalar)
    return result if result else None


def _load_npz(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = [field for field in _REQUIRED_FIELDS if field not in loaded.files]
        if missing:
            raise ValueError(f"NPZ is missing required fields: {missing}")
        data = {field: loaded[field].copy() for field in loaded.files}

    x = data["X"]
    seq_len = int(_scalar(data["seq_len"]))
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"X must have shape [N, 2, seq_len], got {x.shape}")
    if x.shape[2] != seq_len:
        raise ValueError(
            f"X.shape[2] must match seq_len={seq_len}, got {x.shape}"
        )
    return data


def _coordinate_schema(data: dict[str, Any]) -> _CoordinateSchema:
    if all(
        field in data
        for field in ("burst_id", "burst_start_sample", "burst_end_sample")
    ):
        return _CoordinateSchema(
            region_id="burst_id",
            region_start="burst_start_sample",
            region_end="burst_end_sample",
            source_region_start="burst_start_sample",
            source_region_end="burst_end_sample",
            source_window_start="window_start_sample",
            source_window_end="window_end_sample",
            raw_region_start=(
                "raw_burst_start_sample"
                if "raw_burst_start_sample" in data
                else "burst_start_sample"
            ),
            raw_region_end=(
                "raw_burst_end_sample"
                if "raw_burst_end_sample" in data
                else "burst_end_sample"
            ),
            display_name="burst",
        )
    if all(
        field in data
        for field in ("region_id", "region_start_sample", "region_end_sample")
    ):
        dual_rate = _text(data.get("coordinate_schema")) == "dual_rate_v1"
        dual_fields = (
            "source_region_start_sample",
            "source_region_end_sample",
            "source_window_start_sample",
            "source_window_end_sample",
        )
        if dual_rate:
            if "source_sample_rate" not in data:
                raise ValueError(
                    "dual_rate_v1 NPZ is missing source_sample_rate"
                )
            missing = [field for field in dual_fields if field not in data]
            if missing:
                raise ValueError(
                    "dual_rate_v1 NPZ is missing source coordinate fields: "
                    f"{missing}"
                )
            num_samples = int(data["X"].shape[0])
            for field in dual_fields:
                if np.asarray(data[field]).shape != (num_samples,):
                    raise ValueError(
                        f"dual-rate coordinate {field!r} must have shape "
                        f"[{num_samples}]"
                    )
        return _CoordinateSchema(
            region_id="region_id",
            region_start="region_start_sample",
            region_end="region_end_sample",
            source_region_start=(
                "source_region_start_sample"
                if dual_rate
                else "region_start_sample"
            ),
            source_region_end=(
                "source_region_end_sample"
                if dual_rate
                else "region_end_sample"
            ),
            source_window_start=(
                "source_window_start_sample"
                if dual_rate
                else "window_start_sample"
            ),
            source_window_end=(
                "source_window_end_sample"
                if dual_rate
                else "window_end_sample"
            ),
            raw_region_start=(
                "source_region_start_sample"
                if dual_rate
                else "region_start_sample"
            ),
            raw_region_end=(
                "source_region_end_sample"
                if dual_rate
                else "region_end_sample"
            ),
            display_name="region",
            is_dual_rate=dual_rate,
        )
    raise ValueError(
        "NPZ must contain burst_* coordinates or region_* coordinates"
    )


def _resolve_optional_path(
    cli_path: str | None,
    npz_values: tuple[Any, ...],
    npz_path: Path,
) -> Path | None:
    text = cli_path
    if not text:
        for value in npz_values:
            text = _text(value)
            if text:
                break
    if not text:
        return None

    candidate = Path(text)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend((Path.cwd() / candidate, npz_path.parent / candidate))
    for item in candidates:
        if item.exists():
            return item
    return candidate


def _open_original_signal(
    data: dict[str, Any],
    npz_path: Path,
    *,
    raw_path: str | None,
    metadata_path: str | None,
    data_format: str,
    sample_rate: float | None,
    center_frequency: float | None,
    x_key: str | None,
) -> tuple[RawSignal | None, Path | None, Path | None]:
    resolved_raw_path = _resolve_optional_path(
        raw_path,
        (data.get("source_data_path"), data.get("source_path")),
        npz_path,
    )
    resolved_metadata_path = _resolve_optional_path(
        metadata_path,
        (data.get("source_meta_path"),),
        npz_path,
    )
    if resolved_raw_path is None or not resolved_raw_path.exists():
        return None, resolved_raw_path, resolved_metadata_path

    resolved_format = resolve_raw_format(resolved_raw_path, data_format)
    options: dict[str, Any] = {}
    if resolved_format == "sigmf":
        if resolved_metadata_path is not None:
            options["metadata_path"] = resolved_metadata_path
    elif resolved_format == "mat":
        if x_key is not None:
            options["x_key"] = x_key
        if sample_rate is not None:
            options["sample_rate"] = sample_rate
        elif "source_sample_rate" in data:
            options["sample_rate"] = float(_scalar(data["source_sample_rate"]))
        if center_frequency is not None:
            options["center_frequency"] = center_frequency
    else:
        resolved_sample_rate = (
            float(sample_rate)
            if sample_rate is not None
            else float(
                _scalar(data.get("source_sample_rate", data["sample_rate"]))
            )
        )
        options.update(
            sample_rate=resolved_sample_rate,
            center_frequency=center_frequency,
            iq_format="complex64",
        )
    signal = open_raw_signal(
        resolved_raw_path,
        source_id=_text(data.get("source_id")) or npz_path.stem,
        data_format=resolved_format,
        **options,
    )
    return signal, resolved_raw_path, resolved_metadata_path


def _selected_region_ids(region_ids: np.ndarray, maximum: int) -> list[int]:
    unique = sorted(np.unique(region_ids.astype(np.int64)).tolist())
    selected = unique if maximum <= 0 else unique[:maximum]
    return [int(value) for value in selected]


def _select_windows(
    indices: np.ndarray,
    window_ids: np.ndarray,
    starts: np.ndarray,
    count: int,
) -> list[int]:
    if count <= 0:
        raise ValueError("windows_per_region must be positive")
    ordered = sorted(
        (int(index) for index in indices),
        key=lambda index: (int(starts[index]), int(window_ids[index])),
    )
    if len(ordered) <= count:
        return ordered
    if count == 1:
        positions = [len(ordered) // 2]
    elif count == 2:
        positions = [0, len(ordered) - 1]
    elif count == 3:
        positions = [0, len(ordered) // 2, len(ordered) - 1]
    else:
        positions = [
            int(round(value))
            for value in np.linspace(0, len(ordered) - 1, count)
        ]
    return [ordered[position] for position in dict.fromkeys(positions)]


def _coordinate_warnings(
    indices: np.ndarray,
    data: dict[str, Any],
    schema: _CoordinateSchema,
    hop_len: int,
) -> tuple[list[str], set[int]]:
    warnings: list[str] = []
    warning_indices: set[int] = set()
    ordered = sorted(
        (int(index) for index in indices),
        key=lambda index: int(data["window_start_sample"][index]),
    )
    for index in ordered:
        region_start = int(data[schema.region_start][index])
        region_end = int(data[schema.region_end][index])
        window_start = int(data["window_start_sample"][index])
        window_end = int(data["window_end_sample"][index])
        if window_start < region_start:
            warnings.append(f"sample {index}: window_start is before region_start")
            warning_indices.add(index)
        if window_end > region_end:
            warnings.append(f"sample {index}: window_end is after region_end")
            warning_indices.add(index)
        if window_start >= window_end:
            warnings.append(f"sample {index}: window_start >= window_end")
            warning_indices.add(index)
        if schema.is_dual_rate:
            source_region_start = int(data[schema.source_region_start][index])
            source_region_end = int(data[schema.source_region_end][index])
            source_window_start = int(data[schema.source_window_start][index])
            source_window_end = int(data[schema.source_window_end][index])
            if source_window_start < source_region_start:
                warnings.append(
                    f"sample {index}: source window starts before source region"
                )
                warning_indices.add(index)
            if source_window_end > source_region_end:
                warnings.append(
                    f"sample {index}: source window ends after source region"
                )
                warning_indices.add(index)
            if source_window_start >= source_window_end:
                warnings.append(
                    f"sample {index}: source window start >= source window end"
                )
                warning_indices.add(index)
            for canonical, explicit in (
                ("window_start_sample", "target_window_start_sample"),
                ("window_end_sample", "target_window_end_sample"),
                (schema.region_start, "target_region_start_sample"),
                (schema.region_end, "target_region_end_sample"),
            ):
                if explicit in data and int(data[canonical][index]) != int(
                    data[explicit][index]
                ):
                    warnings.append(
                        f"sample {index}: {canonical} disagrees with {explicit}"
                    )
                    warning_indices.add(index)

    for previous, following in zip(ordered, ordered[1:]):
        gap = int(data["window_start_sample"][following]) - int(
            data["window_start_sample"][previous]
        )
        if gap != hop_len:
            region_id = int(data[schema.region_id][previous])
            warnings.append(
                f"{schema.display_name} {region_id}: window start gap "
                f"{gap} != hop_len {hop_len}"
            )
            warning_indices.update((previous, following))
    return warnings, warning_indices


def _plot_region_overview(
    signal: RawSignal,
    data: dict[str, Any],
    schema: _CoordinateSchema,
    region_id: int,
    indices: np.ndarray,
    selected_indices: list[int],
    output_dir: Path,
    seq_len: int,
    hop_len: int,
    max_plot_points: int,
    has_warning: bool,
) -> str | None:
    first = int(indices[0])
    target_start = int(data[schema.region_start][first])
    target_end = int(data[schema.region_end][first])
    start = int(data[schema.source_region_start][first])
    end = int(data[schema.source_region_end][first])
    raw_start = int(data[schema.raw_region_start][first])
    raw_end = int(data[schema.raw_region_end][first])
    count = end - start
    if count <= 0:
        return None
    iq = signal.read_samples(start, count)
    if len(iq) == 0:
        return None

    step = 1
    if max_plot_points > 0 and len(iq) > max_plot_points:
        step = int(math.ceil(len(iq) / max_plot_points))
    plot_iq = iq[::step]
    samples = start + np.arange(len(iq))[::step]
    warning = " | coordinate warning" if has_warning else ""
    duration_ms = count / signal.sample_rate * 1000.0

    plt = _pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(samples, np.abs(plot_iq), color="#2ca02c", linewidth=0.8)
    axes[0].set_ylabel("abs(I+jQ)")
    axes[1].plot(samples, plot_iq.real, color="#1f77b4", linewidth=0.7)
    axes[1].set_ylabel("I")
    axes[2].plot(samples, plot_iq.imag, color="#d62728", linewidth=0.7)
    axes[2].set_ylabel("Q")
    axes[2].set_xlabel("Source sample index")
    for axis in axes:
        axis.axvline(start, color="0.25", linewidth=1.0)
        axis.axvline(end, color="0.25", linewidth=1.0)
        if (raw_start, raw_end) != (start, end):
            axis.axvline(raw_start, color="#d62728", linestyle="--", linewidth=1.1)
            axis.axvline(raw_end, color="#d62728", linestyle="--", linewidth=1.1)
        axis.grid(True, alpha=0.25)
    for index in selected_indices:
        window_start = int(data[schema.source_window_start][index])
        window_end = int(data[schema.source_window_end][index])
        window_id = int(data["window_id"][index])
        axes[0].axvspan(window_start, window_end, color="#ffbf00", alpha=0.25)
        axes[0].text(
            (window_start + window_end) / 2,
            axes[0].get_ylim()[1],
            f"w{window_id}",
            ha="center",
            va="top",
            fontsize=8,
        )
    target_text = (
        f" | target=[{target_start},{target_end})"
        if schema.is_dual_rate
        else ""
    )
    fig.suptitle(
        f"{schema.display_name}_id={region_id} | source=[{start},{end})"
        f"{target_text} | "
        f"length={count} ({duration_ms:.3f} ms) | windows={len(indices)} | "
        f"seq_len={seq_len} | hop_len={hop_len} | plot step={step}{warning}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = output_dir / f"{schema.display_name}_{region_id}_overview.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _plot_window(
    data: dict[str, Any],
    schema: _CoordinateSchema,
    sample_index: int,
    output_dir: Path,
    seq_len: int,
    has_warning: bool,
) -> str:
    x = data["X"][sample_index]
    samples = np.arange(seq_len)
    region_id = int(data[schema.region_id][sample_index])
    window_id = int(data["window_id"][sample_index])
    start = int(data["window_start_sample"][sample_index])
    end = int(data["window_end_sample"][sample_index])
    warning = " | coordinate warning" if has_warning else ""

    plt = _pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(samples, x[0], color="#1f77b4", linewidth=0.9)
    axes[0].set_ylabel("I")
    axes[1].plot(samples, x[1], color="#d62728", linewidth=0.9)
    axes[1].set_ylabel("Q")
    axes[2].plot(samples, np.abs(x[0] + 1j * x[1]), color="#2ca02c", linewidth=0.9)
    axes[2].set_ylabel("abs")
    axes[2].set_xlabel("Window sample index")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle(
        f"{schema.display_name}_id={region_id} | window_id={window_id} | "
        f"window=[{start},{end}) | seq_len={seq_len}{warning}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path = output_dir / (
        f"{schema.display_name}_{region_id}_window_{window_id}.png"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def inspect_prepared_slices(
    npz_path: str,
    output_dir: str,
    *,
    raw_path: str | None = None,
    metadata_path: str | None = None,
    data_format: str = "auto",
    sample_rate: float | None = None,
    center_frequency: float | None = None,
    x_key: str | None = None,
    max_regions: int = 5,
    windows_per_region: int = 3,
    max_plot_points: int = 20_000,
) -> dict[str, Any]:
    """Inspect windows from either legacy burst or generic region datasets."""

    npz_file = Path(npz_path)
    if not npz_file.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_npz(npz_file)
    schema = _coordinate_schema(data)
    x = data["X"]
    if x.shape[0] == 0:
        return {
            "npz_path": str(npz_file),
            "output_dir": str(out_dir),
            "num_samples": 0,
            "num_regions": 0,
            "num_bursts": 0,
            "plotted_regions": 0,
            "plotted_bursts": 0,
            "plotted_windows": 0,
            "warning_count": 0,
            "coordinate_schema": _text(data.get("coordinate_schema"))
            or "single_rate",
            "generated_files": [],
        }

    signal, resolved_raw, resolved_metadata = _open_original_signal(
        data,
        npz_file,
        raw_path=raw_path,
        metadata_path=metadata_path,
        data_format=data_format,
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        x_key=x_key,
    )
    if signal is None:
        print(
            "Original raw signal was not found. Overview plots will be skipped; "
            "prepared-window plots will still be generated."
        )
        print(f"  raw_path: {resolved_raw}")
        print(f"  metadata_path: {resolved_metadata}")
    elif schema.is_dual_rate:
        source_sample_rate = float(_scalar(data["source_sample_rate"]))
        if not math.isclose(
            signal.sample_rate,
            source_sample_rate,
            rel_tol=1e-9,
            abs_tol=max(1e-12, source_sample_rate * 1e-9),
        ):
            raise ValueError(
                "raw signal sample rate does not match dual-rate source metadata: "
                f"{signal.sample_rate} != {source_sample_rate}"
            )

    sample_rate_value = float(_scalar(data["sample_rate"]))
    seq_len = int(_scalar(data["seq_len"]))
    hop_len = int(_scalar(data["hop_len"]))
    all_ids = sorted(np.unique(data[schema.region_id].astype(np.int64)).tolist())
    selected_ids = _selected_region_ids(data[schema.region_id], max_regions)
    generated_files: list[str] = []
    warning_count = 0
    plotted_regions = 0
    plotted_windows = 0

    for region_id in selected_ids:
        indices = np.where(data[schema.region_id].astype(np.int64) == region_id)[0]
        if len(indices) == 0:
            continue
        warnings, warning_indices = _coordinate_warnings(
            indices, data, schema, hop_len
        )
        for warning in warnings:
            print(f"warning: {warning}")
        warning_count += len(warnings)
        selected = _select_windows(
            indices,
            data["window_id"],
            data["window_start_sample"],
            windows_per_region,
        )
        if signal is not None:
            overview = _plot_region_overview(
                signal,
                data,
                schema,
                region_id,
                indices,
                selected,
                out_dir,
                seq_len,
                hop_len,
                max_plot_points,
                bool(warnings),
            )
            if overview:
                generated_files.append(overview)
        for index in selected:
            generated_files.append(
                _plot_window(
                    data,
                    schema,
                    index,
                    out_dir,
                    seq_len,
                    index in warning_indices,
                )
            )
            plotted_windows += 1
        plotted_regions += 1

    return {
        "npz_path": str(npz_file),
        "output_dir": str(out_dir),
        "num_samples": int(x.shape[0]),
        "num_regions": len(all_ids),
        "num_bursts": len(all_ids),
        "plotted_regions": plotted_regions,
        "plotted_bursts": plotted_regions,
        "plotted_windows": plotted_windows,
        "warning_count": warning_count,
        "coordinate_schema": _text(data.get("coordinate_schema"))
        or "single_rate",
        "source_sample_rate": float(
            _scalar(data.get("source_sample_rate", data["sample_rate"]))
        ),
        "sample_rate": sample_rate_value,
        "generated_files": generated_files,
    }


def inspect_sigmf_slices(
    npz_path: str,
    output_dir: str,
    data_path: str | None = None,
    meta_path: str | None = None,
    max_bursts: int = 5,
    windows_per_burst: int = 3,
    max_plot_points: int = 20_000,
) -> dict[str, Any]:
    """Compatibility adapter for the historical SigMF inspection function."""

    return inspect_prepared_slices(
        npz_path,
        output_dir,
        raw_path=data_path,
        metadata_path=meta_path,
        data_format="sigmf",
        max_regions=max_bursts,
        windows_per_region=windows_per_burst,
        max_plot_points=max_plot_points,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-inspect",
        description="Inspect fixed-length windows in a prepared IQ dataset.",
    )
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--raw_path", "--data_path", dest="raw_path", default=None)
    parser.add_argument(
        "--metadata_path", "--meta_path", dest="metadata_path", default=None
    )
    parser.add_argument(
        "--data_format",
        default="auto",
        choices=["auto", "sigmf", "mat", "dat", "bin"],
    )
    parser.add_argument("--sample_rate", type=float, default=None)
    parser.add_argument("--center_frequency", type=float, default=None)
    parser.add_argument("--x_key", default=None)
    parser.add_argument(
        "--max_regions", "--max_bursts", dest="max_regions", type=int, default=5
    )
    parser.add_argument(
        "--windows_per_region",
        "--windows_per_burst",
        dest="windows_per_region",
        type=int,
        default=3,
    )
    parser.add_argument("--max_plot_points", type=int, default=20_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = inspect_prepared_slices(**vars(args))
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("\nPrepared dataset inspection complete")
    print(f"  NPZ samples:         {result['num_samples']}")
    print(f"  regions in NPZ:      {result['num_regions']}")
    print(f"  plotted regions:     {result['plotted_regions']}")
    print(f"  plotted windows:     {result['plotted_windows']}")
    print(f"  coordinate schema:   {result['coordinate_schema']}")
    if "source_sample_rate" in result:
        print(f"  source sample rate:  {result['source_sample_rate']:.10g} Hz")
        print(f"  target sample rate:  {result['sample_rate']:.10g} Hz")
    print(f"  output_dir:          {result['output_dir']}")
    print(f"  coordinate warnings: {result['warning_count']}")
    print("  generated files:")
    for path in result["generated_files"]:
        print(f"    {path}")
    return 0


__all__ = [
    "WindowSelection",
    "build_arg_parser",
    "inspect_prepared_slices",
    "inspect_sigmf_slices",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
