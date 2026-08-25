"""CLI and file-level compatibility API for IQ feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from signal_fusion.feature_extraction.contracts import FEATURE_COUNT, FeatureResult
from signal_fusion.feature_extraction.service import (
    FeatureExtractionService,
    extract_feature_matrix,
)
from signal_fusion.io.loaders import load_prepared_dataset


def _scalar_float(value: Any, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(
            f"{name} must be a scalar, got shape={np.asarray(value).shape}"
        )
    number = float(array[0])
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive, got {number!r}")
    return number


def _resolve_format_from_path(path: str | Path, data_format: str) -> str:
    fmt = (data_format or "auto").lower()
    if fmt != "auto":
        return "pkl" if fmt == "pickle" else fmt
    suffix = Path(path).suffix.lower()
    return {
        ".npz": "npz",
        ".mat": "mat",
        ".npy": "npy",
        ".pkl": "pkl",
        ".pickle": "pkl",
        ".dat": "dat",
        ".bin": "bin",
    }.get(suffix, fmt)


def _read_file_scalar_metadata(
    path: str | Path,
    data_format: str,
    keys: tuple[str, ...],
) -> Any | None:
    fmt = _resolve_format_from_path(path, data_format)
    if fmt == "npz":
        with np.load(path, allow_pickle=False) as data:
            for key in keys:
                if key in data:
                    return data[key]
    if fmt == "mat":
        import scipy.io as sio

        mat = sio.loadmat(path, variable_names=list(keys))
        for key in keys:
            if key in mat:
                return mat[key]
    return None


def _infer_sample_rate(
    data_path: str | Path,
    data_format: str,
    explicit_sample_rate: float | None,
    meta: dict[str, Any],
) -> float:
    if explicit_sample_rate is not None:
        return _scalar_float(explicit_sample_rate, "sample_rate")
    for key in ("sample_rate", "sampling_rate", "fs", "Fs"):
        if key in meta:
            return _scalar_float(meta[key], key)
    value = _read_file_scalar_metadata(
        data_path,
        data_format,
        ("sample_rate", "sampling_rate", "fs", "Fs"),
    )
    if value is not None:
        return _scalar_float(value, "sample_rate")
    raise ValueError(
        "MATLAB 特征提取需要采样率。请传入 --sample_rate，"
        "或使用包含 sample_rate/fs 标量字段的 npz/mat 数据集。"
    )


def write_feature_result(result: FeatureResult, output_path: str | Path) -> Path:
    """Write the exact five-field NPZ schema consumed by existing tools."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=result.features,
        sample_rate=np.asarray(result.sample_rate, dtype=np.float64),
        seq_len=np.asarray(result.seq_len, dtype=np.int64),
        feature_count=np.asarray(FEATURE_COUNT, dtype=np.int64),
        feature_names=np.asarray(result.feature_names),
    )
    return output


def extract_features_from_dataset(
    data_path: str,
    output_path: str,
    data_format: str = "auto",
    seq_len: int | None = None,
    sample_rate: float | None = None,
    label_path: str | None = None,
    x_key: str | None = None,
    y_key: str | None = None,
    max_samples: int | None = None,
    dll_dir: str | None = None,
    dll_path: str | None = None,
    dependency_dir: list[str] | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Load one dataset, run the core service, and preserve the old NPZ API."""

    dataset = load_prepared_dataset(
        path=data_path,
        data_format=data_format,
        seq_len=seq_len,
        label_path=label_path,
        x_key=x_key,
        y_key=y_key,
    )
    resolved_sample_rate = _infer_sample_rate(
        data_path,
        data_format,
        sample_rate,
        dataset.meta,
    )
    source_id = dataset.source_id or Path(data_path).stem
    result = FeatureExtractionService().extract(
        dataset,
        sample_rate=resolved_sample_rate,
        source_id=source_id,
        max_samples=max_samples,
        dll_dir=dll_dir,
        dll_path=dll_path,
        dependency_dirs=dependency_dir,
        progress_every=progress_every,
    )
    written_path = write_feature_result(result, output_path)
    return {
        "output_path": str(written_path),
        "num_samples": result.num_samples,
        "feature_shape": tuple(int(dimension) for dimension in result.features.shape),
        "feature_dtype": str(result.features.dtype),
        "sample_rate": result.sample_rate,
        "seq_len": result.seq_len,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 IQ 数据集中提取 MATLAB DLL 特征并保存为 NPZ。"
    )
    parser.add_argument("--data_path", required=True, help="输入 IQ 数据集路径。")
    parser.add_argument("--output_path", required=True, help="输出特征 NPZ 路径。")
    parser.add_argument(
        "--data_format",
        default="auto",
        choices=["auto", "pkl", "pickle", "mat", "npz", "npy", "dat", "bin"],
        help="输入数据格式。",
    )
    parser.add_argument("--seq_len", type=int, default=None, help="可选序列长度。")
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=None,
        help="采样率 Hz；不传时尝试从数据集 sample_rate/fs 元数据读取。",
    )
    parser.add_argument("--label_path", default=None, help="可选 y.npy 标签路径。")
    parser.add_argument("--x_key", default=None, help="mat/npz 中的 X 字段名。")
    parser.add_argument("--y_key", default=None, help="mat/npz 中的 y 字段名。")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="只处理前 N 个样本，用于快速测试。",
    )
    parser.add_argument(
        "--dll_dir",
        default=None,
        help="动态库目录；默认使用 signal_fusion 包内当前平台的动态库。",
    )
    parser.add_argument(
        "--dll_path",
        default=None,
        help="显式指定 extractAllFeatures 动态库路径。",
    )
    parser.add_argument(
        "--dependency_dir",
        action="append",
        default=None,
        help="额外依赖库目录，可重复传入；Windows 下可传 MinGW bin 目录。",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="每 N 个样本打印一次进度；0 表示关闭。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = extract_features_from_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        data_format=args.data_format,
        seq_len=args.seq_len,
        sample_rate=args.sample_rate,
        label_path=args.label_path,
        x_key=args.x_key,
        y_key=args.y_key,
        max_samples=args.max_samples,
        dll_dir=args.dll_dir,
        dll_path=args.dll_path,
        dependency_dir=args.dependency_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


__all__ = [
    "build_arg_parser",
    "extract_feature_matrix",
    "extract_features_from_dataset",
    "main",
    "write_feature_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
