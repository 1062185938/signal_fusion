from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .feature_map_utils import feature_code_names, load_feature_map
except ImportError:
    from feature_map_utils import feature_code_names, load_feature_map


REQUIRED_NPZ_FIELDS = (
    "features",
    "sample_rate",
    "seq_len",
    "feature_count",
    "feature_names",
)


def _read_scalar(data: np.lib.npyio.NpzFile, key: str, value_type: type) -> Any:
    """从 NPZ 中读取一个标量，并转换为 Python 标量类型。"""
    array = np.asarray(data[key]).reshape(-1)
    if array.size != 1:
        raise ValueError(f"NPZ 字段 {key!r} 必须是标量，实际 shape={data[key].shape}")
    return value_type(array[0])


def _read_feature_names(data: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    """读取 NPZ 中保存的 code_name 数组。"""
    names = np.asarray(data["feature_names"]).reshape(-1)
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in names
    )


def analyze_feature_results(
    feature_path: str,
    sample_index: int = 0,
    feature_map_path: str | None = None,
) -> dict:
    """
    将一个样本的 62 维特征值与 feature_map.json 中的名称和文档位置组装。

    本函数只返回结构化 Python dict，不对特征含义或信号类别作推断。
    """
    npz_path = Path(feature_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"找不到特征结果文件: {npz_path}")

    mapping = load_feature_map(feature_map_path)
    mapped_names = feature_code_names(mapping)

    with np.load(npz_path, allow_pickle=False) as data:
        missing = [field for field in REQUIRED_NPZ_FIELDS if field not in data]
        if missing:
            raise KeyError(f"特征 NPZ 缺少字段: {missing}")

        features = np.asarray(data["features"])
        if features.ndim != 2:
            raise ValueError(
                f"features 必须是 [N, feature_count]，实际 shape={features.shape}"
            )
        if features.shape[0] == 0:
            raise ValueError("特征 NPZ 中没有可分析的样本。")

        feature_count = _read_scalar(data, "feature_count", int)
        sample_rate = _read_scalar(data, "sample_rate", float)
        seq_len = _read_scalar(data, "seq_len", int)
        saved_names = _read_feature_names(data)

        if feature_count != mapping["feature_count"]:
            raise ValueError(
                "feature_map.json 与 NPZ 的 feature_count 不一致: "
                f"map={mapping['feature_count']}, NPZ={feature_count}"
            )
        if features.shape[1] != feature_count:
            raise ValueError(
                f"features.shape[1]={features.shape[1]}，但 feature_count={feature_count}"
            )
        if saved_names != mapped_names:
            raise ValueError("NPZ 中的 feature_names 顺序与 feature_map.json 不一致。")

        index = int(sample_index)
        if index < 0 or index >= features.shape[0]:
            raise IndexError(
                f"sample_index={index} 超出范围 [0, {features.shape[0] - 1}]"
            )
        values = features[index].astype(np.float64, copy=False)

    assembled_features = []
    for item, value in zip(mapping["features"], values):
        feature_item = dict(item)
        feature_item["value"] = float(value)
        assembled_features.append(feature_item)

    return {
        "sample_index": index,
        "sample_rate": sample_rate,
        "signal_length": seq_len,
        "feature_count": feature_count,
        "features": assembled_features,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="将特征 NPZ 与 feature_map.json 组装为可供 Agent 分析的 JSON。"
    )
    parser.add_argument("--feature_path", required=True, help="特征 NPZ 文件路径。")
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="需要组装的样本索引，默认 0。",
    )
    parser.add_argument(
        "--feature_map_path",
        default=None,
        help="可选映射文件路径；默认使用当前 skill 的 references/feature_map.json。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口：打印可直接供 Agent 使用的 JSON 结果。"""
    args = build_arg_parser().parse_args(argv)
    result = analyze_feature_results(
        feature_path=args.feature_path,
        sample_index=args.sample_index,
        feature_map_path=args.feature_map_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
