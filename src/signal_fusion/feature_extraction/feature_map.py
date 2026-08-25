"""Loading and validation for the stable 62-dimensional feature map."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from signal_fusion.feature_extraction.contracts import FEATURE_COUNT
from signal_fusion.feature_extraction.resource_paths import feature_map_resource_path


REQUIRED_FEATURE_FIELDS = (
    "index",
    "code_name",
    "display_name",
    "display_name_zh",
    "group",
    "reference",
)
EXPECTED_GROUP_COUNTS = {
    "time_domain": 12,
    "frequency_domain": 25,
    "time_frequency": 25,
}


def default_feature_map_path() -> Path:
    """Resolve the canonical feature map packaged with the core runtime."""

    return feature_map_resource_path()


def load_feature_map(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate feature indices, names, groups, and references."""

    map_path = Path(path).resolve() if path is not None else default_feature_map_path()
    if not map_path.is_file():
        raise FileNotFoundError(f"找不到特征映射文件: {map_path}")
    with map_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError("feature_map.json 顶层必须是 JSON object。")

    features = mapping.get("features")
    if not isinstance(features, list):
        raise ValueError("feature_map.json 中的 features 必须是数组。")
    try:
        feature_count = int(mapping["feature_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("feature_map.json 必须包含整数 feature_count。") from exc
    if feature_count != len(features):
        raise ValueError(
            f"feature_count={feature_count}，但 features 实际有 {len(features)} 项。"
        )
    if feature_count != FEATURE_COUNT:
        raise ValueError(
            f"feature_map.json 必须包含 {FEATURE_COUNT} 维特征，实际为 {feature_count}。"
        )

    code_names: set[str] = set()
    for expected_index, item in enumerate(features):
        if not isinstance(item, dict):
            raise ValueError(f"features[{expected_index}] 必须是 JSON object。")
        missing = [field for field in REQUIRED_FEATURE_FIELDS if field not in item]
        if missing:
            raise ValueError(f"features[{expected_index}] 缺少字段: {missing}")
        if item["index"] != expected_index:
            raise ValueError(
                f"features[{expected_index}].index={item['index']}，"
                "索引必须从 0 连续排列。"
            )
        code_name = item["code_name"]
        if not isinstance(code_name, str) or not code_name:
            raise ValueError(
                f"features[{expected_index}].code_name 必须是非空字符串。"
            )
        if code_name in code_names:
            raise ValueError(f"feature_map.json 中存在重复 code_name: {code_name}")
        code_names.add(code_name)
        for field_name in ("display_name", "display_name_zh", "group", "reference"):
            if not isinstance(item[field_name], str) or not item[field_name]:
                raise ValueError(
                    f"features[{expected_index}].{field_name} 必须是非空字符串。"
                )
        reference_file = item["reference"].split("#", 1)[0]
        if not (map_path.parent / reference_file).is_file():
            raise ValueError(
                f"features[{expected_index}] 引用的文档不存在: {reference_file}"
            )

    group_counts = Counter(item["group"] for item in features)
    if group_counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            "feature_map.json 分组数量必须为 "
            f"{EXPECTED_GROUP_COUNTS}，实际为 {dict(group_counts)}。"
        )
    result = dict(mapping)
    result["feature_map_path"] = str(map_path)
    return result


def feature_code_names(mapping: dict[str, Any]) -> tuple[str, ...]:
    """Return code names in stable feature-index order."""

    return tuple(item["code_name"] for item in mapping["features"])


__all__ = [
    "EXPECTED_GROUP_COUNTS",
    "REQUIRED_FEATURE_FIELDS",
    "default_feature_map_path",
    "feature_code_names",
    "load_feature_map",
]
