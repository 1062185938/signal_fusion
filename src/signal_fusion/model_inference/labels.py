"""Label-map loading for index-ordered model classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_label_map(label_map: dict[str, Any]) -> tuple[str, ...]:
    if not isinstance(label_map, dict):
        raise ValueError("label_map must be a JSON object")
    try:
        sorted_items = sorted(label_map.items(), key=lambda item: int(item[0]))
        actual_indices = [int(index) for index, _ in sorted_items]
    except (TypeError, ValueError) as exc:
        raise ValueError("label_map keys must be integer indices") from exc

    expected_indices = list(range(len(sorted_items)))
    if actual_indices != expected_indices:
        raise ValueError(
            f"label_map index must be continuous from 0, got {actual_indices}"
        )
    labels = tuple(str(label) for _, label in sorted_items)
    if not labels or any(not label.strip() for label in labels):
        raise ValueError("label_map labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("label_map labels must be unique")
    return labels


def load_label_map(path: str | Path) -> tuple[str, ...]:
    map_path = Path(path)
    with map_path.open("r", encoding="utf-8") as handle:
        label_map = json.load(handle)
    return normalize_label_map(label_map)


__all__ = ["load_label_map", "normalize_label_map"]
