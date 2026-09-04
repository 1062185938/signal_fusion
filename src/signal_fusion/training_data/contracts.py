"""Contracts for deterministic, offline training-dataset assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SPLIT_NAMES = ("train", "validation", "test")


def _require_non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], field_name: str
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {unknown}")


@dataclass(frozen=True, slots=True)
class AssemblySource:
    """One homogeneous, prepared source dataset assigned to one split."""

    path: Path
    declared_path: str
    label: int
    expected_source_id: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingAssemblyManifest:
    """Validated configuration for building fixed train/validation/test files."""

    schema_version: int
    dataset_id: str
    label_map: dict[int, str]
    splits: dict[str, tuple[AssemblySource, ...]]
    windows_per_region: int
    regions_per_class: int | None
    region_selection: str
    window_selection: str
    remove_dc: bool
    rms_normalize: bool
    rms_epsilon: float
    require_disjoint_sources: bool
    manifest_path: Path

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], *, manifest_path: str | Path
    ) -> "TrainingAssemblyManifest":
        if not isinstance(raw, Mapping):
            raise TypeError("assembly manifest must be a JSON object")
        _reject_unknown_keys(
            raw,
            {"schema_version", "dataset_id", "label_map", "assembly", "splits"},
            "manifest",
        )

        schema_version = raw.get("schema_version")
        if schema_version != 1:
            raise ValueError(
                f"schema_version must be 1, got {schema_version!r}"
            )
        dataset_id = _require_non_empty_text(raw.get("dataset_id"), "dataset_id")

        raw_label_map = raw.get("label_map")
        if not isinstance(raw_label_map, Mapping) or not raw_label_map:
            raise ValueError("label_map must be a non-empty object")
        label_map: dict[int, str] = {}
        for raw_label, raw_name in raw_label_map.items():
            try:
                label = int(raw_label)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"label_map key must be an integer, got {raw_label!r}"
                ) from exc
            if label < 0 or str(label) != str(raw_label):
                raise ValueError(
                    f"label_map key must be a canonical non-negative integer, "
                    f"got {raw_label!r}"
                )
            if label in label_map:
                raise ValueError(f"duplicate label_map key: {label}")
            label_map[label] = _require_non_empty_text(
                raw_name, f"label_map[{raw_label!r}]"
            )
        if len(set(label_map.values())) != len(label_map):
            raise ValueError("label_map class names must be unique")

        assembly = raw.get("assembly", {})
        if not isinstance(assembly, Mapping):
            raise TypeError("assembly must be a JSON object")
        _reject_unknown_keys(
            assembly,
            {
                "windows_per_region",
                "regions_per_class",
                "region_selection",
                "window_selection",
                "remove_dc",
                "rms_normalize",
                "rms_epsilon",
                "require_disjoint_sources",
            },
            "assembly",
        )
        windows_per_region = _require_positive_int(
            assembly.get("windows_per_region"), "assembly.windows_per_region"
        )
        raw_regions_per_class = assembly.get("regions_per_class", "minimum")
        if raw_regions_per_class == "minimum":
            regions_per_class = None
        else:
            regions_per_class = _require_positive_int(
                raw_regions_per_class, "assembly.regions_per_class"
            )
        region_selection = assembly.get("region_selection", "uniform")
        if region_selection != "uniform":
            raise ValueError("assembly.region_selection currently supports only 'uniform'")
        window_selection = assembly.get("window_selection", "uniform")
        if window_selection not in {"uniform", "top_energy"}:
            raise ValueError(
                "assembly.window_selection must be 'uniform' or 'top_energy'"
            )
        remove_dc = _require_bool(
            assembly.get("remove_dc", True), "assembly.remove_dc"
        )
        rms_normalize = _require_bool(
            assembly.get("rms_normalize", True), "assembly.rms_normalize"
        )
        rms_epsilon = assembly.get("rms_epsilon", 1e-12)
        if isinstance(rms_epsilon, bool) or not isinstance(rms_epsilon, (int, float)):
            raise TypeError("assembly.rms_epsilon must be a positive number")
        rms_epsilon = float(rms_epsilon)
        if rms_epsilon <= 0.0:
            raise ValueError("assembly.rms_epsilon must be positive")
        require_disjoint_sources = _require_bool(
            assembly.get("require_disjoint_sources", True),
            "assembly.require_disjoint_sources",
        )

        raw_splits = raw.get("splits")
        if not isinstance(raw_splits, Mapping):
            raise TypeError("splits must be a JSON object")
        split_keys = set(raw_splits)
        expected_split_keys = set(SPLIT_NAMES)
        if split_keys != expected_split_keys:
            raise ValueError(
                "splits must contain exactly train, validation, and test; "
                f"got {sorted(split_keys)}"
            )

        resolved_manifest_path = Path(manifest_path).resolve()
        manifest_directory = resolved_manifest_path.parent
        splits: dict[str, tuple[AssemblySource, ...]] = {}
        seen_paths: dict[Path, str] = {}
        for split_name in SPLIT_NAMES:
            raw_sources = raw_splits[split_name]
            if not isinstance(raw_sources, list) or not raw_sources:
                raise ValueError(f"splits.{split_name} must be a non-empty list")
            sources: list[AssemblySource] = []
            for source_index, raw_source in enumerate(raw_sources):
                field_name = f"splits.{split_name}[{source_index}]"
                if not isinstance(raw_source, Mapping):
                    raise TypeError(f"{field_name} must be a JSON object")
                _reject_unknown_keys(
                    raw_source, {"path", "label", "source_id"}, field_name
                )
                declared_path = _require_non_empty_text(
                    raw_source.get("path"), f"{field_name}.path"
                )
                source_path = Path(declared_path)
                if not source_path.is_absolute():
                    source_path = manifest_directory / source_path
                source_path = source_path.resolve()
                label = raw_source.get("label")
                if isinstance(label, bool) or not isinstance(label, int):
                    raise TypeError(f"{field_name}.label must be an integer")
                if label not in label_map:
                    raise ValueError(
                        f"{field_name}.label={label} is absent from label_map"
                    )
                expected_source_id = raw_source.get("source_id")
                if expected_source_id is not None:
                    expected_source_id = _require_non_empty_text(
                        expected_source_id, f"{field_name}.source_id"
                    )
                if source_path in seen_paths:
                    raise ValueError(
                        f"dataset path appears in both {seen_paths[source_path]} and "
                        f"{split_name}: {source_path}"
                    )
                seen_paths[source_path] = split_name
                sources.append(
                    AssemblySource(
                        path=source_path,
                        declared_path=declared_path,
                        label=label,
                        expected_source_id=expected_source_id,
                    )
                )
            present_labels = {source.label for source in sources}
            missing_labels = sorted(set(label_map).difference(present_labels))
            if missing_labels:
                raise ValueError(
                    f"splits.{split_name} has no source for labels {missing_labels}"
                )
            splits[split_name] = tuple(sources)

        return cls(
            schema_version=1,
            dataset_id=dataset_id,
            label_map=label_map,
            splits=splits,
            windows_per_region=windows_per_region,
            regions_per_class=regions_per_class,
            region_selection=region_selection,
            window_selection=window_selection,
            remove_dc=remove_dc,
            rms_normalize=rms_normalize,
            rms_epsilon=rms_epsilon,
            require_disjoint_sources=require_disjoint_sources,
            manifest_path=resolved_manifest_path,
        )


@dataclass(frozen=True, slots=True)
class TrainingAssemblyResult:
    """Paths and audit data produced by one assembly run."""

    output_dir: Path
    split_paths: dict[str, Path]
    report_path: Path
    report: dict[str, Any]


__all__ = [
    "AssemblySource",
    "SPLIT_NAMES",
    "TrainingAssemblyManifest",
    "TrainingAssemblyResult",
]
