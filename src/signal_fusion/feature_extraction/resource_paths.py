"""Filesystem paths for packaged feature-extraction resources."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import platform


ASSET_PACKAGE = "signal_fusion.feature_extraction.assets"


def asset_path(*parts: str) -> Path:
    """Resolve package data to a real filesystem path.

    Native dynamic libraries cannot be loaded directly from a zip importer, so
    feature assets require a normal unpacked installation, as produced by pip
    for wheels and editable installs.
    """

    resource = resources.files(ASSET_PACKAGE).joinpath(*parts)
    path = resource if isinstance(resource, Path) else Path(str(resource))
    if not path.exists():
        raise FileNotFoundError(
            "Packaged feature resource is unavailable on the filesystem: "
            f"{'/'.join(parts)}"
        )
    return path


def feature_map_resource_path() -> Path:
    return asset_path("feature_map.json")


def native_resource_dir(system_name: str | None = None) -> Path:
    resolved_system = (system_name or platform.system()).lower()
    if resolved_system == "windows":
        platform_dir = "windows"
    elif resolved_system == "linux":
        platform_dir = "linux"
    else:
        platform_dir = resolved_system
    return asset_path("native", platform_dir)


def c_api_header_path() -> Path:
    return asset_path("include", "iq_feature_c_api.h")


__all__ = [
    "ASSET_PACKAGE",
    "asset_path",
    "c_api_header_path",
    "feature_map_resource_path",
    "native_resource_dir",
]
