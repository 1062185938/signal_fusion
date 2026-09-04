"""Hermes-ready evidence fusion for one detected signal region."""

from .region import CompleteRegion, load_complete_region
from .service import analyze_group, analyze_group_branches, build_hermes_input
from .weights import FUSION_METHOD, FusionManifest


__all__ = [
    "CompleteRegion",
    "FUSION_METHOD",
    "FusionManifest",
    "analyze_group",
    "analyze_group_branches",
    "build_hermes_input",
    "load_complete_region",
]
