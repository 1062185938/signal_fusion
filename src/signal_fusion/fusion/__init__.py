"""Hermes-ready evidence fusion for one detected signal region."""

from .ensemble_evidence import (
    FEATURE_REFERENCE_DOCUMENTS,
    REFERENCE_DOCUMENTS,
    build_ensemble_evidence_files,
    build_evidence_records,
    load_feature_artifact,
    write_evidence_records,
)
from .region import CompleteRegion, load_complete_region
from .region_dataset import rebuild_continuous_region_dataset
from .periodicity_gate import (
    TECHNOLOGY_PERIODICITY_CANDIDATES,
    apply_periodicity_gate,
    load_periodicity_gate_manifest,
    lte_dvbt_disagreement_mask,
    score_technology_periodicity,
    select_periodicity_gate_margin,
    summarize_periodicity_gate,
)
from .service import analyze_group, analyze_group_branches, build_hermes_input
from .weights import FUSION_METHOD, FusionManifest


__all__ = [
    "CompleteRegion",
    "FEATURE_REFERENCE_DOCUMENTS",
    "FUSION_METHOD",
    "FusionManifest",
    "REFERENCE_DOCUMENTS",
    "TECHNOLOGY_PERIODICITY_CANDIDATES",
    "analyze_group",
    "analyze_group_branches",
    "apply_periodicity_gate",
    "build_ensemble_evidence_files",
    "build_evidence_records",
    "build_hermes_input",
    "load_complete_region",
    "load_periodicity_gate_manifest",
    "lte_dvbt_disagreement_mask",
    "load_feature_artifact",
    "rebuild_continuous_region_dataset",
    "score_technology_periodicity",
    "select_periodicity_gate_margin",
    "summarize_periodicity_gate",
    "write_evidence_records",
]
