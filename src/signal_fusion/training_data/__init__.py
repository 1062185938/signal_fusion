"""Offline assembly of prepared IQ slices for model training."""

from .assembly import (
    ASSEMBLY_VERSION,
    assemble_training_dataset,
    load_assembly_manifest,
)
from .contracts import (
    AssemblySource,
    SPLIT_NAMES,
    TrainingAssemblyManifest,
    TrainingAssemblyResult,
)

__all__ = [
    "ASSEMBLY_VERSION",
    "AssemblySource",
    "SPLIT_NAMES",
    "TrainingAssemblyManifest",
    "TrainingAssemblyResult",
    "assemble_training_dataset",
    "load_assembly_manifest",
]
