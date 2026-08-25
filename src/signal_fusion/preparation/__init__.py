"""Optional raw-signal preparation pipeline."""

from typing import TYPE_CHECKING

from .contracts import PreparationConfig, RawSignal, SignalRegion
from .detectors import (
    DETECTOR_REGISTRY,
    EnergyDetectorV1,
    EnergyDetectorV1Config,
    FullSignalDetector,
    SignalDetector,
    available_detectors,
    build_detector,
)
from .dataset import build_prepared_dataset
from .energy_v1_dataset import build_energy_v1_dataset, prepare_energy_v1
from .legacy_burst import smart_extract_bursts
from .normalization import NormalizedIQ, normalize_iq
from .pipeline import prepare_file, prepare_signal
from .readers import (
    ComplexDatReader,
    MatReader,
    RawSignalReader,
    SigMFReader,
    open_raw_signal,
    resolve_raw_format,
)
from .segmentation import segment_regions
from .windowing import WindowSpan, window_spans

if TYPE_CHECKING:
    from .inspection import inspect_prepared_slices, inspect_sigmf_slices


def __getattr__(name: str):
    if name in {"inspect_prepared_slices", "inspect_sigmf_slices"}:
        from . import inspection

        return getattr(inspection, name)
    raise AttributeError(name)

__all__ = [
    "ComplexDatReader",
    "DETECTOR_REGISTRY",
    "EnergyDetectorV1",
    "EnergyDetectorV1Config",
    "FullSignalDetector",
    "MatReader",
    "NormalizedIQ",
    "PreparationConfig",
    "RawSignal",
    "RawSignalReader",
    "SigMFReader",
    "SignalDetector",
    "SignalRegion",
    "WindowSpan",
    "available_detectors",
    "build_energy_v1_dataset",
    "build_detector",
    "build_prepared_dataset",
    "inspect_prepared_slices",
    "inspect_sigmf_slices",
    "smart_extract_bursts",
    "normalize_iq",
    "open_raw_signal",
    "prepare_file",
    "prepare_energy_v1",
    "prepare_signal",
    "resolve_raw_format",
    "segment_regions",
    "window_spans",
]
