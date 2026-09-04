"""Optional raw-signal preparation pipeline."""

from typing import TYPE_CHECKING

from .contracts import (
    RESAMPLING_PROFILE_V1,
    PreparationConfig,
    RawSignal,
    ResamplingConfig,
    SignalRegion,
)
from .detectors import (
    BlePacketDetectorV1,
    BlePacketDetectorV1Config,
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
from .resampling import (
    ResampledRegion,
    ResamplingPlan,
    build_resampling_plan,
    resample_iq,
    resample_region,
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
    "BlePacketDetectorV1",
    "BlePacketDetectorV1Config",
    "DETECTOR_REGISTRY",
    "EnergyDetectorV1",
    "EnergyDetectorV1Config",
    "FullSignalDetector",
    "MatReader",
    "NormalizedIQ",
    "PreparationConfig",
    "RawSignal",
    "RawSignalReader",
    "RESAMPLING_PROFILE_V1",
    "ResampledRegion",
    "ResamplingConfig",
    "ResamplingPlan",
    "SigMFReader",
    "SignalDetector",
    "SignalRegion",
    "WindowSpan",
    "available_detectors",
    "build_energy_v1_dataset",
    "build_detector",
    "build_prepared_dataset",
    "build_resampling_plan",
    "inspect_prepared_slices",
    "inspect_sigmf_slices",
    "smart_extract_bursts",
    "normalize_iq",
    "open_raw_signal",
    "prepare_file",
    "prepare_energy_v1",
    "prepare_signal",
    "resolve_raw_format",
    "resample_iq",
    "resample_region",
    "segment_regions",
    "window_spans",
]
