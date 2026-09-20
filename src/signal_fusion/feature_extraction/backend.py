"""ctypes backend for the unchanged MATLAB Coder C ABI."""

from __future__ import annotations

from collections.abc import Iterable
import ctypes
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any

import numpy as np

from signal_fusion.feature_extraction.contracts import FEATURE_COUNT
from signal_fusion.feature_extraction.resource_paths import native_resource_dir


MIN_SIGNAL_LENGTH = 32
MAX_SIGNAL_LENGTH = 16384
STATUS_MESSAGES = {
    0: "IQ_FEATURE_OK",
    -1: "IQ_FEATURE_INVALID_LENGTH",
    -2: "IQ_FEATURE_INVALID_SAMPLE_RATE",
    -3: "IQ_FEATURE_INVALID_VALUE",
}


def default_library_dir() -> Path:
    """Resolve the packaged native directory for the current platform."""

    return native_resource_dir()


def default_library_name() -> str:
    if platform.system().lower() == "windows":
        return "extractAllFeatures.dll"
    return "libextractAllFeatures.so"


class IQFeatureCtypesBackend:
    """Extract one 64-dimensional vector through the stable C interface."""

    def __init__(
        self,
        dll_dir: str | os.PathLike[str] | None = None,
        dll_path: str | os.PathLike[str] | None = None,
        dependency_dirs: Iterable[str | os.PathLike[str]] | None = None,
        auto_initialize: bool = True,
    ) -> None:
        if dll_path is not None:
            self.dll_path = Path(dll_path).resolve()
            self.dll_dir = (
                Path(dll_dir).resolve() if dll_dir else self.dll_path.parent
            )
        else:
            self.dll_dir = (
                Path(dll_dir).resolve() if dll_dir else default_library_dir()
            )
            self.dll_path = self.dll_dir / default_library_name()
        if not self.dll_path.exists():
            raise FileNotFoundError(f"Wrapper DLL not found: {self.dll_path}")

        self._dll_directory_handles: list[Any] = []
        self._temp_dll_dir: Path | None = None
        if platform.system().lower() == "windows":
            search_dirs = [self.dll_dir]
            if dependency_dirs is not None:
                search_dirs.extend(Path(item).resolve() for item in dependency_dirs)
            seen: set[Path] = set()
            for directory in search_dirs:
                if directory in seen or not directory.exists():
                    continue
                self._dll_directory_handles.append(
                    os.add_dll_directory(str(directory))
                )
                seen.add(directory)

        try:
            self._lib = ctypes.CDLL(str(self.dll_path))
        except FileNotFoundError:
            if platform.system().lower() != "windows":
                raise
            temp_dll_path = self._copy_dll_bundle_to_temp()
            self._dll_directory_handles.append(
                os.add_dll_directory(str(temp_dll_path.parent))
            )
            self._lib = ctypes.CDLL(str(temp_dll_path))
        self._initialized = False
        self._configure_signatures()
        if auto_initialize:
            self.initialize()

    def _configure_signatures(self) -> None:
        float_ptr = ctypes.POINTER(ctypes.c_float)
        self._lib.iqFeatureInitialize.argtypes = []
        self._lib.iqFeatureInitialize.restype = None
        self._lib.iqFeatureExtract.argtypes = [
            float_ptr,
            float_ptr,
            ctypes.c_int32,
            ctypes.c_double,
            float_ptr,
        ]
        self._lib.iqFeatureExtract.restype = ctypes.c_int32
        self._lib.iqFeatureTerminate.argtypes = []
        self._lib.iqFeatureTerminate.restype = None

    def initialize(self) -> None:
        self._lib.iqFeatureInitialize()
        self._initialized = True

    def terminate(self) -> None:
        if self._initialized:
            self._lib.iqFeatureTerminate()
            self._initialized = False

    def close(self) -> None:
        self.terminate()
        while self._dll_directory_handles:
            self._dll_directory_handles.pop().close()

    def _copy_dll_bundle_to_temp(self) -> Path:
        temp_root = Path(tempfile.gettempdir()) / "iq_feature_extract_dll"
        target_dir = temp_root / str(os.getpid())
        target_dir.mkdir(parents=True, exist_ok=True)
        for dll_file in self.dll_dir.glob("*.dll"):
            shutil.copy2(dll_file, target_dir / dll_file.name)
        self._temp_dll_dir = target_dir
        return target_dir / self.dll_path.name

    def extract_features(
        self,
        i_data: np.ndarray,
        q_data: np.ndarray,
        sample_rate: float,
    ) -> np.ndarray:
        i_array = np.ascontiguousarray(
            np.asarray(i_data).reshape(-1), dtype=np.float32
        )
        q_array = np.ascontiguousarray(
            np.asarray(q_data).reshape(-1), dtype=np.float32
        )
        if i_array.size != q_array.size:
            raise ValueError(
                f"I/Q length mismatch: I={i_array.size}, Q={q_array.size}"
            )
        if i_array.size < MIN_SIGNAL_LENGTH or i_array.size > MAX_SIGNAL_LENGTH:
            raise ValueError(
                "signal_length must be in "
                f"[{MIN_SIGNAL_LENGTH}, {MAX_SIGNAL_LENGTH}], got {i_array.size}"
            )
        if not np.isfinite(sample_rate) or float(sample_rate) <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")

        signal_length = int(i_array.size)
        i_buffer = np.zeros(MAX_SIGNAL_LENGTH, dtype=np.float32)
        q_buffer = np.zeros(MAX_SIGNAL_LENGTH, dtype=np.float32)
        i_buffer[:signal_length] = i_array
        q_buffer[:signal_length] = q_array
        output = np.empty(FEATURE_COUNT, dtype=np.float32)
        status = int(
            self._lib.iqFeatureExtract(
                i_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                q_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int32(signal_length),
                ctypes.c_double(float(sample_rate)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )
        self._raise_for_status(status)
        return output

    def _raise_for_status(self, status: int) -> None:
        if status != 0:
            message = STATUS_MESSAGES.get(status, f"UNKNOWN_STATUS_{status}")
            raise RuntimeError(f"MATLAB feature wrapper failed: {message} ({status})")

    def __enter__(self) -> "IQFeatureCtypesBackend":
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "FEATURE_COUNT",
    "MAX_SIGNAL_LENGTH",
    "MIN_SIGNAL_LENGTH",
    "STATUS_MESSAGES",
    "IQFeatureCtypesBackend",
    "default_library_dir",
    "default_library_name",
]
