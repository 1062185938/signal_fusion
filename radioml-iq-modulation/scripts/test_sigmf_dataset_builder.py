import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sigmf_dataset_builder import (  # noqa: E402
    SigMFReader,
    build_sigmf_dataset,
    load_sigmf_metadata,
    parse_sigmf_datatype,
)


def _write_meta(path: str, datatype: str, sample_rate: float = 10_000.0) -> None:
    meta = {
        "global": {
            "core:datatype": datatype,
            "core:sample_rate": sample_rate,
            "core:description": "synthetic LoRa-like test data",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": 915_000_000.0,
                "core:datetime": "2026-07-15T00:00:00Z",
            }
        ],
        "annotations": [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _write_sigmf_data(path: str, iq: np.ndarray, datatype: str) -> None:
    spec = parse_sigmf_datatype(datatype)
    raw = np.empty(len(iq), dtype=spec.dtype)
    if datatype.lower().startswith("cf32"):
        raw["r"] = iq.real.astype(np.float32)
        raw["i"] = iq.imag.astype(np.float32)
    elif datatype.lower().startswith("ci16"):
        raw["r"] = np.clip(np.round(iq.real * 32768.0), -32768, 32767).astype(
            spec.dtype["r"]
        )
        raw["i"] = np.clip(np.round(iq.imag * 32768.0), -32768, 32767).astype(
            spec.dtype["i"]
        )
    elif datatype.lower().startswith("ci8"):
        raw["r"] = np.clip(np.round(iq.real * 128.0), -128, 127).astype(
            spec.dtype["r"]
        )
        raw["i"] = np.clip(np.round(iq.imag * 128.0), -128, 127).astype(
            spec.dtype["i"]
        )
    else:
        raise AssertionError(f"Unexpected test datatype: {datatype}")
    raw.tofile(path)


def _make_synthetic_iq() -> tuple[np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(0)
    noise_scale = 0.01

    def noise(n: int) -> np.ndarray:
        return (
            rng.normal(0.0, noise_scale, n)
            + 1j * rng.normal(0.0, noise_scale, n)
        ).astype(np.complex64)

    def burst(n: int, phase_offset: float) -> np.ndarray:
        t = np.arange(n, dtype=np.float32)
        phase = phase_offset + 2.0 * np.pi * (0.02 * t + 0.00008 * t * t)
        signal = 0.6 * np.exp(1j * phase)
        signal += noise(n)
        return signal.astype(np.complex64)

    parts = [
        noise(250),
        burst(300, 0.0),
        noise(250),
        burst(300, 1.0),
        noise(220),
    ]
    iq = np.concatenate(parts).astype(np.complex64)
    expected_regions = [(250, 550), (800, 1100)]
    return iq, expected_regions


def test_datatype_parsing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        src = np.asarray(
            [0.25 - 0.5j, -0.75 + 0.125j, 0.0 + 0.0j],
            dtype=np.complex64,
        )
        for datatype, atol in [("cf32_le", 1e-6), ("ci16_le", 1.0 / 32768), ("ci8", 1.0 / 128)]:
            data_path = base / f"test_{datatype}.sigmf-data"
            meta_path = base / f"test_{datatype}.sigmf-meta"
            _write_sigmf_data(str(data_path), src, datatype)
            _write_meta(str(meta_path), datatype)
            meta = load_sigmf_metadata(str(meta_path))
            reader = SigMFReader(str(data_path), meta.dtype_spec)
            loaded = reader.read_samples(0, len(src))
            assert loaded.dtype == np.complex64
            assert loaded.shape == src.shape
            np.testing.assert_allclose(loaded.real, src.real, atol=atol)
            np.testing.assert_allclose(loaded.imag, src.imag, atol=atol)
    print("test_datatype_parsing passed")


def test_synthetic_burst_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data_path = base / "synthetic.sigmf-data"
        meta_path = base / "synthetic.sigmf-meta"
        output_path = base / "synthetic_dataset.npz"
        iq, expected_regions = _make_synthetic_iq()
        _write_sigmf_data(str(data_path), iq, "cf32_le")
        _write_meta(str(meta_path), "cf32_le", sample_rate=10_000.0)

        result = build_sigmf_dataset(
            data_path=str(data_path),
            meta_path=str(meta_path),
            output_path=str(output_path),
            label=0,
            class_name="LoRa",
            seq_len=128,
            hop_len=64,
            chunk_size=300,
            window_ms=2.0,
            start_threshold_db=10.0,
            end_threshold_db=6.0,
            min_signal_ms=10.0,
            min_gap_ms=5.0,
            pad_before_ms=0.0,
            pad_after_ms=0.0,
            remove_dc=True,
            normalize="rms",
            remainder="drop",
            release_windows=2,
            noise_probe_count=4,
        )

        assert result["num_bursts"] == 2
        assert result["num_samples"] > 0
        assert output_path.exists()
        summary_path = Path(result["summary_path"])
        assert summary_path.exists()

        with np.load(output_path) as loaded:
            x = loaded["X"]
            y = loaded["y"]
            assert x.ndim == 3
            assert x.shape[1:] == (2, 128)
            assert x.dtype == np.float32
            assert y.dtype == np.int64
            assert y.shape[0] == x.shape[0]
            assert loaded["burst_id"].shape[0] == x.shape[0]
            assert loaded["window_id"].shape[0] == x.shape[0]
            assert loaded["window_start_sample"].shape[0] == x.shape[0]

            raw_starts = sorted(np.unique(loaded["raw_burst_start_sample"]).tolist())
            raw_ends = sorted(np.unique(loaded["raw_burst_end_sample"]).tolist())
        assert len(raw_starts) == 2
        assert len(raw_ends) == 2
        for detected, expected in zip(raw_starts, [r[0] for r in expected_regions]):
            assert abs(detected - expected) <= 80
        for detected, expected in zip(raw_ends, [r[1] for r in expected_regions]):
            assert abs(detected - expected) <= 100

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["number_of_valid_bursts"] == 2
        assert summary["number_of_generated_windows"] == x.shape[0]
    print("test_synthetic_burst_dataset passed")


def main() -> int:
    test_datatype_parsing()
    test_synthetic_burst_dataset()
    print("All sigmf_dataset_builder tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
