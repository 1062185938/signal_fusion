from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion.preparation import RawSignal
from signal_fusion.preparation.cli import main as prepare_main
from signal_fusion.preparation.detectors import (
    BlePacketDetectorV1,
    BlePacketDetectorV1Config,
    available_detectors,
    build_detector,
)
from signal_fusion.preparation.detectors.ble_packet_v1 import (
    ble_crc24_bytes,
    ble_whiten_bits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_BLE_PDU = bytes.fromhex(
    "42150605040302010e095344522f42544c452f55535250"
)


def _bytes_to_lsb_bits(value: bytes) -> np.ndarray:
    return np.asarray(
        [
            (byte >> bit_index) & 1
            for byte in value
            for bit_index in range(8)
        ],
        dtype=np.uint8,
    )


def _packet_bits(pdu: bytes) -> np.ndarray:
    access_address = 0x8E89BED6
    preamble = np.asarray([0, 1] * 4, dtype=np.uint8)
    access_bits = np.asarray(
        [(access_address >> index) & 1 for index in range(32)],
        dtype=np.uint8,
    )
    body = pdu + ble_crc24_bytes(pdu)
    whitened = ble_whiten_bits(_bytes_to_lsb_bits(body), channel=37)
    return np.concatenate([preamble, access_bits, whitened])


def _synthetic_capture(packet_starts: tuple[int, ...]) -> np.ndarray:
    samples_per_symbol = 4
    packet_bits = _packet_bits(REAL_BLE_PDU)
    packet_phase = np.repeat(
        np.where(packet_bits == 1, 0.34, -0.34), samples_per_symbol
    )
    sample_count = max(packet_starts) + len(packet_phase) + 500
    rng = np.random.default_rng(2408)
    phase_difference = rng.uniform(-np.pi, np.pi, sample_count - 1)
    for packet_index, start in enumerate(packet_starts):
        polarity = 1.0 if packet_index % 2 == 0 else -1.0
        phase_difference[start : start + len(packet_phase)] = (
            polarity * packet_phase
        )
    phase = np.concatenate([[0.0], np.cumsum(phase_difference)])
    return np.exp(1j * phase).astype(np.complex64)


def _raw_signal(iq: np.ndarray) -> RawSignal:
    return RawSignal(
        source_id="synthetic_ble",
        source_path="synthetic_ble.dat",
        sample_rate=4_000_000,
        sample_count=len(iq),
        sample_format="complex64",
        _sample_reader=lambda start, count: iq[start : start + count],
    )


class BlePrimitiveTests(unittest.TestCase):
    def test_crc_matches_a_real_crc_valid_advertising_packet(self):
        self.assertEqual(ble_crc24_bytes(REAL_BLE_PDU).hex(), "1b4c85")

    def test_whitening_is_its_own_inverse(self):
        source = _bytes_to_lsb_bits(REAL_BLE_PDU)
        whitened = ble_whiten_bits(source, channel=37)
        np.testing.assert_array_equal(
            ble_whiten_bits(whitened, channel=37), source
        )


class BlePacketDetectorTests(unittest.TestCase):
    def test_detects_crc_valid_packets_across_chunk_boundary_and_polarity(self):
        starts = (1900, 4100)
        detector = BlePacketDetectorV1(
            BlePacketDetectorV1Config(chunk_size=2_000)
        )

        regions = detector.detect(_raw_signal(_synthetic_capture(starts)))

        self.assertEqual([region.start_sample for region in regions], list(starts))
        self.assertEqual([region.sample_count for region in regions], [992, 992])
        self.assertTrue(all(region.metadata["crc_valid"] for region in regions))
        self.assertEqual(
            [region.metadata["pdu_type_name"] for region in regions],
            ["ADV_NONCONN_IND", "ADV_NONCONN_IND"],
        )
        self.assertEqual(
            [region.metadata["advertiser_address"] for region in regions],
            ["01:02:03:04:05:06", "01:02:03:04:05:06"],
        )
        self.assertEqual(detector.last_report["crc_valid_packets"], 2)

    def test_registry_exposes_ble_packet_detector(self):
        self.assertIn("ble_packet_v1", available_detectors())
        detector = build_detector(
            "ble_packet_v1", {"channel": 37, "min_sync_matches": 35}
        )
        self.assertIsInstance(detector, BlePacketDetectorV1)

    def test_rejects_non_integer_samples_per_symbol(self):
        detector = BlePacketDetectorV1()
        signal = _raw_signal(np.ones(2_000, dtype=np.complex64))
        signal.sample_rate = 3_500_000
        with self.assertRaisesRegex(ValueError, "integer sample-rate/symbol-rate"):
            detector.detect(signal)

    def test_signal_prepare_cli_writes_packet_bounded_windows_and_report(self):
        starts = (1900, 4100)
        capture = _synthetic_capture(starts)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = base / "ble.dat"
            output_path = base / "ble_prepared.npz"
            capture.tofile(input_path)
            with redirect_stdout(io.StringIO()):
                exit_code = prepare_main(
                    [
                        "--input_path",
                        str(input_path),
                        "--output_path",
                        str(output_path),
                        "--source_id",
                        "synthetic_ble_cli",
                        "--data_format",
                        "dat",
                        "--sample_rate",
                        "4000000",
                        "--center_frequency",
                        "1000000000",
                        "--detector",
                        "ble_packet_v1",
                        "--ble_chunk_size",
                        "2000",
                        "--label",
                        "2",
                        "--class_name",
                        "BLE",
                        "--seq_len",
                        "128",
                        "--hop_len",
                        "128",
                        "--remainder",
                        "drop",
                    ]
                )

            self.assertEqual(exit_code, 0)
            with np.load(output_path, allow_pickle=False) as prepared:
                self.assertEqual(prepared["X"].shape, (14, 2, 128))
                self.assertEqual(set(prepared["region_start_sample"]), set(starts))
                np.testing.assert_array_equal(prepared["y"], np.full(14, 2))
            summary = json.loads(
                (base / "ble_prepared_dataset_summary.json").read_text()
            )
            self.assertEqual(summary["detector"], "ble_packet_v1")
            self.assertEqual(summary["number_of_detected_regions"], 2)
            self.assertEqual(summary["detector_report"]["crc_valid_packets"], 2)


class RealBleFixtureTests(unittest.TestCase):
    def test_r01_and_r06_have_unique_crc_valid_packets(self):
        fixtures = (
            ("BLE_1.0GHz_r01_rx_iq.dat", 24),
            ("BLE_1.0GHz_r06_rx_iq.dat", 16),
        )
        for filename, expected_count in fixtures:
            with self.subTest(filename=filename):
                path = PROJECT_ROOT / "data/raw/BLE" / filename
                if not path.is_file():
                    self.skipTest(f"BLE real fixture is not available: {filename}")
                signal = RawSignal(
                    source_id=path.stem,
                    source_path=str(path),
                    sample_rate=4_000_000,
                    sample_count=path.stat().st_size // 8,
                    sample_format="complex64",
                    center_frequency=1_000_000_000,
                    _sample_reader=lambda start, count, source=path: np.fromfile(
                        source, dtype="<c8", count=count, offset=start * 8
                    ),
                )
                detector = BlePacketDetectorV1()

                regions = detector.detect(signal)

                self.assertEqual(len(regions), expected_count)
                self.assertTrue(
                    all(region.sample_count == 992 for region in regions)
                )
                self.assertTrue(
                    all(region.metadata["crc_valid"] for region in regions)
                )
                self.assertTrue(
                    all(
                        region.metadata["advertiser_address"]
                        == "01:02:03:04:05:06"
                        for region in regions
                    )
                )


if __name__ == "__main__":
    unittest.main()
