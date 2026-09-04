"""CRC-validated BLE LE 1M advertising-packet detector.

The detector uses the BLE preamble and access address only to generate
candidates.  A candidate becomes a :class:`SignalRegion` only after its
advertising header has been dewhitened and its 24-bit CRC has been verified.
Consequently, returned boundaries describe the complete over-the-air packet,
from the first preamble bit through the final CRC bit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from signal_fusion.preparation.contracts import RawSignal, SignalRegion
from signal_fusion.preparation.detectors.base import SignalDetector


BLE_LE_1M_SYMBOL_RATE = 1_000_000.0
BLE_ADVERTISING_ACCESS_ADDRESS = 0x8E89BED6
BLE_ADVERTISING_CRC_INIT = 0x555555
BLE_SYNC_BITS = 40
BLE_HEADER_BITS = 16
BLE_CRC_BITS = 24

_PDU_TYPE_NAMES = {
    0: "ADV_IND",
    1: "ADV_DIRECT_IND",
    2: "ADV_NONCONN_IND",
    3: "SCAN_REQ",
    4: "SCAN_RSP",
    5: "CONNECT_IND",
    6: "ADV_SCAN_IND",
    7: "ADV_EXT_IND",
}


def _byte_to_lsb_bits(value: int) -> np.ndarray:
    return np.asarray(
        [(int(value) >> index) & 1 for index in range(8)], dtype=np.uint8
    )


def _bytes_to_lsb_bits(value: bytes) -> np.ndarray:
    if not value:
        return np.empty(0, dtype=np.uint8)
    return np.concatenate([_byte_to_lsb_bits(byte) for byte in value])


def _lsb_bits_to_bytes(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    byte_count = len(values) // 8
    return bytes(
        sum(
            int(values[byte_index * 8 + bit_index]) << bit_index
            for bit_index in range(8)
        )
        for byte_index in range(byte_count)
    )


def _reverse_byte(value: int) -> int:
    return int(f"{int(value):08b}"[::-1], 2)


def ble_whiten_bits(bits: np.ndarray, channel: int) -> np.ndarray:
    """Apply the BLE whitening sequence.

    Whitening and dewhitening are the same XOR operation.  Bits must be in
    their over-the-air, least-significant-bit-first order.
    """

    channel = int(channel)
    if not 0 <= channel <= 39:
        raise ValueError("BLE channel must be within [0, 39]")
    source = np.asarray(bits, dtype=np.uint8).reshape(-1)
    output = np.empty_like(source)
    state = channel | 0x40
    for index, bit in enumerate(source):
        state_lsb = state & 1
        output[index] = int(bit) ^ state_lsb
        state = (state >> 1) ^ (0x44 if state_lsb else 0)
    return output


def ble_crc24_bytes(pdu: bytes, crc_init: int = BLE_ADVERTISING_CRC_INIT) -> bytes:
    """Return the three BLE CRC bytes in over-the-air byte representation."""

    state = int(crc_init)
    if not 0 <= state <= 0xFFFFFF:
        raise ValueError("BLE CRC init must fit in 24 bits")
    for byte in pdu:
        for bit_index in range(8):
            feedback = ((state >> 23) & 1) ^ ((byte >> bit_index) & 1)
            state = (state << 1) & 0xFFFFFF
            if feedback:
                state ^= 0x00065B
    state_bytes = state.to_bytes(3, "big")
    return bytes(_reverse_byte(byte) for byte in state_bytes)


def _access_address_bits(access_address: int) -> np.ndarray:
    value = int(access_address)
    return np.asarray([(value >> index) & 1 for index in range(32)], dtype=np.uint8)


def _preamble_bits(access_address: int) -> np.ndarray:
    first_access_bit = int(access_address) & 1
    return np.asarray(
        [(first_access_bit + index) & 1 for index in range(8)], dtype=np.uint8
    )


@dataclass(slots=True)
class BlePacketDetectorV1Config:
    """Configuration for legacy BLE advertising packets on the LE 1M PHY."""

    channel: int = 37
    symbol_rate: float = BLE_LE_1M_SYMBOL_RATE
    access_address: int = BLE_ADVERTISING_ACCESS_ADDRESS
    crc_init: int = BLE_ADVERTISING_CRC_INIT
    chunk_size: int = 2_000_000
    min_sync_matches: int = 35
    max_payload_length: int = 37

    def __post_init__(self) -> None:
        self.channel = int(self.channel)
        self.symbol_rate = float(self.symbol_rate)
        self.access_address = int(self.access_address)
        self.crc_init = int(self.crc_init)
        self.chunk_size = int(self.chunk_size)
        self.min_sync_matches = int(self.min_sync_matches)
        self.max_payload_length = int(self.max_payload_length)
        if not 0 <= self.channel <= 39:
            raise ValueError("channel must be within [0, 39]")
        if not math.isfinite(self.symbol_rate) or self.symbol_rate <= 0:
            raise ValueError("symbol_rate must be finite and positive")
        if not 0 <= self.access_address <= 0xFFFFFFFF:
            raise ValueError("access_address must fit in 32 bits")
        if not 0 <= self.crc_init <= 0xFFFFFF:
            raise ValueError("crc_init must fit in 24 bits")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 21 <= self.min_sync_matches <= BLE_SYNC_BITS:
            raise ValueError("min_sync_matches must be within [21, 40]")
        if not 0 <= self.max_payload_length <= 37:
            raise ValueError("max_payload_length must be within [0, 37]")


@dataclass(frozen=True, slots=True)
class _SyncCandidate:
    start_sample: int
    hard_correlation: int
    soft_correlation: float
    polarity: int

    @property
    def sync_matches(self) -> int:
        return (BLE_SYNC_BITS + abs(self.hard_correlation)) // 2


@dataclass(frozen=True, slots=True)
class _DecodedPacket:
    start_sample: int
    end_sample: int
    candidate: _SyncCandidate
    pdu_type: int
    payload_length: int
    pdu: bytes
    crc: bytes


def _phase_discriminator(iq: np.ndarray) -> np.ndarray:
    if len(iq) < 2:
        return np.empty(0, dtype=np.float32)
    phase = np.angle(iq[1:] * np.conj(iq[:-1]))
    return phase.astype(np.float32, copy=False)


def _candidate_peaks(
    phase_difference: np.ndarray,
    *,
    samples_per_symbol: int,
    sync_template: np.ndarray,
    min_sync_matches: int,
) -> list[_SyncCandidate]:
    threshold = 2 * int(min_sync_matches) - BLE_SYNC_BITS
    candidates: list[_SyncCandidate] = []
    template_float = sync_template.astype(np.float32)
    template_energy = float(np.dot(template_float, template_float))

    for timing_offset in range(samples_per_symbol):
        bit_count = (len(phase_difference) - timing_offset) // samples_per_symbol
        if bit_count < BLE_SYNC_BITS:
            continue
        soft_bits = phase_difference[
            timing_offset : timing_offset + bit_count * samples_per_symbol
        ].reshape(bit_count, samples_per_symbol).sum(axis=1)
        hard_bits = np.where(soft_bits >= 0.0, 1, -1).astype(np.int8)
        correlation = np.correlate(hard_bits, sync_template, mode="valid")
        absolute = np.abs(correlation)
        peaks, _ = find_peaks(
            absolute,
            height=threshold,
            distance=max(1, BLE_SYNC_BITS // 2),
        )
        peak_indices = set(int(index) for index in peaks)
        if len(absolute) > 0 and absolute[0] >= threshold:
            peak_indices.add(0)
        if len(absolute) > 1 and absolute[-1] >= threshold:
            peak_indices.add(len(absolute) - 1)

        for bit_start in sorted(peak_indices):
            hard_correlation = int(correlation[bit_start])
            if abs(hard_correlation) < threshold:
                continue
            prefix = soft_bits[bit_start : bit_start + BLE_SYNC_BITS]
            denominator = math.sqrt(
                max(float(np.dot(prefix, prefix)) * template_energy, 1e-20)
            )
            soft_correlation = float(np.dot(prefix, template_float) / denominator)
            candidates.append(
                _SyncCandidate(
                    start_sample=timing_offset + bit_start * samples_per_symbol,
                    hard_correlation=hard_correlation,
                    soft_correlation=soft_correlation,
                    polarity=1 if hard_correlation >= 0 else -1,
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.start_sample)


def _group_candidates(
    candidates: list[_SyncCandidate], *, tolerance_samples: int
) -> list[list[_SyncCandidate]]:
    groups: list[list[_SyncCandidate]] = []
    for candidate in candidates:
        if (
            not groups
            or candidate.start_sample - groups[-1][-1].start_sample
            > tolerance_samples
        ):
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    return groups


def _decode_candidate(
    phase_difference: np.ndarray,
    candidate: _SyncCandidate,
    *,
    samples_per_symbol: int,
    sync_bits: np.ndarray,
    config: BlePacketDetectorV1Config,
) -> _DecodedPacket | None:
    maximum_bits = (
        BLE_SYNC_BITS
        + BLE_HEADER_BITS
        + 8 * config.max_payload_length
        + BLE_CRC_BITS
    )
    available_bits = min(
        maximum_bits,
        (len(phase_difference) - candidate.start_sample) // samples_per_symbol,
    )
    if available_bits < BLE_SYNC_BITS + BLE_HEADER_BITS + BLE_CRC_BITS:
        return None
    soft_bits = phase_difference[
        candidate.start_sample : candidate.start_sample
        + available_bits * samples_per_symbol
    ].reshape(available_bits, samples_per_symbol).sum(axis=1)
    air_bits = (soft_bits >= 0.0).astype(np.uint8)
    if candidate.polarity < 0:
        air_bits = 1 - air_bits
    sync_matches = int(np.sum(air_bits[:BLE_SYNC_BITS] == sync_bits))
    if sync_matches < config.min_sync_matches:
        return None

    decoded = ble_whiten_bits(air_bits[BLE_SYNC_BITS:], config.channel)
    header = _lsb_bits_to_bytes(decoded[:BLE_HEADER_BITS])
    if len(header) != 2:
        return None
    pdu_type = header[0] & 0x0F
    payload_length = header[1] & 0x3F
    if pdu_type not in _PDU_TYPE_NAMES:
        return None
    if header[1] & 0xC0:
        return None
    if payload_length > config.max_payload_length:
        return None

    packet_bits = BLE_SYNC_BITS + BLE_HEADER_BITS + 8 * payload_length + BLE_CRC_BITS
    if available_bits < packet_bits:
        return None
    decoded_byte_count = 2 + payload_length + 3
    decoded_bytes = _lsb_bits_to_bytes(decoded[: decoded_byte_count * 8])
    pdu = decoded_bytes[: 2 + payload_length]
    received_crc = decoded_bytes[2 + payload_length : decoded_byte_count]
    if ble_crc24_bytes(pdu, config.crc_init) != received_crc:
        return None

    return _DecodedPacket(
        start_sample=candidate.start_sample,
        end_sample=candidate.start_sample + packet_bits * samples_per_symbol,
        candidate=candidate,
        pdu_type=pdu_type,
        payload_length=payload_length,
        pdu=pdu,
        crc=received_crc,
    )


class BlePacketDetectorV1(SignalDetector):
    """Find unique, CRC-valid legacy BLE advertising packets."""

    name = "ble_packet_v1"

    def __init__(self, config: BlePacketDetectorV1Config | None = None):
        self.config = config or BlePacketDetectorV1Config()
        self.last_report: dict[str, Any] | None = None

    def _samples_per_symbol(self, sample_rate: float) -> int:
        ratio = float(sample_rate) / self.config.symbol_rate
        samples_per_symbol = int(round(ratio))
        if samples_per_symbol < 2 or not math.isclose(
            ratio, samples_per_symbol, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "ble_packet_v1 requires an integer sample-rate/symbol-rate "
                f"ratio of at least 2, got sample_rate={sample_rate}, "
                f"symbol_rate={self.config.symbol_rate}"
            )
        return samples_per_symbol

    def detect(self, signal: RawSignal) -> list[SignalRegion]:
        samples_per_symbol = self._samples_per_symbol(signal.sample_rate)
        sync_bits = np.concatenate(
            [
                _preamble_bits(self.config.access_address),
                _access_address_bits(self.config.access_address),
            ]
        )
        sync_template = (sync_bits.astype(np.int8) * 2 - 1).astype(np.int8)
        maximum_packet_bits = (
            BLE_SYNC_BITS
            + BLE_HEADER_BITS
            + 8 * self.config.max_payload_length
            + BLE_CRC_BITS
        )
        overlap = (
            maximum_packet_bits * samples_per_symbol
            + BLE_SYNC_BITS * samples_per_symbol
        )

        raw_candidate_count = 0
        candidate_group_count = 0
        decoded_packets: list[_DecodedPacket] = []
        chunk_count = 0

        for core_start in range(0, signal.sample_count, self.config.chunk_size):
            chunk_count += 1
            core_end = min(signal.sample_count, core_start + self.config.chunk_size)
            read_start = max(0, core_start - overlap)
            read_end = min(signal.sample_count, core_end + overlap)
            iq = signal.read_samples(read_start, read_end - read_start)
            phase_difference = _phase_discriminator(iq)
            candidates = _candidate_peaks(
                phase_difference,
                samples_per_symbol=samples_per_symbol,
                sync_template=sync_template,
                min_sync_matches=self.config.min_sync_matches,
            )
            raw_candidate_count += len(candidates)
            groups = _group_candidates(
                candidates, tolerance_samples=2 * samples_per_symbol
            )
            candidate_group_count += len(groups)

            for group in groups:
                valid: list[_DecodedPacket] = []
                for candidate in sorted(
                    group,
                    key=lambda value: (
                        value.sync_matches,
                        abs(value.soft_correlation),
                        -value.start_sample,
                    ),
                    reverse=True,
                ):
                    packet = _decode_candidate(
                        phase_difference,
                        candidate,
                        samples_per_symbol=samples_per_symbol,
                        sync_bits=sync_bits,
                        config=self.config,
                    )
                    if packet is not None:
                        valid.append(packet)
                if not valid:
                    continue
                packet = max(
                    valid,
                    key=lambda value: (
                        value.candidate.sync_matches,
                        abs(value.candidate.soft_correlation),
                        -value.start_sample,
                    ),
                )
                absolute_start = read_start + packet.start_sample
                if not core_start <= absolute_start < core_end:
                    continue
                decoded_packets.append(
                    _DecodedPacket(
                        start_sample=absolute_start,
                        end_sample=read_start + packet.end_sample,
                        candidate=_SyncCandidate(
                            start_sample=absolute_start,
                            hard_correlation=packet.candidate.hard_correlation,
                            soft_correlation=packet.candidate.soft_correlation,
                            polarity=packet.candidate.polarity,
                        ),
                        pdu_type=packet.pdu_type,
                        payload_length=packet.payload_length,
                        pdu=packet.pdu,
                        crc=packet.crc,
                    )
                )

        decoded_packets.sort(key=lambda packet: packet.start_sample)
        unique_packets: list[_DecodedPacket] = []
        for packet in decoded_packets:
            if (
                unique_packets
                and packet.start_sample - unique_packets[-1].start_sample
                <= 2 * samples_per_symbol
            ):
                previous = unique_packets[-1]
                if abs(packet.candidate.soft_correlation) > abs(
                    previous.candidate.soft_correlation
                ):
                    unique_packets[-1] = packet
            else:
                unique_packets.append(packet)

        regions: list[SignalRegion] = []
        packet_reports: list[dict[str, Any]] = []
        for packet_id, packet in enumerate(unique_packets):
            payload = packet.pdu[2:]
            advertiser_address = None
            if packet.pdu_type in {0, 2, 4, 6} and len(payload) >= 6:
                advertiser_address = ":".join(
                    f"{byte:02X}" for byte in reversed(payload[:6])
                )
            metadata = {
                "packet_id": packet_id,
                "phy": "LE_1M",
                "channel": self.config.channel,
                "access_address": f"0x{self.config.access_address:08X}",
                "samples_per_symbol": samples_per_symbol,
                "sync_matches": packet.candidate.sync_matches,
                "soft_correlation": abs(packet.candidate.soft_correlation),
                "polarity": packet.candidate.polarity,
                "pdu_type": packet.pdu_type,
                "pdu_type_name": _PDU_TYPE_NAMES[packet.pdu_type],
                "payload_length": packet.payload_length,
                "advertiser_address": advertiser_address,
                "pdu_hex": packet.pdu.hex(),
                "crc_hex": packet.crc.hex(),
                "crc_valid": True,
                "packet_sample_count": packet.end_sample - packet.start_sample,
                "timing_uncertainty_samples": samples_per_symbol - 1,
            }
            regions.append(
                SignalRegion(
                    start_sample=packet.start_sample,
                    end_sample=packet.end_sample,
                    detector=self.name,
                    score=packet.candidate.sync_matches / BLE_SYNC_BITS,
                    metadata=metadata,
                )
            )
            packet_reports.append(
                {
                    "start_sample": packet.start_sample,
                    "end_sample": packet.end_sample,
                    **metadata,
                }
            )

        self.last_report = {
            "detector": self.name,
            "config": asdict(self.config),
            "sample_rate": float(signal.sample_rate),
            "samples_per_symbol": samples_per_symbol,
            "chunk_count": chunk_count,
            "raw_sync_candidates": raw_candidate_count,
            "candidate_groups": candidate_group_count,
            "crc_valid_packets": len(regions),
            "packet_regions": packet_reports,
        }
        return regions


__all__ = [
    "BLE_ADVERTISING_ACCESS_ADDRESS",
    "BLE_ADVERTISING_CRC_INIT",
    "BLE_LE_1M_SYMBOL_RATE",
    "BlePacketDetectorV1",
    "BlePacketDetectorV1Config",
    "ble_crc24_bytes",
    "ble_whiten_bits",
]
