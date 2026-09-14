"""Unified command-line entry for versioned raw-signal preparation."""

from __future__ import annotations

import argparse
import json
import sys

from signal_fusion.preparation.contracts import (
    PreparationConfig,
    ResamplingConfig,
)
from signal_fusion.preparation.dataset import build_prepared_dataset
from signal_fusion.preparation.detectors import (
    BlePacketDetectorV1,
    EnergyDetectorV1,
    FixedBlockDetector,
    available_detectors,
    build_detector,
)
from signal_fusion.preparation.energy_v1_dataset import build_energy_v1_dataset
from signal_fusion.preparation.readers import resolve_raw_format


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-prepare",
        description="Prepare fixed-length IQ datasets from raw signal files.",
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--source_id", required=True)
    parser.add_argument(
        "--data_format",
        default="auto",
        choices=["auto", "sigmf", "mat", "dat", "bin"],
    )
    parser.add_argument(
        "--detector",
        default="energy_v1",
        choices=available_detectors(),
        help="Signal-region detection strategy, independent of input format.",
    )

    reader_group = parser.add_argument_group("raw reader")
    reader_group.add_argument("--meta_path", default=None)
    reader_group.add_argument("--x_key", default=None)
    reader_group.add_argument("--sample_rate", type=float, default=None)
    reader_group.add_argument("--center_frequency", type=float, default=None)
    reader_group.add_argument(
        "--iq_format", choices=["complex64"], default="complex64"
    )

    output_group = parser.add_argument_group("prepared dataset")
    output_group.add_argument("--label", type=int, default=None)
    output_group.add_argument("--class_name", default=None)
    output_group.add_argument("--seq_len", type=int, default=128)
    output_group.add_argument("--hop_len", type=int, default=128)
    output_group.add_argument(
        "--normalize", choices=["rms", "none", "peak"], default="rms"
    )
    output_group.add_argument(
        "--remainder", choices=["drop", "pad", "zero_pad"], default="drop"
    )
    dc_group = output_group.add_mutually_exclusive_group()
    dc_group.add_argument("--remove_dc", dest="remove_dc", action="store_true")
    dc_group.add_argument("--no_remove_dc", dest="remove_dc", action="store_false")
    parser.set_defaults(remove_dc=True)

    resampling_group = parser.add_argument_group("optional region resampling")
    resampling_group.add_argument(
        "--target_sample_rate",
        "--target-sample-rate",
        dest="target_sample_rate",
        type=float,
        default=None,
        help=(
            "Resample detected regions to this rate before windowing. "
            "Detection and segmentation remain at the native source rate."
        ),
    )

    full_signal_group = parser.add_argument_group("full_signal detector")
    full_signal_group.add_argument("--start_sample", type=int, default=0)
    full_signal_group.add_argument("--end_sample", type=int, default=None)

    fixed_blocks_group = parser.add_argument_group("fixed_blocks detector")
    fixed_blocks_group.add_argument(
        "--block_size_samples", type=int, default=4096
    )
    fixed_blocks_group.add_argument("--block_count", type=int, default=None)

    segmentation_group = parser.add_argument_group("standard segmentation")
    segmentation_group.add_argument("--min_region_samples", type=int, default=1)
    segmentation_group.add_argument("--merge_gap_samples", type=int, default=0)
    segmentation_group.add_argument("--pad_before_samples", type=int, default=0)
    segmentation_group.add_argument("--pad_after_samples", type=int, default=0)

    detector_group = parser.add_argument_group("energy_v1 detector")
    detector_group.add_argument("--chunk_size", type=int, default=1_000_000)
    detector_group.add_argument("--window_ms", type=float, default=1.0)
    detector_group.add_argument("--start_threshold_db", type=float, default=6.0)
    detector_group.add_argument("--end_threshold_db", type=float, default=5.0)
    detector_group.add_argument("--min_signal_ms", type=float, default=8.0)
    detector_group.add_argument("--min_gap_ms", type=float, default=2.0)
    detector_group.add_argument("--pad_before_ms", type=float, default=0.5)
    detector_group.add_argument("--pad_after_ms", type=float, default=0.2)
    detector_group.add_argument("--release_windows", type=int, default=2)
    detector_group.add_argument("--noise_percentile", type=float, default=20.0)
    detector_group.add_argument("--noise_probe_count", type=int, default=8)
    detector_group.add_argument("--ignore_initial_ms", type=float, default=0.0)
    detector_group.add_argument("--window_power_ratio", type=float, default=0.05)

    ble_group = parser.add_argument_group("ble_packet_v1 detector")
    ble_group.add_argument("--ble_channel", type=int, default=37)
    ble_group.add_argument("--ble_symbol_rate", type=float, default=1_000_000.0)
    ble_group.add_argument(
        "--ble_access_address",
        type=lambda value: int(value, 0),
        default=0x8E89BED6,
        help="BLE access address as decimal or 0x-prefixed hexadecimal.",
    )
    ble_group.add_argument(
        "--ble_crc_init",
        type=lambda value: int(value, 0),
        default=0x555555,
        help="BLE 24-bit CRC init as decimal or 0x-prefixed hexadecimal.",
    )
    ble_group.add_argument("--ble_chunk_size", type=int, default=2_000_000)
    ble_group.add_argument("--ble_min_sync_matches", type=int, default=35)
    ble_group.add_argument("--ble_max_payload_length", type=int, default=37)
    return parser


def _reader_options(args: argparse.Namespace) -> dict[str, object]:
    resolved_format = resolve_raw_format(args.input_path, args.data_format)
    if resolved_format == "sigmf":
        return {} if args.meta_path is None else {"metadata_path": args.meta_path}
    if resolved_format == "mat":
        options: dict[str, object] = {}
        if args.x_key is not None:
            options["x_key"] = args.x_key
        if args.sample_rate is not None:
            options["sample_rate"] = args.sample_rate
        if args.center_frequency is not None:
            options["center_frequency"] = args.center_frequency
        return options
    if args.sample_rate is None:
        raise ValueError("--sample_rate is required for DAT/BIN input")
    return {
        "sample_rate": args.sample_rate,
        "center_frequency": args.center_frequency,
        "iq_format": args.iq_format,
    }


def _resampling_config(args: argparse.Namespace) -> ResamplingConfig | None:
    if args.target_sample_rate is None:
        return None
    return ResamplingConfig(target_sample_rate=args.target_sample_rate)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        reader_options = _reader_options(args)
        resampling = _resampling_config(args)
        if args.detector == "energy_v1":
            if args.normalize == "peak":
                raise ValueError("energy_v1 supports --normalize rms or none")
            label = 0 if args.label is None else args.label
            class_name = "LoRa" if args.class_name is None else args.class_name
            detector = build_detector(
                "energy_v1",
                {
                    "chunk_size": args.chunk_size,
                    "window_ms": args.window_ms,
                    "start_threshold_db": args.start_threshold_db,
                    "end_threshold_db": args.end_threshold_db,
                    "min_signal_ms": args.min_signal_ms,
                    "min_gap_ms": args.min_gap_ms,
                    "pad_before_ms": args.pad_before_ms,
                    "pad_after_ms": args.pad_after_ms,
                    "release_windows": args.release_windows,
                    "noise_percentile": args.noise_percentile,
                    "noise_probe_count": args.noise_probe_count,
                    "ignore_initial_ms": args.ignore_initial_ms,
                    "window_power_ratio": args.window_power_ratio,
                },
            )
            if not isinstance(detector, EnergyDetectorV1):
                raise RuntimeError("energy_v1 registry returned an invalid detector")
            if resampling is None:
                if args.remainder == "pad":
                    raise ValueError(
                        "legacy energy_v1 uses --remainder zero_pad, not pad"
                    )
                result = build_energy_v1_dataset(
                    input_path=args.input_path,
                    output_path=args.output_path,
                    source_id=args.source_id,
                    data_format=args.data_format,
                    reader_options=reader_options,
                    detector=detector,
                    label=label,
                    class_name=class_name,
                    seq_len=args.seq_len,
                    hop_len=args.hop_len,
                    remove_dc=args.remove_dc,
                    normalize=args.normalize,
                    remainder=args.remainder,
                )
            else:
                remainder = (
                    "pad" if args.remainder == "zero_pad" else args.remainder
                )
                result = build_prepared_dataset(
                    input_path=args.input_path,
                    output_path=args.output_path,
                    source_id=args.source_id,
                    data_format=args.data_format,
                    reader_options=reader_options,
                    detector=detector,
                    config=PreparationConfig(
                        seq_len=args.seq_len,
                        hop_len=args.hop_len,
                        remainder=remainder,
                        normalization=args.normalize,
                        remove_dc=args.remove_dc,
                        min_region_samples=args.min_region_samples,
                        merge_gap_samples=args.merge_gap_samples,
                        pad_before_samples=args.pad_before_samples,
                        pad_after_samples=args.pad_after_samples,
                        label=label,
                        class_name=class_name,
                    ),
                    resampling=resampling,
                )
        elif args.detector == "ble_packet_v1":
            detector = build_detector(
                "ble_packet_v1",
                {
                    "channel": args.ble_channel,
                    "symbol_rate": args.ble_symbol_rate,
                    "access_address": args.ble_access_address,
                    "crc_init": args.ble_crc_init,
                    "chunk_size": args.ble_chunk_size,
                    "min_sync_matches": args.ble_min_sync_matches,
                    "max_payload_length": args.ble_max_payload_length,
                },
            )
            if not isinstance(detector, BlePacketDetectorV1):
                raise RuntimeError(
                    "ble_packet_v1 registry returned an invalid detector"
                )
            remainder = "pad" if args.remainder == "zero_pad" else args.remainder
            result = build_prepared_dataset(
                input_path=args.input_path,
                output_path=args.output_path,
                source_id=args.source_id,
                data_format=args.data_format,
                reader_options=reader_options,
                detector=detector,
                config=PreparationConfig(
                    seq_len=args.seq_len,
                    hop_len=args.hop_len,
                    remainder=remainder,
                    normalization=args.normalize,
                    remove_dc=args.remove_dc,
                    min_region_samples=args.min_region_samples,
                    merge_gap_samples=args.merge_gap_samples,
                    pad_before_samples=args.pad_before_samples,
                    pad_after_samples=args.pad_after_samples,
                    label=args.label,
                    class_name=args.class_name,
                ),
                resampling=resampling,
            )
        elif args.detector == "fixed_blocks":
            detector = build_detector(
                "fixed_blocks",
                {
                    "block_size_samples": args.block_size_samples,
                    "block_count": args.block_count,
                },
            )
            if not isinstance(detector, FixedBlockDetector):
                raise RuntimeError(
                    "fixed_blocks registry returned an invalid detector"
                )
            remainder = "pad" if args.remainder == "zero_pad" else args.remainder
            result = build_prepared_dataset(
                input_path=args.input_path,
                output_path=args.output_path,
                source_id=args.source_id,
                data_format=args.data_format,
                reader_options=reader_options,
                detector=detector,
                config=PreparationConfig(
                    seq_len=args.seq_len,
                    hop_len=args.hop_len,
                    remainder=remainder,
                    normalization=args.normalize,
                    remove_dc=args.remove_dc,
                    min_region_samples=args.min_region_samples,
                    merge_gap_samples=args.merge_gap_samples,
                    pad_before_samples=args.pad_before_samples,
                    pad_after_samples=args.pad_after_samples,
                    label=args.label,
                    class_name=args.class_name,
                ),
                resampling=resampling,
            )
        else:
            detector = build_detector(
                args.detector,
                {
                    "start_sample": args.start_sample,
                    "end_sample": args.end_sample,
                },
            )
            remainder = "pad" if args.remainder == "zero_pad" else args.remainder
            result = build_prepared_dataset(
                input_path=args.input_path,
                output_path=args.output_path,
                source_id=args.source_id,
                data_format=args.data_format,
                reader_options=reader_options,
                detector=detector,
                config=PreparationConfig(
                    seq_len=args.seq_len,
                    hop_len=args.hop_len,
                    remainder=remainder,
                    normalization=args.normalize,
                    remove_dc=args.remove_dc,
                    min_region_samples=args.min_region_samples,
                    merge_gap_samples=args.merge_gap_samples,
                    pad_before_samples=args.pad_before_samples,
                    pad_after_samples=args.pad_after_samples,
                    label=args.label,
                    class_name=args.class_name,
                ),
                resampling=resampling,
            )
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
