"""Command line entry for the public LTE/WiFi/DVB-T benchmark."""

from __future__ import annotations

import argparse
import json

from signal_fusion.benchmarks.technology_recognition import (
    ALL_REGIONS_DATASET_ID,
    REGION_SIZE,
    REGIONS_PER_SOURCE,
    WINDOW_SIZES,
    assemble_external_test_dataset,
    build_v1_inventory,
    prepare_v1_sources,
    profile_directory,
    select_external_1msps_recordings,
    write_ablation_manifests,
    write_all_location_manifest,
    write_inventory,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and prepare the public LTE/WiFi/DVB-T benchmark."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--recording-set",
        choices=("core", "external-1msps"),
        default="core",
        help="Choose the benchmark recording set to audit or prepare.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create prepared sources for every requested window size.",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        choices=WINDOW_SIZES,
        default=list(WINDOW_SIZES),
        help="Non-overlapping window lengths used within each 4096-point region.",
    )
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help=(
            "Use every complete 4096-point region instead of uniformly "
            "selecting 32 regions per recording."
        ),
    )
    parser.add_argument(
        "--manifests-dir",
        default=None,
        help="Write all four location-fold manifests to this directory.",
    )
    parser.add_argument(
        "--all-location-manifest",
        default=None,
        help=(
            "Write one source-disjoint manifest using runs 1-8 for training, "
            "run 9 for validation, and run 10 for internal testing."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    inventory = build_v1_inventory(args.dataset_root)
    if args.recording_set == "external-1msps":
        inventory = select_external_1msps_recordings(inventory)
        if args.manifests_dir is not None or args.all_location_manifest is not None:
            parser.error("manifests are only valid for the core recording set")
        if args.prepare and args.window_sizes != [4096]:
            parser.error("external-1msps preparation requires --window-sizes 4096")
    if args.manifests_dir is not None and args.all_location_manifest is not None:
        parser.error(
            "--manifests-dir and --all-location-manifest are mutually exclusive"
        )
    if args.all_location_manifest is not None:
        if not args.all_regions:
            parser.error("--all-location-manifest requires --all-regions")
        if len(args.window_sizes) != 1:
            parser.error(
                "--all-location-manifest requires exactly one --window-sizes value"
            )

    use_all_regions = args.all_regions or args.recording_set == "external-1msps"
    if use_all_regions:
        if args.recording_set == "core":
            inventory["dataset_id"] = ALL_REGIONS_DATASET_ID
        selected_region_counts = {
            entry["complex_sample_count"] // REGION_SIZE
            for entry in inventory["recordings"]
            if entry["selected_for_v1"]
        }
        inventory["preparation"]["regions_per_source"] = (
            selected_region_counts.pop()
            if len(selected_region_counts) == 1
            else None
        )
        inventory["preparation"]["region_selection"] = "all_complete_blocks"
    inventory_path = write_inventory(
        inventory, f"{args.output_dir}/inventory.json"
    )
    result: dict[str, object] = {
        "inventory_path": str(inventory_path),
        "recording_set": args.recording_set,
        "selected_recordings": sum(
            bool(entry["selected_for_v1"])
            for entry in inventory["recordings"]
        ),
        "region_selection": inventory["preparation"]["region_selection"],
    }
    if args.prepare:
        prepared_profiles = []
        for window_size in args.window_sizes:
            profile_dir = profile_directory(args.output_dir, window_size)
            report = prepare_v1_sources(
                inventory,
                profile_dir,
                window_size=window_size,
                region_count_per_source=(
                    None if use_all_regions else REGIONS_PER_SOURCE
                ),
            )
            prepared_profiles.append(
                {
                    "window_size": window_size,
                    "prepared_source_count": report["prepared_source_count"],
                    "total_regions": report["total_regions"],
                    "total_windows": report["total_windows"],
                    "preparation_report": str(
                        profile_dir / "preparation_report.json"
                    ),
                }
            )
        result["prepared_profiles"] = prepared_profiles
        if args.recording_set == "external-1msps":
            result["external_test"] = assemble_external_test_dataset(
                inventory,
                profile_directory(args.output_dir, 4096)
                / "prepared_sources",
                f"{args.output_dir}/test.npz",
            )
    if args.manifests_dir is not None:
        manifest_paths = write_ablation_manifests(
            inventory,
            f"{args.output_dir}/profiles",
            args.manifests_dir,
            window_sizes=tuple(args.window_sizes),
        )
        result["manifests"] = [str(path) for path in manifest_paths]
    if args.all_location_manifest is not None:
        window_size = args.window_sizes[0]
        result["all_location_manifest"] = str(
            write_all_location_manifest(
                inventory,
                profile_directory(args.output_dir, window_size)
                / "prepared_sources",
                args.all_location_manifest,
                window_size=window_size,
            )
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
