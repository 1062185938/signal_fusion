"""Command line entry for the public LTE/WiFi/DVB-T benchmark."""

from __future__ import annotations

import argparse
import json

from signal_fusion.benchmarks.technology_recognition import (
    ALL_REGIONS_DATASET_ID,
    REGION_SIZE,
    REGIONS_PER_SOURCE,
    WINDOW_SIZES,
    build_v1_inventory,
    prepare_v1_sources,
    profile_directory,
    write_ablation_manifests,
    write_inventory,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and prepare the public LTE/WiFi/DVB-T benchmark."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    inventory = build_v1_inventory(args.dataset_root)
    if args.all_regions:
        inventory["dataset_id"] = ALL_REGIONS_DATASET_ID
        selected_sample_count = next(
            entry["complex_sample_count"]
            for entry in inventory["recordings"]
            if entry["selected_for_v1"]
        )
        inventory["preparation"]["regions_per_source"] = (
            selected_sample_count // REGION_SIZE
        )
        inventory["preparation"]["region_selection"] = "all_complete_blocks"
    inventory_path = write_inventory(
        inventory, f"{args.output_dir}/inventory.json"
    )
    result: dict[str, object] = {
        "inventory_path": str(inventory_path),
        "selected_recordings": inventory["v1_selection"]["selected_recordings"],
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
                    None if args.all_regions else REGIONS_PER_SOURCE
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
    if args.manifests_dir is not None:
        manifest_paths = write_ablation_manifests(
            inventory,
            f"{args.output_dir}/profiles",
            args.manifests_dir,
            window_sizes=tuple(args.window_sizes),
        )
        result["manifests"] = [str(path) for path in manifest_paths]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
