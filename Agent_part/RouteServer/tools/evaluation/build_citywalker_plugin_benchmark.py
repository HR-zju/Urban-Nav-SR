#!/usr/bin/env python3
"""Build one frozen CityWalker benchmark-plugin package offline."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.benchmark.citywalker_dataset_plugin import (
    build_citywalker_benchmark_package_v1,
    citywalker_default_profile_v1,
)


def main() -> int:
    """Parse build arguments, materialize CityWalker artifacts, and print JSON.

    Returns:
        Process exit code zero after a successful immutable package build.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-label-dir", required=True)
    parser.add_argument("--image-root-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--profile-id", default="citywalker_teleop_openloop_v1")
    parser.add_argument("--horizon-s", type=float, action="append", default=None)
    parser.add_argument("--target-seed", type=int, default=20260808)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()
    profile = citywalker_default_profile_v1(
        profile_id=args.profile_id,
        horizons_s=tuple(args.horizon_s) if args.horizon_s else (0.5, 1.0, 2.0, 3.0),
    )
    report = build_citywalker_benchmark_package_v1(
        pose_label_dir=args.pose_label_dir,
        image_root_dir=args.image_root_dir,
        output_root=args.output_root,
        split=args.split,
        profile=profile,
        package_id=args.package_id,
        target_seed=args.target_seed,
        allow_missing_images=args.allow_missing_images,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
