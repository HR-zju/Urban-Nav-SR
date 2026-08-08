#!/usr/bin/env python3
"""Project one frozen legacy UrbanNav Stage-II package into plugin V1 artifacts."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.benchmark.urbannav_dataset_plugin import (
    build_urbannav_stage_ii_benchmark_package_v1,
    urbannav_stage_ii_default_profile_v1,
)


def main() -> int:
    """Parse projection arguments, write one package, and print its report.

    Returns:
        Process exit code zero after successful immutable package creation.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-input-jsonl", required=True)
    parser.add_argument("--legacy-reference-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--profile-id", default="urbannav_stage_ii_openloop_v1")
    parser.add_argument("--horizon-s", type=float, action="append", default=None)
    args = parser.parse_args()
    profile = urbannav_stage_ii_default_profile_v1(
        profile_id=args.profile_id,
        horizons_s=tuple(args.horizon_s) if args.horizon_s else (0.5, 1.0, 2.0),
    )
    report = build_urbannav_stage_ii_benchmark_package_v1(
        legacy_input_jsonl=args.legacy_input_jsonl,
        legacy_reference_jsonl=args.legacy_reference_jsonl,
        output_root=args.output_root,
        split=args.split,
        profile=profile,
        package_id=args.package_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
