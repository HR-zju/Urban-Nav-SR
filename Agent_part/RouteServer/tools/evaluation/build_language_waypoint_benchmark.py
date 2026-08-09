#!/usr/bin/env python3
"""Build typed language-goal candidates, planner guidance, or frozen packages."""

from __future__ import annotations

import argparse
import json

from mcp_server.contracts.language_waypoint_benchmark import (
    LanguageWaypointBenchmarkBuildConfigV1,
)
from mcp_server.data.benchmark.language_waypoint_dataset_plugin import (
    build_language_goal_candidates_v1,
    build_language_waypoint_package_v1,
    build_waypoint_guidance_manifest_v1,
    language_waypoint_default_profile_v1,
)


def main() -> int:
    """Dispatch a versioned offline preprocessing operation."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    candidate_parser = subparsers.add_parser("candidates")
    candidate_parser.add_argument("--source-jsonl", required=True)
    candidate_parser.add_argument("--trajectory-sidecar-jsonl", required=True)
    candidate_parser.add_argument("--output-jsonl", required=True)
    candidate_parser.add_argument("--report-json", required=True)
    candidate_parser.add_argument("--candidate-time-s", type=float, default=4.0)
    guidance_parser = subparsers.add_parser("guidance")
    guidance_parser.add_argument("--planner-route-jsonl", required=True)
    guidance_parser.add_argument("--output-jsonl", required=True)
    guidance_parser.add_argument("--report-json", required=True)
    _add_route_sampling_args(guidance_parser)
    guidance_parser.add_argument("--skip-errors", action="store_true")
    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--source-jsonl", required=True)
    package_parser.add_argument("--trajectory-sidecar-jsonl", required=True)
    package_parser.add_argument("--language-annotation-jsonl")
    package_parser.add_argument("--waypoint-guidance-jsonl")
    package_parser.add_argument("--output-root", required=True)
    package_parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    package_parser.add_argument("--package-id", required=True)
    package_parser.add_argument("--arm", choices=("fpv_only", "fpv_language", "fpv_waypoint", "fpv_language_waypoint"), required=True)
    package_parser.add_argument("--include-bev", action="store_true")
    package_parser.add_argument("--required-fpv-frame-count", type=int, default=5)
    _add_route_sampling_args(package_parser)
    package_parser.add_argument("--no-require-language-annotation-for-controls", action="store_true")
    package_parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()
    if args.command == "candidates":
        report = build_language_goal_candidates_v1(
            source_jsonl=args.source_jsonl,
            trajectory_sidecar_jsonl=args.trajectory_sidecar_jsonl,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            candidate_time_s=args.candidate_time_s,
        )
    elif args.command == "guidance":
        report = build_waypoint_guidance_manifest_v1(
            planner_route_jsonl=args.planner_route_jsonl,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            route_step_m=args.public_route_step_m,
            route_horizon_m=args.public_route_horizon_m,
            route_max_points=args.public_route_max_points,
            local_goal_distance_m=args.public_local_goal_distance_m,
            skip_errors=args.skip_errors,
        )
    else:
        config = LanguageWaypointBenchmarkBuildConfigV1(
            arm=args.arm,
            include_bev=args.include_bev,
            required_fpv_frame_count=args.required_fpv_frame_count,
            public_route_horizon_m=args.public_route_horizon_m,
            public_route_step_m=args.public_route_step_m,
            public_route_max_points=args.public_route_max_points,
            public_local_goal_distance_m=args.public_local_goal_distance_m,
            require_language_annotation_for_controls=not args.no_require_language_annotation_for_controls,
            allow_missing_images=args.allow_missing_images,
        )
        profile = language_waypoint_default_profile_v1(
            arm=config.arm,
            include_bev=config.include_bev,
        )
        report = build_language_waypoint_package_v1(
            source_jsonl=args.source_jsonl,
            trajectory_sidecar_jsonl=args.trajectory_sidecar_jsonl,
            language_annotation_jsonl=args.language_annotation_jsonl,
            waypoint_guidance_jsonl=args.waypoint_guidance_jsonl,
            output_root=args.output_root,
            split=args.split,
            package_id=args.package_id,
            profile=profile,
            config=config,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _add_route_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Register one shared public planner-normalization policy."""

    parser.add_argument("--public-route-step-m", type=float, default=2.0)
    parser.add_argument("--public-route-horizon-m", type=float, default=30.0)
    parser.add_argument("--public-route-max-points", type=int, default=8)
    parser.add_argument("--public-local-goal-distance-m", type=float, default=10.0)


if __name__ == "__main__":
    raise SystemExit(main())
