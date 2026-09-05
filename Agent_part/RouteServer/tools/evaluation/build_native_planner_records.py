#!/usr/bin/env python3
"""Build fail-closed native localization anchors and planner-route records.

The ``foxglove-anchors`` operation extracts only the current GPS/heading
anchor from the native recording.  It does not expose a route.  The
``planner-records`` operation accepts an independently logged planner route
only after timestamp and pose-consistency checks pass.
"""

from __future__ import annotations

import argparse
import json
import math

from mcp_server.contracts.language_waypoint_benchmark import NativePlannerAdapterConfigV1
from mcp_server.data.benchmark.native_planner_adapter import (
    build_foxglove_native_gps_anchors_from_source_v1,
    build_foxglove_native_gps_anchors_from_stage_i_views_v1,
    build_planner_records_from_native_logs_v1,
    build_sidewalk_native_gps_anchors_v1,
)


def main() -> int:
    """Dispatch one versioned offline native-planner adapter operation.

    Returns:
        Process exit code zero after writing deterministic adapter artifacts.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    anchor_parser = subparsers.add_parser("foxglove-anchors")
    anchor_parser.add_argument("--source-jsonl", required=True)
    anchor_parser.add_argument("--foxglove-raw-root", required=True)
    anchor_parser.add_argument("--output-jsonl", required=True)
    anchor_parser.add_argument("--report-json", required=True)
    anchor_parser.add_argument("--heading-lookback-frames", type=int, default=120)
    anchor_parser.add_argument("--min-heading-displacement-m", type=float, default=0.25)
    anchor_parser.add_argument("--timestamp-scale-to-seconds", type=float, default=1e-9)
    sidewalk_anchor_parser = subparsers.add_parser("sidewalk-anchors")
    sidewalk_anchor_parser.add_argument("--source-jsonl", required=True)
    sidewalk_anchor_parser.add_argument("--sidewalk-raw-root", required=True)
    sidewalk_anchor_parser.add_argument("--output-jsonl", required=True)
    sidewalk_anchor_parser.add_argument("--report-json", required=True)
    sidewalk_anchor_parser.add_argument("--heading-lookback-frames", type=int, default=120)
    sidewalk_anchor_parser.add_argument("--min-heading-displacement-m", type=float, default=0.25)
    sidewalk_anchor_parser.add_argument("--timestamp-scale-to-seconds", type=float, default=1e-9)
    view_anchor_parser = subparsers.add_parser("stage-i-view-anchors")
    view_anchor_parser.add_argument("--source-jsonl", required=True)
    view_anchor_parser.add_argument("--foxglove-raw-root", required=True)
    view_anchor_parser.add_argument("--output-jsonl", required=True)
    view_anchor_parser.add_argument("--report-json", required=True)
    view_anchor_parser.add_argument("--heading-lookback-frames", type=int, default=120)
    view_anchor_parser.add_argument("--min-heading-displacement-m", type=float, default=0.25)
    view_anchor_parser.add_argument("--timestamp-scale-to-seconds", type=float, default=1e-9)
    view_anchor_parser.add_argument("--limit", type=int)
    route_parser = subparsers.add_parser("planner-records")
    route_parser.add_argument("--anchor-jsonl", required=True)
    route_parser.add_argument("--planner-log-jsonl", required=True)
    route_parser.add_argument("--output-jsonl", required=True)
    route_parser.add_argument("--decisions-jsonl", required=True)
    route_parser.add_argument("--report-json", required=True)
    route_parser.add_argument("--max-plan-age-s", type=float, default=10.0)
    route_parser.add_argument("--max-origin-distance-m", type=float, default=15.0)
    route_parser.add_argument("--max-heading-delta-deg", type=float, default=45.0)
    args = parser.parse_args()

    if args.command == "foxglove-anchors":
        report = build_foxglove_native_gps_anchors_from_source_v1(
            source_jsonl=args.source_jsonl,
            foxglove_raw_root=args.foxglove_raw_root,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            heading_lookback_frames=args.heading_lookback_frames,
            min_heading_displacement_m=args.min_heading_displacement_m,
            timestamp_scale_to_seconds=args.timestamp_scale_to_seconds,
        )
    elif args.command == "sidewalk-anchors":
        report = build_sidewalk_native_gps_anchors_v1(
            source_jsonl=args.source_jsonl,
            sidewalk_raw_root=args.sidewalk_raw_root,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            heading_lookback_frames=args.heading_lookback_frames,
            min_heading_displacement_m=args.min_heading_displacement_m,
            timestamp_scale_to_seconds=args.timestamp_scale_to_seconds,
        )
    elif args.command == "stage-i-view-anchors":
        report = build_foxglove_native_gps_anchors_from_stage_i_views_v1(
            source_jsonl=args.source_jsonl,
            foxglove_raw_root=args.foxglove_raw_root,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            heading_lookback_frames=args.heading_lookback_frames,
            min_heading_displacement_m=args.min_heading_displacement_m,
            timestamp_scale_to_seconds=args.timestamp_scale_to_seconds,
            limit=args.limit,
        )
    else:
        if args.max_heading_delta_deg <= 0.0 or args.max_heading_delta_deg > 180.0:
            parser.error("--max-heading-delta-deg must be in (0, 180]")
        report = build_planner_records_from_native_logs_v1(
            anchor_jsonl=args.anchor_jsonl,
            planner_log_jsonl=args.planner_log_jsonl,
            output_jsonl=args.output_jsonl,
            decisions_jsonl=args.decisions_jsonl,
            report_json=args.report_json,
            config=NativePlannerAdapterConfigV1(
                max_plan_age_s=args.max_plan_age_s,
                max_origin_distance_m=args.max_origin_distance_m,
                max_heading_delta_rad=math.radians(args.max_heading_delta_deg),
            ),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
