#!/usr/bin/env python3
"""Project standalone UrbanNav map routes into public waypoint source rows.

The output is a route-condition source for the existing waypoint exporter. It
is intentionally separate from SideWalk/Foxglove media parents: UrbanNav
``task_gt.jsonl`` has no verified video join or sensor heading. A route-start
tangent may be requested explicitly for a standalone benchmark coordinate
frame, but it is not a native heading measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic


SCHEMA_VERSION = "urbannav_standalone_waypoint_source_v1"
ROUTE_START_TANGENT = "route_start_tangent"


def build_standalone_waypoint_source(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    rejects_jsonl: str | Path,
    report_json: str | Path,
    heading_source: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert independent UrbanNav route cases to public condition rows.

    Args:
        input_jsonl: UrbanNav ``task_gt.jsonl`` source.
        output_jsonl: Public rows consumed by the waypoint exporter.
        rejects_jsonl: Deterministic row-level reject ledger.
        report_json: Conversion report destination.
        heading_source: Explicit heading policy for rows without a heading.
            ``route_start_tangent`` defines the standalone benchmark frame.
        limit: Optional number of source rows to inspect.

    Returns:
        Counts and source fingerprint for the conversion.

    Raises:
        ValueError: If a source identity is duplicated or the limit is invalid.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if heading_source not in (None, ROUTE_START_TANGENT):
        raise ValueError(f"unsupported heading_source: {heading_source}")

    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    inspected = 0
    for row_index, source in enumerate(iter_jsonl(input_jsonl), start=1):
        if limit is not None and inspected >= limit:
            break
        inspected += 1
        case_id = source.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"row {row_index}: missing string id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate UrbanNav id: {case_id}")
        seen_ids.add(case_id)
        converted, reason = _convert_row(source, heading_source=heading_source)
        if converted is None:
            rejects.append({"row_index": row_index, "sample_id": case_id, "reason": reason})
        else:
            rows.append(converted)

    write_jsonl_atomic(output_jsonl, rows)
    write_jsonl_atomic(rejects_jsonl, rejects)
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_role": "standalone_independent_benchmark_route_source",
        "input_jsonl": str(input_jsonl),
        "input_jsonl_sha256": file_sha256(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "rejects_jsonl": str(rejects_jsonl),
        "heading_source": heading_source,
        "input_rows": inspected,
        "accepted_rows": len(rows),
        "rejected_rows": len(rejects),
        "current_clip_training_join": {
            "status": "not_joined",
            "accepted_training_rows": 0,
            "reason": "standalone UrbanNav cases do not bind to SideWalk/Foxglove/EgoWalk media parents",
        },
        "reject_counts": _count_rejects(rejects),
    }
    report_path = Path(report_json).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _convert_row(source: dict[str, Any], *, heading_source: str | None) -> tuple[dict[str, Any] | None, str]:
    origin = _point(source.get("origin"))
    route = [_point(item) for item in source.get("gt_route") or []]
    route = [point for point in route if point is not None]
    if origin is None:
        return None, "missing_origin_gps"
    if len(route) < 2:
        return None, "missing_gt_route"
    heading, resolved_source = _heading(source, origin, route, heading_source)
    if heading is None:
        return None, "missing_current_heading"
    provider = _route_provider(source)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": source["id"],
        "sample_id": source["id"],
        "parent_sample_id": source["id"],
        "source_record_id": source["id"],
        "source_dataset": "urbannav_standalone_benchmark",
        "dataset_source": "urbannav_standalone_benchmark",
        "source_scope": "standalone_benchmark",
        "instruction": source.get("instruction"),
        "city": source.get("city"),
        "scene_id": source.get("scene_id"),
        "category": source.get("category"),
        "subcategory": source.get("subcategory"),
        "current_gps": {"lat": origin[0], "lon": origin[1]},
        "current_heading_rad": heading,
        "route_context": {
            "route_waypoints_gps": [{"lat": lat, "lon": lon} for lat, lon in route],
            "route_source": provider,
            "source_type": "standalone_benchmark_gt_route",
            "source_record_id": source["id"],
            "heading_source": resolved_source,
            "independent_of_future_target": True,
        },
    }, "accepted"


def _heading(
    source: dict[str, Any],
    origin: tuple[float, float],
    route: list[tuple[float, float]],
    heading_source: str | None,
) -> tuple[float | None, str]:
    for key in ("origin_heading_rad", "current_heading_rad", "heading_rad"):
        value = _finite(source.get(key))
        if value is not None:
            return value, key
    for key in ("origin_heading_deg", "current_heading_deg", "heading_deg"):
        value = _finite(source.get(key))
        if value is not None:
            return math.radians(value), key
    if heading_source != ROUTE_START_TANGENT:
        return None, "unavailable"
    for latitude, longitude in route[1:]:
        north = math.radians(latitude - origin[0])
        east = math.radians(longitude - origin[1]) * math.cos(math.radians(origin[0]))
        if math.hypot(north, east) > 1e-12:
            return math.atan2(east, north), ROUTE_START_TANGENT
    return None, "unavailable"


def _route_provider(source: dict[str, Any]) -> str:
    scope = source.get("optimality_scope")
    if isinstance(scope, dict) and scope.get("route_provider"):
        return str(scope["route_provider"])
    metadata = source.get("metadata")
    if isinstance(metadata, dict):
        provider = metadata.get("provider")
        if isinstance(provider, dict) and provider.get("default_router_type"):
            return str(provider["default_router_type"])
    return "urbannav_gt_route"


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    latitude = _finite(value.get("lat", value.get("latitude")))
    longitude = _finite(value.get("lon", value.get("lng", value.get("longitude"))))
    if latitude is None or longitude is None or not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        return None
    return latitude, longitude


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _count_rejects(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the standalone route conversion CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--rejects-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--heading-source", choices=(ROUTE_START_TANGENT,))
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the standalone route conversion and print its report."""

    args = parse_args(argv)
    report = build_standalone_waypoint_source(
        args.input_jsonl,
        args.output_jsonl,
        rejects_jsonl=args.rejects_jsonl,
        report_json=args.report_json,
        heading_source=args.heading_source,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
