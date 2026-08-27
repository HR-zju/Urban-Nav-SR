#!/usr/bin/env python3
"""Build typed local-planner route logs for an explicit waypoint smoke set.

The route destination is constructed from the current native pose and a fixed
mission pattern.  It is a map-planner smoke source, not the recorded future
trajectory and not a replacement for operator missions in the production
dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mcp_server.contracts.language_waypoint_benchmark import (
    NativePlannerAnchorV1,
    NativePlannerRouteLogV1,
    Wgs84CoordinateV1,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic


EARTH_RADIUS_M = 6_378_137.0
MISSION_OFFSETS_DEG = (0.0, 30.0, -30.0, 60.0, -60.0)
MISSION_DISTANCES_M = (24.0, 32.0, 40.0)


def build_native_planner_route_logs(
    *,
    anchor_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    osrm_url: str,
    osrm_profile: str = "foot",
    timeout_s: float = 10.0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Query one frozen local OSRM planner for deterministic smoke missions.

    Args:
        anchor_jsonl: Native current-pose anchor ledger.
        output_jsonl: Typed ``NativePlannerRouteLogV1`` destination.
        report_json: Row and request audit destination.
        osrm_url: Local OSRM base URL, for example ``http://127.0.0.1:5000``.
        osrm_profile: OSRM routing profile.
        timeout_s: Per-request HTTP timeout.
        limit: Optional number of anchor rows to inspect.

    Returns:
        Deterministic request and route counts.

    Raises:
        ValueError: If the planner returns malformed route geometry.
        OSError: If the local planner cannot be reached.
    """

    if timeout_s <= 0.0 or not math.isfinite(timeout_s):
        raise ValueError("timeout_s must be positive and finite")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    logs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    requests: list[dict[str, Any]] = []
    for index, payload in enumerate(iter_jsonl(anchor_jsonl)):
        if limit is not None and index >= limit:
            break
        anchor = NativePlannerAnchorV1.model_validate(payload)
        counts["anchor_rows"] += 1
        if not anchor.available:
            counts[f"reject:{anchor.reason}"] += 1
            continue
        assert anchor.current_wgs84 is not None
        assert anchor.current_heading_rad is not None
        offset_deg = MISSION_OFFSETS_DEG[index % len(MISSION_OFFSETS_DEG)]
        distance_m = MISSION_DISTANCES_M[index % len(MISSION_DISTANCES_M)]
        goal = _destination(
            anchor.current_wgs84,
            bearing_rad=anchor.current_heading_rad + math.radians(offset_deg),
            distance_m=distance_m,
        )
        route = _request_route(
            base_url=osrm_url,
            profile=osrm_profile,
            origin=anchor.current_wgs84,
            destination=goal,
            timeout_s=timeout_s,
        )
        counts["planner_requests"] += 1
        request_record = {
            "parent_sample_id": anchor.parent_sample_id,
            "source_record_id": anchor.source_record_id,
            "mission_offset_deg": offset_deg,
            "mission_distance_m": distance_m,
            "requested_goal": goal.model_dump(mode="json"),
            "route_distance_m": route["distance_m"],
        }
        requests.append(request_record)
        route_log = NativePlannerRouteLogV1(
            route_log_id=f"local_osrm_waypoint_smoke:{anchor.parent_sample_id}",
            dataset_source=anchor.dataset_source,
            source_record_id=anchor.source_record_id,
            plan_timestamp_s=anchor.observation_timestamp_s,
            current_wgs84=anchor.current_wgs84,
            current_heading_rad=anchor.current_heading_rad,
            goal_wgs84=route["polyline"][-1],
            planner_polyline_wgs84=route["polyline"],
            planner_id="osrm_local_constructed_waypoint_smoke",
            route_version="urbannav_waypoint_smoke_v1",
            datum="WGS84",
            source_kind="map_planner",
            independent_of_future_target=True,
        )
        logs.append(route_log.model_dump(mode="json"))
        counts["accepted_routes"] += 1

    write_jsonl_atomic(output_jsonl, logs)
    report = {
        "schema_version": "urbannav_native_planner_route_log_smoke_report_v1",
        "source_role": "constructed_local_map_mission_smoke",
        "independence_statement": (
            "destinations are generated from current native pose and a fixed mission pattern; "
            "recorded future GPS and trajectory targets are not read"
        ),
        "anchor_jsonl": str(anchor_jsonl),
        "output_jsonl": str(output_jsonl),
        "osrm_url": osrm_url.rstrip("/"),
        "osrm_profile": osrm_profile,
        "mission_offsets_deg": list(MISSION_OFFSETS_DEG),
        "mission_distances_m": list(MISSION_DISTANCES_M),
        "counts": dict(sorted(counts.items())),
        "requests": requests,
    }
    target = Path(report_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _destination(origin: Wgs84CoordinateV1, *, bearing_rad: float, distance_m: float) -> Wgs84CoordinateV1:
    """Project a short explicit mission offset on a local tangent plane."""

    north = distance_m * math.cos(bearing_rad)
    east = distance_m * math.sin(bearing_rad)
    latitude = origin.latitude_deg + math.degrees(north / EARTH_RADIUS_M)
    longitude = origin.longitude_deg + math.degrees(
        east / (EARTH_RADIUS_M * math.cos(math.radians(origin.latitude_deg)))
    )
    return Wgs84CoordinateV1(latitude_deg=latitude, longitude_deg=longitude)


def _request_route(
    *,
    base_url: str,
    profile: str,
    origin: Wgs84CoordinateV1,
    destination: Wgs84CoordinateV1,
    timeout_s: float,
) -> dict[str, Any]:
    """Request one route and retain only its metric and WGS84 polyline."""

    coordinates = (
        f"{origin.longitude_deg:.8f},{origin.latitude_deg:.8f};"
        f"{destination.longitude_deg:.8f},{destination.latitude_deg:.8f}"
    )
    query = urlencode({"overview": "full", "geometries": "geojson", "steps": "false"})
    url = f"{base_url.rstrip('/')}/route/v1/{profile}/{coordinates}?{query}"
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.load(response)
    if payload.get("code") != "Ok" or not payload.get("routes"):
        raise ValueError(f"local OSRM returned no route: {payload.get('code')!r}")
    route = payload["routes"][0]
    raw_coordinates = ((route.get("geometry") or {}).get("coordinates") or [])
    polyline = [
        Wgs84CoordinateV1(latitude_deg=float(point[1]), longitude_deg=float(point[0]))
        for point in raw_coordinates
        if isinstance(point, list) and len(point) >= 2
    ]
    if len(polyline) < 2:
        raise ValueError("local OSRM returned a route with fewer than two coordinates")
    return {"distance_m": float(route["distance"]), "polyline": polyline}


def main() -> int:
    """Parse CLI arguments and build one route-log ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--osrm-url", required=True)
    parser.add_argument("--osrm-profile", default="foot")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = build_native_planner_route_logs(
        anchor_jsonl=args.anchor_jsonl,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        osrm_url=args.osrm_url,
        osrm_profile=args.osrm_profile,
        timeout_s=args.timeout_s,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
