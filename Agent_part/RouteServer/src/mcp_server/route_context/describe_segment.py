"""Convert route progress into local navigation semantics.

This module turns low-level route matching into the compact intent fields used
by the visual prompt generator: turn type, corridor direction, instruction
hint, and the near-field route polyline.

Core rules:

    lookbehind = min(10 m, 0.5 * lookahead)
    local_segment = route[s_match - lookbehind, s_match + lookahead]

    signed_delta = wrap(local_route_bearing - current_heading)

    corridor_type =
        front              if |signed_delta| <= 20 deg
        slight_left/right  if 20 deg < |signed_delta| <= 60 deg
        left/right_front   otherwise

The output is intentionally language-light: it carries enough semantics for the
LVLM to choose a visible support region, while segmentation remains the hard
safety authority.
"""

from __future__ import annotations

import re
from typing import Any

from .common import (
    bearing_deg,
    clamp,
    normalize_coarse_waypoints,
    normalize_route_polyline,
    project_point_to_polyline,
    route_cumulative_distances,
    signed_heading_delta_deg,
)
from .schema import LocalGeometry, LocalSemantics, PoseMatch, RoutePoint, RouteProgress


def _interpolate_point_at_distance(
    route_points: list[RoutePoint],
    cumulative_distances: list[float],
    target_distance_m: float,
) -> RoutePoint:
    """Interpolate a route point at arc-length ``target_distance_m``."""
    if not route_points:
        raise ValueError("route_polyline is empty")
    if len(route_points) == 1:
        return route_points[0]

    total_m = cumulative_distances[-1]
    target = clamp(target_distance_m, 0.0, total_m)
    if target <= 0.0:
        return route_points[0]
    if target >= total_m:
        return route_points[-1]

    for index in range(len(route_points) - 1):
        start_distance = cumulative_distances[index]
        end_distance = cumulative_distances[index + 1]
        if target > end_distance:
            continue
        segment_distance = max(end_distance - start_distance, 1e-6)
        ratio = clamp((target - start_distance) / segment_distance, 0.0, 1.0)
        start = route_points[index]
        end = route_points[index + 1]
        return RoutePoint(
            lat=start.lat + (end.lat - start.lat) * ratio,
            lng=start.lng + (end.lng - start.lng) * ratio,
        )
    return route_points[-1]


def _dedupe_points(points: list[RoutePoint]) -> list[RoutePoint]:
    deduped: list[RoutePoint] = []
    for point in points:
        if deduped and abs(deduped[-1].lat - point.lat) < 1e-9 and abs(deduped[-1].lng - point.lng) < 1e-9:
            continue
        deduped.append(point)
    return deduped


def _connector_polyline(
    *,
    current_point: RoutePoint,
    route_start: RoutePoint,
    merge_point: RoutePoint,
) -> list[RoutePoint]:
    points = [current_point]
    if abs(current_point.lat - route_start.lat) > 1e-9 or abs(current_point.lng - route_start.lng) > 1e-9:
        points.append(route_start)
    if abs(points[-1].lat - merge_point.lat) > 1e-9 or abs(points[-1].lng - merge_point.lng) > 1e-9:
        points.append(merge_point)
    return _dedupe_points(points)


def _slice_local_segment(
    route_points: list[RoutePoint],
    cumulative_distances: list[float],
    matched_distance_m: float,
    lookahead_m: float,
) -> list[RoutePoint]:
    """Return a short route slice around the matched pose for local planning."""
    total_m = cumulative_distances[-1] if cumulative_distances else 0.0
    lookahead = max(float(lookahead_m), 5.0)
    lookbehind = min(10.0, lookahead * 0.5)
    start_distance = max(0.0, matched_distance_m - lookbehind)
    end_distance = min(total_m, matched_distance_m + lookahead)

    points = [_interpolate_point_at_distance(route_points, cumulative_distances, start_distance)]
    for index, point in enumerate(route_points):
        route_distance = cumulative_distances[index]
        if start_distance < route_distance < end_distance:
            points.append(point)
    points.append(_interpolate_point_at_distance(route_points, cumulative_distances, end_distance))
    return _dedupe_points(points)


def _local_bearing_deg(local_points: list[RoutePoint]) -> float | None:
    if len(local_points) < 2:
        return None
    start = local_points[0]
    for candidate in local_points[1:]:
        if abs(candidate.lat - start.lat) < 1e-9 and abs(candidate.lng - start.lng) < 1e-9:
            continue
        return bearing_deg(start.lat, start.lng, candidate.lat, candidate.lng)
    return None


def _corridor_type(
    *,
    signed_heading_delta: float | None,
    turn_type: str,
    off_route: bool,
) -> str:
    """Map heading-route disagreement into a coarse visual corridor sector."""
    if off_route:
        return "uncertain"

    if signed_heading_delta is None:
        if "left" in turn_type:
            return "left_front"
        if "right" in turn_type:
            return "right_front"
        return "front"

    abs_delta = abs(float(signed_heading_delta))
    if abs_delta <= 20.0:
        return "front"
    if abs_delta <= 60.0:
        return "slight_left_front" if signed_heading_delta > 0 else "slight_right_front"
    return "left_front" if signed_heading_delta > 0 else "right_front"


def _turn_semantics(turn_type: str, distance_to_turn_m: float | None, off_route: bool) -> str:
    if off_route:
        return "off_route_recovery"

    action = str(turn_type or "go_straight")
    distance = float(distance_to_turn_m) if distance_to_turn_m is not None else None

    if action == "arrive":
        if distance is not None and distance <= 8.0:
            return "arrival_imminent"
        if distance is not None and distance <= 25.0:
            return "approach_destination"
        return "destination_ahead"

    if action == "turn_left":
        return "approach_left_turn" if distance is None or distance <= 35.0 else "left_turn_ahead"
    if action == "turn_right":
        return "approach_right_turn" if distance is None or distance <= 35.0 else "right_turn_ahead"
    if action in {"keep_left", "bear_left"}:
        return "approach_left_bearing"
    if action in {"keep_right", "bear_right"}:
        return "approach_right_bearing"
    if action == "u_turn":
        return "approach_u_turn"
    return "continue_straight"


def _looks_english_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return re.search(r"[\u3400-\u9fff]", raw) is None


def _short_action_summary(turn_type: str, distance_to_turn_m: float | None, off_route: bool) -> str:
    if off_route:
        return "You may be off route. Return to the planned path first."
    if turn_type == "arrive":
        if distance_to_turn_m is not None and distance_to_turn_m <= 8.0:
            return "Continue forward. Arrival is imminent."
        return "Continue forward. You are approaching the destination."
    if turn_type == "turn_left":
        return "Continue forward and prepare to turn left."
    if turn_type == "turn_right":
        return "Continue forward and prepare to turn right."
    if turn_type in {"keep_left", "bear_left"}:
        return "Continue forward and shift toward the left-front corridor."
    if turn_type in {"keep_right", "bear_right"}:
        return "Continue forward and shift toward the right-front corridor."
    if turn_type == "u_turn":
        return "Continue forward and prepare for a U-turn."
    return "Continue along the current route."


def _instruction_hint(
    *,
    turn_type: str,
    distance_to_turn_m: float | None,
    road_name: str,
    raw_instruction: str,
    off_route: bool,
) -> str:
    if off_route:
        return "The current pose deviates from the planned route. Move back toward the route first."

    if raw_instruction and _looks_english_text(raw_instruction):
        return raw_instruction

    road_text = f"along {road_name}" if road_name and _looks_english_text(road_name) else "along the current road"
    if distance_to_turn_m is not None and distance_to_turn_m > 0:
        distance_text = f" in about {int(round(distance_to_turn_m))} m"
    else:
        distance_text = ""

    if turn_type == "arrive":
        return f"Continue {road_text}{distance_text} to reach the destination."
    if turn_type == "turn_left":
        return f"Continue {road_text}{distance_text}, then turn left."
    if turn_type == "turn_right":
        return f"Continue {road_text}{distance_text}, then turn right."
    if turn_type in {"keep_left", "bear_left"}:
        return f"Continue {road_text}{distance_text}, then keep toward the left-front corridor."
    if turn_type in {"keep_right", "bear_right"}:
        return f"Continue {road_text}{distance_text}, then keep toward the right-front corridor."
    if turn_type == "u_turn":
        return f"Continue {road_text}{distance_text}, then make a U-turn."
    return f"Continue {road_text}."


def describe_local_segment(
    *,
    gps: dict[str, float],
    pose_match: PoseMatch,
    progress: RouteProgress,
    route_polyline: list[dict[str, Any]] | list[list[float]],
    coarse_waypoints: list[dict[str, Any]],
    heading_deg: float | None,
    lookahead_m: float = 20.0,
) -> tuple[LocalSemantics, LocalGeometry]:
    """Build local semantics and geometry from route progress plus pose match."""
    route_points = normalize_route_polyline(route_polyline)
    if not route_points:
        raise ValueError("route_polyline is empty")

    current_point = RoutePoint(lat=float(gps["lat"]), lng=float(gps["lng"]))
    route_start = route_points[0]
    cumulative_distances = route_cumulative_distances(route_points)
    matched_projection = project_point_to_polyline(pose_match.matched_point, route_points)
    matched_distance_m = float(matched_projection["route_distance_m"])
    local_points = _slice_local_segment(route_points, cumulative_distances, matched_distance_m, lookahead_m)
    local_bearing = _local_bearing_deg(local_points) or matched_projection.get("segment_bearing_deg")
    signed_delta = signed_heading_delta_deg(heading_deg, local_bearing)

    normalized_waypoints = normalize_coarse_waypoints(coarse_waypoints)
    next_waypoint = next(
        (item for item in normalized_waypoints if item["index"] == progress.next_waypoint_index),
        None,
    )
    road_name = str((next_waypoint or {}).get("road") or "").strip()
    raw_instruction = str((next_waypoint or {}).get("instruction") or "").strip()
    turn_type = str(progress.next_turn_type or "go_straight")

    if pose_match.start_phase and pose_match.start_grace_applied:
        semantics = LocalSemantics(
            turn_type=turn_type,
            turn_semantics="approach_route_start",
            corridor_type="front",
            short_action_summary="Approach the route start and merge into the planned path.",
            instruction_hint="The current pose is still approaching the provider route start. Move toward the route start before following the planned path.",
            road_name=road_name,
        )
        geometry = LocalGeometry(
            local_segment_polyline=local_points,
            connector_polyline=_connector_polyline(
                current_point=current_point,
                route_start=route_start,
                merge_point=local_points[0] if local_points else route_start,
            ),
            connector_active=True,
            lookahead_m=float(lookahead_m),
        )
        return semantics, geometry

    semantics = LocalSemantics(
        turn_type=turn_type,
        turn_semantics=_turn_semantics(turn_type, progress.distance_to_next_turn_m, pose_match.off_route),
        corridor_type=_corridor_type(
            signed_heading_delta=signed_delta,
            turn_type=turn_type,
            off_route=pose_match.off_route,
        ),
        short_action_summary=_short_action_summary(turn_type, progress.distance_to_next_turn_m, pose_match.off_route),
        instruction_hint=_instruction_hint(
            turn_type=turn_type,
            distance_to_turn_m=progress.distance_to_next_turn_m,
            road_name=road_name,
            raw_instruction=raw_instruction,
            off_route=pose_match.off_route,
        ),
        road_name=road_name,
    )
    geometry = LocalGeometry(
        local_segment_polyline=local_points,
        connector_polyline=[],
        connector_active=False,
        lookahead_m=float(lookahead_m),
    )
    return semantics, geometry
