"""Pure route geometry and presentation helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import os
from mcp_server.map_providers import encode_polyline_text, sample_route_coordinates

def _normalize_route_coords(route_polyline: List[Dict[str, Any]] | List[List[float]]) -> List[List[float]]:
    normalized: List[List[float]] = []
    for item in route_polyline or []:
        longitude = None
        latitude = None
        if isinstance(item, dict):
            longitude = item.get("longitude", item.get("lng", item.get("lon")))
            latitude = item.get("latitude", item.get("lat"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            longitude = item[0]
            latitude = item[1]
        if longitude is None or latitude is None:
            continue
        point = [float(longitude), float(latitude)]
        if normalized and abs(normalized[-1][0] - point[0]) < 1e-9 and abs(normalized[-1][1] - point[1]) < 1e-9:
            continue
        normalized.append(point)
    return normalized


def _build_amap_staticmap_overlay_url(
    *,
    api_key: str,
    base_route_coords: List[List[float]],
    extra_paths: List[Tuple[str, List[List[float]]]],
    markers: List[str],
    size: str = "800*600",
    scale: int = 2,
    zoom: Optional[int] = None,
    traffic: int = 0,
) -> str:
    if not api_key:
        raise RuntimeError("AMAP_KEY is required to build AMap static map URLs.")

    sampled_base = sample_route_coordinates(
        base_route_coords,
        max_points=int(os.getenv("AMAP_STATIC_ROUTE_MAX_POINTS", os.getenv("STATIC_ROUTE_MAX_POINTS", "240")) or 240),
    )
    if not sampled_base:
        raise ValueError("No base route coordinates available for RouteContext BEV rendering.")

    path_specs = [f"6,0x2E8B57,1,,:{encode_polyline_text(sampled_base)}"]
    for style, coords in extra_paths:
        sampled = sample_route_coordinates(coords, max_points=80)
        if sampled:
            path_specs.append(f"{style}:{encode_polyline_text(sampled)}")

    params: List[Tuple[str, str]] = [
        ("key", api_key),
        ("size", size),
        ("scale", str(max(1, min(int(scale), 2)))),
        ("traffic", str(1 if int(traffic) else 0)),
        ("paths", "|".join(path_specs)),
    ]
    if zoom is not None:
        params.append(("zoom", str(int(zoom))))
    if markers:
        params.append(("markers", "|".join(markers)))

    query = "&".join(f"{key}={value}" for key, value in params)
    return f"https://restapi.amap.com/v3/staticmap?{query}"


def _single_char_marker_label(label: str, fallback: str) -> str:
    """AMap static marker labels are safest as a single ASCII alphanumeric char."""
    text = str(label or "").strip()
    if len(text) == 1 and text.isascii() and text.isalnum():
        return text
    fallback_text = str(fallback or "").strip()
    if len(fallback_text) == 1 and fallback_text.isascii() and fallback_text.isalnum():
        return fallback_text
    return "P"


def _destination_id(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("id") or candidate.get("location") or "").strip()


def _select_itinerary_destination(
    *,
    recommendation: Dict[str, Any],
    used_destination_ids: set[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    candidates = recommendation.get("recommendations") or []
    if not isinstance(candidates, list):
        candidates = []

    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        destination_id = _destination_id(candidate)
        if destination_id and destination_id in used_destination_ids:
            continue
        score = float(candidate.get("score", 0.0) or 0.0)
        ranked.append((score, candidate))

    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1], "ranked_nonduplicate_open_vocab_recommendation"

    chosen = recommendation.get("chosen_destination")
    if isinstance(chosen, dict):
        destination_id = _destination_id(chosen)
        if not destination_id or destination_id not in used_destination_ids:
            return chosen, "chosen_destination_fallback"

    return None, "no_nonduplicate_candidate"


def _project_bev_point(
    point: Dict[str, Any],
    *,
    bbox: Dict[str, Any],
    width: int,
    height: int,
    padding: int = 24,
) -> Tuple[float, float]:
    x_min = float(bbox["x_min_m"])
    x_max = float(bbox["x_max_m"])
    y_min = float(bbox["y_min_m"])
    y_max = float(bbox["y_max_m"])
    x_m = float(point["x_m"])
    y_m = float(point["y_m"])
    usable_width = max(width - 2 * padding, 1)
    usable_height = max(height - 2 * padding, 1)
    x = padding + (x_m - x_min) / max(x_max - x_min, 1e-6) * usable_width
    y = padding + (y_max - y_m) / max(y_max - y_min, 1e-6) * usable_height
    return x, y


def _svg_polyline_from_bev(
    points: List[Dict[str, Any]],
    *,
    bbox: Dict[str, Any],
    width: int,
    height: int,
    stroke: str,
    stroke_width: int,
    dasharray: str = "",
) -> str:
    if not points:
        return ""
    coords = []
    for point in points:
        if point.get("x_m") is None or point.get("y_m") is None:
            continue
        x, y = _project_bev_point(point, bbox=bbox, width=width, height=height)
        coords.append(f"{x:.1f},{y:.1f}")
    if not coords:
        return ""
    dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return (
        f'<polyline fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash_attr} points="{" ".join(coords)}" />'
    )


def _svg_circle_from_bev(
    point: Optional[Dict[str, Any]],
    *,
    bbox: Dict[str, Any],
    width: int,
    height: int,
    fill: str,
    radius: int,
) -> str:
    if not point or point.get("x_m") is None or point.get("y_m") is None:
        return ""
    x, y = _project_bev_point(point, bbox=bbox, width=width, height=height)
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" />'


def _svg_label_from_bev(
    point: Optional[Dict[str, Any]],
    *,
    bbox: Dict[str, Any],
    width: int,
    height: int,
    text: str,
    fill: str,
) -> str:
    if not point or point.get("x_m") is None or point.get("y_m") is None:
        return ""
    x, y = _project_bev_point(point, bbox=bbox, width=width, height=height)
    return f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" fill="{fill}" font-size="12" font-family="monospace">{text}</text>'


def _build_bev_sector_path(
    *,
    bbox: Dict[str, Any],
    width: int,
    height: int,
    center_deg: float,
    span_deg: float,
    radius_m: float,
) -> str:
    cx, cy = _project_bev_point({"x_m": 0.0, "y_m": 0.0}, bbox=bbox, width=width, height=height)
    points: List[str] = [f"{cx:.1f},{cy:.1f}"]
    start_deg = center_deg - span_deg / 2.0
    end_deg = center_deg + span_deg / 2.0
    samples = 12
    for i in range(samples + 1):
        deg = start_deg + (end_deg - start_deg) * i / samples
        rad = math.radians(deg)
        x_m = math.sin(rad) * radius_m
        y_m = math.cos(rad) * radius_m
        x, y = _project_bev_point({"x_m": x_m, "y_m": y_m}, bbox=bbox, width=width, height=height)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _build_local_bev_svg(
    *,
    local_bev_context: Dict[str, Any],
    width: int = 640,
    height: int = 640,
) -> str:
    bbox = local_bev_context.get("bev_bbox", {}) or {}
    route_polyline_local = local_bev_context.get("route_polyline_local", []) or []
    local_segment_bev = local_bev_context.get("local_segment_bev", []) or []
    connector_bev = local_bev_context.get("connector_bev", []) or []
    current_pose = local_bev_context.get("current_pose_bev", {}) or {}
    matched_point = local_bev_context.get("matched_point_bev", {}) or {}
    next_turn = local_bev_context.get("next_turn_point_bev", {}) or {}
    next_waypoint = local_bev_context.get("next_waypoint_bev", {}) or {}
    corridor_sector = local_bev_context.get("corridor_sector_deg", {}) or {}
    turn_sector = local_bev_context.get("turn_sector_deg", {}) or {}

    corridor_path = _build_bev_sector_path(
        bbox=bbox,
        width=width,
        height=height,
        center_deg=float(corridor_sector.get("center_deg", 0.0)),
        span_deg=float(corridor_sector.get("span_deg", 60.0)),
        radius_m=max(float(local_bev_context.get("lookahead_m", 20.0)), 12.0),
    )
    turn_path = _build_bev_sector_path(
        bbox=bbox,
        width=width,
        height=height,
        center_deg=float(turn_sector.get("center_deg", 0.0)),
        span_deg=float(turn_sector.get("span_deg", 35.0)),
        radius_m=max(float(local_bev_context.get("lookahead_m", 20.0)) * 0.85, 10.0),
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#020617" />',
        f'<polygon points="{corridor_path}" fill="#60a5fa" fill-opacity="0.10" stroke="none" />',
        f'<polygon points="{turn_path}" fill="#f59e0b" fill-opacity="0.10" stroke="none" />',
    ]

    parts.extend([
        _svg_polyline_from_bev(route_polyline_local, bbox=bbox, width=width, height=height, stroke="#22c55e", stroke_width=4),
        _svg_polyline_from_bev(local_segment_bev, bbox=bbox, width=width, height=height, stroke="#3b82f6", stroke_width=5),
        _svg_polyline_from_bev(connector_bev, bbox=bbox, width=width, height=height, stroke="#fb923c", stroke_width=4, dasharray="8 6"),
        _svg_circle_from_bev(current_pose, bbox=bbox, width=width, height=height, fill="#3b82f6", radius=6),
        _svg_circle_from_bev(matched_point, bbox=bbox, width=width, height=height, fill="#10b981", radius=5),
        _svg_circle_from_bev(next_waypoint, bbox=bbox, width=width, height=height, fill="#facc15", radius=5),
        _svg_circle_from_bev(next_turn, bbox=bbox, width=width, height=height, fill="#f97316", radius=5),
        _svg_label_from_bev(current_pose, bbox=bbox, width=width, height=height, text="C", fill="#93c5fd"),
        _svg_label_from_bev(matched_point, bbox=bbox, width=width, height=height, text="M", fill="#6ee7b7"),
        _svg_label_from_bev(next_waypoint, bbox=bbox, width=width, height=height, text="W", fill="#fde68a"),
        _svg_label_from_bev(next_turn, bbox=bbox, width=width, height=height, text="T", fill="#fdba74"),
        '<text x="24" y="28" fill="#e2e8f0" font-size="14" font-family="monospace">Local BEV Context</text>',
        f'<text x="24" y="50" fill="#94a3b8" font-size="12" font-family="monospace">turn={local_bev_context.get("turn_semantics", "unknown")} corridor={local_bev_context.get("corridor_type", "unknown")}</text>',
        '</svg>',
    ])
    return "".join(part for part in parts if part)
