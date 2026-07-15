#!/usr/bin/env python3
"""Build a paper-oriented UrbanNav classic-case map figure and review page.

The default case is a successful two-stop replay Plan-and-Execute example. The
figure is self-contained after generation: it uses benchmark JSONL records and
embeds an SVG map with an optional OSM raster crop. OSM tiles are fetched once
and embedded as a data URL, so the artifact does not require browser network.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from io import BytesIO
import html
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the benchmark environment normally has Pillow
    Image = None


DEFAULT_CASE_ID = "urbannav-v3-urbannav-multi_stop_route-00458"
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "benchmark"
    / "urbannav_v2_osm_local_osrm_foot_final_15k"
)
DEFAULT_RUN = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "benchmark"
    / "urbannav_module1_runs"
    / "doubao_external_replay_strict_final15k_corrected_v1"
)

SVG_WIDTH = 1180
SVG_HEIGHT = 700
MAP_LEFT = 48
MAP_TOP = 72
MAP_RIGHT = 1132
MAP_BOTTOM = 600
OSM_TILE_SIZE = 256
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
DEFAULT_ONLINE_OSRM_URL = "https://routing.openstreetmap.de/routed-foot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-osm-background",
        action="store_true",
        help="Do not fetch/embed an OpenStreetMap raster background.",
    )
    parser.add_argument("--osm-zoom", type=int, default=None, help="Optional OSM tile zoom override.")
    parser.add_argument(
        "--osm-tile-cache",
        type=Path,
        default=Path("/tmp/urban_nav_osm_tiles"),
        help="On-disk cache for downloaded OSM tiles.",
    )
    parser.add_argument(
        "--pred-route-source",
        choices=("replay", "online-osrm"),
        default="replay",
        help=(
            "Route geometry shown for the predicted destination sequence. "
            "'online-osrm' is visualization-only and does not alter offline evaluation results."
        ),
    )
    parser.add_argument(
        "--online-osrm-url",
        default=DEFAULT_ONLINE_OSRM_URL,
        help="OSRM endpoint prefix used only with --pred-route-source online-osrm.",
    )
    parser.add_argument(
        "--online-osrm-profile",
        default="foot",
        help="OSRM routing profile used only with --pred-route-source online-osrm.",
    )
    parser.add_argument(
        "--online-route-timeout-s",
        type=float,
        default=30.0,
        help="Timeout for each visualization-only online OSRM route request.",
    )
    return parser.parse_args()


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                value = payload.get(key)
                if value:
                    rows[str(value)] = payload
    return rows


def point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return float(value["lng"]), float(value["lat"])
    except (KeyError, TypeError, ValueError):
        return None


def route_points(values: Any) -> list[tuple[float, float]]:
    if not isinstance(values, list):
        return []
    return [item for value in values if (item := point(value)) is not None]


def _route_distance_m(values: list[dict[str, float]]) -> float:
    """Return the approximate geographic length of a polyline in metres."""

    total = 0.0
    for current, following in zip(values, values[1:]):
        lat1, lng1 = math.radians(float(current["lat"])), math.radians(float(current["lng"]))
        lat2, lng2 = math.radians(float(following["lat"])), math.radians(float(following["lng"]))
        delta_lat = lat2 - lat1
        delta_lng = lng2 - lng1
        value = math.sin(delta_lat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lng / 2.0) ** 2
        total += 2.0 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(value)))
    return total


def _online_osrm_route(
    *,
    origin: dict[str, Any],
    destination: dict[str, Any],
    osrm_url: str,
    profile: str,
    timeout_s: float,
) -> tuple[list[dict[str, float]], float, float]:
    """Fetch one visualization-only pedestrian route from a public OSRM service.

    Args:
        origin: Starting coordinate with ``lat`` and ``lng``.
        destination: Destination coordinate with ``lat`` and ``lng``.
        osrm_url: OSRM endpoint prefix.
        profile: OSRM profile, normally ``foot``.
        timeout_s: HTTP timeout in seconds.

    Returns:
        ``(polyline, distance_m, duration_s)`` for the requested route.

    Raises:
        RuntimeError: If OSRM returns no usable route geometry.
    """

    origin_point = point(origin)
    destination_point = point(destination)
    if origin_point is None or destination_point is None:
        raise RuntimeError("online_route_missing_coordinate")
    coords = f"{origin_point[0]},{origin_point[1]};{destination_point[0]},{destination_point[1]}"
    query = urlencode({"overview": "full", "geometries": "geojson", "steps": "false"})
    url = f"{osrm_url.rstrip('/')}/route/v1/{profile}/{coords}?{query}"
    request = Request(url, headers={"User-Agent": "UrbanNav-paper-figure/1.0"})
    with urlopen(request, timeout=timeout_s) as response:  # nosec B310 - explicit paper-figure provider endpoint
        payload = json.loads(response.read().decode("utf-8"))
    route = (payload.get("routes") or [{}])[0]
    coordinates = ((route.get("geometry") or {}).get("coordinates") or [])
    polyline = [{"lng": float(item[0]), "lat": float(item[1])} for item in coordinates if len(item) >= 2]
    if len(polyline) < 2:
        raise RuntimeError(f"online_osrm_invalid_route:{payload.get('code') or 'missing_geometry'}")
    # Keep the visualization semantically exact even when OSRM snaps endpoints to a nearby road vertex.
    polyline[0] = {"lng": origin_point[0], "lat": origin_point[1]}
    polyline[-1] = {"lng": destination_point[0], "lat": destination_point[1]}
    return polyline, float(route.get("distance") or _route_distance_m(polyline)), float(route.get("duration") or 0.0)


def with_online_predicted_routes(
    *,
    case: dict[str, Any],
    prediction: dict[str, Any],
    osrm_url: str,
    profile: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Return a visualization copy whose predicted legs are re-routed online.

    The model's selected destinations remain unchanged. Only the geometry shown
    in the paper figure is replaced with routes queried from the current
    origin through the selected sequence. This avoids reusing a frozen GT leg
    after an earlier predicted stop differs from the GT stop.

    Args:
        case: Frozen benchmark case.
        prediction: Stored model prediction.
        osrm_url: OSRM endpoint prefix.
        profile: OSRM routing profile.
        timeout_s: Per-request timeout.

    Returns:
        Deep-copied prediction annotated with online visualization routes.

    Raises:
        RuntimeError: If a selected destination or online route is unavailable.
    """

    rendered = deepcopy(prediction)
    source_legs = rendered.get("pred_route_legs") or []
    selected = [leg.get("chosen_destination") for leg in source_legs if point(leg.get("chosen_destination")) is not None]
    if not selected:
        selected = [item for item in rendered.get("pred_destination_sequence") or [] if point(item) is not None]
    # Single-destination predictions use the canonical top-level field rather
    # than a one-item route-leg sequence.
    if not selected and point(rendered.get("chosen_destination")) is not None:
        selected = [rendered["chosen_destination"]]
    current_origin = case.get("origin")
    if point(current_origin) is None or not selected:
        raise RuntimeError("online_route_missing_origin_or_selected_destination")

    rebuilt_legs: list[dict[str, Any]] = []
    full_route: list[dict[str, float]] = []
    total_distance = 0.0
    total_duration = 0.0
    for index, destination in enumerate(selected, start=1):
        route, distance_m, duration_s = _online_osrm_route(
            origin=current_origin,
            destination=destination,
            osrm_url=osrm_url,
            profile=profile,
            timeout_s=timeout_s,
        )
        original_leg = deepcopy(source_legs[index - 1]) if index <= len(source_legs) else {}
        original_leg.update(
            {
                "leg_index": index,
                "origin": {"lat": float(current_origin["lat"]), "lng": float(current_origin["lng"])},
                "chosen_destination": destination,
                "route": route,
                "route_summary": {"distance_m": distance_m, "duration_s": duration_s, "source": "online_osrm"},
            }
        )
        rebuilt_legs.append(original_leg)
        if not full_route:
            full_route.extend(route)
        else:
            full_route.extend(route[1:])
        total_distance += distance_m
        total_duration += duration_s
        current_origin = destination

    rendered["pred_route_legs"] = rebuilt_legs
    rendered["route_polyline"] = full_route
    rendered["route_summary"] = {
        **(rendered.get("route_summary") or {}),
        "distance_m": total_distance,
        "duration_s": total_duration,
        "source": "online_osrm_visualization",
    }
    rendered.setdefault("trace", {})["visualization_route_source"] = {
        "source": "online_osrm",
        "osrm_url": osrm_url,
        "profile": profile,
    }
    return rendered


def multi_stop_gt_route(case: dict[str, Any]) -> list[dict[str, float]]:
    """Return the task-aware concatenated GT route for multi-stop cases."""

    legs = case.get("gt_route_legs") or []
    if not isinstance(legs, list):
        return list(case.get("gt_route") or [])
    route: list[dict[str, float]] = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        destination = leg.get("gt_destination") or {}
        leg_route = destination.get("reference_route") or leg.get("route") or []
        for value in leg_route:
            if not isinstance(value, dict):
                continue
            if route and value.get("lat") == route[-1].get("lat") and value.get("lng") == route[-1].get("lng"):
                continue
            route.append(value)
    return route or list(case.get("gt_route") or [])


def destination_name(value: dict[str, Any] | None) -> str:
    if not isinstance(value, dict):
        return "unknown"
    return str(value.get("name") or value.get("provider_poi_id") or "unnamed POI")


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def multi_stop_status(evaluation: dict[str, Any]) -> str:
    """Return a presentation status without changing evaluator semantics."""

    multi = (evaluation.get("extended_metrics") or {}).get("multi_stop") or {}
    return "success" if multi.get("sequence_success") and multi.get("leg_route_success") else "failure"


def visualization_route_source(prediction: dict[str, Any]) -> str:
    """Return the route-geometry source used by the generated figure."""

    return str(((prediction.get("trace") or {}).get("visualization_route_source") or {}).get("source") or "replay")


def visualization_route_distance_m(prediction: dict[str, Any]) -> float:
    """Return total displayed route length from the visualization prediction."""

    total = 0.0
    for leg in prediction.get("pred_route_legs") or []:
        value = ((leg.get("route_summary") or {}).get("distance_m"))
        if value is not None:
            total += float(value)
        else:
            total += _route_distance_m([item for item in leg.get("route") or [] if point(item) is not None])
    return total


def load_case(case_id: str, dataset_dir: Path, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    gt_rows = load_jsonl_by_id(dataset_dir / "task_gt.jsonl", "id")
    pred_rows = load_jsonl_by_id(run_dir / "raw_predictions.jsonl", "id")
    eval_rows = load_jsonl_by_id(run_dir / "eval" / "eval_rows.jsonl", "case_id")
    missing = [name for name, rows in (("GT", gt_rows), ("prediction", pred_rows), ("evaluation", eval_rows)) if case_id not in rows]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} row for case_id={case_id}")
    return gt_rows[case_id], pred_rows[case_id], eval_rows[case_id]


def project_points(case: dict[str, Any], prediction: dict[str, Any]) -> tuple[dict[tuple[float, float], tuple[float, float]], float, float, float, float]:
    values: list[tuple[float, float]] = []
    for value in [case.get("origin"), case.get("gt_destination")]:
        if item := point(value):
            values.append(item)
    values.extend(route_points(multi_stop_gt_route(case)))
    values.extend(route_points(prediction.get("route_polyline")))
    for sequence_key in ("gt_destination_sequence",):
        for value in case.get(sequence_key) or []:
            if item := point(value):
                values.append(item)
    for leg in prediction.get("pred_route_legs") or []:
        for value in leg.get("candidate_destinations") or []:
            if item := point(value):
                values.append(item)
        if item := point(leg.get("chosen_destination")):
            values.append(item)
    if not values:
        raise SystemExit("No coordinates available for the selected case")
    lngs = [value[0] for value in values]
    lats = [value[1] for value in values]
    min_lng, max_lng = min(lngs), max(lngs)
    min_lat, max_lat = min(lats), max(lats)
    lng_span = max(max_lng - min_lng, 0.0002)
    lat_span = max(max_lat - min_lat, 0.0002)
    margin_lng = lng_span * 0.12
    margin_lat = lat_span * 0.12
    min_lng -= margin_lng
    max_lng += margin_lng
    min_lat -= margin_lat
    max_lat += margin_lat
    width = float(MAP_RIGHT - MAP_LEFT)
    height = float(MAP_BOTTOM - MAP_TOP)

    def mercator_y(lat: float) -> float:
        clipped = max(-85.05112878, min(85.05112878, lat))
        return math.log(math.tan(math.pi / 4.0 + math.radians(clipped) / 2.0))

    max_mercator_y = mercator_y(max_lat)
    min_mercator_y = mercator_y(min_lat)

    def transform(item: tuple[float, float]) -> tuple[float, float]:
        x = MAP_LEFT + (item[0] - min_lng) / (max_lng - min_lng) * width
        mercator = mercator_y(item[1])
        y = MAP_TOP + (max_mercator_y - mercator) / (max_mercator_y - min_mercator_y) * height
        return x, y

    return {item: transform(item) for item in set(values)}, min_lng, max_lng, min_lat, max_lat


def svg_polyline(values: Any, projected: dict[tuple[float, float], tuple[float, float]]) -> str:
    coords = []
    for item in route_points(values):
        if item in projected:
            coords.append(f"{projected[item][0]:.1f},{projected[item][1]:.1f}")
    return " ".join(coords)


def _osm_tile_xy(lng: float, lat: float, zoom: int) -> tuple[float, float]:
    """Return continuous Web-Mercator pixel coordinates for an OSM tile grid."""

    scale = float(OSM_TILE_SIZE * (2**zoom))
    x = (lng + 180.0) / 360.0 * scale
    clipped = max(-85.05112878, min(85.05112878, lat))
    y = (
        1.0
        - math.asinh(math.tan(math.radians(clipped))) / math.pi
    ) / 2.0 * scale
    return x, y


def _choose_osm_zoom(min_lng: float, max_lng: float, min_lat: float, max_lat: float) -> int:
    """Choose a readable zoom while keeping the tile mosaic compact."""

    span_lng = max(max_lng - min_lng, 1e-6)
    target_width = float(MAP_RIGHT - MAP_LEFT)
    estimate = math.log2(target_width * 360.0 / (span_lng * OSM_TILE_SIZE))
    return max(12, min(19, int(round(estimate))))


def _fetch_osm_tile(*, x: int, y: int, zoom: int, cache_dir: Path) -> bytes:
    """Fetch one OSM tile, using a local cache to make regeneration deterministic."""

    n = 2**zoom
    wrapped_x = x % n
    if y < 0 or y >= n:
        raise ValueError(f"OSM tile y out of range: {zoom}/{x}/{y}")
    path = cache_dir / str(zoom) / str(wrapped_x) / f"{y}.png"
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    url = OSM_TILE_URL.format(z=zoom, x=wrapped_x, y=y)
    request = Request(url, headers={"User-Agent": "UrbanNav-paper-figure/1.0"})
    with urlopen(request, timeout=12) as response:  # nosec B310 - fixed HTTPS OSM host
        data = response.read()
    path.write_bytes(data)
    return data


def build_osm_background(
    *,
    min_lng: float,
    max_lng: float,
    min_lat: float,
    max_lat: float,
    cache_dir: Path,
    zoom: int | None,
) -> tuple[str | None, str | None]:
    """Build an embedded OSM raster crop aligned with the map projection.

    Returns:
        ``(data_url, error)``. The caller can render a graceful fallback when
        network access or Pillow is unavailable.
    """

    if Image is None:
        return None, "Pillow is unavailable; cannot compose OSM raster tiles."
    selected_zoom = int(zoom) if zoom is not None else _choose_osm_zoom(min_lng, max_lng, min_lat, max_lat)
    selected_zoom = max(12, min(19, selected_zoom))
    while True:
        x0, y0 = _osm_tile_xy(min_lng, max_lat, selected_zoom)
        x1, y1 = _osm_tile_xy(max_lng, min_lat, selected_zoom)
        start_x = math.floor(x0 / OSM_TILE_SIZE)
        end_x = math.floor(x1 / OSM_TILE_SIZE)
        start_y = math.floor(y0 / OSM_TILE_SIZE)
        end_y = math.floor(y1 / OSM_TILE_SIZE)
        if end_x - start_x <= 6 and end_y - start_y <= 6:
            break
        if selected_zoom == 12:
            return None, f"OSM crop needs too many tiles even at zoom {selected_zoom}."
        selected_zoom -= 1
    mosaic = Image.new("RGB", ((end_x - start_x + 1) * OSM_TILE_SIZE, (end_y - start_y + 1) * OSM_TILE_SIZE), "#e5e7eb")
    try:
        for tile_x in range(start_x, end_x + 1):
            for tile_y in range(start_y, end_y + 1):
                raw = _fetch_osm_tile(x=tile_x, y=tile_y, zoom=selected_zoom, cache_dir=cache_dir)
                tile = Image.open(BytesIO(raw)).convert("RGB")
                mosaic.paste(tile.resize((OSM_TILE_SIZE, OSM_TILE_SIZE)), ((tile_x - start_x) * OSM_TILE_SIZE, (tile_y - start_y) * OSM_TILE_SIZE))
    except Exception as exc:
        return None, f"OSM tile fetch failed: {exc}"
    crop_left = max(0, int(round(x0 - start_x * OSM_TILE_SIZE)))
    crop_top = max(0, int(round(y0 - start_y * OSM_TILE_SIZE)))
    crop_right = min(mosaic.width, int(round(x1 - start_x * OSM_TILE_SIZE)) + 1)
    crop_bottom = min(mosaic.height, int(round(y1 - start_y * OSM_TILE_SIZE)) + 1)
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return None, "OSM crop bounds are empty."
    crop = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom)).resize(
        (MAP_RIGHT - MAP_LEFT, MAP_BOTTOM - MAP_TOP)
    )
    buffer = BytesIO()
    crop.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", None


def _label_box_width(lines: list[str]) -> float:
    return max(34.0, max((len(line) for line in lines), default=1) * 6.2 + 12.0)


def _marker_groups(case: dict[str, Any], prediction: dict[str, Any], projected: dict[tuple[float, float], tuple[float, float]]) -> list[dict[str, Any]]:
    """Merge coincident semantic markers before laying out labels."""

    groups: dict[tuple[float, float], dict[str, Any]] = {}

    def group_for(item: tuple[float, float]) -> dict[str, Any]:
        if item not in groups:
            groups[item] = {"item": item, "candidates": [], "gt": [], "selected": []}
        return groups[item]

    for leg_index, leg in enumerate(prediction.get("pred_route_legs") or [], start=1):
        for rank, candidate in enumerate(leg.get("candidate_destinations") or [], start=1):
            item = point(candidate)
            if item in projected:
                group_for(item)["candidates"].append(f"L{leg_index}C{rank}")
    for index, destination in enumerate(case.get("gt_destination_sequence") or [], start=1):
        item = point(destination)
        if item in projected:
            group_for(item)["gt"].append(f"GT{index}")
    for index, leg in enumerate(prediction.get("pred_route_legs") or [], start=1):
        item = point(leg.get("chosen_destination"))
        if item in projected:
            group_for(item)["selected"].append(f"SEL{index}")
    return list(groups.values())


def _marker_layers(case: dict[str, Any], prediction: dict[str, Any], projected: dict[tuple[float, float], tuple[float, float]]) -> list[str]:
    """Render compact, collision-aware marker labels."""

    groups = _marker_groups(case, prediction, projected)
    occupied: list[tuple[float, float, float, float]] = []
    layers: list[str] = []
    offsets = [(12, -12), (12, 12), (-12, -12), (-12, 12), (16, 0), (-16, 0)]
    origin = point(case.get("origin"))
    if origin in projected:
        origin_x, origin_y = projected[origin]
        occupied.append((origin_x + 12.0, origin_y - 24.0, origin_x + 60.0, origin_y - 6.0))

    def overlaps(box: tuple[float, float, float, float]) -> bool:
        return any(not (box[2] < other[0] or box[0] > other[2] or box[3] < other[1] or box[1] > other[3]) for other in occupied)

    for group in groups:
        item = group["item"]
        x, y = projected[item]
        lines = group["gt"] + group["selected"] + group["candidates"]
        width = _label_box_width(lines)
        height = 8.0 + 14.0 * max(1, len(lines))
        chosen = None
        for offset_x, offset_y in offsets:
            anchor_x = x + offset_x
            anchor_y = y + offset_y
            left = anchor_x if offset_x >= 0 else anchor_x - width
            top = anchor_y - height if offset_y <= 0 else anchor_y
            box = (left, top, left + width, top + height)
            if MAP_LEFT + 2 <= box[0] <= box[2] <= MAP_RIGHT - 2 and MAP_TOP + 2 <= box[1] <= box[3] <= MAP_BOTTOM - 2 and not overlaps(box):
                chosen = (box, anchor_x, anchor_y)
                break
        if chosen is None:
            fallback_positions = []
            for row in range(0, 12):
                top = MAP_TOP + 4 + row * (height + 4)
                fallback_positions.extend(
                    [
                        (MAP_LEFT + 4, top),
                        (MAP_RIGHT - width - 4, top),
                    ]
                )
            for left, top in fallback_positions:
                box = (left, top, left + width, top + height)
                if box[3] <= MAP_BOTTOM - 2 and not overlaps(box):
                    chosen = (box, left, top + height / 2)
                    break
        if chosen is None:
            left = min(max(MAP_LEFT + 2, x + 12), MAP_RIGHT - width - 2)
            top = min(max(MAP_TOP + 2, y - height - 12), MAP_BOTTOM - height - 2)
            chosen = ((left, top, left + width, top + height), left, top + height / 2)
        box, anchor_x, anchor_y = chosen
        occupied.append(box)
        fill = "#f97316" if group["selected"] else "#111827" if group["gt"] else "#94a3b8"
        stroke = "#ffffff"
        layers.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{anchor_x:.1f}" y2="{anchor_y:.1f}" stroke="#64748b" stroke-width="1" opacity="0.7"/>')
        layers.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{10 if group["gt"] or group["selected"] else 5:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        layers.append(f'<rect x="{box[0]:.1f}" y="{box[1]:.1f}" width="{width:.1f}" height="{height:.1f}" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1" stroke-width="0.8"/>')
        for line_index, label in enumerate(lines):
            layers.append(f'<text x="{box[0] + 6:.1f}" y="{box[1] + 14 + line_index * 14:.1f}" class="marker-label">{esc(label)}</text>')
    return layers


def build_svg(
    case: dict[str, Any],
    prediction: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    osm_tile_cache: Path = Path("/tmp/urban_nav_osm_tiles"),
    osm_zoom: int | None = None,
    include_osm: bool = True,
) -> str:
    projected, min_lng, max_lng, min_lat, max_lat = project_points(case, prediction)
    gt_route = svg_polyline(multi_stop_gt_route(case), projected)
    pred_route = svg_polyline(prediction.get("route_polyline"), projected)
    layers: list[str] = []
    osm_data_url, osm_error = (None, "disabled by --no-osm-background")
    if include_osm:
        osm_data_url, osm_error = build_osm_background(
            min_lng=min_lng,
            max_lng=max_lng,
            min_lat=min_lat,
            max_lat=max_lat,
            cache_dir=osm_tile_cache,
            zoom=osm_zoom,
        )
    layers.append('<rect x="0" y="0" width="1180" height="700" fill="#f7fafc"/>')
    if osm_data_url:
        layers.append(f'<image x="{MAP_LEFT}" y="{MAP_TOP}" width="{MAP_RIGHT - MAP_LEFT}" height="{MAP_BOTTOM - MAP_TOP}" href="{osm_data_url}" preserveAspectRatio="none" opacity="0.96"/>')
        layers.append(f'<rect x="{MAP_LEFT}" y="{MAP_TOP}" width="{MAP_RIGHT - MAP_LEFT}" height="{MAP_BOTTOM - MAP_TOP}" fill="#ffffff" opacity="0.025"/>')
    else:
        layers.append(f'<rect x="{MAP_LEFT}" y="{MAP_TOP}" width="{MAP_RIGHT - MAP_LEFT}" height="{MAP_BOTTOM - MAP_TOP}" fill="#eef3f5"/>')
        layers.append(f'<text x="{MAP_LEFT + 12}" y="{MAP_TOP + 22}" class="subtitle">OSM background unavailable: {esc(osm_error or "unknown error")}</text>')
    layers.append(f'<rect x="{MAP_LEFT}" y="{MAP_TOP}" width="{MAP_RIGHT - MAP_LEFT}" height="{MAP_BOTTOM - MAP_TOP}" fill="none" stroke="#cbd5e1" stroke-width="1"/>')
    if gt_route:
        layers.append(f'<polyline points="{gt_route}" fill="none" stroke="#ffffff" stroke-width="11" opacity="0.78"/>')
        layers.append(f'<polyline points="{gt_route}" fill="none" stroke="#475569" stroke-width="7" stroke-dasharray="12 9" opacity="0.62"/>')
    predicted_legs = prediction.get("pred_route_legs") or []
    if not predicted_legs and pred_route:
        layers.append(f'<polyline points="{pred_route}" fill="none" stroke="#ffffff" stroke-width="9" opacity="0.82"/>')
        layers.append(f'<polyline points="{pred_route}" fill="none" stroke="#dc2626" stroke-width="5" opacity="0.85"/>')
    for index, leg in enumerate(predicted_legs):
        route = svg_polyline(leg.get("route"), projected)
        if route:
            layers.append(f'<polyline points="{route}" fill="none" stroke="#ffffff" stroke-width="8" opacity="0.82"/>')
            layers.append(f'<polyline points="{route}" fill="none" stroke="#dc2626" stroke-width="5" opacity="0.88"/>')
    origin = point(case.get("origin"))
    if origin in projected:
        x, y = projected[origin]
        layers.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#16a34a" stroke="#ffffff" stroke-width="2"/>')
        layers.append(f'<rect x="{x + 12:.1f}" y="{y - 24:.1f}" width="48" height="18" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/><text x="{x + 18:.1f}" y="{y - 11:.1f}" class="marker-label">origin</text>')
    layers.extend(_marker_layers(case, prediction, projected))
    title = f"{case.get('category')} | {case.get('city')} | {case.get('id')}"
    multi = (evaluation.get("extended_metrics") or {}).get("multi_stop") or {}
    is_multi_stop = case.get("category") == "multi_stop_route"
    route_source = visualization_route_source(prediction)
    source_label = "online OSM replan" if route_source == "online_osrm" else "frozen replay"
    if is_multi_stop:
        subtitle = (
            f"Reference multi-leg {float(multi.get('reference_total_distance_m') or 0):.1f} m | "
            f"Pred multi-leg {visualization_route_distance_m(prediction):.1f} m ({source_label})"
        )
    else:
        subtitle = (
            f"Reference route {float(evaluation.get('reference_route_distance_m') or 0):.1f} m | "
            f"Predicted route {visualization_route_distance_m(prediction):.1f} m ({source_label})"
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 700" role="img" aria-label="{esc(title)}">
<style>
  text {{ font-family: Arial, sans-serif; fill: #172026; }}
  .title {{ font-size: 19px; font-weight: 700; }}
  .subtitle {{ font-size: 13px; fill: #52606d; }}
  .marker-label {{ font-size: 11px; font-weight: 600; }}
  .legend {{ font-size: 12px; }}
</style>
{''.join(layers)}
<text x="48" y="28" class="title">{esc(title)}</text>
<text x="48" y="46" class="subtitle">{esc(subtitle)} | OSM background | dashed gray = GT, red = predicted</text>
<g transform="translate(48 622)">
  <line x1="0" y1="0" x2="34" y2="0" stroke="#475569" stroke-width="7" stroke-dasharray="10 7"/><text x="42" y="4" class="legend">GT route</text>
  <line x1="120" y1="0" x2="154" y2="0" stroke="#dc2626" stroke-width="5"/><text x="162" y="4" class="legend">predicted route</text>
  <circle cx="300" cy="0" r="7" fill="#94a3b8" stroke="#fff" stroke-width="2"/><text x="312" y="4" class="legend">candidate</text>
  <circle cx="390" cy="0" r="8" fill="#f97316" stroke="#fff" stroke-width="2"/><text x="402" y="4" class="legend">selected</text>
</g>
<text x="48" y="680" class="subtitle">Bounds: lng [{min_lng:.5f}, {max_lng:.5f}], lat [{min_lat:.5f}, {max_lat:.5f}] | © OpenStreetMap contributors</text>
</svg>'''


def build_analysis(case: dict[str, Any], prediction: dict[str, Any], evaluation: dict[str, Any]) -> str:
    multi = (evaluation.get("extended_metrics") or {}).get("multi_stop") or {}
    goals = (prediction.get("trace") or {}).get("goal_decomposition") or {}
    status = multi_stop_status(evaluation)
    predicted_legs = prediction.get("pred_route_legs") or []
    route_source = visualization_route_source(prediction)
    diagnosis: list[str] = []
    if status == "failure":
        if multi.get("sequence_hit@5") and not multi.get("sequence_success"):
            diagnosis.append("All ordered goals are covered by the per-stop top-5 candidates, but the selected Top-1 sequence is incorrect; the dominant error is candidate selection/reranking rather than replay retrieval.")
        failed_legs: list[int] = []
        for leg in multi.get("legs") or []:
            if leg.get("destination_success") is False:
                index = int(leg.get("leg_index") or 0) - 1
                chosen = predicted_legs[index].get("chosen_destination") if 0 <= index < len(predicted_legs) else {}
                name = (chosen or {}).get("name") or "the selected candidate"
                failed_legs.append(int(leg.get("leg_index") or 0))
                diagnosis.append(f"Leg {leg.get('leg_index')}: selected {name!r}, while the acceptable destination was not selected; predicted leg distance was {leg.get('pred_route_distance_m')}m versus the reference {leg.get('reference_route_distance_m')}m.")
        if failed_legs:
            first_failed = min(failed_legs)
            if first_failed > 1:
                diagnosis.append(f"The error first appears at leg {first_failed}; earlier legs remain valid, but the ordered sequence and full-route SPL fail once that stop is not matched.")
            else:
                diagnosis.append("The first-stop error changes the origin for subsequent legs, so the ordered sequence and full-route SPL fail even if later legs reach their selected candidates.")
    else:
        diagnosis.append("All ordered goals are selected correctly and every predicted leg reaches its selected destination; the replay evaluator therefore marks the sequence and full multi-stop route successful.")
    lines = [
        f"# UrbanNav Classic Case: Multi-stop {status.title()}",
        "",
        f"- Case: `{case.get('id')}`",
        f"- Category: `{case.get('category')}` / `{case.get('subcategory')}`",
        f"- City: `{case.get('city')}`",
        f"- Instruction: {case.get('instruction')}",
        "",
        "## Pipeline narrative",
        "",
        "1. **Instruction**: the user asks for an ordered multi-stop walking route.",
        f"2. **Goal decomposition**: the runner creates {goals.get('goal_count') or len(goals.get('goals') or [])} ordered goals with separate target types and constraints.",
        "3. **Replay POI search**: each goal reads its frozen POI cache entry; no live map search is performed.",
        "4. **Candidate reranking**: candidates are semantically ordered for each leg.",
        "5. **Route lookup**: the benchmark runner uses frozen replay routes for scoring; this paper figure may replan the displayed selected sequence through online OSM routing.",
        "6. **Selected destination**: the model/runner selects one destination for each stop and uses the selected point as the next leg origin.",
        "7. **Evaluator**: the evaluator checks ordered stop identity, each leg endpoint, and the full multi-stop SPL.",
        "",
        "## Evidence",
        "",
        f"- Goal count: `{goals.get('goal_count') or len(goals.get('goals') or [])}`.",
        f"- Predicted stops: `{multi.get('pred_stop_count')}`; stop recall: `{multi.get('stop_recall')}`.",
        f"- Sequence success: `{multi.get('sequence_success')}`; sequence Hit@5: `{multi.get('sequence_hit@5')}`; stop Hit@5: `{multi.get('stop_hit@5')}`.",
        f"- Leg route success: `{multi.get('leg_route_success')}`; multi-stop SPL: `{multi.get('multi_stop_spl')}`.",
        f"- Benchmark reference/replay predicted distance: `{multi.get('reference_total_distance_m')}m` / `{multi.get('pred_total_distance_m')}m`.",
        f"- Displayed predicted-route source: `{route_source}`; displayed distance: `{visualization_route_distance_m(prediction):.1f}m`.",
        "",
        "## Failure diagnosis" if status == "failure" else "## Success interpretation",
        "",
        *[f"- {item}" for item in diagnosis],
        "",
        "## Figure interpretation",
        "",
        "The dashed gray line is the frozen reference route, the red line is the predicted route, gray markers are replay candidates, black markers are acceptable GT stops, and orange markers are selected predicted stops. When the route source is `online_osrm`, the red geometry is replanned consecutively from the origin through the model-selected stops; it is a visualization artifact and does not alter the offline replay score.",
        "",
        "This is a controlled-replay Plan-and-Execute illustration, not an explicit ReAct tool-call trace. The POI and route stages are deterministic replay operations; the model-facing decisions are intent/goal decomposition and candidate ordering.",
    ]
    return "\n".join(lines) + "\n"


def build_html(case: dict[str, Any], prediction: dict[str, Any], evaluation: dict[str, Any], svg: str) -> str:
    trace = prediction.get("trace") or {}
    multi = (evaluation.get("extended_metrics") or {}).get("multi_stop") or {}
    is_multi_stop = case.get("category") == "multi_stop_route"
    status = multi_stop_status(evaluation) if is_multi_stop else ("success" if evaluation.get("route_success") else "failure")
    status_label = "SUCCESS" if status == "success" else "FAILURE"
    if is_multi_stop:
        metric_cards = f'''<div class="metric"><b>Stop recall</b><span>{esc(multi.get('stop_recall'))}</span></div>
<div class="metric"><b>Sequence success</b><span>{esc(multi.get('sequence_success'))}</span></div>
<div class="metric"><b>Leg route success</b><span>{esc(multi.get('leg_route_success'))}</span></div>
<div class="metric"><b>Multi-stop SPL</b><span>{esc(multi.get('multi_stop_spl'))}</span></div>'''
        decomposition = trace.get("goal_decomposition") or {}
        evaluation_detail = multi.get("legs") or []
        decomposition_label = "Goal decomposition"
        evaluation_label = "Per-leg evaluation"
    else:
        metric_cards = f'''<div class="metric"><b>Destination success</b><span>{esc(evaluation.get('destination_success'))}</span></div>
<div class="metric"><b>Hit@5</b><span>{esc(evaluation.get('hit@5'))}</span></div>
<div class="metric"><b>Route success</b><span>{esc(evaluation.get('route_success'))}</span></div>
<div class="metric"><b>SPL</b><span>{esc(evaluation.get('spl'))}</span></div>'''
        intent = dict(trace.get("intent") or {})
        decomposition = {
            "goal_index": 0,
            "goal_type": intent.get("destination_type"),
            "destination_kind": intent.get("destination_kind"),
            "search_keywords": intent.get("search_keywords"),
            "mode": intent.get("mode"),
            "radius_m": intent.get("radius_m"),
            "optimize_for": intent.get("optimize_for"),
            "hard_constraints": intent.get("hard_constraints"),
            "source": "trace.intent",
        }
        evaluation_detail = {
            key: evaluation.get(key)
            for key in (
                "destination_success", "hit@5", "route_success", "spl",
                "matched_destination_rank", "matched_destination_id",
                "endpoint_error_m", "pred_route_distance_m",
                "reference_route_distance_m",
            )
        }
        decomposition_label = "Intent decomposition"
        evaluation_label = "Single-route evaluation"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UrbanNav classic case</title>
<style>
body {{ margin:0; background:#f4f6f8; color:#172026; font:14px/1.5 Arial,sans-serif; }}
main {{ max-width:1400px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 8px; font-size:25px; }} h2 {{ margin:24px 0 10px; font-size:18px; }}
.status {{ display:inline-block; margin-left:8px; padding:3px 8px; border-radius:4px; font-size:12px; vertical-align:middle; }}
.status.success {{ background:#dcfce7; color:#166534; }} .status.failure {{ background:#fee2e2; color:#991b1b; }}
.muted {{ color:#5c6972; }} .figure {{ background:white; border:1px solid #dce3e8; border-radius:8px; padding:12px; }}
.figure svg {{ width:100%; height:auto; display:block; }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
.metric {{ background:white; border:1px solid #dce3e8; border-radius:6px; padding:11px; }} .metric b {{ display:block;color:#60717d;font-size:12px; }} .metric span {{ font-size:20px;font-weight:700; }}
.two {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-top:14px; }}
pre {{ margin:0; max-height:360px; overflow:auto; background:#0f1720;color:#e7edf3;border-radius:6px;padding:12px;font-size:12px; }}
.flow {{ background:#101827;color:white;padding:14px;border-radius:6px;word-spacing:4px; }}
@media(max-width:900px) {{ .grid,.two {{ grid-template-columns:1fr; }} main {{ padding:14px; }} }}
</style></head><body><main>
<h1>UrbanNav classic case: controlled-replay Plan-and-Execute <span class="status {status}">{status_label}</span></h1>
<p class="muted"><b>{esc(case.get('id'))}</b> | {esc(case.get('city'))} | {esc(case.get('subcategory'))}</p>
<p>{esc(case.get('instruction'))}</p>
<div class="flow">Instruction -> Intent / Goal Decomposition -> Replay POI Search -> Candidate Reranking -> Replay Route Lookup -> Selected Destination -> Evaluator</div>
<h2>Map visualization</h2><section class="figure">{svg}</section>
<h2>Evaluation</h2><div class="grid">
{metric_cards}
</div>
<h2>Structured evidence</h2><div class="two">
<div><h3>{decomposition_label}</h3><pre>{esc(compact_json(decomposition))}</pre></div>
<div><h3>{evaluation_label}</h3><pre>{esc(compact_json(evaluation_detail))}</pre></div>
</div></main></body></html>'''


def main() -> None:
    args = parse_args()
    case, prediction, evaluation = load_case(args.case_id, args.dataset_dir, args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visualization_prediction = prediction
    if args.pred_route_source == "online-osrm":
        visualization_prediction = with_online_predicted_routes(
            case=case,
            prediction=prediction,
            osrm_url=args.online_osrm_url,
            profile=args.online_osrm_profile,
            timeout_s=args.online_route_timeout_s,
        )
    svg = build_svg(
        case,
        visualization_prediction,
        evaluation,
        osm_tile_cache=args.osm_tile_cache,
        osm_zoom=args.osm_zoom,
        include_osm=not args.no_osm_background,
    )
    (args.output_dir / "classic_case_map.svg").write_text(svg, encoding="utf-8")
    (args.output_dir / "classic_case_review.html").write_text(build_html(case, visualization_prediction, evaluation, svg), encoding="utf-8")
    (args.output_dir / "analysis.md").write_text(build_analysis(case, visualization_prediction, evaluation), encoding="utf-8")
    (args.output_dir / "case.json").write_text(
        json.dumps(
            {
                "case": case,
                "prediction": prediction,
                "visualization_prediction": visualization_prediction,
                "evaluation": evaluation,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "case_id": args.case_id,
        "svg": str(args.output_dir / "classic_case_map.svg"),
        "html": str(args.output_dir / "classic_case_review.html"),
        "analysis": str(args.output_dir / "analysis.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
