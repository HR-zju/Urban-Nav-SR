"""MCP server exposing OpenStreetMap-powered location tools.

This module wires FastMCP tool/resource handlers to public geospatial services:
- Nominatim for geocoding and reverse geocoding
- OSRM for routing
- Overpass for POI and category search
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import json
import math
import os
from urllib.parse import quote
from uuid import uuid4

import aiohttp
from mcp.server.fastmcp import Context, FastMCP

from mcp_server.llm import (
    GoalDecomposer,
    ItineraryCandidateJudge,
    analyze_image_with_llm,
    analyze_local_image_with_llm,
)
from mcp_server.macos_location import get_current_location
from mcp_server.map_providers import (
    MapProviderClient,
    build_amap_staticmap_url,
    create_map_client,
    encode_polyline_text,
    format_location,
    parse_location_text,
    sample_route_coordinates,
)
from mcp_server.recommendation import DestinationRecommendationEngine
from mcp_server.route_context import (
    build_local_bev_context,
    build_route_context,
    build_route_state,
    local_bev_context_to_dict,
    route_context_to_dict,
    route_state_to_dict,
)
from mcp_server.services.route_session_service import (
    ROUTE_SESSIONS,
    _cache_route_plan_on_session,
    _compute_combined_route_plan,
    _compute_incremental_route_plan_with_inserted_via,
    _get_persisted_route_session,
    _load_route_sessions,
    _locations_equivalent,
    _normalize_via_points,
    _route_plan_for_session,
    _route_session_store_info,
    _save_persisted_route_session,
)
from mcp_server.services.fpv_segmentation_service import segment_fpv_frame as segment_fpv_frame_service
from mcp_server.services.visual_prompt_service import run_zero_shot_visual_prompt_pipeline

# Create application context
@dataclass
class AppContext:
    osm_client: MapProviderClient
    provider: str


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

# Define lifespan manager
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage OSM client lifecycle"""
    # FastMCP enters this context once on startup and closes it on shutdown.
    osm_client, provider = create_map_client()
    try:
        await osm_client.connect()
        yield AppContext(osm_client=osm_client, provider=provider)
    finally:
        await osm_client.disconnect()

# Create the MCP server
mcp = FastMCP(
    "Location-Based App MCP Server",
    dependencies=["aiohttp", "geojson", "shapely", "haversine"],
    lifespan=app_lifespan
)


@mcp.tool()
async def get_current_location_macos(
    ctx: Context,
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """
    Get the current device location from macOS CoreLocation.

    This tool is intended for local deployment on macOS so the MCP server can
    source the robot or laptop position directly from the host operating system.

    Args:
        timeout_s: Maximum seconds to wait for a location fix

    Returns:
        Current coordinates and accuracy metadata
    """
    await ctx.info(f"Requesting current macOS location with timeout={timeout_s:.1f}s")
    location = await asyncio.to_thread(get_current_location, timeout_s)
    return {
        "provider": "macos_corelocation",
        "location": f"{location['longitude']},{location['latitude']}",
        "coordinates": {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
        },
        "accuracy_m": location.get("accuracy_m"),
        "altitude_m": location.get("altitude_m"),
        "timestamp": location.get("timestamp"),
        "source": location.get("source"),
        "platform": location.get("platform"),
    }


@mcp.tool()
async def geocode_address(address: str, ctx: Context) -> List[Dict]:
    """
    Convert an address or place name to geographic coordinates with detailed location information.
    
    This tool takes a text description of a location (such as an address, landmark name, or
    place of interest) and returns its precise geographic coordinates along with rich metadata.
    The results can be used for mapping, navigation, location-based analysis, and as input to
    other geospatial tools.
    
    Args:
        address: The address, place name, landmark, or description to geocode (e.g., "Empire State Building", 
                "123 Main St, Springfield", "Golden Gate Park, San Francisco")
        
    Returns:
        List of matching locations with:
        - Geographic coordinates (latitude/longitude)
        - Formatted address
        - Administrative boundaries (city, state, country)
        - OSM type and ID
        - Bounding box (if applicable)
        - Importance ranking
    """
    osm_client = ctx.request_context.lifespan_context.osm_client
    results = await osm_client.geocode(address)
    
    # Enhance results with additional context
    for result in results:
        if "lat" in result and "lon" in result:
            result["coordinates"] = {
                "latitude": float(result["lat"]),
                "longitude": float(result["lon"])
            }
    
    return results

@mcp.tool()
async def reverse_geocode(latitude: float, longitude: float, ctx: Context) -> Dict:
    """
    Convert geographic coordinates to a detailed address and location description.
    
    This tool takes a specific point on Earth (latitude and longitude) and returns 
    comprehensive information about that location, including its address, nearby landmarks,
    administrative boundaries, and other contextual information. Useful for translating
    GPS coordinates into human-readable locations.
    
    Args:
        latitude: The latitude coordinate (decimal degrees, WGS84)
        longitude: The longitude coordinate (decimal degrees, WGS84)
        
    Returns:
        Detailed address and location information including:
        - Formatted address
        - Building, street, city, state, country
        - Administrative hierarchy
        - OSM metadata
        - Postal code and other relevant identifiers
    """
    osm_client = ctx.request_context.lifespan_context.osm_client
    return await osm_client.reverse_geocode(latitude, longitude)


@mcp.tool()
async def analyze_image(
    image_url: str,
    ctx: Context,
    question: str = "What is in this image?",
    detail: str = "auto",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze an image with the configured multimodal LLM endpoint.

    This tool sends an image URL and an optional question to the configured LLM
    provider (for example Volcengine ARK / Doubao or another OpenAI-compatible
    multimodal endpoint) and returns a concise text answer.

    Args:
        image_url: Publicly accessible image URL
        question: Prompt or question about the image
        detail: Vision detail level ("auto", "low", "high")
        system_prompt: Optional system instruction override

    Returns:
        Provider/model metadata and the model's answer
    """
    await ctx.info(f"Analyzing image with detail={detail}")
    return await analyze_image_with_llm(
        image_url=image_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )


@mcp.tool()
async def analyze_local_image(
    image_path: str,
    ctx: Context,
    question: str = "What is in this image?",
    detail: str = "auto",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a local image file with the configured multimodal LLM endpoint.

    This tool reads a local image path, converts it into an inline data URL,
    and sends it to the configured multimodal model. It is useful for local
    FPV / PoV experiments where the image is not publicly hosted.

    Args:
        image_path: Absolute or user-resolvable local image path
        question: Prompt or question about the image
        detail: Vision detail level ("auto", "low", "high")
        system_prompt: Optional system instruction override

    Returns:
        Provider/model metadata and the model's answer
    """
    await ctx.info(f"Analyzing local image with detail={detail}")
    return await analyze_local_image_with_llm(
        image_path=image_path,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )


@mcp.tool()
async def search_nearby_pois(
    location: str,
    keywords: str,
    ctx: Context,
    radius: int = 1000,
    sortrule: str = "distance",
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search nearby POIs using semantic keywords against the active map provider.

    This tool is designed for domestic-map-first workflows where the LLM first
    converts user intent into provider-friendly search keywords such as
    "bookstore|cafe" or "convenience store|tea shop", then retrieves cleaned nearby candidates.

    Args:
        location: Current position in "longitude,latitude" format
        keywords: Provider-facing semantic keywords, optionally pipe-delimited
        radius: Search radius in meters
        sortrule: Sorting rule, typically "distance"
        limit: Maximum number of cleaned POI candidates to return

    Returns:
        Cleaned POI candidates with provider, coordinates, distance, type, and address
    """
    app_context = ctx.request_context.lifespan_context
    await ctx.info(f"Semantic POI search: provider={app_context.provider}, keywords={keywords}, radius={radius}m")
    results = await app_context.osm_client.search_semantic_pois(
        location=location,
        keywords=keywords,
        radius=radius,
        sortrule=sortrule,
        limit=limit,
    )
    return {
        "provider": app_context.provider,
        "query": {
            "location": location,
            "keywords": keywords,
            "radius": radius,
            "sortrule": sortrule,
            "limit": limit,
        },
        "results": results,
        "count": len(results),
    }


@mcp.tool()
async def get_coarse_waypoints(
    origin: str,
    destination: str,
    ctx: Context,
    mode: str = "foot",
) -> Dict[str, Any]:
    """
    Generate coarse semantic waypoints between origin and destination.

    Instead of returning dense geometry, this tool extracts route steps with
    topological meaning, suitable for hierarchical navigation or embodied-agent
    waypoint planning.

    Args:
        origin: Start point in "longitude,latitude" format
        destination: End point in "longitude,latitude" format
        mode: Travel mode ("foot", "car", "bike")

    Returns:
        Route summary plus coarse semantic waypoints derived from provider steps
    """
    app_context = ctx.request_context.lifespan_context
    await ctx.info(f"Coarse waypoint route: provider={app_context.provider}, mode={mode}")
    return await app_context.osm_client.get_coarse_waypoints(
        origin=origin,
        destination=destination,
        mode=mode,
    )


@mcp.tool()
async def plan_route_with_via_points(
    origin: str,
    destination: str,
    ctx: Context,
    via_points: Optional[List[Dict[str, Any]]] = None,
    mode: str = "foot",
) -> Dict[str, Any]:
    """
    Plan a multi-leg route with optional via points and return a combined route.

    This tool keeps the current provider interface simple by routing each leg with the
    existing single-origin-single-destination API, then stitching the legs together
    into one combined route polyline and one combined semantic waypoint list.

    Args:
        origin: Start point in "longitude,latitude" format
        destination: Final destination in "longitude,latitude" format
        via_points: Optional ordered list of intermediate stops. Each item may include:
            - location: "longitude,latitude"
            - name: optional label for UI/debugging
        mode: Travel mode ("foot", "car", "bike")

    Returns:
        Combined route summary, combined route polyline, combined coarse waypoints,
        and per-leg route payloads.
    """
    app_context = ctx.request_context.lifespan_context
    normalized_via_points = _normalize_via_points(via_points)
    await ctx.info(
        f"Planning route with via points: provider={app_context.provider}, mode={mode}, leg_count={max(0, len(normalized_via_points) + 1)}"
    )
    return await _compute_combined_route_plan(
        app_context=app_context,
        origin=origin,
        destination=destination,
        via_points=normalized_via_points,
        mode=mode,
    )


@mcp.tool()
async def add_via_point_to_route_plan(
    origin: str,
    destination: str,
    new_via_point: Dict[str, Any],
    ctx: Context,
    existing_via_points: Optional[List[Dict[str, Any]]] = None,
    insert_before_index: Optional[int] = None,
    mode: str = "foot",
) -> Dict[str, Any]:
    """
    Insert a new via point into an existing route plan and return the replanned route.

    This is a stateless helper for interaction loops. The caller keeps the current
    via_points list and passes it back on each interaction turn.

    Args:
        origin: Start point in "longitude,latitude" format
        destination: Final destination in "longitude,latitude" format
        new_via_point: New intermediate stop payload with location and optional name
        existing_via_points: Current ordered via point list
        insert_before_index: Optional insertion index. When omitted, append before destination.
        mode: Travel mode ("foot", "car", "bike")

    Returns:
        Updated via point list and the replanned combined route.
    """
    normalized_existing = _normalize_via_points(existing_via_points)
    normalized_new_list = _normalize_via_points([new_via_point])
    if not normalized_new_list:
        raise ValueError("new_via_point must include a valid location")
    inserted = normalized_new_list[0]

    if insert_before_index is None:
        normalized_existing.append(inserted)
    else:
        idx = max(0, min(int(insert_before_index), len(normalized_existing)))
        normalized_existing.insert(idx, inserted)

    for index, item in enumerate(normalized_existing):
        item["index"] = index

    return await plan_route_with_via_points(
        origin=origin,
        destination=destination,
        via_points=normalized_existing,
        mode=mode,
        ctx=ctx,
    )


@mcp.tool()
async def create_route_session(
    origin: str,
    destination: str,
    ctx: Context,
    via_points: Optional[List[Dict[str, Any]]] = None,
    mode: str = "foot",
    session_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a backend-persisted route session with optional via points.

    The returned session can later be updated interactively by adding/removing via points,
    replacing the destination, or replanning from a new current origin.
    """
    app_context = ctx.request_context.lifespan_context
    normalized_via_points = _normalize_via_points(via_points)
    route_plan = await _compute_combined_route_plan(
        app_context=app_context,
        origin=origin,
        destination=destination,
        via_points=normalized_via_points,
        mode=mode,
    )
    session_id = f"route_{uuid4().hex[:10]}"
    session_payload = {
        "session_id": session_id,
        "session_name": str(session_name or "").strip() or session_id,
        "provider": app_context.provider,
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "via_points": normalized_via_points,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    _cache_route_plan_on_session(session_payload, route_plan)
    _save_persisted_route_session(session_payload)
    return {
        "session": session_payload,
        "route_plan": route_plan,
        "route_plan_cache_hit": False,
        "session_store": _route_session_store_info(),
    }


@mcp.tool()
async def get_route_session(
    session_id: str,
    ctx: Context,
    force_replan: bool = False,
) -> Dict[str, Any]:
    """
    Fetch a route session and return its cached route plan by default.

    Set force_replan=true only when the caller explicitly wants to refresh the
    provider route. This avoids hitting provider QPS limits on frequent UI polls.
    """
    app_context = ctx.request_context.lifespan_context
    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    route_plan, cache_hit = await _route_plan_for_session(
        app_context=app_context,
        session=session,
        force_replan=bool(force_replan),
    )
    return {
        "session": session,
        "route_plan": route_plan,
        "route_plan_cache_hit": cache_hit,
        "session_store": _route_session_store_info(),
    }


@mcp.tool()
async def list_route_sessions(
    ctx: Context,
    limit: int = 20,
    include_route_plan_cache: bool = False,
) -> Dict[str, Any]:
    """
    List persisted route sessions known to this backend.

    This is useful for UI recovery: after a page refresh or backend restart, the
    UI can show recoverable sessions or verify that a session_id is still valid.
    """
    _load_route_sessions()
    bounded_limit = max(1, min(int(limit), 100))
    sessions = sorted(
        ROUTE_SESSIONS.values(),
        key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
        reverse=True,
    )[:bounded_limit]
    if not include_route_plan_cache:
        sessions = [
            {
                key: value
                for key, value in session.items()
                if key not in {"route_plan_cache"}
            }
            for session in sessions
        ]
    return {
        "sessions": sessions,
        "count": len(sessions),
        "total_count": len(ROUTE_SESSIONS),
        "session_store": _route_session_store_info(),
    }


@mcp.tool()
async def add_via_point_to_session(
    session_id: str,
    new_via_point: Dict[str, Any],
    ctx: Context,
    insert_before_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Insert or append a via point inside an existing route session.
    """
    app_context = ctx.request_context.lifespan_context
    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    normalized_new = _normalize_via_points([new_via_point])
    if not normalized_new:
        raise ValueError("new_via_point must include a valid location")
    old_via_points = _normalize_via_points(session.get("via_points"))
    via_points = [dict(item) for item in old_via_points]
    item = normalized_new[0]
    same_as_destination = _locations_equivalent(item.get("location"), session.get("destination"))
    duplicate_index = next(
        (
            index for index, via in enumerate(via_points)
            if _locations_equivalent(via.get("location"), item.get("location"))
        ),
        None,
    )
    if same_as_destination or duplicate_index is not None:
        route_plan, cache_hit = await _route_plan_for_session(
            app_context=app_context,
            session=session,
            force_replan=False,
        )
        return {
            "session": session,
            "route_plan": route_plan,
            "route_plan_cache_hit": cache_hit,
            "session_changed": False,
            "change_reason": "matches_destination" if same_as_destination else f"duplicate_via_point:{duplicate_index}",
            "route_plan_update_mode": "cache_reuse_noop",
            "session_store": _route_session_store_info(),
        }
    inserted_index = len(via_points)
    if insert_before_index is None:
        via_points.append(item)
    else:
        inserted_index = max(0, min(int(insert_before_index), len(via_points)))
        via_points.insert(inserted_index, item)
    for index, via in enumerate(via_points):
        via["index"] = index
    session["via_points"] = via_points
    session["updated_at"] = datetime.now().isoformat()
    cached_route_plan = session.get("route_plan_cache")
    route_plan = None
    route_plan_update_mode = "full_replan"
    if isinstance(cached_route_plan, dict) and cached_route_plan:
        route_plan = await _compute_incremental_route_plan_with_inserted_via(
            app_context=app_context,
            origin=session["origin"],
            destination=session["destination"],
            mode=session["mode"],
            cached_route_plan=cached_route_plan,
            old_via_points=old_via_points,
            new_via_points=via_points,
            insert_index=inserted_index,
        )
        if isinstance(route_plan, dict) and route_plan:
            route_plan_update_mode = str(route_plan.get("replan_strategy") or "incremental_insert_via_split_leg")
    if route_plan is None:
        route_plan = await _compute_combined_route_plan(
            app_context=app_context,
            origin=session["origin"],
            destination=session["destination"],
            via_points=via_points,
            mode=session["mode"],
        )
    _cache_route_plan_on_session(session, route_plan)
    _save_persisted_route_session(session)
    return {
        "session": session,
        "route_plan": route_plan,
        "route_plan_cache_hit": False,
        "session_changed": True,
        "change_reason": "via_point_added",
        "route_plan_update_mode": route_plan_update_mode,
        "session_store": _route_session_store_info(),
    }


@mcp.tool()
async def remove_via_point_from_session(
    session_id: str,
    ctx: Context,
    via_point_index: Optional[int] = None,
    via_point_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Remove one via point from an existing route session by index or by name.
    """
    app_context = ctx.request_context.lifespan_context
    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    via_points = _normalize_via_points(session.get("via_points"))
    removed = None
    if via_point_index is not None and 0 <= int(via_point_index) < len(via_points):
        removed = via_points.pop(int(via_point_index))
    elif str(via_point_name or "").strip():
        name = str(via_point_name).strip().lower()
        for index, via in enumerate(via_points):
            if str(via.get("name") or "").strip().lower() == name:
                removed = via_points.pop(index)
                break
    if removed is None:
        raise KeyError("No matching via point found to remove")
    for index, via in enumerate(via_points):
        via["index"] = index
    session["via_points"] = via_points
    session["updated_at"] = datetime.now().isoformat()
    route_plan = await _compute_combined_route_plan(
        app_context=app_context,
        origin=session["origin"],
        destination=session["destination"],
        via_points=via_points,
        mode=session["mode"],
    )
    _cache_route_plan_on_session(session, route_plan)
    _save_persisted_route_session(session)
    return {
        "session": session,
        "removed_via_point": removed,
        "route_plan": route_plan,
        "route_plan_cache_hit": False,
        "session_store": _route_session_store_info(),
    }


@mcp.tool()
async def replace_destination_in_session(
    session_id: str,
    destination: str,
    ctx: Context,
) -> Dict[str, Any]:
    """
    Replace the final destination of an existing route session and replan.
    """
    app_context = ctx.request_context.lifespan_context
    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    session["destination"] = destination
    session["updated_at"] = datetime.now().isoformat()
    route_plan = await _compute_combined_route_plan(
        app_context=app_context,
        origin=session["origin"],
        destination=session["destination"],
        via_points=session.get("via_points"),
        mode=session["mode"],
    )
    _cache_route_plan_on_session(session, route_plan)
    _save_persisted_route_session(session)
    return {
        "session": session,
        "route_plan": route_plan,
        "route_plan_cache_hit": False,
        "session_store": _route_session_store_info(),
    }


@mcp.tool()
async def replan_route_session_from_origin(
    session_id: str,
    origin: str,
    ctx: Context,
) -> Dict[str, Any]:
    """
    Update the current origin of a route session and replan from the new location.
    """
    app_context = ctx.request_context.lifespan_context
    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    session["origin"] = origin
    session["updated_at"] = datetime.now().isoformat()
    route_plan = await _compute_combined_route_plan(
        app_context=app_context,
        origin=session["origin"],
        destination=session["destination"],
        via_points=session.get("via_points"),
        mode=session["mode"],
    )
    _cache_route_plan_on_session(session, route_plan)
    _save_persisted_route_session(session)
    return {
        "session": session,
        "route_plan": route_plan,
        "route_plan_cache_hit": False,
        "session_store": _route_session_store_info(),
    }


@mcp.tool()
async def render_bev_route_snapshot(
    origin: str,
    destination: str,
    ctx: Context,
    mode: str = "foot",
    refresh_interval_s: float = 5.0,
    size: str = "800*600",
    scale: int = 2,
    zoom: Optional[int] = None,
    traffic: bool = False,
) -> Dict[str, Any]:
    """
    Render an AMap static bird's-eye-view route snapshot and return a refreshable image URL.

    This tool computes the current route from origin to destination using the active AMap
    provider, then builds an AMap static-map URL with the route polyline overlaid. Call it
    periodically (for example every 2-5 seconds) with updated origin coordinates to obtain
    refreshed BEV planning images.

    Args:
        origin: Start point in "longitude,latitude" format
        destination: End point in "longitude,latitude" format
        mode: Travel mode ("foot", "car", "bike")
        refresh_interval_s: Suggested polling interval in seconds
        size: Static image size in "width*height" format
        scale: Static map scale, typically 1 or 2
        zoom: Optional zoom level override
        traffic: Whether to enable traffic layer

    Returns:
        Static map URL, route summary, route polyline, and refresh metadata
    """
    app_context = ctx.request_context.lifespan_context
    if app_context.provider != "amap":
        raise RuntimeError("render_bev_route_snapshot currently requires MAP_PROVIDER=amap")

    await ctx.info(f"Rendering AMap BEV snapshot: mode={mode}, refresh_interval_s={refresh_interval_s}")
    route_data = await app_context.osm_client.get_coarse_waypoints(
        origin=origin,
        destination=destination,
        mode=mode,
    )

    route_polyline = route_data.get("route_polyline") or []
    if not route_polyline:
        coarse_waypoints = route_data.get("coarse_waypoints", []) or []
        route_polyline = [
            [
                float((item.get("coordinate", {}) or {}).get("longitude")),
                float((item.get("coordinate", {}) or {}).get("latitude")),
            ]
            for item in coarse_waypoints
            if isinstance(item, dict)
            and (item.get("coordinate", {}) or {}).get("longitude") is not None
            and (item.get("coordinate", {}) or {}).get("latitude") is not None
        ]
    if not route_polyline:
        raise RuntimeError("No route polyline available for BEV snapshot rendering.")

    amap_key = os.getenv("AMAP_KEY", "").strip()
    if not amap_key:
        raise RuntimeError("AMAP_KEY is required for AMap static BEV snapshots.")

    image_url = build_amap_staticmap_url(
        api_key=amap_key,
        path_coords=route_polyline,
        origin=origin,
        destination=destination,
        size=size,
        scale=scale,
        zoom=zoom,
        traffic=1 if traffic else 0,
    )
    cache_busted_url = f"{image_url}&_ts={int(datetime.now().timestamp())}"

    origin_lon, origin_lat = parse_location_text(origin)
    destination_lon, destination_lat = parse_location_text(destination)

    return {
        "provider": app_context.provider,
        "snapshot_provider": "amap_staticmap",
        "origin": {
            "location": origin,
            "coordinates": {"latitude": origin_lat, "longitude": origin_lon},
        },
        "destination": {
            "location": destination,
            "coordinates": {"latitude": destination_lat, "longitude": destination_lon},
        },
        "mode": mode,
        "route_summary": route_data.get("route_summary", {}),
        "route_polyline": [
            {"longitude": float(point[0]), "latitude": float(point[1])}
            for point in route_polyline
            if isinstance(point, (list, tuple)) and len(point) == 2
        ],
        "image_url": image_url,
        "image_url_cache_busted": cache_busted_url,
        "refresh_interval_s": max(0.5, float(refresh_interval_s)),
        "generated_at": datetime.now().isoformat(),
        "usage": {
            "polling": "Call this tool again with updated origin coordinates at refresh_interval_s cadence.",
            "recommended_refresh_after_s": max(0.5, float(refresh_interval_s)),
        },
    }


@mcp.tool()
async def render_itinerary_bev_snapshot(
    ctx: Context,
    route_plan: Optional[Dict[str, Any]] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    via_points: Optional[List[Dict[str, Any]]] = None,
    mode: str = "foot",
    size: str = "800*600",
    scale: int = 2,
    zoom: Optional[int] = None,
    traffic: bool = False,
) -> Dict[str, Any]:
    """
    Render an AMap static BEV snapshot for a multi-leg itinerary route.

    Pass either an existing route_plan returned by recommend_itinerary_and_waypoints /
    plan_route_with_via_points, or pass origin + destination + via_points so this tool
    can compute the route plan first.
    """
    app_context = ctx.request_context.lifespan_context
    if app_context.provider != "amap":
        raise RuntimeError("render_itinerary_bev_snapshot currently requires MAP_PROVIDER=amap")

    amap_key = os.getenv("AMAP_KEY", "").strip()
    if not amap_key:
        raise RuntimeError("AMAP_KEY is required for AMap static itinerary snapshots.")

    if not isinstance(route_plan, dict) or not route_plan.get("route_polyline"):
        if not origin or not destination:
            raise ValueError("Either route_plan or both origin and destination are required")
        route_plan = await _compute_combined_route_plan(
            app_context=app_context,
            origin=origin,
            destination=destination,
            via_points=via_points,
            mode=mode,
        )

    route_polyline = _normalize_route_coords(route_plan.get("route_polyline", []) or [])
    if not route_polyline:
        raise RuntimeError("No route polyline available for itinerary BEV snapshot rendering.")

    legs = route_plan.get("legs", []) or []
    leg_colors = ["0x2E86FF", "0xFF8C42", "0x9B5DE5", "0x00A86B", "0xFF4D6D"]
    path_specs: List[str] = []
    valid_legs = [leg for leg in legs if isinstance(leg, dict)]
    max_total_points = int(os.getenv("AMAP_ITINERARY_STATIC_MAX_POINTS", "120") or 120)
    per_leg_points = max(12, min(45, max_total_points // max(1, len(valid_legs))))
    for index, leg in enumerate(legs):
        if not isinstance(leg, dict):
            continue
        leg_coords = _normalize_route_coords(leg.get("route_polyline", []) or [])
        if leg_coords:
            color = leg_colors[index % len(leg_colors)]
            sampled = sample_route_coordinates(leg_coords, max_points=per_leg_points)
            path_specs.append(f"7,{color},1,,:{encode_polyline_text(sampled)}")

    if not path_specs:
        sampled = sample_route_coordinates(route_polyline, max_points=max_total_points)
        path_specs.append(f"6,0x2E8B57,1,,:{encode_polyline_text(sampled)}")

    combined_sampled = sample_route_coordinates(route_polyline, max_points=max_total_points)
    combined_path_spec = f"7,0x2E8B57,1,,:{encode_polyline_text(combined_sampled)}"

    stop_sequence = route_plan.get("stop_sequence", []) or []
    markers: List[str] = []
    via_marker_index = 0
    for index, stop in enumerate(stop_sequence):
        if not isinstance(stop, dict):
            continue
        location = str(stop.get("location") or "").strip()
        if not location:
            continue
        try:
            lon, lat = parse_location_text(location)
        except Exception:
            continue
        stop_type = str(stop.get("type") or "").strip().lower()
        if stop_type == "origin":
            markers.append(f"mid,0x2E86FF,O:{lon},{lat}")
        elif stop_type == "destination":
            markers.append(f"mid,0xFF5533,D:{lon},{lat}")
        else:
            via_marker_index += 1
            marker_label = _single_char_marker_label(str(via_marker_index), fallback="V")
            markers.append(f"mid,0xFF8C42,{marker_label}:{lon},{lat}")

    params: List[Tuple[str, str]] = [
        ("key", amap_key),
        ("size", size),
        ("scale", str(max(1, min(int(scale), 2)))),
        ("traffic", str(1 if traffic else 0)),
        ("paths", combined_path_spec),
    ]
    if zoom is not None:
        params.append(("zoom", str(int(zoom))))
    if markers:
        params.append(("markers", "|".join(markers)))

    query = "&".join(f"{key}={value}" for key, value in params)
    image_url = f"https://restapi.amap.com/v3/staticmap?{query}"
    cache_busted_url = f"{image_url}&_ts={int(datetime.now().timestamp())}"

    leg_overlay_image_url: Optional[str] = None
    if len(path_specs) > 1:
        leg_overlay_params = [(key, value) for key, value in params if key != "paths"]
        leg_overlay_params.append(("paths", "|".join(path_specs)))
        leg_overlay_query = "&".join(f"{key}={value}" for key, value in leg_overlay_params)
        leg_overlay_image_url = f"https://restapi.amap.com/v3/staticmap?{leg_overlay_query}"

    route_only_params = [(key, value) for key, value in params if key != "markers"]
    route_only_query = "&".join(f"{key}={value}" for key, value in route_only_params)
    route_only_image_url = f"https://restapi.amap.com/v3/staticmap?{route_only_query}"

    markers_only_image_url: Optional[str] = None
    if markers:
        markers_only_params: List[Tuple[str, str]] = [
            ("key", amap_key),
            ("size", size),
            ("scale", str(max(1, min(int(scale), 2)))),
            ("traffic", str(1 if traffic else 0)),
            ("markers", "|".join(markers)),
        ]
        if zoom is not None:
            markers_only_params.append(("zoom", str(int(zoom))))
        markers_only_query = "&".join(f"{key}={value}" for key, value in markers_only_params)
        markers_only_image_url = f"https://restapi.amap.com/v3/staticmap?{markers_only_query}"

    return {
        "provider": app_context.provider,
        "snapshot_provider": "amap_staticmap_itinerary",
        "mode": route_plan.get("mode", mode),
        "route_summary": route_plan.get("route_summary", {}),
        "via_points": route_plan.get("via_points", []),
        "stop_sequence": stop_sequence,
        "leg_count": len(legs),
        "image_url": image_url,
        "image_url_cache_busted": cache_busted_url,
        "leg_overlay_image_url": leg_overlay_image_url,
        "route_only_image_url": route_only_image_url,
        "markers_only_image_url": markers_only_image_url,
        "marker_labels": [marker.split(":", 1)[0].rsplit(",", 1)[-1] for marker in markers],
        "url_length": len(cache_busted_url),
        "rendering_note": "Primary image_url uses one simplified combined route for AMap reliability; leg_overlay_image_url keeps per-leg colors when supported.",
        "generated_at": datetime.now().isoformat(),
    }


@mcp.tool()
async def get_route_context(
    gps: Dict[str, float],
    route_polyline: List[Dict[str, Any]] | List[List[float]],
    coarse_waypoints: List[Dict[str, Any]],
    ctx: Context,
    heading_deg: Optional[float] = None,
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
) -> Dict[str, Any]:
    """
    Build a lightweight runtime RouteContext from current pose and an existing route.

    This tool does not perform fresh routing. It takes an already computed route polyline
    plus coarse semantic waypoints and compresses them into a stable local route context
    suitable for downstream embodied reasoning.

    Args:
        gps: Current pose as {"lat": ..., "lng": ...}
        route_polyline: Full route polyline from recommendation or routing output
        coarse_waypoints: Coarse semantic route steps
        heading_deg: Optional current heading in degrees
        lookahead_m: Forward local context distance in meters
        off_route_threshold_m: Cross-track error threshold in meters for off-route detection

    Returns:
        RouteContext with pose match, progress, local semantics, and local geometry
    """
    await ctx.info(
        "Building RouteContext "
        f"heading={heading_deg if heading_deg is not None else 'none'} "
        f"lookahead_m={lookahead_m:.1f}"
    )
    route_context = build_route_context(
        gps=gps,
        heading_deg=heading_deg,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        lookahead_m=lookahead_m,
        off_route_threshold_m=off_route_threshold_m,
    )
    return route_context_to_dict(route_context)


@mcp.tool()
async def get_local_bev_context(
    gps: Dict[str, float],
    route_polyline: List[Dict[str, Any]] | List[List[float]],
    coarse_waypoints: List[Dict[str, Any]],
    ctx: Context,
    heading_deg: Optional[float] = None,
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
    front_extent_m: float = 30.0,
    back_extent_m: float = 10.0,
    side_extent_m: float = 15.0,
) -> Dict[str, Any]:
    """
    Build a vectorized LocalBEVContext aligned to the current pose.

    This is the explicit local-BEV branch parallel to RouteContext. It preserves
    a local route patch in pose-aligned coordinates for downstream BEV-to-PoV reasoning.
    """
    await ctx.info(
        "Building LocalBEVContext "
        f"heading={heading_deg if heading_deg is not None else 'none'} "
        f"lookahead_m={lookahead_m:.1f}"
    )
    route_context = build_route_context(
        gps=gps,
        heading_deg=heading_deg,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        lookahead_m=lookahead_m,
        off_route_threshold_m=off_route_threshold_m,
    )
    local_bev_context = build_local_bev_context(
        gps=gps,
        heading_deg=heading_deg,
        route_context=route_context,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        front_extent_m=front_extent_m,
        back_extent_m=back_extent_m,
        side_extent_m=side_extent_m,
    )
    return local_bev_context_to_dict(local_bev_context)


@mcp.tool()
async def get_route_state(
    gps: Dict[str, float],
    route_polyline: List[Dict[str, Any]] | List[List[float]],
    coarse_waypoints: List[Dict[str, Any]],
    ctx: Context,
    heading_deg: Optional[float] = None,
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
    front_extent_m: float = 30.0,
    back_extent_m: float = 10.0,
    side_extent_m: float = 15.0,
) -> Dict[str, Any]:
    """
    Build the full route runtime state:
    - structured RouteContext
    - explicit LocalBEVContext
    """
    await ctx.info(
        "Building route state "
        f"heading={heading_deg if heading_deg is not None else 'none'} "
        f"lookahead_m={lookahead_m:.1f}"
    )
    state = build_route_state(
        gps=gps,
        heading_deg=heading_deg,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        lookahead_m=lookahead_m,
        off_route_threshold_m=off_route_threshold_m,
        front_extent_m=front_extent_m,
        back_extent_m=back_extent_m,
        side_extent_m=side_extent_m,
    )
    return route_state_to_dict(state)


@mcp.tool()
async def segment_fpv_frame(
    ctx: Context,
    sequence_name: str,
    sam3_output_base_dir: str,
    image_path: Optional[str] = None,
    frame_name: Optional[str] = None,
    camera_view: str = "camera_front",
    run_sam3: bool = False,
    sam3_env_name: str = "sam3",
    sam3_script: str = "/home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/prepare_frames_and_run_batch_sam3.py",
    rides_base_dir: Optional[str] = None,
    frames_base_dir: Optional[str] = None,
    recordings_subdir: str = "recordings",
    stream_pattern: str = "*_video.m3u8",
    fps: float = 1.0,
    gpus: Optional[List[int]] = None,
    min_score: float = 0.3,
    finetuned: bool = True,
    checkpoint_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    overwrite_frames: bool = False,
) -> Dict[str, Any]:
    """
    Build or load SAM3 segmentation artifacts for an FPV frame.

    This tool supports two modes:
    - artifact lookup only: run_sam3=false, expects existing SAM3 outputs
    - full execution: run_sam3=true, launches external SAM3 pipeline first
    """
    await ctx.info(
        "Resolving SAM3 segmentation artifacts "
        f"sequence={sequence_name} camera={camera_view} run_sam3={run_sam3}"
    )
    return await segment_fpv_frame_service(
        sequence_name=sequence_name,
        sam3_output_base_dir=sam3_output_base_dir,
        image_path=image_path,
        frame_name=frame_name,
        camera_view=camera_view,
        run_sam3=run_sam3,
        sam3_env_name=sam3_env_name,
        sam3_script=sam3_script,
        rides_base_dir=rides_base_dir,
        frames_base_dir=frames_base_dir,
        recordings_subdir=recordings_subdir,
        stream_pattern=stream_pattern,
        fps=fps,
        gpus=gpus,
        min_score=min_score,
        finetuned=finetuned,
        checkpoint_path=checkpoint_path,
        max_frames=max_frames,
        overwrite_frames=overwrite_frames,
    )


@mcp.tool()
async def run_zero_shot_visual_prompt_agent(
    gps: Dict[str, float],
    route_polyline: List[Dict[str, Any]] | List[List[float]],
    coarse_waypoints: List[Dict[str, Any]],
    ctx: Context,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    fpv_segmentation: Optional[Dict[str, Any]] = None,
    sam3_request: Optional[Dict[str, Any]] = None,
    heading_deg: Optional[float] = None,
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
    front_extent_m: float = 30.0,
    back_extent_m: float = 10.0,
    side_extent_m: float = 15.0,
    visual_top_k: int = 5,
    candidate_top_k: int = 5,
    llm_detail: str = "auto",
    navigation_mode: str = "foot",
    enable_verifier: bool = True,
    emit_qa_bundle: bool = False,
) -> Dict[str, Any]:
    """
    Run a zero-shot visual prompt agent over the current FPV frame.

    The online decision path is mode-aware and image-first:
    1. build RouteContext + LocalBEVContext
    2. pass compact RouteContext directly to the LVLM
    3. let the LVLM render one conservative image-plane prompt from image + RouteContext
    4. if the LVLM fails, return reject_all instead of switching to heuristic fallbacks

    The public input/output schema is unchanged; candidate/debug fields remain for
    backward compatibility but are empty in this path.
    """
    await ctx.info(
        "Running zero-shot visual prompt agent "
        f"heading={heading_deg if heading_deg is not None else 'none'} "
        f"visual_top_k={visual_top_k} candidate_top_k={candidate_top_k} "
        f"enable_verifier={enable_verifier}"
    )
    return await run_zero_shot_visual_prompt_pipeline(
        gps=gps,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        image_path=image_path,
        image_url=image_url,
        fpv_segmentation=fpv_segmentation,
        sam3_request=sam3_request,
        heading_deg=heading_deg,
        lookahead_m=lookahead_m,
        off_route_threshold_m=off_route_threshold_m,
        front_extent_m=front_extent_m,
        back_extent_m=back_extent_m,
        side_extent_m=side_extent_m,
        visual_top_k=visual_top_k,
        candidate_top_k=candidate_top_k,
        llm_detail=llm_detail,
        navigation_mode=navigation_mode,
        enable_verifier=enable_verifier,
        emit_qa_bundle=emit_qa_bundle,
    )


@mcp.tool()
async def render_local_bev_patch(
    gps: Dict[str, float],
    route_polyline: List[Dict[str, Any]] | List[List[float]],
    coarse_waypoints: List[Dict[str, Any]],
    ctx: Context,
    heading_deg: Optional[float] = None,
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
    front_extent_m: float = 30.0,
    back_extent_m: float = 10.0,
    side_extent_m: float = 15.0,
    width: int = 640,
    height: int = 640,
) -> Dict[str, Any]:
    """
    Render a local pose-aligned BEV patch from LocalBEVContext.

    This tool returns an explicit local BEV SVG patch for debugging.
    The patch is centered on the current pose and aligned so the current heading points upward.
    """
    await ctx.info(
        "Rendering LocalBEVContext patch "
        f"heading={heading_deg if heading_deg is not None else 'none'} "
        f"lookahead_m={lookahead_m:.1f}"
    )
    route_state = build_route_state(
        gps=gps,
        heading_deg=heading_deg,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        lookahead_m=lookahead_m,
        off_route_threshold_m=off_route_threshold_m,
        front_extent_m=front_extent_m,
        back_extent_m=back_extent_m,
        side_extent_m=side_extent_m,
    )
    route_state_dict = route_state_to_dict(route_state)
    local_bev_context = route_state_dict.get("local_bev_context", {}) or {}
    svg = _build_local_bev_svg(
        local_bev_context=local_bev_context,
        width=max(256, int(width)),
        height=max(256, int(height)),
    )
    return {
        "local_bev_context": local_bev_context,
        "route_context": route_state_dict.get("route_context", {}),
        "svg": svg,
        "svg_data_url": f"data:image/svg+xml;charset=utf-8,{quote(svg)}",
        "generated_at": datetime.now().isoformat(),
        "usage": {
            "legend": {
                "green": "route polyline local",
                "blue": "local segment",
                "orange": "connector",
                "blue sector": "corridor prior",
                "orange sector": "turn prior",
                "C": "current pose",
                "M": "matched point",
                "W": "next waypoint",
                "T": "next turn",
            }
        },
    }


@mcp.tool()
async def render_bev_route_context_snapshot(
    gps: Dict[str, float],
    route_polyline: List[Dict[str, Any]] | List[List[float]],
    coarse_waypoints: List[Dict[str, Any]],
    ctx: Context,
    heading_deg: Optional[float] = None,
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
    size: str = "800*600",
    scale: int = 2,
    zoom: Optional[int] = None,
    traffic: bool = False,
) -> Dict[str, Any]:
    """
    Render an AMap BEV snapshot with RouteContext overlays.

    The rendered image overlays:
    - the full planned route
    - the local route segment around the matched point
    - the connector polyline when the pose is still merging toward the provider route start

    Args:
        gps: Current pose as {"lat": ..., "lng": ...}
        route_polyline: Full route polyline from recommendation or routing output
        coarse_waypoints: Coarse semantic route steps
        heading_deg: Optional current heading in degrees
        lookahead_m: Forward local context distance in meters
        off_route_threshold_m: Cross-track error threshold in meters for off-route detection
        size: Static image size in "width*height" format
        scale: Static map scale, typically 1 or 2
        zoom: Optional zoom level override
        traffic: Whether to enable traffic layer

    Returns:
        RouteContext plus a refreshable BEV image URL with context overlays
    """
    app_context = ctx.request_context.lifespan_context
    if app_context.provider != "amap":
        raise RuntimeError("render_bev_route_context_snapshot currently requires MAP_PROVIDER=amap")

    await ctx.info(
        "Rendering RouteContext BEV snapshot "
        f"heading={heading_deg if heading_deg is not None else 'none'} "
        f"lookahead_m={lookahead_m:.1f}"
    )

    route_context = build_route_context(
        gps=gps,
        heading_deg=heading_deg,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        lookahead_m=lookahead_m,
        off_route_threshold_m=off_route_threshold_m,
    )
    route_context_dict = route_context_to_dict(route_context)
    local_bev_context = build_local_bev_context(
        gps=gps,
        heading_deg=heading_deg,
        route_context=route_context,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
    )
    local_bev_context_dict = local_bev_context_to_dict(local_bev_context)

    base_route_coords = _normalize_route_coords(route_polyline)
    if not base_route_coords:
        raise RuntimeError("No route_polyline available for RouteContext BEV snapshot rendering.")

    amap_key = os.getenv("AMAP_KEY", "").strip()
    if not amap_key:
        raise RuntimeError("AMAP_KEY is required for AMap static RouteContext BEV snapshots.")

    local_segment_coords = _normalize_route_coords(route_context_dict.get("geometry", {}).get("local_segment_polyline", []))
    connector_coords = _normalize_route_coords(route_context_dict.get("geometry", {}).get("connector_polyline", []))

    extra_paths: List[Tuple[str, List[List[float]]]] = []
    if local_segment_coords:
        extra_paths.append(("8,0x2E86FF,1,,", local_segment_coords))
    if connector_coords:
        extra_paths.append(("6,0xFF8C42,1,,", connector_coords))

    current_lon = float(gps["lng"])
    current_lat = float(gps["lat"])
    destination_lon, destination_lat = base_route_coords[-1]
    matched_point = route_context_dict.get("pose_match", {}).get("matched_point", {}) or {}
    matched_lon = matched_point.get("lng")
    matched_lat = matched_point.get("lat")

    markers = [
        f"mid,0x2E86FF,C:{current_lon},{current_lat}",
        f"mid,0xFF5533,D:{destination_lon},{destination_lat}",
    ]
    if matched_lon is not None and matched_lat is not None:
        markers.append(f"mid,0x00A86B,M:{matched_lon},{matched_lat}")

    image_url = _build_amap_staticmap_overlay_url(
        api_key=amap_key,
        base_route_coords=base_route_coords,
        extra_paths=extra_paths,
        markers=markers,
        size=size,
        scale=scale,
        zoom=zoom,
        traffic=1 if traffic else 0,
    )
    cache_busted_url = f"{image_url}&_ts={int(datetime.now().timestamp())}"

    return {
        "provider": app_context.provider,
        "snapshot_provider": "amap_staticmap_route_context",
        "gps": {"latitude": current_lat, "longitude": current_lon},
        "heading_deg": heading_deg,
        "route_context": route_context_dict,
        "local_bev_context": local_bev_context_dict,
        "route_polyline": [
            {"longitude": float(point[0]), "latitude": float(point[1])}
            for point in base_route_coords
        ],
        "image_url": image_url,
        "image_url_cache_busted": cache_busted_url,
        "refresh_interval_s": 3.0,
        "generated_at": datetime.now().isoformat(),
        "usage": {
            "polling": "Call this tool again with updated GPS and heading at refresh cadence.",
            "recommended_refresh_after_s": 3.0,
            "overlay_legend": {
                "green": "full route",
                "blue": "local route segment",
                "orange": "connector polyline",
                "C": "current pose",
                "M": "matched point",
                "D": "destination",
            },
        },
    }


@mcp.tool()
async def find_nearby_places(
    latitude: float,
    longitude: float,
    ctx: Context,
    radius: float = 1000,  # meters
    categories: List[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Discover points of interest and amenities near a specific location.
    
    This tool performs a comprehensive search around a geographic point to identify
    nearby establishments, amenities, and points of interest. Results are organized by
    category and subcategory, making it easy to find specific types of places. Essential
    for location-based recommendations, neighborhood analysis, and proximity-based decision making.
    
    Args:
        latitude: Center point latitude (decimal degrees)
        longitude: Center point longitude (decimal degrees)
        radius: Search radius in meters (defaults to 1000m/1km)
        categories: List of OSM categories to search for (e.g., ["amenity", "shop", "tourism"]).
                   If omitted, searches common categories.
        limit: Maximum number of total results to return
        
    Returns:
        Structured dictionary containing:
        - Original query parameters
        - Total count of places found
        - Results grouped by category and subcategory
        - Each place includes name, coordinates, and associated tags
    """
    osm_client = ctx.request_context.lifespan_context.osm_client
    
    # Set default categories if not provided
    if not categories:
        categories = ["amenity", "shop", "tourism", "leisure"]
    
    await ctx.info(f"Searching for places within {radius}m of ({latitude}, {longitude})")
    places = await osm_client.get_nearby_pois(latitude, longitude, radius, categories)
    
    # Group results into category -> subcategory -> list[place] for easier LLM summarization.
    results_by_category = {}
    
    for place in places[:limit]:
        tags = place.get("tags", {})
        
        # Find the matching category
        for category in categories:
            if category in tags:
                subcategory = tags[category]
                if category not in results_by_category:
                    results_by_category[category] = {}
                
                if subcategory not in results_by_category[category]:
                    results_by_category[category][subcategory] = []
                
                # Add place to appropriate category and subcategory
                place_info = {
                    "id": place.get("id"),
                    "name": tags.get("name", "Unnamed"),
                    "latitude": place.get("lat"),
                    "longitude": place.get("lon"),
                    "tags": tags
                }
                
                results_by_category[category][subcategory].append(place_info)
    
    # Calculate total count
    total_count = sum(
        len(places)
        for category_data in results_by_category.values()
        for places in category_data.values()
    )
    
    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius
        },
        "categories": results_by_category,
        "total_count": total_count
    }

@mcp.tool()
async def get_route_directions(
    from_latitude: float,
    from_longitude: float,
    to_latitude: float,
    to_longitude: float,
    ctx: Context,
    mode: str = "car",
    steps: bool = False,
    overview: str = "simplified",
    annotations: bool = False
) -> Dict[str, Any]:
    """
    Calculate detailed route directions between two geographic points.
    
    This tool provides comprehensive routing information between two locations using OpenStreetMap/OSRM.
    The output can be minimized using the steps, overview, and annotations parameters to reduce the response size.
    
    Args:
        from_latitude: Starting point latitude (decimal degrees)
        from_longitude: Starting point longitude (decimal degrees)
        to_latitude: Destination latitude (decimal degrees)
        to_longitude: Destination longitude (decimal degrees)
        ctx: Context (provided internally by MCP)
        mode: Transportation mode ("car", "bike", "foot")
        steps: Turn-by-turn instructions (True/False, Default: False)
        overview: Geometry output ("full", "simplified", "false"; Default: "simplified")
        annotations: Additional segment info (True/False, Default: False)
    
    Returns:
        Dictionary with routing information (summary, directions, geometry, waypoints)

    Example:
        {
          "from_latitude": 51.3334193,
          "from_longitude": 9.4540423,
          "to_latitude": 51.3295516,
          "to_longitude": 9.4576721,
          "mode": "car",
          "steps": false,
          "overview": "simplified",
          "annotations": false
        }
    """
    osm_client = ctx.request_context.lifespan_context.osm_client
    
    # Validate transportation mode
    valid_modes = ["car", "bike", "foot"]
    if mode not in valid_modes:
        await ctx.warning(f"Invalid mode '{mode}'. Using 'car' instead.")
        mode = "car"
    
    await ctx.info(f"Calculating {mode} route from ({from_latitude}, {from_longitude}) to ({to_latitude}, {to_longitude})")
    
    # Get route from OSRM
    route_data = await osm_client.get_route(
        from_latitude, from_longitude,
        to_latitude, to_longitude,
        mode,
        steps=steps,
        overview=overview,
        annotations=annotations
    )
    
    # Process and simplify the response
    if "routes" in route_data and len(route_data["routes"]) > 0:
        route = route_data["routes"][0]
        
        # Extract turn-by-turn directions
        steps_list = []
        if "legs" in route:
            for leg in route["legs"]:
                for step in leg.get("steps", []):
                    maneuver = step.get("maneuver", {})
                    waypoint = maneuver.get("location")
                    if not waypoint and step.get("geometry", {}).get("coordinates"):
                        waypoint = step["geometry"]["coordinates"][-1]
                    steps_list.append({
                        "instruction": maneuver.get("instruction", ""),
                        "distance": step.get("distance"),
                        "duration": step.get("duration"),
                        "name": step.get("name", ""),
                        "waypoint": waypoint
                    })
        
        return {
            "summary": {
                "distance": route.get("distance"),  # meters
                "duration": route.get("duration"),  # seconds
                "mode": mode
            },
            "directions": steps_list,
            "geometry": route.get("geometry"),
            "waypoints": route_data.get("waypoints", [])
        }
    else:
        raise Exception("No route found")

@mcp.tool()
async def search_category(
    category: str,
    min_latitude: float,
    min_longitude: float,
    max_latitude: float,
    max_longitude: float,
    ctx: Context,
    subcategories: List[str] = None
) -> Dict[str, Any]:
    """
    Search for specific types of places within a defined geographic area.
    
    This tool allows targeted searches for places matching specific categories within
    a rectangular geographic region. It's particularly useful for filtering places by type
    (restaurants, schools, parks, etc.) within a neighborhood or city district. Results include
    complete location details and metadata about each matching place.
    
    Args:
        category: Main OSM category to search for (e.g., "amenity", "shop", "tourism", "building")
        min_latitude: Southern boundary of search area (decimal degrees)
        min_longitude: Western boundary of search area (decimal degrees)
        max_latitude: Northern boundary of search area (decimal degrees)
        max_longitude: Eastern boundary of search area (decimal degrees)
        subcategories: Optional list of specific subcategories to filter by (e.g., ["restaurant", "cafe"])
        
    Returns:
        Structured results including:
        - Query parameters
        - Count of matching places
        - List of matching places with coordinates, names, and metadata
    """
    osm_client = ctx.request_context.lifespan_context.osm_client
    
    bbox = (min_longitude, min_latitude, max_longitude, max_latitude)
    
    await ctx.info(f"Searching for {category} in bounding box")
    features = await osm_client.search_features_by_category(bbox, category, subcategories)
    
    # Process results
    results = []
    for feature in features:
        tags = feature.get("tags", {})
        
        # Get coordinates based on feature type
        coords = {}
        if feature.get("type") == "node":
            coords = {
                "latitude": feature.get("lat"),
                "longitude": feature.get("lon")
            }
        # For ways and relations, use center coordinates if available
        elif "center" in feature:
            coords = {
                "latitude": feature.get("center", {}).get("lat"),
                "longitude": feature.get("center", {}).get("lon")
            }
        
        # Only include features with valid coordinates
        if coords:
            results.append({
                "id": feature.get("id"),
                "type": feature.get("type"),
                "name": tags.get("name", "Unnamed"),
                "coordinates": coords,
                "category": category,
                "subcategory": tags.get(category),
                "tags": tags
            })
    
    return {
        "query": {
            "category": category,
            "subcategories": subcategories,
            "bbox": {
                "min_latitude": min_latitude,
                "min_longitude": min_longitude,
                "max_latitude": max_latitude,
                "max_longitude": max_longitude
            }
        },
        "results": results,
        "count": len(results)
    }

@mcp.tool()
async def suggest_meeting_point(
    locations: List[Dict[str, float]],
    ctx: Context,
    venue_type: str = "cafe"
) -> Dict[str, Any]:
    """
    Find the optimal meeting place for multiple people coming from different locations.
    
    This tool calculates a central meeting point based on the locations of multiple individuals,
    then recommends suitable venues near that central point. Ideal for planning social gatherings,
    business meetings, or any situation where multiple people need to converge from different
    starting points.
    
    Args:
        locations: List of dictionaries, each containing the latitude and longitude of a person's location
                  Example: [{"latitude": 37.7749, "longitude": -122.4194}, {"latitude": 37.3352, "longitude": -121.8811}]
        venue_type: Type of venue to suggest as a meeting point. Options include:
                   "cafe", "restaurant", "bar", "library", "park", etc.
        
    Returns:
        Meeting point recommendations including:
        - Calculated center point coordinates
        - List of suggested venues with names and details
        - Total number of matching venues in the area
    """
    osm_client = ctx.request_context.lifespan_context.osm_client
    
    if len(locations) < 2:
        raise ValueError("Need at least two locations to suggest a meeting point")
    
    # Calculate the center point (simple average)
    avg_lat = sum(loc.get("latitude", 0) for loc in locations) / len(locations)
    avg_lon = sum(loc.get("longitude", 0) for loc in locations) / len(locations)
    
    await ctx.info(f"Calculating center point for {len(locations)} locations: ({avg_lat}, {avg_lon})")
    
    # Search for venues around this center point
    venues = await osm_client.get_nearby_pois(
        avg_lat, avg_lon, 
        radius=500,  # Search within 500m of center
        categories=["amenity"]
    )
    
    # Filter venues by type
    matching_venues = []
    for venue in venues:
        tags = venue.get("tags", {})
        if tags.get("amenity") == venue_type:
            matching_venues.append({
                "id": venue.get("id"),
                "name": tags.get("name", "Unnamed Venue"),
                "latitude": venue.get("lat"),
                "longitude": venue.get("lon"),
                "tags": tags
            })
    
    # If no venues found, expand search
    if not matching_venues:
        await ctx.info(f"No {venue_type} found within 500m, expanding search to 1000m")
        venues = await osm_client.get_nearby_pois(
            avg_lat, avg_lon, 
            radius=1000,
            categories=["amenity"]
        )
        
        for venue in venues:
            tags = venue.get("tags", {})
            if tags.get("amenity") == venue_type:
                matching_venues.append({
                    "id": venue.get("id"),
                    "name": tags.get("name", "Unnamed Venue"),
                    "latitude": venue.get("lat"),
                    "longitude": venue.get("lon"),
                    "tags": tags
                })
    
    # Return the result
    return {
        "center_point": {
            "latitude": avg_lat,
            "longitude": avg_lon
        },
        "suggested_venues": matching_venues[:5],  # Top 5 venues
        "venue_type": venue_type,
        "total_options": len(matching_venues)
    }

@mcp.tool()
async def explore_area(
    latitude: float,
    longitude: float,
    ctx: Context,
    radius: float = 500
) -> Dict[str, Any]:
    """
    Generate a comprehensive profile of an area including all amenities and features.
    
    This powerful analysis tool creates a detailed overview of a neighborhood or area by
    identifying and categorizing all geographic features, amenities, and points of interest.
    Results are organized by category for easy analysis. Excellent for neighborhood research,
    area comparisons, and location-based decision making.
    
    Args:
        latitude: Center point latitude (decimal degrees)
        longitude: Center point longitude (decimal degrees)
        radius: Search radius in meters (defaults to 500m)
        
    Returns:
        In-depth area profile including:
        - Address and location context
        - Total feature count
        - Features organized by category and subcategory
        - Each feature includes name, coordinates, and detailed metadata
    """
    osm_client = ctx.request_context.lifespan_context.osm_client
    
    # Categories to search for
    categories = [
        "amenity", "shop", "tourism", "leisure", 
        "natural", "historic", "public_transport"
    ]
    
    results = {}
    for i, category in enumerate(categories):
        await ctx.report_progress(i, len(categories))
        await ctx.info(f"Exploring {category} features...")
        
        try:
            # Convert radius to bounding box
            lat_delta = radius / 111000
            lon_delta = radius / (111000 * math.cos(math.radians(latitude)))
            
            bbox = (
                longitude - lon_delta,
                latitude - lat_delta,
                longitude + lon_delta,
                latitude + lat_delta
            )
            
            features = await osm_client.search_features_by_category(bbox, category)
            
            # Group by subcategory
            subcategories = {}
            for feature in features:
                tags = feature.get("tags", {})
                subcategory = tags.get(category)
                
                if subcategory:
                    if subcategory not in subcategories:
                        subcategories[subcategory] = []
                    
                    # Get coordinates based on feature type
                    coords = {}
                    if feature.get("type") == "node":
                        coords = {
                            "latitude": feature.get("lat"),
                            "longitude": feature.get("lon")
                        }
                    elif "center" in feature:
                        coords = {
                            "latitude": feature.get("center", {}).get("lat"),
                            "longitude": feature.get("center", {}).get("lon")
                        }
                    
                    subcategories[subcategory].append({
                        "id": feature.get("id"),
                        "name": tags.get("name", "Unnamed"),
                        "coordinates": coords,
                        "type": feature.get("type"),
                        "tags": tags
                    })
            
            results[category] = subcategories
            
        except Exception as e:
            await ctx.warning(f"Error fetching {category} features: {str(e)}")
            results[category] = {}
    
    # Get address information for the center point
    try:
        address_info = await osm_client.reverse_geocode(latitude, longitude)
    except Exception:
        address_info = {"error": "Could not retrieve address information"}
    
    # Report completion
    await ctx.report_progress(len(categories), len(categories))
    
    # Count total features
    total_features = sum(
        len(places)
        for category_data in results.values()
        for places in category_data.values()
    )
    
    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius
        },
        "address": address_info,
        "categories": results,
        "total_features": total_features,
        "timestamp": datetime.now().isoformat()
    }

# Add resource endpoints for common location-based app needs
@mcp.resource("location://place/{query}")
async def get_place_resource(query: str) -> str:
    """
    Get information about a place by name.
    
    Args:
        query: Place name or address to look up
        
    Returns:
        JSON string with place information
    """
    async with aiohttp.ClientSession() as session:
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        async with session.get(
            nominatim_url,
            params={
                "q": query,
                "format": "json",
                "limit": 1
            },
            headers={"User-Agent": "LocationApp-MCP-Server/1.0"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return json.dumps(data)
            else:
                raise Exception(f"Failed to get place info for {query}: {response.status}")

@mcp.resource("location://map/{style}/{z}/{x}/{y}")
async def get_map_style(style: str, z: int, x: int, y: int) -> Tuple[bytes, str]:
    """
    Get a styled map tile at the specified coordinates.
    
    Args:
        style: Map style (standard, cycle, transport, etc.)
        z: Zoom level
        x: X coordinate
        y: Y coordinate
        
    Returns:
        Tuple of (tile image bytes, mime type)
    """
    # Map styles to their respective tile servers
    tile_servers = {
        "standard": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "cycle": "https://tile.thunderforest.com/cycle/{z}/{x}/{y}.png",
        "transport": "https://tile.thunderforest.com/transport/{z}/{x}/{y}.png",
        "landscape": "https://tile.thunderforest.com/landscape/{z}/{x}/{y}.png",
        "outdoor": "https://tile.thunderforest.com/outdoors/{z}/{x}/{y}.png"
    }
    
    
    if style not in tile_servers:
        style = "standard"
    
    tile_url = tile_servers[style].replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
    
    async with aiohttp.ClientSession() as session:
        async with session.get(tile_url) as response:
            if response.status == 200:
                tile_data = await response.read()
                return tile_data, "image/png"
            else:
                raise Exception(f"Failed to get {style} tile at {z}/{x}/{y}: {response.status}")


@mcp.tool()
async def find_schools_nearby(
    latitude: float,
    longitude: float,
    ctx: Context,
    radius: float = 2000,
    education_levels: List[str] = None
) -> Dict[str, Any]:
    """
    Locate educational institutions near a specific location, filtered by education level.
    
    This specialized search tool identifies schools, colleges, and other educational institutions
    within a specified distance from a location. Results can be filtered by education level
    (elementary, middle, high school, university, etc.). Essential for families evaluating
    neighborhoods or real estate purchases with education considerations.
    
    Args:
        latitude: Center point latitude (decimal degrees)
        longitude: Center point longitude (decimal degrees)
        radius: Search radius in meters (defaults to 2000m/2km)
        education_levels: Optional list of specific education levels to filter by
                         (e.g., ["elementary", "secondary", "university"])
        
    Returns:
        List of educational institutions with:
        - Name and type
        - Distance from search point
        - Education levels offered
        - Contact information if available
        - Other relevant metadata
    """
    # Convert radius to bounding box (approximate)
    lat_delta = radius / 111000
    lon_delta = radius / (111000 * math.cos(math.radians(latitude)))
    
    bbox = (
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta
    )
    
    # Build Overpass query for educational institutions
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Create query for amenity=school and other education-related tags
    education_filters = [
        'node["amenity"="school"]({{bbox}});',
        'way["amenity"="school"]({{bbox}});',
        'node["amenity"="university"]({{bbox}});',
        'way["amenity"="university"]({{bbox}});',
        'node["amenity"="kindergarten"]({{bbox}});',
        'way["amenity"="kindergarten"]({{bbox}});',
        'node["amenity"="college"]({{bbox}});',
        'way["amenity"="college"]({{bbox}});'
    ]
    
    query = f"""
    [out:json];
    (
        {" ".join(education_filters)}
    );
    out body;
    """
    
    query = query.replace("{bbox}", f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(overpass_url, data={"data": query}) as response:
            if response.status == 200:
                data = await response.json()
                schools = data.get("elements", [])
            else:
                raise Exception(f"Failed to find schools: {response.status}")
    
    # Process and filter results
    results = []
    for school in schools:
        tags = school.get("tags", {})
        school_type = tags.get("school", "")
        
        # Filter by education level if specified
        if education_levels and school_type and school_type not in education_levels:
            continue
        
        # Get coordinates based on feature type
        coords = {}
        if school.get("type") == "node":
            coords = {
                "latitude": school.get("lat"),
                "longitude": school.get("lon")
            }
        elif "center" in school:
            coords = {
                "latitude": school.get("center", {}).get("lat"),
                "longitude": school.get("center", {}).get("lon")
            }
        
        # Skip if no valid coordinates
        if not coords:
            continue
        
        # Calculate distance from search point
        # Using Haversine formula for quick distance calculation
        from math import radians, sin, cos, sqrt, asin
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            dLat = radians(lat2 - lat1)
            dLon = radians(lon2 - lon1)
            a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
            c = 2 * asin(sqrt(a))
            return R * c
        
        distance = haversine(latitude, longitude, coords["latitude"], coords["longitude"])
        
        results.append({
            "id": school.get("id"),
            "name": tags.get("name", "Unnamed School"),
            "amenity_type": tags.get("amenity", ""),
            "school_type": school_type,
            "education_level": tags.get("isced", ""),
            "coordinates": coords,
            "distance": round(distance, 1),
            "address": {
                "street": tags.get("addr:street", ""),
                "housenumber": tags.get("addr:housenumber", ""),
                "city": tags.get("addr:city", ""),
                "postcode": tags.get("addr:postcode", "")
            },
            "tags": tags
        })
    
    # Sort by distance
    results.sort(key=lambda x: x["distance"])
    
    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius,
            "education_levels": education_levels
        },
        "schools": results,
        "count": len(results)
    }

@mcp.tool()
async def analyze_commute(
    home_latitude: float,
    home_longitude: float,
    work_latitude: float,
    work_longitude: float,
    ctx: Context,
    modes: List[str] = ["car", "foot", "bike"],
    depart_at: str = None  # Time in HH:MM format, e.g. "08:30"
) -> Dict[str, Any]:
    """
    Perform a detailed commute analysis between home and work locations.
    
    This advanced tool analyzes commute options between two locations (typically home and work),
    comparing multiple transportation modes and providing detailed metrics for each option.
    Includes estimated travel times, distances, turn-by-turn directions, and other commute-relevant
    data. Essential for real estate decisions, lifestyle planning, and workplace relocation analysis.
    
    Args:
        home_latitude: Home location latitude (decimal degrees)
        home_longitude: Home location longitude (decimal degrees)
        work_latitude: Workplace location latitude (decimal degrees)
        work_longitude: Workplace location longitude (decimal degrees)
        modes: List of transportation modes to analyze (options: "car", "foot", "bike")
        depart_at: Optional departure time (format: "HH:MM") for time-sensitive routing
        
    Returns:
        Comprehensive commute analysis with:
        - Summary comparing all transportation modes
        - Detailed route information for each mode
        - Total distance and duration for each option
        - Turn-by-turn directions
    """
    osm_client = ctx.request_context.lifespan_context.osm_client
    
    # Get address information for both locations
    home_info = await osm_client.reverse_geocode(home_latitude, home_longitude)
    work_info = await osm_client.reverse_geocode(work_latitude, work_longitude)
    
    # Get commute information for each mode
    commute_options = []
    
    for mode in modes:
        await ctx.info(f"Calculating {mode} route for commute analysis")
        
        # Get route from OSRM
        try:
            route_data = await osm_client.get_route(
                home_latitude, home_longitude,
                work_latitude, work_longitude,
                mode
            )
            
            if "routes" in route_data and len(route_data["routes"]) > 0:
                route = route_data["routes"][0]
                
                # Extract directions
                steps = []
                if "legs" in route:
                    for leg in route["legs"]:
                        for step in leg.get("steps", []):
                            steps.append({
                                "instruction": step.get("maneuver", {}).get("instruction", ""),
                                "distance": step.get("distance"),
                                "duration": step.get("duration"),
                                "name": step.get("name", "")
                            })
                
                commute_options.append({
                    "mode": mode,
                    "distance_km": round(route.get("distance", 0) / 1000, 2),
                    "duration_minutes": round(route.get("duration", 0) / 60, 1),
                    "directions": steps
                })
        except Exception as e:
            await ctx.warning(f"Error getting {mode} route: {str(e)}")
            commute_options.append({
                "mode": mode,
                "error": str(e)
            })
    
    # Sort by duration (fastest first)
    commute_options.sort(key=lambda x: x.get("duration_minutes", float("inf")))
    
    return {
        "home": {
            "coordinates": {
                "latitude": home_latitude,
                "longitude": home_longitude
            },
            "address": home_info.get("display_name", "Unknown location")
        },
        "work": {
            "coordinates": {
                "latitude": work_latitude,
                "longitude": work_longitude
            },
            "address": work_info.get("display_name", "Unknown location")
        },
        "commute_options": commute_options,
        "fastest_option": commute_options[0]["mode"] if commute_options else None,
        "depart_at": depart_at
    }

@mcp.tool()
async def find_ev_charging_stations(
    latitude: float,
    longitude: float,
    ctx: Context,
    radius: float = 5000,
    connector_types: List[str] = None,
    min_power: float = None
) -> Dict[str, Any]:
    """
    Locate electric vehicle charging stations near a specific location.
    
    This specialized search tool identifies EV charging infrastructure within a specified
    distance from a location. Results can be filtered by connector type (Tesla, CCS, CHAdeMO, etc.)
    and minimum power delivery. Essential for EV owners planning trips or evaluating potential
    charging stops.
    
    Args:
        latitude: Center point latitude (decimal degrees)
        longitude: Center point longitude (decimal degrees)
        radius: Search radius in meters (defaults to 5000m/5km)
        connector_types: Optional list of specific connector types to filter by
                        (e.g., ["type2", "ccs", "tesla"])
        min_power: Minimum charging power in kW
        
    Returns:
        List of charging stations with:
        - Location name and operator
        - Available connector types
        - Charging speeds
        - Number of charging points
        - Access restrictions
        - Other relevant metadata
    """
    # Convert radius to bounding box
    lat_delta = radius / 111000
    lon_delta = radius / (111000 * math.cos(math.radians(latitude)))
    
    bbox = (
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta
    )
    
    # Build Overpass query for EV charging stations
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    query = """
    [out:json];
    (
        node["amenity"="charging_station"]({{bbox}});
        way["amenity"="charging_station"]({{bbox}});
    );
    out body;
    """
    
    query = query.replace("{bbox}", f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(overpass_url, data={"data": query}) as response:
            if response.status == 200:
                data = await response.json()
                stations = data.get("elements", [])
            else:
                raise Exception(f"Failed to find charging stations: {response.status}")
    
    # Process and filter results
    results = []
    for station in stations:
        tags = station.get("tags", {})
        
        # Get coordinates based on feature type
        coords = {}
        if station.get("type") == "node":
            coords = {
                "latitude": station.get("lat"),
                "longitude": station.get("lon")
            }
        elif "center" in station:
            coords = {
                "latitude": station.get("center", {}).get("lat"),
                "longitude": station.get("center", {}).get("lon")
            }
        
        # Skip if no valid coordinates
        if not coords:
            continue
        
        # Extract connector information
        connectors = []
        for key, value in tags.items():
            if key.startswith("socket:"):
                connector_type = key.split(":", 1)[1]
                connectors.append({
                    "type": connector_type,
                    "count": value if value.isdigit() else 1
                })
        
        # Filter by connector type if specified
        if connector_types:
            has_matching_connector = False
            for connector in connectors:
                if connector["type"] in connector_types:
                    has_matching_connector = True
                    break
            if not has_matching_connector:
                continue
        
        # Extract power information
        power = None
        if "maxpower" in tags:
            try:
                power = float(tags["maxpower"])
            except ValueError:
                pass
        
        # Filter by minimum power if specified
        if min_power is not None and (power is None or power < min_power):
            continue
        
        # Calculate distance from search point
        from math import radians, sin, cos, sqrt, asin
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            dLat = radians(lat2 - lat1)
            dLon = radians(lon2 - lon1)
            a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
            c = 2 * asin(sqrt(a))
            return R * c
        
        distance = haversine(latitude, longitude, coords["latitude"], coords["longitude"])
        
        results.append({
            "id": station.get("id"),
            "name": tags.get("name", "Unnamed Charging Station"),
            "operator": tags.get("operator", "Unknown"),
            "coordinates": coords,
            "distance": round(distance, 1),
            "connectors": connectors,
            "capacity": tags.get("capacity", "Unknown"),
            "power": power,
            "fee": tags.get("fee", "Unknown"),
            "access": tags.get("access", "public"),
            "opening_hours": tags.get("opening_hours", "Unknown"),
            "address": {
                "street": tags.get("addr:street", ""),
                "housenumber": tags.get("addr:housenumber", ""),
                "city": tags.get("addr:city", ""),
                "postcode": tags.get("addr:postcode", "")
            },
            "tags": tags
        })
    
    # Sort by distance
    results.sort(key=lambda x: x["distance"])
    
    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius,
            "connector_types": connector_types,
            "min_power": min_power
        },
        "stations": results,
        "count": len(results)
    }

@mcp.tool()
async def analyze_neighborhood(
    latitude: float,
    longitude: float,
    ctx: Context,
    radius: float = 1000
) -> Dict[str, Any]:
    """
    Generate a comprehensive neighborhood analysis focused on livability factors.
    
    This advanced analysis tool evaluates a neighborhood based on multiple livability factors,
    including amenities, transportation options, green spaces, and services. Results include
    counts and proximity scores for various categories, helping to assess the overall quality
    and convenience of a residential area. Invaluable for real estate decisions, relocation
    planning, and neighborhood comparisons.
    
    Args:
        latitude: Center point latitude (decimal degrees)
        longitude: Center point longitude (decimal degrees)
        radius: Analysis radius in meters (defaults to 1000m/1km)
        
    Returns:
        Comprehensive neighborhood profile including:
        - Overall neighborhood score
        - Walkability assessment
        - Public transportation access
        - Nearby amenities (shops, restaurants, services)
        - Green spaces and recreation
        - Education and healthcare facilities
        - Detailed counts and distance metrics for each category
    """
    osm_client = ctx.request_context.lifespan_context.osm_client
    
    # Get address information for the center point
    address_info = await osm_client.reverse_geocode(latitude, longitude)
    
    # Categories to analyze for neighborhood quality
    categories = [
        # Essential services
        {"name": "groceries", "tags": ["shop=supermarket", "shop=convenience", "shop=grocery"]},
        {"name": "restaurants", "tags": ["amenity=restaurant", "amenity=cafe", "amenity=fast_food"]},
        {"name": "healthcare", "tags": ["amenity=hospital", "amenity=doctors", "amenity=pharmacy"]},
        {"name": "education", "tags": ["amenity=school", "amenity=kindergarten", "amenity=university"]},
        
        # Transportation
        {"name": "public_transport", "tags": ["public_transport=stop_position", "railway=station", "amenity=bus_station"]},
        
        # Recreation
        {"name": "parks", "tags": ["leisure=park", "leisure=garden", "leisure=playground"]},
        {"name": "sports", "tags": ["leisure=sports_centre", "leisure=fitness_centre", "leisure=swimming_pool"]},
        
        # Culture and entertainment
        {"name": "entertainment", "tags": ["amenity=theatre", "amenity=cinema", "amenity=arts_centre"]},
        
        # Other amenities
        {"name": "shopping", "tags": ["shop=mall", "shop=department_store", "shop=clothes"]},
        {"name": "services", "tags": ["amenity=bank", "amenity=post_office", "amenity=atm"]}
    ]
    
    # Build overpass queries and collect results
    results = {}
    scores = {}
    
    for i, category in enumerate(categories):
        await ctx.report_progress(i, len(categories))
        await ctx.info(f"Analyzing {category['name']} in neighborhood...")
        
        # Convert radius to bounding box
        lat_delta = radius / 111000
        lon_delta = radius / (111000 * math.cos(math.radians(latitude)))
        
        bbox = (
            longitude - lon_delta,
            latitude - lat_delta,
            longitude + lon_delta,
            latitude + lat_delta
        )
        
        # Build Overpass query
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Create query for category tags
        tag_filters = []
        for tag in category["tags"]:
            key, value = tag.split("=")
            tag_filters.append(f'node["{key}"="{value}"]({{bbox}});')
            tag_filters.append(f'way["{key}"="{value}"]({{bbox}});')
        
        query = f"""
        [out:json];
        (
            {" ".join(tag_filters)}
        );
        out body;
        """
        
        query = query.replace("{bbox}", f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(overpass_url, data={"data": query}) as response:
                    if response.status == 200:
                        data = await response.json()
                        features = data.get("elements", [])
                    else:
                        await ctx.warning(f"Failed to analyze {category['name']}: {response.status}")
                        features = []
                        
            # Process and calculate metrics
            feature_list = []
            distances = []
            
            for feature in features:
                tags = feature.get("tags", {})
                
                # Get coordinates based on feature type
                coords = {}
                if feature.get("type") == "node":
                    coords = {
                        "latitude": feature.get("lat"),
                        "longitude": feature.get("lon")
                    }
                elif "center" in feature:
                    coords = {
                        "latitude": feature.get("center", {}).get("lat"),
                        "longitude": feature.get("center", {}).get("lon")
                    }
                
                # Skip if no valid coordinates
                if not coords:
                    continue
                
                # Calculate distance from center point
                from math import radians, sin, cos, sqrt, asin
                
                def haversine(lat1, lon1, lat2, lon2):
                    R = 6371000  # Earth radius in meters
                    dLat = radians(lat2 - lat1)
                    dLon = radians(lon2 - lon1)
                    a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
                    c = 2 * asin(sqrt(a))
                    return R * c
                
                distance = haversine(latitude, longitude, coords["latitude"], coords["longitude"])
                distances.append(distance)
                
                feature_list.append({
                    "id": feature.get("id"),
                    "name": tags.get("name", "Unnamed"),
                    "type": feature.get("type"),
                    "coordinates": coords,
                    "distance": round(distance, 1),
                    "tags": tags
                })
            
            # Sort by distance
            feature_list.sort(key=lambda x: x["distance"])
            
            # Calculate metrics
            count = len(feature_list)
            avg_distance = sum(distances) / count if count > 0 else None
            min_distance = min(distances) if count > 0 else None
            
            # Score this category (0-10): count contributes up to 5 points,
            # and nearest-feature proximity contributes up to 5 points.
            if count == 0:
                category_score = 0
            else:
                count_score = min(count / 5, 1) * 5  # Up to 5 points for count
                proximity_score = 5 - min(min_distance / radius, 1) * 5  # Up to 5 points for proximity
                category_score = count_score + proximity_score
            
            # Store results
            results[category["name"]] = {
                "count": count,
                "features": feature_list[:10],  # Limit to top 10
                "metrics": {
                    "total_count": count,
                    "avg_distance": round(avg_distance, 1) if avg_distance else None,
                    "min_distance": round(min_distance, 1) if min_distance else None
                }
            }
            
            scores[category["name"]] = category_score
            
        except Exception as e:
            await ctx.warning(f"Error analyzing {category['name']}: {str(e)}")
            results[category["name"]] = {"error": str(e)}
            scores[category["name"]] = 0
    
    # Calculate overall neighborhood score
    if scores:
        overall_score = sum(scores.values()) / len(scores)
    else:
        overall_score = 0
    
    # Walkability score favors variety plus raw amenity count within 500m.
    walkable_amenities = 0
    walkable_categories = 0
    
    for category_name, category_data in results.items():
        if "metrics" in category_data:
            # Count amenities within walking distance
            walking_count = sum(1 for feature in category_data.get("features", []) 
                               if feature.get("distance", float("inf")) <= 500)
            
            if walking_count > 0:
                walkable_amenities += walking_count
                walkable_categories += 1
    
    walkability_score = min(walkable_amenities + walkable_categories, 10)
    
    # Report completion
    await ctx.report_progress(len(categories), len(categories))
    
    return {
        "location": {
            "coordinates": {
                "latitude": latitude,
                "longitude": longitude
            },
            "address": address_info.get("display_name", "Unknown location")
        },
        "scores": {
            "overall": round(overall_score, 1),
            "walkability": walkability_score,
            "categories": {k: round(v, 1) for k, v in scores.items()}
        },
        "categories": results,
        "analysis_radius": radius,
        "timestamp": datetime.now().isoformat()
    }

@mcp.tool()
async def find_parking_facilities(
    latitude: float,
    longitude: float,
    ctx: Context,
    radius: float = 1000,
    parking_type: str = None  # e.g., "surface", "underground", "multi-storey"
) -> Dict[str, Any]:
    """
    Locate parking facilities near a specific location.
    
    This tool finds parking options (lots, garages, street parking) near a specified location.
    Results can be filtered by parking type and include capacity information where available.
    Useful for trip planning, city navigation, and evaluating parking availability in urban areas.
    
    Args:
        latitude: Center point latitude (decimal degrees)
        longitude: Center point longitude (decimal degrees)
        radius: Search radius in meters (defaults to 1000m/1km)
        parking_type: Optional filter for specific types of parking facilities
                     ("surface", "underground", "multi-storey", etc.)
        
    Returns:
        List of parking facilities with:
        - Name and type
        - Capacity information if available
        - Fee structure if available
        - Access restrictions
        - Distance from search point
    """
    # Convert radius to bounding box
    lat_delta = radius / 111000
    lon_delta = radius / (111000 * math.cos(math.radians(latitude)))
    
    bbox = (
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta
    )
    
    # Build Overpass query for parking facilities
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    query = """
    [out:json];
    (
        node["amenity"="parking"]({{bbox}});
        way["amenity"="parking"]({{bbox}});
        relation["amenity"="parking"]({{bbox}});
    );
    out body;
    """
    
    query = query.replace("{bbox}", f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(overpass_url, data={"data": query}) as response:
            if response.status == 200:
                data = await response.json()
                parking_facilities = data.get("elements", [])
            else:
                raise Exception(f"Failed to find parking facilities: {response.status}")
    
    # Process and filter results
    results = []
    for facility in parking_facilities:
        tags = facility.get("tags", {})
        
        # Filter by parking type if specified
        if parking_type and tags.get("parking", "") != parking_type:
            continue
        
        # Get coordinates based on feature type
        coords = {}
        if facility.get("type") == "node":
            coords = {
                "latitude": facility.get("lat"),
                "longitude": facility.get("lon")
            }
        elif "center" in facility:
            coords = {
                "latitude": facility.get("center", {}).get("lat"),
                "longitude": facility.get("center", {}).get("lon")
            }
        
        # Skip if no valid coordinates
        if not coords:
            continue
        
        # Calculate distance from search point
        from math import radians, sin, cos, sqrt, asin
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            dLat = radians(lat2 - lat1)
            dLon = radians(lon2 - lon1)
            a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
            c = 2 * asin(sqrt(a))
            return R * c
        
        distance = haversine(latitude, longitude, coords["latitude"], coords["longitude"])
        
        results.append({
            "id": facility.get("id"),
            "name": tags.get("name", "Unnamed Parking"),
            "type": tags.get("parking", "surface"),
            "coordinates": coords,
            "distance": round(distance, 1),
            "capacity": tags.get("capacity", "Unknown"),
            "fee": tags.get("fee", "Unknown"),
            "access": tags.get("access", "public"),
            "opening_hours": tags.get("opening_hours", "Unknown"),
            "levels": tags.get("levels", "1"),
            "address": {
                "street": tags.get("addr:street", ""),
                "housenumber": tags.get("addr:housenumber", ""),
                "city": tags.get("addr:city", ""),
                "postcode": tags.get("addr:postcode", "")
            },
            "tags": tags
        })
    
    # Sort by distance
    results.sort(key=lambda x: x["distance"])
    
    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius,
            "parking_type": parking_type
        },
        "parking_facilities": results,
        "count": len(results)
    }


@mcp.tool()
async def recommend_destination_and_waypoints(
    user_instruction: str,
    current_latitude: float,
    current_longitude: float,
    ctx: Context,
    max_candidates: int = 5,
    search_radius: float = 3000,
    mode: Optional[str] = None,
    resolution_scope: str = "global_destination",
    anchor_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parse free-form human instructions, recommend candidate destinations, and return route key waypoints.

    This first version combines a lightweight prompt-engineering parser (LLM when configured,
    heuristic fallback otherwise) with map search and routing to produce practical destination
    recommendations and waypoint sequences for navigation.
    """
    app_context = ctx.request_context.lifespan_context
    engine = DestinationRecommendationEngine(
        map_client=app_context.osm_client,
        provider=app_context.provider,
    )
    return await engine.recommend(
        user_instruction=user_instruction,
        current_latitude=current_latitude,
        current_longitude=current_longitude,
        max_candidates=max_candidates,
        search_radius=search_radius,
        mode=mode,
        resolution_scope=resolution_scope,
        anchor_location=anchor_location,
        ctx=ctx,
    )


@mcp.tool()
async def decompose_itinerary_goals(
    user_instruction: str,
    ctx: Context,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decompose a free-form instruction into an ordered list of itinerary goals.

    This tool does not recommend concrete POIs yet. It extracts the ordered goal
    skeleton that can later be realized as via points plus a final destination.
    """
    decomposer = GoalDecomposer()
    decomposition = await decomposer.parse(user_instruction)

    override_mode = str(mode or "").strip().lower()
    if override_mode in {"car", "foot", "bike"}:
        decomposition["global_mode"] = override_mode
        for goal in decomposition.get("goals", []) or []:
            if isinstance(goal, dict):
                goal["mode"] = override_mode

    return {
        "input": {
            "user_instruction": user_instruction,
            "mode_override": override_mode or None,
        },
        "decomposition": decomposition,
        "timestamp": datetime.now().isoformat(),
    }


@mcp.tool()
async def recommend_itinerary_and_waypoints(
    user_instruction: str,
    current_latitude: float,
    current_longitude: float,
    ctx: Context,
    max_candidates: int = 5,
    search_radius: float = 3000,
    mode: Optional[str] = None,
    max_goals: int = 4,
) -> Dict[str, Any]:
    """
    Decompose a multi-goal instruction, recommend one stop for each ordered goal,
    and build a combined multi-leg route with via points.

    This is the first itinerary-oriented extension on top of the single-destination
    recommendation stack. It keeps each sub-goal provider-retrievable and realizes
    earlier goals as via points plus one final destination.
    """
    app_context = ctx.request_context.lifespan_context
    engine = DestinationRecommendationEngine(
        map_client=app_context.osm_client,
        provider=app_context.provider,
    )
    decomposer = GoalDecomposer()
    candidate_judge = ItineraryCandidateJudge()
    decomposition = await decomposer.parse(user_instruction)

    goals = [
        goal
        for goal in (decomposition.get("goals", []) or [])
        if isinstance(goal, dict) and str(goal.get("request_text") or "").strip()
    ][: max(1, int(max_goals))]

    if not goals:
        single = await engine.recommend(
            user_instruction=user_instruction,
            current_latitude=current_latitude,
            current_longitude=current_longitude,
            max_candidates=max_candidates,
            search_radius=search_radius,
            mode=mode,
            ctx=ctx,
        )
        return {
            "provider": app_context.provider,
            "input": {
                "user_instruction": user_instruction,
                "current_latitude": current_latitude,
                "current_longitude": current_longitude,
            },
            "decomposition": decomposition,
            "goal_recommendations": [],
            "via_points": [],
            "final_destination": single.get("chosen_destination"),
            "route_plan": {
                "route_summary": single.get("route_summary", {}),
                "route_polyline": single.get("route_polyline", []),
                "coarse_waypoints": single.get("key_waypoints", []),
                "legs": [],
            },
            "status": "single_goal_fallback",
        }

    override_mode = str(mode or "").strip().lower()
    global_mode = override_mode if override_mode in {"car", "foot", "bike"} else str(
        decomposition.get("global_mode") or "car"
    ).strip().lower()
    if global_mode not in {"car", "foot", "bike"}:
        global_mode = "car"

    goal_results: List[Dict[str, Any]] = []
    selected_stops: List[Dict[str, Any]] = []
    sequential_lat = float(current_latitude)
    sequential_lon = float(current_longitude)
    used_destination_ids: set[str] = set()

    for goal in goals:
        goal_mode = override_mode if override_mode in {"car", "foot", "bike"} else str(
            goal.get("mode") or global_mode
        ).strip().lower()
        if goal_mode not in {"car", "foot", "bike"}:
            goal_mode = global_mode
        goal_radius = goal.get("radius_m")
        if not isinstance(goal_radius, (int, float)) or float(goal_radius) <= 0:
            goal_radius = search_radius

        goal_instruction = str(goal.get("request_text") or "").strip()
        search_keywords = str(goal.get("search_keywords") or "").strip()
        goal_type = str(goal.get("goal_type") or "").strip()
        if search_keywords:
            goal_instruction = (
                f"{goal_instruction}. Goal type: {goal_type}. "
                f"Use these provider-facing search keywords: {search_keywords}."
            )
        recommendation = await engine.recommend(
            user_instruction=goal_instruction,
            current_latitude=sequential_lat,
            current_longitude=sequential_lon,
            max_candidates=max_candidates,
            search_radius=float(goal_radius),
            mode=goal_mode,
            ctx=ctx,
        )

        top_candidates = [
            candidate
            for candidate in (recommendation.get("recommendations") or [])
            if isinstance(candidate, dict)
            and (
                not _destination_id(candidate)
                or _destination_id(candidate) not in used_destination_ids
            )
        ][: max(1, int(max_candidates))]
        judge_decision = await candidate_judge.select_candidate(
            goal=goal,
            candidates=top_candidates,
            previous_stops=selected_stops,
        )
        chosen = None
        selection_reason = str(judge_decision.get("judge_source") or "llm_judge")
        selected_id = str(judge_decision.get("selected_candidate_id") or "").strip()
        if selected_id:
            for candidate in top_candidates:
                if _destination_id(candidate) == selected_id:
                    chosen = candidate
                    break
        if not chosen:
            fallback_chosen, fallback_reason = _select_itinerary_destination(
                recommendation=recommendation,
                used_destination_ids=used_destination_ids,
            )
            chosen = fallback_chosen
            selection_reason = f"{selection_reason}_fallback_{fallback_reason}"
        if chosen:
            recommendation["chosen_destination"] = chosen
        goal_entry = {
            "goal": goal,
            "recommendation": recommendation,
            "status": "selected" if chosen else "failed",
            "selection_reason": selection_reason,
            "judge_decision": judge_decision,
            "judge_fallback_used": not bool(selected_id and chosen),
        }
        goal_results.append(goal_entry)

        if not chosen:
            break

        destination_id = _destination_id(chosen)
        if destination_id:
            used_destination_ids.add(destination_id)
        selected_stops.append(
            {
                "name": str(chosen.get("name") or ""),
                "location": str(chosen.get("location") or ""),
                "goal_type": str(goal.get("goal_type") or ""),
                "goal_index": goal.get("goal_index"),
            }
        )

        coords = (chosen.get("coordinates") or {})
        next_lat = coords.get("latitude")
        next_lon = coords.get("longitude")
        if isinstance(next_lat, (int, float)) and isinstance(next_lon, (int, float)):
            sequential_lat = float(next_lat)
            sequential_lon = float(next_lon)

    successful_goals = [item for item in goal_results if item.get("status") == "selected"]
    chosen_stops = [item.get("recommendation", {}).get("chosen_destination") for item in successful_goals]
    chosen_stops = [item for item in chosen_stops if isinstance(item, dict)]

    via_points = []
    final_destination = chosen_stops[-1] if chosen_stops else None
    if len(chosen_stops) >= 2:
        for index, stop in enumerate(chosen_stops[:-1]):
            goal = successful_goals[index].get("goal") or {}
            via_points.append(
                {
                    "location": str(stop.get("location") or "").strip(),
                    "name": str(stop.get("name") or goal.get("goal_type") or f"Stop {index + 1}").strip(),
                    "goal_index": goal.get("goal_index"),
                    "goal_type": goal.get("goal_type"),
                }
            )

    route_plan = None
    if final_destination and str(final_destination.get("location") or "").strip():
        route_plan = await _compute_combined_route_plan(
            app_context=app_context,
            origin=format_location(current_longitude, current_latitude),
            destination=str(final_destination["location"]),
            via_points=via_points,
            mode=global_mode,
        )

    status = "failed"
    if goal_results:
        if len(successful_goals) == len(goals):
            status = "completed"
        elif successful_goals:
            status = "partial"

    return {
        "provider": app_context.provider,
        "input": {
            "user_instruction": user_instruction,
            "current_latitude": current_latitude,
            "current_longitude": current_longitude,
            "mode_override": override_mode or None,
        },
        "decomposition": decomposition,
        "goal_recommendations": goal_results,
        "fulfilled_goal_count": len(successful_goals),
        "requested_goal_count": len(goals),
        "via_points": via_points,
        "final_destination": final_destination,
        "route_plan": route_plan,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }

if __name__ == "__main__":
    mcp.run()
