"""MCP endpoints for route visualization."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import os
from urllib.parse import quote
from mcp.server.fastmcp import Context
from mcp_server.map_providers import build_amap_staticmap_url, encode_polyline_text, parse_location_text, sample_route_coordinates
from mcp_server.route_context import build_local_bev_context, build_route_context, build_route_state, local_bev_context_to_dict, route_context_to_dict, route_state_to_dict
from mcp_server.services.route_session_service import _compute_combined_route_plan
from mcp_server.mcp_api.presentation import _build_amap_staticmap_overlay_url, _build_local_bev_svg, _normalize_route_coords, _single_char_marker_label
from mcp_server.mcp_api.app import mcp

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
