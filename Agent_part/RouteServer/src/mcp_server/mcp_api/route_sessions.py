"""MCP endpoints for route sessions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import Context
from mcp_server.core.route_session_ops import add_via_point as core_add_via_point, create_route_session as core_create_route_session, get_route_session_snapshot as core_get_route_session_snapshot, list_route_sessions as core_list_route_sessions, remove_via_point as core_remove_via_point, replace_destination as core_replace_destination, replan_route_session_from_origin as core_replan_route_session_from_origin
from mcp_server.services.route_session_service import _compute_combined_route_plan, _normalize_via_points
from mcp_server.mcp_api.app import mcp

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
    return await core_create_route_session(
        app_context=app_context,
        origin=origin,
        destination=destination,
        via_points=via_points,
        mode=mode,
        session_name=session_name,
    )


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
    return await core_get_route_session_snapshot(
        app_context=app_context,
        session_id=session_id,
        force_replan=bool(force_replan),
    )


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
    return core_list_route_sessions(
        limit=limit,
        include_route_plan_cache=include_route_plan_cache,
    )


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
    return await core_add_via_point(
        app_context=app_context,
        session_id=session_id,
        new_via_point=new_via_point,
        insert_before_index=insert_before_index,
    )


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
    return await core_remove_via_point(
        app_context=app_context,
        session_id=session_id,
        via_point_index=via_point_index,
        via_point_name=via_point_name,
    )


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
    return await core_replace_destination(
        app_context=app_context,
        session_id=session_id,
        destination=destination,
    )


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
    return await core_replan_route_session_from_origin(
        app_context=app_context,
        session_id=session_id,
        origin=origin,
    )
