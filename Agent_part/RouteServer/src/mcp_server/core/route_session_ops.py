"""Route session core operations shared by HTTP, MCP, and Agent tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

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


async def create_route_session(
    *,
    app_context: Any,
    origin: str,
    destination: str,
    via_points: Any = None,
    mode: str = "foot",
    session_name: str | None = None,
) -> dict[str, Any]:
    """Create a persisted route session and return the existing API payload."""

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


async def get_route_session_snapshot(
    *,
    app_context: Any,
    session_id: str,
    force_replan: bool = False,
) -> dict[str, Any]:
    """Fetch a route session and its cached or refreshed route plan."""

    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    route_plan, cache_hit = await _route_plan_for_session(
        app_context=app_context,
        session=session,
        force_replan=force_replan,
    )
    return {
        "session": session,
        "route_plan": route_plan,
        "route_plan_cache_hit": cache_hit,
        "session_store": _route_session_store_info(),
    }


def list_route_sessions(*, limit: int = 20, include_route_plan_cache: bool = False) -> dict[str, Any]:
    """List persisted route sessions using the existing response shape."""

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


async def add_via_point(
    *,
    app_context: Any,
    session_id: str,
    new_via_point: dict[str, Any],
    insert_before_index: int | None = None,
) -> dict[str, Any]:
    """Insert or append a via point and replan with the existing semantics."""

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
            index
            for index, via in enumerate(via_points)
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


async def remove_via_point(
    *,
    app_context: Any,
    session_id: str,
    via_point_index: int | None = None,
    via_point_name: str | None = None,
) -> dict[str, Any]:
    """Remove a via point by index or name and replan."""

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


async def replace_destination(
    *,
    app_context: Any,
    session_id: str,
    destination: str,
) -> dict[str, Any]:
    """Replace the final destination and replan."""

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


async def replan_route_session_from_origin(
    *,
    app_context: Any,
    session_id: str,
    origin: str,
) -> dict[str, Any]:
    """Update the current route-session origin and replan."""

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
