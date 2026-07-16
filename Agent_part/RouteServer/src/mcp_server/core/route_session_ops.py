"""Route session core operations shared by HTTP, MCP, and Agent tools."""

from __future__ import annotations

import copy
import threading
from datetime import datetime
from typing import Any
from uuid import uuid4

from mcp_server.services.route_session_service import (
    ROUTE_SESSIONS,
    _cache_route_plan_on_session,
    _compute_combined_route_plan,
    _compute_incremental_route_plan_with_inserted_via,
    _get_persisted_route_session,
    _get_route_mutation_result,
    _load_route_sessions,
    _locations_equivalent,
    _normalize_via_points,
    _route_plan_for_session,
    _route_session_store_info,
    _save_persisted_route_session,
    _save_route_mutation_result,
)


_ROUTE_MUTATION_COMMIT_LOCK = threading.RLock()


class RouteRevisionConflict(ValueError):
    """Raised when a route mutation attempts to commit against a stale revision."""


def _route_revision(session: dict[str, Any]) -> int:
    """Return the canonical integer route revision for one session."""

    return int(session.get("route_revision") or 0)


def _validate_expected_revision(session: dict[str, Any], expected_state_revision: int | None) -> int:
    """Validate an optional caller revision and return the current revision."""

    current = _route_revision(session)
    if expected_state_revision is not None and int(expected_state_revision) != current:
        raise RouteRevisionConflict(
            f"route revision conflict: expected={expected_state_revision}, current={current}"
        )
    return current


def _commit_route_mutation(
    *,
    session_id: str,
    base_revision: int,
    candidate: dict[str, Any],
    route_plan: dict[str, Any],
    operation: str,
    idempotency_key: str | None,
    source_event_id: str | None,
    extra_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Commit a precomputed route candidate with compare-and-swap semantics.

    Args:
        session_id: Route session id.
        base_revision: Revision used to compute the candidate.
        candidate: Detached candidate session.
        route_plan: Provider result computed outside the commit lock.
        operation: Stable route mutation operation.
        idempotency_key: Optional exact-replay key.
        source_event_id: External event grounding this mutation.
        extra_result: Operation-specific response fields.

    Returns:
        Canonical mutation result with revision and state delta.

    Raises:
        KeyError: If the session disappeared before commit.
        RouteRevisionConflict: If another mutation committed first.
    """

    with _ROUTE_MUTATION_COMMIT_LOCK:
        cached = _get_route_mutation_result(session_id, idempotency_key)
        if cached is not None:
            cached["idempotent_replay"] = True
            return cached
        current = _get_persisted_route_session(session_id)
        if current is None:
            raise KeyError(f"Unknown route session: {session_id}")
        current_revision = _route_revision(current)
        if current_revision != base_revision:
            raise RouteRevisionConflict(
                f"route revision conflict: expected={base_revision}, current={current_revision}"
            )
        committed = copy.deepcopy(candidate)
        committed["route_revision"] = base_revision + 1
        committed["updated_at"] = datetime.now().isoformat()
        _cache_route_plan_on_session(committed, route_plan)
        _save_persisted_route_session(committed)
        result = {
            "session": committed,
            "route_plan": route_plan,
            "route_plan_cache_hit": False,
            "base_state_revision": base_revision,
            "state_revision": base_revision + 1,
            "idempotency_key": idempotency_key,
            "idempotent_replay": False,
            "state_delta": {
                "operation": operation,
                "base_state_revision": base_revision,
                "state_revision": base_revision + 1,
                "source_event_id": source_event_id,
            },
            "session_store": _route_session_store_info(),
        }
        result.update(extra_result or {})
        _save_route_mutation_result(session_id, idempotency_key, result)
        return result


async def create_route_session(
    *,
    app_context: Any,
    origin: str,
    destination: str,
    via_points: Any = None,
    mode: str = "foot",
    session_name: str | None = None,
    idempotency_key: str | None = None,
    source_event_id: str | None = None,
) -> dict[str, Any]:
    """Create a persisted route session with optional exact retry replay.

    Args:
        app_context: Map-provider application context.
        origin: Route origin.
        destination: Route destination.
        via_points: Optional initial via points.
        mode: Travel mode.
        session_name: Optional display name.
        idempotency_key: Optional exact-replay key.
        source_event_id: External event grounding this creation.

    Returns:
        Created session, route plan and revision metadata.
    """

    cached = _get_route_mutation_result("__create__", idempotency_key)
    if cached is not None:
        cached["idempotent_replay"] = True
        return cached

    normalized_via_points = _normalize_via_points(via_points)
    route_plan = await _compute_combined_route_plan(
        app_context=app_context,
        origin=origin,
        destination=destination,
        via_points=normalized_via_points,
        mode=mode,
    )
    # provider 计算不持锁；提交前再查幂等键，防止并发重试创建两个 session。
    with _ROUTE_MUTATION_COMMIT_LOCK:
        cached = _get_route_mutation_result("__create__", idempotency_key)
        if cached is not None:
            cached["idempotent_replay"] = True
            return cached
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
            "route_revision": 0,
        }
        _cache_route_plan_on_session(session_payload, route_plan)
        _save_persisted_route_session(session_payload)
        result = {
            "session": session_payload,
            "route_plan": route_plan,
            "route_plan_cache_hit": False,
            "base_state_revision": None,
            "state_revision": 0,
            "idempotency_key": idempotency_key,
            "idempotent_replay": False,
            "state_delta": {
                "operation": "create_route_session",
                "state_revision": 0,
                "source_event_id": source_event_id,
            },
            "session_store": _route_session_store_info(),
        }
        _save_route_mutation_result("__create__", idempotency_key, result)
        return result


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
    expected_state_revision: int | None = None,
    idempotency_key: str | None = None,
    source_event_id: str | None = None,
) -> dict[str, Any]:
    """Insert a via point using provider-outside-lock and CAS commit semantics.

    Args:
        app_context: Map-provider application context.
        session_id: Route session id.
        new_via_point: Via point to insert.
        insert_before_index: Optional insertion index.
        expected_state_revision: Optional optimistic concurrency guard.
        idempotency_key: Optional exact-replay key.
        source_event_id: External event grounding the mutation.

    Returns:
        Committed route plan and revision metadata.

    Raises:
        KeyError: If the session does not exist.
        RouteRevisionConflict: If the expected revision is stale.
        ValueError: If the via point is invalid.
    """

    cached = _get_route_mutation_result(session_id, idempotency_key)
    if cached is not None:
        cached["idempotent_replay"] = True
        return cached

    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    base = copy.deepcopy(session)
    base_revision = _validate_expected_revision(base, expected_state_revision)
    normalized_new = _normalize_via_points([new_via_point])
    if not normalized_new:
        raise ValueError("new_via_point must include a valid location")

    old_via_points = _normalize_via_points(base.get("via_points"))
    via_points = [dict(item) for item in old_via_points]
    item = normalized_new[0]
    same_as_destination = _locations_equivalent(item.get("location"), base.get("destination"))
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
            session=base,
            force_replan=False,
        )
        result = {
            "session": base,
            "route_plan": route_plan,
            "route_plan_cache_hit": cache_hit,
            "session_changed": False,
            "change_reason": "matches_destination" if same_as_destination else f"duplicate_via_point:{duplicate_index}",
            "route_plan_update_mode": "cache_reuse_noop",
            "base_state_revision": base_revision,
            "state_revision": base_revision,
            "idempotency_key": idempotency_key,
            "idempotent_replay": False,
            "state_delta": {},
            "session_store": _route_session_store_info(),
        }
        _save_route_mutation_result(session_id, idempotency_key, result)
        return result

    inserted_index = len(via_points)
    if insert_before_index is None:
        via_points.append(item)
    else:
        inserted_index = max(0, min(int(insert_before_index), len(via_points)))
        via_points.insert(inserted_index, item)
    for index, via in enumerate(via_points):
        via["index"] = index
    candidate = copy.deepcopy(base)
    candidate["via_points"] = via_points

    cached_route_plan = base.get("route_plan_cache")
    route_plan = None
    route_plan_update_mode = "full_replan"
    if isinstance(cached_route_plan, dict) and cached_route_plan:
        route_plan = await _compute_incremental_route_plan_with_inserted_via(
            app_context=app_context,
            origin=base["origin"],
            destination=base["destination"],
            mode=base["mode"],
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
            origin=base["origin"],
            destination=base["destination"],
            via_points=via_points,
            mode=base["mode"],
        )
    return _commit_route_mutation(
        session_id=session_id,
        base_revision=base_revision,
        candidate=candidate,
        route_plan=route_plan,
        operation="add_via_and_replan",
        idempotency_key=idempotency_key,
        source_event_id=source_event_id,
        extra_result={
            "session_changed": True,
            "change_reason": "via_point_added",
            "route_plan_update_mode": route_plan_update_mode,
        },
    )


async def remove_via_point(
    *,
    app_context: Any,
    session_id: str,
    via_point_index: int | None = None,
    via_point_name: str | None = None,
    expected_state_revision: int | None = None,
    idempotency_key: str | None = None,
    source_event_id: str | None = None,
) -> dict[str, Any]:
    """Remove a via point and commit only against the observed revision.

    Args:
        app_context: Map-provider application context.
        session_id: Route session id.
        via_point_index: Optional via index.
        via_point_name: Optional via name.
        expected_state_revision: Optional optimistic concurrency guard.
        idempotency_key: Optional exact-replay key.
        source_event_id: External event grounding the mutation.

    Returns:
        Committed route plan and removed point metadata.

    Raises:
        KeyError: If the session or via point does not exist.
        RouteRevisionConflict: If the expected revision is stale.
    """

    cached = _get_route_mutation_result(session_id, idempotency_key)
    if cached is not None:
        cached["idempotent_replay"] = True
        return cached

    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    base = copy.deepcopy(session)
    base_revision = _validate_expected_revision(base, expected_state_revision)
    via_points = _normalize_via_points(base.get("via_points"))
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
    candidate = copy.deepcopy(base)
    candidate["via_points"] = via_points
    route_plan = await _compute_combined_route_plan(
        app_context=app_context,
        origin=base["origin"],
        destination=base["destination"],
        via_points=via_points,
        mode=base["mode"],
    )
    return _commit_route_mutation(
        session_id=session_id,
        base_revision=base_revision,
        candidate=candidate,
        route_plan=route_plan,
        operation="remove_via_and_replan",
        idempotency_key=idempotency_key,
        source_event_id=source_event_id,
        extra_result={"removed_via_point": removed},
    )


async def replace_destination(
    *,
    app_context: Any,
    session_id: str,
    destination: str,
    expected_state_revision: int | None = None,
    idempotency_key: str | None = None,
    source_event_id: str | None = None,
) -> dict[str, Any]:
    """Replace the destination through a revision-checked commit.

    Args:
        app_context: Map-provider application context.
        session_id: Route session id.
        destination: New final destination.
        expected_state_revision: Optional optimistic concurrency guard.
        idempotency_key: Optional exact-replay key.
        source_event_id: External event grounding the mutation.

    Returns:
        Committed route plan and revision metadata.

    Raises:
        KeyError: If the session does not exist.
        RouteRevisionConflict: If the expected revision is stale.
    """

    cached = _get_route_mutation_result(session_id, idempotency_key)
    if cached is not None:
        cached["idempotent_replay"] = True
        return cached

    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    base = copy.deepcopy(session)
    base_revision = _validate_expected_revision(base, expected_state_revision)
    candidate = copy.deepcopy(base)
    candidate["destination"] = destination
    route_plan = await _compute_combined_route_plan(
        app_context=app_context,
        origin=base["origin"],
        destination=destination,
        via_points=base.get("via_points"),
        mode=base["mode"],
    )
    return _commit_route_mutation(
        session_id=session_id,
        base_revision=base_revision,
        candidate=candidate,
        route_plan=route_plan,
        operation="replace_destination",
        idempotency_key=idempotency_key,
        source_event_id=source_event_id,
    )


async def replan_route_session_from_origin(
    *,
    app_context: Any,
    session_id: str,
    origin: str,
    expected_state_revision: int | None = None,
    idempotency_key: str | None = None,
    source_event_id: str | None = None,
) -> dict[str, Any]:
    """Update the current origin and replan through a CAS commit.

    Args:
        app_context: Map-provider application context.
        session_id: Route session id.
        origin: New current origin.
        expected_state_revision: Optional optimistic concurrency guard.
        idempotency_key: Optional exact-replay key.
        source_event_id: External event grounding the mutation.

    Returns:
        Committed route plan and revision metadata.

    Raises:
        KeyError: If the session does not exist.
        RouteRevisionConflict: If the expected revision is stale.
    """

    cached = _get_route_mutation_result(session_id, idempotency_key)
    if cached is not None:
        cached["idempotent_replay"] = True
        return cached

    session = _get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    base = copy.deepcopy(session)
    base_revision = _validate_expected_revision(base, expected_state_revision)
    candidate = copy.deepcopy(base)
    candidate["origin"] = origin
    route_plan = await _compute_combined_route_plan(
        app_context=app_context,
        origin=origin,
        destination=base["destination"],
        via_points=base.get("via_points"),
        mode=base["mode"],
    )
    return _commit_route_mutation(
        session_id=session_id,
        base_revision=base_revision,
        candidate=candidate,
        route_plan=route_plan,
        operation="replan_from_current_pose",
        idempotency_key=idempotency_key,
        source_event_id=source_event_id,
    )
