from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from typing import Any
from uuid import uuid4

from aiohttp import ClientSession, WSMsgType, web

from mcp_server.map_providers import create_map_client
from mcp_server.recommendation import DestinationRecommendationEngine
from mcp_server import server as mcp_server


LOGGER = logging.getLogger("mcp_server.http_api")


@dataclass
class HTTPAPIState:
    app_context: mcp_server.AppContext

    @property
    def provider(self) -> str:
        return self.app_context.provider

    @property
    def map_client(self):
        return self.app_context.osm_client


class LoggingContext:
    async def info(self, message: str) -> None:
        LOGGER.info(message)

    async def warning(self, message: str) -> None:
        LOGGER.warning(message)

    async def report_progress(self, current: int, total: int) -> None:
        LOGGER.info("progress %s/%s", current, total)


def _json_error(message: str, *, status: int = 400, code: str = "bad_request", detail: Any = None) -> web.Response:
    payload = {
        "error": {
            "code": code,
            "message": message,
        },
        "timestamp": datetime.now().isoformat(),
    }
    if detail is not None:
        payload["error"]["detail"] = detail
    return web.json_response(payload, status=status)


def _bool_query(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def _read_json(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise web.HTTPBadRequest(text=json.dumps({
            "error": {
                "code": "invalid_json",
                "message": "Request body must be valid JSON.",
                "detail": str(exc),
            }
        }, ensure_ascii=False), content_type="application/json") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text=json.dumps({
            "error": {
                "code": "invalid_json",
                "message": "Request body must be a JSON object.",
            }
        }, ensure_ascii=False), content_type="application/json")
    return payload


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing required field: {key}")
    return value


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"Missing required field: {key}")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Field {key} must be numeric") from exc


def _extract_nav_payload_from_route_plan(route_plan: dict[str, Any]) -> dict[str, Any]:
    route_id = f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    coarse_waypoints = route_plan.get("coarse_waypoints", []) or []
    waypoints: list[dict[str, Any]] = []
    for wp in coarse_waypoints:
        if not isinstance(wp, dict):
            continue
        location = str(wp.get("location") or "").strip()
        if not location or "," not in location:
            continue
        try:
            lon_text, lat_text = location.split(",", 1)
            longitude = float(lon_text)
            latitude = float(lat_text)
        except Exception:
            continue
        waypoints.append(
            {
                "waypoint_index": int(wp.get("global_waypoint_index", 0) or 0),
                "longitude": longitude,
                "latitude": latitude,
                "is_terminal": bool(wp.get("is_leg_terminal", False)),
                "is_via_stop": bool(wp.get("is_via_stop", False)),
            }
        )
    return {
        "route_id": route_id,
        "total_count": len(waypoints),
        "waypoints": waypoints,
    }


async def _session_snapshot(state: HTTPAPIState, session_id: str, *, force_replan: bool = False) -> dict[str, Any]:
    session = mcp_server._get_persisted_route_session(session_id)
    if not session:
        raise KeyError(f"Unknown route session: {session_id}")
    route_plan, cache_hit = await mcp_server._route_plan_for_session(
        app_context=state.app_context,
        session=session,
        force_replan=force_replan,
    )
    return {
        "session": session,
        "route_plan": route_plan,
        "route_plan_cache_hit": cache_hit,
        "session_store": mcp_server._route_session_store_info(),
    }


async def _handle_health(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    return web.json_response(
        {
            "status": "ok",
            "provider": state.provider,
            "session_store": mcp_server._route_session_store_info(),
            "timestamp": datetime.now().isoformat(),
        }
    )


async def _handle_recommendations(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    payload = await _read_json(request)
    try:
        user_instruction = _required_str(payload, "user_instruction")
        current_latitude = _required_float(payload, "current_latitude")
        current_longitude = _required_float(payload, "current_longitude")
    except ValueError as exc:
        return _json_error(str(exc))

    engine = DestinationRecommendationEngine(
        map_client=state.map_client,
        provider=state.provider,
    )
    ctx = LoggingContext()
    result = await engine.recommend(
        user_instruction=user_instruction,
        current_latitude=current_latitude,
        current_longitude=current_longitude,
        max_candidates=int(payload.get("max_candidates", 5) or 5),
        search_radius=float(payload.get("search_radius", 3000) or 3000),
        mode=str(payload.get("mode") or "").strip().lower() or None,
        resolution_scope=str(payload.get("resolution_scope") or "global_destination").strip() or "global_destination",
        anchor_location=str(payload.get("anchor_location") or "").strip() or None,
        ctx=ctx,
    )
    return web.json_response(result)


async def _handle_plan_route(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    payload = await _read_json(request)
    try:
        origin = _required_str(payload, "origin")
        destination = _required_str(payload, "destination")
    except ValueError as exc:
        return _json_error(str(exc))

    route_plan = await mcp_server._compute_combined_route_plan(
        app_context=state.app_context,
        origin=origin,
        destination=destination,
        via_points=payload.get("via_points"),
        mode=str(payload.get("mode") or "foot").strip().lower() or "foot",
    )
    return web.json_response(
        {
            "provider": state.provider,
            "route_plan": route_plan,
        }
    )


async def _handle_create_route_session(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    payload = await _read_json(request)
    try:
        origin = _required_str(payload, "origin")
        destination = _required_str(payload, "destination")
    except ValueError as exc:
        return _json_error(str(exc))

    normalized_via_points = mcp_server._normalize_via_points(payload.get("via_points"))
    mode = str(payload.get("mode") or "foot").strip().lower() or "foot"
    route_plan = await mcp_server._compute_combined_route_plan(
        app_context=state.app_context,
        origin=origin,
        destination=destination,
        via_points=normalized_via_points,
        mode=mode,
    )
    session_id = f"route_{uuid4().hex[:10]}"
    session_payload = {
        "session_id": session_id,
        "session_name": str(payload.get("session_name") or "").strip() or session_id,
        "provider": state.provider,
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "via_points": normalized_via_points,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    mcp_server._cache_route_plan_on_session(session_payload, route_plan)
    mcp_server._save_persisted_route_session(session_payload)
    return web.json_response(
        {
            "session": session_payload,
            "route_plan": route_plan,
            "route_plan_cache_hit": False,
            "session_store": mcp_server._route_session_store_info(),
        }
    )


async def _handle_get_route_session(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    session_id = request.match_info["session_id"]
    try:
        snapshot = await _session_snapshot(
            state,
            session_id,
            force_replan=_bool_query(request.query.get("force_replan"), default=False),
        )
    except KeyError as exc:
        return _json_error(str(exc), status=404, code="not_found")
    return web.json_response(snapshot)


async def _handle_add_via_point(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    session_id = request.match_info["session_id"]
    payload = await _read_json(request)
    session = mcp_server._get_persisted_route_session(session_id)
    if not session:
        return _json_error(f"Unknown route session: {session_id}", status=404, code="not_found")

    normalized_new = mcp_server._normalize_via_points([payload.get("new_via_point") or payload])
    if not normalized_new:
        return _json_error("new_via_point must include a valid location")

    old_via_points = mcp_server._normalize_via_points(session.get("via_points"))
    via_points = [dict(item) for item in old_via_points]
    item = normalized_new[0]
    insert_before_index = payload.get("insert_before_index")
    same_as_destination = mcp_server._locations_equivalent(item.get("location"), session.get("destination"))
    duplicate_index = next(
        (
            index for index, via in enumerate(via_points)
            if mcp_server._locations_equivalent(via.get("location"), item.get("location"))
        ),
        None,
    )
    if same_as_destination or duplicate_index is not None:
        snapshot = await _session_snapshot(state, session_id, force_replan=False)
        snapshot.update(
            {
                "session_changed": False,
                "change_reason": "matches_destination" if same_as_destination else f"duplicate_via_point:{duplicate_index}",
                "route_plan_update_mode": "cache_reuse_noop",
            }
        )
        return web.json_response(snapshot)

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
        route_plan = await mcp_server._compute_incremental_route_plan_with_inserted_via(
            app_context=state.app_context,
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
        route_plan = await mcp_server._compute_combined_route_plan(
            app_context=state.app_context,
            origin=session["origin"],
            destination=session["destination"],
            via_points=via_points,
            mode=session["mode"],
        )

    mcp_server._cache_route_plan_on_session(session, route_plan)
    mcp_server._save_persisted_route_session(session)
    return web.json_response(
        {
            "session": session,
            "route_plan": route_plan,
            "route_plan_cache_hit": False,
            "session_changed": True,
            "change_reason": "via_point_added",
            "route_plan_update_mode": route_plan_update_mode,
            "session_store": mcp_server._route_session_store_info(),
        }
    )


async def _handle_replan_route_session(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    session_id = request.match_info["session_id"]
    payload = await _read_json(request)
    session = mcp_server._get_persisted_route_session(session_id)
    if not session:
        return _json_error(f"Unknown route session: {session_id}", status=404, code="not_found")
    try:
        origin = _required_str(payload, "origin")
    except ValueError as exc:
        return _json_error(str(exc))

    session["origin"] = origin
    session["updated_at"] = datetime.now().isoformat()
    route_plan = await mcp_server._compute_combined_route_plan(
        app_context=state.app_context,
        origin=session["origin"],
        destination=session["destination"],
        via_points=session.get("via_points"),
        mode=session["mode"],
    )
    mcp_server._cache_route_plan_on_session(session, route_plan)
    mcp_server._save_persisted_route_session(session)
    return web.json_response(
        {
            "session": session,
            "route_plan": route_plan,
            "route_plan_cache_hit": False,
            "session_store": mcp_server._route_session_store_info(),
        }
    )


async def _handle_push_waypoints(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    payload = await _read_json(request)
    route_plan = payload.get("route_plan")
    session = None

    session_id = str(payload.get("session_id") or "").strip()
    if not route_plan:
        if not session_id:
            return _json_error("Either session_id or route_plan is required.")
        session = mcp_server._get_persisted_route_session(session_id)
        if not session:
            return _json_error(f"Unknown route session: {session_id}", status=404, code="not_found")
        snapshot = await _session_snapshot(state, session_id, force_replan=False)
        route_plan = snapshot.get("route_plan") or {}

    if not isinstance(route_plan, dict) or not route_plan:
        return _json_error("No valid route_plan available for waypoint push.")

    nav_payload = _extract_nav_payload_from_route_plan(route_plan)
    robot_host = str(payload.get("robot_host") or "192.168.168.110").strip() or "192.168.168.110"
    robot_port = int(payload.get("robot_port", 8765) or 8765)
    ws_url = f"ws://{robot_host}:{robot_port}"

    try:
        async with ClientSession() as client:
            async with client.ws_connect(ws_url) as ws:
                await ws.send_json(nav_payload)
    except Exception as exc:
        return web.json_response(
            {
                "status": "failed",
                "ws_info": ws_url,
                "error": str(exc),
                **nav_payload,
            },
            status=502,
        )

    return web.json_response(
        {
            "status": "sent",
            "ws_info": ws_url,
            "session_id": session_id or None,
            "provider": (session or {}).get("provider") if isinstance(session, dict) else state.provider,
            **nav_payload,
        }
    )


async def _handle_route_session_ws(request: web.Request) -> web.StreamResponse:
    state: HTTPAPIState = request.app["state"]
    session_id = request.match_info["session_id"]
    poll_interval_s = max(float(request.query.get("poll_interval_s", "2.0") or 2.0), 0.5)
    ws = web.WebSocketResponse(heartbeat=20.0)
    await ws.prepare(request)

    try:
        snapshot = await _session_snapshot(state, session_id, force_replan=False)
    except KeyError as exc:
        await ws.send_json({"error": {"code": "not_found", "message": str(exc)}})
        await ws.close()
        return ws

    await ws.send_json({"type": "snapshot", "data": snapshot})
    last_version = (
        str((snapshot.get("session") or {}).get("updated_at") or ""),
        str(((snapshot.get("session") or {}).get("route_plan_cached_at") or "")),
    )

    while not ws.closed:
        try:
            message = await ws.receive(timeout=poll_interval_s)
        except asyncio.TimeoutError:
            current = mcp_server._get_persisted_route_session(session_id)
            if not current:
                await ws.send_json(
                    {
                        "type": "deleted",
                        "data": {"session_id": session_id},
                    }
                )
                break
            version = (
                str(current.get("updated_at") or ""),
                str(current.get("route_plan_cached_at") or ""),
            )
            if version != last_version:
                snapshot = await _session_snapshot(state, session_id, force_replan=False)
                last_version = version
                await ws.send_json({"type": "snapshot", "data": snapshot})
            else:
                await ws.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
            continue

        if message.type == WSMsgType.TEXT:
            try:
                data = json.loads(message.data)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "error": {"code": "invalid_json", "message": "Message must be valid JSON."}})
                continue

            action = str((data or {}).get("action") or "refresh").strip().lower()
            if action == "refresh":
                try:
                    snapshot = await _session_snapshot(
                        state,
                        session_id,
                        force_replan=bool((data or {}).get("force_replan", False)),
                    )
                    last_version = (
                        str((snapshot.get("session") or {}).get("updated_at") or ""),
                        str(((snapshot.get("session") or {}).get("route_plan_cached_at") or "")),
                    )
                    await ws.send_json({"type": "snapshot", "data": snapshot})
                except KeyError as exc:
                    await ws.send_json({"type": "error", "error": {"code": "not_found", "message": str(exc)}})
                    break
            else:
                await ws.send_json({"type": "error", "error": {"code": "unsupported_action", "message": f"Unsupported action: {action}"}})
        elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED}:
            break
        elif message.type == WSMsgType.ERROR:
            LOGGER.warning("route session websocket error: %s", ws.exception())
            break

    await ws.close()
    return ws


@web.middleware
async def json_error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except web.HTTPException as exc:
        if exc.content_type == "application/json" and exc.text:
            raise
        return _json_error(exc.reason or "HTTP error", status=exc.status, code="http_error")
    except ValueError as exc:
        return _json_error(str(exc), status=400, code="validation_error")
    except KeyError as exc:
        return _json_error(str(exc), status=404, code="not_found")
    except Exception as exc:
        LOGGER.exception("Unhandled HTTP API error")
        return _json_error("Internal server error", status=500, code="internal_error", detail=str(exc))


async def app_cleanup_ctx(app: web.Application):
    map_client, provider = create_map_client()
    await map_client.connect()
    state = HTTPAPIState(app_context=mcp_server.AppContext(osm_client=map_client, provider=provider))
    app["state"] = state
    try:
        yield
    finally:
        await map_client.disconnect()


def create_app() -> web.Application:
    app = web.Application(middlewares=[json_error_middleware])
    app.cleanup_ctx.append(app_cleanup_ctx)
    app.router.add_get("/api/health", _handle_health)
    app.router.add_post("/api/recommendations", _handle_recommendations)
    app.router.add_post("/api/routes/plan", _handle_plan_route)
    app.router.add_post("/api/route-sessions", _handle_create_route_session)
    app.router.add_get("/api/route-sessions/{session_id}", _handle_get_route_session)
    app.router.add_post("/api/route-sessions/{session_id}/via-points", _handle_add_via_point)
    app.router.add_post("/api/route-sessions/{session_id}/replan", _handle_replan_route_session)
    app.router.add_post("/api/waypoints/push", _handle_push_waypoints)
    app.router.add_get("/ws/route-sessions/{session_id}", _handle_route_session_ws)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenStreetMcp HTTP API server")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8080, help="HTTP bind port")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    web.run_app(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
