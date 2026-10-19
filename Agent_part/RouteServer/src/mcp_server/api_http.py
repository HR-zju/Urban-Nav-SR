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
from mcp_server.core.route_session_ops import (
    RouteRevisionConflict,
    add_via_point as core_add_via_point,
    create_route_session as core_create_route_session,
    get_route_session_snapshot,
    replan_route_session_from_origin as core_replan_route_session_from_origin,
)


LOGGER = logging.getLogger("mcp_server.http_api")
EPISODE_SERVICE_KEY = web.AppKey("episode_service", object)
EPISODE_EVENT_SERVICE_KEY = web.AppKey("episode_event_service", object)


@dataclass
class HTTPAPIState:
    app_context: mcp_server.AppContext | None
    episode_service: Any | None = None
    episode_event_service: Any | None = None

    @property
    def provider(self) -> str:
        return self.app_context.provider if self.app_context is not None else "episode_only"

    @property
    def map_client(self):
        if self.app_context is None:
            raise RuntimeError("map API is unavailable in episode-only mode")
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
    return await get_route_session_snapshot(
        app_context=state.app_context,
        session_id=session_id,
        force_replan=force_replan,
    )


async def _handle_health(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    return web.json_response(
        {
            "status": "ok",
            "provider": state.provider,
            "session_store": mcp_server._route_session_store_info(),
            "episode_service": state.episode_service is not None,
            "episode_event_service": (state.episode_event_service or state.episode_service) is not None,
            "timestamp": datetime.now().isoformat(),
        }
    )


async def _handle_episode_reset(request: web.Request) -> web.Response:
    """Reset one externally-driven training or benchmark episode.

    Args:
        request: Aiohttp request containing an episode payload.

    Returns:
        Strict JSON response consumed by ``UrbanNavEnvClient``.
    """

    state: HTTPAPIState = request.app["state"]
    if state.episode_service is None:
        return _json_error(
            "Episode service is not configured.",
            status=503,
            code="episode_service_unavailable",
        )
    from mcp_server.runtime.service import EpisodeResetResponse

    payload = await _read_json(request)
    response = await state.episode_service.reset(payload)
    validated = response if isinstance(response, EpisodeResetResponse) else EpisodeResetResponse.model_validate(response)
    return web.json_response(validated.model_dump(mode="json", exclude_none=True))


async def _handle_episode_step(request: web.Request) -> web.Response:
    """Apply one external model action to an active episode.

    Args:
        request: Aiohttp request containing action and message history.

    Returns:
        Strict reward/observation response consumed by ``UrbanNavEnvClient``.
    """

    state: HTTPAPIState = request.app["state"]
    if state.episode_service is None:
        return _json_error(
            "Episode service is not configured.",
            status=503,
            code="episode_service_unavailable",
        )
    from mcp_server.runtime.service import EpisodeStepResponse

    payload = await _read_json(request)
    episode_id = str(request.match_info["episode_id"])
    response = await state.episode_service.step(episode_id, payload)
    validated = response if isinstance(response, EpisodeStepResponse) else EpisodeStepResponse.model_validate(response)
    return web.json_response(validated.model_dump(mode="json", exclude_none=True))


async def _handle_episode_event(request: web.Request) -> web.Response:
    """Submit one out-of-band task/control/dialogue event.

    Args:
        request: Aiohttp request containing an ``ExternalUserEvent`` payload.

    Returns:
        Strict idempotent event-ingress response.
    """

    state: HTTPAPIState = request.app["state"]
    event_service = state.episode_event_service or state.episode_service
    if event_service is None or not hasattr(event_service, "submit_event"):
        return _json_error(
            "Episode event service is not configured.",
            status=503,
            code="episode_event_service_unavailable",
        )
    from mcp_server.runtime.service import EpisodeEventSubmitResponse

    payload = await _read_json(request)
    episode_id = str(request.match_info["episode_id"])
    response = await event_service.submit_event(episode_id, payload)
    validated = (
        response
        if isinstance(response, EpisodeEventSubmitResponse)
        else EpisodeEventSubmitResponse.model_validate(response)
    )
    return web.json_response(validated.model_dump(mode="json", exclude_none=True))


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

    mode = str(payload.get("mode") or "foot").strip().lower() or "foot"
    result = await core_create_route_session(
        app_context=state.app_context,
        origin=origin,
        destination=destination,
        via_points=payload.get("via_points"),
        mode=mode,
        session_name=str(payload.get("session_name") or "").strip() or None,
        idempotency_key=str(payload.get("idempotency_key") or "").strip() or None,
        source_event_id=str(payload.get("source_event_id") or "").strip() or None,
    )
    return web.json_response(result)


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
    try:
        result = await core_add_via_point(
            app_context=state.app_context,
            session_id=session_id,
            new_via_point=payload.get("new_via_point") or payload,
            insert_before_index=payload.get("insert_before_index"),
            expected_state_revision=payload.get("expected_state_revision"),
            idempotency_key=str(payload.get("idempotency_key") or "").strip() or None,
            source_event_id=str(payload.get("source_event_id") or "").strip() or None,
        )
    except RouteRevisionConflict as exc:
        return _json_error(str(exc), status=409, code="route_revision_conflict")
    except KeyError as exc:
        return _json_error(str(exc.args[0] if exc.args else exc), status=404, code="not_found")
    except ValueError as exc:
        return _json_error(str(exc))
    return web.json_response(result)


async def _handle_replan_route_session(request: web.Request) -> web.Response:
    state: HTTPAPIState = request.app["state"]
    session_id = request.match_info["session_id"]
    payload = await _read_json(request)
    try:
        origin = _required_str(payload, "origin")
        result = await core_replan_route_session_from_origin(
            app_context=state.app_context,
            session_id=session_id,
            origin=origin,
            expected_state_revision=payload.get("expected_state_revision"),
            idempotency_key=str(payload.get("idempotency_key") or "").strip() or None,
            source_event_id=str(payload.get("source_event_id") or "").strip() or None,
        )
    except RouteRevisionConflict as exc:
        return _json_error(str(exc), status=409, code="route_revision_conflict")
    except KeyError as exc:
        return _json_error(str(exc.args[0] if exc.args else exc), status=404, code="not_found")
    except ValueError as exc:
        return _json_error(str(exc))
    return web.json_response(result)


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
    state = HTTPAPIState(
        app_context=mcp_server.AppContext(osm_client=map_client, provider=provider),
        episode_service=app.get(EPISODE_SERVICE_KEY),
        episode_event_service=app.get(EPISODE_EVENT_SERVICE_KEY),
    )
    app["state"] = state
    try:
        yield
    finally:
        await map_client.disconnect()


async def episode_app_cleanup_ctx(app: web.Application):
    """Install an episode-only state without opening any map connection."""

    app["state"] = HTTPAPIState(
        app_context=None,
        episode_service=app.get(EPISODE_SERVICE_KEY),
        episode_event_service=app.get(EPISODE_EVENT_SERVICE_KEY),
    )
    yield


def create_app(
    *,
    episode_service: Any | None = None,
    episode_event_service: Any | None = None,
    episode_only: bool = False,
) -> web.Application:
    """Create the RouteServer HTTP application.

    Args:
        episode_service: Optional externally-driven episode backend. The API
            returns HTTP 503 instead of fabricating transitions when absent.
        episode_event_service: Optional registered live-runtime event backend.
        episode_only: Expose only frozen episode endpoints and avoid creating a
            map provider client. This is required for deterministic benchmark
            serving.

    Returns:
        Configured aiohttp application.
    """

    app = web.Application(middlewares=[json_error_middleware])
    app[EPISODE_SERVICE_KEY] = episode_service
    app[EPISODE_EVENT_SERVICE_KEY] = episode_event_service
    app.cleanup_ctx.append(episode_app_cleanup_ctx if episode_only else app_cleanup_ctx)
    app.router.add_get("/api/health", _handle_health)
    if not episode_only:
        app.router.add_post("/api/recommendations", _handle_recommendations)
        app.router.add_post("/api/routes/plan", _handle_plan_route)
        app.router.add_post("/api/route-sessions", _handle_create_route_session)
        app.router.add_get("/api/route-sessions/{session_id}", _handle_get_route_session)
        app.router.add_post("/api/route-sessions/{session_id}/via-points", _handle_add_via_point)
        app.router.add_post("/api/route-sessions/{session_id}/replan", _handle_replan_route_session)
        app.router.add_post("/api/waypoints/push", _handle_push_waypoints)
        app.router.add_get("/ws/route-sessions/{session_id}", _handle_route_session_ws)
    app.router.add_post("/v1/episodes/reset", _handle_episode_reset)
    app.router.add_post("/v1/episodes/{episode_id}/step", _handle_episode_step)
    app.router.add_post("/v1/episodes/{episode_id}/events", _handle_episode_event)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenStreetMcp HTTP API server")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8080, help="HTTP bind port")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--episode-cases",
        default=None,
        help="Optional frozen EpisodeCase JSONL exposed at /v1/episodes.",
    )
    parser.add_argument(
        "--episode-only",
        action="store_true",
        help="Serve frozen episodes without connecting to a map provider.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    episode_service = None
    if args.episode_cases:
        from mcp_server.eval.agent_tools.episode import FrozenExternalEpisodeService, load_frozen_episode_cases

        episode_service = FrozenExternalEpisodeService(load_frozen_episode_cases(args.episode_cases))
    if args.episode_only and episode_service is None:
        parser.error("--episode-only requires --episode-cases")
    web.run_app(
        create_app(episode_service=episode_service, episode_only=args.episode_only),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
