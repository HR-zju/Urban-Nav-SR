from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional


DEFAULT_CACHE_PATH = str(Path.home() / ".cache" / "urban_nav" / "realtime_instruction_session_cache.json")
TIMING_LOGGER = logging.getLogger("realtime_instruction_session.demo.timing")


def _setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def _log_timing(stage: str, elapsed_s: float, **fields: Any) -> None:
    payload = {
        "component": "realtime_instruction_session_demo",
        "stage": stage,
        "elapsed_ms": round(float(elapsed_s) * 1000.0, 2),
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    TIMING_LOGGER.info("TIMING %s", json.dumps(payload, ensure_ascii=True, sort_keys=True))


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two points."""
    radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_location_coords(location: str) -> tuple[float, float] | None:
    """Parse 'lon,lat' string into a (lat, lon) tuple."""
    if not location or "," not in location:
        return None
    parts = location.split(",")
    if len(parts) != 2:
        return None
    try:
        lon, lat = float(parts[0]), float(parts[1])
        return (lat, lon)
    except (TypeError, ValueError):
        return None


def compute_route_summary_from_polyline(polyline: list[dict], mode: str) -> dict:
    """Compute a lightweight route summary when the server payload omits one."""
    if not polyline or len(polyline) < 2:
        return {}

    total_distance = 0.0
    for i in range(1, len(polyline)):
        p0, p1 = polyline[i - 1], polyline[i]
        lat0 = p0.get("latitude") or p0.get("lat")
        lon0 = p0.get("longitude") or p0.get("lon")
        lat1 = p1.get("latitude") or p1.get("lat")
        lon1 = p1.get("longitude") or p1.get("lon")
        if all(isinstance(v, (int, float)) for v in [lat0, lon0, lat1, lon1]):
            total_distance += haversine_distance_m(lat0, lon0, lat1, lon1)

    speed_ms = {
        "foot": 1.2,
        "walking": 1.2,
        "bike": 4.0,
        "cycling": 4.0,
        "car": 13.0,
        "driving": 13.0,
    }.get(mode, 1.2)
    duration_s = total_distance / speed_ms if speed_ms > 0 else 0.0
    return {
        "distance_m": round(total_distance, 1),
        "duration_s": round(duration_s, 1),
        "duration_min": round(duration_s / 60.0, 1),
    }


def compute_leg_summaries_from_stop_sequence(stop_sequence: list[dict], polyline: list[dict], mode: str) -> list[dict]:
    """Attach approximate per-leg summaries for app payloads when missing."""
    if not stop_sequence:
        return []

    speed_ms = {
        "foot": 1.2,
        "walking": 1.2,
        "bike": 4.0,
        "cycling": 4.0,
        "car": 13.0,
        "driving": 13.0,
    }.get(mode, 1.2)

    leg_summaries: list[dict] = []
    for idx, stop in enumerate(stop_sequence):
        stop_type = stop.get("stop_type") or stop.get("type") or ""
        name = stop.get("name") or ""
        location = stop.get("location") or ""

        if stop_type == "origin":
            next_stop = stop_sequence[idx + 1] if idx + 1 < len(stop_sequence) else None
            to_name = next_stop.get("name") or "" if next_stop else ""
            prev_location = location
            next_location = next_stop.get("location") if next_stop else None
            from_name = name
        elif stop_type in ("via", "destination"):
            prev_stop = stop_sequence[idx - 1] if idx > 0 else None
            from_name = prev_stop.get("name") or "" if prev_stop else ""
            to_name = name
            prev_location = prev_stop.get("location") if prev_stop else None
            next_location = location
        else:
            enriched = dict(stop)
            enriched.setdefault("stop_type", stop_type)
            enriched.setdefault("route_leg_summary", None)
            leg_summaries.append(enriched)
            continue

        dist, dur = 0.0, 0.0
        prev_coords = parse_location_coords(str(prev_location or ""))
        next_coords = parse_location_coords(str(next_location or ""))
        if prev_coords and next_coords:
            dist = haversine_distance_m(prev_coords[0], prev_coords[1], next_coords[0], next_coords[1])
            dur = dist / speed_ms if speed_ms > 0 else 0.0

        enriched = dict(stop)
        enriched["stop_type"] = stop_type
        enriched["route_leg_summary"] = (
            {
                "distance_m": round(dist, 1),
                "duration_min": round(dur / 60.0, 1),
                "from_stop": from_name,
                "to_stop": to_name,
            }
            if dist > 0
            else None
        )
        leg_summaries.append(enriched)

    return leg_summaries


def first_text_content(result) -> str:
    if not result or not getattr(result, "content", None):
        return ""
    for content_item in result.content:
        if content_item.type == "text" and isinstance(content_item.text, str):
            return content_item.text
    return ""


def parse_first_text_payload(result) -> Optional[dict]:
    text = first_text_content(result)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def describe_tool_result(result: Any) -> str:
    parts: list[str] = []
    if result is None:
        return "result=None"
    is_error = getattr(result, "isError", None)
    if is_error is None:
        is_error = getattr(result, "is_error", None)
    if is_error is not None:
        parts.append(f"is_error={is_error!r}")

    content_items = getattr(result, "content", None) or []
    parts.append(f"content_count={len(content_items)}")
    for index, item in enumerate(content_items):
        item_type = getattr(item, "type", None)
        text = getattr(item, "text", None)
        data = getattr(item, "data", None)
        if isinstance(text, str):
            preview = text if len(text) <= 2000 else text[:2000] + "...<truncated>"
            parts.append(f"content[{index}].type={item_type!r} text={preview!r}")
        elif data is not None:
            parts.append(f"content[{index}].type={item_type!r} data={data!r}")
        else:
            parts.append(f"content[{index}]={item!r}")

    structured_content = getattr(result, "structuredContent", None)
    if structured_content is None:
        structured_content = getattr(result, "structured_content", None)
    if structured_content:
        parts.append(f"structured_content={structured_content!r}")
    return "; ".join(parts)


async def call_tool_json(session: ClientSession, tool_name: str, arguments: dict) -> dict:
    result = await session.call_tool(tool_name, arguments)
    payload = parse_first_text_payload(result)
    if payload is not None:
        return payload
    raw_text = first_text_content(result)
    detail = describe_tool_result(result)
    if raw_text.startswith("Error executing tool "):
        raise RuntimeError(f"{raw_text}\nTool arguments: {json.dumps(arguments, ensure_ascii=False)}\nMCP result: {detail}")
    raise RuntimeError(
        f"{tool_name} returned non-JSON payload.\n"
        f"Tool arguments: {json.dumps(arguments, ensure_ascii=False)}\n"
        f"MCP result: {detail}"
    )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _stable_cache_key(parts: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _default_cache_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "instruction_results": {},
        "name_locations": {},
    }


def _load_json_cache(cache_path: str) -> dict[str, Any]:
    path = Path(cache_path).expanduser()
    if not path.exists():
        return _default_cache_payload()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_cache_payload()
    if not isinstance(payload, dict) or int(payload.get("version", -1)) != 1:
        return _default_cache_payload()
    payload.setdefault("instruction_results", {})
    payload.setdefault("name_locations", {})
    return payload


def _save_json_cache(cache_path: str, payload: dict[str, Any]) -> None:
    path = Path(cache_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _instruction_cache_key(
    *,
    instruction: str,
    current_latitude: float,
    current_longitude: float,
    mode: str,
    search_radius: float,
    max_candidates: int,
    resolution_scope: str = "global_destination",
    anchor_location: str | None = None,
    cache_namespace: str = "",
) -> str:
    return _stable_cache_key(
        [
            "instruction_resolution",
            str(cache_namespace or "").strip().lower(),
            _normalize_text(instruction),
            round(float(current_latitude), 6),
            round(float(current_longitude), 6),
            str(mode).strip().lower(),
            round(float(search_radius), 2),
            int(max_candidates),
            str(resolution_scope).strip().lower(),
            str(anchor_location or "").strip(),
        ]
    )


def _name_location_cache_key(name: str, *, cache_namespace: str = "") -> str:
    return _stable_cache_key(["name_location", str(cache_namespace or "").strip().lower(), _normalize_text(name)])


def as_location_text(chosen_destination: Dict[str, Any]) -> Optional[str]:
    location = str(chosen_destination.get("location") or "").strip()
    if "," in location:
        return location

    coordinates = chosen_destination.get("coordinates") or {}
    lat = coordinates.get("latitude")
    lon = coordinates.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return f"{float(lon)},{float(lat)}"
    return None


async def resolve_instruction_to_destination(
    *,
    session: ClientSession,
    instruction: str,
    current_latitude: float,
    current_longitude: float,
    mode: str,
    search_radius: float,
    max_candidates: int,
    cache_payload: Optional[Dict[str, Any]] = None,
    resolution_scope: str = "global_destination",
    anchor_location: Optional[str] = None,
    cache_enabled: bool = True,
    cache_namespace: str = "",
) -> Dict[str, Any]:
    cache_is_enabled = bool(cache_enabled) and isinstance(cache_payload, dict)
    instruction_cache = (cache_payload or {}).get("instruction_results", {}) if cache_is_enabled else {}
    name_cache = (cache_payload or {}).get("name_locations", {}) if cache_is_enabled else {}

    name_cached = (
        name_cache.get(_name_location_cache_key(instruction, cache_namespace=cache_namespace))
        if isinstance(name_cache, dict)
        else None
    )
    if isinstance(name_cached, dict) and str(name_cached.get("location") or "").strip():
        _log_timing(
            "cache_hit",
            0.0,
            cache_kind="name_location",
            instruction=instruction,
        )
        return {
            "name": str(name_cached.get("name") or instruction).strip() or instruction,
            "location": str(name_cached.get("location") or "").strip(),
            "raw": dict(name_cached.get("raw") or {}),
            "recommend_result": {},
            "cache_source": "name_location",
            "resolution_path": resolution_scope,
        }

    cache_key = _instruction_cache_key(
        instruction=instruction,
        current_latitude=current_latitude,
        current_longitude=current_longitude,
        mode=mode,
        search_radius=search_radius,
        max_candidates=max_candidates,
        resolution_scope=resolution_scope,
        anchor_location=anchor_location,
        cache_namespace=cache_namespace,
    )
    cached = instruction_cache.get(cache_key) if isinstance(instruction_cache, dict) else None
    if isinstance(cached, dict) and str(cached.get("location") or "").strip():
        _log_timing(
            "cache_hit",
            0.0,
            cache_kind="instruction_result",
            instruction=instruction,
        )
        return dict(cached)

    tool_arguments = {
        "user_instruction": instruction,
        "current_latitude": current_latitude,
        "current_longitude": current_longitude,
        "mode": mode,
        "search_radius": search_radius,
        "max_candidates": max_candidates,
        "resolution_scope": resolution_scope,
        "anchor_location": anchor_location,
    }
    started_at = perf_counter()
    result = await call_tool_json(
        session,
        "recommend_destination_and_waypoints",
        tool_arguments,
    )
    _log_timing(
        "mcp_tool_call",
        perf_counter() - started_at,
        tool="recommend_destination_and_waypoints",
        instruction=instruction,
        status="ok",
        resolution_scope=resolution_scope,
    )

    chosen = result.get("chosen_destination") or {}

    if not isinstance(chosen, dict) or not chosen:
        raise RuntimeError(
            f"No chosen_destination for instruction: {instruction}\n"
            f"resolution_scope={resolution_scope}, anchor_location={anchor_location}\n"
            f"message={result.get('message')!r}, "
            f"retrieval_errors={result.get('retrieval_errors')!r}, "
            f"meta={result.get('meta')!r}"
        )
    location = as_location_text(chosen)
    if not location:
        raise RuntimeError(f"chosen_destination has no valid location: {chosen}")

    resolved = {
        "name": str(chosen.get("name") or instruction).strip() or instruction,
        "location": location,
        "raw": chosen,
        "recommend_result": result,
        "resolution_path": str((result.get("meta") or {}).get("resolution_scope") or resolution_scope),
    }
    if cache_is_enabled and isinstance(instruction_cache, dict):
        instruction_cache[cache_key] = resolved
    if cache_is_enabled and isinstance(name_cache, dict):
        name_cache[_name_location_cache_key(resolved["name"], cache_namespace=cache_namespace)] = {
            "name": resolved["name"],
            "location": resolved["location"],
            "raw": chosen,
        }
    return resolved


def _location_equal(a: str, b: str, *, tol: float = 1e-6) -> bool:
    try:
        a_lon, a_lat = [float(item) for item in str(a).split(",", 1)]
        b_lon, b_lat = [float(item) for item in str(b).split(",", 1)]
    except Exception:
        return str(a).strip() == str(b).strip()
    return abs(a_lon - b_lon) <= tol and abs(a_lat - b_lat) <= tol


def _session_update_is_noop(
    *,
    session_payload: Dict[str, Any],
    resolved_location: str,
) -> bool:
    if _location_equal(str(session_payload.get("destination") or ""), resolved_location):
        return True
    for via in session_payload.get("via_points", []) or []:
        if isinstance(via, dict) and _location_equal(str(via.get("location") or ""), resolved_location):
            return True
    return False


def _session_search_anchor(
    *,
    current_latitude: float,
    current_longitude: float,
    session_payload: Dict[str, Any],
) -> str:
    via_points = session_payload.get("via_points", []) or []
    if via_points and isinstance(via_points[-1], dict) and str(via_points[-1].get("location") or "").strip():
        return str(via_points[-1]["location"]).strip()
    origin = str(session_payload.get("origin") or "").strip()
    if origin:
        return origin
    return f"{float(current_longitude)},{float(current_latitude)}"


def _build_result_payload(
    *,
    active_session_id: str,
    resolved_destination_location: Optional[str],
    last_updated: dict,
    mode: str,
) -> dict:
    route_plan = (last_updated.get("route_plan", {}) or {}) or {}
    route_summary = route_plan.get("route_summary") or {}
    if not route_summary:
        route_summary = compute_route_summary_from_polyline(route_plan.get("route_polyline") or [], mode)

    stop_sequence = route_plan.get("stop_sequence") or []
    has_leg_summaries = any((s.get("route_leg_summary") or s.get("leg_summary")) for s in stop_sequence)
    if not has_leg_summaries and stop_sequence:
        enriched = compute_leg_summaries_from_stop_sequence(stop_sequence, route_plan.get("route_polyline") or [], mode)
        if any(e.get("route_leg_summary") for e in enriched):
            stop_sequence = enriched

    return {
        "session_id": active_session_id,
        "resolved_destination_location": resolved_destination_location,
        "session": (last_updated.get("session", {}) or {}) or {},
        "route_plan": {
            "route_summary": route_summary,
            "stop_sequence": stop_sequence,
            "route_polyline": route_plan.get("route_polyline", []),
            "coarse_waypoints": route_plan.get("coarse_waypoints", []),
            "leg_count": route_plan.get("leg_count"),
        },
        "via_points": (last_updated.get("session", {}) or {}).get("via_points", []),
    }


async def run_demo(
    *,
    current_latitude: float,
    current_longitude: float,
    mode: str,
    init_instruction: Optional[str],
    append_instruction: list[str],
    search_radius: float,
    max_candidates: int,
    cache_path: str,
    clear_cache: bool,
    append_cache_mode: str = "disabled",
    perf_profile: str = "quality",
    session_id: str | None = None,
    non_interactive: bool = False,
    map_provider: Optional[str] = None,
    amap_key: Optional[str] = None,
    intent_model: Optional[str] = None,
) -> None:
    overall_started_at = perf_counter()
    normalized_append_cache_mode = str(append_cache_mode or "disabled").strip().lower()
    if normalized_append_cache_mode not in {"disabled", "session", "global"}:
        normalized_append_cache_mode = "disabled"
    use_local_cache_for_init = session_id is None
    use_local_cache_for_append = normalized_append_cache_mode in {"session", "global"}
    should_access_local_cache = use_local_cache_for_init or use_local_cache_for_append
    if clear_cache:
        try:
            Path(cache_path).expanduser().unlink(missing_ok=True)
        except Exception:
            pass
    cache_payload = _load_json_cache(cache_path) if should_access_local_cache else _default_cache_payload()

    server_env = os.environ.copy()
    if map_provider:
        server_env["MAP_PROVIDER"] = map_provider
    if amap_key:
        server_env["AMAP_KEY"] = amap_key
    if intent_model:
        server_env["LLM_INTENT_MODEL"] = intent_model
    if perf_profile == "latency":
        latency_defaults = {
            "LLM_INTENT_PROFILE": "compact",
            "LLM_INTENT_MAX_TOKENS": "256",
            "ROUTE_EVAL_LIMIT_LOCAL_VIA": "5",
            "AMAP_POI_RETRIES": "1",
            "AMAP_POI_TOTAL_TIMEOUT_S": "20",
            "AMAP_ROUTE_RETRIES": "0",
            "AMAP_ROUTE_TOTAL_TIMEOUT_S": "35",
            "AMAP_ROUTE_ALLOW_ESTIMATED_FALLBACK": "1",
        }
        for key, value in latency_defaults.items():
            server_env.setdefault(key, value)

    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=server_env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            session_payload: dict[str, Any] = {}
            resolved_destination_location: Optional[str] = None
            last_updated: dict = {}

            if session_id:
                active_session_id = session_id
                get_started_at = perf_counter()
                last_updated = await call_tool_json(
                    session,
                    "get_route_session",
                    {"session_id": active_session_id},
                )
                _log_timing(
                    "mcp_tool_call",
                    perf_counter() - get_started_at,
                    tool="get_route_session",
                    status="ok",
                    reason="reuse_existing_session",
                )
                session_payload = last_updated.get("session", {}) or {}
            else:
                if not init_instruction:
                    raise RuntimeError("--init-instruction is required when --session-id is not provided")
                if not non_interactive:
                    print("=== Step 1: Resolve Initial Destination Instruction ===")
                init_started_at = perf_counter()
                init_resolved = await resolve_instruction_to_destination(
                    session=session,
                    instruction=init_instruction,
                    current_latitude=current_latitude,
                    current_longitude=current_longitude,
                    mode=mode,
                    search_radius=search_radius,
                    max_candidates=max_candidates,
                    cache_payload=cache_payload,
                    cache_enabled=True,
                )
                _log_timing(
                    "demo_stage",
                    perf_counter() - init_started_at,
                    stage_name="resolve_initial_instruction",
                    cache_source=init_resolved.get("cache_source"),
                )
                resolved_destination_location = init_resolved["location"]
                if not non_interactive:
                    print(
                        json.dumps(
                            {
                                "instruction": init_instruction,
                                "resolved_destination_name": init_resolved["name"],
                                "resolved_destination_location": init_resolved["location"],
                            },
                            indent=2,
                            ensure_ascii=True,
                        )
                    )

                origin = f"{float(current_longitude)},{float(current_latitude)}"
                create_started_at = perf_counter()
                last_updated = await call_tool_json(
                    session,
                    "create_route_session",
                    {
                        "session_name": "realtime_instruction_session",
                        "origin": origin,
                        "destination": init_resolved["location"],
                        "mode": mode,
                    },
                )
                _log_timing(
                    "mcp_tool_call",
                    perf_counter() - create_started_at,
                    tool="create_route_session",
                    status="ok",
                )

                session_payload = last_updated.get("session", {}) or {}
                active_session_id = str(session_payload.get("session_id") or "").strip()
                if not active_session_id:
                    raise RuntimeError("create_route_session returned empty session_id")

                if not non_interactive:
                    print("\n=== Step 2: Create Session ===")
                    print(
                        json.dumps(
                            {
                                "session_id": active_session_id,
                                "origin": session_payload.get("origin"),
                                "destination": session_payload.get("destination"),
                                "mode": session_payload.get("mode"),
                                "via_points": session_payload.get("via_points", []),
                                "route_summary": (last_updated.get("route_plan", {}) or {}).get("route_summary", {}),
                            },
                            indent=2,
                            ensure_ascii=True,
                        )
                    )

            for idx, instruction in enumerate(append_instruction or [], start=1):
                if not non_interactive:
                    print(f"\n=== Step 3.{idx}: Resolve And Append Instruction ===")
                append_started_at = perf_counter()
                anchor_location = _session_search_anchor(
                    current_latitude=current_latitude,
                    current_longitude=current_longitude,
                    session_payload=session_payload,
                )

                append_resolve_started_at = perf_counter()
                append_cache_enabled = normalized_append_cache_mode in {"session", "global"}
                append_cache_namespace = ""
                if normalized_append_cache_mode == "session":
                    append_cache_namespace = f"session:{active_session_id}"
                try:
                    resolved = await resolve_instruction_to_destination(
                        session=session,
                        instruction=instruction,
                        current_latitude=current_latitude,
                        current_longitude=current_longitude,
                        mode=mode,
                        search_radius=search_radius,
                        max_candidates=max_candidates,
                        cache_payload=cache_payload,
                        resolution_scope="session_local_via",
                        anchor_location=anchor_location,
                        cache_enabled=append_cache_enabled,
                        cache_namespace=append_cache_namespace,
                    )
                except RuntimeError as exc:
                    if "No chosen_destination for instruction:" not in str(exc):
                        raise
                    _log_timing(
                        "demo_stage",
                        perf_counter() - append_started_at,
                        stage_name="append_instruction",
                        instruction=instruction,
                        status="skipped_no_candidate",
                    )
                    if not non_interactive:
                        print(
                            json.dumps(
                                {
                                    "instruction": instruction,
                                    "session_changed": False,
                                    "skipped": True,
                                    "reason": "no_candidate",
                                    "error": str(exc),
                                },
                                indent=2,
                                ensure_ascii=True,
                            )
                        )
                    continue
                resolution_path = str(resolved.get("resolution_path") or "session_local_via")
                _log_timing(
                    "demo_stage",
                    perf_counter() - append_resolve_started_at,
                    stage_name="append_resolve_instruction",
                    instruction=instruction,
                    cache_source=resolved.get("cache_source"),
                    resolution_path=resolution_path,
                )

                append_update_started_at = perf_counter()
                if _session_update_is_noop(
                    session_payload=session_payload,
                    resolved_location=resolved["location"],
                ):
                    get_started_at = perf_counter()
                    last_updated = await call_tool_json(
                        session,
                        "get_route_session",
                        {"session_id": active_session_id},
                    )
                    _log_timing(
                        "mcp_tool_call",
                        perf_counter() - get_started_at,
                        tool="get_route_session",
                        status="ok",
                        reason="noop_append",
                    )
                    change_applied = False
                else:
                    add_started_at = perf_counter()
                    last_updated = await call_tool_json(
                        session,
                        "add_via_point_to_session",
                        {
                            "session_id": active_session_id,
                            "new_via_point": {
                                "location": resolved["location"],
                                "name": resolved["name"],
                            },
                        },
                    )
                    _log_timing(
                        "mcp_tool_call",
                        perf_counter() - add_started_at,
                        tool="add_via_point_to_session",
                        status="ok",
                    )
                    change_applied = bool(last_updated.get("session_changed", True))

                session_payload = last_updated.get("session", {}) or session_payload
                _log_timing(
                    "demo_stage",
                    perf_counter() - append_update_started_at,
                    stage_name="append_update_session",
                    instruction=instruction,
                    resolution_path=resolution_path,
                    session_changed=change_applied,
                    change_reason=last_updated.get("change_reason"),
                    route_plan_update_mode=last_updated.get("route_plan_update_mode"),
                )
                _log_timing(
                    "demo_stage",
                    perf_counter() - append_started_at,
                    stage_name="append_instruction",
                    instruction=instruction,
                    cache_source=resolved.get("cache_source"),
                    resolution_path=resolution_path,
                    session_changed=change_applied,
                    change_reason=last_updated.get("change_reason"),
                    route_plan_update_mode=last_updated.get("route_plan_update_mode"),
                )

                if not non_interactive:
                    print(
                        json.dumps(
                            {
                                "instruction": instruction,
                                "resolved_via_name": resolved["name"],
                                "resolved_via_location": resolved["location"],
                                "resolution_path": resolution_path,
                                "session_changed": change_applied,
                                "route_plan_update_mode": last_updated.get("route_plan_update_mode"),
                                "via_points": (last_updated.get("session", {}) or {}).get("via_points", []),
                                "route_summary": (last_updated.get("route_plan", {}) or {}).get("route_summary", {}),
                            },
                            indent=2,
                            ensure_ascii=True,
                        )
                    )

            final_started_at = perf_counter()
            final_state = await call_tool_json(
                session,
                "get_route_session",
                {"session_id": active_session_id},
            )
            _log_timing(
                "mcp_tool_call",
                perf_counter() - final_started_at,
                tool="get_route_session",
                status="ok",
                reason="final_snapshot",
            )
            last_updated = final_state or last_updated

            if not non_interactive:
                print("\n=== Final Session Snapshot ===")
                print(
                    json.dumps(
                        {
                            "session": last_updated.get("session", {}),
                            "route_summary": (last_updated.get("route_plan", {}) or {}).get("route_summary", {}),
                            "stop_sequence": (last_updated.get("route_plan", {}) or {}).get("stop_sequence", []),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )

            if should_access_local_cache:
                _save_json_cache(cache_path, cache_payload)
            _log_timing(
                "demo_total",
                perf_counter() - overall_started_at,
                status="ok",
                append_instruction_count=len(append_instruction or []),
            )

            print("\n---RESULT---")
            print(
                json.dumps(
                    _build_result_payload(
                        active_session_id=active_session_id,
                        resolved_destination_location=resolved_destination_location,
                        last_updated=last_updated,
                        mode=mode,
                    ),
                    ensure_ascii=True,
                )
            )


if __name__ == "__main__":
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Realtime instruction-to-via-point demo: instruction -> destination -> append to route session"
    )
    parser.add_argument("--lat", type=float, required=True, help="Current latitude")
    parser.add_argument("--lon", type=float, required=True, help="Current longitude")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--init-instruction", help="Initial destination instruction; required without --session-id")
    parser.add_argument(
        "--append-instruction",
        action="append",
        default=[],
        help="Additional instruction to resolve and append as via point. Repeat this argument for multiple additions.",
    )
    parser.add_argument("--radius", type=float, default=3000.0, help="Search radius in meters")
    parser.add_argument("--max-candidates", type=int, default=5, help="Max recommendation candidates")
    parser.add_argument(
        "--cache-path",
        default=DEFAULT_CACHE_PATH,
        help="Local JSON cache path for instruction and name-location resolutions",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete the local session demo cache before running for a cold-start measurement",
    )
    parser.add_argument(
        "--append-cache-mode",
        choices=["disabled", "session", "global"],
        default="disabled",
        help=(
            "Local cache behavior for append instructions: disabled skips local cache reads/writes; "
            "session isolates cache by session_id; global uses shared cache."
        ),
    )
    parser.add_argument(
        "--perf-profile",
        choices=["quality", "latency"],
        default="quality",
        help="quality keeps full route behavior; latency uses compact intent parsing and bounded AMap route fallback",
    )
    parser.add_argument(
        "--session-id",
        dest="session_id",
        default=None,
        help="Reuse existing route session ID instead of creating a new one",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Suppress human-readable output, emit only --RESULT-- JSON",
    )
    parser.add_argument(
        "--map-provider",
        choices=["osm", "amap", "baidu", "google"],
        default=None,
        help="Override MAP_PROVIDER for the spawned MCP server",
    )
    parser.add_argument(
        "--amap-key",
        default=None,
        help="Override AMAP_KEY for the spawned MCP server when --map-provider amap is used",
    )
    parser.add_argument(
        "--intent-model",
        default=None,
        help="Override LLM_INTENT_MODEL for intent parsing without changing VLM/visual prompt model settings",
    )
    args = parser.parse_args()

    asyncio.run(
        run_demo(
            current_latitude=args.lat,
            current_longitude=args.lon,
            mode=args.mode,
            init_instruction=args.init_instruction,
            append_instruction=args.append_instruction,
            search_radius=args.radius,
            max_candidates=args.max_candidates,
            cache_path=args.cache_path,
            clear_cache=args.clear_cache,
            append_cache_mode=args.append_cache_mode,
            perf_profile=args.perf_profile,
            session_id=args.session_id,
            non_interactive=args.non_interactive,
            map_provider=args.map_provider,
            amap_key=args.amap_key,
            intent_model=args.intent_model,
        )
    )
