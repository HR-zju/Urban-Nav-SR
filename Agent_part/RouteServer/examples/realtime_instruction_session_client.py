from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import hashlib
import json
import logging
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


def first_text_content(result) -> str:
    if not result or not getattr(result, "content", None):
        return ""
    for content_item in result.content:
        if content_item.type == "text" and isinstance(content_item.text, str):
            return content_item.text
    return ""


def parse_first_text_payload(result) -> dict:
    text = first_text_content(result)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def call_tool_json(session: ClientSession, tool_name: str, arguments: dict) -> dict:
    result = await session.call_tool(tool_name, arguments)
    payload = parse_first_text_payload(result)
    if payload:
        return payload
    raw_text = first_text_content(result)
    if raw_text.startswith("Error executing tool "):
        raise RuntimeError(raw_text)
    raise RuntimeError(f"{tool_name} returned non-JSON payload: {raw_text}")


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
) -> str:
    return _stable_cache_key(
        [
            "instruction_resolution",
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


def _name_location_cache_key(name: str) -> str:
    return _stable_cache_key(["name_location", _normalize_text(name)])


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
) -> Dict[str, Any]:
    instruction_cache = (cache_payload or {}).get("instruction_results", {}) if isinstance(cache_payload, dict) else {}
    name_cache = (cache_payload or {}).get("name_locations", {}) if isinstance(cache_payload, dict) else {}
    name_cached = name_cache.get(_name_location_cache_key(instruction)) if isinstance(name_cache, dict) else None
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

    started_at = perf_counter()
    result = await call_tool_json(
        session,
        "recommend_destination_and_waypoints",
        {
            "user_instruction": instruction,
            "current_latitude": current_latitude,
            "current_longitude": current_longitude,
            "mode": mode,
            "search_radius": search_radius,
            "max_candidates": max_candidates,
            "resolution_scope": resolution_scope,
            "anchor_location": anchor_location,
        },
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
        raise RuntimeError(f"No chosen_destination for instruction: {instruction}")
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
    if isinstance(instruction_cache, dict):
        instruction_cache[cache_key] = resolved
    name_key = _name_location_cache_key(resolved["name"])
    if isinstance(name_cache, dict):
        name_cache[name_key] = {
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


async def run_demo(
    *,
    current_latitude: float,
    current_longitude: float,
    mode: str,
    init_instruction: str,
    append_instruction: list[str],
    search_radius: float,
    max_candidates: int,
    cache_path: str,
    clear_cache: bool,
) -> None:
    overall_started_at = perf_counter()
    if clear_cache:
        try:
            Path(cache_path).expanduser().unlink(missing_ok=True)
        except Exception:
            pass
    cache_payload = _load_json_cache(cache_path)
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

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
            )
            _log_timing(
                "demo_stage",
                perf_counter() - init_started_at,
                stage_name="resolve_initial_instruction",
                cache_source=init_resolved.get("cache_source"),
            )
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
            created = await call_tool_json(
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

            session_payload = created.get("session", {}) or {}
            session_id = str(session_payload.get("session_id") or "").strip()
            if not session_id:
                raise RuntimeError("create_route_session returned empty session_id")

            print("\n=== Step 2: Create Session ===")
            print(
                json.dumps(
                    {
                        "session_id": session_id,
                        "origin": session_payload.get("origin"),
                        "destination": session_payload.get("destination"),
                        "mode": session_payload.get("mode"),
                        "via_points": session_payload.get("via_points", []),
                        "route_summary": (created.get("route_plan", {}) or {}).get("route_summary", {}),
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )

            # Incrementally add via points by natural-language instructions.
            for idx, instruction in enumerate(append_instruction or [], start=1):
                print(f"\n=== Step 3.{idx}: Resolve And Append Instruction ===")
                append_started_at = perf_counter()
                anchor_location = _session_search_anchor(
                    current_latitude=current_latitude,
                    current_longitude=current_longitude,
                    session_payload=session_payload,
                )
                append_resolve_started_at = perf_counter()
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
                )
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
                    get_session_started_at = perf_counter()
                    updated = await call_tool_json(
                        session,
                        "get_route_session",
                        {"session_id": session_id},
                    )
                    _log_timing(
                        "mcp_tool_call",
                        perf_counter() - get_session_started_at,
                        tool="get_route_session",
                        status="ok",
                        reason="noop_append",
                    )
                    change_applied = False
                else:
                    add_via_started_at = perf_counter()
                    updated = await call_tool_json(
                        session,
                        "add_via_point_to_session",
                        {
                            "session_id": session_id,
                            "new_via_point": {
                                "location": resolved["location"],
                                "name": resolved["name"],
                            },
                        },
                    )
                    _log_timing(
                        "mcp_tool_call",
                        perf_counter() - add_via_started_at,
                        tool="add_via_point_to_session",
                        status="ok",
                    )
                    change_applied = bool(updated.get("session_changed", True))
                session_payload = updated.get("session", {}) or session_payload
                _log_timing(
                    "demo_stage",
                    perf_counter() - append_update_started_at,
                    stage_name="append_update_session",
                    instruction=instruction,
                    resolution_path=resolution_path,
                    session_changed=change_applied,
                    change_reason=updated.get("change_reason"),
                    route_plan_update_mode=updated.get("route_plan_update_mode"),
                )
                _log_timing(
                    "demo_stage",
                    perf_counter() - append_started_at,
                    stage_name="append_instruction",
                    instruction=instruction,
                    cache_source=resolved.get("cache_source"),
                    resolution_path=resolution_path,
                    session_changed=change_applied,
                    change_reason=updated.get("change_reason"),
                    route_plan_update_mode=updated.get("route_plan_update_mode"),
                )
                print(
                    json.dumps(
                        {
                            "instruction": instruction,
                            "resolved_via_name": resolved["name"],
                            "resolved_via_location": resolved["location"],
                            "resolution_path": resolution_path,
                            "session_changed": change_applied,
                            "route_plan_update_mode": updated.get("route_plan_update_mode"),
                            "via_points": (updated.get("session", {}) or {}).get("via_points", []),
                            "route_summary": (updated.get("route_plan", {}) or {}).get("route_summary", {}),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )

            final_started_at = perf_counter()
            final_state = await call_tool_json(
                session,
                "get_route_session",
                {"session_id": session_id},
            )
            _log_timing(
                "mcp_tool_call",
                perf_counter() - final_started_at,
                tool="get_route_session",
                status="ok",
                reason="final_snapshot",
            )
            print("\n=== Final Session Snapshot ===")
            print(
                json.dumps(
                    {
                        "session": final_state.get("session", {}),
                        "route_summary": (final_state.get("route_plan", {}) or {}).get("route_summary", {}),
                        "stop_sequence": (final_state.get("route_plan", {}) or {}).get("stop_sequence", []),
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )
            _save_json_cache(cache_path, cache_payload)
            _log_timing(
                "demo_total",
                perf_counter() - overall_started_at,
                status="ok",
                append_instruction_count=len(append_instruction or []),
            )


if __name__ == "__main__":
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Realtime instruction-to-via-point demo: instruction -> destination -> append to route session"
    )
    parser.add_argument("--lat", type=float, required=True, help="Current latitude")
    parser.add_argument("--lon", type=float, required=True, help="Current longitude")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--init-instruction", required=True, help="Initial destination instruction")
    parser.add_argument(
        "--append-instruction",
        action="append",
        default=[],
        help="Additional instruction to resolve and append as via point. Repeat this argument for multiple additions.",
    )
    parser.add_argument("--radius", type=float, default=3000.0, help="Search radius in meters")
    parser.add_argument("--max-candidates", type=int, default=5, help="Max recommendation candidates")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Local JSON cache path for instruction and name-location resolutions")
    parser.add_argument("--clear-cache", action="store_true", help="Delete the local session demo cache before running for a cold-start measurement")
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
        )
    )
