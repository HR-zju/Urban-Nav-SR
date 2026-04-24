from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp_server.map_providers import (
    dedupe_route_coordinates,
    encode_polyline_text,
    format_location,
    parse_location_text,
)


def _coerce_location_text(value: Any) -> str:
    if isinstance(value, str) and "," in value:
        parse_location_text(value)
        return value.strip()
    if isinstance(value, dict):
        if isinstance(value.get("location"), str) and "," in str(value.get("location")):
            location = str(value.get("location")).strip()
            parse_location_text(location)
            return location
        coordinate = value.get("coordinate", {}) or {}
        longitude = coordinate.get("longitude", value.get("longitude"))
        latitude = coordinate.get("latitude", value.get("latitude"))
        if longitude is not None and latitude is not None:
            return format_location(float(longitude), float(latitude))
    raise ValueError(f"Unsupported location payload: {value!r}")


def _normalize_via_points(via_points: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(via_points, list):
        return normalized
    for index, item in enumerate(via_points):
        try:
            location = _coerce_location_text(item)
        except Exception:
            continue
        name = None
        if isinstance(item, dict):
            raw_name = str(item.get("name") or item.get("label") or "").strip()
            name = raw_name or None
        normalized.append(
            {
                "index": len(normalized),
                "location": location,
                "name": name or f"Via {index + 1}",
            }
        )
    return normalized


def _locations_equivalent(a: Any, b: Any, *, tol: float = 1e-6) -> bool:
    try:
        a_text = _coerce_location_text(a)
        b_text = _coerce_location_text(b)
        a_lon, a_lat = parse_location_text(a_text)
        b_lon, b_lat = parse_location_text(b_text)
        return abs(a_lon - b_lon) <= tol and abs(a_lat - b_lat) <= tol
    except Exception:
        return str(a or "").strip() == str(b or "").strip()


def _merge_route_segments(
    *,
    provider: str,
    origin: str,
    destination: str,
    mode: str,
    via_points: List[Dict[str, Any]],
    leg_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    combined_polyline: List[List[float]] = []
    combined_waypoints: List[Dict[str, Any]] = []
    total_distance_m = 0.0
    total_duration_s = 0.0

    stop_sequence: List[Dict[str, Any]] = [{"type": "origin", "location": origin, "name": "Origin"}]

    for leg_index, leg in enumerate(leg_results):
        leg_summary = dict(leg.get("route_summary", {}) or {})
        total_distance_m += float(leg_summary.get("distance_m", 0.0) or 0.0)
        total_duration_s += float(leg_summary.get("duration_s", 0.0) or 0.0)

        if leg_index < len(via_points):
            via = via_points[leg_index]
            stop_sequence.append(
                {
                    "type": "via",
                    "location": via.get("location") or leg.get("destination"),
                    "name": via.get("name") or f"Via {leg_index + 1}",
                    "route_leg_summary": {
                        "distance_m": round(float(leg_summary.get("distance_m", 0.0) or 0.0), 1),
                        "duration_s": round(float(leg_summary.get("duration_s", 0.0) or 0.0), 1),
                        "duration_min": round(float(leg_summary.get("duration_s", 0.0) or 0.0) / 60.0, 1),
                    },
                }
            )
        else:
            final_destination = leg.get("destination") or destination
            stop_sequence.append(
                {
                    "type": "destination",
                    "location": final_destination,
                    "name": "Destination",
                    "route_leg_summary": {
                        "distance_m": round(float(leg_summary.get("distance_m", 0.0) or 0.0), 1),
                        "duration_s": round(float(leg_summary.get("duration_s", 0.0) or 0.0), 1),
                        "duration_min": round(float(leg_summary.get("duration_s", 0.0) or 0.0) / 60.0, 1),
                    },
                }
            )

    global_waypoint_index = 0
    legs_payload: List[Dict[str, Any]] = []
    for leg_index, leg in enumerate(leg_results):
        leg_origin = leg.get("origin")
        leg_destination = leg.get("destination")

        leg_polyline = []
        for point in leg.get("route_polyline", []) or []:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                leg_polyline.append([float(point[0]), float(point[1])])
            elif isinstance(point, dict):
                longitude = point.get("longitude", point.get("lng", point.get("lon")))
                latitude = point.get("latitude", point.get("lat"))
                if longitude is None or latitude is None:
                    continue
                leg_polyline.append([float(longitude), float(latitude)])
        combined_polyline.extend(leg_polyline)

        leg_waypoints = []
        raw_waypoints = leg.get("coarse_waypoints", []) or []
        for waypoint_index, waypoint in enumerate(raw_waypoints):
            payload = dict(waypoint)
            payload["leg_index"] = leg_index
            payload["waypoint_index_in_leg"] = waypoint_index
            payload["global_waypoint_index"] = global_waypoint_index
            payload["leg_origin"] = leg_origin
            payload["leg_destination"] = leg_destination
            payload["is_leg_terminal"] = waypoint_index == (len(raw_waypoints) - 1)
            payload["is_via_stop"] = payload["is_leg_terminal"] and leg_index < (len(leg_results) - 1)
            if payload["is_via_stop"]:
                payload["via_point_index"] = leg_index
                payload["via_point_name"] = (
                    via_points[leg_index].get("name") if leg_index < len(via_points) else f"Via {leg_index + 1}"
                )
            global_waypoint_index += 1
            combined_waypoints.append(payload)
            leg_waypoints.append(payload)

        legs_payload.append(
            {
                "leg_index": leg_index,
                "origin": leg_origin,
                "destination": leg_destination,
                "route_summary": leg_summary,
                "coarse_waypoints": leg_waypoints,
                "route_polyline": [{"longitude": point[0], "latitude": point[1]} for point in leg_polyline],
            }
        )

    combined_polyline = dedupe_route_coordinates(combined_polyline)
    return {
        "provider": provider,
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "via_points": via_points,
        "stop_sequence": stop_sequence,
        "route_summary": {
            "distance_m": round(total_distance_m, 1),
            "duration_s": round(total_duration_s, 1),
            "duration_min": round(total_duration_s / 60.0, 1),
            "step_count": len(combined_waypoints),
            "leg_count": len(leg_results),
        },
        "coarse_waypoints": combined_waypoints,
        "route_polyline": [{"longitude": point[0], "latitude": point[1]} for point in combined_polyline],
        "route_polyline_text": encode_polyline_text(combined_polyline),
        "legs": legs_payload,
    }


def _route_stop_locations(
    *,
    origin: str,
    destination: str,
    via_points: List[Dict[str, Any]],
) -> List[str]:
    return [origin] + [str(item.get("location") or "").strip() for item in via_points] + [destination]


def _leg_matches_stop_pair(leg: Dict[str, Any], start: str, end: str) -> bool:
    try:
        return _locations_equivalent(leg.get("origin"), start) and _locations_equivalent(leg.get("destination"), end)
    except Exception:
        return False


def _cached_legs_match_stops(
    *,
    legs: List[Dict[str, Any]],
    stops: List[str],
) -> bool:
    if len(legs) != max(0, len(stops) - 1):
        return False
    for index, leg in enumerate(legs):
        if not isinstance(leg, dict):
            return False
        if not _leg_matches_stop_pair(leg, stops[index], stops[index + 1]):
            return False
    return True


async def _compute_incremental_route_plan_with_inserted_via(
    *,
    app_context: Any,
    origin: str,
    destination: str,
    mode: str,
    cached_route_plan: Dict[str, Any],
    old_via_points: List[Dict[str, Any]],
    new_via_points: List[Dict[str, Any]],
    insert_index: int,
) -> Optional[Dict[str, Any]]:
    cached_legs = cached_route_plan.get("legs") or []
    if not isinstance(cached_legs, list):
        return None

    old_stops = _route_stop_locations(origin=origin, destination=destination, via_points=old_via_points)
    if not _cached_legs_match_stops(legs=cached_legs, stops=old_stops):
        return None

    if insert_index < 0 or insert_index > len(old_via_points):
        return None

    split_leg_index = insert_index
    if split_leg_index >= len(cached_legs):
        return None

    previous_stop = old_stops[split_leg_index]
    next_stop = old_stops[split_leg_index + 1]
    inserted_via = new_via_points[insert_index]
    inserted_location = str(inserted_via.get("location") or "").strip()
    if not inserted_location:
        return None

    first_new_leg = await app_context.osm_client.get_coarse_waypoints(
        origin=previous_stop,
        destination=inserted_location,
        mode=mode,
    )
    second_new_leg = await app_context.osm_client.get_coarse_waypoints(
        origin=inserted_location,
        destination=next_stop,
        mode=mode,
    )

    reused_prefix = cached_legs[:split_leg_index]
    reused_suffix = cached_legs[split_leg_index + 1 :]
    merged_legs = reused_prefix + [first_new_leg, second_new_leg] + reused_suffix
    route_plan = _merge_route_segments(
        provider=app_context.provider,
        origin=origin,
        destination=destination,
        mode=mode,
        via_points=new_via_points,
        leg_results=merged_legs,
    )
    route_plan["replan_strategy"] = "incremental_insert_via_split_leg"
    route_plan["incremental_replan"] = {
        "reused_prefix_leg_count": len(reused_prefix),
        "reused_suffix_leg_count": len(reused_suffix),
        "recomputed_leg_count": 2,
        "split_leg_index": split_leg_index,
    }
    return route_plan


async def _compute_combined_route_plan(
    *,
    app_context: Any,
    origin: str,
    destination: str,
    via_points: Optional[List[Dict[str, Any]]] = None,
    mode: str = "foot",
) -> Dict[str, Any]:
    normalized_via_points = _normalize_via_points(via_points)
    stops = [origin] + [item["location"] for item in normalized_via_points] + [destination]
    leg_results: List[Dict[str, Any]] = []
    for leg_index in range(len(stops) - 1):
        leg_origin = stops[leg_index]
        leg_destination = stops[leg_index + 1]
        leg = await app_context.osm_client.get_coarse_waypoints(
            origin=leg_origin,
            destination=leg_destination,
            mode=mode,
        )
        leg_results.append(leg)
    return _merge_route_segments(
        provider=app_context.provider,
        origin=origin,
        destination=destination,
        mode=mode,
        via_points=normalized_via_points,
        leg_results=leg_results,
    )


ROUTE_SESSIONS: Dict[str, Dict[str, Any]] = {}
ROUTE_SESSIONS_LOADED = False


def _route_session_store_backend() -> str:
    return str(os.getenv("ROUTE_SESSION_STORE", "file") or "file").strip().lower()


def _route_session_store_path() -> Path:
    configured = str(os.getenv("ROUTE_SESSION_STORE_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".urban_nav" / "route_sessions.json"


def _load_route_sessions() -> None:
    global ROUTE_SESSIONS_LOADED
    if ROUTE_SESSIONS_LOADED:
        return
    ROUTE_SESSIONS_LOADED = True
    if _route_session_store_backend() == "memory":
        return

    path = _route_session_store_path()
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    sessions = payload.get("sessions") if isinstance(payload, dict) else payload
    if not isinstance(sessions, dict):
        return
    for session_id, session in sessions.items():
        if isinstance(session, dict) and str(session.get("session_id") or session_id).strip():
            normalized_id = str(session.get("session_id") or session_id).strip()
            session["session_id"] = normalized_id
            ROUTE_SESSIONS[normalized_id] = session


def _persist_route_sessions() -> None:
    if _route_session_store_backend() == "memory":
        return

    path = _route_session_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": datetime.now().isoformat(),
        "sessions": ROUTE_SESSIONS,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _get_persisted_route_session(session_id: str) -> Optional[Dict[str, Any]]:
    _load_route_sessions()
    session = ROUTE_SESSIONS.get(str(session_id or "").strip())
    if isinstance(session, dict):
        return session
    return None


def _save_persisted_route_session(session: Dict[str, Any]) -> Dict[str, Any]:
    _load_route_sessions()
    session_id = str(session.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("Route session payload is missing session_id")
    ROUTE_SESSIONS[session_id] = session
    _persist_route_sessions()
    return session


def _cache_route_plan_on_session(session: Dict[str, Any], route_plan: Dict[str, Any]) -> None:
    if not isinstance(route_plan, dict):
        return
    session["route_plan_cache"] = route_plan
    session["route_plan_cached_at"] = datetime.now().isoformat()


async def _route_plan_for_session(
    *,
    app_context: Any,
    session: Dict[str, Any],
    force_replan: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    cached = session.get("route_plan_cache")
    if isinstance(cached, dict) and cached and not force_replan:
        return cached, True

    route_plan = await _compute_combined_route_plan(
        app_context=app_context,
        origin=session["origin"],
        destination=session["destination"],
        via_points=session.get("via_points"),
        mode=session["mode"],
    )
    _cache_route_plan_on_session(session, route_plan)
    _save_persisted_route_session(session)
    return route_plan, False


def _route_session_store_info() -> Dict[str, Any]:
    backend = _route_session_store_backend()
    info: Dict[str, Any] = {"backend": backend}
    if backend != "memory":
        info["path"] = str(_route_session_store_path())
    return info
