"""
route_session_bridge.py — single-action MCP bridge for route session management.

Usage:
    python route_session_bridge.py --action create \
        --origin "121.4737,31.2304" --destination "121.480,31.235" --mode foot

    python route_session_bridge.py --action add_via \
        --session-id "abc123" \
        --via-location "121.475,31.228" --via-name "便利店"

    python route_session_bridge.py --action replan \
        --session-id "abc123" \
        --origin "121.4737,31.2304"

    python route_session_bridge.py --action resolve_and_add_via \
        --session-id "abc123" \
        --instruction "顺路经过一个便利店" \
        --current-lat 31.2304 --current-lon 121.4737 \
        --origin "121.4737,31.2304" \
        --mode foot --radius 3000 --max-candidates 5

Each action outputs a single JSON object to stdout.
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def first_text_content(result) -> str:
    if not result or not getattr(result, "content", None):
        return ""
    for item in result.content:
        if item.type == "text" and isinstance(item.text, str):
            return item.text
    return ""


def parse_json(result) -> dict:
    text = first_text_content(result)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def call_tool(session: ClientSession, tool_name: str, arguments: dict) -> dict:
    result = await session.call_tool(tool_name, arguments)
    payload = parse_json(result)
    if payload:
        return payload
    raw = first_text_content(result)
    raise RuntimeError(f"{tool_name} returned non-JSON: {raw}")

def as_location_text(chosen: Dict[str, Any]) -> Optional[str]:
    location = str(chosen.get("location") or "").strip()
    if "," in location:
        return location
    coords = chosen.get("coordinates") or {}
    lat = coords.get("latitude")
    lon = coords.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return f"{float(lon)},{float(lat)}"
    return None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

async def action_create(session: ClientSession, args: argparse.Namespace) -> dict:
    """create_route_session: origin → destination, returns session + route_plan."""
    return await call_tool(session, "create_route_session", {
        "session_name": getattr(args, "session_name", "nav_app_session"),
        "origin": args.origin,
        "destination": args.destination,
        "mode": args.mode,
    })


async def action_add_via(session: ClientSession, args: argparse.Namespace) -> dict:
    """add_via_point_to_session: append a resolved via-point to an existing session."""
    return await call_tool(session, "add_via_point_to_session", {
        "session_id": args.session_id,
        "new_via_point": {
            "location": args.via_location,
            "name": args.via_name,
        },
    })


async def action_replan(session: ClientSession, args: argparse.Namespace) -> dict:
    """replan_route_session_from_origin: re-route from a (possibly updated) origin."""
    return await call_tool(session, "replan_route_session_from_origin", {
        "session_id": args.session_id,
        "new_origin": args.origin,
    })


async def action_resolve_and_add_via(session: ClientSession, args: argparse.Namespace) -> dict:
    """
    Compound action used by the app for "顺路经过X" flow:
      1. recommend_destination_and_waypoints → resolve instruction to a location
      2. add_via_point_to_session → add via point (already returns updated route_plan)

    Matches the behavior of realtime_instruction_session_client.py which does NOT call
    replan_route_session_from_origin after adding a via point.
    """
    # Step 1: resolve instruction
    recommend = await call_tool(session, "recommend_destination_and_waypoints", {
        "user_instruction": args.instruction,
        "current_latitude": args.current_lat,
        "current_longitude": args.current_lon,
        "mode": args.mode,
        "search_radius": args.radius,
        "max_candidates": args.max_candidates,
    })

    chosen = recommend.get("chosen_destination") or {}
    if not isinstance(chosen, dict) or not chosen:
        raise RuntimeError(f"No chosen_destination for instruction: {args.instruction!r}")

    location = as_location_text(chosen)
    if not location:
        raise RuntimeError(f"chosen_destination has no valid location: {chosen}")

    via_name = str(chosen.get("name") or args.instruction).strip() or args.instruction

    # Step 2: add via point — already returns complete route_plan with updated polyline
    add_result = await call_tool(session, "add_via_point_to_session", {
        "session_id": args.session_id,
        "new_via_point": {
            "location": location,
            "name": via_name,
        },
    })

    return {
        "resolved_via": {
            "name": via_name,
            "location": location,
            "raw": chosen,
        },
        "add_result": add_result,
        "route_plan": add_result.get("route_plan"),
    }


async def _extract_nav_payload_from_result(
    route_plan: dict,
    args: argparse.Namespace,
) -> dict:
    """
    从已获取的 route_plan 字典中提取导航载荷，构造发送给机器人的 waypoints JSON。
    route_plan 结构: {"coarse_waypoints": [...], ...}
    """
    import datetime
    now = datetime.datetime.now()
    route_id = f"route_{now.strftime('%Y%m%d_%H%M%S')}_{now.strftime('%f')[:3]}"

    coarse_waypoints = route_plan.get("coarse_waypoints", []) or []

    waypoints = []
    for wp in coarse_waypoints:
        location = wp.get("location", "")
        if not location or "," not in location:
            continue
        lon_str, lat_str = location.split(",")
        waypoints.append({
            "waypoint_index": int(wp.get("global_waypoint_index", 0)),
            "longitude": float(lon_str),
            "latitude": float(lat_str),
            "is_terminal": bool(wp.get("is_leg_terminal", False)),
            "is_via_stop": bool(wp.get("is_via_stop", False)),
        })

    return {
        "route_id": route_id,
        "total_count": len(waypoints),
        "waypoints": waypoints,
        "_raw_route_plan": route_plan,
    }


async def _extract_nav_payload(session: ClientSession, args: argparse.Namespace) -> dict:
    """
    调用 create_route_session，从 coarse_waypoints 提取导航数据
    返回: {"route_id": ..., "total_count": N, "waypoints": [...]}
    """
    result = await call_tool(session, "create_route_session", {
        "session_name": getattr(args, "session_name", "nav_app_session"),
        "origin": args.origin,
        "destination": args.destination,
        "mode": args.mode,
    })

    route_plan = result.get("route_plan", {}) or {}
    return await _extract_nav_payload_from_result(route_plan, args)


async def action_send_ws(session: ClientSession, args: argparse.Namespace) -> dict:
    """
    通过 WebSocket 发送导航点到车上。

    两种模式：
    1. --session-id 提供时：从已有 session 取出 route_plan（调用 get_route_session），
       避免重复创建 session 和重新请求路线（零 QPS 消耗）。
    2. 无 --session-id 时：调用 create_route_session 重新规划（向后兼容 CLI 用法）。
    """
    if getattr(args, "session_id", None):
        # 模式 1: 从已有 session 获取 route_plan
        return await action_get_and_send_ws(session, args)

    # 模式 2: 重新创建 session（向后兼容）
    payload = await _extract_nav_payload(session, args)

    if not WEBSOCKETS_AVAILABLE:
        return {
            "status": "error",
            "error": "websockets library not installed. Run: pip install websockets",
            **payload,
        }

    ws_url = f"ws://{args.robot_host}:{args.robot_port}"
    json_str = json.dumps(payload, ensure_ascii=True)

    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json_str)
            ws_status = "sent"
            ws_info = f"ws://{args.robot_host}:{args.robot_port}"
    except Exception as e:
        ws_status = f"failed: {e}"
        ws_info = ws_url

    return {
        "status": ws_status,
        "ws_info": ws_info,
        **payload,
    }


async def action_get_and_send_ws(session: ClientSession, args: argparse.Namespace) -> dict:
    """
    从已有 route session 获取 route_plan，提取 waypoints 后通过 WebSocket 发送到机器人。
    复用缓存的 route_plan，不触发新的路线规划请求。
    """
    if not WEBSOCKETS_AVAILABLE:
        return {
            "status": "error",
            "error": "websockets library not installed. Run: pip install websockets",
        }

    # 调用 get_route_session 获取已有 session 的 route_plan
    result = await call_tool(session, "get_route_session", {
        "session_id": args.session_id,
        "force_replan": False,  # 复用缓存，不消耗 QPS
    })

    session_data = result.get("session", {}) or {}
    route_plan = result.get("route_plan", {}) or {}

    # 提取导航载荷
    payload = await _extract_nav_payload_from_result(route_plan, args)

    # 附加 session 上下文信息
    payload["_session_id"] = args.session_id
    payload["_session_name"] = session_data.get("session_name", "")
    payload["_origin"] = session_data.get("origin", "")
    payload["_destination"] = session_data.get("destination", "")
    payload["_route_plan_cache_hit"] = result.get("route_plan_cache_hit", None)

    # 发送到机器人
    ws_url = f"ws://{args.robot_host}:{args.robot_port}"
    json_str = json.dumps(payload, ensure_ascii=True)

    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json_str)
            ws_status = "sent"
            ws_info = ws_url
    except Exception as e:
        ws_status = f"failed: {e}"
        ws_info = ws_url

    return {
        "status": ws_status,
        "ws_info": ws_info,
        **payload,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ACTIONS = {
    "create": action_create,
    "add_via": action_add_via,
    "replan": action_replan,
    "resolve_and_add_via": action_resolve_and_add_via,
    "send_ws": action_send_ws,
}


async def run(args: argparse.Namespace) -> None:
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            handler = ACTIONS[args.action]
            result = await handler(session, args)
            print(json.dumps(result, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-action MCP bridge for route session management."
    )
    p.add_argument("--action", required=True, choices=list(ACTIONS.keys()),
                   help="Which MCP operation to perform")

    # create / send_ws
    p.add_argument("--origin", help="Origin as 'lon,lat'")
    p.add_argument("--destination", help="Destination as 'lon,lat'")
    p.add_argument("--mode", default="foot", choices=["car", "foot", "bike"])
    p.add_argument("--session-name", default="nav_app_session", dest="session_name")

    # send_ws (WebSocket 发送)
    p.add_argument("--robot-host", default="192.168.1.100", dest="robot_host",
                    help="Robot/WebSocket server IP")
    p.add_argument("--robot-port", type=int, default=8765, dest="robot_port",
                    help="Robot/WebSocket server port")

    # add_via / replan / resolve_and_add_via
    p.add_argument("--session-id", dest="session_id", help="Existing session ID")

    # add_via (direct coordinates)
    p.add_argument("--via-location", dest="via_location", help="Via-point as 'lon,lat'")
    p.add_argument("--via-name", dest="via_name", default="途经点")

    # resolve_and_add_via (natural language)
    p.add_argument("--instruction", help="Natural-language via-point instruction")
    p.add_argument("--current-lat", type=float, dest="current_lat")
    p.add_argument("--current-lon", type=float, dest="current_lon")
    p.add_argument("--radius", type=float, default=3000.0)
    p.add_argument("--max-candidates", type=int, default=5, dest="max_candidates")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(run(args))
