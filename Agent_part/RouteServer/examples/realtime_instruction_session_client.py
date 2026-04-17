from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional


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
    raise RuntimeError(f"{tool_name} returned non-JSON payload: {raw_text}")


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
) -> Dict[str, Any]:
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
        },
    )
    chosen = result.get("chosen_destination") or {}
    if not isinstance(chosen, dict) or not chosen:
        raise RuntimeError(f"No chosen_destination for instruction: {instruction}")
    location = as_location_text(chosen)
    if not location:
        raise RuntimeError(f"chosen_destination has no valid location: {chosen}")
    return {
        "name": str(chosen.get("name") or instruction).strip() or instruction,
        "location": location,
        "raw": chosen,
        "recommend_result": result,
    }


async def run_demo(
    *,
    current_latitude: float,
    current_longitude: float,
    mode: str,
    init_instruction: str,
    append_instruction: list[str],
    search_radius: float,
    max_candidates: int,
) -> None:
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=== Step 1: Resolve Initial Destination Instruction ===")
            init_resolved = await resolve_instruction_to_destination(
                session=session,
                instruction=init_instruction,
                current_latitude=current_latitude,
                current_longitude=current_longitude,
                mode=mode,
                search_radius=search_radius,
                max_candidates=max_candidates,
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
                resolved = await resolve_instruction_to_destination(
                    session=session,
                    instruction=instruction,
                    current_latitude=current_latitude,
                    current_longitude=current_longitude,
                    mode=mode,
                    search_radius=search_radius,
                    max_candidates=max_candidates,
                )
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
                print(
                    json.dumps(
                        {
                            "instruction": instruction,
                            "resolved_via_name": resolved["name"],
                            "resolved_via_location": resolved["location"],
                            "via_points": (updated.get("session", {}) or {}).get("via_points", []),
                            "route_summary": (updated.get("route_plan", {}) or {}).get("route_summary", {}),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )

            final_state = await call_tool_json(
                session,
                "get_route_session",
                {"session_id": session_id},
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


if __name__ == "__main__":
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
        )
    )
