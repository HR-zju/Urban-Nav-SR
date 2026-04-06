from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os
from typing import Any


def first_text_content(result: Any) -> str:
    if not result or not getattr(result, "content", None):
        return ""
    for content_item in result.content:
        if content_item.type == "text" and isinstance(content_item.text, str):
            return content_item.text
    return ""


def parse_first_text_payload(result: Any) -> dict[str, Any]:
    text = first_text_content(result)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def destination_location(candidate: dict[str, Any]) -> str | None:
    location = candidate.get("location")
    if isinstance(location, str) and "," in location:
        return location
    coordinates = candidate.get("coordinates", {}) or {}
    latitude = coordinates.get("latitude")
    longitude = coordinates.get("longitude")
    if latitude is None or longitude is None:
        return None
    return f"{longitude},{latitude}"


async def call_tool_json(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
    result = await session.call_tool(tool_name, arguments)
    raw_text = first_text_content(result)
    return parse_first_text_payload(result), raw_text


async def main() -> None:
    parser = argparse.ArgumentParser(description="Poll AMap static BEV route snapshots.")
    parser.add_argument("--prompt", required=True, help="Natural language destination prompt")
    parser.add_argument("--lat", type=float, required=True, help="Current latitude")
    parser.add_argument("--lon", type=float, required=True, help="Current longitude")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--radius", type=int, default=3000, help="Search radius in meters")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum recommendation candidates")
    parser.add_argument("--refresh-interval-s", type=float, default=3.0, help="Snapshot refresh interval")
    parser.add_argument("--iterations", type=int, default=5, help="Number of refresh iterations")
    parser.add_argument("--size", default="800*600", help="Static map size, e.g. 800*600")
    parser.add_argument("--scale", type=int, default=2, help="Static map scale, usually 1 or 2")
    parser.add_argument("--zoom", type=int, default=None, help="Optional AMap static map zoom")
    args = parser.parse_args()

    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            recommendation, raw_text = await call_tool_json(
                session,
                "recommend_destination_and_waypoints",
                {
                    "user_instruction": args.prompt,
                    "current_latitude": args.lat,
                    "current_longitude": args.lon,
                    "mode": args.mode,
                    "max_candidates": args.max_candidates,
                    "search_radius": args.radius,
                },
            )
            if not recommendation:
                raise RuntimeError(f"recommend_destination_and_waypoints returned non-JSON: {raw_text}")

            chosen = recommendation.get("chosen_destination") or {}
            destination = destination_location(chosen)
            if not destination:
                raise RuntimeError("No chosen destination with valid location was returned.")

            origin = f"{args.lon},{args.lat}"
            for index in range(args.iterations):
                snapshot, snapshot_raw = await call_tool_json(
                    session,
                    "render_bev_route_snapshot",
                    {
                        "origin": origin,
                        "destination": destination,
                        "mode": args.mode,
                        "refresh_interval_s": args.refresh_interval_s,
                        "size": args.size,
                        "scale": args.scale,
                        "zoom": args.zoom,
                    },
                )
                if not snapshot:
                    raise RuntimeError(f"render_bev_route_snapshot returned non-JSON: {snapshot_raw}")

                print(f"\n=== BEV Snapshot {index + 1}/{args.iterations} ===")
                print(json.dumps(snapshot, indent=2, ensure_ascii=False))

                if index < args.iterations - 1:
                    await asyncio.sleep(max(0.5, args.refresh_interval_s))


if __name__ == "__main__":
    asyncio.run(main())
