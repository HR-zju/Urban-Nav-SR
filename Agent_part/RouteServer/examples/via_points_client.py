from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os


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


async def run_demo(
    *,
    origin: str,
    destination: str,
    mode: str,
    via_locations: list[str],
) -> None:
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            via_points = [{"location": item, "name": f"Via {index + 1}"} for index, item in enumerate(via_locations)]
            base = await call_tool_json(
                session,
                "plan_route_with_via_points",
                {
                    "origin": origin,
                    "destination": destination,
                    "via_points": via_points,
                    "mode": mode,
                },
            )
            print("=== Multi-leg Route ===")
            print(
                json.dumps(
                    {
                        "provider": base.get("provider"),
                        "mode": base.get("mode"),
                        "route_summary": base.get("route_summary", {}),
                        "via_points": base.get("via_points", []),
                        "legs": [
                            {
                                "leg_index": item.get("leg_index"),
                                "origin": item.get("origin"),
                                "destination": item.get("destination"),
                                "route_summary": item.get("route_summary", {}),
                            }
                            for item in (base.get("legs", []) or [])
                        ],
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )

            if via_locations:
                appended = await call_tool_json(
                    session,
                    "add_via_point_to_route_plan",
                    {
                        "origin": origin,
                        "destination": destination,
                        "existing_via_points": via_points,
                        "new_via_point": {"location": via_locations[0], "name": "Inserted copy"},
                        "insert_before_index": 0,
                        "mode": mode,
                    },
                )
                print("\n=== After Inserting One Via Point At Index 0 ===")
                print(
                    json.dumps(
                        {
                            "via_points": appended.get("via_points", []),
                            "route_summary": appended.get("route_summary", {}),
                            "leg_count": (appended.get("route_summary", {}) or {}).get("leg_count"),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan a route with via points and test incremental insertion")
    parser.add_argument("--origin", required=True, help="Origin in 'longitude,latitude' format")
    parser.add_argument("--destination", required=True, help="Destination in 'longitude,latitude' format")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--via", action="append", default=[], help="Via point location in 'longitude,latitude' format; repeat for multiple stops")
    args = parser.parse_args()

    asyncio.run(
        run_demo(
            origin=args.origin,
            destination=args.destination,
            mode=args.mode,
            via_locations=args.via,
        )
    )
