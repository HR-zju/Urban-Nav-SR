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
    add_via: list[str],
    first_via: str | None,
    second_via: str | None,
    replan_origin: str | None,
) -> None:
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            created = await call_tool_json(
                session,
                "create_route_session",
                {
                    "session_name": "demo_route_session",
                    "origin": origin,
                    "destination": destination,
                    "mode": mode,
                },
            )
            session_payload = created.get("session", {})
            route_plan = created.get("route_plan", {})
            session_id = session_payload.get("session_id")

            print("=== Created Route Session ===")
            print(
                json.dumps(
                    {
                        "session": session_payload,
                        "route_summary": route_plan.get("route_summary", {}),
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )

            if not session_id:
                return

            via_points_to_add = list(add_via or [])
            if first_via:
                via_points_to_add.append(first_via)
            if second_via:
                via_points_to_add.append(second_via)

            for via_index, via_location in enumerate(via_points_to_add, start=1):
                updated = await call_tool_json(
                    session,
                    "add_via_point_to_session",
                    {
                        "session_id": session_id,
                        "new_via_point": {"location": via_location, "name": f"Stop {via_index}"},
                    },
                )
                print(f"\n=== After Adding Via Point {via_index} ===")
                print(
                    json.dumps(
                        {
                            "added_via": {"location": via_location, "name": f"Stop {via_index}"},
                            "via_points": (updated.get("session", {}) or {}).get("via_points", []),
                            "route_summary": (updated.get("route_plan", {}) or {}).get("route_summary", {}),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )

            # Legacy demo path: keep the old named flags functional for users who
            # relied on them before --add-via was introduced.
            if via_points_to_add:
                first_via = None
                second_via = None

            if first_via:
                updated = await call_tool_json(
                    session,
                    "add_via_point_to_session",
                    {
                        "session_id": session_id,
                        "new_via_point": {"location": first_via, "name": "First stop"},
                    },
                )
                print("\n=== After Adding First Via Point ===")
                print(
                    json.dumps(
                        {
                            "via_points": (updated.get("session", {}) or {}).get("via_points", []),
                            "route_summary": (updated.get("route_plan", {}) or {}).get("route_summary", {}),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )

            if second_via:
                updated = await call_tool_json(
                    session,
                    "add_via_point_to_session",
                    {
                        "session_id": session_id,
                        "new_via_point": {"location": second_via, "name": "Second stop"},
                        "insert_before_index": 0,
                    },
                )
                print("\n=== After Inserting Second Via Point At Index 0 ===")
                print(
                    json.dumps(
                        {
                            "via_points": (updated.get("session", {}) or {}).get("via_points", []),
                            "route_summary": (updated.get("route_plan", {}) or {}).get("route_summary", {}),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )

            fetched = await call_tool_json(
                session,
                "get_route_session",
                {
                    "session_id": session_id,
                },
            )
            print("\n=== Fetched Session Snapshot ===")
            print(
                json.dumps(
                    {
                        "session": fetched.get("session", {}),
                        "leg_count": ((fetched.get("route_plan", {}) or {}).get("route_summary", {}) or {}).get("leg_count"),
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )

            if second_via:
                removed = await call_tool_json(
                    session,
                    "remove_via_point_from_session",
                    {
                        "session_id": session_id,
                        "via_point_name": "Second stop",
                    },
                )
                print("\n=== After Removing Second Via Point ===")
                print(
                    json.dumps(
                        {
                            "removed_via_point": removed.get("removed_via_point", {}),
                            "via_points": (removed.get("session", {}) or {}).get("via_points", []),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )

            if replan_origin:
                replanned = await call_tool_json(
                    session,
                    "replan_route_session_from_origin",
                    {
                        "session_id": session_id,
                        "origin": replan_origin,
                    },
                )
                print("\n=== After Replanning From New Origin ===")
                print(
                    json.dumps(
                        {
                            "origin": (replanned.get("session", {}) or {}).get("origin"),
                            "route_summary": (replanned.get("route_plan", {}) or {}).get("route_summary", {}),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate interactive route sessions with via points")
    parser.add_argument("--origin", required=True, help="Origin in 'longitude,latitude' format")
    parser.add_argument("--destination", required=True, help="Destination in 'longitude,latitude' format")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument(
        "--add-via",
        action="append",
        default=[],
        help="Via point to append in 'longitude,latitude' format. Can be provided multiple times.",
    )
    parser.add_argument("--first-via", help="First via point to append")
    parser.add_argument("--second-via", help="Second via point to insert at index 0")
    parser.add_argument("--replan-origin", help="Optional new origin to replan the session from")
    args = parser.parse_args()

    asyncio.run(
        run_demo(
            origin=args.origin,
            destination=args.destination,
            mode=args.mode,
            add_via=args.add_via,
            first_via=args.first_via,
            second_via=args.second_via,
            replan_origin=args.replan_origin,
        )
    )
