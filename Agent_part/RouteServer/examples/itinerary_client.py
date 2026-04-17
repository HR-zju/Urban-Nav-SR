from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os
import webbrowser


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
    prompt: str,
    latitude: float,
    longitude: float,
    mode: str | None,
    radius: float,
    max_candidates: int,
    render_map: bool,
    open_map: bool,
) -> None:
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            decomposition = await call_tool_json(
                session,
                "decompose_itinerary_goals",
                {
                    "user_instruction": prompt,
                    "mode": mode,
                },
            )
            print("=== Goal Decomposition ===")
            print(json.dumps(decomposition, indent=2, ensure_ascii=True))

            itinerary = await call_tool_json(
                session,
                "recommend_itinerary_and_waypoints",
                {
                    "user_instruction": prompt,
                    "current_latitude": latitude,
                    "current_longitude": longitude,
                    "mode": mode,
                    "search_radius": radius,
                    "max_candidates": max_candidates,
                },
            )
            print("\n=== Itinerary Recommendation ===")
            compact = {
                "provider": itinerary.get("provider"),
                "status": itinerary.get("status"),
                "fulfilled_goal_count": itinerary.get("fulfilled_goal_count"),
                "requested_goal_count": itinerary.get("requested_goal_count"),
                "via_points": itinerary.get("via_points", []),
                "final_destination": itinerary.get("final_destination"),
                "route_summary": ((itinerary.get("route_plan") or {}).get("route_summary") or {}),
                "goals": [
                    {
                        "goal_type": ((item.get("goal") or {}).get("goal_type")),
                        "request_text": ((item.get("goal") or {}).get("request_text")),
                        "status": item.get("status"),
                        "chosen_destination": ((item.get("recommendation") or {}).get("chosen_destination")),
                    }
                    for item in (itinerary.get("goal_recommendations") or [])
                ],
            }
            print(json.dumps(compact, indent=2, ensure_ascii=True))

            if render_map:
                route_plan = itinerary.get("route_plan")
                if not isinstance(route_plan, dict) or not route_plan.get("route_polyline"):
                    print("\n=== Itinerary BEV Snapshot ===")
                    print("No route_plan.route_polyline available; skip rendering.")
                    return

                snapshot = await call_tool_json(
                    session,
                    "render_itinerary_bev_snapshot",
                    {
                        "route_plan": route_plan,
                        "mode": mode or "foot",
                    },
                )
                image_url = str(snapshot.get("image_url_cache_busted") or snapshot.get("image_url") or "").strip()
                print("\n=== Itinerary BEV Snapshot ===")
                print(
                    json.dumps(
                        {
                            "snapshot_provider": snapshot.get("snapshot_provider"),
                            "route_summary": snapshot.get("route_summary", {}),
                            "leg_count": snapshot.get("leg_count"),
                            "image_url": snapshot.get("image_url"),
                            "image_url_cache_busted": snapshot.get("image_url_cache_busted"),
                            "url_length": snapshot.get("url_length"),
                            "rendering_note": snapshot.get("rendering_note"),
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )
                if open_map and image_url:
                    await asyncio.to_thread(webbrowser.open, image_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose a multi-goal instruction and recommend an itinerary")
    parser.add_argument("--prompt", required=True, help="Free-form multi-goal instruction")
    parser.add_argument("--lat", type=float, required=True, help="Current latitude")
    parser.add_argument("--lon", type=float, required=True, help="Current longitude")
    parser.add_argument("--mode", choices=["car", "foot", "bike"], default=None, help="Optional mode override")
    parser.add_argument("--radius", type=float, default=3000.0, help="Search radius in meters")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum candidates per goal")
    parser.add_argument("--render-map", action="store_true", help="Render an AMap static itinerary BEV snapshot")
    parser.add_argument("--open-map", action="store_true", help="Open the rendered AMap snapshot URL in a browser")
    args = parser.parse_args()

    asyncio.run(
        run_demo(
            prompt=args.prompt,
            latitude=args.lat,
            longitude=args.lon,
            mode=args.mode,
            radius=args.radius,
            max_candidates=args.max_candidates,
            render_map=args.render_map or args.open_map,
            open_map=args.open_map,
        )
    )
