from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os


async def run_demo(prompt: str, latitude: float, longitude: float, mode: str, max_candidates: int, radius: int):
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "recommend_destination_and_waypoints",
                {
                    "user_instruction": prompt,
                    "current_latitude": latitude,
                    "current_longitude": longitude,
                    "mode": mode,
                    "max_candidates": max_candidates,
                    "search_radius": radius,
                },
            )

            payload = {}
            for content_item in result.content:
                if content_item.type == "text":
                    payload = json.loads(content_item.text)
                    break

            print("=== Recommendation Result ===")
            print(json.dumps(payload, indent=2, ensure_ascii=True))
            print(f"Provider: {payload.get('provider', 'unknown')}")

            chosen = payload.get("chosen_destination", {})
            waypoints = payload.get("key_waypoints", [])
            if chosen:
                print("\n=== Summary ===")
                print(f"Chosen: {chosen.get('name', 'Unknown')}")
                route = payload.get("route_summary", {})
                print(f"Distance (m): {route.get('distance_m', 0)}")
                print(f"Duration (min): {route.get('duration_min', 0)}")
                print(f"Waypoints: {len(waypoints)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt-to-destination recommendation demo")
    parser.add_argument("--prompt", required=True, help="Natural language recommendation prompt")
    parser.add_argument("--lat", type=float, required=True, help="Current latitude")
    parser.add_argument("--lon", type=float, required=True, help="Current longitude")
    parser.add_argument("--mode", default="car", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum recommendation candidates")
    parser.add_argument("--radius", type=int, default=3000, help="Search radius in meters")

    args = parser.parse_args()
    asyncio.run(
        run_demo(
            prompt=args.prompt,
            latitude=args.lat,
            longitude=args.lon,
            mode=args.mode,
            max_candidates=args.max_candidates,
            radius=args.radius,
        )
    )
