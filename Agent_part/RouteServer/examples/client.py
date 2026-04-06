from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import json
import os


def parse_first_text_payload(result):
    if not result or not result.content:
        return {}
    for content_item in result.content:
        if content_item.type == "text":
            try:
                return json.loads(content_item.text)
            except json.JSONDecodeError:
                continue
    return {}


async def main():
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            response = await session.list_tools()
            tools = response.tools
            print(f"Available tools: {[tool.name for tool in tools]}")

            print("\n--- Example 1: Geocode a Location ---")

            location_result = await session.call_tool(
                "geocode_address",
                {"address": "San Francisco"},
            )

            locations = []
            for content_item in location_result.content:
                if content_item.type == "text":
                    try:
                        locations.append(json.loads(content_item.text))
                    except json.JSONDecodeError:
                        continue

            if locations:
                print(f"Found place: {locations[0].get('display_name', 'Unknown')}")
                lat = float(locations[0].get('lat', 0))
                lon = float(locations[0].get('lon', 0))
                print(f"Coordinates: {lat}, {lon}")

                location_text = f"{lon},{lat}"

                print("\n--- Example 2: Semantic POI Search ---")
                poi_result = await session.call_tool(
                    "search_nearby_pois",
                    {
                        "location": location_text,
                        "keywords": "restaurant|cafe",
                        "radius": 800,
                        "sortrule": "distance",
                        "limit": 5,
                    },
                )
                poi_info = parse_first_text_payload(poi_result)
                poi_candidates = poi_info.get("results", [])
                print(f"Server provider: {poi_info.get('provider', 'unknown')}")
                print(f"Found {poi_info.get('count', 0)} semantic POI candidates")

                for candidate in poi_candidates[:3]:
                    print(
                        f"- {candidate.get('name', 'Unknown')} "
                        f"({candidate.get('type', 'unknown')}) "
                        f"{candidate.get('distance_m', 0)}m"
                    )

                if poi_candidates:
                    destination = poi_candidates[0]
                    print("\n--- Example 3: Coarse Waypoints ---")
                    coarse_result = await session.call_tool(
                        "get_coarse_waypoints",
                        {
                            "origin": location_text,
                            "destination": destination.get("location"),
                            "mode": "foot",
                        },
                    )
                    coarse_info = parse_first_text_payload(coarse_result)
                    summary = coarse_info.get("route_summary", {})
                    print(f"Route distance: {summary.get('distance_m', 0)} m")
                    print(f"Route duration: {summary.get('duration_s', 0)} s")
                    print(f"Coarse waypoints: {len(coarse_info.get('coarse_waypoints', []))}")

                print("\n--- Example 4: Recommendation from Natural Language ---")
                recommend_result = await session.call_tool(
                    "recommend_destination_and_waypoints",
                    {
                        "user_instruction": "Find a quiet cafe nearby with convenient parking",
                        "current_latitude": lat,
                        "current_longitude": lon,
                        "max_candidates": 3,
                        "search_radius": 2500,
                        "mode": "car",
                    },
                )
                recommend_info = parse_first_text_payload(recommend_result)
                if recommend_info:
                    print(f"Recommendation provider: {recommend_info.get('provider', 'unknown')}")

                chosen = recommend_info.get("chosen_destination", {})
                if chosen:
                    print(f"Chosen destination: {chosen.get('name', 'Unknown')}")
                    route_summary = recommend_info.get("route_summary", {})
                    print(f"Distance: {route_summary.get('distance_m', 0)} m")
                    print(f"Duration: {route_summary.get('duration_min', 0)} min")
                    print(f"Key waypoints: {len(recommend_info.get('key_waypoints', []))}")


if __name__ == "__main__":
    asyncio.run(main())
