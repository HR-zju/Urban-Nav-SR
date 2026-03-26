from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os


def first_text_content(result):
    if not result or not getattr(result, "content", None):
        return ""
    for content_item in result.content:
        if content_item.type == "text" and isinstance(content_item.text, str):
            return content_item.text
    return ""


def parse_first_text_payload(result):
    text = first_text_content(result)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def extract_coordinates(payload):
    coordinates = (payload or {}).get("coordinates", {}) or {}
    latitude = coordinates.get("latitude")
    longitude = coordinates.get("longitude")
    if latitude is None or longitude is None:
        return None
    return float(latitude), float(longitude)


async def run_demo(
    prompt: str,
    mode: str,
    max_candidates: int,
    radius: int,
    timeout_s: float,
    fallback_lat: float | None,
    fallback_lon: float | None,
):
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            location_result = await session.call_tool(
                "get_current_location_macos",
                {"timeout_s": timeout_s},
            )
            raw_location_text = first_text_content(location_result)
            location_payload = parse_first_text_payload(location_result)
            coords = extract_coordinates(location_payload)

            if coords is None:
                if fallback_lat is not None and fallback_lon is not None:
                    latitude, longitude = float(fallback_lat), float(fallback_lon)
                    print("=== Current Location ===")
                    print("macOS location unavailable, using CLI fallback coordinates.")
                    print(f"Fallback coordinates: {latitude}, {longitude}")
                    if raw_location_text:
                        print(f"Location tool message: {raw_location_text}")
                else:
                    detail = raw_location_text or json.dumps(location_payload, ensure_ascii=True)
                    raise RuntimeError(
                        "Failed to read coordinates from get_current_location_macos. "
                        f"Tool response: {detail}. "
                        "Run this on macOS with Location Services enabled, or provide --lat/--lon fallback."
                    )
            else:
                latitude, longitude = coords
                print("=== Current macOS Location ===")
                print(json.dumps(location_payload, indent=2, ensure_ascii=True))


            recommendation_result = await session.call_tool(
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
            payload = parse_first_text_payload(recommendation_result)

            print("\n=== Recommendation Result ===")
            print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="macOS current-location recommendation demo")
    parser.add_argument("--prompt", required=True, help="Natural language recommendation prompt")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum recommendation candidates")
    parser.add_argument("--radius", type=int, default=3000, help="Search radius in meters")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="Wait time for CoreLocation fix")
    parser.add_argument("--lat", type=float, default=None, help="Optional fallback latitude")
    parser.add_argument("--lon", type=float, default=None, help="Optional fallback longitude")

    args = parser.parse_args()
    if (args.lat is None) != (args.lon is None):
        raise ValueError("Use --lat and --lon together when providing fallback coordinates.")

    asyncio.run(
        run_demo(
            prompt=args.prompt,
            mode=args.mode,
            max_candidates=args.max_candidates,
            radius=args.radius,
            timeout_s=args.timeout_s,
            fallback_lat=args.lat,
            fallback_lon=args.lon,
        )
    )
