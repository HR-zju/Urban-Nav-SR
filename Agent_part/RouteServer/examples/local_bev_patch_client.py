from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os
from typing import Any
import webbrowser


def first_text_content(result) -> str:
    if not result or not getattr(result, "content", None):
        return ""
    for content_item in result.content:
        if content_item.type == "text" and isinstance(content_item.text, str):
            return content_item.text
    return ""


def parse_first_text_payload(result) -> dict[str, Any]:
    text = first_text_content(result)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def call_tool_json(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await session.call_tool(tool_name, arguments)
    payload = parse_first_text_payload(result)
    if payload:
        return payload
    raw_text = first_text_content(result)
    raise RuntimeError(f"{tool_name} returned non-JSON payload: {raw_text}")


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


def polyline_from_coarse_waypoints(coarse_waypoints: list[dict[str, Any]]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for item in coarse_waypoints or []:
        coordinate = item.get("coordinate", {}) or {}
        latitude = coordinate.get("latitude", item.get("latitude"))
        longitude = coordinate.get("longitude", item.get("longitude"))
        if latitude is None or longitude is None:
            location = item.get("location")
            if isinstance(location, str) and "," in location:
                try:
                    lon_text, lat_text = str(location).split(",", 1)
                    longitude = float(lon_text)
                    latitude = float(lat_text)
                except Exception:
                    latitude = None
                    longitude = None
        if latitude is None or longitude is None:
            continue
        point = {"longitude": float(longitude), "latitude": float(latitude)}
        if points and abs(points[-1]["latitude"] - point["latitude"]) < 1e-9 and abs(points[-1]["longitude"] - point["longitude"]) < 1e-9:
            continue
        points.append(point)
    return points


async def ensure_route_geometry(
    session: ClientSession,
    *,
    recommendation: dict[str, Any],
    latitude: float,
    longitude: float,
    mode: str,
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    route_polyline = recommendation.get("route_polyline") or []
    coarse_waypoints = recommendation.get("coarse_waypoints") or recommendation.get("key_waypoints") or []
    if route_polyline and coarse_waypoints:
        return route_polyline, coarse_waypoints

    chosen_destination = recommendation.get("chosen_destination", {}) or {}
    destination = destination_location(chosen_destination)
    if not destination:
        raise RuntimeError(
            "recommend_destination_and_waypoints did not return usable route geometry, "
            "and chosen_destination has no usable location"
        )

    route_payload = await call_tool_json(
        session,
        "get_coarse_waypoints",
        {
            "origin": f"{longitude},{latitude}",
            "destination": destination,
            "mode": mode,
        },
    )
    route_polyline = route_payload.get("route_polyline") or []
    coarse_waypoints = route_payload.get("coarse_waypoints") or []
    if not route_polyline and coarse_waypoints:
        route_polyline = polyline_from_coarse_waypoints(coarse_waypoints)

    if not route_polyline:
        raise RuntimeError("No route_polyline available after fallback get_coarse_waypoints")
    if not coarse_waypoints:
        raise RuntimeError("No coarse_waypoints available after fallback get_coarse_waypoints")
    return route_polyline, coarse_waypoints


async def run_demo(
    prompt: str,
    latitude: float,
    longitude: float,
    mode: str,
    radius: int,
    max_candidates: int,
    heading_deg: float | None,
    lookahead_m: float,
    width: int,
    height: int,
    open_patch: bool,
) -> None:
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            recommendation = await call_tool_json(
                session,
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

            route_polyline, coarse_waypoints = await ensure_route_geometry(
                session,
                recommendation=recommendation,
                latitude=latitude,
                longitude=longitude,
                mode=mode,
            )

            patch_payload = await call_tool_json(
                session,
                "render_local_bev_patch",
                {
                    "gps": {"lat": latitude, "lng": longitude},
                    "heading_deg": heading_deg,
                    "route_polyline": route_polyline,
                    "coarse_waypoints": coarse_waypoints,
                    "lookahead_m": lookahead_m,
                    "width": width,
                    "height": height,
                },
            )

            summary = {
                "provider": recommendation.get("provider"),
                "chosen_destination": recommendation.get("chosen_destination", {}),
                "turn_semantics": ((patch_payload.get("local_bev_context", {}) or {}).get("turn_semantics")),
                "corridor_type": ((patch_payload.get("local_bev_context", {}) or {}).get("corridor_type")),
                "svg_data_url": patch_payload.get("svg_data_url"),
            }
            print("=== LocalBEV Patch Summary ===")
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            print("\n=== LocalBEVContext ===")
            print(json.dumps(patch_payload.get("local_bev_context", {}), indent=2, ensure_ascii=True))

            if open_patch:
                svg_data_url = str(patch_payload.get("svg_data_url") or "").strip()
                if svg_data_url:
                    await asyncio.to_thread(webbrowser.open, svg_data_url)
                else:
                    print("\nNo svg_data_url available to open.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a local pose-aligned BEV patch from LocalBEVContext")
    parser.add_argument("--prompt", required=True, help="Natural language destination prompt")
    parser.add_argument("--lat", type=float, required=True, help="Current latitude")
    parser.add_argument("--lon", type=float, required=True, help="Current longitude")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--radius", type=int, default=3000, help="Search radius in meters")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum recommendation candidates")
    parser.add_argument("--heading-deg", type=float, default=None, help="Optional current heading in degrees")
    parser.add_argument("--lookahead-m", type=float, default=20.0, help="Local segment lookahead distance in meters")
    parser.add_argument("--width", type=int, default=640, help="Rendered SVG width")
    parser.add_argument("--height", type=int, default=640, help="Rendered SVG height")
    parser.add_argument("--open", action="store_true", help="Open the rendered SVG data URL in the default browser")

    args = parser.parse_args()
    asyncio.run(
        run_demo(
            prompt=args.prompt,
            latitude=args.lat,
            longitude=args.lon,
            mode=args.mode,
            radius=args.radius,
            max_candidates=args.max_candidates,
            heading_deg=args.heading_deg,
            lookahead_m=args.lookahead_m,
            width=args.width,
            height=args.height,
            open_patch=args.open,
        )
    )
