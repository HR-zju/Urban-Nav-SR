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
) -> tuple[list[Any], list[dict[str, Any]]]:
    route_polyline = recommendation.get("route_polyline") or []
    coarse_waypoints = recommendation.get("coarse_waypoints") or recommendation.get("key_waypoints") or []
    if route_polyline and coarse_waypoints:
        return route_polyline, coarse_waypoints

    chosen_destination = recommendation.get("chosen_destination", {}) or {}
    destination = destination_location(chosen_destination)
    if not destination:
        raise RuntimeError("Recommendation has no route geometry and no usable chosen destination location")

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
    if not route_polyline or not coarse_waypoints:
        raise RuntimeError("No usable route geometry after fallback get_coarse_waypoints")
    return route_polyline, coarse_waypoints


async def run_demo(
    *,
    prompt: str,
    latitude: float,
    longitude: float,
    mode: str,
    radius: int,
    max_candidates: int,
    heading_deg: float | None,
    lookahead_m: float,
    image_path: str | None,
    image_url: str | None,
    visual_top_k: int,
    width: int,
    height: int,
    open_svg: bool,
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

            common_args = {
                "gps": {"lat": latitude, "lng": longitude},
                "heading_deg": heading_deg,
                "route_polyline": route_polyline,
                "coarse_waypoints": coarse_waypoints,
                "lookahead_m": lookahead_m,
            }

            route_context = await call_tool_json(session, "get_route_context", common_args)
            bev_patch = await call_tool_json(
                session,
                "render_local_bev_patch",
                {
                    **common_args,
                    "width": width,
                    "height": height,
                },
            )
            bev_proposals = await call_tool_json(session, "generate_bev_prompt_proposals", common_args)

            pov_v2 = None
            if image_path or image_url:
                pov_v2 = await call_tool_json(
                    session,
                    "generate_pov_prompt_candidates_v2",
                    {
                        **common_args,
                        "image_path": image_path,
                        "image_url": image_url,
                        "visual_top_k": visual_top_k,
                    },
                )

            print("=== Route Context Debug Summary ===")
            print(
                json.dumps(
                    {
                        "provider": recommendation.get("provider"),
                        "chosen_destination": recommendation.get("chosen_destination", {}),
                        "route_summary": recommendation.get("route_summary", {}),
                        "turn_semantics": (route_context.get("local_semantics", {}) or {}).get("turn_semantics"),
                        "corridor_type": (route_context.get("local_semantics", {}) or {}).get("corridor_type"),
                        "distance_to_next_turn_m": (route_context.get("progress", {}) or {}).get("distance_to_next_turn_m"),
                        "bev_proposal_count": len(bev_proposals.get("prompt_proposals", []) or []),
                        "pov_candidate_count": len((pov_v2 or {}).get("pov_prompt_candidates", []) or []),
                        "svg_data_url": bev_patch.get("svg_data_url"),
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )

            print("\n=== RouteContext ===")
            print(json.dumps(route_context, indent=2, ensure_ascii=True))
            print("\n=== BEV Prompt Proposals ===")
            print(json.dumps(bev_proposals.get("prompt_proposals", []), indent=2, ensure_ascii=True))

            if pov_v2 is not None:
                print("\n=== Visual Regions ===")
                print(json.dumps((pov_v2.get("visual_regions", {}) or {}).get("candidate_regions", []), indent=2, ensure_ascii=True))
                print("\n=== PoV Prompt Candidates v2 ===")
                print(json.dumps(pov_v2.get("pov_prompt_candidates", []), indent=2, ensure_ascii=True))

            if open_svg:
                svg_data_url = str(bev_patch.get("svg_data_url") or "").strip()
                if svg_data_url:
                    await asyncio.to_thread(webbrowser.open, svg_data_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug RouteContext, LocalBEV, BEV proposals, and optional RGB-aware PoV candidates")
    parser.add_argument("--prompt", required=True, help="Natural language destination prompt")
    parser.add_argument("--lat", type=float, required=True, help="Current latitude")
    parser.add_argument("--lon", type=float, required=True, help="Current longitude")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--radius", type=int, default=3000, help="Search radius in meters")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum recommendation candidates")
    parser.add_argument("--heading-deg", type=float, default=None, help="Optional current heading in degrees")
    parser.add_argument("--lookahead-m", type=float, default=20.0, help="Local segment lookahead distance in meters")
    parser.add_argument("--image-path", help="Optional local RGB/FPV image path for PromptProposal v2")
    parser.add_argument("--image-url", help="Optional image URL or data URL for PromptProposal v2")
    parser.add_argument("--visual-top-k", type=int, default=5, help="Maximum visual candidate regions")
    parser.add_argument("--width", type=int, default=640, help="Rendered BEV SVG width")
    parser.add_argument("--height", type=int, default=640, help="Rendered BEV SVG height")
    parser.add_argument("--open", action="store_true", help="Open the rendered BEV SVG data URL")
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
            image_path=args.image_path,
            image_url=args.image_url,
            visual_top_k=args.visual_top_k,
            width=args.width,
            height=args.height,
            open_svg=args.open,
        )
    )
