from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any


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


def destination_location(candidate: dict) -> str | None:
    location = candidate.get("location")
    if isinstance(location, str) and "," in location:
        return location
    coordinates = candidate.get("coordinates", {}) or {}
    latitude = coordinates.get("latitude")
    longitude = coordinates.get("longitude")
    if latitude is None or longitude is None:
        return None
    return f"{longitude},{latitude}"


def polyline_from_coarse_waypoints(coarse_waypoints: list[dict]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for item in coarse_waypoints or []:
        coordinate = item.get("coordinate", {}) or {}
        latitude = coordinate.get("latitude", item.get("latitude"))
        longitude = coordinate.get("longitude", item.get("longitude"))
        if latitude is None or longitude is None:
            continue
        point = {"longitude": float(longitude), "latitude": float(latitude)}
        if points and abs(points[-1]["latitude"] - point["latitude"]) < 1e-9 and abs(points[-1]["longitude"] - point["longitude"]) < 1e-9:
            continue
        points.append(point)
    return points


async def ensure_route_geometry(session: ClientSession, *, recommendation: dict, latitude: float, longitude: float, mode: str):
    route_polyline = recommendation.get("route_polyline") or []
    coarse_waypoints = recommendation.get("coarse_waypoints") or recommendation.get("key_waypoints") or []
    if route_polyline and coarse_waypoints:
        return route_polyline, coarse_waypoints

    chosen_destination = recommendation.get("chosen_destination", {}) or {}
    destination = destination_location(chosen_destination)
    if not destination:
        raise RuntimeError("No usable destination location for fallback routing")

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
    fpv_segmentation: dict[str, Any] | None,
    visual_top_k: int,
    candidate_top_k: int,
    llm_detail: str,
    emit_qa_bundle: bool,
    save_overlay: str | None,
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

            payload = await call_tool_json(
                session,
                "run_zero_shot_visual_prompt_agent",
                {
                    "gps": {"lat": latitude, "lng": longitude},
                    "heading_deg": heading_deg,
                    "route_polyline": route_polyline,
                    "coarse_waypoints": coarse_waypoints,
                    "lookahead_m": lookahead_m,
                    "image_path": image_path,
                    "image_url": image_url,
                    "fpv_segmentation": fpv_segmentation,
                    "visual_top_k": visual_top_k,
                    "candidate_top_k": candidate_top_k,
                    "llm_detail": llm_detail,
                    "emit_qa_bundle": emit_qa_bundle,
                },
            )

            print("=== Zero-shot Visual Prompt Agent ===")
            print(
                json.dumps(
                    {
                        "provider": recommendation.get("provider"),
                        "chosen_destination": recommendation.get("chosen_destination", {}),
                        "arbiter_decision": payload.get("arbiter_decision", {}),
                        "fpv_segmentation_summary": payload.get("fpv_segmentation_summary"),
                        "qa_bundle_summary": (
                            ((payload.get("arbiter_decision", {}) or {}).get("debug", {}) or {}).get("qa_bundle_summary")
                        ),
                        "route_context": {
                            "turn_semantics": ((payload.get("route_context", {}) or {}).get("local_semantics", {}) or {}).get("turn_semantics"),
                            "corridor_type": ((payload.get("route_context", {}) or {}).get("local_semantics", {}) or {}).get("corridor_type"),
                            "off_route": ((payload.get("route_context", {}) or {}).get("pose_match", {}) or {}).get("off_route"),
                        },
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )

            overlay_svg = payload.get("overlay_svg")
            if save_overlay and isinstance(overlay_svg, str) and overlay_svg.strip():
                output_path = Path(save_overlay)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(overlay_svg, encoding="utf-8")
                print(f"\nSaved overlay SVG to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot visual prompt agent over one FPV frame")
    parser.add_argument("--prompt", required=True, help="Natural language destination prompt")
    parser.add_argument("--lat", type=float, required=True, help="Current latitude")
    parser.add_argument("--lon", type=float, required=True, help="Current longitude")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--radius", type=int, default=3000, help="Search radius in meters")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum recommendation candidates")
    parser.add_argument("--heading-deg", type=float, default=None, help="Optional current heading in degrees")
    parser.add_argument("--lookahead-m", type=float, default=20.0, help="Local segment lookahead distance in meters")
    parser.add_argument("--image-path", help="Local RGB/FPV image path")
    parser.add_argument("--image-url", help="Image URL or data URL")
    parser.add_argument("--fpv-segmentation-npy", help="Optional SAM3 segmentation .npy path aligned with the FPV frame")
    parser.add_argument("--fpv-segmentation-class-mapping", help="Optional SAM3 class_mapping.json path")
    parser.add_argument("--fpv-segmentation-visualization", help="Optional SAM3 visualization image path")
    parser.add_argument("--visual-top-k", type=int, default=5, help="Maximum visual regions")
    parser.add_argument("--candidate-top-k", type=int, default=5, help="Maximum PoV prompt candidates sent to the arbiter")
    parser.add_argument("--llm-detail", default="auto", choices=["auto", "low", "high"], help="Vision detail level")
    parser.add_argument("--emit-qa-bundle", action="store_true", help="Include structured QA teacher samples in the result")
    parser.add_argument("--save-overlay", help="Optional output path for the rendered prompt SVG overlay")

    args = parser.parse_args()
    if not args.image_path and not args.image_url:
        parser.error("one of --image-path or --image-url is required")

    fpv_segmentation = None
    if args.fpv_segmentation_npy:
        mapping_path = args.fpv_segmentation_class_mapping
        if not mapping_path:
            candidate = Path(args.fpv_segmentation_npy).expanduser().with_name("class_mapping.json")
            if candidate.exists():
                mapping_path = str(candidate)
        fpv_segmentation = {
            "segmentation_npy_path": str(Path(args.fpv_segmentation_npy).expanduser()),
            "class_mapping_path": str(Path(mapping_path).expanduser()) if mapping_path else None,
            "visualization_path": str(Path(args.fpv_segmentation_visualization).expanduser()) if args.fpv_segmentation_visualization else None,
        }

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
            fpv_segmentation=fpv_segmentation,
            visual_top_k=args.visual_top_k,
            candidate_top_k=args.candidate_top_k,
            llm_detail=args.llm_detail,
            emit_qa_bundle=args.emit_qa_bundle,
            save_overlay=args.save_overlay,
        )
    )
