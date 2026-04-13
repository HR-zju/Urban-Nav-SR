from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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


def _normalize_frame(item: dict[str, Any], index: int) -> dict[str, Any]:
    latitude = item.get("lat", item.get("latitude"))
    longitude = item.get("lng", item.get("longitude", item.get("lon")))
    if latitude is None or longitude is None:
        raise ValueError(f"Trajectory frame {index} is missing lat/lng fields")
    heading_deg = item.get("heading_deg", item.get("heading"))
    return {
        "frame_index": index,
        "timestamp": item.get("timestamp"),
        "gps": {"lat": float(latitude), "lng": float(longitude)},
        "heading_deg": float(heading_deg) if heading_deg is not None else None,
        "image_path": item.get("image_path"),
        "image_url": item.get("image_url"),
        "image_width": item.get("image_width"),
        "image_height": item.get("image_height"),
        "raw": item,
    }


def load_trajectory(path: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {path}. "
            "Pass --trajectory with an existing JSON/JSONL file, or use --lat/--lon for a single-frame capture."
        )

    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Trajectory file is empty: {path}")

    records: list[dict[str, Any]] = []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON trajectory file must contain a list")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Trajectory item {index} is not an object")
            records.append(_normalize_frame(item, index))
        return records

    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise ValueError(f"Trajectory item {index} is not an object")
        records.append(_normalize_frame(item, index))
    if not records:
        raise ValueError(f"No valid trajectory frames found in {path}")
    return records


def build_inline_trajectory(
    *,
    latitude: float,
    longitude: float,
    heading_deg: float | None,
    frame_count: int,
    image_path: str | None,
    image_url: str | None,
    image_width: int | None,
    image_height: int | None,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for index in range(max(int(frame_count), 1)):
        frames.append(
            {
                "frame_index": index,
                "timestamp": None,
                "gps": {"lat": float(latitude), "lng": float(longitude)},
                "heading_deg": float(heading_deg) if heading_deg is not None else None,
                "image_path": image_path,
                "image_url": image_url,
                "image_width": image_width,
                "image_height": image_height,
                "raw": {
                    "lat": float(latitude),
                    "lng": float(longitude),
                    "heading_deg": float(heading_deg) if heading_deg is not None else None,
                    "image_path": image_path,
                    "image_url": image_url,
                    "image_width": image_width,
                    "image_height": image_height,
                },
            }
        )
    return frames


def image_path_to_data_url(path: str | None) -> str | None:
    if not path:
        return None
    input_path = Path(path).expanduser()
    if not input_path.exists() or not input_path.is_file():
        return None
    mime_type, _ = mimetypes.guess_type(input_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    encoded = base64.b64encode(input_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


async def ensure_route_geometry(
    session: ClientSession,
    *,
    recommendation: dict[str, Any],
    origin_latitude: float,
    origin_longitude: float,
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
            "origin": f"{origin_longitude},{origin_latitude}",
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


def build_debug_frame(frame: dict[str, Any], route_context: dict[str, Any]) -> dict[str, Any]:
    pose_match = route_context.get("pose_match", {}) or {}
    progress = route_context.get("progress", {}) or {}
    local_semantics = route_context.get("local_semantics", {}) or {}
    geometry = route_context.get("geometry", {}) or {}
    return {
        "frame_index": frame["frame_index"],
        "timestamp": frame.get("timestamp"),
        "gps": frame["gps"],
        "heading_deg": frame.get("heading_deg"),
        "matched_segment_index": pose_match.get("matched_segment_index"),
        "cross_track_error_m": pose_match.get("cross_track_error_m"),
        "heading_delta_deg": pose_match.get("heading_delta_deg"),
        "distance_to_next_turn_m": progress.get("distance_to_next_turn_m"),
        "turn_type": local_semantics.get("turn_type"),
        "turn_semantics": local_semantics.get("turn_semantics"),
        "corridor_type": local_semantics.get("corridor_type"),
        "off_route": pose_match.get("off_route"),
        "connector_active": geometry.get("connector_active"),
        "local_segment_polyline": geometry.get("local_segment_polyline", []),
        "connector_polyline": geometry.get("connector_polyline", []),
    }


async def run_capture(
    *,
    trajectory_path: str | None,
    latitude: float | None,
    longitude: float | None,
    heading_deg: float | None,
    frame_count: int,
    image_path: str | None,
    image_url: str | None,
    image_width: int,
    image_height: int,
    prompt: str,
    mode: str,
    radius: int,
    max_candidates: int,
    lookahead_m: float,
    output_path: str,
    server_command: str,
    server_args: list[str],
) -> None:
    if trajectory_path:
        frames = load_trajectory(trajectory_path)
    else:
        if latitude is None or longitude is None:
            raise ValueError("Either --trajectory or both --lat/--lon must be provided.")
        frames = build_inline_trajectory(
            latitude=latitude,
            longitude=longitude,
            heading_deg=heading_deg,
            frame_count=frame_count,
            image_path=image_path,
            image_url=image_url,
            image_width=image_width,
            image_height=image_height,
        )
    origin = frames[0]["gps"]

    server_params = StdioServerParameters(
        command=server_command,
        args=server_args,
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
                    "current_latitude": origin["lat"],
                    "current_longitude": origin["lng"],
                    "mode": mode,
                    "max_candidates": max_candidates,
                    "search_radius": radius,
                },
            )

            route_polyline, coarse_waypoints = await ensure_route_geometry(
                session,
                recommendation=recommendation,
                origin_latitude=origin["lat"],
                origin_longitude=origin["lng"],
                mode=mode,
            )

            captured_frames: list[dict[str, Any]] = []
            for frame in frames:
                image_width_px = int(frame.get("image_width") or image_width)
                image_height_px = int(frame.get("image_height") or image_height)
                resolved_image_path = str(frame.get("image_path") or image_path or "").strip() or None
                resolved_image_url = str(frame.get("image_url") or image_url or "").strip() or None
                state_args: dict[str, Any] = {
                    "gps": frame["gps"],
                    "heading_deg": frame.get("heading_deg"),
                    "route_polyline": route_polyline,
                    "coarse_waypoints": coarse_waypoints,
                    "lookahead_m": lookahead_m,
                }
                route_state = await call_tool_json(session, "get_route_state", state_args)
                route_context = route_state.get("route_context", {}) or {}
                visual_prompt = None
                if resolved_image_path or resolved_image_url:
                    visual_prompt = await call_tool_json(
                        session,
                        "run_zero_shot_visual_prompt_agent",
                        {
                            **state_args,
                            "image_path": resolved_image_path,
                            "image_url": resolved_image_url,
                            "visual_top_k": 5,
                        },
                    )
                captured_frames.append(
                    {
                        "frame_index": frame["frame_index"],
                        "timestamp": frame.get("timestamp"),
                        "gps": frame["gps"],
                        "heading_deg": frame.get("heading_deg"),
                        "image_path": resolved_image_path,
                        "image_url": resolved_image_url,
                        "image_data_url": image_path_to_data_url(resolved_image_path),
                        "image_width": image_width_px,
                        "image_height": image_height_px,
                        "debug": build_debug_frame(frame, route_context),
                        "route_context": route_context,
                        "local_bev_context": route_state.get("local_bev_context", {}),
                        "visual_prompt_decision": (visual_prompt or {}).get("arbiter_decision", {}) or {},
                        "visual_prompt_overlay_svg_data_url": (visual_prompt or {}).get("overlay_svg_data_url"),
                        "visual_prompt_overlay_svg": (visual_prompt or {}).get("overlay_svg"),
                        "prompt_generation_tool": "run_zero_shot_visual_prompt_agent" if visual_prompt else None,
                    }
                )

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "prompt": prompt,
            "mode": mode,
            "radius": radius,
            "max_candidates": max_candidates,
            "lookahead_m": lookahead_m,
            "trajectory_path": trajectory_path,
            "inline_capture": trajectory_path is None,
            "image_path": image_path,
            "image_url": image_url,
            "image_width": image_width,
            "image_height": image_height,
            "provider": recommendation.get("provider"),
            "parsed_intent": recommendation.get("parsed_intent", {}),
            "chosen_destination": recommendation.get("chosen_destination", {}),
            "route_summary": recommendation.get("route_summary", {}),
            "route_polyline": route_polyline,
            "coarse_waypoints": coarse_waypoints,
        },
        "frames": captured_frames,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote RouteContext capture to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RouteContext frame sequences from a trajectory file")
    parser.add_argument("--trajectory", help="Path to JSON or JSONL trajectory file")
    parser.add_argument("--lat", type=float, help="Inline capture latitude when no trajectory file is provided")
    parser.add_argument("--lon", type=float, help="Inline capture longitude when no trajectory file is provided")
    parser.add_argument("--heading-deg", type=float, default=None, help="Inline capture heading in degrees")
    parser.add_argument("--frame-count", type=int, default=1, help="Inline capture frame count for repeated single-pose snapshots")
    parser.add_argument("--image-path", help="Shared local image path for inline capture or default frame image")
    parser.add_argument("--image-url", help="Shared image URL for inline capture or default frame image")
    parser.add_argument("--image-width", type=int, default=1280, help="Target RGB width for PoV projection")
    parser.add_argument("--image-height", type=int, default=720, help="Target RGB height for PoV projection")
    parser.add_argument("--prompt", required=True, help="Natural language destination prompt")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--radius", type=int, default=3000, help="Search radius in meters")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum recommendation candidates")
    parser.add_argument("--lookahead-m", type=float, default=20.0, help="Local segment lookahead distance in meters")
    parser.add_argument("--output", required=True, help="Output JSON path for captured frames")
    parser.add_argument("--server-command", default="osm-mcp-server", help="MCP server command")
    parser.add_argument("--server-arg", action="append", default=[], help="Additional server args")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_capture(
            trajectory_path=args.trajectory,
            latitude=args.lat,
            longitude=args.lon,
            heading_deg=args.heading_deg,
            frame_count=args.frame_count,
            image_path=args.image_path,
            image_url=args.image_url,
            image_width=args.image_width,
            image_height=args.image_height,
            prompt=args.prompt,
            mode=args.mode,
            radius=args.radius,
            max_candidates=args.max_candidates,
            lookahead_m=args.lookahead_m,
            output_path=args.output,
            server_command=args.server_command,
            server_args=args.server_arg,
        )
    )
