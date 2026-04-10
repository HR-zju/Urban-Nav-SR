from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os
from pathlib import Path


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


def default_test_context() -> dict:
    return {
        "pose_match": {
            "off_route": False,
            "route_confidence": 0.46,
            "cross_track_error_m": 8.0,
            "heading_delta_deg": 4.0,
        },
        "progress": {
            "distance_to_next_turn_m": 18.0,
            "next_turn_type": "go_straight",
        },
        "local_semantics": {
            "turn_type": "go_straight",
            "turn_semantics": "approach_route_start",
            "corridor_type": "front",
            "short_action_summary": "Proceed along the visible sidewalk corridor toward the route start.",
            "instruction_hint": "Pedestrian navigation should stay on the sidewalk and avoid motor lanes.",
            "road_name": "",
        },
        "geometry": {
            "lookahead_m": 20.0,
        },
    }


async def run_demo(
    *,
    image_path: str | None,
    image_url: str | None,
    context_path: str | None,
    navigation_mode: str,
    llm_detail: str,
    save_overlay: str | None,
) -> None:
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    if context_path:
        route_context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    else:
        route_context = default_test_context()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            payload = await call_tool_json(
                session,
                "run_visual_prompt_only_from_context",
                {
                    "route_context": route_context,
                    "image_path": image_path,
                    "image_url": image_url,
                    "llm_detail": llm_detail,
                    "navigation_mode": navigation_mode,
                },
            )
            print("=== Visual Prompt Only ===")
            print(
                json.dumps(
                    {
                        "route_context": payload.get("route_context", {}),
                        "prompt_proposals": payload.get("prompt_proposals", []),
                        "decision": payload.get("decision", {}),
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
    parser = argparse.ArgumentParser(description="Run visual prompt generation directly from a test route context")
    parser.add_argument("--image-path", help="Local RGB/FPV image path")
    parser.add_argument("--image-url", help="Image URL or data URL")
    parser.add_argument("--context-path", help="Optional JSON file containing a route_context-like object")
    parser.add_argument("--navigation-mode", default="foot", choices=["car", "foot", "bike"], help="Navigation mode constraints")
    parser.add_argument("--llm-detail", default="auto", choices=["auto", "low", "high"], help="Vision detail level")
    parser.add_argument("--save-overlay", help="Optional output path for the rendered prompt SVG overlay")
    args = parser.parse_args()

    if not args.image_path and not args.image_url:
        parser.error("one of --image-path or --image-url is required")

    asyncio.run(
        run_demo(
            image_path=args.image_path,
            image_url=args.image_url,
            context_path=args.context_path,
            navigation_mode=args.navigation_mode,
            llm_detail=args.llm_detail,
            save_overlay=args.save_overlay,
        )
    )
