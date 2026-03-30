from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os
from typing import Any

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


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


def print_candidates(payload: dict[str, Any]) -> None:
    recommendations = payload.get("recommendations", []) or []
    if not recommendations:
        print("No candidate destinations returned.")
        return

    print("\n=== Candidate Destinations ===")
    for index, item in enumerate(recommendations, start=1):
        summary = item.get("route_summary", {}) or {}
        tags = item.get("tags", {}) or {}
        print(
            f"{index}. {item.get('name', 'Unknown')} | "
            f"score={parse_float(item.get('score'), 0):.4f} | "
            f"distance={parse_float(summary.get('distance_m'), 0):.0f}m | "
            f"duration={parse_float(summary.get('duration_min'), 0):.1f}min | "
            f"type={tags.get('type', 'unknown')}"
        )


def build_waypoint_sequence(route_payload: dict[str, Any]) -> list[dict[str, float]]:
    waypoints = route_payload.get("coarse_waypoints", []) or []
    sequence = []
    for item in waypoints:
        coordinate = item.get("coordinate", {}) or {}
        latitude = coordinate.get("latitude")
        longitude = coordinate.get("longitude")
        if latitude is None or longitude is None:
            location_text = item.get("location", "")
            try:
                lon_text, lat_text = str(location_text).split(",", 1)
                longitude = float(lon_text)
                latitude = float(lat_text)
            except Exception:
                continue
        sequence.append({"longitude": float(longitude), "latitude": float(latitude)})
    return sequence


def input_or_keep(prompt: str, current: str) -> str:
    entered = input(f"{prompt} [{current}]: ").strip()
    return entered or current


def input_int_or_keep(prompt: str, current: int) -> int:
    entered = input(f"{prompt} [{current}]: ").strip()
    if not entered:
        return current
    try:
        return int(entered)
    except ValueError:
        print(f"Invalid number: {entered}. Keeping {current}.")
        return current


def input_mode_or_keep(current: str) -> str:
    entered = input(f"New mode (car/foot/bike) [{current}]: ").strip().lower()
    if not entered:
        return current
    if entered not in {"car", "foot", "bike"}:
        print(f"Invalid mode: {entered}. Keeping {current}.")
        return current
    return entered


async def call_tool_json(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
    result = await session.call_tool(tool_name, arguments)
    raw_text = first_text_content(result)
    return parse_first_text_payload(result), raw_text


async def generate_candidate_destinations(
    session: ClientSession,
    prompt: str,
    latitude: float,
    longitude: float,
    mode: str,
    max_candidates: int,
    radius: int,
) -> tuple[dict[str, Any], str]:
    return await call_tool_json(
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


async def run_demo(
    prompt: str,
    mode: str,
    max_candidates: int,
    radius: int,
    timeout_s: float,
    fallback_lat: float | None,
    fallback_lon: float | None,
    non_interactive: bool,
    selection_index: int,
    json_only: bool,
):
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # In app-driven flow, CLI coordinates are already from user/device interaction.
            if fallback_lat is not None and fallback_lon is not None:
                latitude, longitude = float(fallback_lat), float(fallback_lon)
                if not json_only:
                    print("=== Current Location ===")
                    print("Using coordinates passed from caller.")
                    print(f"Coordinates: {latitude}, {longitude}")
            else:
                location_payload, raw_location_text = await call_tool_json(
                    session,
                    "get_current_location_macos",
                    {"timeout_s": timeout_s},
                )
                coords = extract_coordinates(location_payload)

                if coords is None:
                    detail = raw_location_text or json.dumps(location_payload, ensure_ascii=True)
                    raise RuntimeError(
                        "Failed to read coordinates from get_current_location_macos. "
                        f"Tool response: {detail}. "
                        "Run this on macOS with Location Services enabled, or provide --lat/--lon."
                    )

                latitude, longitude = coords
                if not json_only:
                    print("=== Current macOS Location ===")
                    print(json.dumps(location_payload, indent=2, ensure_ascii=True))

            current_prompt = prompt
            current_mode = mode
            current_radius = radius
            current_max_candidates = max_candidates

            while True:
                if not json_only:
                    print("\n=== Candidate Generation Request ===")
                    print(
                        json.dumps(
                            {
                                "prompt": current_prompt,
                                "mode": current_mode,
                                "radius": current_radius,
                                "max_candidates": current_max_candidates,
                                "current_location": {"latitude": latitude, "longitude": longitude},
                            },
                            indent=2,
                            ensure_ascii=True,
                        )
                    )

                payload, raw_text = await generate_candidate_destinations(
                    session,
                    prompt=current_prompt,
                    latitude=latitude,
                    longitude=longitude,
                    mode=current_mode,
                    max_candidates=current_max_candidates,
                    radius=current_radius,
                )
                if not payload:
                    if json_only:
                        result = {
                            "provider": None,
                            "recommendations": [],
                            "chosen_destination": None,
                            "key_waypoints": [],
                            "route_summary": None,
                            "message": f"Recommendation tool returned non-JSON payload: {raw_text}",
                        }
                        print(json.dumps(result, ensure_ascii=True))
                    else:
                        print(f"Recommendation tool returned non-JSON payload: {raw_text}")
                    return

                if not json_only:
                    print("\n=== Recommendation Result ===")
                    print(json.dumps(payload, indent=2, ensure_ascii=True))
                    print_candidates(payload)

                recommendations = payload.get("recommendations", []) or []
                if not recommendations:
                    if non_interactive:
                        result = {
                            "provider": payload.get("provider"),
                            "recommendations": [],
                            "chosen_destination": None,
                            "key_waypoints": [],
                            "route_summary": None,
                            "message": "No candidate destinations returned.",
                        }
                        print(json.dumps(result, ensure_ascii=True))
                        return
                    command = input("No candidates. Enter `p` to edit prompt, `r` radius, `m` mode, `q` quit: ").strip().lower()
                    if command == "p":
                        current_prompt = input_or_keep("New prompt", current_prompt)
                    elif command == "r":
                        current_radius = input_int_or_keep("New radius (meters)", current_radius)
                    elif command == "m":
                        current_mode = input_mode_or_keep(current_mode)
                    elif command == "q":
                        return
                    continue

                if non_interactive:
                    selected_index = selection_index
                else:
                    command = input(
                        "Choose destination index, or `p` edit prompt, `r` radius, `m` mode, `c` max candidates, `g` regenerate, `q` quit: "
                    ).strip().lower()

                    if command == "q":
                        return
                    if command == "p":
                        current_prompt = input_or_keep("New prompt", current_prompt)
                        continue
                    if command == "r":
                        current_radius = input_int_or_keep("New radius (meters)", current_radius)
                        continue
                    if command == "m":
                        current_mode = input_mode_or_keep(current_mode)
                        continue
                    if command == "c":
                        current_max_candidates = input_int_or_keep("New max candidates", current_max_candidates)
                        continue
                    if command == "g" or command == "":
                        continue

                    try:
                        selected_index = int(command)
                    except ValueError:
                        print("Invalid input. Please enter an index or a command.")
                        continue

                if selected_index < 1 or selected_index > len(recommendations):
                    if non_interactive:
                        selected_index = 1
                    else:
                        print(f"Invalid index: {selected_index}")
                        continue

                selected = recommendations[selected_index - 1]
                selected_location = destination_location(selected)
                if not selected_location:
                    if non_interactive:
                        result = {
                            "provider": payload.get("provider"),
                            "recommendations": recommendations,
                            "chosen_destination": None,
                            "key_waypoints": [],
                            "route_summary": None,
                            "message": "Selected destination has no valid location field.",
                        }
                        print(json.dumps(result, ensure_ascii=True))
                        return
                    print("Selected destination has no valid location field.")
                    continue

                reused_recommended_route = (
                    selected_index == 1
                    and isinstance(payload.get("route_summary"), dict)
                    and isinstance(payload.get("key_waypoints"), list)
                    and bool(payload.get("route_summary"))
                )

                if reused_recommended_route:
                    final_payload = {
                        "provider": payload.get("provider"),
                        "recommendations": recommendations,
                        "chosen_destination": selected,
                        "key_waypoints": payload.get("key_waypoints", []),
                        "route_summary": payload.get("route_summary", {}),
                        "message": None,
                    }
                else:
                    origin_location = f"{longitude},{latitude}"
                    route_payload, route_raw_text = await call_tool_json(
                        session,
                        "get_coarse_waypoints",
                        {
                            "origin": origin_location,
                            "destination": selected_location,
                            "mode": current_mode,
                        },
                    )
                    if not route_payload:
                        if non_interactive:
                            result = {
                                "provider": payload.get("provider"),
                                "recommendations": recommendations,
                                "chosen_destination": selected,
                                "key_waypoints": [],
                                "route_summary": None,
                                "message": f"get_coarse_waypoints returned non-JSON payload: {route_raw_text}",
                            }
                            print(json.dumps(result, ensure_ascii=True))
                            return
                        print(f"get_coarse_waypoints returned non-JSON payload: {route_raw_text}")
                        continue
                    final_payload = {
                        "provider": payload.get("provider"),
                        "recommendations": recommendations,
                        "chosen_destination": selected,
                        "key_waypoints": build_waypoint_sequence(route_payload),
                        "route_summary": route_payload.get("route_summary", {}),
                        "message": None,
                    }

                if not json_only:
                    print("\n=== Selected Destination Route ===")
                    print(json.dumps(final_payload, indent=2, ensure_ascii=True))
                else:
                    print(json.dumps(final_payload, ensure_ascii=True))
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="macOS current-location recommendation demo")
    parser.add_argument("--prompt", required=True, help="Natural language recommendation prompt")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing mode")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum recommendation candidates")
    parser.add_argument("--radius", type=int, default=3000, help="Search radius in meters")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="Wait time for CoreLocation fix")
    parser.add_argument("--lat", type=float, default=None, help="Optional fallback latitude")
    parser.add_argument("--lon", type=float, default=None, help="Optional fallback longitude")
    parser.add_argument("--non-interactive", action="store_true", help="Run once without interactive prompts")
    parser.add_argument("--select-index", type=int, default=1, help="1-based candidate index in non-interactive mode")
    parser.add_argument("--json-only", action="store_true", help="Output a single JSON object only")

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
            non_interactive=args.non_interactive,
            selection_index=args.select_index,
            json_only=args.json_only,
        )
    )
