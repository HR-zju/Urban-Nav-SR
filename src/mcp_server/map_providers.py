"""Map provider implementations with a shared interface."""

from __future__ import annotations

from typing import Any, Protocol
import math
import os
import re

import aiohttp

from mcp_server.config import (
    default_category_keywords,
    default_search_categories,
    load_json_env,
)


class MapProviderClient(Protocol):
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def geocode(self, query: str) -> list[dict[str, Any]]: ...
    async def reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]: ...
    async def search_semantic_pois(
        self,
        location: str,
        keywords: str,
        radius: int = 1000,
        sortrule: str = "distance",
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...
    async def get_coarse_waypoints(
        self,
        origin: str,
        destination: str,
        mode: str = "foot",
    ) -> dict[str, Any]: ...
    async def get_route(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str = "car",
        steps: bool = False,
        overview: str = "overview",
        annotations: bool = True,
    ) -> dict[str, Any]: ...
    async def get_nearby_pois(
        self,
        lat: float,
        lon: float,
        radius: float = 1000,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...
    async def search_features_by_category(
        self,
        bbox: tuple[float, float, float, float],
        category: str,
        subcategories: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...


def strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text)


def _category_keyword(category: str) -> str:
    mapping = load_json_env("MAP_CATEGORY_KEYWORDS_JSON", default_category_keywords())
    return mapping.get(category, category)


def parse_location_text(location: str) -> tuple[float, float]:
    try:
        lon_str, lat_str = str(location).split(",", 1)
        return float(lon_str.strip()), float(lat_str.strip())
    except Exception as exc:
        raise ValueError(f"Invalid location format: {location!r}. Expected 'longitude,latitude'.") from exc


def format_location(longitude: float, latitude: float) -> str:
    return f"{longitude},{latitude}"


def parse_polyline(polyline: str) -> list[list[float]]:
    coords = []
    for pair in (polyline or "").split(";"):
        if not pair:
            continue
        try:
            lon_str, lat_str = pair.split(",", 1)
            coords.append([float(lon_str), float(lat_str)])
        except Exception:
            continue
    return coords


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def normalize_distance_m(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def normalize_duration_s(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def infer_action(instruction: str, fallback: str = "") -> str:
    text = (instruction or "").strip()
    lowered = text.lower()
    if not text:
        return fallback

    candidates = [
        ("arrive", "arrive"),
        ("destination", "arrive"),
        ("left", "turn_left"),
        ("right", "turn_right"),
        ("straight", "go_straight"),
        ("u-turn", "u_turn"),
        ("roundabout", "roundabout"),
        ("merge", "merge"),
        ("keep left", "keep_left"),
        ("keep right", "keep_right"),
    ]
    for token, action in candidates:
        if token in text or token in lowered:
            return action
    return fallback or text


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_ascii_text(text: str) -> str:
    normalized = (
        str(text or "")
        .replace("（", "(")
        .replace("）", ")")
        .replace("，", ", ")
        .replace("、", ", ")
        .replace("；", "; ")
        .replace("：", ": ")
    )
    normalized = re.sub(r"[\u3400-\u9fff]+", " ", normalized)
    normalized = collapse_whitespace(normalized)
    normalized = re.sub(r"\s+([,;:.)])", r"\1", normalized)
    normalized = re.sub(r"([(])\s+", r"\1", normalized)
    return normalized


def dedupe_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for token in tokens:
        key = token.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(token.strip())
    return deduped


def normalize_compare_text(text: str) -> str:
    normalized = normalize_ascii_text(text).lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    return collapse_whitespace(normalized)


def normalize_poi_type(value: str, typecode: str = "") -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unknown"

    tokens = []
    for segment in raw.split(";"):
        cleaned = segment.strip()
        if not cleaned:
            continue
        normalized_segment = normalize_compare_text(cleaned)
        if not normalized_segment:
            continue
        tokens.append(normalized_segment.replace(" ", "_"))

    tokens = dedupe_tokens(tokens)
    if tokens:
        return ";".join(tokens)
    if typecode:
        return f"typecode_{typecode}"
    if not contains_cjk(raw):
        return raw
    return "unknown"


def normalize_navigation_action(*values: str) -> str:
    merged = " ".join(str(value or "").strip() for value in values if str(value or "").strip()).strip()
    if not merged:
        return "go_straight"
    inferred = infer_action(merged, "")
    if inferred and inferred != merged:
        return inferred
    normalized = normalize_compare_text(merged).replace(" ", "_")
    if normalized in {"arrive", "turn_left", "turn_right", "go_straight", "bear_left", "bear_right", "keep_left", "keep_right", "u_turn"}:
        return normalized
    return "go_straight"


def normalize_address_text(value: str) -> str:
    return str(value or "").strip()


def normalize_road_text(value: str) -> str:
    return str(value or "").strip()


def build_navigation_instruction(action: str, distance_m: int = 0, road: str = "") -> str:
    distance_text = f" for {distance_m} m" if distance_m > 0 else ""
    road_text = f" along {road}" if road else ""

    if action == "arrive":
        return f"Continue{road_text}{distance_text} to arrive at the destination.".strip()
    if action == "turn_left":
        return f"Continue{road_text}{distance_text}, then turn left.".strip()
    if action == "turn_right":
        return f"Continue{road_text}{distance_text}, then turn right.".strip()
    if action == "bear_left":
        return f"Continue{road_text}{distance_text}, then bear left.".strip()
    if action == "bear_right":
        return f"Continue{road_text}{distance_text}, then bear right.".strip()
    if action == "keep_left":
        return f"Continue{road_text}{distance_text}, then keep left.".strip()
    if action == "keep_right":
        return f"Continue{road_text}{distance_text}, then keep right.".strip()
    if action == "u_turn":
        return f"Continue{road_text}{distance_text}, then make a U-turn.".strip()
    return f"Continue{road_text}{distance_text}.".strip()


def build_semantic_poi_record(
    *,
    provider: str,
    provider_poi_id: str,
    name: str,
    longitude: float,
    latitude: float,
    distance_m: int,
    poi_type: str,
    address: str = "",
    raw: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "provider": provider,
        "provider_poi_id": provider_poi_id,
        "name": name or "Unnamed",
        "location": format_location(longitude, latitude),
        "coordinates": {
            "longitude": longitude,
            "latitude": latitude,
        },
        "distance_m": distance_m,
        "type": poi_type or "unknown",
        "address": address or "",
    }
    if extra:
        record.update(extra)
    if raw is not None:
        record["raw"] = raw
    return record


def build_coarse_waypoint_record(
    *,
    longitude: float,
    latitude: float,
    instruction: str,
    action: str,
    road: str = "",
    distance_m: int = 0,
    duration_s: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "location": format_location(longitude, latitude),
        "coordinate": {
            "longitude": longitude,
            "latitude": latitude,
        },
        "instruction": instruction,
        "action": action,
        "road": road,
        "distance_m": distance_m,
        "duration_s": duration_s,
    }
    if extra:
        record.update(extra)
    return record


def parse_google_duration_s(value: Any, default: int = 0) -> int:
    raw = str(value or "").strip()
    if not raw:
        return default
    if raw.endswith("s"):
        raw = raw[:-1]
    return normalize_duration_s(raw, default)


def decode_google_polyline(encoded: str) -> list[list[float]]:
    if not encoded:
        return []

    coordinates: list[list[float]] = []
    index = 0
    latitude = 0
    longitude = 0

    while index < len(encoded):
        shift = 0
        result = 0
        while True:
            if index >= len(encoded):
                return coordinates
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lat = ~(result >> 1) if result & 1 else result >> 1
        latitude += delta_lat

        shift = 0
        result = 0
        while True:
            if index >= len(encoded):
                return coordinates
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lon = ~(result >> 1) if result & 1 else result >> 1
        longitude += delta_lon
        coordinates.append([longitude / 1e5, latitude / 1e5])

    return coordinates


def extract_google_lat_lon(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, dict):
        return None, None

    if "latitude" in value and "longitude" in value:
        try:
            return float(value["latitude"]), float(value["longitude"])
        except Exception:
            return None, None

    if "lat" in value and "lng" in value:
        try:
            return float(value["lat"]), float(value["lng"])
        except Exception:
            return None, None

    lat_lng = value.get("latLng") or value.get("lat_lng")
    if isinstance(lat_lng, dict):
        return extract_google_lat_lon(lat_lng)

    location = value.get("location")
    if isinstance(location, dict):
        return extract_google_lat_lon(location)

    return None, None


def parse_google_places(
    places: list[dict[str, Any]],
    *,
    origin_lat: float,
    origin_lon: float,
    limit: int = 10,
    provider: str = "google",
) -> list[dict[str, Any]]:
    normalized = []
    for place in places:
        latitude, longitude = extract_google_lat_lon(place.get("location") or place)
        if latitude is None or longitude is None:
            continue

        display_name = place.get("displayName", {})
        if isinstance(display_name, dict):
            name = str(display_name.get("text") or "").strip()
        else:
            name = str(display_name or "").strip()
        if not name:
            name = str(place.get("name") or "Unnamed")

        address = normalize_address_text(
            str(place.get("formattedAddress") or place.get("shortFormattedAddress") or "")
        )
        primary_type = str(place.get("primaryType") or "").strip()
        types = [str(item).strip() for item in (place.get("types") or []) if str(item).strip()]
        type_raw = primary_type or ";".join(types)
        distance_m = round(haversine_m(origin_lat, origin_lon, latitude, longitude))
        provider_poi_id = str(place.get("id") or place.get("name") or format_location(longitude, latitude))
        normalized.append(
            build_semantic_poi_record(
                provider=provider,
                provider_poi_id=provider_poi_id,
                name=name,
                longitude=longitude,
                latitude=latitude,
                distance_m=distance_m,
                poi_type=normalize_poi_type(type_raw or "unknown"),
                address=address,
                raw=place,
                extra={
                    "type_raw": type_raw,
                    "address_raw": address,
                    "types": types,
                    "primary_type": primary_type,
                },
            )
        )

    normalized.sort(key=lambda item: item["distance_m"])
    return normalized[: max(1, limit)]


def parse_amap_pois(
    pois: list[dict[str, Any]],
    *,
    origin_lat: float,
    origin_lon: float,
    limit: int = 10,
) -> list[dict[str, Any]]:
    normalized = []
    for poi in pois:
        try:
            longitude, latitude = parse_location_text(poi.get("location", ""))
        except ValueError:
            continue

        distance_m = normalize_distance_m(
            poi.get("distance"),
            default=round(haversine_m(origin_lat, origin_lon, latitude, longitude)),
        )
        normalized.append(
            build_semantic_poi_record(
                provider="amap",
                provider_poi_id=str(poi.get("id") or format_location(longitude, latitude)),
                name=str(poi.get("name") or "Unnamed"),
                longitude=longitude,
                latitude=latitude,
                distance_m=distance_m,
                poi_type=normalize_poi_type(
                    str(poi.get("type") or "unknown"),
                    str(poi.get("typecode") or ""),
                ),
                address=normalize_address_text(str(poi.get("address") or "")),
                raw=poi,
                extra={
                    "type_raw": str(poi.get("type") or ""),
                    "address_raw": str(poi.get("address") or ""),
                    "typecode": str(poi.get("typecode") or ""),
                    "adname": str(poi.get("adname") or ""),
                    "business_area": str(poi.get("business_area") or ""),
                },
            )
        )

    normalized.sort(key=lambda item: item["distance_m"])
    return normalized[: max(1, limit)]


def parse_baidu_pois(
    pois: list[dict[str, Any]],
    *,
    origin_lat: float,
    origin_lon: float,
    limit: int = 10,
) -> list[dict[str, Any]]:
    normalized = []
    for poi in pois:
        location = poi.get("location", {}) or {}
        latitude = location.get("lat")
        longitude = location.get("lng")
        if latitude is None or longitude is None:
            continue

        detail_info = poi.get("detail_info", {}) or {}
        distance_m = normalize_distance_m(
            detail_info.get("distance"),
            default=round(haversine_m(origin_lat, origin_lon, float(latitude), float(longitude))),
        )
        normalized.append(
            build_semantic_poi_record(
                provider="baidu",
                provider_poi_id=str(poi.get("uid") or format_location(float(longitude), float(latitude))),
                name=str(poi.get("name") or "Unnamed"),
                longitude=float(longitude),
                latitude=float(latitude),
                distance_m=distance_m,
                poi_type=normalize_poi_type(str(detail_info.get("tag") or "unknown")),
                address=normalize_address_text(str(poi.get("address") or "")),
                raw=poi,
                extra={
                    "type_raw": str(detail_info.get("tag") or ""),
                    "address_raw": str(poi.get("address") or ""),
                    "province": str(poi.get("province") or ""),
                    "city": str(poi.get("city") or ""),
                    "area": str(poi.get("area") or ""),
                },
            )
        )

    normalized.sort(key=lambda item: item["distance_m"])
    return normalized[: max(1, limit)]


def parse_amap_steps(route_data: dict[str, Any], *, origin: str, destination: str, mode: str) -> dict[str, Any]:
    route = route_data.get("route", {}) or {}
    paths = route.get("paths", []) or []
    if not paths:
        return {
            "provider": "amap",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "route_summary": {"distance_m": 0, "duration_s": 0, "step_count": 0},
            "coarse_waypoints": [],
        }

    path = paths[0]
    coarse_waypoints = []
    for step in path.get("steps", []) or []:
        coords = parse_polyline(step.get("polyline", ""))
        if not coords:
            continue
        longitude, latitude = coords[-1]
        instruction_raw = str(step.get("instruction") or "").strip()
        action_raw = str(step.get("action") or step.get("assistant_action") or "").strip()
        road_raw = str(step.get("road") or "").strip()
        distance_m = normalize_distance_m(step.get("distance"))
        duration_s = normalize_duration_s(step.get("duration"))
        action = normalize_navigation_action(action_raw, instruction_raw)
        road = normalize_road_text(road_raw)
        instruction = build_navigation_instruction(action, distance_m=distance_m, road=road)
        coarse_waypoints.append(
            build_coarse_waypoint_record(
                longitude=longitude,
                latitude=latitude,
                instruction=instruction,
                action=action,
                road=road,
                distance_m=distance_m,
                duration_s=duration_s,
                extra={
                    "instruction_raw": instruction_raw,
                    "action_raw": action_raw,
                    "road_raw": road_raw,
                    "assistant_action": str(step.get("assistant_action") or ""),
                },
            )
        )

    return {
        "provider": "amap",
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "route_summary": {
            "distance_m": normalize_distance_m(path.get("distance")),
            "duration_s": normalize_duration_s(path.get("duration")),
            "step_count": len(coarse_waypoints),
        },
        "coarse_waypoints": coarse_waypoints,
    }


def parse_baidu_steps(route_data: dict[str, Any], *, origin: str, destination: str, mode: str) -> dict[str, Any]:
    route_result = route_data.get("result", {}) or {}
    routes = route_result.get("routes", []) or []
    if not routes:
        return {
            "provider": "baidu",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "route_summary": {"distance_m": 0, "duration_s": 0, "step_count": 0},
            "coarse_waypoints": [],
        }

    best_route = routes[0]
    coarse_waypoints = []
    for step in best_route.get("steps", []) or []:
        coords = parse_polyline(step.get("path", ""))
        if not coords:
            continue
        longitude, latitude = coords[-1]
        instruction_raw = strip_html(str(step.get("instruction") or "")).strip()
        action_raw = str(step.get("turn_type") or "").strip()
        road_raw = str(step.get("road_name") or "").strip()
        distance_m = normalize_distance_m(step.get("distance"))
        duration_s = normalize_duration_s(step.get("duration"))
        action = normalize_navigation_action(action_raw, instruction_raw)
        road = normalize_road_text(road_raw)
        instruction = build_navigation_instruction(action, distance_m=distance_m, road=road)
        coarse_waypoints.append(
            build_coarse_waypoint_record(
                longitude=longitude,
                latitude=latitude,
                instruction=instruction,
                action=action,
                road=road,
                distance_m=distance_m,
                duration_s=duration_s,
                extra={
                    "instruction_raw": instruction_raw,
                    "action_raw": action_raw,
                    "road_raw": road_raw,
                    "turn_type": action_raw,
                },
            )
        )

    return {
        "provider": "baidu",
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "route_summary": {
            "distance_m": normalize_distance_m(best_route.get("distance")),
            "duration_s": normalize_duration_s(best_route.get("duration")),
            "step_count": len(coarse_waypoints),
        },
        "coarse_waypoints": coarse_waypoints,
    }


def parse_google_steps(route_data: dict[str, Any], *, origin: str, destination: str, mode: str) -> dict[str, Any]:
    routes = route_data.get("routes", []) or []
    if not routes:
        return {
            "provider": "google",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "route_summary": {"distance_m": 0, "duration_s": 0, "step_count": 0},
            "coarse_waypoints": [],
        }

    route = routes[0]
    coarse_waypoints = []
    for leg in route.get("legs", []) or []:
        for step in leg.get("steps", []) or []:
            coords = decode_google_polyline(((step.get("polyline") or {}).get("encodedPolyline") or ""))
            latitude, longitude = extract_google_lat_lon(step.get("endLocation") or {})
            if latitude is None or longitude is None:
                if not coords:
                    continue
                longitude, latitude = coords[-1]

            navigation = step.get("navigationInstruction", {}) or {}
            instruction_raw = str(navigation.get("instructions") or "").strip()
            maneuver_raw = str(navigation.get("maneuver") or step.get("maneuver") or "").strip()
            distance_m = normalize_distance_m(step.get("distanceMeters"))
            duration_s = parse_google_duration_s(step.get("staticDuration") or step.get("duration"))
            action = normalize_navigation_action(maneuver_raw.replace("_", " ").lower(), instruction_raw)
            instruction = instruction_raw or build_navigation_instruction(action, distance_m=distance_m)
            coarse_waypoints.append(
                build_coarse_waypoint_record(
                    longitude=longitude,
                    latitude=latitude,
                    instruction=instruction,
                    action=action,
                    road="",
                    distance_m=distance_m,
                    duration_s=duration_s,
                    extra={
                        "instruction_raw": instruction_raw,
                        "maneuver_raw": maneuver_raw,
                    },
                )
            )

    return {
        "provider": "google",
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "route_summary": {
            "distance_m": normalize_distance_m(route.get("distanceMeters")),
            "duration_s": parse_google_duration_s(route.get("duration")),
            "step_count": len(coarse_waypoints),
        },
        "coarse_waypoints": coarse_waypoints,
    }


class BaseHTTPMapClient:
    def __init__(self) -> None:
        self.session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        self.session = aiohttp.ClientSession()

    async def disconnect(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    def require_session(self) -> aiohttp.ClientSession:
        if not self.session:
            raise RuntimeError(f"{self.__class__.__name__} not connected")
        return self.session


class OSMClient(BaseHTTPMapClient):
    def __init__(self, base_url: str = "https://api.openstreetmap.org/api/0.6") -> None:
        super().__init__()
        self.base_url = base_url
        self.cache: dict[str, Any] = {}

    async def geocode(self, query: str) -> list[dict[str, Any]]:
        session = self.require_session()
        async with session.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 5},
            headers={"User-Agent": "OSM-MCP-Server/1.0"},
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to geocode '{query}': {response.status}")
            return await response.json()

    async def reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        session = self.require_session()
        async with session.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "OSM-MCP-Server/1.0"},
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to reverse geocode ({lat}, {lon}): {response.status}")
            return await response.json()

    async def search_semantic_pois(
        self,
        location: str,
        keywords: str,
        radius: int = 1000,
        sortrule: str = "distance",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        longitude, latitude = parse_location_text(location)
        keyword_tokens = [token.strip().lower() for token in keywords.split("|") if token.strip()]
        places = await self.get_nearby_pois(latitude, longitude, radius, default_search_categories())

        normalized = []
        for place in places:
            tags = place.get("tags", {}) or {}
            place_lat = place.get("lat")
            place_lon = place.get("lon")
            if place_lat is None or place_lon is None:
                continue

            name = str(tags.get("name") or "Unnamed")
            text_blob = f"{name} {tags}".lower()
            if keyword_tokens and not any(token in text_blob for token in keyword_tokens):
                continue

            distance_m = round(haversine_m(latitude, longitude, float(place_lat), float(place_lon)))
            normalized.append(
                build_semantic_poi_record(
                    provider="osm",
                    provider_poi_id=str(place.get("id") or format_location(float(place_lon), float(place_lat))),
                    name=name,
                    longitude=float(place_lon),
                    latitude=float(place_lat),
                    distance_m=distance_m,
                    poi_type=str(tags.get("type") or ""),
                    address=str(tags.get("address") or ""),
                    raw=place,
                )
            )

        if sortrule == "distance":
            normalized.sort(key=lambda item: item["distance_m"])
        return normalized[: max(1, limit)]

    async def get_coarse_waypoints(
        self,
        origin: str,
        destination: str,
        mode: str = "foot",
    ) -> dict[str, Any]:
        origin_lon, origin_lat = parse_location_text(origin)
        destination_lon, destination_lat = parse_location_text(destination)
        route_data = await self.get_route(
            from_lat=origin_lat,
            from_lon=origin_lon,
            to_lat=destination_lat,
            to_lon=destination_lon,
            mode=mode,
            steps=True,
            overview="simplified",
            annotations=False,
        )

        routes = route_data.get("routes", []) or []
        if not routes:
            return {
                "provider": "osm",
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "route_summary": {"distance_m": 0, "duration_s": 0, "step_count": 0},
                "coarse_waypoints": [],
            }

        route = routes[0]
        coarse_waypoints = []
        for leg in route.get("legs", []) or []:
            for step in leg.get("steps", []) or []:
                maneuver = step.get("maneuver", {}) or {}
                location_point = maneuver.get("location")
                if not location_point and step.get("geometry", {}).get("coordinates"):
                    location_point = step["geometry"]["coordinates"][-1]
                if not location_point or len(location_point) != 2:
                    continue
                longitude, latitude = float(location_point[0]), float(location_point[1])
                instruction = str(maneuver.get("instruction") or "").strip()
                coarse_waypoints.append(
                    build_coarse_waypoint_record(
                        longitude=longitude,
                        latitude=latitude,
                        instruction=instruction,
                        action=infer_action(instruction, "go_straight"),
                        road=str(step.get("name") or ""),
                        distance_m=normalize_distance_m(step.get("distance")),
                        duration_s=normalize_duration_s(step.get("duration")),
                    )
                )

        return {
            "provider": "osm",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "route_summary": {
                "distance_m": normalize_distance_m(route.get("distance")),
                "duration_s": normalize_duration_s(route.get("duration")),
                "step_count": len(coarse_waypoints),
            },
            "coarse_waypoints": coarse_waypoints,
        }

    async def get_route(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str = "car",
        steps: bool = False,
        overview: str = "overview",
        annotations: bool = True,
    ) -> dict[str, Any]:
        session = self.require_session()
        osrm_url = f"http://router.project-osrm.org/route/v1/{mode}/{from_lon},{from_lat};{to_lon},{to_lat}"
        params = {
            "overview": overview,
            "geometries": "geojson",
            "steps": str(steps).lower(),
            "annotations": str(annotations).lower(),
        }
        async with session.get(osrm_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to get route: {response.status}")
            return await response.json()

    async def get_nearby_pois(
        self,
        lat: float,
        lon: float,
        radius: float = 1000,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        session = self.require_session()

        lat_delta = radius / 111000
        lon_delta = radius / (111000 * math.cos(math.radians(lat)))
        bbox = (lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta)

        categories = categories or default_search_categories()
        tag_filters = [f'node["{category}"]({{bbox}});' for category in categories]
        query = f"""
        [out:json];
        (
            {" ".join(tag_filters)}
        );
        out body;
        """.replace("{bbox}", f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}")

        async with session.post("https://overpass-api.de/api/interpreter", data={"data": query}) as response:
            if response.status != 200:
                raise Exception(f"Failed to get nearby POIs: {response.status}")
            data = await response.json()
            return data.get("elements", [])

    async def search_features_by_category(
        self,
        bbox: tuple[float, float, float, float],
        category: str,
        subcategories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        session = self.require_session()
        if subcategories:
            subcategory_filters = " or ".join([f'"{category}"="{sub}"' for sub in subcategories])
            query_filter = f"({subcategory_filters})"
        else:
            query_filter = f'"{category}"'

        query = f"""
        [out:json];
        (
          node[{query_filter}]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          way[{query_filter}]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          relation[{query_filter}]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        """

        async with session.post("https://overpass-api.de/api/interpreter", data={"data": query}) as response:
            if response.status != 200:
                raise Exception(f"Failed to search features by category: {response.status}")
            data = await response.json()
            return data.get("elements", [])


class AMapClient(BaseHTTPMapClient):
    def __init__(self, api_key: str) -> None:
        super().__init__()
        self.api_key = api_key

    @staticmethod
    def _split_location(location: str) -> tuple[float | None, float | None]:
        try:
            longitude, latitude = parse_location_text(location)
            return latitude, longitude
        except Exception:
            return None, None

    async def geocode(self, query: str) -> list[dict[str, Any]]:
        session = self.require_session()
        results: list[dict[str, Any]] = []

        # Phase 1: address geocoding (works well for concrete addresses).
        async with session.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params={"key": self.api_key, "address": query},
        ) as response:
            data = await response.json(content_type=None)
            if response.status == 200 and data.get("status") == "1":
                for item in data.get("geocodes", []):
                    lat, lon = self._split_location(item.get("location", ""))
                    if lat is None or lon is None:
                        continue
                    results.append(
                        {
                            "display_name": item.get("formatted_address", query),
                            "lat": str(lat),
                            "lon": str(lon),
                            "source": "amap_geocode",
                            "raw": item,
                        }
                    )

        if results:
            return results

        # Phase 2: POI text search for named places/landmarks (e.g. "West Lake").
        async with session.get(
            "https://restapi.amap.com/v3/place/text",
            params={
                "key": self.api_key,
                "keywords": query,
                "offset": 10,
                "page": 1,
            },
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != "1":
                raise Exception(f"AMap geocode/place-text failed: {data.get('info', response.status)}")

        poi_results: list[dict[str, Any]] = []
        for poi in data.get("pois", []):
            lat, lon = self._split_location(poi.get("location", ""))
            if lat is None or lon is None:
                continue
            name = str(poi.get("name") or query).strip() or query
            address = str(poi.get("address") or "").strip()
            display_name = name if not address else f"{name} ({address})"
            poi_results.append(
                {
                    "display_name": display_name,
                    "lat": str(lat),
                    "lon": str(lon),
                    "source": "amap_poi_text",
                    "raw": poi,
                }
            )

        return poi_results

    async def reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        session = self.require_session()
        async with session.get(
            "https://restapi.amap.com/v3/geocode/regeo",
            params={"key": self.api_key, "location": f"{lon},{lat}", "extensions": "base"},
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != "1":
                raise Exception(f"AMap reverse geocode failed: {data.get('info', response.status)}")

        regeocode = data.get("regeocode", {})
        return {
            "display_name": regeocode.get("formatted_address", "Unknown location"),
            "address": regeocode.get("addressComponent", {}),
            "source": "amap",
        }

    async def search_semantic_pois(
        self,
        location: str,
        keywords: str,
        radius: int = 1000,
        sortrule: str = "distance",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        session = self.require_session()
        origin_lon, origin_lat = parse_location_text(location)
        async with session.get(
            "https://restapi.amap.com/v3/place/around",
            params={
                "key": self.api_key,
                "location": location,
                "keywords": keywords,
                "radius": int(radius),
                "sortrule": sortrule,
                "offset": max(1, min(limit, 25)),
                "page": 1,
            },
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != "1":
                raise Exception(f"AMap semantic POI search failed: {data.get('info', response.status)}")

        return parse_amap_pois(
            data.get("pois", []),
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            limit=limit,
        )

    async def get_coarse_waypoints(
        self,
        origin: str,
        destination: str,
        mode: str = "foot",
    ) -> dict[str, Any]:
        session = self.require_session()
        amap_mode = {"car": "driving", "foot": "walking", "bike": "walking"}.get(mode, "walking")
        async with session.get(
            f"https://restapi.amap.com/v3/direction/{amap_mode}",
            params={
                "key": self.api_key,
                "origin": origin,
                "destination": destination,
                "extensions": "all",
            },
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != "1":
                raise Exception(f"AMap coarse waypoint route failed: {data.get('info', response.status)}")

        return parse_amap_steps(data, origin=origin, destination=destination, mode=mode)

    async def get_nearby_pois(
        self,
        lat: float,
        lon: float,
        radius: float = 1000,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        session = self.require_session()
        categories = categories or default_search_categories()

        results_by_id: dict[str, dict[str, Any]] = {}
        for category in categories:
            async with session.get(
                "https://restapi.amap.com/v3/place/around",
                params={
                    "key": self.api_key,
                    "location": f"{lon},{lat}",
                    "radius": int(radius),
                    "keywords": _category_keyword(category),
                    "sortrule": "distance",
                    "offset": 25,
                    "page": 1,
                },
            ) as response:
                data = await response.json(content_type=None)
                if response.status != 200 or data.get("status") != "1":
                    continue

            for poi in data.get("pois", []):
                poi_id = poi.get("id") or f"amap-{poi.get('location', '')}-{poi.get('name', '')}"
                poi_lat, poi_lon = self._split_location(poi.get("location", ""))
                if poi_lat is None or poi_lon is None:
                    continue
                results_by_id[poi_id] = {
                    "id": poi_id,
                    "lat": poi_lat,
                    "lon": poi_lon,
                    "tags": {
                        "name": poi.get("name", "Unnamed"),
                        category: poi.get("type", "unknown"),
                        "type": poi.get("type", "unknown"),
                        "address": poi.get("address", ""),
                        "source": "amap",
                    },
                }

        return list(results_by_id.values())

    async def search_features_by_category(
        self,
        bbox: tuple[float, float, float, float],
        category: str,
        subcategories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2
        rough_radius = max(200, int(max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])) * 111000))
        categories = subcategories if subcategories else [category]
        pois = await self.get_nearby_pois(center_lat, center_lon, rough_radius, categories)
        return [
            {
                "id": poi.get("id"),
                "type": "node",
                "lat": poi.get("lat"),
                "lon": poi.get("lon"),
                "tags": poi.get("tags", {}),
            }
            for poi in pois
        ]

    async def get_route(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str = "car",
        steps: bool = False,
        overview: str = "overview",
        annotations: bool = True,
    ) -> dict[str, Any]:
        session = self.require_session()
        amap_mode = {"car": "driving", "foot": "walking", "bike": "walking"}.get(mode, "driving")
        async with session.get(
            f"https://restapi.amap.com/v3/direction/{amap_mode}",
            params={
                "key": self.api_key,
                "origin": f"{from_lon},{from_lat}",
                "destination": f"{to_lon},{to_lat}",
                "extensions": "all",
            },
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != "1":
                raise Exception(f"AMap route failed: {data.get('info', response.status)}")

        route = data.get("route", {})
        paths = route.get("paths", [])
        if not paths:
            return {"routes": [], "waypoints": []}

        path = paths[0]
        leg_steps = []
        geometry_coords = []
        for step in path.get("steps", []):
            step_coords = []
            for pair in step.get("polyline", "").split(";"):
                if not pair:
                    continue
                try:
                    lon_str, lat_str = pair.split(",", 1)
                    coord = [float(lon_str), float(lat_str)]
                    step_coords.append(coord)
                    geometry_coords.append(coord)
                except Exception:
                    continue

            maneuver: dict[str, Any] = {"instruction": step.get("instruction", "")}
            if step_coords:
                maneuver["location"] = step_coords[-1]

            leg_steps.append(
                {
                    "maneuver": maneuver,
                    "distance": float(step.get("distance", 0) or 0),
                    "duration": float(step.get("duration", 0) or 0),
                    "name": step.get("road", ""),
                    "geometry": {"type": "LineString", "coordinates": step_coords},
                }
            )

        return {
            "routes": [
                {
                    "distance": float(path.get("distance", 0) or 0),
                    "duration": float(path.get("duration", 0) or 0),
                    "legs": [{"steps": leg_steps}],
                    "geometry": {"type": "LineString", "coordinates": geometry_coords},
                }
            ],
            "waypoints": [
                {"location": [from_lon, from_lat], "name": "origin"},
                {"location": [to_lon, to_lat], "name": "destination"},
            ],
            "source": "amap",
        }


class BaiduClient(BaseHTTPMapClient):
    def __init__(self, ak: str) -> None:
        super().__init__()
        self.ak = ak

    @staticmethod
    def _parse_path(path: str) -> list[list[float]]:
        return parse_polyline(path)

    async def geocode(self, query: str) -> list[dict[str, Any]]:
        session = self.require_session()
        async with session.get(
            "https://api.map.baidu.com/geocoding/v3/",
            params={"address": query, "output": "json", "ak": self.ak},
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != 0:
                raise Exception(f"Baidu geocode failed: {data.get('message', response.status)}")

        result = data.get("result", {})
        location = result.get("location", {})
        lat = location.get("lat")
        lon = location.get("lng")
        if lat is None or lon is None:
            return []
        return [
            {
                "display_name": result.get("formatted_address", query),
                "lat": str(lat),
                "lon": str(lon),
                "source": "baidu",
                "raw": result,
            }
        ]

    async def reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        session = self.require_session()
        async with session.get(
            "https://api.map.baidu.com/reverse_geocoding/v3/",
            params={"location": f"{lat},{lon}", "output": "json", "ak": self.ak},
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != 0:
                raise Exception(f"Baidu reverse geocode failed: {data.get('message', response.status)}")

        result = data.get("result", {})
        return {
            "display_name": result.get("formatted_address", "Unknown location"),
            "address": result.get("addressComponent", {}),
            "source": "baidu",
        }

    async def search_semantic_pois(
        self,
        location: str,
        keywords: str,
        radius: int = 1000,
        sortrule: str = "distance",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        session = self.require_session()
        origin_lon, origin_lat = parse_location_text(location)
        async with session.get(
            "https://api.map.baidu.com/place/v2/search",
            params={
                "query": keywords,
                "location": f"{origin_lat},{origin_lon}",
                "radius": int(radius),
                "output": "json",
                "ak": self.ak,
                "scope": 2,
                "page_size": max(1, min(limit, 20)),
                "page_num": 0,
            },
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != 0:
                raise Exception(f"Baidu semantic POI search failed: {data.get('message', response.status)}")

        parsed = parse_baidu_pois(
            data.get("results", []),
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            limit=limit,
        )
        if sortrule == "distance":
            parsed.sort(key=lambda item: item["distance_m"])
        return parsed

    async def get_coarse_waypoints(
        self,
        origin: str,
        destination: str,
        mode: str = "foot",
    ) -> dict[str, Any]:
        session = self.require_session()
        origin_lon, origin_lat = parse_location_text(origin)
        destination_lon, destination_lat = parse_location_text(destination)
        baidu_mode = {"car": "driving", "foot": "walking", "bike": "riding"}.get(mode, "walking")
        async with session.get(
            f"https://api.map.baidu.com/directionlite/v1/{baidu_mode}",
            params={
                "origin": f"{origin_lat},{origin_lon}",
                "destination": f"{destination_lat},{destination_lon}",
                "ak": self.ak,
            },
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != 0:
                raise Exception(f"Baidu coarse waypoint route failed: {data.get('message', response.status)}")

        return parse_baidu_steps(data, origin=origin, destination=destination, mode=mode)

    async def get_nearby_pois(
        self,
        lat: float,
        lon: float,
        radius: float = 1000,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        session = self.require_session()
        categories = categories or default_search_categories()

        results_by_id: dict[str, dict[str, Any]] = {}
        for category in categories:
            async with session.get(
                "https://api.map.baidu.com/place/v2/search",
                params={
                    "query": _category_keyword(category),
                    "location": f"{lat},{lon}",
                    "radius": int(radius),
                    "output": "json",
                    "ak": self.ak,
                    "scope": 2,
                    "page_size": 20,
                    "page_num": 0,
                },
            ) as response:
                data = await response.json(content_type=None)
                if response.status != 200 or data.get("status") != 0:
                    continue

            for poi in data.get("results", []):
                location = poi.get("location", {})
                poi_lat = location.get("lat")
                poi_lon = location.get("lng")
                if poi_lat is None or poi_lon is None:
                    continue

                poi_id = poi.get("uid") or f"baidu-{poi_lon}-{poi_lat}-{poi.get('name', '')}"
                results_by_id[poi_id] = {
                    "id": poi_id,
                    "lat": poi_lat,
                    "lon": poi_lon,
                    "tags": {
                        "name": poi.get("name", "Unnamed"),
                        category: poi.get("detail_info", {}).get("tag", "unknown"),
                        "type": poi.get("detail_info", {}).get("tag", "unknown"),
                        "address": poi.get("address", ""),
                        "source": "baidu",
                    },
                }

        return list(results_by_id.values())

    async def search_features_by_category(
        self,
        bbox: tuple[float, float, float, float],
        category: str,
        subcategories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2
        rough_radius = max(200, int(max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])) * 111000))
        categories = subcategories if subcategories else [category]
        pois = await self.get_nearby_pois(center_lat, center_lon, rough_radius, categories)
        return [
            {
                "id": poi.get("id"),
                "type": "node",
                "lat": poi.get("lat"),
                "lon": poi.get("lon"),
                "tags": poi.get("tags", {}),
            }
            for poi in pois
        ]

    async def get_route(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str = "car",
        steps: bool = False,
        overview: str = "overview",
        annotations: bool = True,
    ) -> dict[str, Any]:
        session = self.require_session()
        baidu_mode = {"car": "driving", "foot": "walking", "bike": "riding"}.get(mode, "driving")
        async with session.get(
            f"https://api.map.baidu.com/directionlite/v1/{baidu_mode}",
            params={"origin": f"{from_lat},{from_lon}", "destination": f"{to_lat},{to_lon}", "ak": self.ak},
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("status") != 0:
                raise Exception(f"Baidu route failed: {data.get('message', response.status)}")

        route_result = data.get("result", {})
        routes = route_result.get("routes", [])
        if not routes:
            return {"routes": [], "waypoints": []}

        best_route = routes[0]
        leg_steps = []
        geometry_coords = []
        for step in best_route.get("steps", []):
            step_coords = self._parse_path(step.get("path", ""))
            geometry_coords.extend(step_coords)
            maneuver: dict[str, Any] = {"instruction": strip_html(step.get("instruction", ""))}
            if step_coords:
                maneuver["location"] = step_coords[-1]

            leg_steps.append(
                {
                    "maneuver": maneuver,
                    "distance": float(step.get("distance", 0) or 0),
                    "duration": float(step.get("duration", 0) or 0),
                    "name": step.get("road_name", ""),
                    "geometry": {"type": "LineString", "coordinates": step_coords},
                }
            )

        return {
            "routes": [
                {
                    "distance": float(best_route.get("distance", 0) or 0),
                    "duration": float(best_route.get("duration", 0) or 0),
                    "legs": [{"steps": leg_steps}],
                    "geometry": {"type": "LineString", "coordinates": geometry_coords},
                }
            ],
            "waypoints": [
                {"location": [from_lon, from_lat], "name": "origin"},
                {"location": [to_lon, to_lat], "name": "destination"},
            ],
            "source": "baidu",
        }


class GoogleMapsClient(BaseHTTPMapClient):
    def __init__(self, api_key: str, *, language: str = "", region: str = "") -> None:
        super().__init__()
        self.api_key = api_key
        self.language = language.strip()
        self.region = region.strip().upper()

    @staticmethod
    def _normalize_query(query: str) -> str:
        tokens = [token.strip() for token in str(query or "").replace("|", " ").split() if token.strip()]
        return collapse_whitespace(" ".join(tokens))

    @staticmethod
    def _travel_mode(mode: str) -> str:
        return {
            "car": "DRIVE",
            "foot": "WALK",
            "bike": "BICYCLE",
        }.get(mode, "DRIVE")

    def _places_headers(self, field_mask: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

    def _routes_headers(self, field_mask: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

    async def _text_search_places(
        self,
        *,
        query: str,
        latitude: float | None = None,
        longitude: float | None = None,
        radius: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        session = self.require_session()
        body: dict[str, Any] = {
            "textQuery": self._normalize_query(query),
            "pageSize": max(1, min(limit, 20)),
        }
        if latitude is not None and longitude is not None and radius:
            body["locationBias"] = {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude,
                    },
                    "radius": float(radius),
                }
            }
        if self.language:
            body["languageCode"] = self.language
        if self.region:
            body["regionCode"] = self.region

        async with session.post(
            "https://places.googleapis.com/v1/places:searchText",
            json=body,
            headers=self._places_headers(
                "places.id,places.name,places.displayName,places.formattedAddress,places.shortFormattedAddress,"
                "places.location,places.primaryType,places.types"
            ),
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200:
                raise Exception(f"Google Places text search failed: {data}")
        return data.get("places", []) or []

    async def _compute_routes(
        self,
        *,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str,
        include_steps: bool,
    ) -> dict[str, Any]:
        session = self.require_session()
        body: dict[str, Any] = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": from_lat,
                        "longitude": from_lon,
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": to_lat,
                        "longitude": to_lon,
                    }
                }
            },
            "travelMode": self._travel_mode(mode),
            "units": "METRIC",
        }
        if self.language:
            body["languageCode"] = self.language

        field_mask = ["routes.distanceMeters", "routes.duration", "routes.polyline.encodedPolyline"]
        if include_steps:
            field_mask.extend(
                [
                    "routes.legs.steps.distanceMeters",
                    "routes.legs.steps.staticDuration",
                    "routes.legs.steps.polyline.encodedPolyline",
                    "routes.legs.steps.navigationInstruction.instructions",
                    "routes.legs.steps.navigationInstruction.maneuver",
                    "routes.legs.steps.startLocation",
                    "routes.legs.steps.endLocation",
                ]
            )

        async with session.post(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            json=body,
            headers=self._routes_headers(",".join(field_mask)),
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200:
                raise Exception(f"Google Routes computeRoutes failed: {data}")
        return data

    async def geocode(self, query: str) -> list[dict[str, Any]]:
        session = self.require_session()
        params: dict[str, Any] = {
            "address": query,
            "key": self.api_key,
        }
        if self.language:
            params["language"] = self.language
        if self.region:
            params["region"] = self.region

        async with session.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params=params,
        ) as response:
            data = await response.json(content_type=None)
            status = str(data.get("status") or "")
            if response.status != 200 or status not in {"OK", "ZERO_RESULTS"}:
                raise Exception(f"Google geocode failed: {status or response.status}")

        results = []
        for item in data.get("results", []) or []:
            geometry = item.get("geometry", {}) or {}
            location = geometry.get("location", {}) or {}
            latitude, longitude = extract_google_lat_lon(location)
            if latitude is None or longitude is None:
                continue
            results.append(
                {
                    "display_name": str(item.get("formatted_address") or query),
                    "lat": str(latitude),
                    "lon": str(longitude),
                    "source": "google_geocode",
                    "raw": item,
                }
            )
        return results

    async def reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        session = self.require_session()
        params: dict[str, Any] = {
            "latlng": f"{lat},{lon}",
            "key": self.api_key,
        }
        if self.language:
            params["language"] = self.language
        if self.region:
            params["region"] = self.region

        async with session.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params=params,
        ) as response:
            data = await response.json(content_type=None)
            status = str(data.get("status") or "")
            if response.status != 200 or status not in {"OK", "ZERO_RESULTS"}:
                raise Exception(f"Google reverse geocode failed: {status or response.status}")

        result = (data.get("results", []) or [{}])[0]
        return {
            "display_name": str(result.get("formatted_address") or "Unknown location"),
            "address": result.get("address_components", []),
            "source": "google",
        }

    async def search_semantic_pois(
        self,
        location: str,
        keywords: str,
        radius: int = 1000,
        sortrule: str = "distance",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        origin_lon, origin_lat = parse_location_text(location)
        places = await self._text_search_places(
            query=keywords,
            latitude=origin_lat,
            longitude=origin_lon,
            radius=int(radius),
            limit=limit,
        )
        parsed = parse_google_places(
            places,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            limit=limit,
            provider="google",
        )
        if sortrule == "distance":
            parsed.sort(key=lambda item: item["distance_m"])
        return parsed

    async def get_coarse_waypoints(
        self,
        origin: str,
        destination: str,
        mode: str = "foot",
    ) -> dict[str, Any]:
        origin_lon, origin_lat = parse_location_text(origin)
        destination_lon, destination_lat = parse_location_text(destination)
        route_data = await self._compute_routes(
            from_lat=origin_lat,
            from_lon=origin_lon,
            to_lat=destination_lat,
            to_lon=destination_lon,
            mode=mode,
            include_steps=True,
        )
        return parse_google_steps(route_data, origin=origin, destination=destination, mode=mode)

    async def get_route(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str = "car",
        steps: bool = False,
        overview: str = "overview",
        annotations: bool = True,
    ) -> dict[str, Any]:
        route_data = await self._compute_routes(
            from_lat=from_lat,
            from_lon=from_lon,
            to_lat=to_lat,
            to_lon=to_lon,
            mode=mode,
            include_steps=steps,
        )
        routes = route_data.get("routes", []) or []
        if not routes:
            return {"routes": [], "waypoints": []}

        route = routes[0]
        route_coords = decode_google_polyline(((route.get("polyline") or {}).get("encodedPolyline") or ""))
        legs = []
        for leg in route.get("legs", []) or []:
            leg_steps = []
            for step in leg.get("steps", []) or []:
                step_coords = decode_google_polyline(((step.get("polyline") or {}).get("encodedPolyline") or ""))
                latitude, longitude = extract_google_lat_lon(step.get("endLocation") or {})
                maneuver = step.get("navigationInstruction", {}) or {}
                instruction = str(maneuver.get("instructions") or "").strip()
                maneuver_payload: dict[str, Any] = {"instruction": instruction}
                if latitude is not None and longitude is not None:
                    maneuver_payload["location"] = [longitude, latitude]
                elif step_coords:
                    maneuver_payload["location"] = step_coords[-1]

                leg_steps.append(
                    {
                        "maneuver": maneuver_payload,
                        "distance": float(step.get("distanceMeters", 0) or 0),
                        "duration": float(parse_google_duration_s(step.get("staticDuration") or step.get("duration"))),
                        "name": "",
                        "geometry": {"type": "LineString", "coordinates": step_coords},
                    }
                )
            legs.append({"steps": leg_steps})

        return {
            "routes": [
                {
                    "distance": float(route.get("distanceMeters", 0) or 0),
                    "duration": float(parse_google_duration_s(route.get("duration"))),
                    "legs": legs,
                    "geometry": {"type": "LineString", "coordinates": route_coords},
                }
            ],
            "waypoints": [
                {"location": [from_lon, from_lat], "name": "origin"},
                {"location": [to_lon, to_lat], "name": "destination"},
            ],
            "source": "google",
        }

    async def get_nearby_pois(
        self,
        lat: float,
        lon: float,
        radius: float = 1000,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        categories = categories or default_search_categories()
        results_by_id: dict[str, dict[str, Any]] = {}
        for category in categories:
            places = await self._text_search_places(
                query=_category_keyword(category),
                latitude=lat,
                longitude=lon,
                radius=int(radius),
                limit=20,
            )
            for poi in parse_google_places(
                places,
                origin_lat=lat,
                origin_lon=lon,
                limit=20,
                provider="google",
            ):
                poi_id = poi.get("provider_poi_id") or f"google-{poi.get('location', '')}-{poi.get('name', '')}"
                tags = {
                    "name": poi.get("name", "Unnamed"),
                    category: poi.get("type", "unknown"),
                    "type": poi.get("type", "unknown"),
                    "address": poi.get("address", ""),
                    "source": "google",
                }
                results_by_id[str(poi_id)] = {
                    "id": str(poi_id),
                    "lat": poi["coordinates"]["latitude"],
                    "lon": poi["coordinates"]["longitude"],
                    "tags": tags,
                }

        return list(results_by_id.values())

    async def search_features_by_category(
        self,
        bbox: tuple[float, float, float, float],
        category: str,
        subcategories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2
        rough_radius = max(200, int(max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])) * 111000))
        categories = subcategories if subcategories else [category]
        pois = await self.get_nearby_pois(center_lat, center_lon, rough_radius, categories)
        return [
            {
                "id": poi.get("id"),
                "type": "node",
                "lat": poi.get("lat"),
                "lon": poi.get("lon"),
                "tags": poi.get("tags", {}),
            }
            for poi in pois
        ]


def create_map_client() -> tuple[MapProviderClient, str]:
    provider = os.getenv("MAP_PROVIDER", "osm").strip().lower()
    if provider == "amap":
        amap_key = os.getenv("AMAP_KEY", "").strip()
        if not amap_key:
            raise RuntimeError("MAP_PROVIDER=amap requires AMAP_KEY")
        return AMapClient(api_key=amap_key), "amap"
    if provider == "baidu":
        baidu_ak = os.getenv("BAIDU_AK", "").strip()
        if not baidu_ak:
            raise RuntimeError("MAP_PROVIDER=baidu requires BAIDU_AK")
        return BaiduClient(ak=baidu_ak), "baidu"
    if provider == "google":
        google_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
        if not google_key:
            raise RuntimeError("MAP_PROVIDER=google requires GOOGLE_MAPS_API_KEY")
        return GoogleMapsClient(
            api_key=google_key,
            language=os.getenv("GOOGLE_MAPS_LANGUAGE", "").strip(),
            region=os.getenv("GOOGLE_MAPS_REGION", "").strip(),
        ), "google"
    return OSMClient(), "osm"


__all__ = [
    "AMapClient",
    "BaiduClient",
    "GoogleMapsClient",
    "MapProviderClient",
    "OSMClient",
    "create_map_client",
    "decode_google_polyline",
    "format_location",
    "infer_action",
    "parse_amap_pois",
    "parse_amap_steps",
    "parse_baidu_pois",
    "parse_baidu_steps",
    "parse_google_places",
    "parse_google_steps",
    "parse_location_text",
    "parse_polyline",
    "strip_html",
]
