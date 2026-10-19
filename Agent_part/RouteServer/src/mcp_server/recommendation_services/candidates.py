"""Candidates responsibilities for destination recommendation."""
from __future__ import annotations

from typing import Any
import math
import re
from mcp_server.recommendation_services.policy import _annotate_candidate, _compact_candidate_tags, _display_name_from_tags, _normalize_text_key
from mcp_server.recommendation_services.geometry import haversine_m

def _extract_locality_hints(reverse_geocode_result: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    if not isinstance(reverse_geocode_result, dict):
        return hints

    address = reverse_geocode_result.get("address")
    if isinstance(address, dict):
        for field in ("city", "district", "township", "county", "state", "province"):
            value = str(address.get(field) or "").strip()
            if value:
                hints.append(value)

    display_name = str(reverse_geocode_result.get("display_name") or "").strip()
    if display_name:
        chunks = re.split(r"[,/]", display_name)
        for chunk in chunks[:3]:
            value = str(chunk).strip()
            if value:
                hints.append(value)

    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        key = _normalize_text_key(hint)
        if key and key not in seen:
            seen.add(key)
            deduped.append(hint)
    return deduped[:4]


def _normalize_named_place_candidates(
    geocode_results: list[dict[str, Any]],
    current_latitude: float,
    current_longitude: float,
    max_direct_distance_m: float | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in geocode_results:
        try:
            lat = float(item.get("lat"))
            lon = float(item.get("lon"))
        except Exception:
            continue

        source = str(item.get("source", "geocode")).strip() or "geocode"
        display_name = str(item.get("display_name", "Named Place")).strip() or "Named Place"
        candidate_id = f"named:{source}:{round(lat, 6)}:{round(lon, 6)}"
        if candidate_id in seen:
            continue

        direct_distance = haversine_m(current_latitude, current_longitude, lat, lon)
        if isinstance(max_direct_distance_m, (int, float)) and direct_distance > float(max_direct_distance_m):
            continue

        seen.add(candidate_id)
        raw_item = item.get("raw", item)
        raw_type = ""
        raw_address = display_name
        if isinstance(raw_item, dict):
            raw_type = str(raw_item.get("type") or raw_item.get("tag") or "").strip()
            raw_address = str(raw_item.get("address") or raw_item.get("formatted_address") or display_name).strip() or display_name
        raw_candidate = {
            "name": display_name,
            "type": raw_type or "named_place",
            "address": raw_address,
            "provider": source,
            "raw": raw_item,
        }
        normalized.append(
            {
                "id": candidate_id,
                "name": display_name,
                "latitude": lat,
                "longitude": lon,
                "location": f"{lon},{lat}",
                "tags": {
                    "type": raw_type or "named_place",
                    "address": raw_address,
                    "provider": source,
                    "provider_poi_id": candidate_id,
                },
                "raw_candidate": raw_candidate,
                "direct_distance_m": round(direct_distance, 1),
            }
        )

    normalized.sort(key=lambda item: item["direct_distance_m"])
    return normalized


def _normalize_semantic_poi_candidates(
    candidates_raw: list[dict[str, Any]],
    *,
    destination_kind: str,
) -> list[dict[str, Any]]:
    normalized_candidates: list[dict[str, Any]] = []
    for item in candidates_raw:
        coordinates = item.get("coordinates", {}) or {}
        lat = coordinates.get("latitude")
        lon = coordinates.get("longitude")
        if lat is None or lon is None:
            continue

        name = item.get("name") or "Unnamed"
        normalized_candidates.append(
            _annotate_candidate(
                {
                    "id": item.get("provider_poi_id") or item.get("location") or f"{lat}-{lon}-{name}",
                    "name": name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "location": item.get("location") or f"{lon},{lat}",
                    "tags": _compact_candidate_tags(item),
                    "raw_candidate": item,
                    "direct_distance_m": float(item.get("distance_m", 0) or 0),
                },
                destination_kind=destination_kind,
            )
        )
    return normalized_candidates


def _normalize_nearby_poi_candidates(
    pois: list[dict[str, Any]],
    *,
    current_latitude: float,
    current_longitude: float,
    destination_kind: str,
    source_prefix: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in pois:
        tags = item.get("tags", {}) or {}
        lat = item.get("lat")
        lon = item.get("lon")
        if lat is None or lon is None:
            continue

        poi_type = str(tags.get("type") or "").strip()
        name = _display_name_from_tags(tags=tags, fallback_type=poi_type)
        provider_poi_id = str(item.get("id") or f"{source_prefix}:{lon}:{lat}:{name}")
        address = str(tags.get("address") or "").strip()
        normalized.append(
            _annotate_candidate(
                {
                    "id": provider_poi_id,
                    "name": name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "location": f"{float(lon)},{float(lat)}",
                    "tags": {
                        "type": poi_type,
                        "address": address,
                        "provider": str(tags.get("source") or source_prefix),
                        "provider_poi_id": provider_poi_id,
                    },
                    "raw_candidate": {
                        "name": name,
                        "type": poi_type,
                        "address": address,
                        "provider": str(tags.get("source") or source_prefix),
                        "raw": item,
                    },
                    "direct_distance_m": round(
                        haversine_m(current_latitude, current_longitude, float(lat), float(lon)),
                        1,
                    ),
                },
                destination_kind=destination_kind,
            )
        )
    return normalized


def _bbox_from_center(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    lat_delta = radius_m / 111000
    lon_divisor = max(0.2, math.cos(math.radians(lat)))
    lon_delta = radius_m / (111000 * lon_divisor)
    return (lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta)


def _normalize_feature_candidates(
    features: list[dict[str, Any]],
    *,
    current_latitude: float,
    current_longitude: float,
    destination_kind: str,
    source_prefix: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in features:
        center = item.get("center", {}) or {}
        lat = item.get("lat", center.get("lat"))
        lon = item.get("lon", center.get("lon"))
        if lat is None or lon is None:
            continue

        tags = item.get("tags", {}) or {}
        name = str(tags.get("name") or item.get("id") or "Unnamed").strip() or "Unnamed"
        provider_poi_id = str(item.get("id") or f"{source_prefix}:{lon}:{lat}:{name}")

        poi_type_parts = []
        for key in ("tourism", "leisure", "natural", "historic"):
            value = str(tags.get(key) or "").strip()
            if value:
                poi_type_parts.append(value)
        poi_type = ";".join(poi_type_parts) or "unknown"
        name = _display_name_from_tags(tags=tags, fallback_type=poi_type)
        address = str(tags.get("addr:full") or tags.get("address") or "").strip()

        normalized.append(
            _annotate_candidate(
                {
                    "id": provider_poi_id,
                    "name": name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "location": f"{float(lon)},{float(lat)}",
                    "tags": {
                        "type": poi_type,
                        "address": address,
                        "provider": source_prefix,
                        "provider_poi_id": provider_poi_id,
                    },
                    "raw_candidate": {
                        "name": name,
                        "type": poi_type,
                        "address": address,
                        "provider": source_prefix,
                        "raw": item,
                    },
                    "direct_distance_m": round(
                        haversine_m(current_latitude, current_longitude, float(lat), float(lon)),
                        1,
                    ),
                },
                destination_kind=destination_kind,
            )
        )
    return normalized
