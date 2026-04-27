from __future__ import annotations

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL.ExifTags import GPSTAGS, TAGS
import argparse
import asyncio
import json
import logging
import math
import hashlib
import os
from pathlib import Path
from time import perf_counter
import time
import re
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import urlopen

from mcp_server.map_providers import encode_polyline_text, sample_route_coordinates


DEFAULT_IMAGE_PATH = "/home/ubuntu/WorkSpace/project/SR/test4.jpg"
DEFAULT_ORIGIN_NAME = "浙江大学玉泉校区南门 杭州"
DEFAULT_DESTINATION_NAME = "浙江大学玉泉校区北门 杭州"
DEFAULT_PERF_PROFILE = "latency"
DEFAULT_CACHE_PATH = str(Path.home() / ".cache" / "urban_nav" / "yuquan_demo_cache.json")

TIMING_LOGGER = logging.getLogger("yuquan.demo.timing")

PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "latency": {
        "visual_top_k": 3,
        "candidate_top_k": 3,
        "llm_detail": "low",
        "enable_verifier": False,
        "auto_local_bev_svg": False,
        "auto_map_bev_html": False,
        "auto_map_bev_png": False,
        "auto_planner_route_html": False,
        "auto_planner_route_png": False,
    },
    "balanced": {
        "visual_top_k": 4,
        "candidate_top_k": 4,
        "llm_detail": "auto",
        "enable_verifier": True,
        "auto_local_bev_svg": False,
        "auto_map_bev_html": False,
        "auto_map_bev_png": False,
        "auto_planner_route_html": True,
        "auto_planner_route_png": False,
    },
    "debug": {
        "visual_top_k": 5,
        "candidate_top_k": 5,
        "llm_detail": "auto",
        "enable_verifier": True,
        "auto_local_bev_svg": True,
        "auto_map_bev_html": True,
        "auto_map_bev_png": False,
        "auto_planner_route_html": True,
        "auto_planner_route_png": False,
    },
}

CACHE_VERSION = 1
DIRECT_AMAP_LOOKUP_TIMEOUT_S = max(float(os.getenv("YUQUAN_AMAP_LOOKUP_TIMEOUT_S", "20") or 20.0), 1.0)
DIRECT_AMAP_LOOKUP_RETRIES = max(int(os.getenv("YUQUAN_AMAP_LOOKUP_RETRIES", "2") or 2), 0)
DIRECT_AMAP_LOOKUP_BACKOFF_S = max(float(os.getenv("YUQUAN_AMAP_LOOKUP_BACKOFF_S", "1.0") or 1.0), 0.0)


def _setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def _log_timing(stage: str, elapsed_s: float, **fields: Any) -> None:
    payload = {
        "component": "yuquan_demo",
        "stage": stage,
        "elapsed_ms": round(float(elapsed_s) * 1000.0, 2),
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    TIMING_LOGGER.info("TIMING %s", json.dumps(payload, ensure_ascii=True, sort_keys=True))


def _profile_settings(profile: str) -> dict[str, Any]:
    return dict(PROFILE_DEFAULTS.get(str(profile or DEFAULT_PERF_PROFILE), PROFILE_DEFAULTS[DEFAULT_PERF_PROFILE]))


def _default_cache_payload() -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "locations": {},
        "routes": {},
    }


def _load_json_cache(cache_path: str) -> dict[str, Any]:
    path = Path(cache_path).expanduser()
    if not path.exists():
        return _default_cache_payload()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_cache_payload()
    if not isinstance(payload, dict) or int(payload.get("version", -1)) != CACHE_VERSION:
        return _default_cache_payload()
    payload.setdefault("locations", {})
    payload.setdefault("routes", {})
    return payload


def _save_json_cache(cache_path: str, payload: dict[str, Any]) -> None:
    path = Path(cache_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _stable_cache_key(parts: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _location_cache_key(*, coordinate: str | None, name: str) -> str:
    if coordinate:
        return _stable_cache_key(["coordinate", str(coordinate).strip()])
    return _stable_cache_key(["name", _normalize_text(name)])


def _route_cache_key(*, origin: str, destination: str, mode: str) -> str:
    return _stable_cache_key(["route", str(origin).strip(), str(destination).strip(), str(mode).strip().lower()])


def _build_fpv_segmentation_payload(
    *,
    segmentation_npy_path: str | None,
    class_mapping_path: str | None,
    visualization_path: str | None,
) -> dict[str, Any] | None:
    raw_npy = str(segmentation_npy_path or "").strip()
    if not raw_npy:
        return None
    npy_path = Path(raw_npy).expanduser()
    mapping_path = Path(str(class_mapping_path).strip()).expanduser() if str(class_mapping_path or "").strip() else npy_path.with_name("class_mapping.json")
    payload: dict[str, Any] = {
        "segmentation_npy_path": str(npy_path),
        "class_mapping_path": str(mapping_path),
    }
    raw_vis = str(visualization_path or "").strip()
    if raw_vis:
        payload["visualization_path"] = str(Path(raw_vis).expanduser())
    return payload


def _urlopen_json_with_retries(url: str, *, timeout_s: float, retries: int, backoff_s: float) -> dict[str, Any]:
    attempts = max(int(retries), 0) + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8", errors="ignore"))
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if backoff_s > 0:
                time.sleep(backoff_s * attempt)
    assert last_error is not None
    raise RuntimeError(f"{last_error}")


def first_text_content(result: Any) -> str:
    if not result or not getattr(result, "content", None):
        return ""
    for content_item in result.content:
        if content_item.type == "text" and isinstance(content_item.text, str):
            return content_item.text
    return ""


_JSON_PARSE_MISSING = object()


def parse_first_text_payload(result: Any) -> Any:
    text = first_text_content(result)
    if not text:
        return _JSON_PARSE_MISSING
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _JSON_PARSE_MISSING


async def call_tool_json(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> Any:
    result = await session.call_tool(tool_name, arguments)
    payload = parse_first_text_payload(result)
    if payload is not _JSON_PARSE_MISSING:
        return payload
    raw_text = first_text_content(result)
    raw_text = raw_text.strip()
    if raw_text.startswith("Error executing tool "):
        raise RuntimeError(raw_text)
    raise RuntimeError(f"{tool_name} returned non-JSON payload: {raw_text}")


def _rational_to_float(value: Any) -> float:
    try:
        return float(value)
    except TypeError:
        return float(value[0]) / float(value[1])


def _dms_to_decimal(dms: Any, ref: str) -> float:
    value = (
        _rational_to_float(dms[0])
        + _rational_to_float(dms[1]) / 60.0
        + _rational_to_float(dms[2]) / 3600.0
    )
    return -value if ref in ("S", "W") else value


def read_image_pose_from_exif(image_path: str) -> dict[str, Any]:
    image = Image.open(image_path)
    exif = image.getexif()
    display_image = ImageOps.exif_transpose(image)
    named = {TAGS.get(tag_id, tag_id): value for tag_id, value in exif.items()}
    gps_tag_id = next(
        (tag_id for tag_id in exif if TAGS.get(tag_id, tag_id) == "GPSInfo"),
        None,
    )
    if gps_tag_id is None:
        raise RuntimeError(f"No GPSInfo found in image EXIF: {image_path}")

    gps = {GPSTAGS.get(k, k): v for k, v in exif.get_ifd(gps_tag_id).items()}
    latitude = _dms_to_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
    longitude = _dms_to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    heading = None
    if "GPSImgDirection" in gps:
        heading = _rational_to_float(gps["GPSImgDirection"])

    return {
        "image_path": image_path,
        "image_size": {"width": display_image.size[0], "height": display_image.size[1]},
        "raw_image_size": {"width": image.size[0], "height": image.size[1]},
        "latitude": latitude,
        "longitude": longitude,
        "heading_deg": heading,
        "datetime": named.get("DateTime"),
        "model": named.get("Model"),
        "gps_h_error_m": _rational_to_float(gps["GPSHPositioningError"])
        if "GPSHPositioningError" in gps
        else None,
    }


def parse_lnglat(location: str) -> tuple[float, float]:
    lon_text, lat_text = str(location).split(",", 1)
    return float(lon_text), float(lat_text)


def format_lnglat(longitude: float, latitude: float) -> str:
    return f"{longitude},{latitude}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def _outside_china(latitude: float, longitude: float) -> bool:
    return longitude < 72.004 or longitude > 137.8347 or latitude < 0.8293 or latitude > 55.8271


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(longitude: float, latitude: float) -> tuple[float, float]:
    if _outside_china(latitude, longitude):
        return longitude, latitude
    a = 6378245.0
    ee = 0.00669342162296594323
    d_lat = _transform_lat(longitude - 105.0, latitude - 35.0)
    d_lon = _transform_lon(longitude - 105.0, latitude - 35.0)
    rad_lat = latitude / 180.0 * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - ee * magic * magic
    sqrt_magic = math.sqrt(magic)
    d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * math.pi)
    d_lon = (d_lon * 180.0) / (a / sqrt_magic * math.cos(rad_lat) * math.pi)
    return longitude + d_lon, latitude + d_lat


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()


def _extract_gate_hint(text: str) -> str | None:
    value = _normalize_text(text)
    gate_patterns = [
        ("south", ("南门", "南校门", "southgate", "southentrance")),
        ("north", ("北门", "北校门", "northgate", "northentrance")),
        ("east", ("东门", "东校门", "eastgate", "eastentrance")),
        ("west", ("西门", "西校门", "westgate", "westentrance")),
    ]
    for gate, patterns in gate_patterns:
        if any(pattern in value for pattern in patterns):
            return gate
    return None


def _place_name_score(query_name: str, result: dict[str, Any]) -> float:
    query = _normalize_text(query_name)
    name = _normalize_text(result.get("name") or result.get("display_name"))
    address = _normalize_text(result.get("address") or result.get("formatted_address"))
    location = _normalize_text(result.get("location"))
    haystack = f"{name}|{address}|{location}"

    score = 0.0
    query_gate = _extract_gate_hint(query)
    result_gate = _extract_gate_hint(haystack)
    if query_gate is not None:
        if result_gate == query_gate:
            score += 0.7
        elif result_gate is not None and result_gate != query_gate:
            score -= 0.9

    query_tokens = [
        token for token in re.split(r"[^\w\u3400-\u9fff]+", str(query_name or ""))
        if token and token not in {"杭州", "浙大", "浙江大学"}
    ]
    for token in query_tokens:
        token_norm = _normalize_text(token)
        if not token_norm:
            continue
        if token_norm in name:
            score += 0.18
        elif token_norm in address:
            score += 0.08

    if query and query in name:
        score += 0.45
    if "玉泉" in query and "玉泉" in haystack:
        score += 0.16
    if "校区" in query and "校区" in haystack:
        score += 0.08

    try:
        distance = float(result.get("distance_m", result.get("distance", 999999)) or 999999)
    except Exception:
        distance = 999999.0
    score -= min(distance / 2500.0, 0.35)
    return score


def _amap_text_search_queries(name: str) -> list[str]:
    text = str(name or "").strip()
    if not text:
        return []
    variants = [text]
    without_city = re.sub(r"\s*杭州(?:市)?\s*$", "", text).strip()
    if without_city and without_city not in variants:
        variants.append(without_city)
    return variants


def _fetch_amap_place_text_candidates(name: str) -> list[dict[str, Any]]:
    amap_key = _amap_key_from_env_or_url()
    if not amap_key:
        raise RuntimeError("AMAP_KEY is required for direct AMap place/text lookup")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in _amap_text_search_queries(name):
        url = (
            "https://restapi.amap.com/v3/place/text?"
            + urlencode(
                {
                    "key": amap_key,
                    "keywords": query,
                    "city": "杭州",
                    "citylimit": "true",
                    "offset": 20,
                    "page": 1,
                    "extensions": "base",
                }
            )
        )
        data = _urlopen_json_with_retries(
            url,
            timeout_s=DIRECT_AMAP_LOOKUP_TIMEOUT_S,
            retries=DIRECT_AMAP_LOOKUP_RETRIES,
            backoff_s=DIRECT_AMAP_LOOKUP_BACKOFF_S,
        )
        if data.get("status") != "1":
            raise RuntimeError(f"AMap place/text failed for {query!r}: {data.get('info') or data}")
        for poi in data.get("pois", []) or []:
            location = str(poi.get("location") or "").strip()
            if not location or location in seen:
                continue
            try:
                lon, lat = parse_lnglat(location)
            except Exception:
                continue
            seen.add(location)
            name_text = str(poi.get("name") or "").strip()
            address = str(poi.get("address") or "").strip()
            results.append(
                {
                    "provider": "amap",
                    "provider_poi_id": str(poi.get("id") or location),
                    "name": name_text,
                    "display_name": f"{name_text} ({address})" if address else name_text,
                    "location": format_lnglat(lon, lat),
                    "coordinates": {"longitude": lon, "latitude": lat},
                    "distance_m": poi.get("distance"),
                    "type": poi.get("type"),
                    "address": address,
                    "typecode": poi.get("typecode"),
                    "raw": poi,
                    "source": "amap_place_text",
                    "query": query,
                }
            )
    if not results:
        raise RuntimeError(f"No AMap place/text candidates for {name!r}")
    return results


def _fetch_amap_inputtips_candidates(name: str) -> list[dict[str, Any]]:
    amap_key = _amap_key_from_env_or_url()
    if not amap_key:
        raise RuntimeError("AMAP_KEY is required for direct AMap inputtips lookup")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in _amap_text_search_queries(name):
        url = (
            "https://restapi.amap.com/v3/assistant/inputtips?"
            + urlencode(
                {
                    "key": amap_key,
                    "keywords": query,
                    "city": "杭州",
                    "citylimit": "true",
                }
            )
        )
        data = _urlopen_json_with_retries(
            url,
            timeout_s=DIRECT_AMAP_LOOKUP_TIMEOUT_S,
            retries=DIRECT_AMAP_LOOKUP_RETRIES,
            backoff_s=DIRECT_AMAP_LOOKUP_BACKOFF_S,
        )
        if data.get("status") != "1":
            raise RuntimeError(f"AMap inputtips failed for {query!r}: {data.get('info') or data}")
        for tip in data.get("tips", []) or []:
            location = str(tip.get("location") or "").strip()
            if not location or location == "[]" or location in seen:
                continue
            try:
                lon, lat = parse_lnglat(location)
            except Exception:
                continue
            seen.add(location)
            name_text = str(tip.get("name") or "").strip()
            address = str(tip.get("address") or "").strip()
            results.append(
                {
                    "provider": "amap",
                    "provider_poi_id": str(tip.get("id") or location),
                    "name": name_text,
                    "display_name": f"{name_text} ({address})" if address else name_text,
                    "location": format_lnglat(lon, lat),
                    "coordinates": {"longitude": lon, "latitude": lat},
                    "typecode": tip.get("typecode"),
                    "address": address,
                    "raw": tip,
                    "source": "amap_inputtips",
                    "query": query,
                }
            )
    if not results:
        raise RuntimeError(f"No AMap inputtips candidates for {name!r}")
    return results


def location_from_candidates(results: Any, *, preferred_name: str | None = None) -> tuple[str, dict[str, Any]]:
    if not isinstance(results, list) or not results:
        raise RuntimeError("No candidates with coordinates")
    ranked_results = [result for result in results if isinstance(result, dict)]
    if preferred_name:
        ranked_results.sort(key=lambda item: _place_name_score(preferred_name, item), reverse=True)
    for result in ranked_results:
        if not isinstance(result, dict):
            continue
        location = result.get("location")
        if isinstance(location, str) and "," in location:
            lon, lat = parse_lnglat(location)
            return format_lnglat(lon, lat), result
        coordinates = (result or {}).get("coordinates") or {}
        latitude = coordinates.get("latitude", result.get("lat"))
        longitude = coordinates.get("longitude", result.get("lon"))
        if latitude is None or longitude is None:
            continue
        return format_lnglat(float(longitude), float(latitude)), result
    raise RuntimeError("No candidates with coordinates")


async def resolve_location(
    session: ClientSession,
    *,
    coordinate: str | None,
    name: str,
    search_anchor: str,
    search_radius: int,
    cache_payload: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    cache_key = _location_cache_key(coordinate=coordinate, name=name)
    location_cache = (cache_payload or {}).get("locations", {}) if isinstance(cache_payload, dict) else {}
    cached = location_cache.get(cache_key) if isinstance(location_cache, dict) else None
    if isinstance(cached, dict):
        cached_location = str(cached.get("location") or "").strip()
        cached_selected = cached.get("selected")
        if cached_location and isinstance(cached_selected, dict):
            _log_timing(
                "cache_hit",
                0.0,
                cache_kind="location",
                name=name,
                source=cached_selected.get("source"),
            )
            return cached_location, dict(cached_selected)

    if coordinate:
        lon, lat = parse_lnglat(coordinate)
        location = format_lnglat(lon, lat)
        selected = {"source": "argument", "name": name, "coordinates": {"latitude": lat, "longitude": lon}}
        if isinstance(location_cache, dict):
            location_cache[cache_key] = {"location": location, "selected": selected}
        return location, selected

    errors: list[str] = []
    try:
        started_at = perf_counter()
        results = await asyncio.to_thread(_fetch_amap_inputtips_candidates, name)
        _log_timing(
            "resolve_location_direct_amap",
            perf_counter() - started_at,
            source="amap_inputtips",
            query_name=name,
            candidate_count=len(results),
        )
        location, selected = location_from_candidates(results, preferred_name=name)
        selected = dict(selected)
        selected["source"] = "amap_inputtips_direct"
        _log_timing(
            "resolve_location_attempt",
            perf_counter() - started_at,
            source="amap_inputtips_direct",
            name=name,
            status="ok",
        )
        _log_timing(
            "resolve_location",
            perf_counter() - started_at,
            source="amap_inputtips_direct",
            name=name,
            status="ok",
        )
        if isinstance(location_cache, dict):
            location_cache[cache_key] = {"location": location, "selected": selected}
        return location, selected
    except Exception as exc:
        errors.append(f"amap_inputtips_direct: {exc}")

    try:
        started_at = perf_counter()
        results = await asyncio.to_thread(_fetch_amap_place_text_candidates, name)
        _log_timing(
            "resolve_location_direct_amap",
            perf_counter() - started_at,
            source="amap_place_text",
            query_name=name,
            candidate_count=len(results),
        )
        location, selected = location_from_candidates(results, preferred_name=name)
        selected = dict(selected)
        selected["source"] = "amap_place_text_direct"
        _log_timing(
            "resolve_location_attempt",
            perf_counter() - started_at,
            source="amap_place_text_direct",
            name=name,
            status="ok",
        )
        _log_timing(
            "resolve_location",
            perf_counter() - started_at,
            source="amap_place_text_direct",
            name=name,
            status="ok",
        )
        if isinstance(location_cache, dict):
            location_cache[cache_key] = {"location": location, "selected": selected}
        return location, selected
    except Exception as exc:
        errors.append(f"amap_place_text_direct: {exc}")

    try:
        started_at = perf_counter()
        results = await call_tool_json(session, "geocode_address", {"address": name})
        location, selected = location_from_candidates(results, preferred_name=name)
        selected = dict(selected)
        selected["source"] = "geocode_address"
        selected["fallback_errors"] = errors
        _log_timing(
            "resolve_location_attempt",
            perf_counter() - started_at,
            source="geocode_address",
            name=name,
            status="ok",
        )
        _log_timing(
            "resolve_location",
            perf_counter() - started_at,
            source="geocode_address",
            name=name,
            status="ok",
        )
        if isinstance(location_cache, dict):
            location_cache[cache_key] = {"location": location, "selected": selected}
        return location, selected
    except Exception as exc:
        errors.append(f"geocode_address: {exc}")

    raise RuntimeError(f"Unable to resolve location '{name}'. " + " | ".join(errors))


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
                    longitude, latitude = parse_lnglat(location)
                except Exception:
                    latitude = None
                    longitude = None
        if latitude is None or longitude is None:
            continue
        point = {"longitude": float(longitude), "latitude": float(latitude)}
        if (
            points
            and abs(points[-1]["latitude"] - point["latitude"]) < 1e-9
            and abs(points[-1]["longitude"] - point["longitude"]) < 1e-9
        ):
            continue
        points.append(point)
    return points


def _scaled_point(point: Any, scale: float) -> tuple[float, float] | None:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        return float(point[0]) * scale, float(point[1]) * scale
    except Exception:
        return None


def _draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    size: float,
    fill: tuple[int, int, int, int],
) -> None:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    norm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
    ux, uy = vx / norm, vy / norm
    px, py = -uy, ux
    p1 = (end[0], end[1])
    p2 = (end[0] - ux * size - px * size * 0.55, end[1] - uy * size - py * size * 0.55)
    p3 = (end[0] - ux * size + px * size * 0.55, end[1] - uy * size + py * size * 0.55)
    draw.polygon([p1, p2, p3], fill=fill)


def _write_text(path: str, content: str) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return str(output)


def _save_remote_image(url: str, output_path: str, timeout_s: float = 20.0) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout_s) as response:
        output.write_bytes(response.read())
    return str(output)


def _build_map_snapshot_html(*, title: str, image_url: str, metadata: dict[str, Any]) -> str:
    metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; background: #0f172a; color: #e2e8f0; margin: 24px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #334155; border-radius: 12px; background: white; }}
    pre {{ white-space: pre-wrap; background: #111827; padding: 16px; border-radius: 12px; overflow: auto; }}
    a {{ color: #7dd3fc; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p><a href="{image_url}" target="_blank" rel="noreferrer">Open raw AMap snapshot URL</a></p>
  <img src="{image_url}" alt="{title}" />
  <pre>{metadata_json}</pre>
</body>
</html>
"""


def _amap_key_from_env_or_url(fallback_url: str | None = None) -> str:
    for env_name in ("AMAP_KEY", "AMAP_API_KEY", "GAODE_API_KEY"):
        key = str(os.getenv(env_name, "")).strip()
        if key:
            return key
    if fallback_url:
        try:
            values = parse_qs(urlparse(fallback_url).query).get("key") or []
            return str(values[0]).strip() if values else ""
        except Exception:
            return ""
    return ""


def _route_point_lnglat(item: Any) -> list[float] | None:
    if isinstance(item, dict):
        longitude = item.get("longitude", item.get("lng", item.get("lon")))
        latitude = item.get("latitude", item.get("lat"))
        if longitude is None or latitude is None:
            coordinate = item.get("coordinate")
            if isinstance(coordinate, dict):
                longitude = coordinate.get("longitude", coordinate.get("lng", coordinate.get("lon")))
                latitude = coordinate.get("latitude", coordinate.get("lat"))
        if longitude is None or latitude is None:
            location = item.get("location")
            if isinstance(location, str) and "," in location:
                try:
                    longitude, latitude = parse_lnglat(location)
                except Exception:
                    longitude = None
                    latitude = None
        if longitude is None or latitude is None:
            return None
        return [float(longitude), float(latitude)]
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return [float(item[0]), float(item[1])]
    return None


def _normalize_route_polyline_lnglat(route_polyline: list[Any]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for item in route_polyline or []:
        point = _route_point_lnglat(item)
        if point is not None:
            normalized.append(point)
    return normalized


def _add_amap_marker(image_url: str, *, label: str, color: str, location: str) -> str:
    if not image_url or not location:
        return image_url
    parsed = urlparse(image_url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    marker = f"mid,{color},{label}:{location}"
    updated_pairs: list[tuple[str, str]] = []
    marker_added = False
    for key, value in query_pairs:
        if key == "markers" and value:
            if marker not in value:
                value = f"{value}|{marker}"
            marker_added = True
        updated_pairs.append((key, value))
    if not marker_added:
        updated_pairs.append(("markers", marker))
    return urlunparse(parsed._replace(query=urlencode(updated_pairs, doseq=True, safe=",:;|*")))


def _build_planner_route_overview_url(
    *,
    route_polyline: list[Any],
    origin_location: str,
    destination_location: str,
    current_location: str | None = None,
    amap_key: str | None = None,
    size: str = "800*600",
    scale: int = 2,
) -> str | None:
    amap_key = str(amap_key or "").strip() or _amap_key_from_env_or_url()
    if not amap_key:
        return None
    normalized = _normalize_route_polyline_lnglat(route_polyline)
    sampled = sample_route_coordinates(normalized, max_points=220)
    if not sampled:
        return None
    path_spec = f"7,0x2E8B57,1,,:{encode_polyline_text(sampled)}"
    markers = [
        f"mid,0x2E86FF,O:{origin_location}",
        f"mid,0xFF5533,D:{destination_location}",
    ]
    if current_location:
        markers.append(f"mid,0xFF8C42,P:{current_location}")
    return (
        "https://restapi.amap.com/v3/staticmap"
        f"?key={amap_key}"
        f"&size={size}"
        f"&scale={int(scale)}"
        "&traffic=0"
        f"&paths={path_spec}"
        f"&markers={'|'.join(markers)}"
    )


def render_prompt_preview_png(
    *,
    image_path: str,
    rendered_prompt: dict[str, Any] | None,
    output_path: str,
    max_width: int = 1200,
) -> str:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    src_width, src_height = image.size
    scale = min(float(max_width) / max(float(src_width), 1.0), 1.0)
    dst_width = max(1, int(round(src_width * scale)))
    dst_height = max(1, int(round(src_height * scale)))
    preview = image.resize((dst_width, dst_height), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if isinstance(rendered_prompt, dict):
        prompt_type = str(rendered_prompt.get("type") or "arrow_path").lower()
        mask = rendered_prompt.get("mask")
        coverage_polygon = rendered_prompt.get("coverage_polygon_uv")
        path_samples = rendered_prompt.get("path_samples")
        if isinstance(coverage_polygon, list) and len(coverage_polygon) >= 3:
            polygon = [pt for pt in (_scaled_point(point, scale) for point in coverage_polygon) if pt is not None]
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=(34, 197, 94, 60), outline=(34, 197, 94, 230))
        if isinstance(mask, list) and len(mask) >= 3:
            polygon = [pt for pt in (_scaled_point(point, scale) for point in mask) if pt is not None]
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=(56, 189, 248, 58), outline=(56, 189, 248, 230))
        if isinstance(path_samples, list):
            for sample in path_samples:
                if not isinstance(sample, dict):
                    continue
                center = _scaled_point(sample.get("center_uv"), scale)
                left_edge = _scaled_point(sample.get("left_edge_uv"), scale)
                right_edge = _scaled_point(sample.get("right_edge_uv"), scale)
                if center and left_edge and right_edge:
                    draw.line([left_edge, right_edge], fill=(250, 204, 21, 210), width=max(2, int(round(scale * 10))))
                    r_edge = max(3, int(round(scale * 18)))
                    r_center = max(3, int(round(scale * 14)))
                    draw.ellipse(
                        [left_edge[0] - r_edge, left_edge[1] - r_edge, left_edge[0] + r_edge, left_edge[1] + r_edge],
                        fill=(253, 224, 71, 255),
                    )
                    draw.ellipse(
                        [right_edge[0] - r_edge, right_edge[1] - r_edge, right_edge[0] + r_edge, right_edge[1] + r_edge],
                        fill=(253, 224, 71, 255),
                    )
                    draw.ellipse(
                        [center[0] - r_center, center[1] - r_center, center[0] + r_center, center[1] + r_center],
                        fill=(251, 113, 133, 255),
                    )

        path_uv = rendered_prompt.get("path_uv")
        path_points = []
        if isinstance(path_uv, list):
            path_points = [pt for pt in (_scaled_point(point, scale) for point in path_uv) if pt is not None]

        span = max(float(rendered_prompt.get("span_px") or 120.0) * scale, 24.0)
        stroke_width = max(int(round(min(dst_width, dst_height) * 0.012)), 6)
        arrow_color = (249, 115, 22, 255)

        if prompt_type in {"arrival", "arrival_marker", "destination_marker"}:
            center = _scaled_point(rendered_prompt.get("center_uv"), scale) or (dst_width * 0.5, dst_height * 0.68)
            radius = max(int(round(min(dst_width, dst_height) * 0.05)), 24)
            draw.ellipse(
                [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
                fill=(34, 197, 94, 70),
                outline=(34, 197, 94, 255),
                width=max(stroke_width // 2, 3),
            )
            draw.ellipse(
                [center[0] - radius * 0.28, center[1] - radius * 0.28, center[0] + radius * 0.28, center[1] + radius * 0.28],
                fill=(34, 197, 94, 255),
            )
            draw.text((center[0], center[1] + radius * 1.35), "ARRIVED", anchor="mm", fill=(220, 252, 231, 255))
        elif len(path_points) >= 2:
            draw.line(path_points, fill=arrow_color, width=stroke_width, joint="curve")
            _draw_arrow_head(
                draw,
                start=path_points[-2],
                end=path_points[-1],
                size=max(span * 0.16, stroke_width * 2.6),
                fill=arrow_color,
            )
        else:
            center = _scaled_point(rendered_prompt.get("center_uv"), scale) or (dst_width * 0.5, dst_height * 0.55)
            anchor = (dst_width * 0.5, dst_height * 0.92)
            draw.line([anchor, center], fill=arrow_color, width=stroke_width)
            _draw_arrow_head(draw, start=anchor, end=center, size=max(span * 0.16, stroke_width * 2.6), fill=arrow_color)

        center = _scaled_point(rendered_prompt.get("center_uv"), scale)
        if center is not None:
            marker_r = max(stroke_width, 8)
            draw.ellipse(
                [center[0] - marker_r, center[1] - marker_r, center[0] + marker_r, center[1] + marker_r],
                fill=(226, 232, 240, 255),
            )
            label = str(rendered_prompt.get("semantic_tag") or prompt_type)[:40]
            text_xy = (min(center[0] + marker_r * 1.7, dst_width - 10), max(center[1] - marker_r * 2.0, 10))
            draw.text(text_xy, label, fill=(226, 232, 240, 255), font=ImageFont.load_default())
            termination = str(rendered_prompt.get("termination_semantics") or "").strip()
            if termination:
                term_xy = (min(center[0] + marker_r * 1.7, dst_width - 10), min(center[1] + marker_r * 1.5, dst_height - 18))
                draw.text(term_xy, termination[:36], fill=(187, 247, 208, 255), font=ImageFont.load_default())
    else:
        draw.text((dst_width * 0.5, dst_height * 0.5), "No rendered prompt", anchor="mm", fill=(252, 165, 165, 255))

    composed = Image.alpha_composite(preview.convert("RGBA"), overlay).convert("RGB")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    composed.save(out, format="PNG")
    return str(out)


def _compact_route_context(route_context: dict[str, Any]) -> dict[str, Any]:
    pose = route_context.get("pose_match", {}) or {}
    progress = route_context.get("progress", {}) or {}
    local = route_context.get("local_semantics", {}) or {}
    geometry = route_context.get("geometry", {}) or {}
    return {
        "off_route": pose.get("off_route"),
        "route_confidence": pose.get("route_confidence"),
        "cross_track_error_m": pose.get("cross_track_error_m"),
        "heading_delta_deg": pose.get("heading_delta_deg"),
        "remaining_distance_m": progress.get("remaining_distance_m"),
        "distance_to_next_turn_m": progress.get("distance_to_next_turn_m"),
        "progress_ratio": progress.get("progress_ratio"),
        "turn_semantics": local.get("turn_semantics"),
        "corridor_type": local.get("corridor_type"),
        "instruction_hint": local.get("instruction_hint"),
        "local_segment_polyline": geometry.get("local_segment_polyline", []),
    }


def _compact_arbiter_decision(decision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    compact = {
        key: decision.get(key)
        for key in (
            "selected_prompt_type",
            "instruction",
            "confidence",
            "reason",
            "reject_all",
            "fallback_used",
            "arbiter_source",
            "rendered_prompt",
        )
    }
    debug = decision.get("debug", {}) or {}
    if debug.get("llm_error"):
        compact["llm_error"] = debug.get("llm_error")
    if isinstance(debug.get("llm_raw"), dict):
        raw = debug["llm_raw"]
        compact["llm_raw_summary"] = {
            key: raw.get(key)
            for key in ("selected_prompt_type", "instruction", "confidence", "reason", "reject_all")
        }
    for debug_key in ("generator_raw", "verifier_raw"):
        if isinstance(debug.get(debug_key), dict):
            raw = debug[debug_key]
            compact[f"{debug_key}_summary"] = {
                key: raw.get(key)
                for key in ("selected_prompt_type", "instruction", "confidence", "reason", "reject_all")
            }
            rendered_prompt = raw.get("rendered_prompt")
            if isinstance(rendered_prompt, dict):
                compact[f"{debug_key}_rendered_prompt"] = {
                    key: rendered_prompt.get(key)
                    for key in (
                        "type",
                        "center_uv",
                        "path_uv",
                        "path_samples",
                        "coverage_polygon_uv",
                        "termination_semantics",
                        "support_confidence",
                    )
                }
    if debug.get("verifier_error"):
        compact["verifier_error"] = debug.get("verifier_error")
    return compact


def _compact_map_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    return {
        "snapshot_provider": snapshot.get("snapshot_provider"),
        "route_summary": snapshot.get("route_summary"),
        "image_url": snapshot.get("image_url"),
        "image_url_cache_busted": snapshot.get("image_url_cache_busted"),
        "refresh_interval_s": snapshot.get("refresh_interval_s"),
        "generated_at": snapshot.get("generated_at"),
    }


async def run_demo(
    *,
    image_path: str,
    fpv_segmentation: dict[str, Any] | None,
    fpv_segmentation_overlay_image_path: str | None,
    fpv_segmentation_overlay_image_url: str | None,
    enforce_alignment: bool,
    origin: str | None,
    destination: str | None,
    origin_name: str,
    destination_name: str,
    mode: str,
    gps_source: str,
    heading_deg: float | None,
    lookahead_m: float,
    off_route_threshold_m: float,
    visual_top_k: int,
    candidate_top_k: int,
    llm_detail: str,
    perf_profile: str,
    enable_verifier: bool,
    enable_segmentation_overlay_input: bool,
    save_overlay: str,
    save_preview_png: str | None,
    save_local_bev_svg: str | None,
    save_map_bev_png: str | None,
    save_map_bev_html: str | None,
    save_planner_route_png: str | None,
    save_planner_route_html: str | None,
    save_json: str | None,
    cache_path: str,
) -> None:
    overall_started_at = perf_counter()
    cache_payload = _load_json_cache(cache_path)
    image_pose = read_image_pose_from_exif(image_path)
    image_gcj_longitude, image_gcj_latitude = wgs84_to_gcj02(
        float(image_pose["longitude"]),
        float(image_pose["latitude"]),
    )
    image_pose_for_amap = {
        **image_pose,
        "coordinate_system": "WGS84_EXIF",
        "amap_coordinate_system": "GCJ-02",
        "amap_longitude": image_gcj_longitude,
        "amap_latitude": image_gcj_latitude,
    }

    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            _log_timing(
                "demo_profile_selected",
                0.0,
                perf_profile=perf_profile,
                visual_top_k=visual_top_k,
                candidate_top_k=candidate_top_k,
                llm_detail=llm_detail,
                enable_verifier=enable_verifier,
                auto_enable_heavy_png=bool(save_map_bev_png or save_planner_route_png),
            )

            search_anchor = format_lnglat(image_gcj_longitude, image_gcj_latitude)
            origin_task = resolve_location(
                session,
                coordinate=origin,
                name=origin_name,
                search_anchor=search_anchor,
                search_radius=3000,
                cache_payload=cache_payload,
            )
            destination_task = resolve_location(
                session,
                coordinate=destination,
                name=destination_name,
                search_anchor=search_anchor,
                search_radius=5000,
                cache_payload=cache_payload,
            )
            origin_result, destination_result = await asyncio.gather(
                origin_task,
                destination_task,
                return_exceptions=True,
            )
            resolution_errors: list[str] = []
            if isinstance(origin_result, Exception):
                resolution_errors.append(f"origin: {origin_result}")
            if isinstance(destination_result, Exception):
                resolution_errors.append(f"destination: {destination_result}")
            if resolution_errors:
                raise RuntimeError("Location resolution failed. " + " | ".join(resolution_errors))
            origin_location, origin_geocode = origin_result
            destination_location, destination_geocode = destination_result
            _log_timing(
                "demo_stage",
                perf_counter() - overall_started_at,
                stage_name="after_resolve_locations",
            )
            if (
                str(origin_location).strip() == str(destination_location).strip()
                or str((origin_geocode or {}).get("provider_poi_id") or "").strip()
                and str((origin_geocode or {}).get("provider_poi_id") or "").strip()
                == str((destination_geocode or {}).get("provider_poi_id") or "").strip()
            ):
                raise RuntimeError(
                    "Origin and destination resolved to the same POI/location. "
                    f"origin={origin_name} -> {origin_location} ({(origin_geocode or {}).get('name')}); "
                    f"destination={destination_name} -> {destination_location} ({(destination_geocode or {}).get('name')})."
                )

            route_cache = cache_payload.get("routes", {}) if isinstance(cache_payload, dict) else {}
            route_cache_key = _route_cache_key(
                origin=origin_location,
                destination=destination_location,
                mode=mode,
            )
            cached_route_payload = route_cache.get(route_cache_key) if isinstance(route_cache, dict) else None
            if isinstance(cached_route_payload, dict):
                route_payload = dict(cached_route_payload)
                _log_timing(
                    "cache_hit",
                    0.0,
                    cache_kind="route",
                    tool="get_coarse_waypoints",
                    mode=mode,
                )
            else:
                route_started_at = perf_counter()
                try:
                    route_payload = await call_tool_json(
                        session,
                        "get_coarse_waypoints",
                        {
                            "origin": origin_location,
                            "destination": destination_location,
                            "mode": mode,
                        },
                    )
                    _log_timing(
                        "mcp_tool_call",
                        perf_counter() - route_started_at,
                        tool="get_coarse_waypoints",
                        status="ok",
                    )
                    if isinstance(route_cache, dict):
                        route_cache[route_cache_key] = route_payload
                except Exception:
                    _log_timing(
                        "mcp_tool_call",
                        perf_counter() - route_started_at,
                        tool="get_coarse_waypoints",
                        status="non_json",
                    )
                    raise
            route_polyline = route_payload.get("route_polyline") or []
            coarse_waypoints = route_payload.get("coarse_waypoints") or []
            if not route_polyline and coarse_waypoints:
                route_polyline = polyline_from_coarse_waypoints(coarse_waypoints)
            if not route_polyline or not coarse_waypoints:
                raise RuntimeError("No usable route geometry returned by get_coarse_waypoints")
            _log_timing(
                "demo_stage",
                perf_counter() - overall_started_at,
                stage_name="after_route_payload",
            )

            origin_lon, origin_lat = parse_lnglat(origin_location)
            if gps_source == "origin":
                current_latitude = origin_lat
                current_longitude = origin_lon
            else:
                current_latitude = image_gcj_latitude
                current_longitude = image_gcj_longitude

            current_heading = heading_deg if heading_deg is not None else image_pose.get("heading_deg")
            runtime_pose_location = format_lnglat(current_longitude, current_latitude)
            fpv_image_location = format_lnglat(image_gcj_longitude, image_gcj_latitude)

            agent_started_at = perf_counter()
            agent_payload = await call_tool_json(
                session,
                "run_zero_shot_visual_prompt_agent",
                {
                    "gps": {"lat": current_latitude, "lng": current_longitude},
                    "heading_deg": current_heading,
                    "route_polyline": route_polyline,
                    "coarse_waypoints": coarse_waypoints,
                    "lookahead_m": lookahead_m,
                    "off_route_threshold_m": off_route_threshold_m,
                    "image_path": image_path,
                    "fpv_segmentation": fpv_segmentation,
                    "fpv_segmentation_overlay_image_path": fpv_segmentation_overlay_image_path,
                    "fpv_segmentation_overlay_image_url": fpv_segmentation_overlay_image_url,
                    "enforce_alignment": bool(enforce_alignment),
                    "visual_top_k": visual_top_k,
                    "candidate_top_k": candidate_top_k,
                    "llm_detail": llm_detail,
                    "navigation_mode": mode,
                    "enable_verifier": enable_verifier,
                    "enable_segmentation_overlay_input": bool(enable_segmentation_overlay_input),
                },
            )
            _log_timing(
                "mcp_tool_call",
                perf_counter() - agent_started_at,
                tool="run_zero_shot_visual_prompt_agent",
                status="ok",
            )
            _log_timing(
                "demo_stage",
                perf_counter() - overall_started_at,
                stage_name="after_visual_prompt_agent",
            )

            overlay_svg = agent_payload.get("overlay_svg")
            if isinstance(overlay_svg, str) and overlay_svg.strip():
                output_path = Path(save_overlay)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(overlay_svg, encoding="utf-8")
            preview_png = None
            rendered_prompt = (agent_payload.get("arbiter_decision") or {}).get("rendered_prompt")
            if save_preview_png:
                preview_png = render_prompt_preview_png(
                    image_path=image_path,
                    rendered_prompt=rendered_prompt,
                    output_path=save_preview_png,
                )

            local_bev_payload = None
            local_bev_error = None
            if save_local_bev_svg:
                try:
                    local_bev_started_at = perf_counter()
                    local_bev_payload = await call_tool_json(
                        session,
                        "render_local_bev_patch",
                        {
                            "gps": {"lat": current_latitude, "lng": current_longitude},
                            "heading_deg": current_heading,
                            "route_polyline": route_polyline,
                            "coarse_waypoints": coarse_waypoints,
                            "lookahead_m": lookahead_m,
                            "off_route_threshold_m": off_route_threshold_m,
                        },
                    )
                    svg = str(local_bev_payload.get("svg") or "").strip()
                    if svg:
                        _write_text(save_local_bev_svg, svg)
                    _log_timing(
                        "mcp_tool_call",
                        perf_counter() - local_bev_started_at,
                        tool="render_local_bev_patch",
                        status="ok",
                    )
                    _log_timing(
                        "demo_stage",
                        perf_counter() - overall_started_at,
                        stage_name="render_local_bev_patch",
                        status="ok",
                    )
                except Exception as exc:
                    local_bev_error = str(exc)
                    _log_timing(
                        "demo_stage",
                        perf_counter() - overall_started_at,
                        stage_name="render_local_bev_patch",
                        status="failed",
                    )

            map_bev_payload = None
            map_bev_error = None
            map_bev_png = None
            map_bev_html = None
            map_image_url = ""
            planner_route_png = None
            planner_route_html = None
            planner_route_error = None
            if save_map_bev_png or save_map_bev_html:
                try:
                    map_bev_started_at = perf_counter()
                    map_bev_payload = await call_tool_json(
                        session,
                        "render_bev_route_context_snapshot",
                        {
                            "gps": {"lat": current_latitude, "lng": current_longitude},
                            "heading_deg": current_heading,
                            "route_polyline": route_polyline,
                            "coarse_waypoints": coarse_waypoints,
                            "lookahead_m": lookahead_m,
                            "off_route_threshold_m": off_route_threshold_m,
                        },
                    )
                    map_image_url = str(
                        map_bev_payload.get("image_url_cache_busted")
                        or map_bev_payload.get("image_url")
                        or ""
                    ).strip()
                    if map_image_url:
                        map_image_url = _add_amap_marker(
                            map_image_url,
                            label="P",
                            color="0xFF8C42",
                            location=fpv_image_location,
                        )
                    if save_map_bev_html and map_image_url:
                        html = _build_map_snapshot_html(
                            title="Yuquan RouteContext BEV Snapshot",
                            image_url=map_image_url,
                            metadata=map_bev_payload,
                        )
                        map_bev_html = _write_text(save_map_bev_html, html)
                    if save_map_bev_png and map_image_url:
                        map_bev_png = _save_remote_image(map_image_url, save_map_bev_png)
                    _log_timing(
                        "mcp_tool_call",
                        perf_counter() - map_bev_started_at,
                        tool="render_bev_route_context_snapshot",
                        status="ok",
                    )
                    _log_timing(
                        "demo_stage",
                        perf_counter() - overall_started_at,
                        stage_name="render_bev_route_context_snapshot",
                        status="ok",
                    )
                except (RuntimeError, URLError, OSError) as exc:
                    map_bev_error = str(exc)
                    _log_timing(
                        "demo_stage",
                        perf_counter() - overall_started_at,
                        stage_name="render_bev_route_context_snapshot",
                        status="failed",
                    )

            if save_planner_route_png or save_planner_route_html:
                try:
                    planner_started_at = perf_counter()
                    planner_amap_key = _amap_key_from_env_or_url(map_image_url)
                    planner_route_url = _build_planner_route_overview_url(
                        route_polyline=route_polyline,
                        origin_location=origin_location,
                        destination_location=destination_location,
                        current_location=fpv_image_location,
                        amap_key=planner_amap_key,
                    )
                    if planner_route_url:
                        if save_planner_route_html:
                            planner_route_html = _write_text(
                                save_planner_route_html,
                                _build_map_snapshot_html(
                                    title="Yuquan Planner Route Overview",
                                    image_url=planner_route_url,
                                    metadata={
                                        "origin": origin_location,
                                        "destination": destination_location,
                                        "fpv_image_gps": fpv_image_location,
                                        "runtime_pose": runtime_pose_location,
                                        "mode": mode,
                                        "route_summary": route_payload.get("route_summary", {}),
                                        "route_polyline_count": len(route_polyline),
                                        "image_url": planner_route_url,
                                        "overlay_legend": {
                                            "green": "planner route",
                                            "O": "resolved route origin",
                                            "D": "resolved route destination",
                                            "P": "FPV image GPS / runtime pose",
                                        },
                                    },
                                ),
                            )
                        if save_planner_route_png:
                            planner_route_png = _save_remote_image(planner_route_url, save_planner_route_png)
                        _log_timing(
                            "artifact_build",
                            perf_counter() - planner_started_at,
                            artifact="planner_route_overview",
                            status="ok",
                        )
                    else:
                        if not planner_amap_key:
                            planner_route_error = (
                                "AMAP key unavailable in client env. Set AMAP_KEY, AMAP_API_KEY, "
                                "or GAODE_API_KEY, or generate map_bev first so the key can be reused."
                            )
                        elif not route_polyline:
                            planner_route_error = "route_polyline unavailable"
                        else:
                            planner_route_error = "planner route overview URL unavailable"
                        _log_timing(
                            "artifact_build",
                            perf_counter() - planner_started_at,
                            artifact="planner_route_overview",
                            status="failed",
                        )
                except (RuntimeError, URLError, OSError) as exc:
                    planner_route_error = str(exc)
                    _log_timing(
                        "artifact_build",
                        perf_counter() - planner_started_at,
                        artifact="planner_route_overview",
                        status="failed",
                    )

            origin_distance_m = haversine_m(
                image_gcj_latitude,
                image_gcj_longitude,
                origin_lat,
                origin_lon,
            )
            destination_lon, destination_lat = parse_lnglat(destination_location)
            destination_distance_m = haversine_m(
                image_gcj_latitude,
                image_gcj_longitude,
                destination_lat,
                destination_lon,
            )

            summary = {
                "image_pose": image_pose_for_amap,
                "test_route": {
                    "origin_name": origin_name,
                    "origin": origin_location,
                    "origin_geocode": origin_geocode,
                    "destination_name": destination_name,
                    "destination": destination_location,
                    "destination_geocode": destination_geocode,
                    "mode": mode,
                    "route_summary": route_payload.get("route_summary", {}),
                    "coarse_waypoint_count": len(coarse_waypoints),
                    "route_polyline_count": len(route_polyline),
                },
                "runtime_pose": {
                    "gps_source": gps_source,
                    "latitude": current_latitude,
                    "longitude": current_longitude,
                    "heading_deg": current_heading,
                    "image_to_origin_distance_m": round(origin_distance_m, 2),
                    "image_to_destination_distance_m": round(destination_distance_m, 2),
                },
                "demo_profile": {
                    "perf_profile": perf_profile,
                    "visual_top_k": visual_top_k,
                    "candidate_top_k": candidate_top_k,
                    "llm_detail": llm_detail,
                    "enable_verifier": enable_verifier,
                    "cache_path": str(Path(cache_path).expanduser()),
                },
                "agent": {
                    "route_context": _compact_route_context(agent_payload.get("route_context", {}) or {}),
                    "arbiter_decision": _compact_arbiter_decision(agent_payload.get("arbiter_decision", {}) or {}),
                    "fpv_segmentation_summary": (agent_payload.get("fpv_segmentation_summary") or {}),
                },
                "bev": {
                    "local_bev_context": (local_bev_payload or {}).get("local_bev_context", {}),
                    "map_snapshot": _compact_map_snapshot(map_bev_payload),
                    "errors": {
                        "local_bev_error": local_bev_error,
                        "map_bev_error": map_bev_error,
                        "planner_route_error": planner_route_error,
                    },
                },
                "artifacts": {
                    "overlay_svg": save_overlay if isinstance(overlay_svg, str) and overlay_svg.strip() else None,
                    "preview_png": preview_png,
                    "local_bev_svg": save_local_bev_svg if local_bev_payload and str((local_bev_payload or {}).get("svg") or "").strip() else None,
                    "map_bev_png": map_bev_png,
                    "map_bev_html": map_bev_html,
                    "planner_route_png": planner_route_png,
                    "planner_route_html": planner_route_html,
                },
            }

            print("=== Yuquan Gate Visual Prompt Test ===")
            print(json.dumps(summary, indent=2, ensure_ascii=False))

            if save_json:
                json_path = Path(save_json)
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"\nSaved JSON to: {json_path}")
            if summary["artifacts"]["overlay_svg"]:
                print(f"\nSaved overlay SVG to: {save_overlay}")
            if summary["artifacts"]["preview_png"]:
                print(f"Saved preview PNG to: {summary['artifacts']['preview_png']}")
            if summary["artifacts"]["local_bev_svg"]:
                print(f"Saved local BEV SVG to: {summary['artifacts']['local_bev_svg']}")
            if summary["artifacts"]["map_bev_png"]:
                print(f"Saved map BEV PNG to: {summary['artifacts']['map_bev_png']}")
            if summary["artifacts"]["map_bev_html"]:
                print(f"Saved map BEV HTML to: {summary['artifacts']['map_bev_html']}")
            if summary["artifacts"]["planner_route_png"]:
                print(f"Saved planner route PNG to: {summary['artifacts']['planner_route_png']}")
            if summary["artifacts"]["planner_route_html"]:
                print(f"Saved planner route HTML to: {summary['artifacts']['planner_route_html']}")
            _log_timing(
                "demo_total",
                perf_counter() - overall_started_at,
                status="ok",
            )
            _save_json_cache(cache_path, cache_payload)


if __name__ == "__main__":
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Test the current zero-shot visual prompt agent on an iPhone FPV image for ZJU Yuquan South Gate -> North Gate."
    )
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH, help="Local iPhone FPV image path with EXIF GPS")
    parser.add_argument("--origin", help="Optional AMap/GCJ-02 origin coordinate in 'longitude,latitude' format")
    parser.add_argument("--destination", help="Optional AMap/GCJ-02 destination coordinate in 'longitude,latitude' format")
    parser.add_argument("--origin-name", default=DEFAULT_ORIGIN_NAME, help="Origin place name used when --origin is omitted")
    parser.add_argument("--destination-name", default=DEFAULT_DESTINATION_NAME, help="Destination place name used when --destination is omitted")
    parser.add_argument("--mode", default="foot", choices=["car", "foot", "bike"], help="Routing/navigation mode")
    parser.add_argument(
        "--gps-source",
        default="image",
        choices=["image", "origin"],
        help="Use EXIF GPS as current pose, or use the route origin coordinate for debugging",
    )
    parser.add_argument("--heading-deg", type=float, default=None, help="Override EXIF GPSImgDirection")
    parser.add_argument("--lookahead-m", type=float, default=20.0, help="RouteContext lookahead distance")
    parser.add_argument("--fpv-segmentation-npy", help="Optional SAM3 segmentation .npy path aligned with the FPV image")
    parser.add_argument(
        "--fpv-segmentation-class-mapping",
        help="Optional SAM3 class_mapping.json path; defaults to class_mapping.json next to the segmentation file",
    )
    parser.add_argument(
        "--fpv-segmentation-visualization",
        help="Optional SAM3 visualization image path for bookkeeping/debugging",
    )
    parser.add_argument(
        "--fpv-segmentation-overlay-image-path",
        help="Optional explicit overlay image path passed to MCP",
    )
    parser.add_argument(
        "--fpv-segmentation-overlay-image-url",
        help="Optional explicit overlay image URL passed to MCP",
    )
    parser.add_argument(
        "--enforce-alignment",
        dest="enforce_alignment",
        action="store_true",
        default=True,
        help="Require segmentation/image alignment validation (default: true)",
    )
    parser.add_argument(
        "--no-enforce-alignment",
        dest="enforce_alignment",
        action="store_false",
        help="Allow proceeding when segmentation alignment validation fails",
    )
    parser.add_argument(
        "--off-route-threshold-m",
        type=float,
        default=35.0,
        help="Route matching tolerance. iPhone EXIF + campus map geometry often needs >20m.",
    )
    parser.add_argument(
        "--perf-profile",
        default=DEFAULT_PERF_PROFILE,
        choices=sorted(PROFILE_DEFAULTS.keys()),
        help="Execution profile. latency trims non-essential work; debug enables route-side debug artifacts.",
    )
    parser.add_argument("--visual-top-k", type=int, default=None, help="Maximum visual regions; defaults come from --perf-profile")
    parser.add_argument("--candidate-top-k", type=int, default=None, help="Maximum PoV candidates; defaults come from --perf-profile")
    parser.add_argument("--llm-detail", default=None, choices=["auto", "low", "high"], help="Vision detail level; defaults come from --perf-profile")
    parser.add_argument(
        "--enable-verifier",
        dest="enable_verifier",
        action="store_true",
        default=None,
        help="Force-enable the second-pass visual prompt verifier.",
    )
    parser.add_argument(
        "--disable-verifier",
        dest="enable_verifier",
        action="store_false",
        help="Force-disable the second-pass visual prompt verifier.",
    )
    parser.add_argument(
        "--enable-segmentation-overlay-input",
        action="store_true",
        help="Enable multi-image arbiter input by appending segmentation visualization when available",
    )
    parser.add_argument("--save-overlay", default="/tmp/yuquan_visual_prompt_overlay.svg", help="Output SVG overlay path")
    parser.add_argument(
        "--save-preview-png",
        default=None,
        help="Output PNG preview path. Defaults to the SVG path with .png suffix.",
    )
    parser.add_argument(
        "--save-local-bev-svg",
        default=None,
        help="Output Local BEV patch SVG path. Defaults to the overlay path with _local_bev suffix.",
    )
    parser.add_argument(
        "--save-map-bev-png",
        default=None,
        help="Output downloaded AMap RouteContext BEV PNG path. Defaults to the overlay path with _map_bev suffix.",
    )
    parser.add_argument(
        "--save-map-bev-html",
        default=None,
        help="Output HTML wrapper for the RouteContext BEV snapshot URL. Defaults to the overlay path with _map_bev suffix.",
    )
    parser.add_argument(
        "--save-planner-route-png",
        default=None,
        help="Output pure planner-route overview PNG path. Defaults to the overlay path with _planner_route suffix.",
    )
    parser.add_argument(
        "--save-planner-route-html",
        default=None,
        help="Output HTML wrapper for the pure planner-route overview URL. Defaults to the overlay path with _planner_route suffix.",
    )
    parser.add_argument(
        "--auto-enable-heavy-png",
        action="store_true",
        help="When profile defaults enable debug artifacts, also auto-generate remote PNG downloads in addition to HTML wrappers.",
    )
    parser.add_argument(
        "--cache-path",
        default=DEFAULT_CACHE_PATH,
        help="Local JSON cache path for resolved places and route payloads.",
    )
    parser.add_argument("--save-json", default="/tmp/yuquan_visual_prompt_result.json", help="Optional output JSON path")

    args = parser.parse_args()
    profile_settings = _profile_settings(args.perf_profile)
    visual_top_k = (
        int(args.visual_top_k)
        if args.visual_top_k is not None
        else int(profile_settings["visual_top_k"])
    )
    candidate_top_k = (
        int(args.candidate_top_k)
        if args.candidate_top_k is not None
        else int(profile_settings["candidate_top_k"])
    )
    llm_detail = str(args.llm_detail or profile_settings["llm_detail"])
    enable_verifier = (
        bool(args.enable_verifier)
        if args.enable_verifier is not None
        else bool(profile_settings["enable_verifier"])
    )
    save_preview_png = args.save_preview_png
    overlay_path = Path(args.save_overlay)
    if save_preview_png is None and args.save_overlay:
        save_preview_png = str(overlay_path.with_suffix(".png"))
    save_local_bev_svg = args.save_local_bev_svg
    if save_local_bev_svg is None and args.save_overlay and profile_settings["auto_local_bev_svg"]:
        save_local_bev_svg = str(overlay_path.with_name(f"{overlay_path.stem}_local_bev.svg"))
    save_map_bev_png = args.save_map_bev_png
    if (
        save_map_bev_png is None
        and args.save_overlay
        and profile_settings["auto_map_bev_png"]
        and args.auto_enable_heavy_png
    ):
        save_map_bev_png = str(overlay_path.with_name(f"{overlay_path.stem}_map_bev.png"))
    save_map_bev_html = args.save_map_bev_html
    if save_map_bev_html is None and args.save_overlay and profile_settings["auto_map_bev_html"]:
        save_map_bev_html = str(overlay_path.with_name(f"{overlay_path.stem}_map_bev.html"))
    save_planner_route_png = args.save_planner_route_png
    if (
        save_planner_route_png is None
        and args.save_overlay
        and profile_settings["auto_planner_route_png"]
        and args.auto_enable_heavy_png
    ):
        save_planner_route_png = str(overlay_path.with_name(f"{overlay_path.stem}_planner_route.png"))
    save_planner_route_html = args.save_planner_route_html
    if save_planner_route_html is None and args.save_overlay and profile_settings["auto_planner_route_html"]:
        save_planner_route_html = str(overlay_path.with_name(f"{overlay_path.stem}_planner_route.html"))
    fpv_segmentation = _build_fpv_segmentation_payload(
        segmentation_npy_path=args.fpv_segmentation_npy,
        class_mapping_path=args.fpv_segmentation_class_mapping,
        visualization_path=args.fpv_segmentation_visualization,
    )
    asyncio.run(
        run_demo(
            image_path=args.image_path,
            fpv_segmentation=fpv_segmentation,
            fpv_segmentation_overlay_image_path=args.fpv_segmentation_overlay_image_path,
            fpv_segmentation_overlay_image_url=args.fpv_segmentation_overlay_image_url,
            enforce_alignment=args.enforce_alignment,
            origin=args.origin,
            destination=args.destination,
            origin_name=args.origin_name,
            destination_name=args.destination_name,
            mode=args.mode,
            gps_source=args.gps_source,
            heading_deg=args.heading_deg,
            lookahead_m=args.lookahead_m,
            off_route_threshold_m=args.off_route_threshold_m,
            visual_top_k=visual_top_k,
            candidate_top_k=candidate_top_k,
            llm_detail=llm_detail,
            perf_profile=args.perf_profile,
            enable_verifier=enable_verifier,
            enable_segmentation_overlay_input=args.enable_segmentation_overlay_input,
            save_overlay=args.save_overlay,
            save_preview_png=save_preview_png,
            save_local_bev_svg=save_local_bev_svg,
            save_map_bev_png=save_map_bev_png,
            save_map_bev_html=save_map_bev_html,
            save_planner_route_png=save_planner_route_png,
            save_planner_route_html=save_planner_route_html,
            save_json=args.save_json,
            cache_path=args.cache_path,
        )
    )
