"""Build map-derived semantic BEV layers around the current GNSS pose.

The route-derived BEV context tells the agent where the planned route goes.
This provider adds nearby OpenStreetMap semantics so QA teachers can reason
about what the route is passing through:

    GNSS + heading + local BEV bbox
        -> OSM/Overpass features
        -> ego-aligned BEV anchors

Projection uses the same tangent-plane approximation as ``build_bev``:

    east_m  ~= (lng - lng_0) * 111320 * cos(mean_lat)
    north_m ~= (lat - lat_0) * 110540
    x_bev = east_m * cos(theta) - north_m * sin(theta)
    y_bev = east_m * sin(theta) + north_m * cos(theta)

The first implementation intentionally stores point anchors rather than dense
rasters. They are enough for Stage-1/Stage-2 QA tasks such as "is there a
crossing ahead?" and can later be upgraded to line/polygon masks when the map
provider returns full geometry.
"""

from __future__ import annotations

import math
import os
from typing import Any

BEV_MAP_SCHEMA_VERSION = 1

_QUERY_SPECS: tuple[tuple[str, list[str] | None], ...] = (
    ("highway", ["footway", "path", "pedestrian", "steps", "crossing", "residential", "service", "living_street"]),
    ("footway", ["sidewalk", "crossing"]),
    ("crossing", None),
    ("entrance", None),
    ("building", None),
    ("barrier", None),
)


def _default_query_radius_m(local_bev_context: dict[str, Any], radius_m: float | None) -> float:
    if radius_m is not None:
        return float(radius_m)
    return _radius_from_local_bev(
        local_bev_context,
        float(os.getenv("BEV_MAP_QUERY_RADIUS_M", "60") or 60),
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bev_point_from_wgs84(
    *,
    lat: float,
    lng: float,
    origin_lat: float,
    origin_lng: float,
    rotation_heading_deg: float,
) -> dict[str, float]:
    mean_lat = (float(lat) + float(origin_lat)) / 2.0
    east_m = (float(lng) - float(origin_lng)) * 111320.0 * max(0.1, math.cos(math.radians(mean_lat)))
    north_m = (float(lat) - float(origin_lat)) * 110540.0
    theta = math.radians(float(rotation_heading_deg))
    x_m = east_m * math.cos(theta) - north_m * math.sin(theta)
    y_m = east_m * math.sin(theta) + north_m * math.cos(theta)
    return {"x_m": round(float(x_m), 3), "y_m": round(float(y_m), 3)}


def _radius_from_local_bev(local_bev_context: dict[str, Any], default_radius_m: float) -> float:
    bbox = local_bev_context.get("bev_bbox") if isinstance(local_bev_context, dict) else None
    if not isinstance(bbox, dict):
        return float(default_radius_m)
    values = [
        abs(float(bbox.get("x_min_m", 0.0) or 0.0)),
        abs(float(bbox.get("x_max_m", 0.0) or 0.0)),
        abs(float(bbox.get("y_min_m", 0.0) or 0.0)),
        abs(float(bbox.get("y_max_m", 0.0) or 0.0)),
    ]
    return max(float(default_radius_m), math.hypot(max(values[:2] or [0.0]), max(values[2:] or [0.0])) + 20.0)


def _bbox_around(lat: float, lng: float, radius_m: float) -> tuple[float, float, float, float]:
    lat_delta = float(radius_m) / 111000.0
    lon_delta = float(radius_m) / (111000.0 * max(0.1, math.cos(math.radians(float(lat)))))
    return (float(lng) - lon_delta, float(lat) - lat_delta, float(lng) + lon_delta, float(lat) + lat_delta)


def _feature_lat_lng(feature: dict[str, Any]) -> tuple[float, float] | None:
    lat = _float_or_none(feature.get("lat"))
    lng = _float_or_none(feature.get("lon", feature.get("lng")))
    if lat is not None and lng is not None:
        return lat, lng
    center = feature.get("center") if isinstance(feature.get("center"), dict) else {}
    lat = _float_or_none(center.get("lat"))
    lng = _float_or_none(center.get("lon", center.get("lng")))
    if lat is not None and lng is not None:
        return lat, lng
    return None


def _feature_role(tags: dict[str, Any]) -> str:
    highway = str(tags.get("highway") or "").strip().lower()
    footway = str(tags.get("footway") or "").strip().lower()
    crossing = str(tags.get("crossing") or "").strip().lower()
    if tags.get("entrance") is not None:
        return "entrance"
    if tags.get("barrier") is not None:
        return "barrier"
    if tags.get("building") is not None:
        return "building"
    if highway == "crossing" or footway == "crossing" or crossing:
        return "crossing"
    if highway in {"footway", "path", "pedestrian", "steps"} or footway in {"sidewalk", "crossing"}:
        return "sidewalk"
    if highway in {"residential", "service", "living_street", "tertiary", "secondary", "primary", "unclassified"}:
        return "road"
    return "other"


def _append_role_point(layers: dict[str, Any], role: str, point: dict[str, float]) -> None:
    key_by_role = {
        "sidewalk": "sidewalk_points_bev",
        "crossing": "crossing_points_bev",
        "road": "road_points_bev",
        "building": "building_points_bev",
        "barrier": "barrier_points_bev",
        "entrance": "entrance_candidates_bev",
    }
    key = key_by_role.get(role)
    if key:
        layers.setdefault(key, []).append(point)


def _dedupe_points(points: list[dict[str, float]]) -> list[dict[str, float]]:
    seen: set[tuple[int, int]] = set()
    deduped: list[dict[str, float]] = []
    for point in points:
        key = (round(float(point.get("x_m", 0.0) or 0.0) * 10), round(float(point.get("y_m", 0.0) or 0.0) * 10))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


async def _query_features_by_specs(
    *,
    map_client: Any,
    bbox: tuple[float, float, float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    features_out: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for category, subcategories in _QUERY_SPECS:
        try:
            features = await map_client.search_features_by_category(
                bbox=bbox,
                category=category,
                subcategories=subcategories,
            )
        except Exception as exc:
            errors.append({"category": category, "error": str(exc)})
            continue
        for feature in features or []:
            if isinstance(feature, dict):
                features_out.append(feature)
    return features_out, errors


def _project_features_to_layers(
    *,
    features: list[dict[str, Any]],
    errors: list[dict[str, str]],
    origin_lat: float,
    origin_lng: float,
    rotation_heading_deg: float,
    query_radius_m: float,
    bbox: tuple[float, float, float, float],
    provider_name: str,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layers: dict[str, Any] = {
        "schema_version": BEV_MAP_SCHEMA_VERSION,
        "map_provider": str(provider_name or "osm"),
        "source": "osm_overpass_semantic_bev_v1",
        "map_query": {
            "center": {"lat": round(origin_lat, 8), "lng": round(origin_lng, 8)},
            "radius_m": round(float(query_radius_m), 3),
            "bbox_lng_lat": [round(float(value), 8) for value in bbox],
            "rotation_heading_deg": round(float(rotation_heading_deg), 3),
        },
        "sidewalk_points_bev": [],
        "crossing_points_bev": [],
        "road_points_bev": [],
        "building_points_bev": [],
        "barrier_points_bev": [],
        "entrance_candidates_bev": [],
        "features": [],
        "errors": list(errors or []),
    }
    if cache_metadata:
        layers["cache"] = dict(cache_metadata)
    for feature in features or []:
        if not isinstance(feature, dict):
            continue
        lat_lng = _feature_lat_lng(feature)
        if lat_lng is None:
            continue
        tags = feature.get("tags") if isinstance(feature.get("tags"), dict) else {}
        role = _feature_role(tags)
        point = _bev_point_from_wgs84(
            lat=lat_lng[0],
            lng=lat_lng[1],
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            rotation_heading_deg=rotation_heading_deg,
        )
        _append_role_point(layers, role, point)
        layers["features"].append(
            {
                "provider_feature_id": str(feature.get("id") or ""),
                "osm_type": feature.get("type"),
                "role": role,
                "point_bev": point,
                "lat": round(float(lat_lng[0]), 8),
                "lng": round(float(lat_lng[1]), 8),
                "tags": {str(key): value for key, value in tags.items()},
            }
        )
    for key in [
        "sidewalk_points_bev",
        "crossing_points_bev",
        "road_points_bev",
        "building_points_bev",
        "barrier_points_bev",
        "entrance_candidates_bev",
    ]:
        layers[key] = _dedupe_points(layers.get(key) or [])
    layers["feature_counts"] = {
        "sidewalk": len(layers["sidewalk_points_bev"]),
        "crossing": len(layers["crossing_points_bev"]),
        "road": len(layers["road_points_bev"]),
        "building": len(layers["building_points_bev"]),
        "barrier": len(layers["barrier_points_bev"]),
        "entrance": len(layers["entrance_candidates_bev"]),
        "total_features": len(layers["features"]),
    }
    return layers


class BEVMapProviderSession:
    """Navigation-scoped map provider session with a spatial raw-feature cache.

    A navigation run processes many FPV frames that are only a few meters apart.
    Querying Overpass for every frame is wasteful and can trigger rate limits.
    The session caches provider-native features by a quantized GNSS cell:

        cell_lat = floor(lat_m / cache_cell_m)
        cell_lng = floor(lng_m / cache_cell_m)
        key = (provider, cell_lat, cell_lng, radius_bucket_m)

    Cached features are stored in WGS84 and re-projected for each frame, so
    heading changes and ego-frame origin changes still produce correct BEV
    coordinates while the expensive network call is reused.
    """

    def __init__(
        self,
        *,
        map_client: Any,
        provider_name: str = "osm",
        cache_cell_m: float | None = None,
    ) -> None:
        self.map_client = map_client
        self.provider_name = str(provider_name or "osm")
        self.cache_cell_m = float(cache_cell_m or os.getenv("BEV_MAP_CACHE_CELL_M", "80") or 80)
        self._raw_feature_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._connected = False
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    @classmethod
    def from_env(cls) -> "BEVMapProviderSession":
        from mcp_server.map_providers import create_map_client

        map_client, provider_name = create_map_client()
        return cls(map_client=map_client, provider_name=provider_name)

    async def __aenter__(self) -> "BEVMapProviderSession":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        if self._connected:
            return
        await self.map_client.connect()
        self._connected = True

    async def disconnect(self) -> None:
        if not self._connected:
            return
        await self.map_client.disconnect()
        self._connected = False

    def cache_stats(self) -> dict[str, Any]:
        return {
            "cache_entries": len(self._raw_feature_cache),
            "cache_cell_m": round(float(self.cache_cell_m), 3),
            "query_count": int(self.query_count),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
        }

    def _cache_key(self, *, lat: float, lng: float, radius_m: float) -> tuple[Any, ...]:
        cell_m = max(1.0, float(self.cache_cell_m))
        lat_cell = math.floor((float(lat) * 110540.0) / cell_m)
        lng_cell = math.floor(
            (float(lng) * 111320.0 * max(0.1, math.cos(math.radians(float(lat))))) / cell_m
        )
        radius_bucket_m = int(math.ceil(float(radius_m) / 10.0) * 10)
        return (self.provider_name, int(lat_cell), int(lng_cell), radius_bucket_m)

    async def build_semantic_layers(
        self,
        *,
        gps: dict[str, float],
        heading_deg: float | None,
        local_bev_context: dict[str, Any],
        radius_m: float | None = None,
    ) -> dict[str, Any]:
        origin_lat = float(gps["lat"])
        origin_lng = float(gps["lng"])
        rotation_heading_deg = float(local_bev_context.get("bev_rotation_deg", heading_deg or 0.0) or 0.0)
        query_radius_m = _default_query_radius_m(local_bev_context, radius_m)
        key = self._cache_key(lat=origin_lat, lng=origin_lng, radius_m=query_radius_m)
        cache_hit = key in self._raw_feature_cache
        if cache_hit:
            self.cache_hits += 1
            cached = self._raw_feature_cache[key]
            features = list(cached.get("features") or [])
            errors = list(cached.get("errors") or [])
            bbox = tuple(cached.get("bbox") or _bbox_around(origin_lat, origin_lng, query_radius_m))
        else:
            self.cache_misses += 1
            bbox = _bbox_around(origin_lat, origin_lng, query_radius_m)
            features, errors = await _query_features_by_specs(map_client=self.map_client, bbox=bbox)
            self.query_count += 1
            self._raw_feature_cache[key] = {
                "bbox": bbox,
                "features": list(features),
                "errors": list(errors),
            }
        return _project_features_to_layers(
            features=features,
            errors=errors,
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            rotation_heading_deg=rotation_heading_deg,
            query_radius_m=query_radius_m,
            bbox=tuple(float(value) for value in bbox),
            provider_name=self.provider_name,
            cache_metadata={
                "enabled": True,
                "hit": bool(cache_hit),
                "key": list(key),
                **self.cache_stats(),
            },
        )


async def build_osm_bev_semantic_layers(
    *,
    map_client: Any,
    gps: dict[str, float],
    heading_deg: float | None,
    local_bev_context: dict[str, Any],
    provider_name: str = "osm",
    radius_m: float | None = None,
) -> dict[str, Any]:
    """Query a map provider and project semantic anchors into local BEV.

    ``map_client`` must expose ``search_features_by_category``. The existing
    OSM client implements that method with Overpass; tests can pass a fake
    client with the same method.
    """
    origin_lat = float(gps["lat"])
    origin_lng = float(gps["lng"])
    rotation_heading_deg = float(local_bev_context.get("bev_rotation_deg", heading_deg or 0.0) or 0.0)
    query_radius_m = _default_query_radius_m(local_bev_context, radius_m)
    bbox = _bbox_around(origin_lat, origin_lng, query_radius_m)
    features, errors = await _query_features_by_specs(map_client=map_client, bbox=bbox)
    return _project_features_to_layers(
        features=features,
        errors=errors,
        origin_lat=origin_lat,
        origin_lng=origin_lng,
        rotation_heading_deg=rotation_heading_deg,
        query_radius_m=query_radius_m,
        bbox=bbox,
        provider_name=provider_name,
        cache_metadata={"enabled": False},
    )


__all__ = ["BEVMapProviderSession", "build_osm_bev_semantic_layers"]
