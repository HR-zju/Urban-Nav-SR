"""Local OSM provider for reproducible global UrbanNav GT construction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mcp_server.core.osm_global_provider import _dedupe_points, _parse_osrm_route
from mcp_server.core.osm_local_index import LocalOsmPoiIndex
from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.core.urbannav_provider import (
    UrbanNavLiveProvider,
    UrbanNavPoiSearchRequest,
    UrbanNavProviderCapabilities,
    UrbanNavRouteRequest,
)
from mcp_server.eval.urbannav_schema import UrbanNavPoint


class LocalOsmOsrmProvider(UrbanNavLiveProvider):
    r"""UrbanNav provider backed by frozen local OSM POI indexes and routers.

    Args:
        poi_index_root: Directory containing local POI JSONL indexes.
        router_config: Optional JSON config mapping city/region ids to local
            OSRM or Valhalla endpoints.
        default_osrm_url: Fallback OSRM endpoint prefix.
        osrm_profile: OSRM profile name.
        default_valhalla_url: Fallback Valhalla endpoint prefix.
        valhalla_costing: Valhalla costing model.
        timeout_s: HTTP timeout.

    Unlike ``OsmOverpassOsrmProvider``, this provider never calls public
    Overpass. Its POI retrieval is local:

    \[
    C(o,z)=\operatorname{LocalRadiusSearch}(o,z,r)
    \]

    Route retrieval still uses OSRM, but the intended deployment is a local
    OSRM/Valhalla service built from the same OSM snapshot.
    """

    def __init__(
        self,
        *,
        poi_index_root: Path | str,
        router_config: Path | str | None = None,
        default_osrm_url: str | None = None,
        osrm_profile: str = "foot",
        default_valhalla_url: str | None = None,
        valhalla_costing: str = "pedestrian",
        timeout_s: float | None = None,
    ) -> None:
        """Initialize the local OSM provider.

        Args:
            poi_index_root: Directory containing local POI JSONL indexes.
            router_config: Optional router config JSON.
            default_osrm_url: Fallback OSRM endpoint prefix.
            osrm_profile: OSRM profile name.
            default_valhalla_url: Fallback Valhalla endpoint prefix.
            valhalla_costing: Valhalla costing model.
            timeout_s: HTTP timeout.

        Returns:
            None.
        """

        self.poi_index_root = Path(poi_index_root)
        self.router_config_path = Path(router_config) if router_config else None
        self.router_config = _load_router_config(self.router_config_path)
        self.default_osrm_url = (
            default_osrm_url
            or self.router_config.get("default_osrm_url")
            or os.getenv("OSRM_URL")
            or "http://127.0.0.1:5000"
        ).rstrip("/")
        self.osrm_profile = osrm_profile or str(self.router_config.get("default_osrm_profile") or "foot")
        self.default_valhalla_url = (
            default_valhalla_url
            or self.router_config.get("default_valhalla_url")
            or os.getenv("VALHALLA_URL")
            or "http://127.0.0.1:8002"
        ).rstrip("/")
        self.valhalla_costing = valhalla_costing or str(self.router_config.get("default_valhalla_costing") or "pedestrian")
        self.default_router_type = str(self.router_config.get("default_router_type") or "osrm").lower()
        self.timeout_s = float(timeout_s or os.getenv("URBANNAV_PROVIDER_TIMEOUT_S", "30"))
        self._indexes: dict[str, LocalOsmPoiIndex] = {}

    @property
    def capabilities(self) -> UrbanNavProviderCapabilities:
        """Return provider capabilities.

        Returns:
            Capability descriptor.
        """

        return UrbanNavProviderCapabilities(
            provider_name=f"osm_local_{self.default_router_type}",
            supports_poi_search=True,
            supports_pedestrian_route=True,
            supports_static_map=False,
            raw={
                "poi_index_root": str(self.poi_index_root),
                "router_config": str(self.router_config_path) if self.router_config_path else None,
                "default_router_type": self.default_router_type,
                "default_osrm_url": self.default_osrm_url,
                "osrm_profile": self.osrm_profile,
                "default_valhalla_url": self.default_valhalla_url,
                "valhalla_costing": self.valhalla_costing,
            },
        )

    async def search_pois(self, *, cache_key: str, request: UrbanNavPoiSearchRequest) -> ReplayPoiResult:
        """Search local OSM POIs.

        Args:
            cache_key: Replay cache key.
            request: Normalized POI search request.

        Returns:
            Replay-compatible POI search record.
        """

        index_key = _request_scope_key(request.raw)
        index = self._load_index(index_key)
        candidates = index.search(
            origin=request.origin,
            tags=list(request.tags or []),
            query=request.query,
            radius_m=int(request.radius_m),
            limit=int(request.limit),
        )
        return ReplayPoiResult(
            cache_key=cache_key,
            provider=self.capabilities.provider_name,
            request={
                "origin": request.origin.to_dict(),
                "query": request.query,
                "tags": list(request.tags or []),
                "radius_m": request.radius_m,
                "limit": request.limit,
                "locale": request.locale,
                "scope_key": index_key,
                "poi_index_root": str(self.poi_index_root),
            },
            response={
                "source": index.source_id,
                "candidate_count": len(candidates),
                "mode": "local_osm_radius_search",
            },
            candidates=candidates,
        )

    async def plan_route(self, *, cache_key: str, request: UrbanNavRouteRequest) -> ReplayRouteResult:
        """Plan a route through a local OSRM endpoint.

        Args:
            cache_key: Replay cache key.
            request: Normalized route request.

        Returns:
            Replay-compatible route record.
        """

        if request.destination.lat is None or request.destination.lng is None:
            return ReplayRouteResult(
                cache_key=cache_key,
                provider=self.capabilities.provider_name,
                request={"origin": request.origin.to_dict(), "destination": request.destination.to_dict(), "mode": request.mode},
                response={"error": "destination_missing_coordinate"},
            )
        scope_key = _destination_scope_key(request.destination.raw)
        router = self._router_for_scope(scope_key)
        if router["router_type"] == "valhalla":
            data = self._request_valhalla(request.origin, request.destination.point, router["url"], router["profile"])
            route = _parse_valhalla_route(data, origin=request.origin, destination=request.destination.point)
        else:
            data = self._request_osrm(request.origin, request.destination.point, router["url"], router["profile"])
            route = _parse_osrm_route(data, origin=request.origin, destination=request.destination.point)
        return ReplayRouteResult(
            cache_key=cache_key,
            provider=self.capabilities.provider_name,
            request={
                "origin": request.origin.to_dict(),
                "destination": request.destination.to_dict(),
                "mode": request.mode,
                "scope_key": scope_key,
                "router_type": router["router_type"],
                "router_url": router["url"],
                "profile": router["profile"],
            },
            response=data,
            polyline=route["polyline"],
            distance_m=route["distance_m"],
            duration_s=route["duration_s"],
        )

    def _load_index(self, scope_key: str) -> LocalOsmPoiIndex:
        """Load one POI index by region/city/default key.

        Args:
            scope_key: Region or city key.

        Returns:
            Loaded POI index.

        Raises:
            FileNotFoundError: If no matching local index exists.
        """

        if scope_key in self._indexes:
            return self._indexes[scope_key]
        path = self._index_path(scope_key)
        index = LocalOsmPoiIndex.from_path(path, source_id=scope_key)
        self._indexes[scope_key] = index
        return index

    def _index_path(self, scope_key: str) -> Path:
        indexes = self.router_config.get("indexes") if isinstance(self.router_config.get("indexes"), dict) else {}
        configured = indexes.get(scope_key) or indexes.get("default")
        if configured:
            path = Path(configured)
            return path if path.is_absolute() else self.poi_index_root / path
        for candidate in (
            self.poi_index_root / f"{scope_key}.jsonl",
            self.poi_index_root / scope_key / "pois.jsonl",
            self.poi_index_root / "all.jsonl",
        ):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No local OSM POI index found for scope '{scope_key}' under {self.poi_index_root}")

    def _router_for_scope(self, scope_key: str) -> dict[str, str]:
        """Return router endpoint metadata for a scope.

        Args:
            scope_key: Region, city, country, or ``default``.

        Returns:
            Dictionary with ``router_type``, ``url``, and ``profile``.
        """

        routers = self.router_config.get("routers") if isinstance(self.router_config.get("routers"), dict) else {}
        entry = routers.get(scope_key) or routers.get("default") or {}
        if isinstance(entry, str):
            return {"router_type": "osrm", "url": entry.rstrip("/"), "profile": self.osrm_profile}
        router_type = str(entry.get("router_type") or self.default_router_type or "osrm").lower()
        if router_type == "valhalla":
            url = str(entry.get("valhalla_url") or entry.get("url") or self.default_valhalla_url).rstrip("/")
            profile = str(entry.get("costing") or entry.get("profile") or self.valhalla_costing)
        else:
            router_type = "osrm"
            url = str(entry.get("osrm_url") or entry.get("url") or self.default_osrm_url).rstrip("/")
            profile = str(entry.get("profile") or self.osrm_profile)
        return {"router_type": router_type, "url": url, "profile": profile}

    def _request_osrm(self, origin: UrbanNavPoint, destination: UrbanNavPoint, osrm_url: str, profile: str) -> dict[str, Any]:
        """Request one OSRM route.

        Args:
            origin: Route origin.
            destination: Route destination.
            osrm_url: OSRM endpoint prefix.
            profile: OSRM profile.

        Returns:
            Decoded OSRM response.
        """

        coords = f"{origin.lng},{origin.lat};{destination.lng},{destination.lat}"
        query = urlencode({"overview": "full", "geometries": "geojson", "steps": "true"})
        url = f"{osrm_url}/route/v1/{profile}/{coords}?{query}"
        req = Request(url, headers={"User-Agent": "UrbanNavBenchmark/0.1"})
        with urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_valhalla(self, origin: UrbanNavPoint, destination: UrbanNavPoint, valhalla_url: str, costing: str) -> dict[str, Any]:
        """Request one Valhalla route.

        Args:
            origin: Route origin.
            destination: Route destination.
            valhalla_url: Valhalla service endpoint prefix.
            costing: Valhalla costing model, usually ``pedestrian``.

        Returns:
            Decoded Valhalla response.
        """

        payload = {
            "locations": [
                {"lat": origin.lat, "lon": origin.lng, "type": "break"},
                {"lat": destination.lat, "lon": destination.lng, "type": "break"},
            ],
            "costing": costing,
            "directions_options": {"units": "kilometers"},
            "shape_format": "geojson",
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{valhalla_url}/route",
            data=data,
            headers={"User-Agent": "UrbanNavBenchmark/0.1", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))


def _load_router_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _request_scope_key(raw: dict[str, Any]) -> str:
    return str(raw.get("region_id") or raw.get("city") or raw.get("country") or "default")


def _destination_scope_key(raw: dict[str, Any]) -> str:
    local = raw.get("osm_local") if isinstance(raw.get("osm_local"), dict) else {}
    return str(local.get("region_id") or local.get("city") or local.get("country") or "default")


def _parse_valhalla_route(
    data: dict[str, Any],
    *,
    origin: UrbanNavPoint,
    destination: UrbanNavPoint | None,
) -> dict[str, Any]:
    """Normalize a Valhalla response.

    Args:
        data: Raw Valhalla ``/route`` response.
        origin: Exact benchmark origin.
        destination: Exact benchmark destination.

    Returns:
        Dictionary containing polyline, distance, and duration.
    """

    trip = data.get("trip") if isinstance(data.get("trip"), dict) else {}
    summary = trip.get("summary") if isinstance(trip.get("summary"), dict) else {}
    legs = trip.get("legs") if isinstance(trip.get("legs"), list) else []
    polyline: list[UrbanNavPoint] = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        shape = leg.get("shape")
        polyline.extend(_parse_valhalla_shape(shape))
    if origin and (not polyline or polyline[0] != origin):
        polyline.insert(0, origin)
    if destination and (not polyline or polyline[-1] != destination):
        polyline.append(destination)
    distance_m = _valhalla_distance_m(summary)
    duration_s = _float_or_none(summary.get("time") or summary.get("elapsed_time"))
    return {"polyline": _dedupe_points(polyline), "distance_m": distance_m, "duration_s": duration_s}


def _parse_valhalla_shape(shape: Any) -> list[UrbanNavPoint]:
    """Parse a Valhalla leg shape.

    Args:
        shape: Either GeoJSON coordinates or an encoded Valhalla polyline.

    Returns:
        Decoded route points.
    """

    if isinstance(shape, list):
        return [
            UrbanNavPoint(lat=float(coord[1]), lng=float(coord[0]))
            for coord in shape
            if isinstance(coord, list) and len(coord) >= 2
        ]
    if isinstance(shape, str) and shape:
        return _decode_valhalla_polyline(shape)
    return []


def _decode_valhalla_polyline(encoded: str, *, precision: int = 6) -> list[UrbanNavPoint]:
    """Decode a Valhalla encoded polyline.

    Args:
        encoded: Encoded shape string.
        precision: Coordinate precision. Valhalla uses six decimal places.

    Returns:
        Decoded route points.
    """

    coordinates: list[UrbanNavPoint] = []
    index = 0
    lat = 0
    lng = 0
    factor = 10**precision
    while index < len(encoded):
        result = 1
        shift = 0
        while True:
            value = ord(encoded[index]) - 63 - 1
            index += 1
            result += value << shift
            shift += 5
            if value < 0x1F:
                break
        lat += ~(result >> 1) if result & 1 else result >> 1

        result = 1
        shift = 0
        while True:
            value = ord(encoded[index]) - 63 - 1
            index += 1
            result += value << shift
            shift += 5
            if value < 0x1F:
                break
        lng += ~(result >> 1) if result & 1 else result >> 1
        coordinates.append(UrbanNavPoint(lat=lat / factor, lng=lng / factor))
    return coordinates


def _valhalla_distance_m(summary: dict[str, Any]) -> float | None:
    """Return Valhalla route distance in meters.

    Args:
        summary: Valhalla route summary.

    Returns:
        Distance in meters, or ``None``.
    """

    length = _float_or_none(summary.get("length"))
    if length is None:
        return None
    return length * 1000.0


def _float_or_none(value: Any) -> float | None:
    """Convert value to float if possible.

    Args:
        value: Raw value.

    Returns:
        Float or ``None``.
    """

    try:
        return float(value)
    except Exception:
        return None
