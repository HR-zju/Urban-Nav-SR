"""OSM/Overpass and OSRM provider for global UrbanNav GT construction."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.core.urbannav_provider import (
    UrbanNavLiveProvider,
    UrbanNavPoiSearchRequest,
    UrbanNavProviderCapabilities,
    UrbanNavRouteRequest,
)
from mcp_server.eval.urbannav_metrics import haversine_m
from mcp_server.eval.urbannav_schema import UrbanNavDestination, UrbanNavPoint


class OsmOverpassOsrmProvider(UrbanNavLiveProvider):
    r"""Global open-map provider backed by Overpass POI search and OSRM routes.

    Args:
        overpass_url: Overpass API endpoint.
        osrm_url: OSRM endpoint prefix. For pedestrian routing, a typical
            public endpoint is ``https://routing.openstreetmap.de/routed-foot``.
        osrm_profile: OSRM profile, usually ``"foot"``.
        timeout_s: HTTP timeout in seconds.

    POI retrieval uses OSM tag constraints:

    \[
    C(o,z)=\{d \mid d \in \operatorname{OverpassAround}(o,r),\ tag(d)\in z\}
    \]

    Route retrieval uses OSRM:

    \[
    R(o,d)=\operatorname{OSRMRoute}(o,d,\mathrm{profile}=\mathrm{foot})
    \]
    """

    def __init__(
        self,
        *,
        overpass_url: str | None = None,
        osrm_url: str | None = None,
        osrm_profile: str = "foot",
        timeout_s: float | None = None,
    ) -> None:
        """Initialize the OSM/OSRM provider.

        Args:
            overpass_url: Optional Overpass endpoint.
            osrm_url: Optional OSRM endpoint prefix.
            osrm_profile: OSRM profile name.
            timeout_s: Optional request timeout.

        Returns:
            None.
        """

        self.overpass_url = (overpass_url or os.getenv("OVERPASS_URL") or "https://overpass-api.de/api/interpreter").rstrip("/")
        self.osrm_url = (osrm_url or os.getenv("OSRM_URL") or "https://routing.openstreetmap.de/routed-foot").rstrip("/")
        self.osrm_profile = osrm_profile or os.getenv("OSRM_PROFILE", "foot")
        self.timeout_s = float(timeout_s or os.getenv("URBANNAV_PROVIDER_TIMEOUT_S", "60"))

    @property
    def capabilities(self) -> UrbanNavProviderCapabilities:
        """Return provider capabilities.

        Returns:
            Capability descriptor for the OSM global provider.
        """

        return UrbanNavProviderCapabilities(
            provider_name="osm_overpass_osrm",
            supports_poi_search=True,
            supports_pedestrian_route=True,
            supports_static_map=False,
            raw={"overpass_url": self.overpass_url, "osrm_url": self.osrm_url, "osrm_profile": self.osrm_profile},
        )

    async def search_pois(self, *, cache_key: str, request: UrbanNavPoiSearchRequest) -> ReplayPoiResult:
        """Search OSM POIs through Overpass.

        Args:
            cache_key: Stable cache key.
            request: Normalized POI search request.

        Returns:
            Replay-compatible POI search record.
        """

        tags = list(request.tags or [])
        if not tags:
            tags = _fallback_query_tags(request.query)
        query = _build_overpass_query(origin=request.origin, radius_m=request.radius_m, tags=tags, limit=request.limit)
        data = await asyncio.to_thread(self._post_overpass, query)
        candidates = _parse_overpass_elements(data.get("elements", []), origin=request.origin, limit=request.limit)
        return ReplayPoiResult(
            cache_key=cache_key,
            provider=self.capabilities.provider_name,
            request={
                "origin": request.origin.to_dict(),
                "query": request.query,
                "tags": tags,
                "radius_m": request.radius_m,
                "limit": request.limit,
                "locale": request.locale,
                "overpass_url": self.overpass_url,
            },
            response=data,
            candidates=candidates,
        )

    async def plan_route(self, *, cache_key: str, request: UrbanNavRouteRequest) -> ReplayRouteResult:
        """Plan one route through OSRM.

        Args:
            cache_key: Stable cache key.
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
        data = await asyncio.to_thread(self._request_osrm, request.origin, request.destination)
        route = _parse_osrm_route(data, origin=request.origin, destination=request.destination.point)
        return ReplayRouteResult(
            cache_key=cache_key,
            provider=self.capabilities.provider_name,
            request={
                "origin": request.origin.to_dict(),
                "destination": request.destination.to_dict(),
                "mode": request.mode,
                "osrm_url": self.osrm_url,
                "profile": self.osrm_profile,
            },
            response=data,
            polyline=route["polyline"],
            distance_m=route["distance_m"],
            duration_s=route["duration_s"],
        )

    def _post_overpass(self, query: str) -> dict[str, Any]:
        """POST an Overpass query.

        Args:
            query: Overpass QL query.

        Returns:
            Decoded JSON response.

        Raises:
            RuntimeError: If the response cannot be decoded.
        """

        payload = urlencode({"data": query}).encode("utf-8")
        req = Request(
            self.overpass_url,
            data=payload,
            headers={"User-Agent": "UrbanNavBenchmark/0.1", "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_osrm(self, origin: UrbanNavPoint, destination: UrbanNavDestination) -> dict[str, Any]:
        """Request one OSRM route.

        Args:
            origin: Route origin.
            destination: Destination with coordinates.

        Returns:
            Decoded OSRM response.
        """

        coords = f"{origin.lng},{origin.lat};{destination.lng},{destination.lat}"
        query = urlencode({"overview": "full", "geometries": "geojson", "steps": "true"})
        url = f"{self.osrm_url}/route/v1/{self.osrm_profile}/{coords}?{query}"
        req = Request(url, headers={"User-Agent": "UrbanNavBenchmark/0.1"})
        with urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))


def _build_overpass_query(*, origin: UrbanNavPoint, radius_m: int, tags: list[str], limit: int) -> str:
    """Build an Overpass around query.

    Args:
        origin: Search center.
        radius_m: Search radius.
        tags: OSM tag constraints in ``key=value`` form.
        limit: Maximum output elements.

    Returns:
        Overpass QL query.
    """

    clauses = []
    for tag in tags:
        parsed = _parse_osm_tag(tag)
        if not parsed:
            continue
        key, value = parsed
        selector = f'["{key}"="{value}"]'
        around = f"(around:{int(radius_m)},{origin.lat},{origin.lng})"
        clauses.extend([f"node{around}{selector};", f"way{around}{selector};", f"relation{around}{selector};"])
    if not clauses:
        clauses.append(f'node(around:{int(radius_m)},{origin.lat},{origin.lng})["name"];')
    return "\n".join(
        [
            "[out:json][timeout:30];",
            "(",
            *clauses,
            ");",
            "out center tags qt " + str(max(1, limit * 4)) + ";",
        ]
    )


def _parse_overpass_elements(elements: list[dict[str, Any]], *, origin: UrbanNavPoint, limit: int) -> list[UrbanNavDestination]:
    """Normalize Overpass elements into UrbanNav destinations.

    Args:
        elements: Raw Overpass elements.
        origin: Search origin for distance sorting.
        limit: Maximum candidates.

    Returns:
        Normalized destination list sorted by direct distance.
    """

    candidates: list[UrbanNavDestination] = []
    for element in elements:
        tags = element.get("tags") or {}
        lat = element.get("lat") or (element.get("center") or {}).get("lat")
        lng = element.get("lon") or (element.get("center") or {}).get("lon")
        if lat is None or lng is None:
            continue
        point = UrbanNavPoint(lat=float(lat), lng=float(lng))
        name = str(tags.get("name") or tags.get("brand") or tags.get("operator") or "Unnamed OSM POI")
        poi_type = _osm_type(tags)
        candidates.append(
            UrbanNavDestination(
                lat=point.lat,
                lng=point.lng,
                name=name,
                type=poi_type,
                provider_poi_id=f"osm:{element.get('type')}:{element.get('id')}",
                route_distance_m=round(haversine_m(origin, point), 3),
                raw={"osm": element},
            )
        )
    candidates.sort(key=lambda item: (item.route_distance_m or float("inf"), item.name or ""))
    return candidates[: max(1, limit)]


def _parse_osrm_route(
    data: dict[str, Any],
    *,
    origin: UrbanNavPoint,
    destination: UrbanNavPoint | None,
) -> dict[str, Any]:
    """Normalize an OSRM response.

    Args:
        data: Raw OSRM response.
        origin: Exact benchmark origin.
        destination: Exact benchmark destination.

    Returns:
        Dictionary containing polyline, distance, and duration.
    """

    routes = data.get("routes") or []
    if not routes:
        return {"polyline": [], "distance_m": None, "duration_s": None}
    route = routes[0]
    coords = ((route.get("geometry") or {}).get("coordinates") or [])
    polyline = [UrbanNavPoint(lat=float(coord[1]), lng=float(coord[0])) for coord in coords if isinstance(coord, list) and len(coord) >= 2]
    if origin and (not polyline or polyline[0] != origin):
        polyline.insert(0, origin)
    if destination and (not polyline or polyline[-1] != destination):
        polyline.append(destination)
    return {"polyline": _dedupe_points(polyline), "distance_m": _float_or_none(route.get("distance")), "duration_s": _float_or_none(route.get("duration"))}


def _parse_osm_tag(tag: str) -> tuple[str, str] | None:
    """Parse one OSM tag constraint.

    Args:
        tag: Tag in ``key=value`` form.

    Returns:
        Parsed key/value pair, or ``None``.
    """

    if "=" not in tag:
        return None
    key, value = tag.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        return None
    return key, value


def _fallback_query_tags(query: str) -> list[str]:
    """Map common English query text to OSM tags.

    Args:
        query: Query string.

    Returns:
        Fallback tag list.
    """

    text = query.lower()
    if "restaurant" in text or "food" in text:
        return ["amenity=restaurant", "amenity=fast_food"]
    if "bank" in text:
        return ["amenity=bank"]
    if "atm" in text:
        return ["amenity=atm"]
    if "coffee" in text or "cafe" in text:
        return ["amenity=cafe"]
    if "hotel" in text:
        return ["tourism=hotel"]
    if "convenience" in text or "grocery" in text:
        return ["shop=convenience", "shop=supermarket"]
    return []


def _osm_type(tags: dict[str, Any]) -> str:
    """Return a compact OSM type string.

    Args:
        tags: OSM tags.

    Returns:
        Type string such as ``amenity=restaurant``.
    """

    for key in ("amenity", "shop", "tourism", "leisure", "healthcare", "office"):
        if tags.get(key):
            return f"{key}={tags[key]}"
    return "osm_poi"


def _dedupe_points(points: list[UrbanNavPoint]) -> list[UrbanNavPoint]:
    """Remove adjacent duplicate route points.

    Args:
        points: Route points.

    Returns:
        Deduplicated point list.
    """

    output: list[UrbanNavPoint] = []
    for point in points:
        if output and abs(output[-1].lat - point.lat) < 1e-9 and abs(output[-1].lng - point.lng) < 1e-9:
            continue
        output.append(point)
    return output


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
