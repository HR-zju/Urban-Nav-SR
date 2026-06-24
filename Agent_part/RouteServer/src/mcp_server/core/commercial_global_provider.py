"""Commercial map providers for UrbanNav global review splits."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import quote, urlencode
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


class TomTomUrbanNavProvider(UrbanNavLiveProvider):
    r"""TomTom Search + Routing provider for commercial GT review splits.

    Args:
        api_key: TomTom API key. If omitted, ``TOMTOM_API_KEY`` is used.
        search_url: Search API URL prefix.
        routing_url: Routing API URL prefix.
        timeout_s: HTTP timeout in seconds.

    The provider mirrors the open-map provider boundary:

    \[
    C(o,q)=\operatorname{TomTomSearch}(q,o,r)
    \]

    \[
    R(o,d)=\operatorname{TomTomRoute}(o,d,\mathrm{travelMode}=\mathrm{pedestrian})
    \]
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        search_url: str | None = None,
        routing_url: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Initialize the TomTom provider.

        Args:
            api_key: Optional TomTom API key.
            search_url: Optional Search API prefix.
            routing_url: Optional Routing API prefix.
            timeout_s: Optional request timeout.

        Returns:
            None.
        """

        self.api_key = api_key or os.getenv("TOMTOM_API_KEY") or os.getenv("TOMTOM_KEY") or ""
        self.search_url = (search_url or os.getenv("TOMTOM_SEARCH_URL") or "https://api.tomtom.com/search/2").rstrip("/")
        self.routing_url = (routing_url or os.getenv("TOMTOM_ROUTING_URL") or "https://api.tomtom.com/routing/1").rstrip("/")
        self.timeout_s = float(timeout_s or os.getenv("URBANNAV_PROVIDER_TIMEOUT_S", "60"))
        self.route_strategy = "pedestrian_shortest"

    @property
    def capabilities(self) -> UrbanNavProviderCapabilities:
        """Return provider capabilities.

        Returns:
            Capability descriptor for TomTom.
        """

        return UrbanNavProviderCapabilities(
            provider_name="tomtom_search_routing",
            supports_poi_search=True,
            supports_pedestrian_route=True,
            supports_static_map=False,
            raw={"search_url": self.search_url, "routing_url": self.routing_url, "route_strategy": self.route_strategy},
        )

    async def connect(self) -> None:
        """Validate provider configuration.

        Returns:
            None.

        Raises:
            RuntimeError: If the API key is missing.
        """

        if not self.api_key:
            raise RuntimeError("TomTom provider requires TOMTOM_API_KEY or --tomtom-key")

    async def search_pois(self, *, cache_key: str, request: UrbanNavPoiSearchRequest) -> ReplayPoiResult:
        """Search POIs through TomTom Search.

        Args:
            cache_key: Stable cache key.
            request: Normalized POI search request.

        Returns:
            Replay-compatible POI search record.
        """

        data = await asyncio.to_thread(self._search_pois, request)
        candidates = _parse_tomtom_search_results(data.get("results") or [], origin=request.origin, limit=request.limit)
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
                "raw": dict(request.raw or {}),
                "search_url": self.search_url,
            },
            response=data,
            candidates=candidates,
        )

    async def plan_route(self, *, cache_key: str, request: UrbanNavRouteRequest) -> ReplayRouteResult:
        """Plan a pedestrian route through TomTom Routing.

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
        data = await asyncio.to_thread(self._route, request.origin, request.destination)
        parsed = _parse_tomtom_route(data, origin=request.origin, destination=request.destination.point)
        return ReplayRouteResult(
            cache_key=cache_key,
            provider=self.capabilities.provider_name,
            request={
                "origin": request.origin.to_dict(),
                "destination": request.destination.to_dict(),
                "mode": request.mode,
                "routing_url": self.routing_url,
                "travelMode": "pedestrian",
                "routeType": "shortest",
            },
            response=data,
            polyline=parsed["polyline"],
            distance_m=parsed["distance_m"],
            duration_s=parsed["duration_s"],
        )

    def _search_pois(self, request: UrbanNavPoiSearchRequest) -> dict[str, Any]:
        """Call TomTom POI Search.

        Args:
            request: Normalized POI search request.

        Returns:
            Decoded JSON response.
        """

        params: dict[str, Any] = {
            "key": self.api_key,
            "lat": request.origin.lat,
            "lon": request.origin.lng,
            "radius": int(request.radius_m),
            "limit": int(request.limit),
        }
        if request.locale:
            params["language"] = str(request.locale).split("-")[0]
        category_set = _tomtom_category_set(request)
        if category_set:
            params["categorySet"] = category_set
        query = quote(request.query or "poi")
        url = f"{self.search_url}/poiSearch/{query}.json?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "UrbanNavBenchmark/0.1"})
        with urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _route(self, origin: UrbanNavPoint, destination: UrbanNavDestination) -> dict[str, Any]:
        """Call TomTom Routing.

        Args:
            origin: Route origin.
            destination: Route destination.

        Returns:
            Decoded JSON response.
        """

        locations = f"{origin.lat},{origin.lng}:{destination.lat},{destination.lng}"
        params = urlencode({"key": self.api_key, "travelMode": "pedestrian", "routeType": "shortest"})
        url = f"{self.routing_url}/calculateRoute/{locations}/json?{params}"
        req = Request(url, headers={"User-Agent": "UrbanNavBenchmark/0.1"})
        with urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))


def _tomtom_category_set(request: UrbanNavPoiSearchRequest) -> str | None:
    """Extract TomTom category ids from request metadata.

    Args:
        request: Normalized POI search request.

    Returns:
        Comma-separated category ids, or ``None``.
    """

    raw = dict(request.raw or {})
    values = raw.get("tomtom_category_set") or raw.get("categorySet")
    if isinstance(values, str):
        return values
    if isinstance(values, list):
        clean = [str(item).strip() for item in values if str(item).strip()]
        return ",".join(clean) if clean else None
    return None


def _parse_tomtom_search_results(
    results: list[dict[str, Any]],
    *,
    origin: UrbanNavPoint,
    limit: int,
) -> list[UrbanNavDestination]:
    """Normalize TomTom search results.

    Args:
        results: Raw TomTom result rows.
        origin: Search origin.
        limit: Maximum retained candidates.

    Returns:
        Normalized destinations sorted by provider order.
    """

    candidates: list[UrbanNavDestination] = []
    for result in results:
        position = result.get("position") or {}
        lat = position.get("lat")
        lng = position.get("lon")
        if lat is None or lng is None:
            continue
        point = UrbanNavPoint(lat=float(lat), lng=float(lng))
        poi = result.get("poi") or {}
        categories = poi.get("categories") or []
        category_set = poi.get("categorySet") or []
        poi_type = _tomtom_type(categories=categories, category_set=category_set)
        candidates.append(
            UrbanNavDestination(
                lat=point.lat,
                lng=point.lng,
                name=str(poi.get("name") or result.get("address", {}).get("freeformAddress") or "Unnamed TomTom POI"),
                type=poi_type,
                provider_poi_id=str(result.get("id") or ""),
                route_distance_m=_float_or_default(result.get("dist"), haversine_m(origin, point)),
                raw={"tomtom": result},
            )
        )
    return candidates[: max(1, limit)]


def _parse_tomtom_route(
    data: dict[str, Any],
    *,
    origin: UrbanNavPoint,
    destination: UrbanNavPoint | None,
) -> dict[str, Any]:
    """Normalize a TomTom route response.

    Args:
        data: Raw TomTom Routing response.
        origin: Exact benchmark origin.
        destination: Exact benchmark destination.

    Returns:
        Dictionary containing polyline, distance, and duration.
    """

    routes = data.get("routes") or []
    if not routes:
        return {"polyline": [], "distance_m": None, "duration_s": None}
    route = routes[0]
    points: list[UrbanNavPoint] = []
    for leg in route.get("legs") or []:
        for point in leg.get("points") or []:
            if point.get("latitude") is None or point.get("longitude") is None:
                continue
            points.append(UrbanNavPoint(lat=float(point["latitude"]), lng=float(point["longitude"])))
    if not points or points[0] != origin:
        points.insert(0, origin)
    if destination and (not points or points[-1] != destination):
        points.append(destination)
    summary = route.get("summary") or {}
    return {
        "polyline": _dedupe_points(points),
        "distance_m": _float_or_none(summary.get("lengthInMeters")),
        "duration_s": _float_or_none(summary.get("travelTimeInSeconds")),
    }


def _tomtom_type(*, categories: list[Any], category_set: list[Any]) -> str:
    """Build a compact TomTom POI type.

    Args:
        categories: Human-readable TomTom categories.
        category_set: TomTom category ids.

    Returns:
        Compact type string.
    """

    if category_set:
        first = category_set[0]
        if isinstance(first, dict) and first.get("id") is not None:
            return f"tomtom_category={first['id']}"
        return f"tomtom_category={first}"
    if categories:
        return f"tomtom_category={str(categories[0]).lower().replace(' ', '_')}"
    return "tomtom_poi"


def _dedupe_points(points: list[UrbanNavPoint]) -> list[UrbanNavPoint]:
    """Remove adjacent duplicate points.

    Args:
        points: Route points.

    Returns:
        Deduplicated points.
    """

    output: list[UrbanNavPoint] = []
    for point in points:
        if output and abs(output[-1].lat - point.lat) < 1e-9 and abs(output[-1].lng - point.lng) < 1e-9:
            continue
        output.append(point)
    return output


def _float_or_none(value: Any) -> float | None:
    """Convert value to float.

    Args:
        value: Raw value.

    Returns:
        Float or ``None``.
    """

    try:
        return float(value)
    except Exception:
        return None


def _float_or_default(value: Any, default: float) -> float:
    """Convert value to float with default.

    Args:
        value: Raw value.
        default: Fallback value.

    Returns:
        Float value.
    """

    parsed = _float_or_none(value)
    return default if parsed is None else parsed
