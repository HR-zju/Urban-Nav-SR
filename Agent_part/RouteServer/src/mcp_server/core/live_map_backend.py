"""Live map backend for UrbanNav benchmark construction."""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavDestination, UrbanNavPoint, parse_route
from mcp_server.map_providers import (
    MapProviderClient,
    create_map_client,
    format_location,
    parse_amap_pois,
    parse_amap_steps,
    parse_location_text,
)


@dataclass(frozen=True)
class LiveRouteRequest:
    """Route-planning request issued to a live provider.

    Args:
        origin: Route origin point.
        destination: Route destination.
        mode: Provider route mode. UrbanNav benchmark generation uses
            ``"foot"`` by default.
    """

    origin: UrbanNavPoint
    destination: UrbanNavDestination
    mode: str = "foot"


class LiveMapBackend:
    r"""Adapter from RouteServer map providers to UrbanNav replay records.

    Args:
        client: Connected or connectable RouteServer map provider client.
        provider_name: Provider name stored in replay records.

    The backend converts live responses into frozen records:

    \[
    cache\_record = normalize(provider(request))
    \]

    The GT builder persists those records to JSONL so evaluation never depends
    on live API availability.
    """

    def __init__(self, *, client: MapProviderClient, provider_name: str) -> None:
        """Initialize the live backend.

        Args:
            client: RouteServer map provider client.
            provider_name: Provider name such as ``"amap"``.

        Returns:
            None.
        """

        self.client = client
        self.provider_name = provider_name

    @classmethod
    def from_env(cls, *, map_provider: str | None = None) -> "LiveMapBackend":
        """Create a live backend from RouteServer map-provider environment.

        Args:
            map_provider: Optional provider override. When set, it updates
                ``MAP_PROVIDER`` before calling ``create_map_client``.

        Returns:
            Live backend wrapping the configured provider.

        Raises:
            RuntimeError: If the selected provider is missing required
                credentials, for example ``AMAP_KEY`` for AMap.
        """

        if map_provider:
            os.environ["MAP_PROVIDER"] = map_provider
        try:
            client, provider_name = create_map_client()
        except RuntimeError as exc:
            provider = os.getenv("MAP_PROVIDER", "").strip().lower()
            if provider == "amap" and "aiohttp is required" in str(exc):
                client = AmapUrbanNavProvider(api_key=_required_env("AMAP_KEY"))
                provider_name = "amap"
            else:
                raise
        return cls(client=client, provider_name=provider_name)

    async def connect(self) -> None:
        """Connect the wrapped provider client.

        Returns:
            None.
        """

        await self.client.connect()

    async def disconnect(self) -> None:
        """Disconnect the wrapped provider client.

        Returns:
            None.
        """

        await self.client.disconnect()

    async def search_destinations(
        self,
        *,
        cache_key: str,
        origin: UrbanNavPoint,
        keywords: str,
        radius_m: int,
        limit: int,
        sortrule: str = "distance",
    ) -> ReplayPoiResult:
        """Search live POI candidates and return a replay-compatible record.

        Args:
            cache_key: Stable replay cache key for this search.
            origin: Search origin.
            keywords: Provider keyword query. Multiple alternatives may be
                joined by ``"|"`` when the provider supports it.
            radius_m: Provider search radius in meters.
            limit: Maximum candidates to retain.
            sortrule: Provider sorting rule.

        Returns:
            ``ReplayPoiResult`` containing normalized destination candidates.
        """

        location = format_location(origin.lng, origin.lat)
        response = await self.client.search_semantic_pois(
            location=location,
            keywords=keywords,
            radius=radius_m,
            sortrule=sortrule,
            limit=limit,
        )
        candidates = [
            destination
            for destination in (UrbanNavDestination.from_dict(item) for item in response if isinstance(item, dict))
            if destination.point is not None
        ]
        return ReplayPoiResult(
            cache_key=cache_key,
            provider=self.provider_name,
            request={
                "origin": origin.to_dict(),
                "location": location,
                "keywords": keywords,
                "radius_m": radius_m,
                "limit": limit,
                "sortrule": sortrule,
            },
            response={"normalized_pois": response},
            candidates=candidates,
        )

    async def plan_route(self, *, cache_key: str, request: LiveRouteRequest) -> ReplayRouteResult:
        """Plan one live route and return a replay-compatible record.

        Args:
            cache_key: Stable replay cache key for this route.
            request: Origin, destination, and mode.

        Returns:
            ``ReplayRouteResult`` with normalized polyline, distance, and
            duration.
        """

        if request.destination.lat is None or request.destination.lng is None:
            return ReplayRouteResult(
                cache_key=cache_key,
                provider=self.provider_name,
                request=_route_request_payload(request),
                response={"error": "destination_missing_coordinate"},
            )

        origin_text = format_location(request.origin.lng, request.origin.lat)
        destination_text = format_location(request.destination.lng, request.destination.lat)
        response = await self.client.get_coarse_waypoints(
            origin=origin_text,
            destination=destination_text,
            mode=request.mode,
        )
        summary = response.get("route_summary", {}) if isinstance(response.get("route_summary"), dict) else {}
        polyline = _route_with_exact_endpoints(
            parse_route(response.get("route_polyline") or []),
            origin=request.origin,
            destination=UrbanNavPoint(lat=float(request.destination.lat), lng=float(request.destination.lng)),
        )
        return ReplayRouteResult(
            cache_key=cache_key,
            provider=self.provider_name,
            request=_route_request_payload(request),
            response=response,
            polyline=polyline,
            distance_m=_float_or_none(summary.get("distance_m")),
            duration_s=_float_or_none(summary.get("duration_s")),
        )


def _route_request_payload(request: LiveRouteRequest) -> dict[str, Any]:
    """Serialize a live route request.

    Args:
        request: Route request object.

    Returns:
        JSON-ready request payload.
    """

    return {
        "origin": request.origin.to_dict(),
        "destination": request.destination.to_dict(),
        "mode": request.mode,
    }


def _float_or_none(value: Any) -> float | None:
    """Normalize optional numeric values.

    Args:
        value: Raw value.

    Returns:
        Float value, or ``None`` when conversion fails.
    """

    try:
        return float(value)
    except Exception:
        return None


def _route_with_exact_endpoints(
    route: list[UrbanNavPoint],
    *,
    origin: UrbanNavPoint,
    destination: UrbanNavPoint,
) -> list[UrbanNavPoint]:
    """Ensure a live provider route starts and ends at benchmark endpoints.

    Args:
        route: Provider route polyline.
        origin: Benchmark origin.
        destination: Benchmark destination.

    Returns:
        Route polyline with exact origin as first point and exact destination
        as final point.
    """

    points = list(route)
    if not points:
        return [origin, destination]
    if points[0] != origin:
        points.insert(0, origin)
    if points[-1] != destination:
        points.append(destination)
    return _dedupe_points(points)


def _dedupe_points(points: list[UrbanNavPoint]) -> list[UrbanNavPoint]:
    """Remove adjacent duplicate points.

    Args:
        points: Route points.

    Returns:
        Deduplicated route points.
    """

    output: list[UrbanNavPoint] = []
    for point in points:
        if output and abs(output[-1].lat - point.lat) < 1e-9 and abs(output[-1].lng - point.lng) < 1e-9:
            continue
        output.append(point)
    return output


class AmapUrbanNavProvider:
    """Minimal async AMap provider for UrbanNav live GT construction.

    Args:
        api_key: AMap Web Service API key.

    This fallback exists for environments where ``aiohttp`` is not installed.
    It implements only the live-builder operations and uses ``urllib`` in a
    worker thread to keep the public methods async.
    """

    def __init__(self, *, api_key: str) -> None:
        """Initialize the urllib-backed AMap provider.

        Args:
            api_key: AMap Web Service API key.

        Returns:
            None.
        """

        self.api_key = api_key

    async def connect(self) -> None:
        """Connect the provider.

        Returns:
            None.
        """

    async def disconnect(self) -> None:
        """Disconnect the provider.

        Returns:
            None.
        """

    async def geocode(self, query: str) -> list[dict[str, Any]]:
        """Geocode is not used by the UrbanNav live builder.

        Args:
            query: Geocode query.

        Returns:
            Empty result list.
        """

        return []

    async def reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        """Reverse geocode is not used by the UrbanNav live builder.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Empty result dictionary.
        """

        return {}

    async def search_semantic_pois(
        self,
        location: str,
        keywords: str,
        radius: int = 1000,
        sortrule: str = "distance",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search nearby AMap POIs.

        Args:
            location: Origin as ``"lng,lat"``.
            keywords: AMap keyword query.
            radius: Search radius in meters.
            sortrule: AMap sorting rule.
            limit: Maximum candidates.

        Returns:
            Normalized POI dictionaries.
        """

        origin_lon, origin_lat = parse_location_text(location)
        data = await asyncio.to_thread(
            self._request_json,
            "https://restapi.amap.com/v3/place/around",
            {
                "location": location,
                "keywords": keywords,
                "radius": int(radius),
                "sortrule": sortrule,
                "offset": max(1, min(limit, 25)),
                "page": 1,
            },
            "AMap semantic POI search",
        )
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
        """Request an AMap walking route and normalize waypoints.

        Args:
            origin: Origin as ``"lng,lat"``.
            destination: Destination as ``"lng,lat"``.
            mode: Route mode. Values other than ``"car"`` use walking.

        Returns:
            Normalized coarse-waypoint route payload.
        """

        amap_mode = {"car": "driving", "foot": "walking", "bike": "walking"}.get(mode, "walking")
        data = await asyncio.to_thread(
            self._request_json,
            f"https://restapi.amap.com/v3/direction/{amap_mode}",
            {
                "origin": origin,
                "destination": destination,
                "extensions": os.getenv("AMAP_ROUTE_EXTENSIONS", "all").strip() or "all",
            },
            "AMap route",
        )
        return parse_amap_steps(data, origin=origin, destination=destination, mode=mode)

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
        """Return a normalized route in provider-compatible shape.

        Args:
            from_lat: Origin latitude.
            from_lon: Origin longitude.
            to_lat: Destination latitude.
            to_lon: Destination longitude.
            mode: Route mode.
            steps: Whether to include steps. Ignored because AMap returns steps.
            overview: Route overview mode. Ignored.
            annotations: Whether to include annotations. Ignored.

        Returns:
            Dictionary with route payload fields.
        """

        origin = format_location(from_lon, from_lat)
        destination = format_location(to_lon, to_lat)
        coarse = await self.get_coarse_waypoints(origin=origin, destination=destination, mode=mode)
        return {"routes": [{"geometry": {"coordinates": coarse.get("route_polyline", [])}}], "source": "amap"}

    async def get_nearby_pois(
        self,
        lat: float,
        lon: float,
        radius: float = 1000,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Nearby POI search is not used by the live builder.

        Args:
            lat: Latitude.
            lon: Longitude.
            radius: Search radius.
            categories: Optional category list.

        Returns:
            Empty result list.
        """

        return []

    async def search_features_by_category(
        self,
        bbox: tuple[float, float, float, float],
        category: str,
        subcategories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Feature search is not used by the live builder.

        Args:
            bbox: Bounding box.
            category: Feature category.
            subcategories: Optional subcategories.

        Returns:
            Empty result list.
        """

        return []

    def _request_json(self, endpoint: str, params: dict[str, Any], error_context: str) -> dict[str, Any]:
        """Request one AMap JSON endpoint.

        Args:
            endpoint: AMap endpoint URL.
            params: Query parameters excluding ``key``.
            error_context: Error label.

        Returns:
            Decoded JSON response.

        Raises:
            RuntimeError: If the HTTP response or AMap status is unsuccessful.
        """

        timeout_s = float(os.getenv("AMAP_URLLIB_TIMEOUT_S", os.getenv("MAP_HTTP_TOTAL_TIMEOUT_S", "60")) or 60)
        url = f"{endpoint}?{urlencode({'key': self.api_key, **params})}"
        with urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") != "1":
            raise RuntimeError(f"{error_context} failed: {payload.get('info') or payload}")
        return payload


def _required_env(name: str) -> str:
    """Read a required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Stripped environment value.

    Raises:
        RuntimeError: If the variable is empty.
    """

    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
