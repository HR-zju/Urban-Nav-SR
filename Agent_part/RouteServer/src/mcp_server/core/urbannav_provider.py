"""Provider abstractions for global UrbanNav benchmark construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.eval.urbannav_schema import UrbanNavDestination, UrbanNavPoint


@dataclass(frozen=True)
class UrbanNavProviderCapabilities:
    """Capabilities exposed by one UrbanNav map provider.

    Args:
        provider_name: Stable provider identifier stored in replay caches.
        supports_poi_search: Whether the provider can search POIs.
        supports_pedestrian_route: Whether the provider can plan pedestrian
            routes.
        supports_static_map: Whether the provider can render review maps.
        raw: Provider-specific capability metadata.
    """

    provider_name: str
    supports_poi_search: bool = True
    supports_pedestrian_route: bool = True
    supports_static_map: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UrbanNavPoiSearchRequest:
    """POI search request normalized across providers.

    Args:
        origin: Search center.
        query: Human-readable query string for audit.
        tags: Provider-normalized category tags such as ``amenity=restaurant``.
        radius_m: Search radius in meters.
        limit: Maximum normalized candidates to retain.
        locale: Locale used for query/tag selection.
        raw: Provider-specific request fields.
    """

    origin: UrbanNavPoint
    query: str
    tags: list[str] = field(default_factory=list)
    radius_m: int = 1500
    limit: int = 10
    locale: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UrbanNavRouteRequest:
    """Route request normalized across providers.

    Args:
        origin: Route origin.
        destination: Route destination.
        mode: Route mode. UrbanNav GT construction uses ``"foot"``.
        raw: Provider-specific request fields.
    """

    origin: UrbanNavPoint
    destination: UrbanNavDestination
    mode: str = "foot"
    raw: dict[str, Any] = field(default_factory=dict)


class UrbanNavLiveProvider(ABC):
    r"""Abstract live provider used by UrbanNav GT builders.

    The provider boundary separates benchmark construction from any specific
    map API:

    \[
    provider(q,o) \rightarrow (C(o,q), R(o,d))
    \]

    GT builders persist returned replay records before evaluation, so live
    providers are never used by the offline evaluator.
    """

    @property
    @abstractmethod
    def capabilities(self) -> UrbanNavProviderCapabilities:
        """Return provider capabilities.

        Returns:
            Capability descriptor.
        """

    async def connect(self) -> None:
        """Connect provider resources.

        Returns:
            None.
        """

    async def disconnect(self) -> None:
        """Disconnect provider resources.

        Returns:
            None.
        """

    @abstractmethod
    async def search_pois(self, *, cache_key: str, request: UrbanNavPoiSearchRequest) -> ReplayPoiResult:
        """Search POIs and return a replay-compatible record.

        Args:
            cache_key: Stable cache key for replay.
            request: Normalized POI search request.

        Returns:
            Frozen POI search record.
        """

    @abstractmethod
    async def plan_route(self, *, cache_key: str, request: UrbanNavRouteRequest) -> ReplayRouteResult:
        """Plan a route and return a replay-compatible record.

        Args:
            cache_key: Stable cache key for replay.
            request: Normalized route request.

        Returns:
            Frozen route record.
        """
