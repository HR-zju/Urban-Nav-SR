"""Replay map backend for deterministic UrbanNav benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from mcp_server.eval.urbannav_io import iter_json_records
from mcp_server.eval.urbannav_schema import UrbanNavDestination, UrbanNavPoint, parse_route


@dataclass(frozen=True)
class ReplayPoiResult:
    """Frozen POI search response.

    Args:
        cache_key: Replay cache key.
        provider: Provider name saved by the GT builder.
        request: Original provider request.
        response: Original provider response.
        candidates: Normalized candidate destinations.
    """

    cache_key: str
    provider: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    candidates: list[UrbanNavDestination] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayPoiResult":
        """Parse one POI replay cache record.

        Args:
            payload: Raw cache dictionary. Candidate lists may be stored under
                ``normalized_candidates``, ``candidates``, ``results``, or
                ``pois``.

        Returns:
            Parsed ``ReplayPoiResult``.
        """

        candidates_raw = (
            payload.get("normalized_candidates")
            or payload.get("candidates")
            or payload.get("results")
            or payload.get("pois")
            or []
        )
        candidates = [
            destination
            for destination in (UrbanNavDestination.from_dict(item) for item in candidates_raw if isinstance(item, dict))
            if destination is not None
        ]
        return cls(
            cache_key=str(payload.get("cache_key") or payload.get("key") or ""),
            provider=_str_or_none(payload.get("provider")),
            request=dict(payload.get("request") or {}),
            response=dict(payload.get("response") or {}),
            candidates=candidates,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the POI replay record.

        Returns:
            JSON-ready dictionary.
        """

        return {
            "cache_key": self.cache_key,
            "provider": self.provider,
            "request": dict(self.request),
            "response": dict(self.response),
            "normalized_candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class ReplayRouteResult:
    """Frozen route planning response.

    Args:
        cache_key: Replay cache key.
        provider: Provider name saved by the GT builder.
        request: Original provider request.
        response: Original provider response.
        polyline: Normalized route polyline.
        distance_m: Route distance in meters.
        duration_s: Route duration in seconds when available.
    """

    cache_key: str
    provider: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    polyline: list[UrbanNavPoint] = field(default_factory=list)
    distance_m: float | None = None
    duration_s: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayRouteResult":
        """Parse one route replay cache record.

        Args:
            payload: Raw route cache dictionary.

        Returns:
            Parsed ``ReplayRouteResult``.
        """

        normalized = payload.get("normalized_route") if isinstance(payload.get("normalized_route"), dict) else {}
        polyline = parse_route(normalized.get("polyline") or payload.get("polyline") or payload.get("route") or [])
        return cls(
            cache_key=str(payload.get("cache_key") or payload.get("key") or ""),
            provider=_str_or_none(payload.get("provider")),
            request=dict(payload.get("request") or {}),
            response=dict(payload.get("response") or {}),
            polyline=polyline,
            distance_m=_float_or_none(normalized.get("distance_m") or payload.get("distance_m")),
            duration_s=_float_or_none(normalized.get("duration_s") or payload.get("duration_s")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the route replay record.

        Returns:
            JSON-ready dictionary.
        """

        return {
            "cache_key": self.cache_key,
            "provider": self.provider,
            "request": dict(self.request),
            "response": dict(self.response),
            "normalized_route": {
                "polyline": [point.to_dict() for point in self.polyline],
                "distance_m": self.distance_m,
                "duration_s": self.duration_s,
            },
        }


class ReplayMapBackend:
    r"""Map backend backed by frozen POI and route JSONL caches.

    Args:
        poi_cache_path: Path to ``poi_cache.jsonl``.
        route_cache_path: Path to ``route_cache.jsonl``.

    The backend is deterministic:

    \[
    response = cache[cache\_key]
    \]

    No live map API is called.
    """

    def __init__(self, *, poi_cache_path: str | Path | None = None, route_cache_path: str | Path | None = None) -> None:
        """Initialize a replay backend from cache files.

        Args:
            poi_cache_path: Optional path to ``poi_cache.jsonl``.
            route_cache_path: Optional path to ``route_cache.jsonl``.

        Returns:
            None.
        """

        self.poi_by_key = load_poi_cache(poi_cache_path) if poi_cache_path else {}
        self.route_by_key = load_route_cache(route_cache_path) if route_cache_path else {}

    def search_destinations(self, *, cache_key: str) -> ReplayPoiResult:
        """Return frozen POI search results by key.

        Args:
            cache_key: Key from ``case.cache_keys["poi_search"]``.

        Returns:
            Frozen ``ReplayPoiResult``.

        Raises:
            KeyError: If the key is absent.
        """

        return self.poi_by_key[cache_key]

    def plan_route(self, *, cache_key: str) -> ReplayRouteResult:
        """Return frozen route result by key.

        Args:
            cache_key: Key from ``case.cache_keys["routes"]``.

        Returns:
            Frozen ``ReplayRouteResult``.

        Raises:
            KeyError: If the key is absent.
        """

        return self.route_by_key[cache_key]


class ReplayMapBackendClient:
    r"""HTTP client for a shared replay cache service.

    Args:
        base_url: Base URL of ``serve_urbannav_replay_cache.py``.
        timeout_s: HTTP request timeout.

    This client exposes the same lookup methods as :class:`ReplayMapBackend`
    while avoiding per-worker loading of large replay cache JSONL files.
    """

    def __init__(self, *, base_url: str, timeout_s: float = 30.0) -> None:
        """Initialize a replay cache service client.

        Args:
            base_url: Cache service base URL.
            timeout_s: HTTP request timeout.

        Returns:
            None.
        """

        self.base_url = str(base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("ReplayMapBackendClient requires a non-empty base_url")
        self.timeout_s = max(1.0, float(timeout_s or 30.0))
        self._opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))

    def search_destinations(self, *, cache_key: str) -> ReplayPoiResult:
        """Return frozen POI search results by key.

        Args:
            cache_key: Key from ``case.cache_keys["poi_search"]``.

        Returns:
            Frozen ``ReplayPoiResult``.
        """

        return ReplayPoiResult.from_dict(self._get_json("/poi", cache_key=cache_key))

    def plan_route(self, *, cache_key: str) -> ReplayRouteResult:
        """Return frozen route result by key.

        Args:
            cache_key: Key from ``case.cache_keys["routes"]``.

        Returns:
            Frozen ``ReplayRouteResult``.
        """

        return ReplayRouteResult.from_dict(self._get_json("/route", cache_key=cache_key))

    def _get_json(self, path: str, *, cache_key: str) -> dict[str, Any]:
        """Fetch one cache record from the replay cache service.

        Args:
            path: Service path, for example ``/poi`` or ``/route``.
            cache_key: Replay cache key.

        Returns:
            JSON dictionary payload.

        Raises:
            KeyError: If the service reports a missing key.
            RuntimeError: If the service call fails.
        """

        query = urlparse.urlencode({"cache_key": str(cache_key or "")})
        url = f"{self.base_url}{path}?{query}"
        request = urlrequest.Request(url, method="GET")
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urlerror.HTTPError as exc:
            detail = exc.read(512).decode("utf-8", errors="replace")
            if exc.code == 404:
                raise KeyError(cache_key) from exc
            raise RuntimeError(f"replay_cache_service_http_{exc.code}:{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"replay_cache_service_error:{type(exc).__name__}:{exc}") from exc
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("replay_cache_service_error:non_object_payload")
        return payload


def load_poi_cache(path: str | Path | None) -> dict[str, ReplayPoiResult]:
    """Load a POI replay cache file.

    Args:
        path: JSON/JSONL cache path.

    Returns:
        Mapping from cache key to ``ReplayPoiResult``.
    """

    if not path:
        return {}
    records: dict[str, ReplayPoiResult] = {}
    for payload in iter_json_records(path):
        item = ReplayPoiResult.from_dict(payload)
        if item.cache_key:
            records[item.cache_key] = item
    return records


def load_route_cache(path: str | Path | None) -> dict[str, ReplayRouteResult]:
    """Load a route replay cache file.

    Args:
        path: JSON/JSONL cache path.

    Returns:
        Mapping from cache key to ``ReplayRouteResult``.
    """

    if not path:
        return {}
    records: dict[str, ReplayRouteResult] = {}
    for payload in iter_json_records(path):
        item = ReplayRouteResult.from_dict(payload)
        if item.cache_key:
            records[item.cache_key] = item
    return records


def _str_or_none(value: Any) -> str | None:
    """Normalize optional strings.

    Args:
        value: Raw value.

    Returns:
        Stripped string or ``None``.
    """

    text = str(value or "").strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    """Normalize optional floats.

    Args:
        value: Raw value.

    Returns:
        Parsed float or ``None``.
    """

    try:
        return float(value)
    except Exception:
        return None
