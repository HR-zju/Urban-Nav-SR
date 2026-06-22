"""Schemas for the UrbanNav instruction-to-destination/route benchmark.

These dataclasses are intentionally independent from the existing Agentic
Stage2 trace schemas. UrbanNav evaluates frozen OD benchmark rows and saved
predictions, not dialogue protocol quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


URBANNAV_CASE_SCHEMA_VERSION = "urbannav_case_v1"
URBANNAV_PREDICTION_SCHEMA_VERSION = "urbannav_prediction_v1"


def clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop empty optional fields while preserving valid false/zero values.

    Args:
        payload: JSON-serializable dictionary produced by a schema object.

    Returns:
        A shallow copy without ``None``, empty dict, or empty list values.
    """

    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != {} and value != []
    }


@dataclass(frozen=True)
class UrbanNavPoint:
    """A normalized longitude/latitude point.

    Args:
        lat: Latitude in WGS84-like decimal degrees.
        lng: Longitude in WGS84-like decimal degrees.
    """

    lat: float
    lng: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UrbanNavPoint":
        """Create a point from common coordinate field names.

        Args:
            payload: Dictionary containing ``lat/lng``, ``latitude/longitude``,
                or ``y/x`` fields.

        Returns:
            A normalized ``UrbanNavPoint``.

        Raises:
            ValueError: If either coordinate is missing or not finite.
        """

        lat = _required_float(_pick(payload, "lat", "latitude", "y"), "lat")
        lng = _required_float(_pick(payload, "lng", "lon", "longitude", "x"), "lng")
        return cls(lat=lat, lng=lng)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the point.

        Returns:
            A JSON-ready ``{"lat": ..., "lng": ...}`` dictionary.
        """

        return {"lat": self.lat, "lng": self.lng}


@dataclass(frozen=True)
class UrbanNavDestination:
    """A normalized POI/destination object used by GT and predictions.

    Args:
        lat: Optional latitude of the POI or route endpoint.
        lng: Optional longitude of the POI or route endpoint.
        name: Optional display name.
        type: Optional POI type/category.
        provider_poi_id: Stable provider-specific POI id when available.
        place_id: Alternative stable place id when available.
        rank: Offline GT rank; rank 1 is the optimal destination within scope.
        reference_route_distance_m: Frozen route distance from the case origin
            to this destination, used when this destination is matched.
        reference_route: Frozen reference route polyline for this destination.
        route_distance_m: Candidate route distance used in ranked destinations.
        score_breakdown: Optional ranking score components for audit.
        raw: Original payload for debugging and forward compatibility.
    """

    lat: float | None = None
    lng: float | None = None
    name: str | None = None
    type: str | None = None
    provider_poi_id: str | None = None
    place_id: str | None = None
    rank: int | None = None
    reference_route_distance_m: float | None = None
    reference_route: list[UrbanNavPoint] = field(default_factory=list)
    route_distance_m: float | None = None
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UrbanNavDestination":
        """Normalize a POI/destination dictionary.

        Args:
            payload: Raw destination dictionary. Coordinates can be represented
                as explicit lat/lng fields or a provider ``"lng,lat"`` location
                string.

        Returns:
            A normalized ``UrbanNavDestination``.
        """

        route = parse_route(_pick(payload, "reference_route", "route", "polyline") or [])
        lat = _float_or_none(_pick(payload, "lat", "latitude", "y"))
        lng = _float_or_none(_pick(payload, "lng", "lon", "longitude", "x"))
        if (lat is None or lng is None) and isinstance(payload.get("location"), str):
            parsed = parse_lnglat(payload["location"])
            if parsed is not None:
                lat, lng = parsed.lat, parsed.lng
        return cls(
            lat=lat,
            lng=lng,
            name=_str_or_none(_pick(payload, "name", "display_name", "title")),
            type=_str_or_none(_pick(payload, "type", "category", "poi_type")),
            provider_poi_id=_str_or_none(_pick(payload, "provider_poi_id", "poi_id", "id")),
            place_id=_str_or_none(payload.get("place_id")),
            rank=_int_or_none(payload.get("rank")),
            reference_route_distance_m=_float_or_none(payload.get("reference_route_distance_m")),
            reference_route=route,
            route_distance_m=_float_or_none(_pick(payload, "route_distance_m", "distance_m")),
            score_breakdown=dict(payload.get("score_breakdown") or {}),
            raw=dict(payload),
        )

    @property
    def point(self) -> UrbanNavPoint | None:
        """Return the coordinate point when both coordinates are present.

        Returns:
            ``UrbanNavPoint`` if ``lat`` and ``lng`` are available, otherwise
            ``None``.
        """

        if self.lat is None or self.lng is None:
            return None
        return UrbanNavPoint(lat=self.lat, lng=self.lng)

    @property
    def stable_ids(self) -> set[str]:
        """Return provider ids usable for exact identity matching.

        Returns:
            Set containing ``provider_poi_id`` and/or ``place_id`` when present.
        """

        return {value for value in (self.provider_poi_id, self.place_id) if value}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the destination.

        Returns:
            JSON-ready dictionary with empty optional fields omitted.
        """

        return clean_dict(
            {
                "name": self.name,
                "type": self.type,
                "lat": self.lat,
                "lng": self.lng,
                "provider_poi_id": self.provider_poi_id,
                "place_id": self.place_id,
                "rank": self.rank,
                "reference_route_distance_m": self.reference_route_distance_m,
                "reference_route": [point.to_dict() for point in self.reference_route],
                "route_distance_m": self.route_distance_m,
                "score_breakdown": dict(self.score_breakdown),
            }
        )


@dataclass(frozen=True)
class UrbanNavOptimalityScope:
    """The scope in which rank-1 optimality is claimed.

    Args:
        poi_provider: POI provider used for candidate retrieval.
        search_radius_m: Search radius used by the GT builder.
        candidate_limit: Maximum candidates retained for routing.
        route_provider: Route provider used for reference routes.
        route_mode: Travel mode, such as ``foot``.
        route_strategy: Provider route strategy, such as ``default``.
        cost: Ranking cost, usually ``route_distance_m``.
        builder_version: GT builder version string.
        raw: Original scope payload.
    """

    poi_provider: str | None = None
    search_radius_m: float | None = None
    candidate_limit: int | None = None
    route_provider: str | None = None
    route_mode: str | None = None
    route_strategy: str | None = None
    cost: str | None = None
    builder_version: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "UrbanNavOptimalityScope":
        """Normalize an optimality-scope dictionary.

        Args:
            payload: Raw scope dictionary, or ``None`` for an empty scope.

        Returns:
            A normalized ``UrbanNavOptimalityScope``.
        """

        payload = dict(payload or {})
        return cls(
            poi_provider=_str_or_none(payload.get("poi_provider")),
            search_radius_m=_float_or_none(payload.get("search_radius_m")),
            candidate_limit=_int_or_none(payload.get("candidate_limit")),
            route_provider=_str_or_none(payload.get("route_provider")),
            route_mode=_str_or_none(payload.get("route_mode")),
            route_strategy=_str_or_none(payload.get("route_strategy")),
            cost=_str_or_none(payload.get("cost")),
            builder_version=_str_or_none(payload.get("builder_version")),
            raw=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the optimality scope.

        Returns:
            JSON-ready dictionary with empty optional fields omitted.
        """

        return clean_dict(
            {
                "poi_provider": self.poi_provider,
                "search_radius_m": self.search_radius_m,
                "candidate_limit": self.candidate_limit,
                "route_provider": self.route_provider,
                "route_mode": self.route_mode,
                "route_strategy": self.route_strategy,
                "cost": self.cost,
                "builder_version": self.builder_version,
            }
        )


@dataclass(frozen=True)
class UrbanNavCase:
    """One frozen UrbanNav benchmark case.

    Args:
        case_id: Stable case identifier used to align GT and predictions.
        instruction: User demand instruction.
        origin: Origin coordinate.
        gt_destination: Rank-1 destination under ``optimality_scope``.
        acceptable_destinations: Set of destinations considered correct.
        ranked_destinations: Auditable ranked candidate list.
        gt_route: Reference route to the rank-1 destination.
        gt_distance_m: Reference distance for ``gt_route``.
        category: Task category.
        subcategory: Task subcategory.
        city: City label used for reporting slices.
        scene_id: Scene/region id used for reporting slices.
        difficulty: Difficulty label used for reporting slices.
        optimality_scope: Scope where GT optimality is valid.
        cache_keys: Keys into frozen POI/route replay caches.
        metadata: Additional non-scoring metadata.
        raw: Original payload.
        schema_version: UrbanNav case schema version.
    """

    case_id: str
    instruction: str
    origin: UrbanNavPoint | None
    gt_destination: UrbanNavDestination | None
    acceptable_destinations: list[UrbanNavDestination] = field(default_factory=list)
    ranked_destinations: list[UrbanNavDestination] = field(default_factory=list)
    gt_route: list[UrbanNavPoint] = field(default_factory=list)
    gt_distance_m: float | None = None
    category: str | None = None
    subcategory: str | None = None
    city: str | None = None
    scene_id: str | None = None
    difficulty: str | None = None
    optimality_scope: UrbanNavOptimalityScope = field(default_factory=UrbanNavOptimalityScope)
    cache_keys: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    schema_version: str = URBANNAV_CASE_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UrbanNavCase":
        """Normalize one raw GT case.

        Args:
            payload: Raw UrbanNav GT row. The loader also accepts compatible
                CitySeeker-style fields such as ``question`` and
                ``complete_route``.

        Returns:
            A normalized ``UrbanNavCase``.

        Raises:
            ValueError: If a case id cannot be inferred.
        """

        case_id = case_id_from_record(payload)
        gt_destination = destination_from_any(_pick(payload, "gt_destination", "destination", "target"))
        gt_route = parse_route(_pick(payload, "gt_route", "complete_route", "route") or [])
        if gt_destination is None and gt_route:
            end = gt_route[-1]
            gt_destination = UrbanNavDestination(lat=end.lat, lng=end.lng)

        acceptable = [
            dest
            for dest in (destination_from_any(item) for item in _list_or_empty(payload.get("acceptable_destinations")))
            if dest is not None
        ]
        if not acceptable and gt_destination is not None:
            reference_distance = _float_or_none(payload.get("gt_distance_m"))
            acceptable = [
                UrbanNavDestination(
                    lat=gt_destination.lat,
                    lng=gt_destination.lng,
                    name=gt_destination.name,
                    type=gt_destination.type,
                    provider_poi_id=gt_destination.provider_poi_id,
                    place_id=gt_destination.place_id,
                    rank=1,
                    reference_route_distance_m=gt_destination.reference_route_distance_m or reference_distance,
                    reference_route=gt_destination.reference_route or gt_route,
                    raw=gt_destination.raw,
                )
            ]

        ranked = [
            dest
            for dest in (destination_from_any(item) for item in _list_or_empty(payload.get("ranked_destinations")))
            if dest is not None
        ]

        return cls(
            case_id=case_id,
            instruction=str(_pick(payload, "instruction", "question", "prompt", "user_request") or ""),
            origin=point_from_any(_pick(payload, "origin", "start")),
            gt_destination=gt_destination,
            acceptable_destinations=acceptable,
            ranked_destinations=ranked,
            gt_route=gt_route,
            gt_distance_m=_float_or_none(_pick(payload, "gt_distance_m", "total_weight", "distance_m")),
            category=_str_or_none(payload.get("category")),
            subcategory=_str_or_none(payload.get("subcategory")),
            city=_str_or_none(payload.get("city")),
            scene_id=_str_or_none(payload.get("scene_id")),
            difficulty=_str_or_none(payload.get("difficulty")),
            optimality_scope=UrbanNavOptimalityScope.from_dict(payload.get("optimality_scope")),
            cache_keys=dict(payload.get("cache_keys") or {}),
            metadata=dict(payload.get("metadata") or {}),
            raw=dict(payload),
            schema_version=str(payload.get("schema_version") or URBANNAV_CASE_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the case.

        Returns:
            JSON-ready UrbanNav case dictionary.
        """

        return clean_dict(
            {
                "schema_version": self.schema_version,
                "id": self.case_id,
                "city": self.city,
                "scene_id": self.scene_id,
                "category": self.category,
                "subcategory": self.subcategory,
                "difficulty": self.difficulty,
                "instruction": self.instruction,
                "origin": self.origin.to_dict() if self.origin else None,
                "gt_destination": self.gt_destination.to_dict() if self.gt_destination else None,
                "acceptable_destinations": [item.to_dict() for item in self.acceptable_destinations],
                "ranked_destinations": [item.to_dict() for item in self.ranked_destinations],
                "gt_route": [point.to_dict() for point in self.gt_route],
                "gt_distance_m": self.gt_distance_m,
                "optimality_scope": self.optimality_scope.to_dict(),
                "cache_keys": dict(self.cache_keys),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class UrbanNavPrediction:
    """One saved model/system prediction for an UrbanNav case.

    Args:
        case_id: Case id aligned with ``UrbanNavCase.case_id``.
        chosen_destination: Top-1 predicted destination.
        candidate_destinations: Ranked predicted candidate list.
        route_polyline: Predicted route polyline.
        destination_sequence: Ordered predicted destinations for multi-stop
            tasks. Empty for ordinary single-destination tasks.
        route_legs: Ordered predicted route legs for multi-stop tasks.
        route_distance_m: Predicted route distance, if provided by the system.
        route_summary: Raw route summary fields.
        trace: Optional diagnostic trace; not used by main metrics.
        raw: Original payload.
        schema_version: UrbanNav prediction schema version.
    """

    case_id: str
    chosen_destination: UrbanNavDestination | None = None
    candidate_destinations: list[UrbanNavDestination] = field(default_factory=list)
    route_polyline: list[UrbanNavPoint] = field(default_factory=list)
    destination_sequence: list[UrbanNavDestination] = field(default_factory=list)
    route_legs: list[dict[str, Any]] = field(default_factory=list)
    route_distance_m: float | None = None
    route_summary: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    schema_version: str = URBANNAV_PREDICTION_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload_outer: dict[str, Any]) -> "UrbanNavPrediction":
        """Normalize one raw prediction row.

        Args:
            payload_outer: Prediction dictionary. It may be flat or wrapped
                inside ``prediction/result/output/recommendation``.

        Returns:
            A normalized ``UrbanNavPrediction``.

        Raises:
            ValueError: If a case id cannot be inferred.
        """

        payload = _prediction_payload(payload_outer)
        route_summary = dict(payload.get("route_summary") or {})
        route = parse_route(_pick(payload, "route_polyline", "pred_route", "complete_route", "route") or [])
        candidates = [
            dest
            for dest in (destination_from_any(item) for item in _candidate_items(payload))
            if dest is not None
        ]
        destination_sequence = [
            dest
            for dest in (
                destination_from_any(item)
                for item in _list_or_empty(
                    _pick(payload, "pred_destination_sequence", "destination_sequence", "chosen_destination_sequence")
                )
            )
            if dest is not None
        ]
        chosen = destination_from_any(_pick(payload, "chosen_destination", "pred_destination", "destination"))
        if chosen is None and destination_sequence:
            chosen = destination_sequence[-1]
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is None and route:
            end = route[-1]
            chosen = UrbanNavDestination(lat=end.lat, lng=end.lng)
        return cls(
            case_id=case_id_from_record(payload),
            chosen_destination=chosen,
            candidate_destinations=candidates,
            route_polyline=route,
            destination_sequence=destination_sequence,
            route_legs=list(_list_or_empty(_pick(payload, "pred_route_legs", "route_legs"))),
            route_distance_m=_float_or_none(
                _pick(payload, "pred_distance_m", "route_distance_m", "distance_m")
                or route_summary.get("distance_m")
                or route_summary.get("distance")
            ),
            route_summary=route_summary,
            trace=dict(payload.get("trace") or {}),
            raw=dict(payload_outer),
            schema_version=str(payload.get("schema_version") or URBANNAV_PREDICTION_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction.

        Returns:
            JSON-ready UrbanNav prediction dictionary.
        """

        return clean_dict(
            {
                "schema_version": self.schema_version,
                "id": self.case_id,
                "chosen_destination": self.chosen_destination.to_dict() if self.chosen_destination else None,
                "candidate_destinations": [item.to_dict() for item in self.candidate_destinations],
                "pred_destination_sequence": [item.to_dict() for item in self.destination_sequence],
                "pred_route_legs": list(self.route_legs),
                "route_polyline": [point.to_dict() for point in self.route_polyline],
                "route_summary": dict(self.route_summary),
                "trace": dict(self.trace),
            }
        )


def case_id_from_record(payload: dict[str, Any]) -> str:
    """Infer the stable case id from a raw record.

    Args:
        payload: Raw case or prediction dictionary.

    Returns:
        The first non-empty id from supported id fields.

    Raises:
        ValueError: If no supported id field is present.
    """

    for key in ("id", "case_id", "sample_id", "qid", "question_id"):
        value = _str_or_none(payload.get(key))
        if value:
            return value
    raise ValueError(f"Cannot infer UrbanNav case id from keys={sorted(payload.keys())}")


def point_from_any(value: Any) -> UrbanNavPoint | None:
    """Parse a point from common coordinate formats.

    Args:
        value: A point object, ``"lng,lat"`` string, ``[lng, lat]`` list, or
            dictionary with coordinate fields.

    Returns:
        ``UrbanNavPoint`` when parsing succeeds, otherwise ``None``.
    """

    if value is None:
        return None
    if isinstance(value, UrbanNavPoint):
        return value
    if isinstance(value, str):
        return parse_lnglat(value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lng = _float_or_none(value[0])
        lat = _float_or_none(value[1])
        if lat is None or lng is None:
            return None
        return UrbanNavPoint(lat=lat, lng=lng)
    if isinstance(value, dict):
        if isinstance(value.get("location"), str):
            parsed = parse_lnglat(value["location"])
            if parsed is not None:
                return parsed
        coordinates = value.get("coordinates") if isinstance(value.get("coordinates"), dict) else {}
        lat = _float_or_none(_pick(value, "lat", "latitude", "y") or _pick(coordinates, "lat", "latitude"))
        lng = _float_or_none(_pick(value, "lng", "lon", "longitude", "x") or _pick(coordinates, "lng", "lon", "longitude"))
        if lat is None or lng is None:
            return None
        return UrbanNavPoint(lat=lat, lng=lng)
    return None


def destination_from_any(value: Any) -> UrbanNavDestination | None:
    """Parse a destination from a raw value.

    Args:
        value: Destination dictionary, point-compatible value, or existing
            ``UrbanNavDestination``.

    Returns:
        ``UrbanNavDestination`` when parsing succeeds, otherwise ``None``.
    """

    if value is None:
        return None
    if isinstance(value, UrbanNavDestination):
        return value
    if isinstance(value, dict):
        return UrbanNavDestination.from_dict(value)
    point = point_from_any(value)
    if point is None:
        return None
    return UrbanNavDestination(lat=point.lat, lng=point.lng)


def parse_lnglat(text: str) -> UrbanNavPoint | None:
    """Parse a provider-style ``"lng,lat"`` coordinate string.

    Args:
        text: Coordinate string with longitude first and latitude second.

    Returns:
        Parsed point when valid and in geographic bounds, otherwise ``None``.
    """

    try:
        lng_text, lat_text = str(text).split(",", 1)
        lng = float(lng_text.strip())
        lat = float(lat_text.strip())
    except Exception:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    return UrbanNavPoint(lat=lat, lng=lng)


def parse_route(value: Any) -> list[UrbanNavPoint]:
    """Parse a route polyline from a raw list.

    Args:
        value: List of point-compatible items. Each item can be ``[lng, lat]``,
            ``"lng,lat"``, or a coordinate dictionary.

    Returns:
        List of successfully parsed route points. Invalid items are skipped.
    """

    if not isinstance(value, list):
        return []
    points: list[UrbanNavPoint] = []
    for item in value:
        point = point_from_any(item)
        if point is not None:
            points.append(point)
    return points


def _prediction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap nested prediction payloads while preserving id fields.

    Args:
        payload: Raw prediction record.

    Returns:
        Flat prediction payload.
    """

    for key in ("prediction", "result", "output", "recommendation"):
        value = payload.get(key)
        if isinstance(value, dict):
            merged = dict(value)
            for passthrough in ("id", "case_id", "sample_id", "qid", "question_id"):
                if passthrough in payload and passthrough not in merged:
                    merged[passthrough] = payload[passthrough]
            return merged
    return payload


def _candidate_items(payload: dict[str, Any]) -> list[Any]:
    """Extract a raw candidate list from a prediction payload.

    Args:
        payload: Flat prediction payload.

    Returns:
        Candidate item list, or a single chosen destination as fallback.
    """

    for key in ("candidate_destinations", "recommendations", "candidates", "ranked_destinations"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    chosen = _pick(payload, "chosen_destination", "pred_destination")
    return [chosen] if chosen else []


def _pick(payload: dict[str, Any], *keys: str) -> Any:
    """Return the first present non-empty field.

    Args:
        payload: Source dictionary.
        *keys: Candidate field names in priority order.

    Returns:
        First non-empty value, otherwise ``None``.
    """

    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _list_or_empty(value: Any) -> list[Any]:
    """Normalize a maybe-list value.

    Args:
        value: Any raw value.

    Returns:
        ``value`` if it is a list, otherwise an empty list.
    """

    return value if isinstance(value, list) else []


def _str_or_none(value: Any) -> str | None:
    """Normalize optional string values.

    Args:
        value: Raw value.

    Returns:
        Stripped string, or ``None`` if empty.
    """

    text = str(value or "").strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    """Normalize optional numeric values.

    Args:
        value: Raw value.

    Returns:
        Finite float, or ``None`` when parsing fails.
    """

    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


def _required_float(value: Any, field_name: str) -> float:
    """Parse a required float.

    Args:
        value: Raw value.
        field_name: Field name used in the error message.

    Returns:
        Parsed finite float.

    Raises:
        ValueError: If parsing fails.
    """

    parsed = _float_or_none(value)
    if parsed is None:
        raise ValueError(f"Missing or invalid {field_name}")
    return parsed


def _int_or_none(value: Any) -> int | None:
    """Normalize optional integer values.

    Args:
        value: Raw value.

    Returns:
        Integer, or ``None`` when parsing fails.
    """

    try:
        return int(value)
    except Exception:
        return None
