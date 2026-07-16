"""Pydantic argument contracts for student-visible UrbanNav tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MAP_TOOL_NAME = "map_tool"
IMAGE_TOOL_NAME = "image_tool"
USER_INTERACTION_TOOL_NAME = "user_interaction_tool"


class StrictArguments(BaseModel):
    """Base class that rejects undeclared model-emitted arguments."""

    model_config = ConfigDict(extra="forbid")


class MapToolArguments(StrictArguments):
    """Superset of canonical map and route-session operation arguments.

    Args:
        op: Map operation selected by the model.
        query: Destination, POI or category query.
        address: Address used by geocoding.
        location: Search center as text or coordinate object.
        origin: Route origin as text or coordinate object.
        destination: Route destination as text or coordinate object.
        destination_id: Stable destination id from a previous tool response.
        candidate_id: Stable candidate id from a previous tool response.
        route_id: Stable route id from a previous tool response.
        session_id: Stable route-session id.
        via_point: Coordinate object inserted into the route.
        current_pose: Current coordinate object used for replanning.
        asset_types: Requested deterministic context artifact types.
        horizon_m: Route-context horizon in metres.
        frame: Route-context coordinate frame.
        latitude: Latitude used by reverse geocoding.
        longitude: Longitude used by reverse geocoding.
        bbox: Search bounding box.
        min_latitude: Bounding-box minimum latitude.
        min_longitude: Bounding-box minimum longitude.
        max_latitude: Bounding-box maximum latitude.
        max_longitude: Bounding-box maximum longitude.
        category: Search category.
        subcategories: Optional category filters.
        keywords: POI query alias retained as a canonical provider field.
        radius: Search radius in metres.
        sortrule: Provider result sort rule.
        limit: Maximum result count.
        mode: Route travel mode.
        steps: Whether provider route steps are requested.
        via_points: Initial route-session via points.
        session_name: Optional user-visible session name.
        force_replan: Whether a session read should force provider replanning.
        insert_before_index: Via insertion index.
        via_point_index: Via removal index.
        via_point_name: Via removal name.
        expected_state_revision: Optimistic concurrency revision for mutations.
        idempotency_key: Stable deduplication key for mutations.
        source_event_id: External user event grounding the mutation.
    """

    op: str
    query: str | None = None
    address: str | None = None
    location: str | dict[str, Any] | None = None
    origin: str | dict[str, Any] | None = None
    destination: str | dict[str, Any] | None = None
    destination_id: str | None = None
    candidate_id: str | None = None
    route_id: str | None = None
    session_id: str | None = None
    via_point: dict[str, Any] | None = None
    current_pose: dict[str, Any] | None = None
    asset_types: list[str] | None = None
    horizon_m: float | None = Field(default=None, gt=0)
    frame: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    bbox: dict[str, float] | None = None
    min_latitude: float | None = Field(default=None, ge=-90, le=90)
    min_longitude: float | None = Field(default=None, ge=-180, le=180)
    max_latitude: float | None = Field(default=None, ge=-90, le=90)
    max_longitude: float | None = Field(default=None, ge=-180, le=180)
    category: str | None = None
    subcategories: list[str] | None = None
    keywords: str | None = None
    radius: int | None = Field(default=None, gt=0)
    sortrule: str | None = None
    limit: int | None = Field(default=None, gt=0)
    mode: Literal["foot", "bike", "car"] | None = None
    steps: bool | None = None
    via_points: list[dict[str, Any]] | None = None
    session_name: str | None = None
    force_replan: bool | None = None
    insert_before_index: int | None = Field(default=None, ge=0)
    via_point_index: int | None = Field(default=None, ge=0)
    via_point_name: str | None = None
    stop_index: int | None = Field(default=None, ge=0)
    maneuver: str | None = None
    next_maneuver: str | None = None
    distance_m: float | None = Field(default=None, ge=0)
    eta_sec: float | None = Field(default=None, ge=0)
    next_maneuver_distance_m: float | None = Field(default=None, ge=0)
    instruction_text: str | None = None
    waypoint_ego: list[float] | None = None
    next_waypoint: list[float] | None = None
    image_ref: str | None = None
    expected_state_revision: int | None = Field(default=None, ge=0)
    idempotency_key: str | None = None
    source_event_id: str | None = None


class ImageToolArguments(StrictArguments):
    """Arguments for deterministic image crop and zoom operations.

    Args:
        op: Image operation selected by the model.
        image: Logical image key such as ``fpv`` or ``bev``.
        image_path: Explicit source image path.
        bbox: Pixel bounding box ``[x1, y1, x2, y2]``.
    """

    op: Literal["crop", "zoom_in"]
    image: str | None = None
    image_path: str | None = None
    bbox: tuple[float, float, float, float]


class UserInteractionToolArguments(StrictArguments):
    """Arguments for auditable human interaction actions.

    Args:
        op: Interaction operation selected by the model.
        question: Question displayed to the user.
        message: Notification or prompt displayed to the user.
        choices: Legacy string choices.
        options: Structured options with stable ids.
        recommended_option_id: Recommended structured option id.
        requires_user_response: Whether graph execution must interrupt.
        interaction_mode: Blocking mode for user responses.
        motion_policy_while_waiting: Safety policy while interrupted.
        evidence_refs: Tool-result ids grounding the interaction.
        proposal: Proposed route mutation requiring confirmation.
    """

    op: Literal[
        "ask_user",
        "request_clarification",
        "present_options",
        "confirm_route_change",
        "notify_issue",
        "answer_query",
    ]
    question: str | None = None
    message: str | None = None
    choices: list[str] | None = None
    options: list[dict[str, Any]] | None = None
    recommended_option_id: str | None = None
    requires_user_response: bool | None = None
    interaction_mode: Literal["blocking", "non_blocking"] | None = None
    motion_policy_while_waiting: Literal[
        "stop",
        "slow_down",
        "continue_previous_safe_policy",
        "handover_to_safety_controller",
    ] | None = None
    evidence_refs: list[str] | None = None
    proposal: dict[str, Any] | None = None
    interaction_id: str | None = None


_DESTINATION_ID_ALIASES = (
    "destination_id",
    "chosen_destination_id",
    "selected_destination_id",
    "target_destination_id",
    "target_id",
    "new_destination_id",
    "replacement_destination_id",
    "via_destination_id",
    "via_candidate_id",
    "via_id",
)
_CANDIDATE_ID_ALIASES = ("candidate_id", "poi_id", "place_id")
_ROUTE_ID_ALIASES = ("route_id", "chosen_route_id", "selected_route_id", "last_route_id")
_SESSION_ID_ALIASES = ("session_id", "route_session_id")
_STOP_INDEX_ALIASES = ("stop_index", "goal_index", "goal_order", "leg_index", "stop_number")
_MODE_ALIASES = ("mode", "travel_mode", "route_mode")
_QUERY_ALIASES = ("query", "keyword", "intent", "target_type", "target_types")
_OP_ALIASES = ("op", "operation", "action")


def normalize_map_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize accepted legacy aliases into the strict canonical map schema.

    Args:
        arguments: Raw model-emitted argument mapping.

    Returns:
        Canonical argument mapping. Recognized alias keys are removed so strict
        Pydantic validation can still reject genuinely unknown fields.
    """

    raw = dict(arguments or {})
    normalized = dict(raw)
    _copy_first(normalized, "op", raw, _OP_ALIASES, _as_string)
    _copy_first(normalized, "query", raw, _QUERY_ALIASES, _as_string)
    _copy_first(normalized, "destination_id", raw, _DESTINATION_ID_ALIASES, _as_string)
    _copy_first(normalized, "candidate_id", raw, _CANDIDATE_ID_ALIASES, _as_string)
    _copy_first(normalized, "route_id", raw, _ROUTE_ID_ALIASES, _as_string)
    _copy_first(normalized, "session_id", raw, _SESSION_ID_ALIASES, _as_string)
    _copy_first(normalized, "mode", raw, _MODE_ALIASES, _as_string)
    _copy_first(normalized, "stop_index", raw, _STOP_INDEX_ALIASES, _as_int)
    _copy_point(
        normalized,
        "origin",
        raw,
        (("origin_lat", "origin_lng"), ("origin_latitude", "origin_longitude"),
         ("from_lat", "from_lng"), ("from_latitude", "from_longitude"),
         ("start_lat", "start_lng"), ("start_latitude", "start_longitude")),
    )
    _copy_point(
        normalized,
        "destination",
        raw,
        (("destination_lat", "destination_lng"), ("destination_latitude", "destination_longitude"),
         ("to_lat", "to_lng"), ("to_latitude", "to_longitude"),
         ("target_lat", "target_lng"), ("target_latitude", "target_longitude")),
    )
    _copy_point(
        normalized,
        "current_pose",
        raw,
        (("current_lat", "current_lng"), ("current_latitude", "current_longitude"),
         ("current_origin_lat", "current_origin_lng"),
         ("current_origin_latitude", "current_origin_longitude")),
    )
    _copy_point(
        normalized,
        "via_point",
        raw,
        (("via_lat", "via_lng"), ("via_latitude", "via_longitude"),
         ("waypoint_lat", "waypoint_lng"), ("waypoint_latitude", "waypoint_longitude")),
    )
    alias_keys = set(
        _DESTINATION_ID_ALIASES
        + _CANDIDATE_ID_ALIASES
        + _ROUTE_ID_ALIASES
        + _SESSION_ID_ALIASES
        + _STOP_INDEX_ALIASES
        + _MODE_ALIASES
        + _QUERY_ALIASES
        + _OP_ALIASES
    )
    alias_keys.update(
        key
        for key in raw
        if key.endswith(("_lat", "_lng", "_latitude", "_longitude"))
        and key not in {"latitude", "longitude", "min_latitude", "min_longitude", "max_latitude", "max_longitude"}
    )
    for key in alias_keys:
        if key not in MapToolArguments.model_fields:
            normalized.pop(key, None)
    return normalized


def _copy_first(target: dict[str, Any], canonical: str, source: dict[str, Any], aliases: tuple[str, ...], parser) -> None:
    """Copy the first parseable alias value into a canonical field."""

    if parser(target.get(canonical)) is not None:
        target[canonical] = parser(target.get(canonical))
        return
    for alias in aliases:
        value = parser(source.get(alias))
        if value is not None:
            target[canonical] = value
            return


def _copy_point(
    target: dict[str, Any],
    canonical: str,
    source: dict[str, Any],
    pairs: tuple[tuple[str, str], ...],
) -> None:
    """Build one canonical coordinate object from scalar latitude/longitude aliases."""

    if target.get(canonical) is not None:
        return
    for lat_key, lng_key in pairs:
        if source.get(lat_key) is not None and source.get(lng_key) is not None:
            target[canonical] = {"lat": float(source[lat_key]), "lng": float(source[lng_key])}
            return


def _as_string(value: Any) -> str | None:
    """Return a stripped non-empty string."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any) -> int | None:
    """Return an integer alias value when parseable."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
