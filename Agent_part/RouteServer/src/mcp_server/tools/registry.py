"""Single source of truth for tool schemas, validation and executor capabilities."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mcp_server.tools.specs import (
    IMAGE_TOOL_NAME,
    MAP_TOOL_NAME,
    USER_INTERACTION_TOOL_NAME,
    ImageToolArguments,
    MapToolArguments,
    UserInteractionToolArguments,
    normalize_map_tool_arguments,
)


ArgumentValidator = Callable[[dict[str, Any]], list[str]]


class ToolOperationSpec(BaseModel):
    """One registered operation under a student-visible meta-tool.

    Args:
        tool: Stable meta-tool name.
        op: Stable operation name.
        description: Model-visible operation description.
        argument_model: Pydantic model validating the operation arguments.
        executor_profiles: Executor profiles that implement the operation.
        side_effect: Whether execution mutates route or dialogue state.
        validator: Optional operation-level semantic validator.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    tool: str
    op: str
    description: str
    argument_model: type[BaseModel]
    executor_profiles: frozenset[str] = Field(default_factory=frozenset)
    side_effect: bool = False
    validator: ArgumentValidator | None = None


class ToolSpecRegistry:
    """Registry generating tool schemas and validating calls from one spec set.

    Args:
        specs: Registered operation specifications.
        tool_descriptions: Model-visible descriptions keyed by meta-tool name.

    Raises:
        ValueError: If duplicate ``(tool, op)`` registrations are present.
    """

    def __init__(self, specs: list[ToolOperationSpec], *, tool_descriptions: dict[str, str]) -> None:
        self._specs: dict[tuple[str, str], ToolOperationSpec] = {}
        self._tool_descriptions = dict(tool_descriptions)
        for spec in specs:
            key = (spec.tool, spec.op)
            if key in self._specs:
                raise ValueError(f"duplicate tool operation: {spec.tool}.{spec.op}")
            self._specs[key] = spec

    def operations(self, tool: str) -> set[str]:
        """Return all registered operations for one meta-tool.

        Args:
            tool: Student-visible tool name.

        Returns:
            Registered operation names.
        """

        return {op for registered_tool, op in self._specs if registered_tool == tool}

    def capabilities(self, profile: str) -> set[tuple[str, str]]:
        """Return capabilities implemented by one executor profile.

        Args:
            profile: Executor profile name.

        Returns:
            ``(tool, op)`` capability pairs.
        """

        return {key for key, spec in self._specs.items() if profile in spec.executor_profiles}

    def validate_call(self, call: Any) -> tuple[bool, list[str], dict[str, Any] | None]:
        """Validate and canonicalize a legacy or v2 tool call.

        Args:
            call: Tool call object exposing ``tool`` and ``arguments``, or a mapping.

        Returns:
            Tuple ``(valid, errors, canonical_arguments)``.
        """

        if isinstance(call, dict):
            tool = str(call.get("tool") or call.get("name") or "").strip()
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {
                    key: value for key, value in call.items() if key not in {"tool", "name", "call_id", "id"}
                }
        else:
            tool = str(getattr(call, "tool", "") or "").strip()
            arguments = getattr(call, "arguments", {})
        if not isinstance(arguments, dict):
            return False, ["arguments_not_object"], None
        canonical = normalize_map_tool_arguments(arguments) if tool == MAP_TOOL_NAME else dict(arguments)
        op = str(canonical.get("op") or "").strip()
        if not tool or tool not in self._tool_descriptions:
            return False, [f"unknown_tool:{tool}"], canonical
        if not op:
            return False, ["missing_op"], canonical
        spec = self._specs.get((tool, op))
        if spec is None:
            return False, [f"unsupported_op:{tool}.{op}"], canonical
        try:
            validated = spec.argument_model.model_validate(canonical).model_dump(exclude_none=True)
        except ValidationError as exc:
            errors = [
                "invalid_argument:" + ".".join(str(part) for part in item["loc"]) + ":" + str(item["type"])
                for item in exc.errors()
            ]
            return False, errors, canonical
        semantic_errors = spec.validator(validated) if spec.validator is not None else []
        return not semantic_errors, semantic_errors, validated

    def openai_definitions(self) -> list[dict[str, Any]]:
        """Generate backward-compatible OpenAI function definitions.

        Returns:
            One strict JSON schema per student-visible meta-tool.
        """

        definitions: list[dict[str, Any]] = []
        for tool in self._tool_descriptions:
            operations = sorted(self.operations(tool))
            models = {self._specs[(tool, op)].argument_model for op in operations}
            properties: dict[str, Any] = {}
            required = {"op"}
            for model in models:
                schema = model.model_json_schema()
                properties.update(schema.get("properties") or {})
            properties["op"] = {"type": "string", "enum": operations, "description": "Operation name."}
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool,
                        "description": self._tool_descriptions[tool],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": sorted(required),
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return definitions

    def assert_capabilities(self, profile: str, actual: set[tuple[str, str]]) -> None:
        """Assert an executor exactly matches its declared capability profile.

        Args:
            profile: Registry executor profile.
            actual: Capabilities reported by the executor implementation.

        Raises:
            AssertionError: If operations are missing or unexpectedly exposed.
        """

        expected = self.capabilities(profile)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise AssertionError(f"capability mismatch for {profile}: missing={missing}, extra={extra}")


def _require_any(*groups: tuple[str, ...]) -> ArgumentValidator:
    """Build a semantic validator requiring every field group to have one value."""

    def validate(arguments: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for group in groups:
            if not any(arguments.get(field) is not None for field in group):
                errors.append("missing_any_of:" + "|".join(group))
        return errors

    return validate


def _validate_route_plan(arguments: dict[str, Any]) -> list[str]:
    """Validate route planning by coordinates or prior destination reference."""

    has_pair = arguments.get("origin") is not None and arguments.get("destination") is not None
    has_reference = any(arguments.get(key) is not None for key in ("destination_id", "candidate_id", "route_id"))
    return [] if has_pair or has_reference else ["missing_route_endpoint"]


def _validate_add_via(arguments: dict[str, Any]) -> list[str]:
    """Require a resolved candidate id or an explicit longitude/latitude pair."""

    if arguments.get("destination_id") is not None or arguments.get("candidate_id") is not None:
        return []
    via_point = arguments.get("via_point")
    if not isinstance(via_point, dict):
        return ["missing_grounded_via_point"]
    location = via_point.get("location")
    if isinstance(location, str) and "," in location:
        parts = [part.strip() for part in location.split(",", 1)]
        try:
            longitude, latitude = float(parts[0]), float(parts[1])
        except (TypeError, ValueError):
            return ["invalid_via_point_coordinate"]
        return [] if -180 <= longitude <= 180 and -90 <= latitude <= 90 else ["invalid_via_point_coordinate"]
    coordinate = via_point.get("coordinate") or {}
    longitude = coordinate.get("longitude", via_point.get("longitude"))
    latitude = coordinate.get("latitude", via_point.get("latitude"))
    try:
        valid = longitude is not None and latitude is not None and -180 <= float(longitude) <= 180 and -90 <= float(latitude) <= 90
    except (TypeError, ValueError):
        valid = False
    return [] if valid else ["missing_grounded_via_point"]


def _validate_bbox(arguments: dict[str, Any]) -> list[str]:
    """Validate a complete bounding box in nested or flat form."""

    bbox = arguments.get("bbox") or {}
    keys = ("min_latitude", "min_longitude", "max_latitude", "max_longitude")
    return [] if all(arguments.get(key) is not None or bbox.get(key) is not None for key in keys) else ["missing_bbox"]


def _validate_image(arguments: dict[str, Any]) -> list[str]:
    """Require one image source in addition to the typed bounding box."""

    return [] if arguments.get("image") or arguments.get("image_path") else ["missing_image"]


def _validate_user_interaction(arguments: dict[str, Any]) -> list[str]:
    """Validate user-visible content and blocking interaction semantics."""

    errors: list[str] = []
    op = str(arguments.get("op") or "")
    if not (arguments.get("message") or arguments.get("question")):
        errors.append("missing_message")
    if op == "present_options" and not (arguments.get("options") or arguments.get("choices")):
        errors.append("missing_options")
    requires_user = bool(arguments.get("requires_user_response", op not in {"answer_query", "notify_issue"}))
    if requires_user and arguments.get("interaction_mode") not in {None, "blocking"}:
        errors.append("requires_user_response_must_be_blocking")
    return errors


_LIVE = frozenset({"live"})
_LIVE_REPLAY = frozenset({"live", "urbannav_replay"})


def _spec(
    tool: str,
    op: str,
    description: str,
    model: type[BaseModel],
    profiles: frozenset[str],
    *,
    side_effect: bool = False,
    validator: ArgumentValidator | None = None,
) -> ToolOperationSpec:
    """Construct one concise operation specification."""

    return ToolOperationSpec(
        tool=tool,
        op=op,
        description=description,
        argument_model=model,
        executor_profiles=profiles,
        side_effect=side_effect,
        validator=validator,
    )


_MAP_SPECS = [
    _spec(MAP_TOOL_NAME, "geocode", "Resolve an address.", MapToolArguments, _LIVE, validator=_require_any(("query", "address"))),
    _spec(MAP_TOOL_NAME, "reverse_geocode", "Resolve a coordinate.", MapToolArguments, _LIVE, validator=_require_any(("latitude",), ("longitude",))),
    _spec(MAP_TOOL_NAME, "search_destination", "Search destination candidates.", MapToolArguments, _LIVE_REPLAY, validator=_require_any(("query", "keywords", "category"))),
    _spec(MAP_TOOL_NAME, "search_poi", "Search POI candidates.", MapToolArguments, _LIVE_REPLAY, validator=_require_any(("query", "keywords", "category"))),
    _spec(MAP_TOOL_NAME, "search_category", "Search one category in a bounding box.", MapToolArguments, _LIVE, validator=_validate_bbox),
    _spec(MAP_TOOL_NAME, "plan_route", "Plan a route.", MapToolArguments, _LIVE_REPLAY, validator=_validate_route_plan),
    _spec(MAP_TOOL_NAME, "get_route", "Read or plan a route.", MapToolArguments, _LIVE, validator=_require_any(("session_id", "route_id", "origin"))),
    _spec(MAP_TOOL_NAME, "get_route_context", "Read deterministic route context.", MapToolArguments, _LIVE, validator=_require_any(("session_id", "route_id"))),
    _spec(MAP_TOOL_NAME, "create_route_session", "Create a route session.", MapToolArguments, _LIVE_REPLAY, side_effect=True, validator=_validate_route_plan),
    # Replay benchmark state can supply the active session implicitly; live
    # execution still reports a structured missing-session error when needed.
    _spec(MAP_TOOL_NAME, "get_route_session", "Read a route session.", MapToolArguments, _LIVE_REPLAY),
    _spec(MAP_TOOL_NAME, "add_via", "Add a resolved via point and replan.", MapToolArguments, _LIVE_REPLAY, side_effect=True, validator=_validate_add_via),
    _spec(MAP_TOOL_NAME, "add_via_and_replan", "Add a resolved via point and replan.", MapToolArguments, _LIVE_REPLAY, side_effect=True, validator=_validate_add_via),
    _spec(MAP_TOOL_NAME, "remove_via_and_replan", "Remove a via point and replan.", MapToolArguments, _LIVE, side_effect=True, validator=_require_any(("via_point_index", "via_point_name"))),
    _spec(MAP_TOOL_NAME, "replace_destination", "Replace the active destination.", MapToolArguments, _LIVE_REPLAY, side_effect=True, validator=_require_any(("destination", "destination_id", "candidate_id"))),
    _spec(MAP_TOOL_NAME, "replan_from_current_pose", "Replan from the current pose.", MapToolArguments, _LIVE_REPLAY, side_effect=True, validator=_require_any(("origin", "current_pose"))),
]
_IMAGE_SPECS = [
    _spec(IMAGE_TOOL_NAME, "crop", "Crop an image region.", ImageToolArguments, _LIVE, validator=_validate_image),
    _spec(IMAGE_TOOL_NAME, "zoom_in", "Zoom into an image region.", ImageToolArguments, _LIVE, validator=_validate_image),
]
_USER_SPECS = [
    _spec(USER_INTERACTION_TOOL_NAME, op, "Perform one auditable user interaction.", UserInteractionToolArguments, _LIVE, side_effect=True, validator=_validate_user_interaction)
    for op in ("ask_user", "request_clarification", "present_options", "confirm_route_change", "notify_issue", "answer_query")
]


DEFAULT_TOOL_REGISTRY = ToolSpecRegistry(
    _MAP_SPECS + _IMAGE_SPECS + _USER_SPECS,
    tool_descriptions={
        MAP_TOOL_NAME: "Query map data, plan or update a route session.",
        IMAGE_TOOL_NAME: "Perform deterministic crop or zoom operations on FPV/BEV images.",
        USER_INTERACTION_TOOL_NAME: "Ask, confirm or notify the human user.",
    },
)
