"""Unified observation, action, and step-result contracts for SideWalk agents.

These dataclasses define the stable boundary between RouteServer runtime
logging and ms-swift model training.  They are intentionally lightweight:
existing Stage2 ``AgentStage2Case`` and ``AgentTrace`` objects remain the
canonical JSONL inputs, while this module provides a normalized view over the
same data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from mcp_server.agent.schemas import AgentStage2Case, AgentTrace, ToolCall
from mcp_server.agent.tool_parser import parse_finish_text


NAV_OBSERVATION_CONTRACT_VERSION = "sidewalk_nav_observation_v1"
ACTION_CONTRACT_VERSION = "sidewalk_nav_action_v1"
AGENT_STEP_RESULT_SCHEMA_VERSION = "sidewalk_agent_step_result_v1"

TEXT_ACTION_EXTERNAL_TOOL = "external_tool_call"
TEXT_ACTION_USER_INTERACTION = "user_interaction"
TEXT_ACTION_NAVIGATION_DECISION = "navigation_decision"

RUNTIME_STATE_RUNNING = "RUNNING"
RUNTIME_STATE_WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
RUNTIME_STATE_WAITING_FOR_USER = "WAITING_FOR_USER"
RUNTIME_STATE_READY_FOR_CONTROL = "READY_FOR_CONTROL"
RUNTIME_STATE_SAFETY_STOP = "SAFETY_STOP"


def _clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop empty optional fields without removing valid false/zero values.

    Args:
        payload: JSON-compatible dictionary.

    Returns:
        Dictionary without ``None``, empty dict, or empty list values.
    """

    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != {} and value != []
    }


@dataclass(frozen=True)
class ContextAsset:
    """Model-visible context produced by a tool response.

    Args:
        asset_id: Stable id unique within one observation or tool response.
        asset_type: Asset semantic type, for example
            ``route_instruction_text`` or ``next_waypoint``.
        source_tool: Tool that produced the asset.
        source_tool_response_id: Optional id of the tool response that contains
            this asset.
        modality: One of ``text``, ``structured``, or ``image``.
        visibility: ``model_visible`` for model inputs, or another explicit
            value for future non-visible audit assets.
        payload: Asset payload.  Image assets should reference artifacts rather
            than embedding binary data.
        frame: Coordinate frame for structured assets when applicable.
        timestamp: Optional source timestamp.
        metadata: Additional audit metadata.
    """

    asset_id: str
    asset_type: str
    source_tool: str | None = None
    source_tool_response_id: str | None = None
    modality: str = "structured"
    visibility: str = "model_visible"
    payload: dict[str, Any] = field(default_factory=dict)
    frame: str | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextAsset":
        """Build a context asset from JSON.

        Args:
            payload: Context asset dictionary.

        Returns:
            ``ContextAsset`` instance.
        """

        return cls(
            asset_id=str(payload.get("asset_id") or ""),
            asset_type=str(payload.get("asset_type") or ""),
            source_tool=payload.get("source_tool"),
            source_tool_response_id=payload.get("source_tool_response_id"),
            modality=str(payload.get("modality") or "structured"),
            visibility=str(payload.get("visibility") or "model_visible"),
            payload=dict(payload.get("payload") or {}),
            frame=payload.get("frame"),
            timestamp=float(payload["timestamp"]) if payload.get("timestamp") is not None else None,
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the context asset.

        Returns:
            JSON-compatible context asset dictionary.
        """

        return _clean_dict(
            {
                "asset_id": self.asset_id,
                "asset_type": self.asset_type,
                "source_tool": self.source_tool,
                "source_tool_response_id": self.source_tool_response_id,
                "modality": self.modality,
                "visibility": self.visibility,
                "frame": self.frame,
                "timestamp": self.timestamp,
                "payload": dict(self.payload),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class NavObservation:
    """Runtime-visible navigation observation.

    Args:
        observation_id: Stable observation id, usually ``case_id`` for single
            tick Stage2 data.
        request: User instruction, dialogue history, and resolved instruction
            updates.
        sensors: FPV/BEV images and sensor metadata.
        route_state: Current route/session/pose state visible to the model.
        dialogue_state: Pending user interaction state.
        tool_context: Available tools, prior tool responses, and
            ``ContextAsset`` entries.
        runtime_meta: Split, tick id, source, and other replay metadata.
        contract_version: Observation contract version.
    """

    observation_id: str
    request: dict[str, Any] = field(default_factory=dict)
    sensors: dict[str, Any] = field(default_factory=dict)
    route_state: dict[str, Any] = field(default_factory=dict)
    dialogue_state: dict[str, Any] = field(default_factory=dict)
    tool_context: dict[str, Any] = field(default_factory=dict)
    runtime_meta: dict[str, Any] = field(default_factory=dict)
    contract_version: str = NAV_OBSERVATION_CONTRACT_VERSION

    @classmethod
    def from_case(cls, case: AgentStage2Case) -> "NavObservation":
        """Normalize an existing Stage2 case into ``NavObservation``.

        Args:
            case: Existing Stage2 case.

        Returns:
            Observation view over ``case`` without copying supervision fields
            into model-visible context.
        """

        observation = dict(case.observation or {})
        request = _normalize_request(case)
        sensors = _normalize_sensors(case, observation)
        route_state = _normalize_route_state(observation)
        tool_context = _normalize_tool_context(case, observation)
        dialogue_state = dict(observation.get("dialogue_state") or {})
        return cls(
            observation_id=str(observation.get("observation_id") or case.case_id),
            request=request,
            sensors=sensors,
            route_state=route_state,
            dialogue_state=dialogue_state,
            tool_context=tool_context,
            runtime_meta={
                "case_id": case.case_id,
                "task_type": case.task_type,
                "split": case.split,
                "locale": case.locale,
            },
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NavObservation":
        """Build an observation from JSON.

        Args:
            payload: Observation dictionary.

        Returns:
            ``NavObservation`` instance.
        """

        return cls(
            observation_id=str(payload.get("observation_id") or ""),
            request=dict(payload.get("request") or {}),
            sensors=dict(payload.get("sensors") or {}),
            route_state=dict(payload.get("route_state") or {}),
            dialogue_state=dict(payload.get("dialogue_state") or {}),
            tool_context=dict(payload.get("tool_context") or {}),
            runtime_meta=dict(payload.get("runtime_meta") or {}),
            contract_version=str(payload.get("contract_version") or NAV_OBSERVATION_CONTRACT_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the observation.

        Returns:
            JSON-compatible observation dictionary.
        """

        return _clean_dict(
            {
                "contract_version": self.contract_version,
                "observation_id": self.observation_id,
                "request": dict(self.request),
                "sensors": dict(self.sensors),
                "route_state": dict(self.route_state),
                "dialogue_state": dict(self.dialogue_state),
                "tool_context": dict(self.tool_context),
                "runtime_meta": dict(self.runtime_meta),
            }
        )


@dataclass(frozen=True)
class ToolAction:
    """System2 text-layer executable action.

    Args:
        tool: Student-visible tool name.
        arguments: Tool arguments, including the required ``op`` field.
        action_type: ``external_tool_call`` or ``user_interaction``.
        decision_meta: Optional reason/evidence metadata.
        call_id: Optional tool call id.
        contract_version: Action contract version.
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    action_type: str = TEXT_ACTION_EXTERNAL_TOOL
    decision_meta: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    contract_version: str = ACTION_CONTRACT_VERSION

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> "ToolAction":
        """Build a text action from a RouteServer ``ToolCall``.

        Args:
            call: Parsed tool call.

        Returns:
            ``ToolAction`` using ``user_interaction`` for user-facing tool
            calls and ``external_tool_call`` otherwise.
        """

        action_type = TEXT_ACTION_USER_INTERACTION if call.tool == "user_interaction_tool" else TEXT_ACTION_EXTERNAL_TOOL
        decision_meta = {}
        if isinstance(call.arguments.get("decision_meta"), dict):
            decision_meta = dict(call.arguments.get("decision_meta") or {})
        return cls(
            tool=call.tool,
            arguments=dict(call.arguments),
            action_type=action_type,
            decision_meta=decision_meta,
            call_id=call.call_id,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolAction":
        """Build a tool action from JSON.

        Args:
            payload: Tool action dictionary or tool-call dictionary.

        Returns:
            ``ToolAction`` instance.
        """

        call = ToolCall.from_dict(payload)
        action_type = str(payload.get("action_type") or "")
        if not action_type:
            action_type = TEXT_ACTION_USER_INTERACTION if call.tool == "user_interaction_tool" else TEXT_ACTION_EXTERNAL_TOOL
        return cls(
            tool=call.tool,
            arguments=dict(call.arguments),
            action_type=action_type,
            decision_meta=dict(payload.get("decision_meta") or {}),
            call_id=call.call_id,
            contract_version=str(payload.get("contract_version") or ACTION_CONTRACT_VERSION),
        )

    def to_tool_call(self) -> ToolCall:
        """Return the executable ``ToolCall`` view.

        Returns:
            ``ToolCall`` with the same tool, arguments, and call id.
        """

        return ToolCall(tool=self.tool, arguments=dict(self.arguments), call_id=self.call_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool action.

        Returns:
            JSON-compatible tool action dictionary.
        """

        return _clean_dict(
            {
                "contract_version": self.contract_version,
                "action_type": self.action_type,
                "tool": self.tool,
                "arguments": dict(self.arguments),
                "decision_meta": dict(self.decision_meta),
                "call_id": self.call_id,
            }
        )


@dataclass(frozen=True)
class NavigationDecision:
    """System2 final navigation decision parsed from ``<finish>``.

    Args:
        route_action: High-level route action such as ``follow_route`` or
            ``request_user_input``.
        intent: Direction or behavior intent.
        instruction_text: Optional user-facing navigation instruction.
        tool_used: Whether the final decision consumed a tool.
        finish: Raw finish payload for backward compatibility.
        user_message: Optional query answer or explanation attached to the
            navigation decision.
        contract_version: Action contract version.
    """

    route_action: str | None = None
    intent: str | None = None
    instruction_text: str | None = None
    tool_used: bool | None = None
    finish: dict[str, Any] = field(default_factory=dict)
    user_message: dict[str, Any] | None = None
    contract_version: str = ACTION_CONTRACT_VERSION

    @classmethod
    def from_finish(cls, finish: dict[str, Any] | None) -> "NavigationDecision | None":
        """Build a navigation decision from parsed finish JSON.

        Args:
            finish: Parsed ``<finish>`` dictionary.

        Returns:
            ``NavigationDecision`` when ``finish`` is a dictionary, otherwise
            ``None``.
        """

        if not isinstance(finish, dict):
            return None
        return cls(
            route_action=str(finish.get("route_action")) if finish.get("route_action") is not None else None,
            intent=str(finish.get("intent")) if finish.get("intent") is not None else None,
            instruction_text=finish.get("instruction_text"),
            tool_used=bool(finish.get("tool_used")) if finish.get("tool_used") is not None else None,
            finish=dict(finish),
            user_message=dict(finish.get("user_message")) if isinstance(finish.get("user_message"), dict) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the navigation decision.

        Returns:
            JSON-compatible navigation decision dictionary.
        """

        return _clean_dict(
            {
                "contract_version": self.contract_version,
                "action_type": TEXT_ACTION_NAVIGATION_DECISION,
                "route_action": self.route_action,
                "intent": self.intent,
                "instruction_text": self.instruction_text,
                "tool_used": self.tool_used,
                "user_message": self.user_message,
                "finish": dict(self.finish),
            }
        )


@dataclass(frozen=True)
class System1ControlOutput:
    """System1 trajectory/control output generated from readout hidden states.

    Args:
        trajectory_norm: Optional normalized FPV trajectory.
        trajectory_grid1000: Optional text-compatible norm1000 trajectory.
        action_space: Optional action space, for example ``ego_delta_xyt``.
        action_chunk_ego: Optional continuous ego-frame action chunk.
        action_mask: Optional valid-step mask.
        runtime_commands: Optional runtime command payloads after adapter
            conversion.
        metadata: Extra deterministic replay metadata.
    """

    trajectory_norm: list[list[float]] | None = None
    trajectory_grid1000: list[list[int]] | None = None
    action_space: str | None = None
    action_chunk_ego: list[list[float]] | None = None
    action_mask: list[bool] | None = None
    runtime_commands: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_supervision(cls, supervision: dict[str, Any], finish: dict[str, Any] | None = None) -> "System1ControlOutput":
        """Build a control-output view from supervision fields.

        Args:
            supervision: Trace or row supervision dictionary.
            finish: Optional parsed finish payload.

        Returns:
            ``System1ControlOutput`` using available trajectory/action fields.
        """

        finish = finish if isinstance(finish, dict) else {}
        return cls(
            trajectory_norm=supervision.get("pixel_trajectory") or finish.get("trajectory"),
            trajectory_grid1000=finish.get("trajectory_grid1000"),
            action_space=_nested_get(supervision, ("action_stats", "space")) or _nested_get(supervision, ("trajectory_policy", "action_space")),
            action_chunk_ego=supervision.get("action_ego_t"),
            action_mask=supervision.get("action_mask"),
            metadata={"source": "supervision_view"},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the System1 output.

        Returns:
            JSON-compatible System1 output dictionary.
        """

        return _clean_dict(
            {
                "output_type": "system1_control",
                "trajectory_norm": self.trajectory_norm,
                "trajectory_grid1000": self.trajectory_grid1000,
                "action_space": self.action_space,
                "action_chunk_ego": self.action_chunk_ego,
                "action_mask": self.action_mask,
                "runtime_commands": list(self.runtime_commands),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class AgentStepResult:
    """Unified output envelope for one navigation agent step.

    Args:
        step_id: Stable step id.
        observation: Runtime-visible observation.
        tool_action: Optional System2 tool/user-interaction action.
        navigation_decision: Optional final System2 navigation decision.
        system1_output: Optional trajectory/action head output view.
        tool_trace: Tool call/response messages.
        validation: Validator result dictionary.
        artifacts: Runtime artifacts.
        replay: Replay row/trace references.
        runtime_state: Runtime state such as ``READY_FOR_CONTROL`` or
            ``WAITING_FOR_USER``.
        metadata: Extra audit metadata.
        schema_version: Step result schema version.
    """

    step_id: str
    observation: NavObservation | None = None
    tool_action: ToolAction | None = None
    navigation_decision: NavigationDecision | None = None
    system1_output: System1ControlOutput | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    replay: dict[str, Any] = field(default_factory=dict)
    runtime_state: str = RUNTIME_STATE_RUNNING
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = AGENT_STEP_RESULT_SCHEMA_VERSION

    @classmethod
    def from_trace(
        cls,
        trace: AgentTrace,
        *,
        case: AgentStage2Case | None = None,
        validation: dict[str, Any] | None = None,
        ms_swift_row: dict[str, Any] | None = None,
    ) -> "AgentStepResult":
        """Build one step-result envelope from an existing Stage2 trace.

        Args:
            trace: Replayable Stage2 trace.
            case: Optional source Stage2 case.
            validation: Optional validator result dictionary.
            ms_swift_row: Optional exported training/replay row.

        Returns:
            ``AgentStepResult`` with tool, finish, and supervision views.
        """

        case = case or _case_from_trace(trace)
        observation = NavObservation.from_case(case) if case is not None else None
        # runtime 执行工具后会把 pending user interaction 和 route context assets
        # 提升到 trace.metadata；这里再合入 step result，便于在线推理接口一次性返回。
        if observation is not None:
            dialogue_state = trace.metadata.get("dialogue_state")
            if isinstance(dialogue_state, dict) and dialogue_state:
                observation = replace(observation, dialogue_state=dict(dialogue_state))
            context_assets = trace.metadata.get("context_assets")
            if isinstance(context_assets, list) and context_assets:
                tool_context = dict(observation.tool_context)
                tool_context["context_assets"] = list(context_assets)
                observation = replace(observation, tool_context=tool_context)
        tool_action = _first_tool_action(trace)
        finish = _last_finish(trace)
        navigation_decision = NavigationDecision.from_finish(finish)
        supervision = dict(trace.supervision)
        if case is not None:
            supervision.update(case.ground_truth)
        system1 = System1ControlOutput.from_supervision(supervision, finish)
        runtime_state = str(trace.metadata.get("runtime_state") or _runtime_state_from_trace(trace, navigation_decision))
        replay = {"trace_schema": trace.schema_version}
        if ms_swift_row is not None:
            replay["ms_swift_row_schema"] = ms_swift_row.get("schema_version")
        return cls(
            step_id=trace.case_id,
            observation=observation,
            tool_action=tool_action,
            navigation_decision=navigation_decision,
            system1_output=system1,
            tool_trace=_tool_turns(trace),
            validation=dict(validation or {}),
            artifacts=list(trace.artifacts),
            replay=replay,
            runtime_state=runtime_state,
            metadata={"task_type": trace.task_type, "model_run": dict(trace.model_run)},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent step result.

        Returns:
            JSON-compatible step-result dictionary.
        """

        return _clean_dict(
            {
                "schema_version": self.schema_version,
                "step_id": self.step_id,
                "runtime_state": self.runtime_state,
                "observation": self.observation.to_dict() if self.observation else None,
                "system2": _clean_dict(
                    {
                        "tool_action": self.tool_action.to_dict() if self.tool_action else None,
                        "navigation_decision": self.navigation_decision.to_dict() if self.navigation_decision else None,
                    }
                ),
                "system1": self.system1_output.to_dict() if self.system1_output else None,
                "tool_trace": list(self.tool_trace),
                "validation": dict(self.validation),
                "artifacts": list(self.artifacts),
                "replay": dict(self.replay),
                "metadata": dict(self.metadata),
            }
        )


def _normalize_request(case: AgentStage2Case) -> dict[str, Any]:
    """Return request fields visible to the model."""

    observation = dict(case.observation or {})
    request = dict(observation.get("request") or {})
    request.setdefault("active_instruction", case.user_request)
    request.setdefault("latest_user_utterance", case.user_request)
    if case.dialogue_history:
        request.setdefault("dialogue_history", list(case.dialogue_history))
    return request


def _normalize_sensors(case: AgentStage2Case, observation: dict[str, Any]) -> dict[str, Any]:
    """Return sensor fields without supervision labels."""

    sensors = dict(observation.get("sensors") or {})
    images = observation.get("images")
    if case.images:
        sensors.setdefault("images", case.images)
    if isinstance(images, dict):
        fpv = images.get("fpv") or images.get("front")
        bev = images.get("bev")
        if fpv is not None:
            sensors.setdefault("fpv_images", [fpv])
        if bev is not None:
            sensors.setdefault("bev_images", [bev])
    elif isinstance(images, list):
        sensors.setdefault("images", list(images))
    for key in ("history_fpv_image_paths", "history_bev_image_paths", "fpv_image_paths", "bev_image_paths"):
        if observation.get(key) is not None:
            sensors.setdefault(key, observation[key])
    return sensors


def _normalize_route_state(observation: dict[str, Any]) -> dict[str, Any]:
    """Return visible route/session/pose state."""

    route_state = dict(observation.get("route_state") or {})
    for key in (
        "current_pose",
        "current_location",
        "route_session_id",
        "session_id",
        "active_goal",
        "active_route_id",
        "mode",
        "origin",
        "destination",
    ):
        if observation.get(key) is not None:
            route_state.setdefault(key, observation[key])
    return route_state


def _normalize_tool_context(case: AgentStage2Case, observation: dict[str, Any]) -> dict[str, Any]:
    """Return available tools, tool responses, and context assets."""

    tool_context = dict(observation.get("tool_context") or {})
    tool_context.setdefault("available_tools", list(case.tools))
    tool_context.setdefault("tool_responses", list(tool_context.get("tool_responses") or []))
    tool_context.setdefault("context_assets", list(tool_context.get("context_assets") or []))
    return tool_context


def _first_tool_action(trace: AgentTrace) -> ToolAction | None:
    """Extract the first tool action from a trace."""

    for turn in trace.turns:
        if turn.role != "tool_call":
            continue
        payload = _content_dict(turn.content)
        if payload:
            return ToolAction.from_dict(payload)
    return None


def _last_finish(trace: AgentTrace) -> dict[str, Any] | None:
    """Parse the last assistant finish payload from a trace."""

    for turn in reversed(trace.turns):
        if turn.role != "assistant" or not isinstance(turn.content, str):
            continue
        finish, errors = parse_finish_text(turn.content)
        if finish is not None and not errors:
            return finish
    return None


def _runtime_state_from_trace(trace: AgentTrace, decision: NavigationDecision | None) -> str:
    """Infer a coarse runtime state for one completed trace."""

    if decision and decision.route_action == "request_user_input":
        return RUNTIME_STATE_WAITING_FOR_USER
    for turn in trace.turns:
        payload = _content_dict(turn.content)
        if payload.get("tool") == "user_interaction_tool" or payload.get("name") == "user_interaction_tool":
            return RUNTIME_STATE_WAITING_FOR_USER
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        if result.get("await_user") is True:
            return RUNTIME_STATE_WAITING_FOR_USER
    return RUNTIME_STATE_READY_FOR_CONTROL if decision else RUNTIME_STATE_RUNNING


def _tool_turns(trace: AgentTrace) -> list[dict[str, Any]]:
    """Return tool-call and tool-response turns for step replay."""

    return [turn.to_dict() for turn in trace.turns if turn.role in {"tool_call", "tool_response"}]


def _case_from_trace(trace: AgentTrace) -> AgentStage2Case | None:
    """Rebuild a Stage2 case from trace request metadata when possible."""

    request = trace.request
    if isinstance(request, dict) and request.get("case_id"):
        return AgentStage2Case.from_dict(request)
    return None


def _content_dict(content: Any) -> dict[str, Any]:
    """Parse a turn content object into a dictionary."""

    if isinstance(content, dict):
        return dict(content)
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Return nested dictionary value or ``None``."""

    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
