"""Strong contracts for grounded navigation reasoning targets.

``ReasoningTraceV2`` describes one System2 decision at one observation tick.  It
is deliberately smaller than an episode trace: runtime events remain in
``EpisodeTraceV2`` and continuous System1 targets remain top-level training
fields.  The same five sections are rendered as one canonical Full CoT target.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


REASONING_TRACE_V2_SCHEMA_VERSION = "urbannav_reasoning_trace_v2"
REASONING_SECTION_NAMES = (
    "TaskGrounding",
    "EvidenceGrounding",
    "CrossModalAlignment",
    "ConstraintApplicability",
    "Decision",
)

EvidenceRef = Annotated[
    str,
    Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]*(?:[.:][A-Za-z0-9_-]+)*$",
    ),
]
ReasonCode = Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[a-z][a-z0-9_]*$")]

LocalObjective = Literal[
    "continue_route",
    "approach_goal",
    "enter_crosswalk",
    "exit_crosswalk",
    "follow_instruction",
    "stop_at_goal",
    "unknown",
]
DesiredDirection = Literal["left", "straight", "right", "stop", "unknown"]
RouteToFPV = Literal["left", "center", "right", "not_visible", "unknown"]
AlignmentStatus = Literal[
    "aligned",
    "locally_blocked",
    "conflict",
    "not_visible",
    "insufficient_evidence",
]
ConstraintCode = Literal[
    "maintain_clearance",
    "slow_for_agent",
    "yield_to_crossing_agent",
    "wait_for_gap",
    "follow_agent",
    "pass_left",
    "pass_right",
    "detour_around_agent",
    "wait_for_pedestrian_signal",
    "use_designated_crossing",
    "avoid_vehicle_lane",
]
ConstraintEffect = Literal["go", "slow", "stop", "wait", "detour"]
NotApplicableReason = Literal[
    "wrong_travel_mode",
    "path_does_not_interact",
    "path_does_not_enter_controlled_region",
    "entity_not_on_selected_corridor",
    "insufficient_evidence",
]
MotionPrimitive = Literal["go_straight", "turn_left", "turn_right", "stop"]
SpeedPolicy = Literal["normal", "slow", "stop"]
SocialAction = Literal["none", "yield", "wait", "follow", "pass_left", "pass_right", "detour", "unknown"]
FallbackAction = Literal["continue", "slow", "stop", "request_replan", "unknown"]
ConfidenceLevel = Literal["high", "medium", "low", "unknown"]
System2DecisionType = Literal["navigate", "ask_user", "complete_task", "abort_task"]


class GroundedFactV2(BaseModel):
    """One controlled fact and the input evidence that supports it.

    Args:
        fact_code: Stable lower-snake-case fact identifier.
        evidence_refs: Input-visible references supporting the fact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_code: ReasonCode
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=8)


class TaskGroundingV2(BaseModel):
    """Ground the active instruction into a local route objective.

    Args:
        goal_ref: Stable reference to the active goal when available.
        route_ref: Stable reference to the active route when available.
        local_objective: Current short-horizon task objective.
        instruction_constraints: Controlled instruction constraints that remain
            active at this tick.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_ref: EvidenceRef | None = None
    route_ref: EvidenceRef | None = None
    local_objective: LocalObjective = "unknown"
    instruction_constraints: list[ReasonCode] = Field(default_factory=list, max_length=8)


class EvidenceGroundingV2(BaseModel):
    """Record the evidence read from task, route, tools, and FPV inputs.

    Args:
        task_evidence_refs: Instruction or goal references used by the decision.
        route_evidence_refs: Route, waypoint, map, or BEV references used.
        fpv_evidence_refs: FPV corridor, track, control, or obstacle references.
        tool_evidence_refs: Tool-response references consumed after a tool call.
        relevant_facts: Compact controlled facts supported by those references.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    route_evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    fpv_evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=16)
    tool_evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    relevant_facts: list[GroundedFactV2] = Field(default_factory=list, max_length=16)


class CrossModalAlignmentV2(BaseModel):
    """Align route intent with a currently visible FPV corridor.

    Args:
        available: Whether both route intent and local visual evidence exist.
        desired_direction: Direction implied by instruction and route evidence.
        route_to_fpv: Coarse FPV region corresponding to the active route.
        selected_corridor_ref: Selected input-visible corridor reference.
        status: Alignment, blockage, conflict, visibility, or evidence status.
        evidence_refs: References used to establish the alignment relation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    desired_direction: DesiredDirection = "unknown"
    route_to_fpv: RouteToFPV = "unknown"
    selected_corridor_ref: EvidenceRef | None = None
    status: AlignmentStatus = "insufficient_evidence"
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_availability(self) -> "CrossModalAlignmentV2":
        """Validate that unavailable alignment does not claim a selected corridor.

        Returns:
            The validated section.

        Raises:
            ValueError: If an unavailable section claims an aligned corridor.
        """

        if not self.available and self.status in {"aligned", "locally_blocked", "conflict"}:
            raise ValueError("unavailable alignment cannot claim an executable alignment status")
        if self.status == "not_visible" and self.selected_corridor_ref is not None:
            raise ValueError("not_visible alignment cannot select an FPV corridor")
        return self


class ApplicableConstraintV2(BaseModel):
    """One social or traffic constraint that applies to the intended path.

    Args:
        constraint: Controlled constraint identifier.
        trigger_refs: Evidence and path references that activate the constraint.
        effect: Required local behavior effect.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint: ConstraintCode
    trigger_refs: list[EvidenceRef] = Field(min_length=1, max_length=8)
    effect: ConstraintEffect


class RejectedConstraintV2(BaseModel):
    """One observed but inapplicable constraint for hard negative supervision.

    Args:
        constraint: Controlled constraint identifier considered by the teacher.
        trigger_refs: Observed entities or controls considered for applicability.
        reason: Why the constraint does not affect the intended path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint: ConstraintCode
    trigger_refs: list[EvidenceRef] = Field(default_factory=list, max_length=8)
    reason: NotApplicableReason


class ConstraintApplicabilityV2(BaseModel):
    """Separate observed constraints from constraints applicable to the path.

    Args:
        applicable: Constraints supported by path interaction and travel mode.
        not_applicable: Small hard-negative set, not an exhaustive rule list.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: list[ApplicableConstraintV2] = Field(default_factory=list, max_length=6)
    not_applicable: list[RejectedConstraintV2] = Field(default_factory=list, max_length=3)


class DecisionV2(BaseModel):
    """Represent the evidence-grounded symbolic local navigation policy.

    Args:
        motion: Coarse local motion primitive.
        speed: Symbolic speed policy.
        social_action: Social interaction policy.
        selected_corridor_ref: Input-visible corridor selected for execution.
        fallback: Behavior if the selected policy becomes invalid.
        confidence: Calibrated decision confidence.
        reason_codes: Controlled reasons linked to grounded facts or constraints.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion: MotionPrimitive
    speed: SpeedPolicy
    social_action: SocialAction
    selected_corridor_ref: EvidenceRef | None = None
    fallback: FallbackAction
    confidence: ConfidenceLevel
    reason_codes: list[ReasonCode] = Field(default_factory=list, max_length=8)


class ReasoningTraceV2(BaseModel):
    """Canonical five-section reasoning target for one navigation decision.

    Args:
        schema_version: Fixed V2 contract version.
        task_grounding: Active task and instruction grounding.
        evidence_grounding: Evidence references and controlled facts.
        cross_modal_alignment: Route-to-FPV alignment relation.
        constraint_applicability: Applicable and rejected local constraints.
        decision: Final symbolic local policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_TRACE_V2_SCHEMA_VERSION] = REASONING_TRACE_V2_SCHEMA_VERSION
    task_grounding: TaskGroundingV2
    evidence_grounding: EvidenceGroundingV2
    cross_modal_alignment: CrossModalAlignmentV2
    constraint_applicability: ConstraintApplicabilityV2
    decision: DecisionV2


class UserRequestV2(BaseModel):
    """Structured request emitted by an ``ask_user`` decision.

    Args:
        question: Concise question shown to the user.
        choices: Optional bounded choices.
        motion_policy_while_waiting: Deterministic waiting behavior.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1, max_length=512)
    choices: list[str] = Field(default_factory=list, max_length=8)
    motion_policy_while_waiting: Literal["stop", "slow_down", "handover_to_safety_controller"] = "stop"


class FinishV2(BaseModel):
    """Executable System2 turn contract following ``ReasoningTraceV2``.

    Args:
        schema_version: Fixed V2 finish version.
        decision_type: Runtime control-flow decision.
        motion: Navigate-only motion copied from ``DecisionV2``.
        speed: Navigate-only speed copied from ``DecisionV2``.
        social_action: Navigate-only social policy copied from ``DecisionV2``.
        confidence_level: Navigate-only calibrated confidence.
        trajectory_type: Optional symbolic trajectory class such as stop_hold.
        route_action: Optional route-session operation summary.
        tool_used: Whether this decision consumed a tool response.
        reason_code: Required for task completion or abortion.
        reason: Human-auditable terminal reason.
        evidence_refs: Evidence supporting a terminal request.
        user_request: Structured question for ``ask_user``.
        retryable: Whether an aborted task may be retried.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["urbannav_finish_v2"] = "urbannav_finish_v2"
    decision_type: System2DecisionType
    motion: MotionPrimitive | None = None
    speed: SpeedPolicy | None = None
    social_action: SocialAction | None = None
    confidence_level: ConfidenceLevel | None = None
    trajectory_type: str | None = Field(default=None, min_length=1, max_length=64)
    route_action: str | None = Field(default=None, min_length=1, max_length=96)
    tool_used: bool | None = None
    reason_code: ReasonCode | None = None
    reason: str | None = Field(default=None, max_length=512)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    user_request: UserRequestV2 | None = None
    retryable: bool | None = None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "FinishV2":
        """Validate decision-specific finish fields.

        Returns:
            The validated finish payload.

        Raises:
            ValueError: If required fields are missing or appear on an invalid
                decision type.
        """

        navigation_fields = (self.motion, self.speed, self.social_action, self.confidence_level)
        if self.decision_type == "navigate":
            if any(value is None for value in navigation_fields):
                raise ValueError("navigate finish requires motion, speed, social_action, and confidence_level")
            if self.user_request is not None or self.reason_code is not None:
                raise ValueError("navigate finish cannot contain interaction or terminal fields")
        elif any(value is not None for value in navigation_fields) or self.trajectory_type is not None:
            raise ValueError("non-navigation finish cannot contain navigation fields")

        if self.decision_type == "ask_user" and self.user_request is None:
            raise ValueError("ask_user finish requires user_request")
        if self.decision_type != "ask_user" and self.user_request is not None:
            raise ValueError("only ask_user finish may contain user_request")
        if self.decision_type in {"complete_task", "abort_task"} and self.reason_code is None:
            raise ValueError("terminal finish requires reason_code")
        if self.decision_type not in {"complete_task", "abort_task"} and self.reason_code is not None:
            raise ValueError("only terminal finish may contain reason_code")
        return self


class ReasoningTargetV2(BaseModel):
    """Pair one reasoning trace with its executable finish projection.

    Args:
        reasoning: Canonical five-section reasoning target.
        finish: Runtime control-flow and discrete navigation payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: ReasoningTraceV2
    finish: FinishV2

    @model_validator(mode="after")
    def validate_navigation_consistency(self) -> "ReasoningTargetV2":
        """Require navigate finish fields to agree with the Decision section.

        Returns:
            The validated target.

        Raises:
            ValueError: If duplicated execution fields disagree.
        """

        if self.finish.decision_type != "navigate":
            return self
        decision = self.reasoning.decision
        pairs = {
            "motion": (decision.motion, self.finish.motion),
            "speed": (decision.speed, self.finish.speed),
            "social_action": (decision.social_action, self.finish.social_action),
            "confidence": (decision.confidence, self.finish.confidence_level),
        }
        mismatched = [name for name, (left, right) in pairs.items() if left != right]
        if mismatched:
            raise ValueError(f"Decision/finish mismatch: {','.join(mismatched)}")
        return self


def reasoning_evidence_refs(trace: ReasoningTraceV2) -> set[str]:
    """Return every evidence reference cited by a reasoning trace.

    Args:
        trace: Canonical reasoning trace.

    Returns:
        Unique evidence-reference strings used by all five sections.
    """

    refs: set[str] = set()
    task = trace.task_grounding
    refs.update(ref for ref in (task.goal_ref, task.route_ref) if ref)
    evidence = trace.evidence_grounding
    refs.update(evidence.task_evidence_refs)
    refs.update(evidence.route_evidence_refs)
    refs.update(evidence.fpv_evidence_refs)
    refs.update(evidence.tool_evidence_refs)
    for fact in evidence.relevant_facts:
        refs.update(fact.evidence_refs)
    alignment = trace.cross_modal_alignment
    if alignment.selected_corridor_ref:
        refs.add(alignment.selected_corridor_ref)
    refs.update(alignment.evidence_refs)
    for item in trace.constraint_applicability.applicable:
        refs.update(item.trigger_refs)
    for item in trace.constraint_applicability.not_applicable:
        refs.update(item.trigger_refs)
    if trace.decision.selected_corridor_ref:
        refs.add(trace.decision.selected_corridor_ref)
    return refs


def reasoning_json_schema() -> dict[str, Any]:
    """Return the generated JSON Schema for cross-repository conformance tests.

    Returns:
        JSON-compatible Pydantic schema for ``ReasoningTargetV2``.
    """

    return ReasoningTargetV2.model_json_schema()
