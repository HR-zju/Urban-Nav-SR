"""Render, parse, and adapt grounded reasoning assistant text.

The text grammar is intentionally deterministic.  New rows always serialize the
five ``ReasoningTraceV2`` sections, while the adapter accepts historical
``FPVScene/BEVContext/CrossView/Plan`` targets without preserving their weaker
free-text geometry fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import ValidationError

from mcp_server.contracts.reasoning import (
    REASONING_SECTION_NAMES,
    ApplicableConstraintV2,
    ConstraintApplicabilityV2,
    CrossModalAlignmentV2,
    DecisionV2,
    EvidenceGroundingV2,
    FinishV2,
    GroundedFactV2,
    ReasoningTargetV2,
    ReasoningTraceV2,
    RejectedConstraintV2,
    TaskGroundingV2,
    UserRequestV2,
)


_THINK_RE = re.compile(r"<think>\s*(?P<body>.*?)\s*</think>", re.DOTALL)
_TAG_RE = re.compile(r"<(?P<tag>[A-Za-z][A-Za-z0-9]*)>\s*(?P<body>\{.*?\})\s*</(?P=tag)>", re.DOTALL)
_READOUT_RE = re.compile(r"<readout>\s*(?P<body>.*?)\s*</readout>", re.DOTALL)
_V1_SECTION_NAMES = ("FPVScene", "BEVContext", "CrossView", "Plan")


@dataclass(frozen=True)
class ReasoningTextParseResult:
    """Result of parsing either canonical V2 or compatible V1 reasoning text.

    Args:
        target: Parsed canonical target, or ``None`` after hard errors.
        errors: Machine-readable parsing and validation errors.
        warnings: Non-blocking compatibility warnings.
        adapted_from_v1: Whether the result came from the legacy adapter.
        tool_result_use: Optional parsed agentic tool-use extension.
    """

    target: ReasoningTargetV2 | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    adapted_from_v1: bool = False
    tool_result_use: dict[str, Any] | None = None

    @property
    def valid(self) -> bool:
        """Return whether parsing produced a target without hard errors.

        Returns:
            ``True`` when ``target`` exists and ``errors`` is empty.
        """

        return self.target is not None and not self.errors


def render_reasoning_target_v2(
    target: ReasoningTargetV2,
    *,
    readout_tokens: Iterable[str] = (),
    tool_result_use: dict[str, Any] | None = None,
) -> str:
    """Render one canonical full CoT target.

    Args:
        target: Validated reasoning and finish payload.
        readout_tokens: Optional System1 tokens appended only for navigation.
        tool_result_use: Optional compact Stage2 tool-result-use extension.

    Returns:
        Deterministic XML-like text with compact JSON section payloads.

    Raises:
        ValueError: If readout tokens are supplied for a non-navigation turn.
    """

    tokens = [str(token) for token in readout_tokens if str(token)]
    if tokens and target.finish.decision_type != "navigate":
        raise ValueError("System1 readout is legal only for decision_type=navigate")

    reasoning = target.reasoning
    sections = [
        _render_tag("TaskGrounding", reasoning.task_grounding.model_dump(mode="json", exclude_none=True)),
        _render_tag("EvidenceGrounding", reasoning.evidence_grounding.model_dump(mode="json", exclude_none=True)),
    ]
    if tool_result_use is not None:
        sections.append(_render_tag("ToolResultUse", tool_result_use))
    sections.extend(
        [
            _render_tag(
                "CrossModalAlignment",
                reasoning.cross_modal_alignment.model_dump(mode="json", exclude_none=True),
            ),
            _render_tag(
                "ConstraintApplicability",
                reasoning.constraint_applicability.model_dump(mode="json", exclude_none=True),
            ),
            _render_tag("Decision", reasoning.decision.model_dump(mode="json", exclude_none=True)),
        ]
    )
    text = "<think>\n" + "\n".join(sections) + "\n</think>\n"
    text += _render_tag("finish", target.finish.model_dump(mode="json", exclude_none=True))
    if tokens:
        text += "\n<readout>" + " ".join(tokens) + "</readout>"
    return text


def parse_reasoning_target_v2(text: str) -> ReasoningTextParseResult:
    """Parse strict V2 full CoT text.

    Args:
        text: Assistant text containing five sections and one finish payload.

    Returns:
        Parse result with a strongly validated target or detailed errors.
    """

    errors: list[str] = []
    think_match = _THINK_RE.search(text)
    if think_match is None:
        return ReasoningTextParseResult(target=None, errors=["missing_think"])
    raw_sections = list(_TAG_RE.finditer(think_match.group("body")))
    tag_names = [match.group("tag") for match in raw_sections]
    names = [match.group("tag") for match in raw_sections if match.group("tag") != "ToolResultUse"]
    if names != list(REASONING_SECTION_NAMES):
        errors.append(f"reasoning_section_order:{names}")
    tool_matches = [match for match in raw_sections if match.group("tag") == "ToolResultUse"]
    if len(tool_matches) > 1:
        errors.append("duplicate_tool_result_use")
    expected_with_tool = [
        "TaskGrounding",
        "EvidenceGrounding",
        "ToolResultUse",
        "CrossModalAlignment",
        "ConstraintApplicability",
        "Decision",
    ]
    if tool_matches and tag_names != expected_with_tool:
        errors.append(f"tool_result_use_position:{tag_names}")

    # 匹配后的空白必须为空，避免 <think> 中夹带未审计自然语言。
    stripped_body = _TAG_RE.sub("", think_match.group("body")).strip()
    if stripped_body:
        errors.append("unexpected_text_inside_think")

    payloads: dict[str, dict[str, Any]] = {}
    tool_result_use: dict[str, Any] | None = None
    for match in raw_sections:
        payload, payload_errors = _parse_json_object(match.group("body"))
        errors.extend(f"{match.group('tag')}:{item}" for item in payload_errors)
        if payload is None:
            continue
        if match.group("tag") == "ToolResultUse":
            tool_result_use = payload
        else:
            payloads[match.group("tag")] = payload

    finish_payload, finish_errors = _tag_payload(text, "finish")
    errors.extend(f"finish:{item}" for item in finish_errors)
    if errors:
        return ReasoningTextParseResult(target=None, errors=errors, tool_result_use=tool_result_use)
    try:
        trace = ReasoningTraceV2(
            task_grounding=TaskGroundingV2.model_validate(payloads["TaskGrounding"]),
            evidence_grounding=EvidenceGroundingV2.model_validate(payloads["EvidenceGrounding"]),
            cross_modal_alignment=CrossModalAlignmentV2.model_validate(payloads["CrossModalAlignment"]),
            constraint_applicability=ConstraintApplicabilityV2.model_validate(payloads["ConstraintApplicability"]),
            decision=DecisionV2.model_validate(payloads["Decision"]),
        )
        target = ReasoningTargetV2(reasoning=trace, finish=FinishV2.model_validate(finish_payload))
    except (KeyError, ValidationError, ValueError) as exc:
        return ReasoningTextParseResult(
            target=None,
            errors=[f"reasoning_schema:{_validation_error_text(exc)}"],
            tool_result_use=tool_result_use,
        )
    return ReasoningTextParseResult(target=target, tool_result_use=tool_result_use)


def parse_or_adapt_reasoning_text(text: str) -> ReasoningTextParseResult:
    """Parse V2 text or convert one historical V1 CoT target.

    Args:
        text: Assistant text using V2 or historical SideWalk tags.

    Returns:
        Canonical parse result.  V1 input is marked through
        ``adapted_from_v1`` and emits a compatibility warning.
    """

    if "<TaskGrounding>" in text:
        return parse_reasoning_target_v2(text)
    return adapt_v1_reasoning_text(text)


def adapt_v1_reasoning_text(text: str) -> ReasoningTextParseResult:
    """Convert historical modality-oriented CoT into ``ReasoningTargetV2``.

    Args:
        text: V1 assistant text containing FPV/BEV/CrossView/Plan/finish tags.

    Returns:
        Canonical target using only facts supported by legacy fields.  The
        adapter intentionally does not promote free-text V1 policies into
        strong social-constraint labels.
    """

    payloads: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for tag in _V1_SECTION_NAMES:
        payload, tag_errors = _tag_payload(text, tag)
        errors.extend(f"{tag}:{item}" for item in tag_errors)
        if payload is not None:
            payloads[tag] = payload
    finish_payload, finish_errors = _tag_payload(text, "finish")
    errors.extend(f"finish:{item}" for item in finish_errors)
    if errors:
        return ReasoningTextParseResult(target=None, errors=errors, adapted_from_v1=True)

    fpv = payloads["FPVScene"]
    bev = payloads["BEVContext"]
    cross = payloads["CrossView"]
    plan = payloads["Plan"]
    finish_payload = finish_payload or {}

    motion = _normalize_motion(plan.get("motion_primitive") or finish_payload.get("intent") or bev.get("intent"))
    speed = _normalize_speed(plan.get("speed_policy"), motion=motion)
    social_action = _normalize_social_action(plan.get("social_action"), speed=speed)
    confidence = _normalize_confidence(finish_payload.get("confidence_level") or cross.get("confidence"))
    corridor_ref = _corridor_ref(cross.get("safe_corridor") or cross.get("fpv_route_region"))
    route_available = bool(bev.get("available", True)) and str(bev.get("scene_summary") or "") != "not_available"

    task_refs = ["instruction.active"]
    route_refs = ["route.active"] if route_available else []
    fpv_refs = ["fpv.current"]
    if corridor_ref:
        fpv_refs.append(corridor_ref)
    facts = _legacy_facts(bev=bev, fpv=fpv, motion=motion, corridor_ref=corridor_ref)
    status = _normalize_alignment_status(cross.get("alignment"), blocked_by=fpv.get("blocked_by"))
    alignment_available = route_available and status != "insufficient_evidence"
    if not alignment_available and status in {"aligned", "locally_blocked", "conflict"}:
        status = "insufficient_evidence"

    constraints = _legacy_constraints(fpv=fpv, plan=plan, corridor_ref=corridor_ref)
    trace = ReasoningTraceV2(
        task_grounding=TaskGroundingV2(
            goal_ref="goal.active" if finish_payload.get("route_action") in {"create_route", "update_route"} else None,
            route_ref="route.active" if route_available else None,
            local_objective=_local_objective(bev=bev, finish=finish_payload),
            instruction_constraints=[],
        ),
        evidence_grounding=EvidenceGroundingV2(
            task_evidence_refs=task_refs,
            route_evidence_refs=route_refs,
            fpv_evidence_refs=fpv_refs,
            tool_evidence_refs=["tool.latest"] if finish_payload.get("tool_used") else [],
            relevant_facts=facts,
        ),
        cross_modal_alignment=CrossModalAlignmentV2(
            available=alignment_available,
            desired_direction=_desired_direction(bev.get("intent") or motion),
            route_to_fpv=_route_to_fpv(cross.get("route_to_fpv_direction") or cross.get("fpv_route_region")),
            selected_corridor_ref=corridor_ref if status != "not_visible" else None,
            status=status,
            evidence_refs=[*route_refs, *fpv_refs] if alignment_available else [],
        ),
        constraint_applicability=constraints,
        decision=DecisionV2(
            motion=motion,
            speed=speed,
            social_action=social_action,
            selected_corridor_ref=corridor_ref,
            fallback=_normalize_fallback(plan.get("contingency")),
            confidence=confidence,
            reason_codes=_legacy_reason_codes(fpv=fpv, status=status),
        ),
    )
    try:
        finish = _adapt_v1_finish(
            finish_payload,
            motion=motion,
            speed=speed,
            social_action=social_action,
            confidence=confidence,
        )
        target = ReasoningTargetV2(reasoning=trace, finish=finish)
    except (ValidationError, ValueError) as exc:
        return ReasoningTextParseResult(
            target=None,
            errors=[f"v1_adapter:{_validation_error_text(exc)}"],
            adapted_from_v1=True,
        )
    tool_result_use, _ = _tag_payload(text, "ToolResultUse", required=False)
    return ReasoningTextParseResult(
        target=target,
        warnings=["legacy_reasoning_adapted_to_v2"],
        adapted_from_v1=True,
        tool_result_use=tool_result_use,
    )


def has_system1_readout(text: str) -> bool:
    """Return whether assistant text contains a non-empty readout block.

    Args:
        text: Assistant output.

    Returns:
        ``True`` when ``<readout>`` contains at least one SYS1/TRAJ token.
    """

    match = _READOUT_RE.search(text)
    if match is None:
        return False
    return re.search(r"\[(?:SYS1|TRAJ)_\d+\]", match.group("body")) is not None


def _render_tag(tag: str, payload: dict[str, Any]) -> str:
    """Render one compact JSON tag."""

    return f"<{tag}>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</{tag}>"


def _tag_payload(text: str, tag: str, *, required: bool = True) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract one JSON object from a named tag."""

    matches = re.findall(rf"<{re.escape(tag)}>\s*(\{{.*?\}})\s*</{re.escape(tag)}>", text, flags=re.DOTALL)
    if not matches:
        return (None, ["missing_tag"] if required else [])
    if len(matches) > 1:
        return None, ["duplicate_tag"]
    return _parse_json_object(matches[0])


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse one strict JSON object."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"json_decode:{exc.msg}"]
    if not isinstance(payload, dict):
        return None, ["json_not_object"]
    return payload, []


def _normalize_motion(value: Any) -> str:
    """Map historical intent aliases into V2 motion primitives."""

    raw = str(value or "unknown").strip().lower()
    aliases = {
        "straight": "go_straight",
        "continue": "go_straight",
        "continue_navigation": "go_straight",
        "left": "turn_left",
        "right": "turn_right",
        "hold": "stop",
        "wait": "stop",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in {"go_straight", "turn_left", "turn_right", "stop"} else "stop"


def _normalize_speed(value: Any, *, motion: str) -> str:
    """Map historical speed aliases into V2 speed policy."""

    raw = str(value or "").strip().lower()
    if motion == "stop" or raw in {"stop", "hold", "zero"}:
        return "stop"
    if raw in {"slow", "slow_down", "cautious"}:
        return "slow"
    return "normal"


def _normalize_social_action(value: Any, *, speed: str) -> str:
    """Map V1 social labels without inferring unsupported interactions."""

    raw = str(value or "").strip().lower()
    aliases = {"go": "none", "slow": "none", "stop": "wait"}
    normalized = aliases.get(raw, raw)
    allowed = {"none", "yield", "wait", "follow", "pass_left", "pass_right", "detour", "unknown"}
    if normalized in allowed:
        return normalized
    return "wait" if speed == "stop" else "unknown"


def _normalize_confidence(value: Any) -> str:
    """Map legacy confidence/status labels into calibrated V2 confidence."""

    raw = str(value or "").strip().lower()
    return {
        "normal": "high",
        "conflict": "low",
        "high": "high",
        "medium": "medium",
        "low": "low",
    }.get(raw, "unknown")


def _corridor_ref(value: Any) -> str | None:
    """Return a stable corridor reference from a V1 region label."""

    raw = str(value or "").strip().lower()
    if "left" in raw:
        return "corridor.left"
    if "right" in raw:
        return "corridor.right"
    if "center" in raw or "front" in raw:
        return "corridor.center"
    return None


def _route_to_fpv(value: Any) -> str:
    """Normalize legacy route-to-FPV direction labels."""

    raw = str(value or "").strip().lower()
    if "left" in raw:
        return "left"
    if "right" in raw:
        return "right"
    if "center" in raw or "forward" in raw or "front" in raw:
        return "center"
    if "not_visible" in raw:
        return "not_visible"
    return "unknown"


def _desired_direction(value: Any) -> str:
    """Normalize route intent into the V2 desired-direction enum."""

    raw = str(value or "").strip().lower()
    if "left" in raw:
        return "left"
    if "right" in raw:
        return "right"
    if raw in {"stop", "hold", "arrive", "stop_at_goal"}:
        return "stop"
    if raw in {"go_straight", "straight", "continue", "forward"}:
        return "straight"
    return "unknown"


def _normalize_alignment_status(value: Any, *, blocked_by: Any) -> str:
    """Map legacy alignment plus visible blockage into V2 status."""

    blockers = [str(item).lower() for item in blocked_by or []] if isinstance(blocked_by, list) else []
    blockers = [item for item in blockers if item not in {"none", "unknown"}]
    raw = str(value or "").strip().lower()
    if blockers and raw in {"aligned", "partially_aligned", "weakly_aligned"}:
        return "locally_blocked"
    if raw == "aligned":
        return "aligned"
    if raw in {"conflict", "unaligned"}:
        return "conflict"
    if raw == "not_visible":
        return "not_visible"
    return "insufficient_evidence"


def _normalize_fallback(value: Any) -> str:
    """Reduce free-text V1 contingency into a bounded fallback action."""

    raw = str(value or "").lower()
    if "replan" in raw:
        return "request_replan"
    if "stop" in raw or "hold" in raw:
        return "stop"
    if "slow" in raw or "yield" in raw:
        return "slow"
    if "continue" in raw:
        return "continue"
    return "unknown"


def _local_objective(*, bev: dict[str, Any], finish: dict[str, Any]) -> str:
    """Infer a conservative local objective from legacy explicit fields."""

    text = " ".join(str(value or "").lower() for value in (bev.get("intent"), bev.get("route_summary")))
    decision_type = str(finish.get("decision_type") or "").lower()
    if decision_type == "complete_task" or "arriv" in text:
        return "stop_at_goal"
    if "cross" in text:
        return "enter_crosswalk"
    if finish.get("route_action") in {"create_route", "update_route", "replan"}:
        return "follow_instruction"
    return "continue_route" if bev else "unknown"


def _legacy_facts(
    *,
    bev: dict[str, Any],
    fpv: dict[str, Any],
    motion: str,
    corridor_ref: str | None,
) -> list[GroundedFactV2]:
    """Build only bounded facts directly supported by V1 fields."""

    facts = [GroundedFactV2(fact_code=f"route_{motion}", evidence_refs=["route.active"])] if bev else []
    blockers = [str(item).lower() for item in fpv.get("blocked_by") or []]
    if any(item not in {"none", "unknown"} for item in blockers):
        facts.append(
            GroundedFactV2(
                fact_code="selected_corridor_blocked",
                evidence_refs=[ref for ref in ("fpv.current", corridor_ref) if ref],
            )
        )
    return facts


def _legacy_constraints(
    *,
    fpv: dict[str, Any],
    plan: dict[str, Any],
    corridor_ref: str | None,
) -> ConstraintApplicabilityV2:
    """Promote only V1 constraints with explicit path-interaction evidence.

    Historical ``blocked_by=pedestrian`` labels do not contain a track id or
    crossing relation.  Treating them as ``yield_to_crossing_agent`` would turn
    a weak scene tag into a strong temporal social label, so the adapter keeps
    those cases out of ``applicable``.
    """

    triggers = [ref for ref in ("fpv.current", corridor_ref) if ref]
    applicable: list[ApplicableConstraintV2] = []
    risk_names = {
        str(item.get("name") or "").lower()
        for item in fpv.get("risk_objects") or []
        if isinstance(item, dict)
    }
    if "vehicle_lane" in risk_names:
        applicable.append(ApplicableConstraintV2(constraint="avoid_vehicle_lane", trigger_refs=triggers, effect="detour"))
    return ConstraintApplicabilityV2(applicable=applicable, not_applicable=[])


def _legacy_reason_codes(*, fpv: dict[str, Any], status: str) -> list[str]:
    """Map bounded V1 risk fields into controlled decision reasons."""

    reasons: list[str] = []
    if status == "locally_blocked":
        reasons.append("selected_corridor_blocked")
    elif status == "conflict":
        reasons.append("route_fpv_conflict")
    risk = str(fpv.get("local_risk") or "").lower()
    if risk in {"medium", "high"}:
        reasons.append(f"local_risk_{risk}")
    return reasons


def _adapt_v1_finish(
    payload: dict[str, Any],
    *,
    motion: str,
    speed: str,
    social_action: str,
    confidence: str,
) -> FinishV2:
    """Normalize one historical finish payload without retaining text trajectory."""

    raw_decision = str(payload.get("decision_type") or "").strip().lower()
    route_action = str(payload.get("route_action") or "").strip().lower()
    if not raw_decision:
        if route_action in {"finish", "complete_task"}:
            raw_decision = "complete_task"
        elif route_action in {"request_user_input", "ask_user"}:
            raw_decision = "ask_user"
        else:
            raw_decision = "navigate"
    raw_decision = {"finish": "complete_task", "stop": "abort_task"}.get(raw_decision, raw_decision)

    base: dict[str, Any] = {
        "decision_type": raw_decision,
        "route_action": route_action or None,
        "tool_used": payload.get("tool_used"),
    }
    if raw_decision == "navigate":
        base.update(
            {
                "motion": motion,
                "speed": speed,
                "social_action": social_action,
                "confidence_level": confidence,
                "trajectory_type": payload.get("trajectory_type") or ("stop_hold" if motion == "stop" else "unknown"),
            }
        )
    elif raw_decision == "ask_user":
        raw_request = payload.get("user_request") if isinstance(payload.get("user_request"), dict) else {}
        base["user_request"] = UserRequestV2(
            question=str(raw_request.get("question") or payload.get("reason") or "Please clarify the navigation request."),
            choices=[str(item) for item in raw_request.get("choices") or []],
            motion_policy_while_waiting=str(raw_request.get("motion_policy_while_waiting") or "stop"),
        )
    elif raw_decision in {"complete_task", "abort_task"}:
        base.update(
            {
                "reason_code": payload.get("reason_code")
                or ("arrival_confirmed" if raw_decision == "complete_task" else "unknown_failure"),
                "reason": payload.get("reason"),
                "evidence_refs": payload.get("evidence_refs") or [],
                "retryable": payload.get("retryable"),
            }
        )
    return FinishV2.model_validate(base)


def _validation_error_text(exc: Exception) -> str:
    """Return compact deterministic text for schema exceptions."""

    if isinstance(exc, ValidationError):
        return ";".join(
            f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}" for item in exc.errors()
        )
    return str(exc)
