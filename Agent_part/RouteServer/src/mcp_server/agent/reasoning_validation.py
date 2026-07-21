"""Deterministic validators for grounded reasoning targets and QA projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from mcp_server.agent.reasoning_text import has_system1_readout
from mcp_server.contracts.reasoning import ReasoningTargetV2, ReasoningTraceV2, reasoning_evidence_refs


@dataclass(frozen=True)
class ReasoningValidationResult:
    """Validation result for one V2 reasoning target.

    Args:
        valid: Whether no hard errors were found.
        errors: Machine-readable contract or grounding failures.
        warnings: Review signals that do not block export.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_reasoning_target_v2(
    target: ReasoningTargetV2,
    *,
    evidence_catalog: Iterable[Any] = (),
    available_modalities: Iterable[str] | None = None,
    travel_mode: str | None = None,
    assistant_text: str | None = None,
    require_navigation_readout: bool = False,
    tool_result_use: dict[str, Any] | None = None,
) -> ReasoningValidationResult:
    """Validate references, applicability, and finish/readout legality.

    Reference resolution uses only input-visible catalog entries.  Label source,
    teacher confidence, and future trajectory supervision must not be added to
    this catalog.

    Args:
        target: Strongly parsed reasoning and finish target.
        evidence_catalog: Strings or objects containing ``ref``/``id`` fields.
        available_modalities: Optional input-visible modality names.
        travel_mode: Optional active travel mode for traffic-rule checks.
        assistant_text: Optional rendered assistant output.
        require_navigation_readout: Whether navigate text must carry readout.
        tool_result_use: Optional parsed tool-use summary from the same turn.

    Returns:
        Deterministic validation result.
    """

    errors: list[str] = []
    warnings: list[str] = []
    available_refs = normalize_evidence_catalog(evidence_catalog)
    cited_refs = reasoning_evidence_refs(target.reasoning) | set(target.finish.evidence_refs)
    if available_refs:
        unresolved = sorted(cited_refs - available_refs)
        errors.extend(f"unresolved_evidence_ref:{ref}" for ref in unresolved)
    elif cited_refs:
        warnings.append("evidence_catalog_missing")

    _validate_alignment(
        target.reasoning,
        available_modalities=available_modalities,
        errors=errors,
    )
    _validate_constraints(target.reasoning, travel_mode=travel_mode, errors=errors, warnings=warnings)
    _validate_decision(target.reasoning, errors=errors, warnings=warnings)

    # tool_used、ToolResultUse 与 EvidenceGrounding 必须描述同一条因果链。
    if tool_result_use is not None:
        if target.finish.tool_used is not True:
            errors.append("tool_result_use_requires_finish_tool_used")
        if not target.reasoning.evidence_grounding.tool_evidence_refs:
            errors.append("tool_result_use_requires_tool_evidence_refs")
    elif target.finish.tool_used is True:
        errors.append("finish_tool_used_missing_tool_result_use")
    elif target.reasoning.evidence_grounding.tool_evidence_refs:
        errors.append("tool_evidence_refs_without_tool_use")

    if assistant_text is not None:
        has_readout = has_system1_readout(assistant_text)
        if target.finish.decision_type == "navigate" and require_navigation_readout and not has_readout:
            errors.append("navigate_missing_system1_readout")
        if target.finish.decision_type != "navigate" and has_readout:
            errors.append("non_navigation_has_system1_readout")
    if target.finish.trajectory_type == "stop_hold" and target.finish.decision_type != "navigate":
        errors.append("stop_hold_requires_navigation")
    if target.finish.decision_type == "navigate" and target.finish.motion == "stop":
        if target.finish.speed != "stop":
            errors.append("stop_motion_requires_stop_speed")
        if target.finish.trajectory_type != "stop_hold":
            errors.append("stop_motion_requires_stop_hold")
    return ReasoningValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_reasoning_section_refs(
    trace: ReasoningTraceV2,
    *,
    evidence_catalog: Iterable[Any],
) -> ReasoningValidationResult:
    """Validate evidence references for a full trace or one projected QA source.

    Args:
        trace: Canonical reasoning trace.
        evidence_catalog: Input-visible evidence entries.

    Returns:
        Result containing unresolved-reference errors.
    """

    available_refs = normalize_evidence_catalog(evidence_catalog)
    unresolved = sorted(reasoning_evidence_refs(trace) - available_refs)
    errors = [f"unresolved_evidence_ref:{ref}" for ref in unresolved]
    return ReasoningValidationResult(valid=not errors, errors=errors)


def normalize_evidence_catalog(entries: Iterable[Any]) -> set[str]:
    """Normalize evidence catalog strings and objects into reference ids.

    Args:
        entries: Strings or dictionaries with ``ref``, ``evidence_ref``,
            ``asset_id``, or ``id``.

    Returns:
        Unique input-visible reference strings.
    """

    refs: set[str] = set()
    for entry in entries or ():
        if isinstance(entry, str) and entry:
            refs.add(entry)
            continue
        if not isinstance(entry, dict):
            continue
        value = entry.get("ref") or entry.get("evidence_ref") or entry.get("asset_id") or entry.get("id")
        if value:
            refs.add(str(value))
    return refs


def _validate_alignment(
    trace: ReasoningTraceV2,
    *,
    available_modalities: Iterable[str] | None,
    errors: list[str],
) -> None:
    """Validate route-to-corridor consistency and modality availability."""

    alignment = trace.cross_modal_alignment
    decision = trace.decision
    if available_modalities is not None:
        modalities = {str(item) for item in available_modalities}
        has_bev = "bev" in modalities
        if not has_bev and alignment.available:
            errors.append("alignment_available_without_bev")
        if not has_bev and (
            alignment.selected_corridor_ref is not None
            or alignment.evidence_refs
            or alignment.status not in {"not_visible", "insufficient_evidence"}
        ):
            errors.append("alignment_content_without_bev")
        if has_bev and alignment.available:
            refs = set(alignment.evidence_refs)
            if "bev.current" not in refs:
                errors.append("alignment_missing_bev_ref")
            if not any(ref == "fpv.current" or ref.startswith("corridor.") for ref in refs):
                errors.append("alignment_missing_fpv_ref")
    if alignment.status in {"aligned", "locally_blocked", "conflict"} and not alignment.evidence_refs:
        errors.append("alignment_missing_evidence_refs")
    if (
        alignment.selected_corridor_ref
        and decision.selected_corridor_ref
        and alignment.selected_corridor_ref != decision.selected_corridor_ref
    ):
        errors.append("alignment_decision_corridor_mismatch")


def _validate_constraints(
    trace: ReasoningTraceV2,
    *,
    travel_mode: str | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    r"""Validate evidence prerequisites for social and traffic constraints.

    A local constraint is valid only when evidence, path interaction, and travel
    mode support it:

    \[
    A(c)=E(c)\land I(\tau,c)\land M(c)
    \]
    """

    for item in trace.constraint_applicability.applicable:
        refs = set(item.trigger_refs)
        has_track = any(ref.startswith("track.") for ref in refs)
        has_control = any(ref.startswith("control.") for ref in refs)
        has_path = any(
            ref.startswith(prefix)
            for ref in refs
            for prefix in ("corridor.", "route.", "waypoint.", "crosswalk.", "bev.")
        )
        if item.constraint in {
            "yield_to_crossing_agent",
            "follow_agent",
            "pass_left",
            "pass_right",
        } and not has_track:
            errors.append(f"constraint_missing_track_ref:{item.constraint}")
        if item.constraint in {"wait_for_pedestrian_signal", "use_designated_crossing"}:
            if not has_control:
                errors.append(f"constraint_missing_control_ref:{item.constraint}")
            if not has_path:
                errors.append(f"constraint_missing_controlled_path_ref:{item.constraint}")
        if not has_path:
            errors.append(f"constraint_missing_explicit_path_ref:{item.constraint}")
        if travel_mode and item.constraint == "wait_for_pedestrian_signal" and travel_mode not in {
            "foot",
            "pedestrian",
            "pedestrian_assist",
        }:
            errors.append("pedestrian_signal_wrong_travel_mode")


def _validate_decision(
    trace: ReasoningTraceV2,
    *,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate policy consequences against applicable constraints."""

    decision = trace.decision
    constraints = trace.constraint_applicability.applicable
    constraint_names = {item.constraint for item in constraints}
    if decision.social_action in {"yield", "wait"} and not constraint_names:
        warnings.append("social_wait_without_applicable_constraint")
    if "yield_to_crossing_agent" in constraint_names and decision.social_action not in {"yield", "wait", "detour"}:
        errors.append("crossing_agent_constraint_not_reflected_in_decision")
    if "wait_for_pedestrian_signal" in constraint_names and decision.speed != "stop":
        errors.append("pedestrian_signal_wait_requires_stop_speed")
    if decision.motion == "stop" and decision.speed != "stop":
        errors.append("decision_stop_motion_requires_stop_speed")
