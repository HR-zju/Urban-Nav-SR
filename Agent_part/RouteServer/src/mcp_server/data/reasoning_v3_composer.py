"""Compose canonical V3.3 Full-CoT targets from validated teacher outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3
from mcp_server.contracts.reasoning_v3 import (
    DecisionV3,
    FinishV3,
    InteractionAndRuleGroundingV3,
    LabelProvenanceEntryV3,
    LocalTraversabilityV3,
    NavigationContextGroundingV3,
    NavigationSourceGroundingV3,
    ReasoningTargetV3,
    ReasoningTraceV3,
    SectionLossMaskV3,
    TaskGroundingV3,
)
from mcp_server.contracts.reasoning_v3_observation import (
    DecisionRationaleObservationV3,
    ReasoningObservationV3,
    TaskSemanticObservationV3,
)
from mcp_server.data.reasoning_v3_conditions import (
    build_navigation_context_sources_v3,
    fuse_navigation_context_v3,
)


REASONING_V3_COMPOSER_VERSION = "urbannav_reasoning_v3_3_composer_v1"


@dataclass(frozen=True)
class DecisionOutcomeV3:
    """Represent the frozen symbolic outcome explained by the rationale teacher.

    Args:
        motion: Selected local motion.
        speed: Selected speed policy.
        social_action: Selected social action.
        selected_corridor_ref: Executable FPV corridor, when motion progresses.
        fallback: Conservative fallback policy.
        confidence: Confidence inherited from validated observations.
    """

    motion: str
    speed: str
    social_action: str
    selected_corridor_ref: str | None
    fallback: str
    confidence: str

    def model_dump(self) -> dict[str, str | None]:
        """Return a JSON-compatible teacher-only outcome payload.

        Returns:
            Frozen symbolic outcome fields.
        """

        return {
            "motion": self.motion,
            "speed": self.speed,
            "social_action": self.social_action,
            "selected_corridor_ref": self.selected_corridor_ref,
            "fallback": self.fallback,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class _CompositionStateV3:
    """Hold deterministic intermediate state shared by outcome and target composition."""

    demonstrated_direction: str
    navigation_context: NavigationContextGroundingV3
    selected_corridor: str | None


@dataclass(frozen=True)
class ReasoningCompositionResultV3:
    """Represent a composed target and trainable supervision metadata.

    Args:
        target: Canonical Full-CoT target.
        label_provenance: Field-level deterministic and LVLM provenance.
        section_loss_mask: Post-composition section eligibility.
        warnings: Non-fatal composition audit findings.
    """

    target: ReasoningTargetV3
    label_provenance: dict[str, LabelProvenanceEntryV3]
    section_loss_mask: SectionLossMaskV3
    warnings: list[str] = field(default_factory=list)


def compose_reasoning_target_v3(
    source: ReasoningSourceV3,
    observation: ReasoningObservationV3,
    decision_rationale: DecisionRationaleObservationV3,
) -> ReasoningCompositionResultV3:
    """Compose a V3.3 target while preserving supervision authority boundaries.

    The LVLM owns task semantics, scene/affordance observations, cross-modal
    relations, and a hindsight rationale constrained to the frozen outcome.
    Rules own source identities, authoritative runtime facts, trusted motion,
    evidence closure, and deterministic ``finish`` serialization.

    Args:
        source: Frozen model input and teacher-only deterministic supervision.
        observation: Validated LVLM observation.
        decision_rationale: Validated rationale for the frozen symbolic outcome.

    Returns:
        Composed target, provenance, loss mask, and warnings.

    Raises:
        ValueError: If source coverage, strong labels, or visual execution
            semantics conflict.
    """

    fixed = _fixed_labels(source)
    state = _build_composition_state(source, observation, fixed=fixed)
    specs = build_navigation_context_sources_v3(source)
    source_groundings = state.navigation_context.source_groundings
    authoritative_objective = str(fixed.get("authoritative_local_objective") or "unknown")
    warnings: list[str] = []
    if (
        authoritative_objective != "unknown"
        and observation.task_semantic.inferred_objective != "unknown"
        and authoritative_objective != observation.task_semantic.inferred_objective
    ):
        warnings.append("task_semantic_objective_overridden_by_authoritative_runtime_state")
    task = _compose_task(
        source,
        observation.task_semantic,
        authoritative_objective=authoritative_objective,
        active_navigation_refs=[item.source_ref for item in specs],
    )
    interaction = observation.interaction_and_rule_grounding
    outcome = _derive_decision_outcome(
        observation,
        demonstrated_direction=state.demonstrated_direction,
        navigation_context=state.navigation_context,
        selected_corridor=state.selected_corridor,
        interaction=interaction,
    )
    decision = _compose_decision(outcome, decision_rationale)
    # 等待或让行只暂停即时运动，不应抹掉已经验证的后续执行走廊。
    # 真正没有可执行走廊时，_select_executable_corridor 已经返回 None。
    local = LocalTraversabilityV3(
        applicable=observation.local_traversability.applicable,
        regions=observation.local_traversability.regions,
            preferred_clear_corridor_ref=observation.local_traversability.preferred_clear_corridor_ref,
        visibility=observation.local_traversability.visibility,
        local_risk=observation.local_traversability.local_risk,
        confidence=observation.local_traversability.confidence,
    )
    trace = ReasoningTraceV3(
        task_grounding=task,
        scene_grounding=observation.scene_grounding,
        local_traversability=local,
        navigation_context_grounding=state.navigation_context,
        interaction_and_rule_grounding=interaction,
        decision=decision,
    )
    finish = FinishV3(
        decision_type="navigate",
        motion=decision.motion,
        speed=decision.speed,
        social_action=decision.social_action,
        confidence_level=decision.confidence,
        trajectory_type="stop_hold" if decision.motion == "stop" else "local_navigation",
        route_action="continue_navigation",
        tool_used=any(item.source_type == "tool" for item in source_groundings),
    )
    target = ReasoningTargetV3(reasoning=trace, finish=finish)
    provenance = _composition_provenance(
        source,
        observation,
        decision_rationale=decision_rationale,
        demonstrated_direction=state.demonstrated_direction,
        source_groundings=source_groundings,
    )
    section_mask = _composition_section_mask(
        source.section_loss_mask,
        target,
        provenance=provenance,
    )
    return ReasoningCompositionResultV3(
        target=target,
        label_provenance=provenance,
        section_loss_mask=section_mask,
        warnings=warnings,
    )


def derive_decision_outcome_v3(
    source: ReasoningSourceV3,
    observation: ReasoningObservationV3,
) -> DecisionOutcomeV3:
    """Derive the frozen decision outcome before hindsight-rationale annotation.

    Args:
        source: Frozen model input and teacher-only trusted outcome labels.
        observation: Validated task and visual observations produced without GT.

    Returns:
        Symbolic outcome supplied to the decision-rationale teacher.

    Raises:
        ValueError: If fixed supervision or visual execution support conflicts.
    """

    fixed = _fixed_labels(source)
    state = _build_composition_state(source, observation, fixed=fixed)
    return _derive_decision_outcome(
        observation,
        demonstrated_direction=state.demonstrated_direction,
        navigation_context=state.navigation_context,
        selected_corridor=state.selected_corridor,
        interaction=observation.interaction_and_rule_grounding,
    )


def _compose_task(
    source: ReasoningSourceV3,
    semantic: TaskSemanticObservationV3,
    *,
    authoritative_objective: str,
    active_navigation_refs: list[str],
) -> TaskGroundingV3:
    """Close LVLM task semantics with deterministic source identities and state."""

    catalog_refs = {entry.ref for entry in source.model_input.evidence_catalog}
    applicable = semantic.applicable and "instruction.active" in catalog_refs
    if not applicable:
        return TaskGroundingV3(
            applicable=False,
            instruction_ref="instruction.active" if "instruction.active" in catalog_refs else None,
            caption_role=semantic.caption_role,
            confidence=semantic.confidence,
        )
    objective = (
        authoritative_objective
        if authoritative_objective != "unknown"
        else semantic.inferred_objective
    )
    objective_refs = list(semantic.objective_evidence_refs)
    if authoritative_objective != "unknown" and "goal.active" in catalog_refs:
        objective_refs = _unique([*objective_refs, "goal.active"])
    return TaskGroundingV3(
        applicable=True,
        instruction_ref="instruction.active",
        instruction_type=semantic.instruction_type,
        caption_role=semantic.caption_role,
        goal_ref="goal.active" if "goal.active" in catalog_refs else None,
        active_navigation_refs=_unique(active_navigation_refs),
        local_objective=objective,
        objective_evidence_refs=objective_refs,
        requirements=semantic.requirements,
        referents=semantic.referents,
        ambiguities=semantic.ambiguities,
        confidence=semantic.confidence,
    )


def _fixed_labels(source: ReasoningSourceV3) -> dict:
    """Return and version-check V3.3 teacher-only fixed labels."""

    version = source.teacher_supervision_context.get("fixed_supervision_version")
    if version != "reasoning_v3_3_fixed_supervision_v1":
        raise ValueError("source_requires_reasoning_v3_3_fixed_supervision_v1")
    return dict(source.teacher_supervision_context.get("fixed_labels") or {})


def _build_composition_state(
    source: ReasoningSourceV3,
    observation: ReasoningObservationV3,
    *,
    fixed: dict,
) -> _CompositionStateV3:
    """Build deterministic context fusion and executable-corridor state."""

    demonstrated_direction = str(fixed.get("demonstrated_direction") or "unknown")
    demonstrated_corridor = fixed.get("demonstrated_corridor_ref")
    specs = build_navigation_context_sources_v3(source)
    source_groundings = _close_navigation_sources(source, observation, specs=specs)
    fused_direction, fused_target, consistency, fused_status = fuse_navigation_context_v3(source_groundings)
    navigation_context = NavigationContextGroundingV3(
        applicable=bool(source_groundings),
        source_groundings=source_groundings,
        fused_direction=fused_direction,
        fused_target_corridor_ref=fused_target,
        source_consistency=consistency,
        fused_execution_status=fused_status,
    )
    selected_corridor = _select_executable_corridor(
        observation,
        demonstrated_direction=demonstrated_direction,
        demonstrated_corridor=demonstrated_corridor,
        navigation_context=navigation_context,
    )
    return _CompositionStateV3(
        demonstrated_direction=demonstrated_direction,
        navigation_context=navigation_context,
        selected_corridor=selected_corridor,
    )


def _close_navigation_sources(
    source: ReasoningSourceV3,
    observation: ReasoningObservationV3,
    *,
    specs: list,
) -> list[NavigationSourceGroundingV3]:
    """Bind LVLM source relations to deterministic source identities/directions."""

    observed = {
        item.source_ref: item
        for item in observation.navigation_context_grounding.source_groundings
    }
    if set(observed) != {item.source_ref for item in specs}:
        raise ValueError("navigation_context_source_coverage_mismatch")
    result: list[NavigationSourceGroundingV3] = []
    for spec in specs:
        item = observed[spec.source_ref]
        if item.source_type != spec.source_type:
            raise ValueError(f"navigation_context_source_type_mismatch:{spec.source_ref}")
        directive = item.directive
        if spec.deterministic_direction != "unknown":
            if directive != "unknown" and directive != spec.deterministic_direction:
                raise ValueError(f"observed_context_direction_conflicts_with_fixed:{spec.source_ref}")
            directive = spec.deterministic_direction
        reliability = spec.reliability if spec.reliability != "unknown" else item.reliability
        result.append(item.model_copy(update={"directive": directive, "reliability": reliability}))
    return result


def _select_executable_corridor(
    observation: ReasoningObservationV3,
    *,
    demonstrated_direction: str,
    demonstrated_corridor: str | None,
    navigation_context: NavigationContextGroundingV3,
) -> str | None:
    """Select direct FPV support without mapping odometry direction to image thirds."""

    regions = {item.region_ref: item.state for item in observation.local_traversability.regions}
    direct = {ref for ref, state in regions.items() if state == "direct"}
    if demonstrated_direction == "stop":
        return None
    # 只有通过投影 QC 的 FPV corridor 才能强约束图像区域；odometry 方向本身不做这个映射。
    if demonstrated_corridor is not None:
        return demonstrated_corridor if regions.get(demonstrated_corridor) == "direct" else None
    if (
        navigation_context.fused_execution_status == "clear"
        and navigation_context.fused_target_corridor_ref in direct
    ):
        return navigation_context.fused_target_corridor_ref
    preferred = observation.local_traversability.preferred_clear_corridor_ref
    if preferred in direct:
        return preferred
    if direct:
        return sorted(direct)[0]
    return None


def _derive_decision_outcome(
    observation: ReasoningObservationV3,
    *,
    demonstrated_direction: str,
    navigation_context: NavigationContextGroundingV3,
    selected_corridor: str | None,
    interaction: InteractionAndRuleGroundingV3,
) -> DecisionOutcomeV3:
    """Apply safety-first arbitration after all observations are frozen."""

    relations = [*interaction.interactions, *interaction.traffic_rules]
    required = {item.required_behavior for item in relations}
    if demonstrated_direction != "unknown":
        motion = _motion_from_direction(demonstrated_direction)
    elif {"stop", "wait"} & required:
        motion = "stop"
    elif (
        navigation_context.fused_execution_status == "clear"
        and navigation_context.fused_direction != "unknown"
    ):
        motion = _motion_from_direction(navigation_context.fused_direction)
    else:
        motion = _motion_from_corridor(selected_corridor)
    if selected_corridor is None and motion != "stop" and demonstrated_direction == "unknown":
        motion = "stop"
    speed = "stop" if motion == "stop" else (
        "slow"
        if required & {"slow", "yield", "follow", "pass_left", "pass_right", "detour"}
        or observation.local_traversability.local_risk in {"medium", "high"}
        else "normal"
    )
    social_action = _social_action(required, motion=motion)
    confidence = _minimum_confidence(
        observation.scene_grounding.confidence,
        observation.local_traversability.confidence,
    )
    return DecisionOutcomeV3(
        motion=motion,
        speed=speed,
        social_action=social_action,
        selected_corridor_ref=selected_corridor,
        fallback="stop" if interaction.applicable or observation.local_traversability.local_risk == "high" else "slow",
        confidence=confidence,
    )


def _compose_decision(
    outcome: DecisionOutcomeV3,
    rationale: DecisionRationaleObservationV3,
) -> DecisionV3:
    """Attach a validated hindsight rationale to the frozen symbolic outcome."""

    if rationale.outcome_motion != outcome.motion:
        raise ValueError("decision_rationale_outcome_mismatch")
    return DecisionV3(
        motion=outcome.motion,
        speed=outcome.speed,
        social_action=outcome.social_action,
        selected_corridor_ref=outcome.selected_corridor_ref,
        fallback=outcome.fallback,
        confidence=_minimum_confidence(outcome.confidence, rationale.confidence),
        rationale_status=rationale.status,
        supporting_reasons=rationale.supporting_reasons,
        rejected_alternatives=rationale.rejected_alternatives,
    )


def _composition_provenance(
    source: ReasoningSourceV3,
    observation: ReasoningObservationV3,
    *,
    decision_rationale: DecisionRationaleObservationV3,
    demonstrated_direction: str,
    source_groundings: list[NavigationSourceGroundingV3],
) -> dict[str, LabelProvenanceEntryV3]:
    """Build section provenance after deterministic composition."""

    result = dict(source.label_provenance)
    scene_confidence = _confidence_score(observation.scene_grounding.confidence)
    local_confidence = _confidence_score(observation.local_traversability.confidence)
    task_confidence = _confidence_score(observation.task_semantic.confidence)
    if observation.task_semantic.applicable or observation.task_semantic.caption_role == "scene_description":
        result["task_grounding"] = LabelProvenanceEntryV3(
            source="validated_lvlm_task_semantic",
            strength="weak",
            generator_version=REASONING_V3_COMPOSER_VERSION,
            confidence=task_confidence,
        )
    result["scene_grounding"] = LabelProvenanceEntryV3(
        source="validated_lvlm_observation",
        strength="weak",
        generator_version=REASONING_V3_COMPOSER_VERSION,
        confidence=scene_confidence,
    )
    result["local_traversability"] = LabelProvenanceEntryV3(
        source="validated_lvlm_observation",
        strength="weak",
        generator_version=REASONING_V3_COMPOSER_VERSION,
        confidence=local_confidence,
    )
    if source_groundings:
        result["navigation_context_grounding"] = LabelProvenanceEntryV3(
            source="validated_per_source_context_grounding",
            strength="weak",
            generator_version=REASONING_V3_COMPOSER_VERSION,
            confidence=min((0.9 if item.reliability == "high" else 0.6 for item in source_groundings), default=0.0),
        )
    if observation.interaction_and_rule_grounding.applicable:
        result["interaction_and_rule_grounding"] = LabelProvenanceEntryV3(
            source="validated_actor_path_relations",
            strength="weak",
            generator_version=REASONING_V3_COMPOSER_VERSION,
            confidence=scene_confidence,
        )
    result["decision"] = LabelProvenanceEntryV3(
        source="trusted_demonstration_policy" if demonstrated_direction != "unknown" else "deterministic_safety_composer",
        strength="strong" if demonstrated_direction != "unknown" else "weak",
        generator_version=REASONING_V3_COMPOSER_VERSION,
        confidence=1.0 if demonstrated_direction != "unknown" else min(scene_confidence, local_confidence),
    )
    if decision_rationale.status in {"supported", "partial"}:
        result["decision_rationale"] = LabelProvenanceEntryV3(
            source="validated_hindsight_rationale",
            strength="weak",
            generator_version=REASONING_V3_COMPOSER_VERSION,
            confidence=_confidence_score(decision_rationale.confidence),
        )
    return result


def _composition_section_mask(
    source_mask: SectionLossMaskV3,
    target: ReasoningTargetV3,
    *,
    provenance: dict[str, LabelProvenanceEntryV3],
) -> SectionLossMaskV3:
    """Intersect source authorization with post-composition eligibility."""

    trace = target.reasoning
    context_statuses = {item.status for item in trace.navigation_context_grounding.source_groundings}
    return SectionLossMaskV3(
        task_grounding=(
            source_mask.task_grounding
            and "task_grounding" in provenance
        ),
        scene_grounding=source_mask.scene_grounding and trace.scene_grounding.applicable,
        local_traversability=(
            source_mask.local_traversability
            and trace.local_traversability.applicable
            and bool(trace.local_traversability.regions)
        ),
        navigation_context_grounding=(
            source_mask.navigation_context_grounding
            and trace.navigation_context_grounding.applicable
            and bool(context_statuses)
            and "insufficient_evidence" not in context_statuses
            and "conflict" not in context_statuses
        ),
        interaction_and_rule_grounding=(
            source_mask.interaction_and_rule_grounding
            and trace.interaction_and_rule_grounding.applicable
            and "interaction_and_rule_grounding" in provenance
        ),
        decision=(
            source_mask.decision
            and "decision" in provenance
            and "decision_rationale" in provenance
            and trace.decision.rationale_status in {"supported", "partial"}
        ),
    )


def _social_action(required: set[str], *, motion: str) -> str:
    """Map grounded required behaviors to one final social action."""

    if "wait" in required or ("yield" in required and motion == "stop"):
        return "wait"
    for value in ("yield", "detour", "pass_left", "pass_right", "follow"):
        if value in required:
            return value
    return "none"


def _motion_from_direction(direction: str) -> str:
    """Map coarse direction to the symbolic motion vocabulary."""

    return {
        "left": "turn_left",
        "straight": "go_straight",
        "right": "turn_right",
        "stop": "stop",
    }.get(direction, "stop")


def _motion_from_corridor(corridor: str | None) -> str:
    """Map a canonical corridor address to a fallback motion."""

    return {
        "corridor.left": "turn_left",
        "corridor.center": "go_straight",
        "corridor.right": "turn_right",
    }.get(corridor, "stop")


def _minimum_confidence(*values: str) -> str:
    """Return the conservative minimum of controlled confidence values."""

    order = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    return min(values, key=lambda value: order.get(str(value), 0)) if values else "unknown"


def _confidence_score(value: str) -> float:
    """Map controlled confidence to an audit score."""

    return {"high": 0.9, "medium": 0.75, "low": 0.5, "unknown": 0.0}.get(str(value), 0.0)


def _unique(values: list[str]) -> list[str]:
    """Return ordered unique non-empty strings."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "DecisionOutcomeV3",
    "REASONING_V3_COMPOSER_VERSION",
    "ReasoningCompositionResultV3",
    "compose_reasoning_target_v3",
    "derive_decision_outcome_v3",
]
