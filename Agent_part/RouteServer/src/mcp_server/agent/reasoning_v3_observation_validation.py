"""Validate LVLM V3.3 semantics and rationale before deterministic composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from mcp_server.contracts.reasoning_v3_observation import (
    DecisionRationaleObservationV3,
    ReasoningObservationV3,
)


@dataclass(frozen=True)
class ObservationValidationResultV3:
    """Represent semantic validation of one LVLM observation.

    Args:
        errors: Hard validation failures.
        warnings: Non-fatal audit findings.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether the observation has no hard error.

        Returns:
            ``True`` when ``errors`` is empty.
        """

        return not self.errors


def validate_reasoning_observation_v3(
    observation: ReasoningObservationV3,
    *,
    evidence_catalog: Iterable[Any],
    available_modalities: Iterable[str],
    navigation_context_source_refs: Iterable[str],
    instruction: str,
) -> ObservationValidationResultV3:
    """Validate modality accounting, reference closure, and grounded relations.

    Args:
        observation: Parsed LVLM observation.
        evidence_catalog: Frozen model-visible evidence entries.
        available_modalities: Frozen modality names.
        navigation_context_source_refs: Route-specific context sources that
            must each receive one independent grounding.
        instruction: Exact model-visible instruction or caption text.

    Returns:
        Structured errors and warnings.
    """

    errors: list[str] = []
    warnings: list[str] = []
    catalog = _normalize_catalog(evidence_catalog)
    catalog_refs = set(catalog)
    modalities = {str(item) for item in available_modalities}
    required_context_refs = {
        ref
        for ref, modality in catalog.items()
        if modality in {"bev", "goal", "goal_image", "waypoint", "route", "tool", "tool_route"}
    }
    required_navigation_refs = {str(item) for item in navigation_context_source_refs}

    task = observation.task_semantic
    normalized_instruction = " ".join(str(instruction).casefold().split())
    if task.applicable and "instruction.active" not in catalog_refs:
        errors.append("task_semantic_without_instruction_ref")
    if task.caption_role == "scene_description" and "instruction.active" not in catalog_refs:
        errors.append("caption_role_without_instruction_ref")
    for requirement in task.requirements:
        if "instruction.active" not in requirement.evidence_refs:
            errors.append(f"task_requirement_missing_instruction_ref:{requirement.code}")
        if not _span_supported(requirement.text_span, normalized_instruction):
            errors.append(f"task_requirement_span_not_in_instruction:{requirement.code}")
    for referent in task.referents:
        if "instruction.active" not in referent.evidence_refs:
            errors.append(f"task_referent_missing_instruction_ref:{referent.phrase}")
        if not _span_supported(referent.phrase, normalized_instruction):
            errors.append(f"task_referent_span_not_in_instruction:{referent.phrase}")
    for ambiguity in task.ambiguities:
        if "instruction.active" not in ambiguity.evidence_refs:
            errors.append(f"task_ambiguity_missing_instruction_ref:{ambiguity.type}")
        if ambiguity.text_span and not _span_supported(ambiguity.text_span, normalized_instruction):
            errors.append(f"task_ambiguity_span_not_in_instruction:{ambiguity.type}")

    generated_refs = {
        *(item.entity_ref for item in observation.scene_grounding.entities),
        *(item.control_ref for item in observation.scene_grounding.traffic_controls),
        *(item.region_ref for item in observation.local_traversability.regions),
    }
    for ref in sorted(_observation_input_refs(observation) - catalog_refs - generated_refs):
        errors.append(f"unknown_observation_evidence_ref:{ref}")

    scene = observation.scene_grounding
    if scene.applicable and "fpv" not in modalities:
        errors.append("scene_grounding_without_fpv")
    context_refs = {item.source_ref for item in scene.context_observations}
    for ref in sorted(required_context_refs - context_refs):
        errors.append(f"required_context_silently_omitted:{ref}")
    for ref in sorted(context_refs - required_context_refs):
        errors.append(f"unexpected_context_observation:{ref}")
    relation_sources = {item.source_ref for item in scene.cross_modal_relations}
    for item in scene.context_observations:
        if item.usage == "used" and item.role != "context_only":
            grounded_elsewhere = item.source_ref in relation_sources or item.source_ref in required_navigation_refs
            if not grounded_elsewhere:
                errors.append(f"used_context_missing_relation:{item.source_ref}")
    for item in scene.cross_modal_relations:
        if item.status in {"aligned", "partial", "conflict"}:
            if not _has_fpv_evidence(item.evidence_refs):
                errors.append(f"cross_modal_relation_missing_fpv_evidence:{item.source_ref}")

    entity_refs = {item.entity_ref for item in scene.entities}
    control_refs = {item.control_ref for item in scene.traffic_controls}
    for entity in scene.entities:
        if entity.motion_state in {"crossing_path", "same_direction", "approaching", "receding"}:
            if not _has_temporal_evidence(entity.evidence_refs, catalog):
                errors.append(f"dynamic_entity_missing_temporal_evidence:{entity.entity_ref}")

    local = observation.local_traversability
    if local.applicable and "fpv" not in modalities:
        errors.append("local_traversability_without_fpv")
    region_states = {item.region_ref: item.state for item in local.regions}
    for item in local.regions:
        refs = set(item.evidence_refs)
        if not any(ref == "fpv.current" or ref.startswith("fpv.history_") for ref in refs):
            errors.append(f"traversability_region_missing_fpv_ref:{item.region_ref}")
        for blocker_ref in item.blocker_refs:
            if blocker_ref not in entity_refs and blocker_ref not in control_refs:
                errors.append(f"traversability_unknown_blocker_ref:{item.region_ref}:{blocker_ref}")

    navigation = observation.navigation_context_grounding
    grounded_refs = {item.source_ref for item in navigation.source_groundings}
    for ref in sorted(required_navigation_refs - grounded_refs):
        errors.append(f"navigation_context_source_omitted:{ref}")
    for ref in sorted(grounded_refs - required_navigation_refs):
        errors.append(f"unexpected_navigation_context_source:{ref}")
    if navigation.applicable != bool(required_navigation_refs):
        errors.append("navigation_context_applicability_mismatch")
    for item in navigation.source_groundings:
        if item.status in {"aligned", "locally_blocked"} and not _has_fpv_evidence(item.evidence_refs):
            errors.append(f"navigation_context_missing_fpv_evidence:{item.source_ref}")
        if item.status == "aligned" and item.directive != "stop":
            if region_states.get(item.target_corridor_ref) != "direct":
                errors.append(f"aligned_context_target_not_direct:{item.source_ref}")
        if item.status == "locally_blocked" and item.directive != "stop":
            if region_states.get(item.target_corridor_ref) not in {"conditional", "blocked"}:
                errors.append(f"blocked_context_target_not_unavailable:{item.source_ref}")
        if item.status == "insufficient_evidence":
            warnings.append(f"navigation_context_insufficient_evidence:{item.source_ref}")

    interactions = observation.interaction_and_rule_grounding
    for item in interactions.interactions:
        if item.actor_ref not in entity_refs:
            errors.append(f"interaction_unknown_actor_ref:{item.actor_ref}")
        if item.affected_corridor_ref not in region_states:
            errors.append(f"interaction_unknown_corridor_ref:{item.affected_corridor_ref}")
        if item.affected_corridor_ref not in item.evidence_refs:
            errors.append(f"interaction_missing_corridor_evidence:{item.actor_ref}")
        if not _has_fpv_evidence(item.evidence_refs):
            errors.append(f"interaction_missing_fpv_evidence:{item.actor_ref}")
        if item.temporal_basis == "multi_frame" and not _has_temporal_evidence(item.evidence_refs, catalog):
            errors.append(f"interaction_missing_multiframe_evidence:{item.actor_ref}")
        if item.relation in {"crossing_path", "approaching_path", "ahead_same_direction"}:
            if item.temporal_basis not in {"multi_frame", "tracked_metadata"}:
                errors.append(f"dynamic_interaction_without_temporal_basis:{item.actor_ref}")
    for item in interactions.traffic_rules:
        if item.control_ref not in control_refs:
            errors.append(f"traffic_rule_unknown_control_ref:{item.control_ref}")
        if item.affected_corridor_ref not in region_states:
            errors.append(f"traffic_rule_unknown_corridor_ref:{item.affected_corridor_ref}")
        if item.affected_corridor_ref not in item.evidence_refs:
            errors.append(f"traffic_rule_missing_corridor_evidence:{item.control_ref}")
        if not _has_fpv_evidence(item.evidence_refs):
            errors.append(f"traffic_rule_missing_fpv_evidence:{item.control_ref}")
    return ObservationValidationResultV3(errors=errors, warnings=warnings)


def validate_decision_rationale_v3(
    rationale: DecisionRationaleObservationV3,
    *,
    expected_outcome: Mapping[str, Any],
    observation: ReasoningObservationV3,
    evidence_catalog: Iterable[Any],
    fixed_labels: Mapping[str, Any],
) -> ObservationValidationResultV3:
    """Validate hindsight rationale against frozen observations and outcome.

    Args:
        rationale: LVLM rationale that may select but not create reasons.
        expected_outcome: Deterministically frozen motion and corridor fields.
        observation: Previously validated task and visual observations.
        evidence_catalog: Input-visible evidence catalog.
        fixed_labels: Teacher-only authoritative outcome facts.

    Returns:
        Structured rationale errors and warnings.
    """

    errors: list[str] = []
    warnings: list[str] = []
    expected_motion = str(expected_outcome.get("motion") or "")
    if rationale.outcome_motion != expected_motion:
        errors.append("decision_rationale_outcome_mismatch")

    catalog_refs = set(_normalize_catalog(evidence_catalog))
    scene = observation.scene_grounding
    local = observation.local_traversability
    generated_refs = {
        *(item.entity_ref for item in scene.entities),
        *(item.control_ref for item in scene.traffic_controls),
        *(item.region_ref for item in local.regions),
    }
    allowed_refs = catalog_refs | generated_refs
    supports = [*rationale.supporting_reasons, *rationale.rejected_alternatives]
    for item in supports:
        for ref in item.evidence_refs:
            if ref not in allowed_refs:
                errors.append(f"decision_rationale_unknown_ref:{item.reason}:{ref}")
        errors.extend(
            _validate_decision_reason(
                item.reason,
                evidence_refs=item.evidence_refs,
                observation=observation,
                fixed_labels=fixed_labels,
                expected_outcome=expected_outcome,
            )
        )
    selected_reasons = {item.reason for item in rationale.supporting_reasons}
    selected_corridor = expected_outcome.get("selected_corridor_ref")
    motion = str(expected_outcome.get("motion") or "")
    speed = str(expected_outcome.get("speed") or "")
    social_action = str(expected_outcome.get("social_action") or "")
    if rationale.status == "not_applicable":
        errors.append("decision_rationale_not_applicable_with_frozen_outcome")
    if rationale.status in {"supported", "partial"} and rationale.confidence == "unknown":
        errors.append("decision_rationale_claims_support_with_unknown_confidence")
    if rationale.status == "unsupported" and rationale.confidence == "high":
        errors.append("unsupported_decision_rationale_cannot_be_high_confidence")
    if rationale.status == "supported":
        if motion != "stop" and (
            selected_corridor is None or "direct_corridor_available" not in selected_reasons
        ):
            errors.append("supported_moving_outcome_requires_direct_corridor_reason")
        stop_reasons = {
            "explicit_user_stop",
            "authoritative_goal_reached",
            "navigation_context_blocked",
            "local_path_blocked",
            "interaction_requires_yield",
            "interaction_requires_wait",
            "traffic_control_requires_action",
        }
        if motion == "stop" and not (selected_reasons & stop_reasons):
            errors.append("supported_stop_outcome_requires_grounded_stop_reason")
        required_social_reason = {
            "yield": "interaction_requires_yield",
            "wait": "interaction_requires_wait",
        }.get(social_action)
        if required_social_reason and required_social_reason not in selected_reasons:
            errors.append(f"supported_social_action_missing_reason:{social_action}")
        slow_reasons = {
            "local_risk_requires_slow",
            "interaction_requires_slow",
            "interaction_requires_yield",
            "navigation_context_blocked",
            "accessibility_requirement_applies",
            "traffic_control_requires_action",
        }
        if speed == "slow" and not (selected_reasons & slow_reasons):
            errors.append("supported_slow_outcome_requires_grounded_slow_reason")
    if rationale.status == "partial":
        warnings.append("decision_rationale_partial_support")
    if rationale.status == "unsupported":
        warnings.append("decision_rationale_unsupported")
    return ObservationValidationResultV3(errors=_unique(errors), warnings=warnings)


def _observation_input_refs(observation: ReasoningObservationV3) -> set[str]:
    """Collect only input-visible refs, excluding generated entity/control ids."""

    refs: set[str] = set()
    task = observation.task_semantic
    refs.update(task.objective_evidence_refs)
    for item in task.requirements:
        if item.target_ref:
            refs.add(item.target_ref)
        refs.update(item.evidence_refs)
    for item in task.referents:
        if item.resolved_ref:
            refs.add(item.resolved_ref)
        refs.update(item.evidence_refs)
    for item in task.ambiguities:
        refs.update(item.evidence_refs)
    scene = observation.scene_grounding
    for fact in scene.scene_facts:
        refs.update(fact.evidence_refs)
    for item in (*scene.entities, *scene.traffic_controls):
        refs.update(item.evidence_refs)
    for item in scene.context_observations:
        refs.add(item.source_ref)
        refs.update(item.evidence_refs)
    for item in scene.cross_modal_relations:
        refs.add(item.source_ref)
        if item.target_ref:
            refs.add(item.target_ref)
        refs.update(item.evidence_refs)
    local = observation.local_traversability
    if local.preferred_clear_corridor_ref:
        refs.add(local.preferred_clear_corridor_ref)
    for item in local.regions:
        refs.add(item.region_ref)
        refs.update(item.evidence_refs)
    for item in observation.navigation_context_grounding.source_groundings:
        refs.add(item.source_ref)
        if item.target_corridor_ref:
            refs.add(item.target_corridor_ref)
        refs.update(item.evidence_refs)
    interaction = observation.interaction_and_rule_grounding
    for item in (*interaction.interactions, *interaction.traffic_rules):
        refs.add(item.affected_corridor_ref)
        refs.update(item.evidence_refs)
    return refs


def _validate_decision_reason(
    reason: str,
    *,
    evidence_refs: Iterable[str],
    observation: ReasoningObservationV3,
    fixed_labels: Mapping[str, Any],
    expected_outcome: Mapping[str, Any],
) -> list[str]:
    """Return semantic errors for one controlled hindsight reason."""

    refs = set(evidence_refs)
    local = observation.local_traversability
    navigation = observation.navigation_context_grounding
    interactions = observation.interaction_and_rule_grounding
    required = {
        *(item.required_behavior for item in interactions.interactions),
        *(item.required_behavior for item in interactions.traffic_rules),
    }
    if reason == "authoritative_goal_reached" and not bool(fixed_labels.get("goal_reached")):
        return ["decision_reason_goal_reached_without_authoritative_fact"]
    if reason == "authoritative_goal_reached" and "goal.active" not in refs:
        return ["decision_reason_goal_reached_missing_goal_ref"]
    if reason == "explicit_user_stop" and observation.task_semantic.inferred_objective not in {
        "hold_position",
        "stop_at_goal",
    }:
        return ["decision_reason_explicit_stop_without_task_support"]
    if reason == "explicit_user_stop" and "instruction.active" not in refs:
        return ["decision_reason_explicit_stop_missing_instruction_ref"]
    if reason == "direct_corridor_available":
        direct = {item.region_ref for item in local.regions if item.state == "direct"}
        if expected_outcome.get("selected_corridor_ref") not in direct:
            return ["decision_reason_direct_corridor_without_direct_target"]
        if expected_outcome.get("selected_corridor_ref") not in refs or not _has_fpv_evidence(refs):
            return ["decision_reason_direct_corridor_missing_target_evidence"]
    if reason == "navigation_context_aligned" and navigation.fused_execution_status != "clear":
        return ["decision_reason_alignment_without_clear_context"]
    if reason == "navigation_context_aligned" and not any(
        item.source_ref in refs and item.status == "aligned"
        for item in navigation.source_groundings
    ):
        return ["decision_reason_alignment_missing_source_evidence"]
    if reason == "navigation_context_blocked" and navigation.fused_execution_status != "locally_blocked":
        return ["decision_reason_blocked_without_blocked_context"]
    if reason == "local_path_blocked" and not any(
        item.state in {"conditional", "blocked", "forbidden"} for item in local.regions
    ):
        return ["decision_reason_blocked_without_unavailable_region"]
    if reason == "local_path_blocked" and not any(
        item.region_ref in refs and item.state in {"conditional", "blocked", "forbidden"}
        for item in local.regions
    ):
        return ["decision_reason_blocked_missing_region_evidence"]
    if reason == "local_risk_requires_slow" and local.local_risk not in {"medium", "high"}:
        return ["decision_reason_slow_without_local_risk"]
    behavior_by_reason = {
        "interaction_requires_slow": {"slow", "follow", "pass_left", "pass_right", "detour"},
        "interaction_requires_yield": {"yield"},
        "interaction_requires_wait": {"wait", "stop"},
    }
    if reason in behavior_by_reason and not (required & behavior_by_reason[reason]):
        return [f"decision_reason_without_required_behavior:{reason}"]
    if reason in behavior_by_reason and not any(
        item.affected_corridor_ref in refs
        and item.required_behavior in behavior_by_reason[reason]
        for item in interactions.interactions
    ):
        return [f"decision_reason_missing_interaction_evidence:{reason}"]
    if reason == "traffic_control_requires_action" and not interactions.traffic_rules:
        return ["decision_reason_traffic_control_without_rule"]
    if reason == "traffic_control_requires_action" and not any(
        item.control_ref in refs or item.affected_corridor_ref in refs
        for item in interactions.traffic_rules
    ):
        return ["decision_reason_traffic_control_missing_rule_evidence"]
    if reason == "accessibility_requirement_applies" and not observation.task_semantic.requirements:
        return ["decision_reason_accessibility_without_requirement"]
    return []


def _span_supported(span: str, normalized_instruction: str) -> bool:
    """Return whether a normalized phrase occurs in the instruction."""

    normalized_span = " ".join(str(span).casefold().split())
    return bool(normalized_span) and normalized_span in normalized_instruction


def _unique(values: Iterable[str]) -> list[str]:
    """Return ordered unique validation messages."""

    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _has_temporal_evidence(refs: Iterable[str], catalog: Mapping[str, str]) -> bool:
    """Return whether refs support temporal reasoning rather than one-frame inference."""

    values = {str(ref) for ref in refs}
    fpv_refs = {ref for ref in values if ref == "fpv.current" or ref.startswith("fpv.history_")}
    if len(fpv_refs) >= 2:
        return True
    return any(catalog.get(ref) == "track" or ref.startswith("track.") for ref in values)


def _has_fpv_evidence(refs: Iterable[str]) -> bool:
    """Return whether refs cite at least one actual FPV frame."""

    return any(ref == "fpv.current" or str(ref).startswith("fpv.history_") for ref in refs)


def _normalize_catalog(entries: Iterable[Any]) -> dict[str, str]:
    """Normalize evidence catalog objects to ``ref -> modality``."""

    result: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, Mapping):
            ref = entry.get("ref")
            if ref:
                result[str(ref)] = str(entry.get("modality") or "unknown")
            continue
        ref = getattr(entry, "ref", None)
        if ref:
            result[str(ref)] = str(getattr(entry, "modality", "unknown"))
    return result


__all__ = [
    "ObservationValidationResultV3",
    "validate_decision_rationale_v3",
    "validate_reasoning_observation_v3",
]
