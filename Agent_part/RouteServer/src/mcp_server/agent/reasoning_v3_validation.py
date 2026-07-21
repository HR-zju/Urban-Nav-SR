"""Semantic, provenance, and mask validation for Reasoning V3.3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from mcp_server.agent.reasoning_v3_text import parse_reasoning_target_v3
from mcp_server.contracts.reasoning_v3 import (
    LabelProvenanceEntryV3,
    ReasoningTargetV3,
    SectionLossMaskV3,
    reasoning_evidence_refs_v3,
)


@dataclass(frozen=True)
class ReasoningValidationResultV3:
    """Represent semantic validation of one V3.3 target.

    Args:
        errors: Hard semantic validation failures.
        warnings: Non-fatal audit findings.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether the target has no hard error.

        Returns:
            ``True`` when ``errors`` is empty.
        """

        return not self.errors


def validate_reasoning_target_v3(
    target: ReasoningTargetV3,
    *,
    instruction: str | None = None,
    evidence_catalog: Iterable[Any],
    available_modalities: Iterable[str],
    label_provenance: Mapping[str, LabelProvenanceEntryV3 | Mapping[str, Any]] | None = None,
    section_loss_mask: SectionLossMaskV3 | Mapping[str, Any] | None = None,
    source_section_loss_mask: SectionLossMaskV3 | Mapping[str, Any] | None = None,
    fixed_labels: Mapping[str, Any] | None = None,
    dataset_source: str | None = None,
    assistant_text: str | None = None,
) -> ReasoningValidationResultV3:
    """Validate one V3.3 target against frozen input and private supervision.

    Args:
        target: Composed canonical Full-CoT target.
        instruction: Exact model-visible instruction or caption text.
        evidence_catalog: Frozen model-visible evidence entries.
        available_modalities: Frozen modality names.
        label_provenance: Effective field-level label provenance.
        section_loss_mask: Final post-composition section eligibility.
        source_section_loss_mask: Source-level upper bound for every section.
        fixed_labels: Private deterministic labels used only for validation.
        dataset_source: Canonical dataset family for trust-policy checks.
        assistant_text: Optional rendered target for parser roundtrip checks.

    Returns:
        Structured semantic errors and warnings.
    """

    errors: list[str] = []
    warnings: list[str] = []
    catalog = _normalize_catalog(evidence_catalog)
    catalog_refs = set(catalog)
    generated_refs = _generated_refs(target)
    for ref in sorted(reasoning_evidence_refs_v3(target.reasoning) - catalog_refs - generated_refs):
        errors.append(f"unknown_evidence_ref:{ref}")
    modalities = {str(item) for item in available_modalities}
    _validate_task(target, instruction=instruction, catalog=catalog, errors=errors)
    _validate_scene(target, catalog=catalog, modalities=modalities, errors=errors)
    _validate_traversability(target, modalities=modalities, errors=errors, warnings=warnings)
    _validate_navigation_context(target, catalog=catalog, modalities=modalities, errors=errors, warnings=warnings)
    _validate_interactions(target, errors=errors)
    provenance = _normalize_provenance(label_provenance or {}, errors=errors)
    _validate_provenance(target, provenance=provenance, dataset_source=dataset_source, errors=errors, warnings=warnings)
    _validate_fixed_labels(target, fixed_labels=fixed_labels or {}, provenance=provenance, errors=errors)
    if section_loss_mask is not None:
        try:
            mask = _normalize_mask(section_loss_mask)
            source_mask = _normalize_mask(source_section_loss_mask) if source_section_loss_mask is not None else None
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid_section_loss_mask:{exc}")
        else:
            _validate_section_mask(target, mask=mask, source_mask=source_mask, provenance=provenance, errors=errors)
    if assistant_text is not None:
        parsed = parse_reasoning_target_v3(assistant_text)
        if not parsed.valid:
            errors.extend(f"assistant_text:{item}" for item in parsed.errors)
        elif parsed.target != target:
            errors.append("assistant_text_target_mismatch")
    return ReasoningValidationResultV3(errors=errors, warnings=warnings)


def _validate_task(
    target: ReasoningTargetV3,
    *,
    instruction: str | None,
    catalog: Mapping[str, str],
    errors: list[str],
) -> None:
    """Validate that TaskGrounding contains task semantics rather than scene facts."""

    task = target.reasoning.task_grounding
    if not task.applicable and task.caption_role != "scene_description":
        return
    refs = {task.instruction_ref, task.goal_ref, *task.active_navigation_refs} - {None}
    for ref in sorted(str(item) for item in refs):
        if ref not in catalog:
            errors.append(f"task_grounding_unknown_ref:{ref}")
    if task.instruction_ref and catalog.get(task.instruction_ref) != "instruction":
        errors.append("task_grounding_instruction_ref_modality_mismatch")
    normalized = " ".join(str(instruction or "").casefold().split())
    for requirement in task.requirements:
        if not _span_supported(requirement.text_span, normalized):
            errors.append(f"task_requirement_span_not_in_instruction:{requirement.code}")
    for referent in task.referents:
        if not _span_supported(referent.phrase, normalized):
            errors.append(f"task_referent_span_not_in_instruction:{referent.phrase[:80]}")
    for ambiguity in task.ambiguities:
        if ambiguity.text_span and not _span_supported(ambiguity.text_span, normalized):
            errors.append(f"task_ambiguity_span_not_in_instruction:{ambiguity.type}")


def _validate_scene(
    target: ReasoningTargetV3,
    *,
    catalog: Mapping[str, str],
    modalities: set[str],
    errors: list[str],
) -> None:
    """Validate scene-local ids and complete optional-context accounting."""

    scene = target.reasoning.scene_grounding
    if scene.applicable and "fpv" not in modalities:
        errors.append("scene_grounding_applicable_without_fpv")
    required = {
        ref for ref, modality in catalog.items()
        if modality in {"bev", "goal", "goal_image", "waypoint", "route", "tool", "tool_route"}
    }
    observed = {item.source_ref for item in scene.context_observations}
    for ref in sorted(required - observed):
        errors.append(f"target_context_silently_omitted:{ref}")
    for ref in sorted(observed - required):
        errors.append(f"target_context_unexpected_source:{ref}")
    entity_refs = {item.entity_ref for item in scene.entities}
    control_refs = {item.control_ref for item in scene.traffic_controls}
    if entity_refs & control_refs:
        errors.append("scene_entity_and_control_refs_overlap")


def _validate_traversability(
    target: ReasoningTargetV3,
    *,
    modalities: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate FPV-first semantic geometry and selection."""

    section = target.reasoning.local_traversability
    if section.applicable and "fpv" not in modalities:
        errors.append("traversability_applicable_without_fpv")
    if not section.applicable:
        return
    for item in section.regions:
        if not item.region_ref.startswith(("corridor.", "region.", "crosswalk.")):
            errors.append(f"invalid_traversability_region_ref:{item.region_ref}")
        if not any(ref == "fpv.current" or ref.startswith("fpv.history_") for ref in item.evidence_refs):
            errors.append(f"traversability_region_missing_fpv_ref:{item.region_ref}")
    if target.reasoning.decision.selected_corridor_ref is None and target.reasoning.decision.motion != "stop":
        warnings.append("moving_decision_without_selected_execution_corridor")


def _validate_navigation_context(
    target: ReasoningTargetV3,
    *,
    catalog: Mapping[str, str],
    modalities: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate every optional navigation source independently before fusion."""

    section = target.reasoning.navigation_context_grounding
    if not section.applicable:
        return
    if "fpv" not in modalities:
        errors.append("navigation_context_applicable_without_fpv")
    for item in section.source_groundings:
        modality = catalog.get(item.source_ref)
        if modality is None:
            errors.append(f"navigation_context_unknown_source:{item.source_ref}")
        elif not _source_type_matches_modality(item.source_type, modality):
            errors.append(f"navigation_context_source_modality_mismatch:{item.source_type}:{item.source_ref}:{modality}")
        if item.status in {"aligned", "locally_blocked"}:
            if not any(ref == "fpv.current" or ref.startswith("fpv.history_") for ref in item.evidence_refs):
                errors.append(f"navigation_context_missing_fpv_evidence:{item.source_ref}")
        if item.status == "insufficient_evidence":
            warnings.append(f"navigation_context_insufficient_evidence:{item.source_ref}")


def _validate_interactions(target: ReasoningTargetV3, *, errors: list[str]) -> None:
    """Validate actor/control/path relations against Scene and Local sections."""

    trace = target.reasoning
    entities = {item.entity_ref for item in trace.scene_grounding.entities}
    controls = {item.control_ref for item in trace.scene_grounding.traffic_controls}
    regions = {item.region_ref for item in trace.local_traversability.regions}
    section = trace.interaction_and_rule_grounding
    for item in section.interactions:
        if item.actor_ref not in entities:
            errors.append(f"interaction_actor_missing_from_scene:{item.actor_ref}")
        if item.affected_corridor_ref not in regions:
            errors.append(f"interaction_corridor_missing_from_local:{item.affected_corridor_ref}")
        if not any(ref == "fpv.current" or ref.startswith("fpv.history_") for ref in item.evidence_refs):
            errors.append(f"interaction_missing_fpv_evidence:{item.actor_ref}")
    for item in section.traffic_rules:
        if item.control_ref not in controls:
            errors.append(f"traffic_control_missing_from_scene:{item.control_ref}")
        if item.affected_corridor_ref not in regions:
            errors.append(f"traffic_rule_corridor_missing_from_local:{item.affected_corridor_ref}")
        if not any(ref == "fpv.current" or ref.startswith("fpv.history_") for ref in item.evidence_refs):
            errors.append(f"traffic_rule_missing_fpv_evidence:{item.control_ref}")


def _validate_fixed_labels(
    target: ReasoningTargetV3,
    *,
    fixed_labels: Mapping[str, Any],
    provenance: Mapping[str, LabelProvenanceEntryV3],
    errors: list[str],
) -> None:
    """Prevent composition from silently changing trusted deterministic labels."""

    expected_direction = str(fixed_labels.get("demonstrated_direction") or "unknown")
    direction_provenance = provenance.get("demonstrated_direction")
    if expected_direction != "unknown" and direction_provenance and direction_provenance.strength == "strong":
        expected_motion = {
            "left": "turn_left",
            "right": "turn_right",
            "straight": "go_straight",
            "stop": "stop",
        }[expected_direction]
        if target.reasoning.decision.motion != expected_motion:
            errors.append(f"strong_fixed_direction_changed:decision:{expected_motion}:{target.reasoning.decision.motion}")
        if target.finish.decision_type == "navigate" and target.finish.motion != expected_motion:
            errors.append(f"strong_fixed_direction_changed:finish:{expected_motion}:{target.finish.motion}")

    expected_context = str(fixed_labels.get("condition_direction") or "unknown")
    context_provenance = provenance.get("condition_direction")
    grounding = target.reasoning.navigation_context_grounding
    if expected_context != "unknown" and context_provenance and context_provenance.strength == "strong":
        if not grounding.applicable or grounding.fused_direction != expected_context:
            errors.append(f"strong_context_direction_changed:{expected_context}:{grounding.fused_direction}")

    expected_objective = str(fixed_labels.get("authoritative_local_objective") or "unknown")
    objective_provenance = provenance.get("authoritative_local_objective")
    if expected_objective != "unknown" and objective_provenance and objective_provenance.strength == "strong":
        task = target.reasoning.task_grounding
        if not task.applicable or task.local_objective != expected_objective:
            errors.append(f"strong_fixed_objective_changed:{expected_objective}:{task.local_objective}")

    expected_corridor = fixed_labels.get("demonstrated_corridor_ref")
    corridor_provenance = provenance.get("demonstrated_corridor_ref")
    if expected_corridor and corridor_provenance and corridor_provenance.strength == "strong":
        actual = {
            "local_traversability": target.reasoning.local_traversability.preferred_clear_corridor_ref,
            "decision": target.reasoning.decision.selected_corridor_ref,
        }
        for field_name, value in actual.items():
            if value != expected_corridor:
                errors.append(f"strong_fixed_corridor_changed:{field_name}:{expected_corridor}:{value}")


def _validate_provenance(
    target: ReasoningTargetV3,
    *,
    provenance: Mapping[str, LabelProvenanceEntryV3],
    dataset_source: str | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate required section provenance and dataset trust boundaries."""

    trace = target.reasoning
    required = {"scene_grounding", "local_traversability", "decision"}
    if trace.task_grounding.applicable or trace.task_grounding.caption_role == "scene_description":
        required.add("task_grounding")
    if trace.decision.rationale_status in {"supported", "partial"}:
        required.add("decision_rationale")
    if trace.navigation_context_grounding.applicable:
        required.add("navigation_context_grounding")
    if trace.interaction_and_rule_grounding.applicable:
        required.add("interaction_and_rule_grounding")
    for field_name in sorted(required - set(provenance)):
        warnings.append(f"section_missing_label_provenance:{field_name}")
    normalized_dataset = str(dataset_source or "").strip().lower().replace("_", "-")
    if normalized_dataset.startswith("frodobots"):
        forbidden_sources = {"gt_trajectory", "odometry", "validated_fpv_projection"}
        for field_name, item in provenance.items():
            if item.source in forbidden_sources and item.strength == "strong":
                errors.append(f"frodobots_forbidden_strong_provenance:{field_name}:{item.source}")


def _validate_section_mask(
    target: ReasoningTargetV3,
    *,
    mask: SectionLossMaskV3,
    source_mask: SectionLossMaskV3 | None,
    provenance: Mapping[str, LabelProvenanceEntryV3],
    errors: list[str],
) -> None:
    """Validate section applicability and prevent source-mask escalation."""

    trace = target.reasoning
    if source_mask is not None:
        for field_name, enabled in mask.model_dump(mode="python").items():
            if enabled and not getattr(source_mask, field_name):
                errors.append(f"section_loss_mask_escalated:{field_name}")
    if (
        mask.task_grounding
        and not trace.task_grounding.applicable
        and trace.task_grounding.caption_role != "scene_description"
    ):
        errors.append("task_grounding_loss_enabled_when_not_applicable")
    if mask.scene_grounding and not trace.scene_grounding.applicable:
        errors.append("scene_grounding_loss_enabled_when_not_applicable")
    if mask.local_traversability and not trace.local_traversability.applicable:
        errors.append("local_traversability_loss_enabled_when_not_applicable")
    if mask.navigation_context_grounding and not trace.navigation_context_grounding.applicable:
        errors.append("navigation_context_loss_enabled_when_not_applicable")
    if mask.interaction_and_rule_grounding and not trace.interaction_and_rule_grounding.applicable:
        errors.append("interaction_rule_loss_enabled_when_not_applicable")
    if mask.decision and "decision" not in provenance:
        errors.append("decision_loss_enabled_without_provenance")
    if mask.decision and "decision_rationale" not in provenance:
        errors.append("decision_loss_enabled_without_validated_rationale")


def _generated_refs(target: ReasoningTargetV3) -> set[str]:
    """Collect local ids generated by validated scene and affordance sections."""

    trace = target.reasoning
    return {
        *(item.entity_ref for item in trace.scene_grounding.entities),
        *(item.control_ref for item in trace.scene_grounding.traffic_controls),
        *(item.region_ref for item in trace.local_traversability.regions),
    }


def _span_supported(span: str, normalized_instruction: str) -> bool:
    """Return whether an exact normalized phrase occurs in model-visible text."""

    normalized_span = " ".join(str(span).casefold().split())
    return bool(normalized_span) and normalized_span in normalized_instruction


def _normalize_mask(value: SectionLossMaskV3 | Mapping[str, Any] | None) -> SectionLossMaskV3:
    """Normalize one typed or serialized section mask."""

    if isinstance(value, SectionLossMaskV3):
        return value
    return SectionLossMaskV3.model_validate(value or {})


def _normalize_catalog(entries: Iterable[Any]) -> dict[str, str]:
    """Normalize evidence catalog objects to ``ref -> modality``."""

    result: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, str):
            result[entry] = _modality_for_ref(entry)
            continue
        if isinstance(entry, Mapping):
            ref = entry.get("ref") or entry.get("id")
            if ref:
                result[str(ref)] = str(entry.get("modality") or _modality_for_ref(str(ref)))
            continue
        ref = getattr(entry, "ref", None)
        if ref:
            result[str(ref)] = str(getattr(entry, "modality", None) or _modality_for_ref(str(ref)))
    return result


def _normalize_provenance(
    values: Mapping[str, LabelProvenanceEntryV3 | Mapping[str, Any]],
    *,
    errors: list[str],
) -> dict[str, LabelProvenanceEntryV3]:
    """Normalize provenance entries and record malformed values."""

    result: dict[str, LabelProvenanceEntryV3] = {}
    for key, value in values.items():
        try:
            result[str(key)] = (
                value if isinstance(value, LabelProvenanceEntryV3) else LabelProvenanceEntryV3.model_validate(value)
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid_label_provenance:{key}:{exc}")
    return result


def _source_type_matches_modality(source_type: str, modality: str) -> bool:
    """Return whether a navigation source type matches catalog modality."""

    allowed = {
        "instruction": {"instruction"},
        "bev": {"bev"},
        "goal": {"goal", "goal_image"},
        "waypoint": {"waypoint"},
        "route": {"route"},
        "tool": {"tool", "tool_route"},
    }
    return modality in allowed.get(source_type, set())


def _modality_for_ref(ref: str) -> str:
    """Infer a fallback modality from a canonical evidence prefix."""

    prefix = ref.split(".", 1)[0]
    return {
        "instruction": "instruction",
        "fpv": "fpv",
        "corridor": "fpv",
        "region": "fpv",
        "bev": "bev",
        "goal": "goal",
        "waypoint": "waypoint",
        "route": "route",
        "tool": "tool",
        "control": "control",
        "track": "track",
        "crosswalk": "selected_path",
    }.get(prefix, "unknown")


__all__ = ["ReasoningValidationResultV3", "validate_reasoning_target_v3"]
