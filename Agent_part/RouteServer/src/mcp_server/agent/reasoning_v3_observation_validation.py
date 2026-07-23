"""Semantic validators for the two-pass Reasoning V3.4 weak-label pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from mcp_server.contracts.reasoning_v3_observation import (
    DecisionRationaleObservationV3,
    ReasoningObservationV3,
)


@dataclass(frozen=True)
class ObservationValidationResultV3:
    """Hard errors and non-fatal audit warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether no hard validation error was found."""

        return not self.errors


def validate_reasoning_observation_v3(
    observation: ReasoningObservationV3,
    *,
    instruction: str,
    evidence_catalog: Iterable[Any],
    available_modalities: Iterable[str] = (),
    navigation_context_source_refs: Iterable[str] = (),
) -> ObservationValidationResultV3:
    """Validate Pass-1 labels against student-visible evidence only."""

    del available_modalities
    catalog = _normalize_catalog(evidence_catalog)
    catalog_refs = set(catalog)
    fpv_refs = {ref for ref, modality in catalog.items() if modality == "fpv"}
    errors: list[str] = []
    warnings: list[str] = []

    if any(ref.startswith("track.") or modality == "track" for ref, modality in catalog.items()):
        errors.append("v3_catalog_forbids_track_evidence")

    normalized_instruction = _normalize_text(instruction)
    task = observation.task_semantic
    for index, requirement in enumerate(task.requirements):
        if not _span_supported(requirement.text_span, normalized_instruction):
            errors.append(f"task_requirement_span_not_in_instruction:{index}")
    for index, referent in enumerate(task.referents):
        if not _span_supported(referent.phrase, normalized_instruction):
            errors.append(f"task_referent_phrase_not_in_instruction:{index}")
    for index, ambiguity in enumerate(task.ambiguities):
        if ambiguity.text_span and not _span_supported(
            ambiguity.text_span,
            normalized_instruction,
        ):
            errors.append(f"task_ambiguity_span_not_in_instruction:{index}")

    scene = observation.scene_grounding
    local = observation.local_traversability
    if fpv_refs and not scene.applicable:
        errors.append("scene_grounding_must_be_applicable_when_fpv_exists")
    if fpv_refs and not local.applicable:
        errors.append("local_traversability_must_include_unknown_region_when_fpv_exists")

    entity_refs = {item.entity_ref for item in scene.entities}
    control_refs = {item.control_ref for item in scene.traffic_controls}
    region_refs = {item.region_ref for item in local.regions}
    local_refs = entity_refs | control_refs | region_refs
    allowed_refs = catalog_refs | local_refs

    for index, entity in enumerate(scene.entities):
        _validate_visible_claim(
            f"scene_entity:{index}",
            entity.evidence_refs,
            catalog_refs=catalog_refs,
            fpv_refs=fpv_refs,
            errors=errors,
        )
    for index, control in enumerate(scene.traffic_controls):
        _validate_visible_claim(
            f"traffic_control:{index}",
            control.evidence_refs,
            catalog_refs=catalog_refs,
            fpv_refs=fpv_refs,
            errors=errors,
        )
    for index, region in enumerate(local.regions):
        unknown = set(region.evidence_refs) - allowed_refs
        if unknown:
            errors.append(f"traversability_unknown_evidence:{index}:{sorted(unknown)}")
        if not (set(region.evidence_refs) & fpv_refs):
            errors.append(f"traversability_requires_fpv_evidence:{index}")
        missing_blockers = set(region.blocker_refs) - entity_refs
        if missing_blockers:
            errors.append(f"traversability_unknown_blocker:{index}:{sorted(missing_blockers)}")

    _validate_task_refs(
        observation,
        allowed_refs=allowed_refs,
        errors=errors,
    )

    expected_context_refs = {str(ref) for ref in navigation_context_source_refs}
    navigation = observation.navigation_context_grounding
    observed_context_refs = {item.source_ref for item in navigation.source_groundings}
    if observed_context_refs != expected_context_refs:
        errors.append(
            "navigation_context_source_coverage_mismatch:"
            f"expected={sorted(expected_context_refs)}:"
            f"actual={sorted(observed_context_refs)}"
        )
    for index, grounding in enumerate(navigation.source_groundings):
        unknown = set(grounding.evidence_refs) - catalog_refs
        if unknown:
            errors.append(f"navigation_context_unknown_evidence:{index}:{sorted(unknown)}")
        if grounding.usage == "used" and grounding.relation is None:
            errors.append(f"used_navigation_context_requires_relation:{index}")
        if grounding.usage == "ignored" and grounding.relation is not None:
            errors.append(f"ignored_navigation_context_cannot_claim_relation:{index}")

    if task.applicable and "instruction.active" not in catalog_refs:
        errors.append("applicable_task_requires_visible_instruction_ref")
    if not task.requirements and not task.referents and not task.ambiguities:
        warnings.append("task_semantic_contains_no_extracted_claim")
    return ObservationValidationResultV3(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def validate_decision_rationale_v3(
    rationale: DecisionRationaleObservationV3,
    *,
    expected_outcome: Mapping[str, Any],
    observation: ReasoningObservationV3,
    evidence_catalog: Iterable[Any],
    fixed_labels: Mapping[str, Any] | None = None,
) -> ObservationValidationResultV3:
    """Validate direction identity and visible interaction/rationale evidence."""

    del fixed_labels
    errors: list[str] = []
    warnings: list[str] = []
    expected_identity = (
        str(expected_outcome.get("profile_id") or ""),
        str(expected_outcome.get("profile_version") or ""),
        int(expected_outcome.get("label_index", -1)),
        str(expected_outcome.get("label_name") or ""),
        expected_outcome.get("choice"),
        expected_outcome.get("angle_deg"),
        str(expected_outcome.get("source") or ""),
    )
    actual_identity = (
        rationale.direction_profile_id,
        rationale.direction_profile_version,
        rationale.direction_label_index,
        rationale.direction_label_name,
        rationale.direction_choice,
        rationale.direction_angle_deg,
        rationale.direction_source,
    )
    if actual_identity != expected_identity:
        errors.append(
            "rationale_direction_identity_mismatch:"
            f"expected={expected_identity!r}:actual={actual_identity!r}"
        )

    catalog = _normalize_catalog(evidence_catalog)
    catalog_refs = set(catalog)
    fpv_refs = {ref for ref, modality in catalog.items() if modality == "fpv"}
    scene = observation.scene_grounding
    local = observation.local_traversability
    entity_refs = {item.entity_ref for item in scene.entities}
    control_refs = {item.control_ref for item in scene.traffic_controls}
    region_refs = {item.region_ref for item in local.regions}
    allowed_refs = catalog_refs | entity_refs | control_refs | region_refs

    interaction_group = rationale.interaction_and_rule_grounding
    for index, interaction in enumerate(interaction_group.interactions):
        if interaction.actor_ref not in entity_refs:
            errors.append(f"interaction_unknown_actor:{index}:{interaction.actor_ref}")
        refs = set(interaction.evidence_refs)
        unknown = refs - allowed_refs
        if unknown:
            errors.append(f"interaction_unknown_evidence:{index}:{sorted(unknown)}")
        visible_fpv = refs & fpv_refs
        if not visible_fpv:
            errors.append(f"interaction_requires_fpv_evidence:{index}")
        if interaction.temporal_basis == "multi_frame" and len(visible_fpv) < 2:
            errors.append(f"multi_frame_interaction_requires_two_fpv_refs:{index}")
        if _looks_dynamic(interaction.relation) and (
            interaction.temporal_basis != "multi_frame" or len(visible_fpv) < 2
        ):
            errors.append(f"dynamic_interaction_requires_visible_multiframe_evidence:{index}")

    for index, rule in enumerate(interaction_group.traffic_rules):
        if rule.control_ref not in control_refs:
            errors.append(f"traffic_rule_unknown_control:{index}:{rule.control_ref}")
        refs = set(rule.evidence_refs)
        unknown = refs - allowed_refs
        if unknown:
            errors.append(f"traffic_rule_unknown_evidence:{index}:{sorted(unknown)}")
        if rule.control_ref not in refs:
            errors.append(f"traffic_rule_must_cite_control_ref:{index}")
        if not (refs & fpv_refs):
            errors.append(f"traffic_rule_requires_fpv_evidence:{index}")

    observation_refs = _observation_refs(observation, catalog_refs=catalog_refs)
    for index, support in enumerate(rationale.supporting_reasons):
        unknown = set(support.evidence_refs) - observation_refs
        if unknown:
            errors.append(f"decision_support_unknown_evidence:{index}:{sorted(unknown)}")
    if rationale.rationale_status == "unsupported":
        warnings.append("canonical_direction_has_no_visible_rationale_support")
    return ObservationValidationResultV3(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def _validate_task_refs(
    observation: ReasoningObservationV3,
    *,
    allowed_refs: set[str],
    errors: list[str],
) -> None:
    task = observation.task_semantic
    for prefix, items in (
        ("requirement", task.requirements),
        ("referent", task.referents),
        ("ambiguity", task.ambiguities),
    ):
        for index, item in enumerate(items):
            refs = set(item.evidence_refs)
            unknown = refs - allowed_refs
            if unknown:
                errors.append(f"task_{prefix}_unknown_evidence:{index}:{sorted(unknown)}")
            target_ref = getattr(item, "target_ref", None)
            resolved_ref = getattr(item, "resolved_ref", None)
            for value in (target_ref, resolved_ref):
                if value is not None and value not in allowed_refs:
                    errors.append(f"task_{prefix}_unknown_target:{index}:{value}")


def _validate_visible_claim(
    prefix: str,
    refs: Iterable[str],
    *,
    catalog_refs: set[str],
    fpv_refs: set[str],
    errors: list[str],
) -> None:
    values = set(refs)
    unknown = values - catalog_refs
    if unknown:
        errors.append(f"{prefix}:unknown_evidence:{sorted(unknown)}")
    if not (values & fpv_refs):
        errors.append(f"{prefix}:requires_fpv_evidence")


def _observation_refs(
    observation: ReasoningObservationV3,
    *,
    catalog_refs: set[str],
) -> set[str]:
    refs = set(catalog_refs)
    scene = observation.scene_grounding
    local = observation.local_traversability
    refs.update(item.entity_ref for item in scene.entities)
    refs.update(item.control_ref for item in scene.traffic_controls)
    refs.update(item.region_ref for item in local.regions)
    return refs


def _looks_dynamic(relation: str) -> bool:
    value = str(relation or "").lower()
    return any(
        token in value
        for token in (
            "crossing",
            "approaching",
            "moving",
            "receding",
            "overtaking",
            "same_direction",
        )
    )


def _normalize_catalog(entries: Iterable[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, Mapping):
            ref = str(entry.get("ref") or "")
            modality = str(entry.get("modality") or "")
        else:
            ref = str(getattr(entry, "ref", "") or "")
            modality = str(getattr(entry, "modality", "") or "")
        if ref:
            result[ref] = modality
    return result


def _span_supported(span: str, normalized_instruction: str) -> bool:
    normalized_span = _normalize_text(span)
    return bool(normalized_span and normalized_span in normalized_instruction)


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "ObservationValidationResultV3",
    "validate_decision_rationale_v3",
    "validate_reasoning_observation_v3",
]
