"""End-to-end validation for the minimal Reasoning V3.4 target."""

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
    """Hard contract failures and non-fatal audit warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether the target passed every hard gate."""

        return not self.errors


def validate_reasoning_target_v3(
    target: ReasoningTargetV3,
    *,
    instruction: str,
    evidence_catalog: Iterable[Any],
    available_modalities: Iterable[str] = (),
    label_provenance: Mapping[str, LabelProvenanceEntryV3 | Mapping[str, Any]] | None = None,
    section_loss_mask: SectionLossMaskV3 | Mapping[str, Any] | None = None,
    source_section_loss_mask: SectionLossMaskV3 | Mapping[str, Any] | None = None,
    fixed_labels: Mapping[str, Any] | None = None,
    dataset_source: str = "",
    assistant_text: str | None = None,
) -> ReasoningValidationResultV3:
    """Validate evidence closure, provenance, and exact direction identity."""

    del available_modalities
    errors: list[str] = []
    warnings: list[str] = []
    catalog = _normalize_catalog(evidence_catalog)
    catalog_refs = set(catalog)
    fpv_refs = {ref for ref, modality in catalog.items() if modality == "fpv"}
    trace = target.reasoning
    entity_refs = {item.entity_ref for item in trace.scene_grounding.entities}
    control_refs = {item.control_ref for item in trace.scene_grounding.traffic_controls}
    region_refs = {item.region_ref for item in trace.local_traversability.regions}
    local_refs = entity_refs | control_refs | region_refs
    allowed_refs = catalog_refs | local_refs

    unknown_refs = reasoning_evidence_refs_v3(trace) - allowed_refs
    if unknown_refs:
        errors.append(f"reasoning_unknown_evidence_refs:{sorted(unknown_refs)}")
    if any(ref.startswith("track.") or modality == "track" for ref, modality in catalog.items()):
        errors.append("reasoning_v3_forbids_track_evidence")

    normalized_instruction = _normalize_text(instruction)
    for index, requirement in enumerate(trace.task_grounding.requirements):
        if not _span_supported(requirement.text_span, normalized_instruction):
            errors.append(f"task_requirement_span_not_in_instruction:{index}")
    for index, referent in enumerate(trace.task_grounding.referents):
        if not _span_supported(referent.phrase, normalized_instruction):
            errors.append(f"task_referent_phrase_not_in_instruction:{index}")

    for prefix, items in (
        ("scene_entity", trace.scene_grounding.entities),
        ("traffic_control", trace.scene_grounding.traffic_controls),
        ("traversability", trace.local_traversability.regions),
    ):
        for index, item in enumerate(items):
            refs = set(item.evidence_refs)
            if not (refs & fpv_refs):
                errors.append(f"{prefix}_requires_fpv_evidence:{index}")

    for index, interaction in enumerate(
        trace.interaction_and_rule_grounding.interactions
    ):
        refs = set(interaction.evidence_refs)
        visible_fpv = refs & fpv_refs
        if interaction.actor_ref not in entity_refs:
            errors.append(f"interaction_unknown_actor:{index}:{interaction.actor_ref}")
        if not visible_fpv:
            errors.append(f"interaction_requires_fpv_evidence:{index}")
        if interaction.temporal_basis == "multi_frame" and len(visible_fpv) < 2:
            errors.append(f"multi_frame_interaction_requires_two_fpv_refs:{index}")
        if _looks_dynamic(interaction.relation) and (
            interaction.temporal_basis != "multi_frame" or len(visible_fpv) < 2
        ):
            errors.append(f"dynamic_interaction_requires_visible_multiframe_evidence:{index}")
    for index, rule in enumerate(
        trace.interaction_and_rule_grounding.traffic_rules
    ):
        if rule.control_ref not in control_refs:
            errors.append(f"traffic_rule_unknown_control:{index}:{rule.control_ref}")
        if rule.control_ref not in set(rule.evidence_refs):
            errors.append(f"traffic_rule_must_cite_control_ref:{index}")

    fixed = dict(fixed_labels or {})
    canonical = fixed.get("canonical_direction")
    if not isinstance(canonical, Mapping):
        errors.append("full_cot_requires_canonical_direction_record")
    else:
        expected = (
            str(canonical.get("profile_id") or ""),
            str(canonical.get("profile_version") or ""),
            int(canonical.get("label_index", -1)),
            str(canonical.get("name") or ""),
            canonical.get("choice"),
            canonical.get("angle_deg"),
            str(canonical.get("source") or ""),
        )
        decision = trace.decision
        actual = (
            decision.profile_id,
            decision.profile_version,
            decision.label_index,
            decision.label_name,
            decision.choice,
            decision.angle_deg,
            decision.source,
        )
        if actual != expected:
            errors.append(
                "decision_does_not_copy_canonical_direction:"
                f"expected={expected!r}:actual={actual!r}"
            )

    provenance = _normalize_provenance(label_provenance or {}, errors=errors)
    action_provenance = provenance.get("decision.direction")
    if action_provenance is None:
        errors.append("canonical_direction_provenance_missing")
    elif (
        action_provenance.source_kind != "deterministic_adapter"
        or action_provenance.supervision != "strong"
        or not action_provenance.student_inferable
        or action_provenance.loss_channel != "direction_choice"
    ):
        errors.append("canonical_direction_provenance_is_not_strong_adapter_output")

    for field_path in (
        "task_grounding",
        "scene_grounding",
        "local_traversability",
        "navigation_context_grounding",
        "interaction_and_rule_grounding",
        "decision.rationale",
    ):
        entry = provenance.get(field_path)
        if entry is None:
            continue
        if (
            entry.source_kind != "lvlm_weak"
            or entry.supervision != "weak"
            or not entry.student_inferable
            or entry.loss_channel not in {"none", "weak_lm"}
        ):
            errors.append(f"weak_field_has_invalid_provenance:{field_path}")

    mask = _normalize_mask(section_loss_mask)
    source_mask = _normalize_mask(source_section_loss_mask)
    if mask.decision or source_mask.decision:
        errors.append("decision_language_loss_is_forbidden_in_v3_4")
    if trace.decision.rationale_status == "unsupported":
        warnings.append("canonical_direction_has_no_visible_rationale_support")

    normalized_dataset = str(dataset_source).lower().replace("_", "-")
    if normalized_dataset.startswith("frodobots"):
        errors.append("frodobots_cannot_emit_canonical_direction_full_cot")

    if assistant_text is not None:
        forbidden = (
            "instruction_type",
            "local_objective",
            "goal_distance",
            "route_theta",
            "selected_corridor",
            "preferred_clear_corridor",
            "local_risk",
            "confidence",
            "tracked_metadata",
            "instance_id",
            "track_id",
            "\"bbox\"",
            "\"mask\"",
            "\"ttc\"",
        )
        leaked = [token for token in forbidden if token in assistant_text]
        if leaked:
            errors.append(f"forbidden_v3_4_fields_in_assistant:{leaked}")
        parsed = parse_reasoning_target_v3(assistant_text)
        if not parsed.valid or parsed.target is None:
            errors.extend(f"assistant_text:{item}" for item in parsed.errors)
        elif parsed.target != target:
            errors.append("assistant_text_target_roundtrip_mismatch")
    return ReasoningValidationResultV3(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def _normalize_provenance(
    value: Mapping[str, LabelProvenanceEntryV3 | Mapping[str, Any]],
    *,
    errors: list[str],
) -> dict[str, LabelProvenanceEntryV3]:
    result: dict[str, LabelProvenanceEntryV3] = {}
    for key, raw in value.items():
        try:
            result[str(key)] = (
                raw
                if isinstance(raw, LabelProvenanceEntryV3)
                else LabelProvenanceEntryV3.model_validate(raw)
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid_label_provenance:{key}:{exc}")
    return result


def _normalize_mask(
    value: SectionLossMaskV3 | Mapping[str, Any] | None,
) -> SectionLossMaskV3:
    if value is None:
        return SectionLossMaskV3()
    if isinstance(value, SectionLossMaskV3):
        return value
    return SectionLossMaskV3.model_validate(value)


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


def _span_supported(span: str, normalized_instruction: str) -> bool:
    normalized_span = _normalize_text(span)
    return bool(normalized_span and normalized_span in normalized_instruction)


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "ReasoningValidationResultV3",
    "validate_reasoning_target_v3",
]
