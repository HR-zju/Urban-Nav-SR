"""Structural and provenance audit for minimal Reasoning V3.4 artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from mcp_server.agent.reasoning_v3_observation_validation import (
    validate_decision_rationale_v3,
    validate_reasoning_observation_v3,
)
from mcp_server.agent.reasoning_v3_prompt_policy import (
    STUDENT_FULL_COT_PROFILE_V3,
    TEACHER_DECISION_RATIONALE_PROFILE_V3,
    TEACHER_OBSERVATION_PROFILE_V3,
)
from mcp_server.agent.reasoning_v3_validation import validate_reasoning_target_v3
from mcp_server.contracts.reasoning_dataset_v3 import (
    REASONING_ANNOTATION_V3_SCHEMA_VERSION,
    REASONING_SOURCE_V3_SCHEMA_VERSION,
    REASONING_TRAINING_VIEW_V3_SCHEMA_VERSION,
    ReasoningAnnotationV3,
    ReasoningSourceV3,
    ReasoningTrainingViewV3,
)
from mcp_server.contracts.reasoning_v3 import LabelProvenanceEntryV3
from mcp_server.data.jsonl_io import iter_jsonl
from mcp_server.data.reasoning_v3_composer import (
    REASONING_V3_COMPOSER_VERSION,
    compose_reasoning_target_v3,
    derive_decision_outcome_v3,
)
from mcp_server.data.reasoning_v3_conditions import build_navigation_context_sources_v3
from mcp_server.data.reasoning_v3_views import (
    REASONING_V3_TRAINING_POLICY_VERSION,
    REASONING_V3_TRAINING_PROFILES,
)
from mcp_server.teacher.reasoning_v3_fewshot import REASONING_V3_FEWSHOT_VERSION


REASONING_V3_AUDIT_REPORT_VERSION = "urbannav_reasoning_v3_4_acceptance_audit_v1"
_LOSS_MASK_KEYS = {
    "lm",
    "direction_choice",
    "system1_action",
    "trajectory",
    "segmentation",
}


def audit_reasoning_v3_artifacts(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    view_jsonl: str | Path,
    *,
    max_insufficient_fraction: float = 0.10,
    max_unsupported_rationale_fraction: float = 0.20,
    required_dataset_families: Iterable[str] = (),
) -> dict[str, Any]:
    """Audit schema, evidence closure, provenance, and sidecar loss routing.

    The two fraction arguments are retained for caller compatibility but no
    longer affect V3.4. They previously encoded removed fusion and rationale
    calibration gates. V3.4 reports weak-label availability without claiming
    calibration or geometric accuracy.
    """

    for name, value in {
        "max_insufficient_fraction": max_insufficient_fraction,
        "max_unsupported_rationale_fraction": max_unsupported_rationale_fraction,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    violations: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    sources: dict[str, ReasoningSourceV3] = {}
    for row in iter_jsonl(source_jsonl):
        source = ReasoningSourceV3.model_validate(row)
        source_id = source.model_input.source_group_id
        if source_id in sources:
            _violation(violations, "duplicate_source_id", source_id)
        sources[source_id] = source
        dataset = _dataset_family(source.model_input.dataset_source)
        counts["source_rows"] += 1
        counts[f"source_dataset:{dataset}"] += 1
        counts[f"source_schema:{source.schema_version}"] += 1
        _count_source_availability(source, counts)
        if _weak_lm_fields(source.label_provenance):
            _violation(violations, "source_has_weak_lm_provenance", source_id)

    annotations: dict[str, ReasoningAnnotationV3] = {}
    for row in iter_jsonl(annotation_jsonl):
        annotation = ReasoningAnnotationV3.model_validate(row)
        annotations[annotation.source_group_id] = annotation
        counts["annotation_records"] += 1
        counts[f"annotation_schema:{annotation.schema_version}"] += 1

    for source_id, annotation in annotations.items():
        counts[f"latest_annotation_status:{annotation.status}"] += 1
        source = sources.get(source_id)
        if source is None:
            _violation(violations, "annotation_missing_source", source_id)
            continue
        if annotation.source_fingerprint != source.source_fingerprint:
            _violation(violations, "annotation_source_fingerprint_mismatch", source_id)
            continue
        if annotation.status == "accepted":
            _count_provenance(annotation.label_provenance, counts, prefix="annotation")
        if _weak_lm_fields(annotation.label_provenance):
            _violation(violations, "annotation_has_weak_lm_provenance", source_id)
        if (
            annotation.status in {"accepted", "observation_only"}
            and annotation.observation is not None
        ):
            _audit_observation(source, annotation, violations=violations, counts=counts)
        if annotation.status == "observation_only":
            counts["observation_only_rows"] += 1
            continue
        if annotation.status != "accepted":
            counts["failed_annotation_rows"] += 1
            continue
        _audit_accepted_target(source, annotation, violations=violations, counts=counts)

    view_count_by_source: Counter[str] = Counter()
    for row in iter_jsonl(view_jsonl):
        view = ReasoningTrainingViewV3.model_validate(row)
        view_count_by_source[view.source_group_id] += 1
        _audit_view(
            view,
            source=sources.get(view.source_group_id),
            annotation=annotations.get(view.source_group_id),
            violations=violations,
            counts=counts,
        )

    for source_id, annotation in annotations.items():
        view_count = view_count_by_source[source_id]
        if annotation.status == "observation_only" and view_count:
            _violation(violations, "observation_only_annotation_has_training_view", source_id)
        if annotation.status == "accepted" and view_count != 1:
            _violation(
                violations,
                f"accepted_annotation_view_count_not_one:{view_count}",
                source_id,
            )

    missing_annotations = set(sources) - set(annotations)
    counts["missing_latest_annotations"] = len(missing_annotations)
    for source_id in sorted(missing_annotations):
        _violation(violations, "source_missing_latest_annotation", source_id)

    accepted = counts["accepted_target_rows"]
    observation_only = counts["observation_only_rows"]
    valid_annotations = accepted + observation_only
    source_rows = counts["source_rows"]
    availability = {
        "accepted_target_fraction": _fraction(accepted, source_rows),
        "observation_only_fraction": _fraction(observation_only, source_rows),
        "valid_annotation_fraction": _fraction(valid_annotations, source_rows),
        "direction_available_fraction": _fraction(
            counts["direction_available_rows"], source_rows
        ),
        "system1_available_fraction": _fraction(
            counts["system1_available_rows"], source_rows
        ),
        "weak_observation_fraction_among_valid_annotations": _fraction(
            counts["weak_observation_rows"], valid_annotations
        ),
        "weak_provenance_fraction_among_accepted_targets": _fraction(
            counts["annotation_rows_with_weak_provenance"], accepted
        ),
    }

    present = {
        key.split(":", 1)[1]
        for key in counts
        if key.startswith("source_dataset:") and counts[key] > 0
    }
    required = {_dataset_family(value) for value in required_dataset_families}
    for missing in sorted(required - present):
        _violation(violations, f"required_dataset_missing:{missing}", "global")

    return {
        "schema_version": REASONING_V3_AUDIT_REPORT_VERSION,
        "artifact_schema_versions": {
            "source": REASONING_SOURCE_V3_SCHEMA_VERSION,
            "annotation": REASONING_ANNOTATION_V3_SCHEMA_VERSION,
            "training_view": REASONING_TRAINING_VIEW_V3_SCHEMA_VERSION,
        },
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "view_jsonl": str(view_jsonl),
        "report_semantics": (
            "availability_only:no_calibration_projection_fusion_tracking_or_geometry_claim"
        ),
        "availability": availability,
        "counts": dict(sorted(counts.items())),
        "violation_count": len(violations),
        "violations": violations[:200],
        "passed": not violations,
    }


def _audit_observation(
    source: ReasoningSourceV3,
    annotation: ReasoningAnnotationV3,
    *,
    violations: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    """Validate and count one typed weak observation."""

    observation = annotation.observation
    if observation is None:
        return
    source_id = source.model_input.source_group_id
    context_sources = build_navigation_context_sources_v3(source)
    validation = validate_reasoning_observation_v3(
        observation,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        available_modalities=source.model_input.available_modalities,
        navigation_context_source_refs=[item.source_ref for item in context_sources],
    )
    for error in validation.errors:
        _violation(violations, f"observation_invalid:{error}", source_id)

    counts["weak_observation_rows"] += 1
    task = observation.task_semantic
    scene = observation.scene_grounding
    local = observation.local_traversability
    context = observation.navigation_context_grounding
    for name, applicable in {
        "task_grounding": task.applicable,
        "scene_grounding": scene.applicable,
        "local_traversability": local.applicable,
        "navigation_context_grounding": context.applicable,
    }.items():
        counts[f"weak_field_applicable:{name}"] += int(bool(applicable))
    counts["task_requirement_items"] += len(task.requirements)
    counts["task_referent_items"] += len(task.referents)
    counts["task_ambiguity_items"] += len(task.ambiguities)
    counts["scene_entity_items"] += len(scene.entities)
    counts["traffic_control_items"] += len(scene.traffic_controls)
    counts["traversability_region_items"] += len(local.regions)
    counts["navigation_source_grounding_items"] += len(context.source_groundings)


def _audit_accepted_target(
    source: ReasoningSourceV3,
    annotation: ReasoningAnnotationV3,
    *,
    violations: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    """Recompose and validate one accepted target without accuracy claims."""

    source_id = source.model_input.source_group_id
    if (
        annotation.target is None
        or annotation.observation is None
        or annotation.decision_rationale is None
        or annotation.section_loss_mask is None
    ):
        _violation(violations, "accepted_annotation_missing_required_partition", source_id)
        return
    try:
        outcome = derive_decision_outcome_v3(source, annotation.observation)
        rationale_validation = validate_decision_rationale_v3(
            annotation.decision_rationale,
            expected_outcome=outcome.model_dump(),
            observation=annotation.observation,
            evidence_catalog=source.model_input.evidence_catalog,
            fixed_labels=source.teacher_supervision_context.get("fixed_labels") or {},
        )
        for error in rationale_validation.errors:
            _violation(violations, f"accepted_rationale_invalid:{error}", source_id)
        composition = compose_reasoning_target_v3(
            source,
            annotation.observation,
            annotation.decision_rationale,
        )
    except (TypeError, ValueError) as exc:
        _violation(violations, f"accepted_annotation_recomposition_failed:{exc}", source_id)
        return

    if composition.target != annotation.target:
        _violation(violations, "accepted_target_not_deterministic_composition", source_id)
    if composition.section_loss_mask != annotation.section_loss_mask:
        _violation(violations, "accepted_weak_section_availability_drift", source_id)
    _audit_annotation_metadata(annotation, source_id=source_id, violations=violations)

    validation = validate_reasoning_target_v3(
        annotation.target,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        available_modalities=source.model_input.available_modalities,
        label_provenance={**source.label_provenance, **annotation.label_provenance},
        section_loss_mask=annotation.section_loss_mask,
        source_section_loss_mask=source.section_loss_mask,
        fixed_labels=source.teacher_supervision_context.get("fixed_labels") or {},
        dataset_source=source.model_input.dataset_source,
    )
    for error in validation.errors:
        _violation(violations, f"accepted_target_invalid:{error}", source_id)

    counts["accepted_target_rows"] += 1
    counts[
        f"weak_rationale_status:{annotation.decision_rationale.rationale_status}"
    ] += 1
    counts["weak_rationale_rows"] += 1
    counts["interaction_items"] += len(
        annotation.decision_rationale.interaction_and_rule_grounding.interactions
    )
    counts["traffic_rule_items"] += len(
        annotation.decision_rationale.interaction_and_rule_grounding.traffic_rules
    )
    counts["decision_support_items"] += len(
        annotation.decision_rationale.supporting_reasons
    )
    for name, available in annotation.section_loss_mask.model_dump(mode="python").items():
        counts[f"weak_section_available:{name}"] += int(bool(available))


def _audit_annotation_metadata(
    annotation: ReasoningAnnotationV3,
    *,
    source_id: str,
    violations: list[dict[str, Any]],
) -> None:
    """Check version provenance that binds the two teacher passes."""

    expected = {
        "composer_version": REASONING_V3_COMPOSER_VERSION,
        "teacher_prompt_version": TEACHER_OBSERVATION_PROFILE_V3.profile_id,
        "teacher_prompt_policy_digest": TEACHER_OBSERVATION_PROFILE_V3.policy_digest,
        "rationale_prompt_version": TEACHER_DECISION_RATIONALE_PROFILE_V3.profile_id,
        "rationale_prompt_policy_digest":
            TEACHER_DECISION_RATIONALE_PROFILE_V3.policy_digest,
        "fewshot_registry_version": REASONING_V3_FEWSHOT_VERSION,
    }
    for key, expected_value in expected.items():
        if annotation.metadata.get(key) != expected_value:
            _violation(violations, f"annotation_provenance_mismatch:{key}", source_id)


def _audit_view(
    view: ReasoningTrainingViewV3,
    *,
    source: ReasoningSourceV3 | None,
    annotation: ReasoningAnnotationV3 | None,
    violations: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    """Audit one view's zero-LM policy and independent strong sidecars."""

    counts["view_rows"] += 1
    counts[f"view_schema:{view.schema_version}"] += 1
    counts[f"view_dataset:{_dataset_family(view.dataset_source)}"] += 1
    counts[f"view_type:{view.view_type}"] += 1
    profile = str(view.metadata.get("reasoning_training_profile") or "")
    counts[f"view_training_profile:{profile or 'missing'}"] += 1

    if profile not in REASONING_V3_TRAINING_PROFILES:
        _violation(violations, "training_view_invalid_profile", view.sample_id)
    if (
        view.metadata.get("reasoning_training_policy_version")
        != REASONING_V3_TRAINING_POLICY_VERSION
    ):
        _violation(violations, "training_view_policy_version_mismatch", view.sample_id)
    if source is None or annotation is None or annotation.status != "accepted":
        _violation(
            violations,
            "training_view_missing_accepted_source_annotation",
            view.sample_id,
        )
        return
    if view.reasoning_target != annotation.target:
        _violation(violations, "training_view_target_annotation_mismatch", view.sample_id)
    expected_weak = (
        annotation.section_loss_mask.model_dump(mode="json")
        if annotation.section_loss_mask is not None
        else None
    )
    if view.metadata.get("weak_section_availability") != expected_weak:
        _violation(
            violations,
            "training_view_weak_section_availability_mismatch",
            view.sample_id,
        )
    if any(view.section_loss_mask.model_dump(mode="python").values()):
        _violation(violations, "training_view_has_reasoning_section_loss", view.sample_id)

    logical = view.loss_mask
    if set(logical) != _LOSS_MASK_KEYS:
        _violation(violations, "training_view_loss_mask_keys_mismatch", view.sample_id)
    for forbidden in ("lm", "trajectory", "segmentation"):
        if logical.get(forbidden):
            _violation(violations, f"training_view_forbidden_loss:{forbidden}", view.sample_id)

    direction_available = _direction_available(source)
    system1_available = _system1_available(source)
    direction_sidecar = view.direction_profile_id is not None
    system1_sidecar = view.trajectory_ego_t is not None
    counts["view_direction_sidecar_rows"] += int(direction_sidecar)
    counts["view_system1_sidecar_rows"] += int(system1_sidecar)
    counts["view_direction_choice_loss_rows"] += int(
        bool(logical.get("direction_choice"))
    )
    counts["view_system1_action_loss_rows"] += int(
        bool(logical.get("system1_action"))
    )
    if direction_sidecar != direction_available:
        _violation(violations, "training_view_direction_sidecar_availability_mismatch", view.sample_id)
    if system1_sidecar != system1_available:
        _violation(violations, "training_view_system1_sidecar_availability_mismatch", view.sample_id)

    if profile == "audit_only":
        if logical.get("direction_choice") or logical.get("system1_action"):
            _violation(violations, "audit_only_view_has_enabled_sidecar_loss", view.sample_id)
    elif profile == "strong_action_only":
        if bool(logical.get("direction_choice")) != direction_available:
            _violation(violations, "strong_view_direction_loss_mismatch", view.sample_id)
        if bool(logical.get("system1_action")) != system1_available:
            _violation(violations, "strong_view_system1_loss_mismatch", view.sample_id)
        if not logical.get("direction_choice") and not logical.get("system1_action"):
            _violation(violations, "strong_view_has_no_enabled_strong_sidecar", view.sample_id)

    if view.prompt_profile_id != STUDENT_FULL_COT_PROFILE_V3.profile_id:
        _violation(violations, "training_view_prompt_profile_mismatch", view.sample_id)
    if view.prompt_policy_digest != STUDENT_FULL_COT_PROFILE_V3.policy_digest:
        _violation(violations, "training_view_prompt_policy_digest_mismatch", view.sample_id)
    if view.task_type != "reasoning_v3_4_full_cot":
        _violation(violations, "training_view_task_type_mismatch", view.sample_id)


def _count_source_availability(
    source: ReasoningSourceV3,
    counts: Counter[str],
) -> None:
    """Count source evidence, provenance, and strong sidecar availability."""

    model_input = source.model_input
    counts["evidence_catalog_items"] += len(model_input.evidence_catalog)
    counts["rows_with_evidence_catalog"] += int(bool(model_input.evidence_catalog))
    for item in model_input.evidence_catalog:
        counts[f"evidence_modality:{item.modality}"] += 1
    direction_available = _direction_available(source)
    system1_available = _system1_available(source)
    counts["direction_available_rows"] += int(direction_available)
    counts["direction_unavailable_rows"] += int(not direction_available)
    counts["system1_available_rows"] += int(system1_available)
    counts["system1_unavailable_rows"] += int(not system1_available)
    counts["direction_and_system1_available_rows"] += int(
        direction_available and system1_available
    )
    _count_provenance(source.label_provenance, counts, prefix="source")


def _count_provenance(
    provenance: Mapping[str, LabelProvenanceEntryV3],
    counts: Counter[str],
    *,
    prefix: str,
) -> None:
    """Count typed provenance without converting it into an accuracy score."""

    if provenance:
        counts[f"{prefix}_rows_with_provenance"] += 1
    weak_present = False
    for field_path, entry in provenance.items():
        counts[f"{prefix}_provenance_items"] += 1
        counts[f"{prefix}_provenance_source_kind:{entry.source_kind}"] += 1
        counts[f"{prefix}_provenance_supervision:{entry.supervision}"] += 1
        counts[f"{prefix}_provenance_loss_channel:{entry.loss_channel}"] += 1
        if entry.supervision == "weak":
            weak_present = True
            counts[f"weak_provenance_field:{field_path}"] += 1
    counts[f"{prefix}_rows_with_weak_provenance"] += int(weak_present)


def _direction_available(source: ReasoningSourceV3) -> bool:
    """Return whether one complete strong canonical direction sidecar exists."""

    fixed = source.teacher_supervision_context.get("fixed_labels")
    canonical = fixed.get("canonical_direction") if isinstance(fixed, Mapping) else None
    provenance = source.label_provenance.get("decision.direction")
    return bool(
        isinstance(canonical, Mapping)
        and provenance is not None
        and provenance.source_kind == "deterministic_adapter"
        and provenance.supervision == "strong"
        and provenance.student_inferable
        and provenance.loss_channel == "direction_choice"
    )


def _system1_available(source: ReasoningSourceV3) -> bool:
    """Return whether System1 has an eligible strong dataset sidecar."""

    provenance = source.label_provenance.get("system1.trajectory")
    return bool(
        source.model_input.system1_enabled
        and provenance is not None
        and provenance.source_kind == "dataset_gt"
        and provenance.supervision == "strong"
        and provenance.student_inferable
        and provenance.loss_channel == "system1_action"
    )


def _weak_lm_fields(
    provenance: Mapping[str, LabelProvenanceEntryV3],
) -> list[str]:
    """Return weak fields that still advertise removed language loss."""

    return sorted(
        field_path
        for field_path, entry in provenance.items()
        if entry.supervision == "weak" and entry.loss_channel != "none"
    )


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _violation(violations: list[dict[str, Any]], code: str, sample_id: str) -> None:
    violations.append({"code": code, "sample_id": sample_id})


def _dataset_family(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    for family in ("sidewalk", "foxglove", "frodobots", "egowalk"):
        if normalized.startswith(family):
            return family
    return normalized or "unknown"


__all__ = ["REASONING_V3_AUDIT_REPORT_VERSION", "audit_reasoning_v3_artifacts"]
