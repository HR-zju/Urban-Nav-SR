"""Acceptance audit for source, annotation, and training-view Reasoning V3 artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

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
from mcp_server.data.jsonl_io import iter_jsonl
from mcp_server.data.reasoning_v3_composer import (
    REASONING_V3_COMPOSER_VERSION,
    compose_reasoning_target_v3,
    derive_decision_outcome_v3,
)
from mcp_server.data.reasoning_v3_conditions import build_navigation_context_sources_v3
from mcp_server.teacher.reasoning_v3_fewshot import REASONING_V3_FEWSHOT_VERSION


REASONING_V3_AUDIT_REPORT_VERSION = "urbannav_reasoning_v3_3_acceptance_audit_v1"


def audit_reasoning_v3_artifacts(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    view_jsonl: str | Path,
    *,
    max_insufficient_fraction: float = 0.10,
    max_unsupported_rationale_fraction: float = 0.20,
    required_dataset_families: Iterable[str] = (),
) -> dict[str, Any]:
    """Evaluate machine-checkable Full-CoT V3.3 acceptance conditions.

    Args:
        source_jsonl: Frozen V3.3 source manifest.
        annotation_jsonl: Append-only V3.3 annotation manifest.
        view_jsonl: Exported Full-CoT training views.
        max_insufficient_fraction: Maximum accepted hard-negative fraction.
        max_unsupported_rationale_fraction: Maximum accepted unsupported rationale fraction.
        required_dataset_families: Dataset families that must be represented.

    Returns:
        Deterministic audit report with counts and bounded violations.

    Raises:
        ValueError: If the configured fraction is invalid or any artifact row
            violates its typed schema.
    """

    if not 0.0 <= max_insufficient_fraction <= 1.0:
        raise ValueError("max_insufficient_fraction must be in [0, 1]")
    if not 0.0 <= max_unsupported_rationale_fraction <= 1.0:
        raise ValueError("max_unsupported_rationale_fraction must be in [0, 1]")
    violations: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    sources: dict[str, ReasoningSourceV3] = {}
    for row in iter_jsonl(source_jsonl):
        source = ReasoningSourceV3.model_validate(row)
        source_id = source.model_input.source_group_id
        sources[source_id] = source
        dataset = _dataset_family(source.model_input.dataset_source)
        counts["source_rows"] += 1
        counts[f"source_dataset:{dataset}"] += 1
        counts[f"source_schema:{source.schema_version}"] += 1
        if dataset in {"sidewalk", "foxglove", "egowalk"} and source.model_input.system1_enabled:
            counts["trajectory_qc_eligible_rows"] += 1
            provenance = source.label_provenance.get("demonstrated_direction")
            if provenance is not None and provenance.strength == "strong":
                counts["trajectory_qc_strong_direction_rows"] += 1
            else:
                _violation(violations, "trajectory_qc_missing_strong_direction", source_id)
        if dataset == "frodobots" and source.model_input.system1_enabled:
            _violation(violations, "frodobots_source_system1_enabled", source_id)

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
        if (
            annotation.status != "accepted"
            or annotation.target is None
            or annotation.observation is None
            or annotation.decision_rationale is None
        ):
            continue
        context_sources = build_navigation_context_sources_v3(source)
        observation_validation = validate_reasoning_observation_v3(
            annotation.observation,
            instruction=source.model_input.instruction,
            evidence_catalog=source.model_input.evidence_catalog,
            available_modalities=source.model_input.available_modalities,
            navigation_context_source_refs=[item.source_ref for item in context_sources],
        )
        for error in observation_validation.errors:
            _violation(violations, f"accepted_observation_invalid:{error}", source_id)
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
            continue
        if composition.target != annotation.target:
            _violation(violations, "accepted_annotation_target_not_deterministic_composition", source_id)
        if composition.section_loss_mask != annotation.section_loss_mask:
            _violation(violations, "accepted_annotation_section_mask_drift", source_id)
        if annotation.metadata.get("composer_version") != REASONING_V3_COMPOSER_VERSION:
            _violation(violations, "accepted_annotation_composer_version_mismatch", source_id)
        if annotation.metadata.get("teacher_prompt_version") != TEACHER_OBSERVATION_PROFILE_V3.profile_id:
            _violation(violations, "accepted_annotation_teacher_profile_mismatch", source_id)
        if annotation.metadata.get("teacher_prompt_policy_digest") != TEACHER_OBSERVATION_PROFILE_V3.policy_digest:
            _violation(violations, "accepted_annotation_teacher_policy_digest_mismatch", source_id)
        if annotation.metadata.get("rationale_prompt_version") != TEACHER_DECISION_RATIONALE_PROFILE_V3.profile_id:
            _violation(violations, "accepted_annotation_rationale_profile_mismatch", source_id)
        if (
            annotation.metadata.get("rationale_prompt_policy_digest")
            != TEACHER_DECISION_RATIONALE_PROFILE_V3.policy_digest
        ):
            _violation(violations, "accepted_annotation_rationale_policy_digest_mismatch", source_id)
        if annotation.metadata.get("fewshot_registry_version") != REASONING_V3_FEWSHOT_VERSION:
            _violation(violations, "accepted_annotation_fewshot_registry_mismatch", source_id)
        target = annotation.target
        validation = validate_reasoning_target_v3(
            target,
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
            _violation(violations, f"accepted_annotation_invalid:{error}", source_id)
        grounding = target.reasoning.navigation_context_grounding
        if grounding.applicable and not grounding.source_groundings:
            _violation(violations, "applicable_navigation_context_missing_real_source", source_id)
        expected_optional = {
            entry.ref
            for entry in source.model_input.evidence_catalog
            if entry.modality in {"bev", "goal", "goal_image", "waypoint", "route", "tool", "tool_route"}
        }
        observed_optional = {
            item.source_ref for item in target.reasoning.scene_grounding.context_observations
        }
        if observed_optional != expected_optional:
            _violation(violations, "optional_context_accounting_mismatch", source_id)
        if grounding.fused_execution_status == "insufficient_evidence":
            counts["accepted_insufficient_evidence_rows"] += 1
        counts[f"accepted_rationale_status:{target.reasoning.decision.rationale_status}"] += 1
        counts["accepted_annotation_rows"] += 1

    for row in iter_jsonl(view_jsonl):
        view = ReasoningTrainingViewV3.model_validate(row)
        counts["view_rows"] += 1
        counts[f"view_schema:{view.schema_version}"] += 1
        counts[f"view_dataset:{_dataset_family(view.dataset_source)}"] += 1
        counts[f"view_type:{view.view_type}"] += 1
        source = sources.get(view.source_group_id)
        annotation = annotations.get(view.source_group_id)
        if source is None or annotation is None or annotation.status != "accepted":
            _violation(violations, "training_view_missing_accepted_source_annotation", view.sample_id)
        else:
            if view.reasoning_target != annotation.target:
                _violation(violations, "training_view_target_annotation_mismatch", view.sample_id)
            if view.section_loss_mask != annotation.section_loss_mask:
                _violation(violations, "training_view_section_mask_annotation_mismatch", view.sample_id)
        if view.prompt_profile_id != STUDENT_FULL_COT_PROFILE_V3.profile_id:
            _violation(violations, "training_view_prompt_profile_mismatch", view.sample_id)
        if view.prompt_policy_digest != STUDENT_FULL_COT_PROFILE_V3.policy_digest:
            _violation(violations, "training_view_prompt_policy_digest_mismatch", view.sample_id)
        if _dataset_family(view.dataset_source) == "frodobots":
            if view.trajectory_ego_t is not None or view.loss_mask.get("system1_action"):
                _violation(violations, "frodobots_trajectory_loss_row", view.sample_id)
        if (
            view.reasoning_target is not None
            and view.reasoning_target.reasoning.navigation_context_grounding.fused_execution_status
            == "insufficient_evidence"
        ):
            counts["view_insufficient_evidence_rows"] += 1

    eligible = counts["trajectory_qc_eligible_rows"]
    strong = counts["trajectory_qc_strong_direction_rows"]
    coverage = strong / eligible if eligible else 1.0
    if coverage != 1.0:
        _violation(violations, "trajectory_qc_strong_direction_coverage_below_100_percent", "global")
    accepted = counts["accepted_annotation_rows"]
    insufficient_fraction = counts["accepted_insufficient_evidence_rows"] / accepted if accepted else 0.0
    if insufficient_fraction > max_insufficient_fraction:
        _violation(
            violations,
            f"insufficient_evidence_fraction_exceeds_limit:{insufficient_fraction:.6f}:{max_insufficient_fraction:.6f}",
            "global",
        )
    unsupported_fraction = (
        counts["accepted_rationale_status:unsupported"] / accepted if accepted else 0.0
    )
    if unsupported_fraction > max_unsupported_rationale_fraction:
        _violation(
            violations,
            (
                "unsupported_rationale_fraction_exceeds_limit:"
                f"{unsupported_fraction:.6f}:{max_unsupported_rationale_fraction:.6f}"
            ),
            "global",
        )
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
        "max_insufficient_fraction": max_insufficient_fraction,
        "max_unsupported_rationale_fraction": max_unsupported_rationale_fraction,
        "trajectory_qc_strong_direction_coverage": coverage,
        "insufficient_evidence_fraction": insufficient_fraction,
        "unsupported_rationale_fraction": unsupported_fraction,
        "counts": dict(sorted(counts.items())),
        "violation_count": len(violations),
        "violations": violations[:200],
        "passed": not violations,
    }

def _violation(violations: list[dict[str, Any]], code: str, sample_id: str) -> None:
    violations.append({"code": code, "sample_id": sample_id})


def _dataset_family(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    for family in ("sidewalk", "foxglove", "frodobots", "egowalk"):
        if normalized.startswith(family):
            return family
    return normalized or "unknown"


__all__ = ["REASONING_V3_AUDIT_REPORT_VERSION", "audit_reasoning_v3_artifacts"]
