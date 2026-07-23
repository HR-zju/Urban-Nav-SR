"""Project accepted V3.4 annotations into minimal, provenance-gated views."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from mcp_server.agent.reasoning_v3_prompt_policy import STUDENT_FULL_COT_PROFILE_V3
from mcp_server.agent.reasoning_v3_text import render_reasoning_target_v3
from mcp_server.agent.reasoning_v3_validation import validate_reasoning_target_v3
from mcp_server.agent.stage2 import SYSTEM1_READOUT_TOKENS
from mcp_server.contracts.reasoning_dataset_v3 import (
    ReasoningAnnotationV3,
    ReasoningSourceV3,
    ReasoningTrainingViewV3,
)
from mcp_server.contracts.reasoning_v3 import LabelProvenanceEntryV3, SectionLossMaskV3
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic


REASONING_V3_TRAINING_POLICY_VERSION = "urbannav_reasoning_v3_4_minimal_training_policy_v1"
ReasoningTrainingProfileV3 = Literal["audit_only", "strong_action_only"]
DEFAULT_REASONING_V3_TRAINING_PROFILE: ReasoningTrainingProfileV3 = "audit_only"
REASONING_V3_TRAINING_PROFILES = ("audit_only", "strong_action_only")


@dataclass(frozen=True)
class ReasoningViewBuildConfigV3:
    """Configure the two safe V3.4 export profiles.

    ``audit_only`` emits a typed trace and available sidecars with every loss
    disabled. ``strong_action_only`` may enable only the independently
    supervised direction-choice and System1 sidecars. Full-CoT language and all
    six section masks remain disabled in both profiles.

    The two legacy booleans remain in the constructor only so old callers fail
    with an explicit migration error rather than silently recovering removed
    mixed-distillation semantics.
    """

    training_profile: ReasoningTrainingProfileV3 = DEFAULT_REASONING_V3_TRAINING_PROFILE
    require_complete_full_cot_supervision: bool = False
    allow_flagged_legacy_sources: bool = False

    def __post_init__(self) -> None:
        """Reject removed profiles and legacy section-loss switches."""

        if self.training_profile not in REASONING_V3_TRAINING_PROFILES:
            raise ValueError(f"unsupported Reasoning V3.4 training profile: {self.training_profile}")
        if self.require_complete_full_cot_supervision:
            raise ValueError(
                "require_complete_full_cot_supervision was removed in V3.4; "
                "reasoning section LM loss is always disabled"
            )
        if self.allow_flagged_legacy_sources:
            raise ValueError(
                "allow_flagged_legacy_sources was removed in V3.4; "
                "legacy mixed teacher distillation is not supported"
            )


def build_reasoning_training_views_v3(
    source: ReasoningSourceV3,
    annotation: ReasoningAnnotationV3,
    *,
    config: ReasoningViewBuildConfigV3 | None = None,
) -> list[ReasoningTrainingViewV3]:
    """Build zero or one minimal V3.4 Full-CoT view.

    Observation-only, rejected, and error annotations never produce training
    views. An accepted target remains audit text; only complete deterministic
    sidecars may receive loss under ``strong_action_only``.
    """

    cfg = config or ReasoningViewBuildConfigV3()
    if annotation.status != "accepted" or annotation.target is None:
        return []

    model_input = source.model_input
    if annotation.source_group_id != model_input.source_group_id:
        raise ValueError("V3.4 annotation/source id mismatch")
    if annotation.source_fingerprint != source.source_fingerprint:
        raise ValueError("V3.4 annotation/source fingerprint mismatch")
    if annotation.section_loss_mask is None:
        raise ValueError("accepted V3.4 annotation is missing weak-section availability")

    target = annotation.target
    provenance = _effective_provenance(source, annotation)
    _reject_weak_lm_provenance(provenance)
    section_mask = SectionLossMaskV3()
    system1_available = _system1_available(source)
    readout = SYSTEM1_READOUT_TOKENS if system1_available else ()
    assistant_text = render_reasoning_target_v3(target, readout_tokens=readout)
    validation = validate_reasoning_target_v3(
        target,
        instruction=model_input.instruction,
        evidence_catalog=model_input.evidence_catalog,
        available_modalities=model_input.available_modalities,
        label_provenance=provenance,
        section_loss_mask=section_mask,
        source_section_loss_mask=source.section_loss_mask,
        fixed_labels=source.teacher_supervision_context.get("fixed_labels") or {},
        dataset_source=model_input.dataset_source,
        assistant_text=assistant_text,
    )
    if not validation.valid:
        raise ValueError(f"accepted V3.4 annotation failed view validation: {validation.errors}")

    direction = _canonical_direction_sidecar(source, target=target, provenance=provenance)
    direction_available = direction is not None
    supervision_flags = _source_supervision_flags(source)
    if cfg.training_profile == "strong_action_only" and supervision_flags:
        # A direction- or trajectory-derived instruction makes the requested
        # action trivially recoverable from text.  Keep such rows for audit,
        # but never expose either strong action channel to training.
        return []
    if cfg.training_profile == "strong_action_only" and not (
        direction_available or system1_available
    ):
        return []

    enable_strong = cfg.training_profile == "strong_action_only"
    loss_mask = {
        "lm": False,
        "direction_choice": bool(enable_strong and direction_available),
        "system1_action": bool(enable_strong and system1_available),
        "trajectory": False,
        "segmentation": False,
    }
    common: dict[str, Any] = {
        "source_group_id": model_input.source_group_id,
        "dataset_source": model_input.dataset_source,
        "travel_mode": model_input.travel_mode,
        "images": model_input.images,
        "image_roles": model_input.image_roles,
        "available_modalities": model_input.available_modalities,
        "instruction": model_input.instruction,
        "model_visible_context": model_input.model_visible_context,
        "evidence_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in model_input.evidence_catalog
        ],
        "label_provenance": provenance,
        "prompt_profile_id": STUDENT_FULL_COT_PROFILE_V3.profile_id,
        "prompt_policy_digest": STUDENT_FULL_COT_PROFILE_V3.policy_digest,
    }
    direction_fields = direction or {}
    view = ReasoningTrainingViewV3(
        sample_id=f"{model_input.source_group_id}:full_cot_v3_4",
        view_type="full_cot",
        task_family="grounded_navigation",
        task_type="reasoning_v3_4_full_cot",
        assistant_text=assistant_text,
        reasoning_target=target,
        trajectory_ego_t=model_input.trajectory_ego_t if system1_available else None,
        trajectory_policy=model_input.trajectory_policy if system1_available else None,
        action_mask=model_input.action_mask if system1_available else None,
        loss_mask=loss_mask,
        section_loss_mask=section_mask,
        metadata={
            "annotation_schema_version": annotation.schema_version,
            "source_fingerprint": source.source_fingerprint,
            "teacher_provider": annotation.teacher_provider,
            "teacher_model": annotation.teacher_model,
            "validation_warnings": validation.warnings,
            "readout_token_count": len(readout),
            "composer_version": annotation.metadata.get("composer_version"),
            "teacher_prompt_version": annotation.metadata.get("teacher_prompt_version"),
            "teacher_prompt_policy_digest": annotation.metadata.get(
                "teacher_prompt_policy_digest"
            ),
            "rationale_prompt_version": annotation.metadata.get("rationale_prompt_version"),
            "rationale_prompt_policy_digest": annotation.metadata.get(
                "rationale_prompt_policy_digest"
            ),
            "fewshot_registry_version": annotation.metadata.get("fewshot_registry_version"),
            "decision_rationale_status": target.reasoning.decision.rationale_status,
            "reasoning_training_policy_version": REASONING_V3_TRAINING_POLICY_VERSION,
            "reasoning_training_profile": cfg.training_profile,
            "reasoning_text_training_role": "audit_only",
            "direction_training_role": _sidecar_role(
                available=direction_available,
                enabled=loss_mask["direction_choice"],
            ),
            "system1_training_role": _sidecar_role(
                available=system1_available,
                enabled=loss_mask["system1_action"],
            ),
            "weak_section_availability": annotation.section_loss_mask.model_dump(
                mode="json"
            ),
            "p0_supervision_flags": list(supervision_flags),
        },
        **direction_fields,
        **common,
    )
    return [view]


def export_reasoning_training_views_v3(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    config: ReasoningViewBuildConfigV3 | None = None,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Join frozen V3.4 sources with their latest annotations."""

    cfg = config or ReasoningViewBuildConfigV3()
    annotations = _latest_annotations_v3(annotation_jsonl)
    outputs: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(source_jsonl)):
        if limit is not None and index >= int(limit):
            break
        counters["source_rows"] += 1
        source = ReasoningSourceV3.model_validate(row)
        dataset = source.model_input.dataset_source
        counters[f"dataset:{dataset}"] += 1
        annotation = annotations.get(source.model_input.source_group_id)
        if annotation is None:
            counters["missing_annotation"] += 1
            continue
        counters[f"annotation:{annotation.status}"] += 1
        if annotation.status != "accepted":
            if annotation.status == "observation_only":
                counters["observation_only_without_view"] += 1
            continue
        try:
            views = build_reasoning_training_views_v3(source, annotation, config=cfg)
        except ValueError as exc:
            counters["invalid_accepted_annotation"] += 1
            if len(errors) < 100:
                errors.append(
                    {
                        "source_group_id": source.model_input.source_group_id,
                        "error": str(exc),
                    }
                )
            if not skip_errors:
                raise
            continue
        if not views:
            counters["filtered_without_strong_sidecar"] += 1
        for view in views:
            outputs.append(view.model_dump(mode="json", exclude_none=True))
            counters[f"view:{view.view_type}"] += 1
            counters[f"task:{view.task_type}"] += 1
            counters[f"training_profile:{cfg.training_profile}"] += 1
            counters["audit_only_views"] += int(cfg.training_profile == "audit_only")
            counters["direction_sidecar_views"] += int(view.direction_profile_id is not None)
            counters["system1_sidecar_views"] += int(view.trajectory_ego_t is not None)
            counters["direction_choice_loss_views"] += int(
                bool(view.loss_mask.get("direction_choice"))
            )
            counters["system1_action_loss_views"] += int(
                bool(view.loss_mask.get("system1_action"))
            )
        counters["accepted_sources"] += 1
    write_jsonl_atomic(output_jsonl, outputs)
    return {
        "schema_version": "urbannav_reasoning_training_view_v3_4_report_v1",
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "output_jsonl": str(output_jsonl),
        "reasoning_training_policy_version": REASONING_V3_TRAINING_POLICY_VERSION,
        "reasoning_training_profile": cfg.training_profile,
        "counts": dict(counters),
        "errors": errors,
    }


def _latest_annotations_v3(path: str | Path) -> dict[str, ReasoningAnnotationV3]:
    result: dict[str, ReasoningAnnotationV3] = {}
    for row in iter_jsonl(path):
        annotation = ReasoningAnnotationV3.model_validate(row)
        result[annotation.source_group_id] = annotation
    return result


def _effective_provenance(
    source: ReasoningSourceV3,
    annotation: ReasoningAnnotationV3,
) -> dict[str, LabelProvenanceEntryV3]:
    """Merge typed provenance without inventing confidence or label strength."""

    result = dict(source.label_provenance)
    result.update(annotation.label_provenance)
    annotation_values = annotation.metadata.get("label_provenance")
    if isinstance(annotation_values, Mapping):
        for key, value in annotation_values.items():
            result[str(key)] = (
                value
                if isinstance(value, LabelProvenanceEntryV3)
                else LabelProvenanceEntryV3.model_validate(value)
            )
    return result


def _canonical_direction_sidecar(
    source: ReasoningSourceV3,
    *,
    target: Any,
    provenance: Mapping[str, LabelProvenanceEntryV3],
) -> dict[str, Any] | None:
    """Return a direction sidecar only for a complete strong adapter label."""

    fixed_labels = source.teacher_supervision_context.get("fixed_labels")
    canonical = (
        fixed_labels.get("canonical_direction")
        if isinstance(fixed_labels, Mapping)
        else None
    )
    entry = provenance.get("decision.direction")
    if not isinstance(canonical, Mapping) or entry is None:
        return None
    if (
        entry.source_kind != "deterministic_adapter"
        or entry.supervision != "strong"
        or not entry.student_inferable
        or entry.loss_channel != "direction_choice"
    ):
        return None
    decision = target.reasoning.decision
    if str(entry.source) != str(decision.source):
        return None
    return {
        "direction_profile_id": decision.profile_id,
        "direction_profile_version": decision.profile_version,
        "direction_label_index": decision.label_index,
        "direction_label_name": decision.label_name,
        "direction_choice": decision.choice,
        "direction_angle_deg": decision.angle_deg,
        "direction_source": decision.source,
    }


def _reject_weak_lm_provenance(
    provenance: Mapping[str, LabelProvenanceEntryV3],
) -> None:
    """Reject removed weak-language loss routing at the export boundary."""

    invalid = sorted(
        field_path
        for field_path, entry in provenance.items()
        if entry.supervision == "weak" and entry.loss_channel != "none"
    )
    if invalid:
        raise ValueError(f"V3.4 weak labels must be audit-only: {invalid}")


def _system1_available(source: ReasoningSourceV3) -> bool:
    """Return whether the continuous sidecar has strong dataset provenance."""

    entry = source.label_provenance.get("system1.trajectory")
    return bool(
        source.model_input.system1_enabled
        and entry is not None
        and entry.source_kind == "dataset_gt"
        and entry.supervision == "strong"
        and entry.student_inferable
        and entry.loss_channel == "system1_action"
    )


def _sidecar_role(*, available: bool, enabled: bool) -> str:
    if enabled:
        return "primary"
    if available:
        return "audit_only"
    return "disabled"


def _source_supervision_flags(source: ReasoningSourceV3) -> tuple[str, ...]:
    """Read conservative source warnings without turning them into loss gates."""

    raw = source.teacher_supervision_context.get("p0_supervision_flags")
    if not isinstance(raw, (list, tuple, set)):
        return ()
    return tuple(sorted({str(value) for value in raw if str(value).strip()}))


__all__ = [
    "DEFAULT_REASONING_V3_TRAINING_PROFILE",
    "REASONING_V3_TRAINING_POLICY_VERSION",
    "REASONING_V3_TRAINING_PROFILES",
    "ReasoningViewBuildConfigV3",
    "ReasoningTrainingProfileV3",
    "build_reasoning_training_views_v3",
    "export_reasoning_training_views_v3",
]
