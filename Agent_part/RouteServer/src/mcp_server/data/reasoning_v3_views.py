"""Project accepted V3.3 annotations into Full-CoT training views."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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


@dataclass(frozen=True)
class ReasoningViewBuildConfigV3:
    """Configure canonical V3 Full-CoT export.

    Args:
        require_complete_full_cot_supervision: Require every section loss bit.
    """

    require_complete_full_cot_supervision: bool = False


def build_reasoning_training_views_v3(
    source: ReasoningSourceV3,
    annotation: ReasoningAnnotationV3,
    *,
    config: ReasoningViewBuildConfigV3 | None = None,
) -> list[ReasoningTrainingViewV3]:
    """Build provenance-gated Full-CoT views for one accepted annotation.

    Args:
        source: Frozen V3.3 source.
        annotation: Matching accepted observation/composition annotation.
        config: Full-CoT export and completeness policy.

    Returns:
        Zero or one typed Full-CoT training view.

    Raises:
        ValueError: If source binding, composed masks, or target validation fails.
    """

    cfg = config or ReasoningViewBuildConfigV3()
    if annotation.status != "accepted" or annotation.target is None:
        return []
    model_input = source.model_input
    if annotation.source_group_id != model_input.source_group_id:
        raise ValueError("V3 annotation/source id mismatch")
    if annotation.source_fingerprint != source.source_fingerprint:
        raise ValueError("V3 annotation/source fingerprint mismatch")

    target = annotation.target
    provenance = _effective_provenance(source, annotation)
    if annotation.section_loss_mask is None:
        raise ValueError("accepted V3 annotation is missing composed section loss mask")
    section_mask = _effective_section_mask(annotation.section_loss_mask, target=target)
    readout = SYSTEM1_READOUT_TOKENS if model_input.system1_enabled else ()
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
        raise ValueError(f"accepted V3 annotation failed view validation: {validation.errors}")

    common: dict[str, Any] = {
        "source_group_id": model_input.source_group_id,
        "dataset_source": model_input.dataset_source,
        "travel_mode": model_input.travel_mode,
        "images": model_input.images,
        "image_roles": model_input.image_roles,
        "available_modalities": model_input.available_modalities,
        "instruction": model_input.instruction,
        "model_visible_context": model_input.model_visible_context,
        "evidence_catalog": [item.model_dump(mode="json", exclude_none=True) for item in model_input.evidence_catalog],
        "label_provenance": provenance,
        "prompt_profile_id": STUDENT_FULL_COT_PROFILE_V3.profile_id,
        "prompt_policy_digest": STUDENT_FULL_COT_PROFILE_V3.policy_digest,
    }
    outputs: list[ReasoningTrainingViewV3] = []
    full_cot_eligible = (
        not cfg.require_complete_full_cot_supervision or _all_sections_enabled(section_mask)
    )
    # V3.3 只导出 canonical Full CoT；VQA 必须由后续独立 typed profile 生成。
    if full_cot_eligible:
        outputs.append(
            ReasoningTrainingViewV3(
                sample_id=f"{model_input.source_group_id}:full_cot_v3_3",
                view_type="full_cot",
                task_family="grounded_navigation",
                task_type="reasoning_v3_3_full_cot",
                assistant_text=assistant_text,
                reasoning_target=target,
                trajectory_ego_t=model_input.trajectory_ego_t if model_input.system1_enabled else None,
                trajectory_policy=model_input.trajectory_policy if model_input.system1_enabled else None,
                action_mask=model_input.action_mask if model_input.system1_enabled else None,
                loss_mask={
                    "lm": True,
                    "system1_action": model_input.system1_enabled,
                    "trajectory": False,
                    "segmentation": False,
                },
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
                    "rationale_prompt_version": annotation.metadata.get("rationale_prompt_version"),
                    "rationale_prompt_policy_digest": annotation.metadata.get(
                        "rationale_prompt_policy_digest"
                    ),
                    "decision_rationale_status": target.reasoning.decision.rationale_status,
                    "fewshot_registry_version": annotation.metadata.get("fewshot_registry_version"),
                },
                **common,
            )
        )

    return outputs


def export_reasoning_training_views_v3(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    config: ReasoningViewBuildConfigV3 | None = None,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Join frozen V3.3 sources with latest append-only annotations.

    Args:
        source_jsonl: Frozen source manifest.
        annotation_jsonl: Append-only annotation manifest.
        output_jsonl: Destination Full-CoT view manifest.
        config: Full-CoT export policy.
        limit: Optional maximum source rows.
        skip_errors: Continue after invalid accepted annotations.

    Returns:
        Export counts and bounded validation errors.

    Raises:
        ValueError: If an accepted annotation is invalid and ``skip_errors``
            is false.
    """

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
        if annotation.status != "accepted":
            counters[f"annotation_{annotation.status}"] += 1
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
        for view in views:
            outputs.append(view.model_dump(mode="json", exclude_none=True))
            counters[f"view:{view.view_type}"] += 1
            counters[f"task:{view.task_type}"] += 1
            counters["system1_views" if view.loss_mask.get("system1_action") else "lm_only_views"] += 1
            if (
                view.reasoning_target is not None
                and view.reasoning_target.reasoning.navigation_context_grounding.fused_execution_status
                == "insufficient_evidence"
            ):
                counters["insufficient_evidence_views"] += 1
        counters["accepted_sources"] += 1
    write_jsonl_atomic(output_jsonl, outputs)
    return {
        "schema_version": "urbannav_reasoning_training_view_v3_3_report_v1",
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "output_jsonl": str(output_jsonl),
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
    if annotation.target is not None and annotation.target.reasoning.local_traversability.applicable:
        result.setdefault(
            "local_traversability",
            LabelProvenanceEntryV3(
                source="validated_lvlm_teacher",
                strength="weak",
                generator_version="reasoning_v3_3_teacher_v1",
                confidence=_confidence_score(annotation.target.reasoning.local_traversability.confidence),
            ),
        )
    return result


def _effective_section_mask(
    source_mask: SectionLossMaskV3,
    *,
    target: Any,
) -> SectionLossMaskV3:
    trace = target.reasoning
    return source_mask.model_copy(
        update={
            "task_grounding": (
                source_mask.task_grounding
                and (
                    trace.task_grounding.applicable
                    or trace.task_grounding.caption_role == "scene_description"
                )
            ),
            "scene_grounding": source_mask.scene_grounding and trace.scene_grounding.applicable,
            "local_traversability": source_mask.local_traversability and trace.local_traversability.applicable,
            "navigation_context_grounding": (
                source_mask.navigation_context_grounding
                and trace.navigation_context_grounding.applicable
            ),
            "interaction_and_rule_grounding": (
                source_mask.interaction_and_rule_grounding
                and trace.interaction_and_rule_grounding.applicable
            ),
        }
    )


def _all_sections_enabled(section_mask: SectionLossMaskV3) -> bool:
    """Return whether all canonical Full-CoT sections carry loss."""

    return all(section_mask.model_dump(mode="python").values())


def _confidence_score(value: str) -> float:
    """Map controlled confidence to a provenance score."""

    return {"high": 0.9, "medium": 0.75, "low": 0.5, "unknown": 0.0}.get(str(value), 0.0)


__all__ = [
    "ReasoningViewBuildConfigV3",
    "build_reasoning_training_views_v3",
    "export_reasoning_training_views_v3",
]
