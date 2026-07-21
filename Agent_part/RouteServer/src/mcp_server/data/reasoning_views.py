"""Project accepted V2 annotations into cross-repository training views."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_server.agent.reasoning_text import render_reasoning_target_v2
from mcp_server.agent.reasoning_validation import validate_reasoning_target_v2
from mcp_server.agent.stage2 import SYSTEM1_READOUT_TOKENS
from mcp_server.contracts.reasoning_dataset import (
    ReasoningAnnotationV2,
    ReasoningSourceV2,
    ReasoningTrainingViewV2,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.qa.reasoning_views import build_reasoning_qa_views


@dataclass(frozen=True)
class ReasoningViewBuildConfig:
    """Configure deterministic full-CoT and QA projection.

    Args:
        include_full_cot: Export one full reasoning view per accepted source.
        include_qa: Export eligible one-section QA views.
        minimum_qa_confidence: Minimum task-specific QA confidence.
        derive_observational_teacher_qa: Add explicitly marked VLM-distilled
            cross-view/social-observation QA gates. It never creates tracked
            motion or constraint-applicability ground truth.
    """

    include_full_cot: bool = True
    include_qa: bool = True
    minimum_qa_confidence: float = 0.7
    derive_observational_teacher_qa: bool = True


def build_reasoning_training_views(
    source: ReasoningSourceV2,
    annotation: ReasoningAnnotationV2,
    *,
    config: ReasoningViewBuildConfig | None = None,
) -> list[ReasoningTrainingViewV2]:
    """Build all trainable views for one accepted annotation.

    Args:
        source: Frozen model-visible source and optional System1 sidecars.
        annotation: Matching accepted teacher annotation.
        config: View projection policy.

    Returns:
        Full CoT and sparse QA views sharing ``source_group_id``.

    Raises:
        ValueError: If annotation status/fingerprint/grounding is invalid.
    """

    cfg = config or ReasoningViewBuildConfig()
    if annotation.status != "accepted" or annotation.target is None:
        return []
    if annotation.source_group_id != source.source_group_id:
        raise ValueError("annotation/source id mismatch")
    if annotation.source_fingerprint != source.source_fingerprint:
        raise ValueError("annotation/source fingerprint mismatch")

    target = annotation.target
    readout = SYSTEM1_READOUT_TOKENS if source.system1_enabled else ()
    assistant_text = render_reasoning_target_v2(target, readout_tokens=readout)
    validation = validate_reasoning_target_v2(
        target,
        evidence_catalog=[item.model_dump(mode="json") for item in source.evidence_catalog],
        available_modalities=source.available_modalities,
        travel_mode=source.travel_mode,
        assistant_text=assistant_text,
        require_navigation_readout=source.system1_enabled,
    )
    if not validation.valid:
        raise ValueError(f"accepted annotation failed view validation: {validation.errors}")

    views: list[ReasoningTrainingViewV2] = []
    common = {
        "source_group_id": source.source_group_id,
        "images": source.images,
        "image_roles": source.image_roles,
        "available_modalities": source.available_modalities,
        "instruction": source.instruction,
        "travel_mode": source.travel_mode,
        "model_visible_context": source.model_visible_context,
        "evidence_catalog": source.evidence_catalog,
        "goal_geometry": source.goal_geometry,
        "route_points_ego_t": source.route_points_ego_t,
    }
    if cfg.include_full_cot:
        views.append(
            ReasoningTrainingViewV2(
                sample_id=f"{source.source_group_id}:full_cot",
                view_type="full_cot",
                task_family="grounded_navigation",
                task_type="reasoning_v2_full_cot",
                assistant_text=assistant_text,
                reasoning_target=target,
                trajectory_ego_t=source.trajectory_ego_t if source.system1_enabled else None,
                trajectory_policy=source.trajectory_policy if source.system1_enabled else None,
                action_mask=source.action_mask if source.system1_enabled else None,
                loss_mask={
                    "lm": True,
                    "system1_action": source.system1_enabled,
                    "trajectory": False,
                    "segmentation": False,
                },
                metadata={
                    "annotation_schema_version": annotation.schema_version,
                    "source_fingerprint": source.source_fingerprint,
                    "teacher_provider": annotation.teacher_provider,
                    "teacher_model": annotation.teacher_model,
                    "annotation_warnings": annotation.warnings,
                    "readout_token_count": len(readout),
                },
                **common,
            )
        )

    if cfg.include_qa:
        validity = _effective_qa_validity(source, annotation, config=cfg)
        qa_views = build_reasoning_qa_views(
            target.reasoning,
            source_group_id=source.source_group_id,
            available_modalities=source.available_modalities,
            label_validity=validity,
            minimum_confidence=cfg.minimum_qa_confidence,
        )
        for qa in qa_views:
            views.append(
                ReasoningTrainingViewV2(
                    sample_id=qa.sample_id,
                    view_type="qa",
                    task_family=qa.task_family,
                    task_type=qa.task_type,
                    assistant_text=qa.teacher_answer,
                    section_name=qa.section_name,
                    question=qa.question,
                    loss_mask={
                        "lm": True,
                        "system1_action": False,
                        "trajectory": False,
                        "segmentation": False,
                    },
                    metadata={
                        **qa.metadata,
                        "label_source": qa.label_source,
                        "label_confidence": qa.confidence,
                        "label_evidence_refs": list(qa.evidence_refs),
                        "source_fingerprint": source.source_fingerprint,
                    },
                    **common,
                )
            )
    return views


def export_reasoning_training_views(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    config: ReasoningViewBuildConfig | None = None,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Join frozen sources with latest annotations and write training views.

    Args:
        source_jsonl: Frozen source rows.
        annotation_jsonl: Append-only annotations; latest source id wins.
        output_jsonl: Cross-repository view JSONL.
        config: Projection policy.
        limit: Optional maximum source rows.
        skip_errors: Continue after fingerprint or validation failures.

    Returns:
        Export counts and bounded errors.

    Raises:
        ValueError: On first invalid accepted annotation when ``skip_errors``
            is false.
    """

    cfg = config or ReasoningViewBuildConfig()
    annotations = _latest_annotations(annotation_jsonl)
    outputs: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(source_jsonl)):
        if limit is not None and index >= int(limit):
            break
        counters["source_rows"] += 1
        source = ReasoningSourceV2.model_validate(row)
        annotation = annotations.get(source.source_group_id)
        if annotation is None:
            counters["missing_annotation"] += 1
            continue
        if annotation.status != "accepted":
            counters[f"annotation_{annotation.status}"] += 1
            continue
        try:
            views = build_reasoning_training_views(source, annotation, config=cfg)
        except ValueError as exc:
            counters["invalid_accepted_annotation"] += 1
            errors.append({"source_group_id": source.source_group_id, "error": str(exc)})
            if not skip_errors:
                raise
            continue
        for view in views:
            outputs.append(view.model_dump(mode="json", exclude_none=True))
            counters[f"view_{view.view_type}"] += 1
            counters[f"task_{view.task_type}"] += 1
            if view.loss_mask.get("system1_action"):
                counters["system1_views"] += 1
        counters["accepted_sources"] += 1
    write_jsonl_atomic(output_jsonl, outputs)
    return {
        "schema_version": "urbannav_reasoning_training_view_v2_report",
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "output_jsonl": str(output_jsonl),
        "counts": dict(counters),
        "errors": errors[:100],
    }


def _effective_qa_validity(
    source: ReasoningSourceV2,
    annotation: ReasoningAnnotationV2,
    *,
    config: ReasoningViewBuildConfig,
) -> dict[str, dict[str, Any]]:
    """Merge source truth gates with bounded VLM observational projections."""

    result = {key: dict(value) for key, value in source.reasoning_qa_label_validity.items()}
    if not config.derive_observational_teacher_qa or annotation.target is None:
        return result
    trace = annotation.target.reasoning
    confidence = _confidence_score(trace.decision.confidence)
    alignment_refs = tuple(trace.cross_modal_alignment.evidence_refs)
    if (
        {"fpv", "bev"}.issubset(source.available_modalities)
        and trace.cross_modal_alignment.available
        and alignment_refs
    ):
        for task_type in ("route_to_fpv_grounding", "local_route_executability"):
            result.setdefault(
                task_type,
                {
                    "valid": True,
                    "confidence": confidence,
                    "label_source": "validated_vlm_teacher",
                    "evidence_refs": list(alignment_refs),
                },
            )
    fpv_refs = tuple(trace.evidence_grounding.fpv_evidence_refs)
    if fpv_refs:
        result.setdefault(
            "social_traversability",
            {
                "valid": True,
                "confidence": confidence,
                "label_source": "validated_vlm_teacher",
                "evidence_refs": list(fpv_refs),
            },
        )
    return result


def _confidence_score(value: str) -> float:
    """Map controlled confidence to a conservative QA export score."""

    return {"high": 0.9, "medium": 0.75, "low": 0.5, "unknown": 0.0}.get(str(value), 0.0)


def _latest_annotations(path: str | Path) -> dict[str, ReasoningAnnotationV2]:
    """Load latest append-only annotation for every source id."""

    result: dict[str, ReasoningAnnotationV2] = {}
    for row in iter_jsonl(path):
        annotation = ReasoningAnnotationV2.model_validate(row)
        result[annotation.source_group_id] = annotation
    return result


__all__ = [
    "ReasoningViewBuildConfig",
    "build_reasoning_training_views",
    "export_reasoning_training_views",
]
