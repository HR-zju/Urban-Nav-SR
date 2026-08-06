"""Build leakage-bounded LM-only training views for Navigation VQA V2."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.agent.vqa_v2_prompt_policy import (
    STUDENT_VQA_PROFILE_V2,
)
from mcp_server.agent.vqa_v2_text import render_vqa_answer_v2
from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQACapabilityConfigV2,
    VQASourceV2,
    VQATrainingViewV2,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic


def build_vqa_training_view_v2(
    source: VQASourceV2,
    annotation: VQAAnnotationV2,
    capabilities: VQACapabilityConfigV2,
) -> VQATrainingViewV2:
    """Build one fresh view without private source interfaces.

    Args:
        source: Frozen source.
        annotation: Accepted typed annotation.
        capabilities: Versioned export policy.

    Returns:
        Validated cross-repository LM-only view.

    Raises:
        ValueError: If provenance, evidence, or capability gates fail.
    """

    if annotation.source_group_id != (
        source.observation.source_group_id
    ):
        raise ValueError("annotation/source group mismatch")
    if annotation.source_fingerprint != source.source_fingerprint:
        raise ValueError("annotation/source fingerprint mismatch")
    capability = capabilities.capability(
        annotation.question.task_id
    )
    if not capability.training_export_enabled:
        raise ValueError(
            "training_export_disabled:"
            f"{annotation.question.task_id}"
        )
    if annotation.producer_id not in capability.producer_ids:
        raise ValueError(
            "annotation producer is not enabled for task"
        )
    available_refs = {
        entry.ref
        for entry in source.observation.evidence_catalog
    }
    missing = sorted(
        set(annotation.evidence_refs) - available_refs
    )
    if missing:
        raise ValueError(
            f"annotation evidence is absent from student input: {missing}"
        )
    assistant_text = render_vqa_answer_v2(annotation.answer)
    return VQATrainingViewV2(
        sample_id=annotation.annotation_id,
        source_group_id=annotation.source_group_id,
        split_group_id=source.observation.split_group_id,
        dataset_source=source.observation.dataset_source,
        images=list(source.observation.images),
        image_roles=list(source.observation.image_roles),
        instruction=source.observation.instruction,
        model_visible_context=dict(
            source.observation.model_visible_context
        ),
        evidence_catalog=list(
            source.observation.evidence_catalog
        ),
        input_modalities=source.observation.input_modalities,
        question=annotation.question,
        assistant_text=assistant_text,
        annotation_id=annotation.annotation_id,
        label_source=annotation.label_source,
        source_kind=annotation.source_kind,
        producer_id=annotation.producer_id,
        producer_version=annotation.producer_version,
        prompt_profile_id=STUDENT_VQA_PROFILE_V2.profile_id,
        prompt_policy_digest=(
            STUDENT_VQA_PROFILE_V2.policy_digest
        ),
        capability_config_digest=capabilities.digest(),
        metadata={
            "annotation_context_digest": (
                annotation.annotation_context_digest
            ),
            "teacher_provider": annotation.teacher_provider,
            "teacher_model": annotation.teacher_model,
        },
    )


def export_vqa_training_views_v2(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    capabilities: VQACapabilityConfigV2,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Join source/annotation artifacts into training views.

    Args:
        source_jsonl: Frozen VQA source rows.
        annotation_jsonl: Accepted VQA annotations.
        output_jsonl: Destination training-view JSONL.
        capabilities: Versioned export policy.
        skip_errors: Continue after invalid joins when true.

    Returns:
        Deterministic export report.
    """

    sources = {
        source.observation.source_group_id: source
        for source in (
            VQASourceV2.model_validate(row)
            for row in iter_jsonl(source_jsonl)
        )
    }
    annotations = [
        VQAAnnotationV2.model_validate(row)
        for row in iter_jsonl(annotation_jsonl)
    ]
    outputs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for annotation in annotations:
        counts["annotation_rows"] += 1
        try:
            source = sources[annotation.source_group_id]
            view = build_vqa_training_view_v2(
                source,
                annotation,
                capabilities,
            )
            if view.sample_id in seen_ids:
                raise ValueError(
                    f"duplicate VQA sample id: {view.sample_id}"
                )
            seen_ids.add(view.sample_id)
        except (KeyError, TypeError, ValueError) as exc:
            counts["invalid_rows"] += 1
            if len(errors) < 100:
                errors.append(
                    {
                        "annotation_id": annotation.annotation_id,
                        "error": str(exc),
                    }
                )
            if not skip_errors:
                raise
            continue
        outputs.append(
            view.model_dump(mode="json", exclude_none=True)
        )
        counts["output_rows"] += 1
        counts[f"task:{view.question.task_id}"] += 1
        counts[f"label_source:{view.label_source}"] += 1
        counts[f"dataset:{view.dataset_source}"] += 1
    write_jsonl_atomic(output_jsonl, outputs)
    return {
        "schema_version": (
            "urbannav_navigation_vqa_training_view_export_v2"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "output_jsonl": str(output_jsonl),
        "capability_config_digest": capabilities.digest(),
        "counts": dict(counts),
        "errors": errors,
    }


__all__ = [
    "build_vqa_training_view_v2",
    "export_vqa_training_views_v2",
]
