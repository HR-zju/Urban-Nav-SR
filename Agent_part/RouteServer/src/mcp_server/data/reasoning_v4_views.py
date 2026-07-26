"""Export independent and full language-supervised Reasoning V4 views."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.agent.reasoning_v4_prompt_policy import (
    STUDENT_DECISION_PROFILE_V4,
    STUDENT_FULL_COT_PROFILE_V4,
    STUDENT_GROUNDING_PROFILES_V4,
)
from mcp_server.agent.reasoning_v4_text import (
    render_reasoning_section_v4,
    render_reasoning_target_v4,
)
from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_target_v4,
)
from mcp_server.agent.stage2 import SYSTEM1_READOUT_TOKENS
from mcp_server.contracts.reasoning_dataset_v4 import (
    AnnotationStageRecordV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
    ReasoningTrainingViewV4,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic


REASONING_V4_TRAINING_POLICY_VERSION = (
    "urbannav_reasoning_v4_0_language_sft_policy_v1"
)


def build_reasoning_training_views_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
) -> list[ReasoningTrainingViewV4]:
    """Project one V4 annotation into independently maskable SFT samples.

    Args:
        source: Frozen visible observation and private strong sidecars.
        annotation: Partial or complete section annotation.

    Returns:
        Accepted section views plus a full view when composition succeeded.

    Raises:
        ValueError: If annotation identity or an accepted full target is invalid.
    """

    observation = source.observation
    if annotation.source_group_id != observation.source_group_id:
        raise ValueError("V4 annotation/source id mismatch")
    if annotation.source_fingerprint != source.source_fingerprint:
        raise ValueError("V4 annotation/source fingerprint mismatch")

    views: list[ReasoningTrainingViewV4] = []
    grounding_values = {
        "TaskGrounding": annotation.grounding_outputs.task_grounding,
        "SceneGrounding": annotation.grounding_outputs.scene_grounding,
        "LocalTraversability": annotation.grounding_outputs.local_traversability,
        "NavigationContextGrounding": (
            annotation.grounding_outputs.navigation_context_grounding
        ),
        "InteractionConstraintGrounding": (
            annotation.grounding_outputs.interaction_constraint_grounding
        ),
    }
    for section_name, section in grounding_values.items():
        record = annotation.stage_records.get(section_name)
        if (
            section is None
            or record is None
            or record.status != "accepted"
        ):
            continue
        profile = STUDENT_GROUNDING_PROFILES_V4[section_name]
        views.append(
            ReasoningTrainingViewV4(
                **_common_view_fields(source),
                sample_id=(
                    f"{observation.source_group_id}:"
                    f"{section_name}:v4_0"
                ),
                view_type="grounding_section_sft",
                section_name=section_name,
                assistant_text=render_reasoning_section_v4(
                    section_name,
                    section,
                ),
                prompt_profile_id=profile.profile_id,
                prompt_policy_digest=profile.policy_digest,
                metadata=_view_metadata(
                    source,
                    annotation,
                    record=record,
                ),
            )
        )

    decision_record = annotation.stage_records.get("Decision")
    if (
        annotation.decision is not None
        and annotation.grounding_outputs.complete
        and decision_record is not None
        and decision_record.status == "accepted"
    ):
        decision_context = dict(observation.model_visible_context)
        decision_context["validated_grounding_sections"] = (
            annotation.grounding_outputs.model_dump(
                mode="json",
                exclude_none=True,
            )
        )
        views.append(
            ReasoningTrainingViewV4(
                **_common_view_fields(
                    source,
                    model_visible_context=decision_context,
                ),
                sample_id=f"{observation.source_group_id}:Decision:v4_0",
                view_type="decision_section_sft",
                section_name="Decision",
                assistant_text=render_reasoning_section_v4(
                    "Decision",
                    annotation.decision,
                ),
                prompt_profile_id=STUDENT_DECISION_PROFILE_V4.profile_id,
                prompt_policy_digest=(
                    STUDENT_DECISION_PROFILE_V4.policy_digest
                ),
                metadata=_view_metadata(
                    source,
                    annotation,
                    record=decision_record,
                ),
            )
        )

    if annotation.status == "accepted" and annotation.target is not None:
        include_system1 = (
            source.trajectory_ego_t is not None
            and annotation.target.finish.decision_type == "navigate"
        )
        readout = (
            SYSTEM1_READOUT_TOKENS
            if include_system1
            else ()
        )
        assistant_text = render_reasoning_target_v4(
            annotation.target,
            readout_tokens=readout,
        )
        checked = validate_reasoning_target_v4(
            annotation.target,
            observation=observation,
            assistant_text=assistant_text,
        )
        if not checked.valid:
            raise ValueError(
                "accepted V4 annotation failed full-view validation: "
                f"{checked.errors}"
            )
        views.append(
            ReasoningTrainingViewV4(
                **_common_view_fields(source),
                sample_id=f"{observation.source_group_id}:full_cot:v4_0",
                view_type="full_cot_sft",
                assistant_text=assistant_text,
                reasoning_target=annotation.target,
                prompt_profile_id=STUDENT_FULL_COT_PROFILE_V4.profile_id,
                prompt_policy_digest=(
                    STUDENT_FULL_COT_PROFILE_V4.policy_digest
                ),
                direction_supervision=source.direction_supervision,
                trajectory_ego_t=(
                    source.trajectory_ego_t if include_system1 else None
                ),
                trajectory_policy=(
                    source.trajectory_policy if include_system1 else None
                ),
                action_mask=source.action_mask if include_system1 else None,
                metadata={
                    **_view_metadata(source, annotation),
                    "validation_warnings": checked.warnings,
                    "readout_token_count": len(readout),
                    **(
                        {"system1_omitted_reason": "decision_not_navigate"}
                        if source.trajectory_ego_t is not None
                        and not include_system1
                        else {}
                    ),
                },
            )
        )
    return views


def build_reasoning_v4_user_prompt(
    view: ReasoningTrainingViewV4,
) -> str:
    """Build the exact student user message from one V4 view.

    Args:
        view: Cross-repository V4 training view.

    Returns:
        Prompt-visible JSON context and output request.
    """

    payload = {
        "source_group_id": view.source_group_id,
        "requested_view_type": view.view_type,
        "requested_section": view.section_name,
        "instruction": view.instruction,
        "image_order": [
            {"ordinal": index + 1, "role": role}
            for index, role in enumerate(view.image_roles)
        ],
        "input_modalities": view.input_modalities.model_dump(mode="json"),
        "model_visible_context": view.model_visible_context,
        "evidence_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in view.evidence_catalog
        ],
        "system1_readout_tokens": (
            list(SYSTEM1_READOUT_TOKENS)
            if view.trajectory_ego_t is not None
            else []
        ),
    }
    return (
        "<image>" * len(view.images)
        + "\nUse the attached ordered images and the following visible context to "
        "produce exactly the requested Reasoning V4 output.\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def export_reasoning_training_views_v4(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Join frozen V4 sources and latest annotations into SFT views.

    Args:
        source_jsonl: V4 source manifest.
        annotation_jsonl: Append-only V4 annotation manifest.
        output_jsonl: Destination training-view manifest.
        limit: Optional maximum source rows.
        skip_errors: Continue after invalid joined rows.

    Returns:
        Export counts and bounded errors.
    """

    annotations = _latest_annotations_v4(annotation_jsonl)
    outputs: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(source_jsonl)):
        if limit is not None and index >= int(limit):
            break
        source = ReasoningSourceV4.model_validate(row)
        counters["source_rows"] += 1
        annotation = annotations.get(source.observation.source_group_id)
        if annotation is None:
            counters["missing_annotation"] += 1
            continue
        counters[f"annotation:{annotation.status}"] += 1
        try:
            views = build_reasoning_training_views_v4(
                source,
                annotation,
            )
        except ValueError as exc:
            counters["invalid_join"] += 1
            if len(errors) < 100:
                errors.append(
                    {
                        "source_group_id": (
                            source.observation.source_group_id
                        ),
                        "error": str(exc),
                    }
                )
            if not skip_errors:
                raise
            continue
        for view in views:
            outputs.append(view.model_dump(mode="json", exclude_none=True))
            counters[f"view:{view.view_type}"] += 1
            if view.section_name is not None:
                counters[f"section:{view.section_name}"] += 1
            counters["direction_sidecar_views"] += int(
                view.direction_supervision is not None
            )
            counters["system1_sidecar_views"] += int(
                view.trajectory_ego_t is not None
            )
    write_jsonl_atomic(output_jsonl, outputs)
    return {
        "schema_version": (
            "urbannav_reasoning_training_view_v4_0_report_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "output_jsonl": str(output_jsonl),
        "training_policy_version": REASONING_V4_TRAINING_POLICY_VERSION,
        "counts": dict(counters),
        "errors": errors,
    }


def _common_view_fields(
    source: ReasoningSourceV4,
    *,
    model_visible_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return immutable visible fields shared by every exported view."""

    observation = source.observation
    return {
        "source_group_id": observation.source_group_id,
        "split_group_id": observation.split_group_id,
        "dataset_source": observation.dataset_source,
        "images": observation.images,
        "image_roles": observation.image_roles,
        "instruction": observation.instruction,
        "model_visible_context": (
            model_visible_context
            if model_visible_context is not None
            else observation.model_visible_context
        ),
        "evidence_catalog": observation.evidence_catalog,
        "input_modalities": observation.input_modalities,
    }


def _view_metadata(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    *,
    record: AnnotationStageRecordV4 | None = None,
) -> dict[str, Any]:
    """Build version-complete audit metadata for one exported view."""

    metadata: dict[str, Any] = {
        "source_fingerprint": source.source_fingerprint,
        "annotation_schema_version": annotation.schema_version,
        "teacher_provider": annotation.teacher_provider,
        "teacher_model": annotation.teacher_model,
        "composer_version": annotation.metadata.get("composer_version"),
        "training_policy_version": REASONING_V4_TRAINING_POLICY_VERSION,
        "language_supervision": True,
    }
    if record is not None:
        metadata.update(
            {
                "teacher_prompt_profile_id": record.prompt_profile_id,
                "teacher_prompt_policy_digest": (
                    record.prompt_policy_digest
                ),
                "teacher_stage_warnings": record.warnings,
            }
        )
    return metadata


def _latest_annotations_v4(
    path: str | Path,
) -> dict[str, ReasoningAnnotationV4]:
    """Return the last append-only annotation for each source id."""

    result: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(path):
        annotation = ReasoningAnnotationV4.model_validate(row)
        result[annotation.source_group_id] = annotation
    return result
