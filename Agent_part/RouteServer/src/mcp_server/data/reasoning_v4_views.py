"""Export review-preview and formal-training Reasoning V4.6 views."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.agent.reasoning_v4_prompt_policy import (
    STUDENT_FULL_COT_PROFILE_V4,
)
from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    ActionReviewResolutionV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
    ReasoningTrainingViewV4,
    ReleaseTrackV4,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_v4_compact import (
    render_compact_full_cot_v4,
)
from mcp_server.data.reasoning_v4_reconciliation import (
    validate_action_reconciliation_audit_v4,
)
from mcp_server.data.reasoning_v4_state import (
    derive_reasoning_release_state_v4,
    load_latest_action_resolutions_v4,
)


REASONING_V4_TRAINING_POLICY_VERSION = (
    "urbannav_reasoning_v4_6_language_sft_policy_v1"
)


def build_reasoning_training_views_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    *,
    release_track: ReleaseTrackV4 = "review_preview",
    resolution: ActionReviewResolutionV4 | None = None,
) -> list[ReasoningTrainingViewV4]:
    """Project one joined artifact into at most one compact FullCoT row.

    Review previews are always marked blocked and must never enter training.
    Formal rows are emitted only when every derived eligibility gate passes.

    Args:
        source: Frozen visible input and private strong sidecars.
        annotation: Immutable annotation.
        release_track: Review-only or formal-training track.
        resolution: Latest append-only resolution for the action audit.

    Returns:
        Zero or one compact FullCoT view.

    Raises:
        ValueError: If identities, audit reconstruction, or target validation
            fail.
    """

    if (
        annotation.source_group_id
        != source.observation.source_group_id
        or annotation.source_fingerprint
        != source.source_fingerprint
    ):
        raise ValueError(
            "V4.6 annotation/source identity mismatch"
        )
    if (
        annotation.action_reconciliation is not None
        and annotation.decision_proposal is not None
    ):
        validate_action_reconciliation_audit_v4(
            source,
            annotation.grounding_outputs,
            annotation.decision_proposal,
            annotation.action_reconciliation,
        )
    state = derive_reasoning_release_state_v4(
        source,
        annotation,
        resolution=resolution,
    )
    target = annotation.candidate_target
    direction = source.direction_supervision
    if target is None or direction is None:
        return []
    if (
        release_track == "formal_training"
        and not state.formal_training_eligible
    ):
        return []
    checked = validate_reasoning_target_v4(
        target,
        observation=source.observation,
        direction_supervision=direction,
    )
    if not checked.valid:
        raise ValueError(
            "V4.6 candidate failed full-view validation: "
            f"{checked.errors}"
        )
    compact = render_compact_full_cot_v4(
        target,
        action_profile_id=(
            source.observation.direction_context.action_profile_id
        ),
    )
    formal = release_track == "formal_training"
    observation = source.observation
    return [
        ReasoningTrainingViewV4(
            sample_id=(
                f"{observation.source_group_id}:"
                f"full_cot:v4_6:{release_track}"
            ),
            source_group_id=observation.source_group_id,
            split_group_id=observation.split_group_id,
            dataset_source=observation.dataset_source,
            release_track=release_track,
            audit_id=state.audit_id,
            review_status=state.review_status,
            artifact_status=(
                "training_ready" if formal else "preview_only"
            ),
            artifact_training_eligible=formal,
            formal_training_eligible=(
                state.formal_training_eligible
            ),
            images=observation.images,
            image_roles=observation.image_roles,
            instruction=observation.instruction,
            model_visible_context=(
                observation.model_visible_context
            ),
            evidence_catalog=observation.evidence_catalog,
            input_modalities=observation.input_modalities,
            direction_context=observation.direction_context,
            assistant_text=compact.assistant_text,
            reasoning_target=target,
            full_cot_renderer_id=compact.renderer_id,
            prompt_profile_id=(
                STUDENT_FULL_COT_PROFILE_V4.profile_id
            ),
            prompt_policy_digest=(
                STUDENT_FULL_COT_PROFILE_V4.policy_digest
            ),
            direction_supervision=direction,
            trajectory_ego_t=source.trajectory_ego_t,
            trajectory_policy=source.trajectory_policy,
            action_mask=source.action_mask,
            metadata={
                "source_fingerprint": source.source_fingerprint,
                "annotation_schema_version": (
                    annotation.schema_version
                ),
                "teacher_provider": annotation.teacher_provider,
                "teacher_model": annotation.teacher_model,
                "composer_version": (
                    annotation.candidate_structure_record.composer_version
                ),
                "training_policy_version": (
                    REASONING_V4_TRAINING_POLICY_VERSION
                ),
                "release_blockers": list(state.blockers),
                "validation_warnings": checked.warnings,
                "action_training_role": (
                    "decision_selected_action_lm"
                ),
            },
        )
    ]


def build_reasoning_v4_user_prompt(
    view: ReasoningTrainingViewV4,
) -> str:
    """Build the exact prompt-visible student user message.

    Args:
        view: Valid V4.6 training or preview view.

    Returns:
        Ordered image placeholders and visible JSON context.
    """

    payload = {
        "source_group_id": view.source_group_id,
        "requested_view_type": view.view_type,
        "instruction": view.instruction,
        "image_order": [
            {"ordinal": index + 1, "role": role}
            for index, role in enumerate(view.image_roles)
        ],
        "input_modalities": view.input_modalities.model_dump(
            mode="json"
        ),
        "direction_context": view.direction_context.model_dump(
            mode="json"
        ),
        "action_labels": list(
            view.direction_context.action_labels
        ),
        "affordance_probe_labels": list(
            view.direction_context.affordance_probe_labels
        ),
        "model_visible_context": view.model_visible_context,
        "visible_input_catalog": [
            item.model_dump(
                mode="json",
                exclude_none=True,
            )
            for item in view.evidence_catalog
        ],
    }
    return (
        "<image>" * len(view.images)
        + "\nUse the attached ordered images and visible context to "
        "produce exactly the requested Reasoning V4.6 output.\n"
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
    release_track: ReleaseTrackV4 = "review_preview",
    resolution_jsonl: str | Path | None = None,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Join sources, annotations, and resolutions into compact views.

    Args:
        source_jsonl: Frozen source manifest.
        annotation_jsonl: Append-only annotation manifest.
        output_jsonl: Destination view manifest.
        release_track: Review-only or formal-training export.
        resolution_jsonl: Optional append-only human resolutions.
        limit: Optional source-row limit.
        skip_errors: Continue after invalid joins.

    Returns:
        Machine-readable export report.
    """

    annotations = _latest_annotations_v4(annotation_jsonl)
    resolutions = load_latest_action_resolutions_v4(
        resolution_jsonl
    )
    outputs: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(source_jsonl)):
        if limit is not None and index >= int(limit):
            break
        source = ReasoningSourceV4.model_validate(row)
        counters["source_rows"] += 1
        annotation = annotations.get(
            source.observation.source_group_id
        )
        if annotation is None:
            counters["missing_annotation"] += 1
            continue
        counters[
            f"record_status:{annotation.record_status}"
        ] += 1
        audit = annotation.action_reconciliation
        resolution = (
            resolutions.get(audit.audit_id)
            if audit is not None
            else None
        )
        try:
            views = build_reasoning_training_views_v4(
                source,
                annotation,
                release_track=release_track,
                resolution=resolution,
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
        if not views:
            counters["not_exported"] += 1
        for view in views:
            outputs.append(
                view.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
            counters[f"release_track:{view.release_track}"] += 1
            counters[
                f"review_status:{view.review_status}"
            ] += 1
            counters[
                f"artifact_status:{view.artifact_status}"
            ] += 1
            counters["formal_training_eligible"] += int(
                view.formal_training_eligible
            )
            counters["system1_sidecar_views"] += int(
                view.trajectory_ego_t is not None
            )
    write_jsonl_atomic(output_jsonl, outputs)
    return {
        "schema_version": (
            "urbannav_reasoning_training_view_v4_6_export_report_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "resolution_jsonl": (
            str(resolution_jsonl)
            if resolution_jsonl is not None
            else None
        ),
        "output_jsonl": str(output_jsonl),
        "training_policy_version": (
            REASONING_V4_TRAINING_POLICY_VERSION
        ),
        "release_track": release_track,
        "formal_release_ready": (
            release_track == "formal_training"
            and not errors
            and bool(outputs)
        ),
        "counts": dict(counters),
        "errors": errors,
    }


def _latest_annotations_v4(
    path: str | Path,
) -> dict[str, ReasoningAnnotationV4]:
    """Return latest append-only annotation per source group."""

    result: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(path):
        annotation = ReasoningAnnotationV4.model_validate(
            row
        )
        result[annotation.source_group_id] = annotation
    return result


__all__ = [
    "REASONING_V4_TRAINING_POLICY_VERSION",
    "build_reasoning_training_views_v4",
    "build_reasoning_v4_user_prompt",
    "export_reasoning_training_views_v4",
]
