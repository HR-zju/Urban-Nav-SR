"""Project canonical training views into hidden open-loop references."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp_server.agent.vqa_v2_text import parse_vqa_answer_v2
from mcp_server.contracts.openloop_benchmark import OpenLoopReferenceV1
from mcp_server.contracts.stage_i_sft import (
    StageIDecisionViewV1,
    StageIFullCoTViewV1,
    StageIFullCoTViewV2,
    StageIVQAViewV1,
    parse_stage_i_training_view_v1,
)
from mcp_server.contracts.stage_ii_cfm import (
    StageIITrajectoryTargetV1,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic


def build_stage_i_openloop_reference_v1(
    payload: dict[str, Any],
) -> OpenLoopReferenceV1:
    """Build one hidden Stage I reference from a canonical exchange view.

    Args:
        payload: Canonical Stage I FullCoT, VQA, or Decision view.

    Returns:
        Hidden reference with provenance kept outside model input.

    Raises:
        ValueError: If the stored VQA assistant answer is not canonical.
    """

    view = parse_stage_i_training_view_v1(payload)
    common = {
        "sample_id": view.sample_id,
        "parent_sample_id": view.parent_sample_id,
        "source_group_id": view.source_group_id,
        "split_group_id": view.split_group_id,
        "stage": "stage_i",
        "task_type": view.task_type,
        "dataset_source": view.dataset_source,
    }
    if isinstance(view, StageIFullCoTViewV1):
        field_provenance = {
            "decision.selected_action_label": "deterministic_gt",
            **_declared_field_provenance(view.metadata),
        }
        return OpenLoopReferenceV1(
            **common,
            target={
                "reasoning_target": view.reasoning_target.model_dump(
                    mode="json"
                )
            },
            label_provenance="lvlm_weak",
            field_provenance=field_provenance,
            direction_stamp=view.direction_supervision,
            subgroup_metadata={
                "parent_artifact_kind": view.parent_artifact_kind,
                "parent_audit_id": view.parent_audit_id,
            },
        )
    if isinstance(view, StageIDecisionViewV1):
        field_provenance = {
            "decision.selected_action_label": "deterministic_gt",
            "decision.speed": "reference_annotation",
            "decision.response": "reference_annotation",
            **_declared_field_provenance(view.metadata),
        }
        return OpenLoopReferenceV1(
            **common,
            target={"decision": view.decision.model_dump(mode="json")},
            label_provenance="reference_annotation",
            field_provenance=field_provenance,
            direction_stamp=view.direction_supervision,
            subgroup_metadata={
                "decision_authority": view.decision_authority,
                "parent_audit_id": view.parent_audit_id,
            },
        )
    if not isinstance(view, StageIVQAViewV1):
        raise TypeError(f"unsupported Stage I view: {type(view).__name__}")
    parsed = parse_vqa_answer_v2(
        view.assistant_text,
        expected_task_id=view.question.task_id,
        question=view.question,
    )
    if not parsed.valid or parsed.answer is None:
        raise ValueError(
            f"Stage I VQA reference answer is invalid: {parsed.errors}"
        )
    provenance = {
        "derived_gt": "deterministic_gt",
        "external_gt": "deterministic_gt",
        "lvlm_weak": "lvlm_weak",
    }[view.label_source]
    return OpenLoopReferenceV1(
        **common,
        target={
            "question": view.question.model_dump(mode="json"),
            "answer": parsed.answer.model_dump(mode="json"),
        },
        label_provenance=provenance,
        subgroup_metadata={
            "annotation_id": view.annotation_id,
            "source_kind": str(view.source_kind),
            "producer_id": view.producer_id,
            "producer_version": view.producer_version,
        },
    )


def build_stage_ii_openloop_reference_from_parent_v1(
    stage_i_view: StageIFullCoTViewV1 | StageIFullCoTViewV2,
    trajectory_target: StageIITrajectoryTargetV1,
) -> OpenLoopReferenceV1:
    """Build a Stage II reference before Pass-R/Pass-D survivor filtering.

    Args:
        stage_i_view: Formal V1 or V2-envelope FullCoT input defining public
            case identity.  The name remains V1 because the reference stores
            the frozen CFM-V1 trajectory target, not the Stage-I envelope.
        trajectory_target: Valid deterministic metric-FLU target.

    Returns:
        Hidden trajectory reference on the complete eligible denominator.
    """

    target = trajectory_target
    return OpenLoopReferenceV1(
        sample_id=f"{stage_i_view.sample_id}:stage_ii_cfm",
        parent_sample_id=stage_i_view.parent_sample_id,
        source_group_id=stage_i_view.source_group_id,
        split_group_id=stage_i_view.split_group_id,
        stage="stage_ii",
        task_type="trajectory",
        dataset_source=stage_i_view.dataset_source,
        target={"trajectory_target": target.model_dump(mode="json")},
        label_provenance="deterministic_gt",
        direction_stamp=stage_i_view.direction_supervision,
        trajectory_valid_mask=list(target.action_mask),
        trajectory_policy_id=target.policy.policy_id,
        subgroup_metadata={
            "trajectory_source": str(target.policy.trajectory_source),
            "yaw_source": target.policy.yaw_source,
            "valid_step_count": target.valid_step_count,
            "parent_audit_id": stage_i_view.parent_audit_id,
            "modality_profile_id": stage_i_view.modality_profile_id,
            "parent_stage_i_sample_id": stage_i_view.sample_id,
        },
    )


def build_openloop_reference_jsonl_v1(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """Build unique Stage I references from a canonical view manifest.

    Args:
        input_jsonl: Canonical Stage I exchange views.
        output_jsonl: Hidden reference destination.
        stage: Exact ``stage_i`` selector.

    Returns:
        Deterministic input/output counts.

    Raises:
        ValueError: If stage is unsupported or sample ids duplicate.
    """

    builders = {"stage_i": build_stage_i_openloop_reference_v1}
    try:
        builder = builders[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported open-loop stage: {stage}") from exc
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in iter_jsonl(input_jsonl):
        reference = builder(payload)
        if reference.sample_id in seen:
            raise ValueError(
                f"duplicate open-loop reference id: {reference.sample_id}"
            )
        seen.add(reference.sample_id)
        rows.append(reference.model_dump(mode="json", exclude_none=True))
    write_jsonl_atomic(output_jsonl, rows)
    return {
        "stage": stage,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "reference_count": len(rows),
    }


def _declared_field_provenance(
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Load only explicitly declared benchmark field provenance."""

    value = metadata.get("benchmark_field_provenance")
    if not isinstance(value, dict):
        return {}
    allowed = {
        "deterministic_gt",
        "human_gold",
        "lvlm_weak",
        "reference_annotation",
    }
    result = {str(key): str(item) for key, item in value.items()}
    invalid = sorted(set(result.values()) - allowed)
    if invalid:
        raise ValueError(f"invalid benchmark field provenance: {invalid}")
    return result


__all__ = [
    "build_openloop_reference_jsonl_v1",
    "build_stage_i_openloop_reference_v1",
    "build_stage_ii_openloop_reference_from_parent_v1",
]
