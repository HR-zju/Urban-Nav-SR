"""Project safe Stage I inputs and Stage II sidecars into causal CFM views."""

from __future__ import annotations

from typing import Any, Mapping

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV1,
    StageITrainingViewV1,
    parse_stage_i_training_view_v1,
)
from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION,
    STAGE_II_PROJECTION_DROP_SCHEMA_VERSION,
    STAGE_II_PROJECTION_POLICY_ID,
    STAGE_II_PROJECTION_POLICY_VERSION,
    STAGE_II_TRAINING_CONTRACT_ID,
    StageIICFMTrainingViewV1,
    StageIIContextCacheV1,
    StageIIProjectionDropV1,
    StageIITrajectoryTargetV1,
    stage_ii_digest_v1,
)
from mcp_server.data.stage_ii_context import (
    stage_ii_projected_input_digest_v1,
)


def project_stage_ii_cfm_view_v1(
    stage_i_view: StageITrainingViewV1 | Mapping[str, Any],
    *,
    trajectory_target: StageIITrajectoryTargetV1 | Mapping[str, Any],
    reasoning_context: StageIIContextCacheV1 | Mapping[str, Any],
    decision_context: StageIIContextCacheV1 | Mapping[str, Any],
    review_status: str,
) -> StageIICFMTrainingViewV1:
    """Build one immutable trajectory-only Stage II exchange view.

    Args:
        stage_i_view: A formal FullCoT Stage I view used only for its verified
            projected inputs, parent audit, and private strong direction stamp.
        trajectory_target: Deterministic metric FLU target.
        reasoning_context: Accepted Pass-R five-section cache.
        decision_context: Accepted independent Pass-D cache.
        review_status: ``not_required`` or ``resolved``.

    Returns:
        Validated Stage II CFM exchange view.

    Raises:
        ValueError: If input, audit, cache, or target identities are unsafe.
    """

    typed = (
        stage_i_view
        if isinstance(stage_i_view, StageIFullCoTViewV1)
        else parse_stage_i_training_view_v1(dict(stage_i_view))
    )
    if not isinstance(typed, StageIFullCoTViewV1):
        raise ValueError("Stage II projection requires a formal Stage I FullCoT input view")
    target = (
        trajectory_target
        if isinstance(trajectory_target, StageIITrajectoryTargetV1)
        else StageIITrajectoryTargetV1.model_validate(trajectory_target)
    )
    reasoning = (
        reasoning_context
        if isinstance(reasoning_context, StageIIContextCacheV1)
        else StageIIContextCacheV1.model_validate(reasoning_context)
    )
    decision = (
        decision_context
        if isinstance(decision_context, StageIIContextCacheV1)
        else StageIIContextCacheV1.model_validate(decision_context)
    )
    if typed.parent_audit_id is None:
        raise ValueError("Stage II requires a resolved formal action audit")
    if review_status not in {"not_required", "resolved"}:
        raise ValueError("Stage II excludes pending or rejected human review")
    input_digest = stage_ii_projected_input_digest_v1(
        images=list(typed.images),
        image_roles=list(typed.image_roles),
        instruction=typed.instruction,
        model_visible_context=typed.model_visible_context,
        evidence_catalog=[
            item.model_dump(mode="json", exclude_none=True)
            for item in typed.evidence_catalog
        ],
        input_modalities=typed.input_modalities.model_dump(mode="json"),
    )
    for cache in (reasoning, decision):
        if cache.projected_input_digest != input_digest:
            raise ValueError("Stage II cache was generated from a different input projection")
        if cache.sample_id != typed.sample_id:
            raise ValueError("Stage II cache sample identity mismatch")
    payload: dict[str, Any] = {
        "schema_version": STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION,
        "training_contract": STAGE_II_TRAINING_CONTRACT_ID,
        "sample_id": f"{typed.sample_id}:stage_ii_cfm",
        "source_group_id": typed.source_group_id,
        "split_group_id": typed.split_group_id,
        "dataset_source": typed.dataset_source,
        "modality_profile_id": typed.modality_profile_id,
        "native_availability": typed.native_availability.model_dump(mode="json"),
        "training_selection": typed.training_selection.model_dump(mode="json"),
        "modality_state": typed.modality_state.model_dump(mode="json"),
        "instruction_training_role": typed.instruction_training_role,
        "images": list(typed.images),
        "image_roles": list(typed.image_roles),
        "instruction": typed.instruction,
        "model_visible_context": dict(typed.model_visible_context),
        "evidence_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in typed.evidence_catalog
        ],
        "input_modalities": typed.input_modalities.model_dump(mode="json"),
        "reasoning_context": reasoning.model_dump(mode="json", exclude_none=True),
        "decision_context": decision.model_dump(mode="json", exclude_none=True),
        "trajectory_target": target.model_dump(mode="json"),
        "direction_context": typed.direction_context.model_dump(mode="json"),
        "private_direction_supervision": typed.direction_supervision.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "parent_stage_i_sample_id": typed.sample_id,
        "parent_source_fingerprint": typed.parent_source_fingerprint,
        "parent_audit_id": typed.parent_audit_id,
        "formal_training_eligible": True,
        "review_status": review_status,
        "loss_mask": {
            "lm": False,
            "system1_action": True,
            "trajectory": False,
            "segmentation": False,
        },
        "projection_policy_id": STAGE_II_PROJECTION_POLICY_ID,
        "projection_policy_version": STAGE_II_PROJECTION_POLICY_VERSION,
    }
    payload["view_digest"] = stage_ii_digest_v1(payload)
    return StageIICFMTrainingViewV1.model_validate(payload)


def build_stage_ii_projection_drop_v1(
    *,
    parent_sample_id: str,
    source_group_id: str,
    split_group_id: str,
    dataset_source: str,
    modality_profile_id: str,
    reason_code: str,
    detail: str,
) -> StageIIProjectionDropV1:
    """Build one immutable fail-closed projection record.

    Args:
        parent_sample_id: Best available parent identity.
        source_group_id: Best available source-group identity.
        split_group_id: Best available leakage-safe split identity.
        dataset_source: Dataset family.
        modality_profile_id: Requested modality profile.
        reason_code: Stable bounded failure category.
        detail: Bounded engineering diagnostic.

    Returns:
        Validated immutable drop audit.
    """

    payload = {
        "schema_version": STAGE_II_PROJECTION_DROP_SCHEMA_VERSION,
        "parent_sample_id": parent_sample_id,
        "source_group_id": source_group_id,
        "split_group_id": split_group_id,
        "dataset_source": dataset_source,
        "modality_profile_id": modality_profile_id,
        "reason_code": reason_code,
        "detail": detail,
        "projection_policy_id": STAGE_II_PROJECTION_POLICY_ID,
        "projection_policy_version": STAGE_II_PROJECTION_POLICY_VERSION,
    }
    payload["drop_digest"] = stage_ii_digest_v1(payload)
    return StageIIProjectionDropV1.model_validate(payload)


__all__ = [
    "build_stage_ii_projection_drop_v1",
    "project_stage_ii_cfm_view_v1",
]
