"""Project safe Stage I inputs and Stage II sidecars into causal CFM views."""

from __future__ import annotations

import re
from typing import Any, Mapping

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV1,
    StageIFullCoTViewV2,
    StageIReasoningDropoutViewV2,
    StageITrainingViewV1,
    StageITrainingViewV2,
    parse_stage_i_training_view_v1,
    parse_stage_i_training_view_v2,
)
from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE,
    STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2,
    STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION,
    STAGE_II_CACHED_TRAJ_QUERY_SOURCE_VIEW_V2_SCHEMA_VERSION,
    STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_SCHEMA_VERSION,
    STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_V2_SCHEMA_VERSION,
    STAGE_II_CFM_TRAINING_VIEW_V2_SCHEMA_VERSION,
    STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION,
    STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_ID,
    STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_VERSION,
    STAGE_II_DIRECT_LATENT_VIEW_SCHEMA_VERSION,
    STAGE_II_PROJECTION_DROP_V2_SCHEMA_VERSION,
    STAGE_II_PROJECTION_DROP_SCHEMA_VERSION,
    STAGE_II_PROJECTION_POLICY_ID,
    STAGE_II_PROJECTION_POLICY_VERSION,
    STAGE_II_PROJECTION_POLICY_V2_ID,
    STAGE_II_PROJECTION_POLICY_V2_VERSION,
    STAGE_II_TRAINING_CONTRACT_ID,
    STAGE_II_V2_TRAINING_CONTRACT_ID,
    STAGE_II_V46_DIRECT_LATENT_PROFILE_ID,
    STAGE_II_V46_DIRECT_LATENT_PROFILE_VERSION,
    StageIICFMDirectLatentViewV1,
    StageIICFMTrainingViewV1,
    StageIICFMTrainingViewV2,
    StageIICachedTrajQueryBindingV1,
    StageIICachedTrajQueryBindingV2,
    StageIICachedTrajQueryObservationV1,
    StageIICachedTrajQuerySourceViewV2,
    StageIICachedTrajQueryTrainingViewV1,
    StageIICachedTrajQueryTrainingViewV2,
    StageIIContextCacheV1,
    StageIIProjectionDropV1,
    StageIIProjectionDropV2,
    StageIITrajectoryTargetV1,
    StageIITrajectoryTargetV2,
    stage_ii_digest_v1,
    stage_ii_pass_t_input_digest_v2,
    stage_ii_projected_input_digest_v1,
)
from mcp_server.data.stage_ii.context import (
    build_stage_ii_reasoning_absent_v2,
    build_stage_ii_reasoning_present_v2,
    stage_ii_parent_sample_id_v2,
    stage_ii_projected_input_digest_v1,
    stage_ii_projected_input_digest_from_stage_i_v2,
)


def build_stage_ii_cached_traj_query_observation_v1(
    view: StageIICFMTrainingViewV2 | StageITrainingViewV2 | Mapping[str, Any],
    *,
    observation_timestamp_sec: float,
    frame_timestamps_sec: list[float],
    bev_timestamp_sec: float | None = None,
) -> StageIICachedTrajQueryObservationV1:
    """Project one V2 view into target-free input for a slow query pass."""

    if isinstance(view, StageIICFMTrainingViewV2):
        typed: Any = view
        if typed.system1_condition_mode != "traj_query_only":
            raise ValueError(
                "cached query observation requires a traj_query_only action view"
            )
    elif isinstance(view, (StageIFullCoTViewV2, StageIReasoningDropoutViewV2)):
        typed = view
    elif isinstance(view, Mapping) and view.get("task_type") in {
        "full_cot",
        "reasoning_dropout",
    }:
        typed = parse_stage_i_training_view_v2(dict(view))
    else:
        typed = StageIICFMTrainingViewV2.model_validate(view)
    payload = {
        "schema_version": STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION,
        "sample_id": typed.sample_id,
        "source_group_id": typed.source_group_id,
        "split_group_id": typed.split_group_id,
        "dataset_source": typed.dataset_source,
        "modality_profile_id": typed.modality_profile_id,
        "images": list(typed.images),
        "image_roles": list(typed.image_roles),
        "instruction": typed.instruction,
        "model_visible_context": dict(typed.model_visible_context),
        "evidence_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in typed.evidence_catalog
        ],
        "input_modalities": typed.input_modalities.model_dump(mode="json"),
        "frame_timestamps_sec": list(frame_timestamps_sec),
        "observation_timestamp_sec": float(observation_timestamp_sec),
        "bev_timestamp_sec": bev_timestamp_sec,
    }
    payload["view_digest"] = stage_ii_digest_v1(payload)
    return StageIICachedTrajQueryObservationV1.model_validate(payload)


def build_stage_ii_cached_traj_query_source_view_v2(
    stage_i_view: StageITrainingViewV2 | Mapping[str, Any],
    *,
    trajectory_target: StageIITrajectoryTargetV2 | Mapping[str, Any],
    observation_timestamp_sec: float,
    frame_timestamps_sec: list[float],
    bev_timestamp_sec: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> StageIICachedTrajQuerySourceViewV2:
    """Bind one Stage-I visible row to an accepted private trajectory target.

    This is the pure-latent V2 source boundary.  It reads only the Stage-I
    model-visible envelope and the independently validated trajectory sidecar;
    Pass-R and Pass-D caches are intentionally not accepted here.
    """

    typed = (
        parse_stage_i_training_view_v2(dict(stage_i_view))
        if isinstance(stage_i_view, Mapping)
        else stage_i_view
    )
    if not isinstance(typed, (StageIFullCoTViewV2, StageIReasoningDropoutViewV2)):
        raise ValueError("cached-query V2 source requires a Stage-I reasoning row")
    target = (
        trajectory_target
        if isinstance(trajectory_target, StageIITrajectoryTargetV2)
        else StageIITrajectoryTargetV2.model_validate(trajectory_target)
    )
    observation = build_stage_ii_cached_traj_query_observation_v1(
        typed,
        observation_timestamp_sec=observation_timestamp_sec,
        frame_timestamps_sec=frame_timestamps_sec,
        bev_timestamp_sec=bev_timestamp_sec,
    )
    payload: dict[str, Any] = {
        "schema_version": (
            STAGE_II_CACHED_TRAJ_QUERY_SOURCE_VIEW_V2_SCHEMA_VERSION
        ),
        "training_contract": STAGE_II_V2_TRAINING_CONTRACT_ID,
        "sample_id": f"{typed.source_group_id}:cached_source_v2",
        "source_group_id": typed.source_group_id,
        "split_group_id": typed.split_group_id,
        "dataset_source": typed.dataset_source,
        "observation": observation.model_dump(mode="json"),
        "trajectory_target": target.model_dump(mode="json"),
        "provenance": dict(provenance or {}),
        "formal_training_eligible": True,
    }
    payload["provenance"].setdefault(
        "trajectory_source_group_id", typed.source_group_id
    )
    payload["view_digest"] = stage_ii_digest_v1(
        {key: value for key, value in payload.items() if key != "view_digest"}
    )
    return StageIICachedTrajQuerySourceViewV2.model_validate(payload)


def build_stage_ii_cached_traj_query_training_view_v1(
    action_view: StageIICFMTrainingViewV2 | Mapping[str, Any],
    query_source_view: StageIICFMTrainingViewV2 | Mapping[str, Any],
    *,
    query_source_timestamp_sec: float,
    action_observation_timestamp_sec: float,
    query_source_frame_timestamps_sec: list[float],
    action_frame_timestamps_sec: list[float],
    query_source_bev_timestamp_sec: float | None = None,
    action_bev_timestamp_sec: float | None = None,
) -> StageIICachedTrajQueryTrainingViewV1:
    """Build a causal query/action pair without exposing the action target."""

    action = (
        action_view
        if isinstance(action_view, StageIICFMTrainingViewV2)
        else StageIICFMTrainingViewV2.model_validate(action_view)
    )
    query = (
        query_source_view
        if isinstance(query_source_view, StageIICFMTrainingViewV2)
        else StageIICFMTrainingViewV2.model_validate(query_source_view)
    )
    if action.split_group_id != query.split_group_id:
        raise ValueError("cached pair requires one split_group_id")
    if action.dataset_source != query.dataset_source:
        raise ValueError("cached pair requires one dataset_source")
    if action.modality_profile_id != query.modality_profile_id:
        raise ValueError("cached pair requires one modality profile")
    query_observation = build_stage_ii_cached_traj_query_observation_v1(
        query,
        observation_timestamp_sec=query_source_timestamp_sec,
        frame_timestamps_sec=query_source_frame_timestamps_sec,
        bev_timestamp_sec=query_source_bev_timestamp_sec,
    )
    action_observation = build_stage_ii_cached_traj_query_observation_v1(
        action,
        observation_timestamp_sec=action_observation_timestamp_sec,
        frame_timestamps_sec=action_frame_timestamps_sec,
        bev_timestamp_sec=action_bev_timestamp_sec,
    )
    binding = StageIICachedTrajQueryBindingV1(
        condition_mode=STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE,
        query_source_timestamp_sec=float(query_source_timestamp_sec),
        action_observation_timestamp_sec=float(action_observation_timestamp_sec),
        query_age_sec=float(action_observation_timestamp_sec - query_source_timestamp_sec),
        modality_profile_id=action.modality_profile_id,
    )
    payload = {
        "schema_version": STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_SCHEMA_VERSION,
        "training_contract": "stage_ii_cfm_v2",
        "sample_id": f"{action.sample_id}:{STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE}",
        "query_source": query_observation.model_dump(mode="json"),
        "action_observation": action_observation.model_dump(mode="json"),
        "action_view": action.model_dump(mode="json"),
        "cached_traj_query": binding.model_dump(mode="json"),
        "formal_training_eligible": True,
    }
    payload["view_digest"] = stage_ii_digest_v1(payload)
    return StageIICachedTrajQueryTrainingViewV1.model_validate(payload)


def build_stage_ii_cached_traj_query_training_view_v2(
    query_source: StageIICachedTrajQueryObservationV1 | Mapping[str, Any],
    action_observation: StageIICachedTrajQueryObservationV1 | Mapping[str, Any],
    *,
    trajectory_target: StageIITrajectoryTargetV2 | Mapping[str, Any],
    condition_digest: str,
    provenance: Mapping[str, Any] | None = None,
) -> StageIICachedTrajQueryTrainingViewV2:
    """Build one Pass-R/Pass-D-free cached-query training view.

    The input observations are already projected from model-visible source
    rows.  This function only binds their causal relationship to the private
    4-second, 32-step target; it never creates a target from prompt text.

    Args:
        query_source: Target-free observation used by the slow path.
        action_observation: Current observation used by the fast path.
        trajectory_target: Accepted private 4-second trajectory target.
        condition_digest: Digest of stable navigation conditions shared by both
            observations.
        provenance: Optional private lineage metadata.

    Returns:
        A validated V2 cached-query training view.

    Raises:
        ValueError: If the observations, timing, target, or condition digest
            cannot form one immutable pair.
    """

    query = (
        query_source
        if isinstance(query_source, StageIICachedTrajQueryObservationV1)
        else StageIICachedTrajQueryObservationV1.model_validate(query_source)
    )
    action = (
        action_observation
        if isinstance(action_observation, StageIICachedTrajQueryObservationV1)
        else StageIICachedTrajQueryObservationV1.model_validate(action_observation)
    )
    target = (
        trajectory_target
        if isinstance(trajectory_target, StageIITrajectoryTargetV2)
        else StageIITrajectoryTargetV2.model_validate(trajectory_target)
    )
    if query.dataset_source != action.dataset_source:
        raise ValueError("cached trajectory-query V2 requires one dataset_source")
    if query.split_group_id != action.split_group_id:
        raise ValueError("cached trajectory-query V2 requires one split_group_id")
    if not re.fullmatch(r"[0-9a-f]{64}", condition_digest):
        raise ValueError("cached trajectory-query V2 condition_digest must be SHA256")
    binding = StageIICachedTrajQueryBindingV2(
        condition_mode=STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2,
        query_source_timestamp_sec=query.observation_timestamp_sec,
        action_observation_timestamp_sec=action.observation_timestamp_sec,
        query_age_sec=(
            action.observation_timestamp_sec - query.observation_timestamp_sec
        ),
        query_source_group_id=query.source_group_id,
        action_source_group_id=action.source_group_id,
        query_modality_profile_id=query.modality_profile_id,
        action_modality_profile_id=action.modality_profile_id,
        condition_digest=condition_digest,
    )
    payload = {
        "schema_version": STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_V2_SCHEMA_VERSION,
        "training_contract": STAGE_II_V2_TRAINING_CONTRACT_ID,
        "condition_mode": STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2,
        "sample_id": f"{action.source_group_id}:{STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2}",
        "source_group_id": action.source_group_id,
        "split_group_id": action.split_group_id,
        "dataset_source": action.dataset_source,
        "query_source": query.model_dump(mode="json"),
        "action_observation": action.model_dump(mode="json"),
        "temporal_binding": binding.model_dump(mode="json"),
        "trajectory_target": target.model_dump(mode="json"),
        "provenance": dict(provenance or {}),
        "formal_training_eligible": True,
        "loss_mask": {
            "lm": False,
            "system1_action": True,
            "trajectory": False,
            "segmentation": False,
        },
    }
    payload["view_digest"] = stage_ii_digest_v1(payload)
    return StageIICachedTrajQueryTrainingViewV2.model_validate(payload)


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


def project_stage_ii_cfm_view_v1_from_stage_i_v2(
    stage_i_view: StageIFullCoTViewV2 | Mapping[str, Any],
    *,
    trajectory_target: StageIITrajectoryTargetV1 | Mapping[str, Any],
    reasoning_context: StageIIContextCacheV1 | Mapping[str, Any],
    decision_context: StageIIContextCacheV1 | Mapping[str, Any],
    review_status: str,
) -> StageIICFMTrainingViewV1:
    """Project a V2 FullCoT parent into the unchanged 16-point CFM contract.

    The compatibility boundary is deliberately narrow: only a formal V2
    FullCoT parent is accepted, Pass-R/Pass-D still bind the exact visible
    input digest, and the target remains the independent 16-step CFM truth.
    It does not reinterpret a V2 target as V1 or promote a reasoning-dropout
    route into a FullCoT parent.

    Args:
        stage_i_view: Current V2 FullCoT source view.
        trajectory_target: Existing clean 16-step metric FLU target.
        reasoning_context: Accepted target-free Pass-R cache.
        decision_context: Accepted target-free Pass-D cache.
        review_status: Resolved review admission state.

    Returns:
        A validated V1 CFM exchange view with explicit V2-parent lineage
        retained in its external build report.

    Raises:
        ValueError: If source, target, cache, or review identities drift.
    """

    typed = (
        parse_stage_i_training_view_v2(dict(stage_i_view))
        if isinstance(stage_i_view, Mapping)
        else stage_i_view
    )
    if not isinstance(typed, StageIFullCoTViewV2):
        raise ValueError("Stage II V1 compatibility requires a V2 FullCoT input view")
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
        raise ValueError("Stage II V1 compatibility requires a resolved formal action audit")
    if review_status not in {"not_required", "resolved"}:
        raise ValueError("Stage II V1 compatibility excludes pending or rejected review")
    input_digest = stage_ii_projected_input_digest_from_stage_i_v2(typed)
    for cache, kind in ((reasoning, "reasoning_grounding"), (decision, "decision")):
        identities = (
            cache.parse_status == "accepted",
            cache.context_kind == kind,
            cache.projected_input_digest == input_digest,
            cache.sample_id == typed.sample_id,
            cache.source_group_id == typed.source_group_id,
            cache.split_group_id == typed.split_group_id,
            cache.dataset_source == typed.dataset_source,
            cache.modality_profile_id == typed.modality_profile_id,
            cache.parent_source_fingerprint == typed.parent_source_fingerprint,
        )
        if not all(identities):
            raise ValueError(f"Stage II V1 compatibility {kind} cache identity mismatch")
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


def project_stage_ii_direct_latent_view_from_stage_i_v2(
    stage_i_view: StageIFullCoTViewV2 | Mapping[str, Any],
    *,
    trajectory_target: StageIITrajectoryTargetV1 | Mapping[str, Any],
    review_status: str,
) -> StageIICFMDirectLatentViewV1:
    """Project one current-V4 FullCoT parent into a cacheless Pass-T view.

    The projection reuses only the exact prompt-visible Stage-I V2 input and
    the independently QC-accepted private trajectory target.  It does not
    materialize, accept, or consume Pass-R/Pass-D context caches.

    Args:
        stage_i_view: Formal current-V4 FullCoT parent.
        trajectory_target: Existing clean 16-step metric FLU target.
        review_status: Resolved trajectory-review admission state.

    Returns:
        A validated direct-latent CFM exchange view.

    Raises:
        ValueError: If the parent is not a formal FullCoT V2 row, its action
            audit is unresolved, or trajectory/review lineage is unsafe.
    """

    typed = (
        parse_stage_i_training_view_v2(dict(stage_i_view))
        if isinstance(stage_i_view, Mapping)
        else stage_i_view
    )
    if not isinstance(typed, StageIFullCoTViewV2):
        raise ValueError("Stage II direct-latent requires a V2 FullCoT input view")
    target = (
        trajectory_target
        if isinstance(trajectory_target, StageIITrajectoryTargetV1)
        else StageIITrajectoryTargetV1.model_validate(trajectory_target)
    )
    if typed.parent_audit_id is None:
        raise ValueError("Stage II direct-latent requires a resolved formal action audit")
    if review_status not in {"not_required", "resolved"}:
        raise ValueError("Stage II direct-latent excludes pending or rejected review")
    projected_input_digest = stage_ii_projected_input_digest_v1(
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
    payload: dict[str, Any] = {
        "schema_version": STAGE_II_DIRECT_LATENT_VIEW_SCHEMA_VERSION,
        "training_contract": STAGE_II_TRAINING_CONTRACT_ID,
        "direct_latent_profile_id": STAGE_II_V46_DIRECT_LATENT_PROFILE_ID,
        "direct_latent_profile_version": STAGE_II_V46_DIRECT_LATENT_PROFILE_VERSION,
        "sample_id": f"{typed.sample_id}:stage_ii_direct_latent",
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
        "projected_input_digest": projected_input_digest,
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
        "projection_policy_id": STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_ID,
        "projection_policy_version": STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_VERSION,
    }
    payload["view_digest"] = stage_ii_digest_v1(payload)
    return StageIICFMDirectLatentViewV1.model_validate(payload)


def project_stage_ii_cfm_view_v2(
    stage_i_view: StageITrainingViewV2 | Mapping[str, Any],
    *,
    trajectory_target: StageIITrajectoryTargetV2 | Mapping[str, Any],
    reasoning_context: StageIIContextCacheV1 | Mapping[str, Any] | None,
    decision_context: StageIIContextCacheV1 | Mapping[str, Any],
    review_status: str,
    split_group_id: str | None = None,
) -> StageIICFMTrainingViewV2:
    """Build one V2 CFM view from exactly one Stage I reasoning route.

    FullCoT parents require accepted independent Pass R and Pass D caches.
    Reasoning-dropout parents require only accepted Pass D and must not carry
    any Pass R cache, including a rejected cache.
    """

    typed = (
        parse_stage_i_training_view_v2(dict(stage_i_view))
        if isinstance(stage_i_view, Mapping)
        else stage_i_view
    )
    if not isinstance(typed, (StageIFullCoTViewV2, StageIReasoningDropoutViewV2)):
        raise ValueError("Stage II V2 requires a Stage I Reasoning route")
    target = (
        trajectory_target
        if isinstance(trajectory_target, StageIITrajectoryTargetV2)
        else StageIITrajectoryTargetV2.model_validate(trajectory_target)
    )
    decision = (
        decision_context
        if isinstance(decision_context, StageIIContextCacheV1)
        else StageIIContextCacheV1.model_validate(decision_context)
    )
    if typed.parent_audit_id is None:
        raise ValueError("Stage II V2 requires a resolved formal action audit")
    if review_status not in {"not_required", "resolved"}:
        raise ValueError("Stage II V2 excludes pending or rejected human review")
    projected_input_digest = stage_ii_projected_input_digest_from_stage_i_v2(
        typed
    )
    parent_stage_i_sample_id = stage_ii_parent_sample_id_v2(typed)
    if isinstance(typed, StageIFullCoTViewV2):
        if reasoning_context is None:
            raise ValueError("reasoning_plus_decision requires a Pass-R cache")
        reasoning_cache = (
            reasoning_context
            if isinstance(reasoning_context, StageIIContextCacheV1)
            else StageIIContextCacheV1.model_validate(reasoning_context)
        )
        reasoning = build_stage_ii_reasoning_present_v2(reasoning_cache)
        context_mode = "reasoning_plus_decision"
    else:
        if reasoning_context is not None:
            raise ValueError(
                "reasoning dropout forbids both accepted and rejected Pass-R caches"
            )
        reasoning = build_stage_ii_reasoning_absent_v2(typed)
        context_mode = "decision_only"
    pass_t_input_digest = stage_ii_pass_t_input_digest_v2(
        projected_input_digest=projected_input_digest,
        context_mode=context_mode,
        reasoning=reasoning,
        decision_context=decision,
    )
    payload: dict[str, Any] = {
        "schema_version": STAGE_II_CFM_TRAINING_VIEW_V2_SCHEMA_VERSION,
        "training_contract": STAGE_II_V2_TRAINING_CONTRACT_ID,
        "sample_id": (
            f"{parent_stage_i_sample_id}:stage_ii_cfm_v2:{context_mode}"
        ),
        "source_group_id": typed.source_group_id,
        "split_group_id": split_group_id or typed.split_group_id,
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
        "projected_input_digest": projected_input_digest,
        "context_mode": context_mode,
        "reasoning": reasoning.model_dump(mode="json", exclude_none=True),
        "decision_context": decision.model_dump(mode="json", exclude_none=True),
        "pass_t_input_digest": pass_t_input_digest,
        "system1_condition_mode": "traj_query_only",
        "trajectory_target": target.model_dump(mode="json"),
        "direction_context": typed.direction_context.model_dump(mode="json"),
        "private_direction_supervision": typed.direction_supervision.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "parent_stage_i_sample_id": parent_stage_i_sample_id,
        "stage_i_route_sample_id": typed.sample_id,
        "stage_i_route_task_type": typed.task_type,
        "stage_i_route_projection_digest": typed.projection_digest,
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
        "projection_policy_id": STAGE_II_PROJECTION_POLICY_V2_ID,
        "projection_policy_version": STAGE_II_PROJECTION_POLICY_V2_VERSION,
    }
    payload["view_digest"] = stage_ii_digest_v1(payload)
    return StageIICFMTrainingViewV2.model_validate(payload)


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


def build_stage_ii_projection_drop_v2(
    *,
    parent_sample_id: str,
    stage_i_route_sample_id: str,
    stage_i_route_task_type: str,
    requested_context_mode: str,
    source_group_id: str,
    split_group_id: str,
    dataset_source: str,
    modality_profile_id: str,
    reason_code: str,
    detail: str,
) -> StageIIProjectionDropV2:
    """Build one immutable V2 fail-closed projection record."""

    payload = {
        "schema_version": STAGE_II_PROJECTION_DROP_V2_SCHEMA_VERSION,
        "parent_sample_id": parent_sample_id,
        "stage_i_route_sample_id": stage_i_route_sample_id,
        "stage_i_route_task_type": stage_i_route_task_type,
        "requested_context_mode": requested_context_mode,
        "source_group_id": source_group_id,
        "split_group_id": split_group_id,
        "dataset_source": dataset_source,
        "modality_profile_id": modality_profile_id,
        "reason_code": reason_code,
        "detail": detail[:1000],
        "projection_policy_id": STAGE_II_PROJECTION_POLICY_V2_ID,
        "projection_policy_version": STAGE_II_PROJECTION_POLICY_V2_VERSION,
    }
    payload["drop_digest"] = stage_ii_digest_v1(payload)
    return StageIIProjectionDropV2.model_validate(payload)


def project_stage_ii_cfm_view_with_audit_v2(
    stage_i_view: StageITrainingViewV2 | Mapping[str, Any],
    *,
    trajectory_target: StageIITrajectoryTargetV2 | Mapping[str, Any],
    reasoning_context: StageIIContextCacheV1 | Mapping[str, Any] | None,
    decision_context: StageIIContextCacheV1 | Mapping[str, Any],
    review_status: str,
) -> StageIICFMTrainingViewV2 | StageIIProjectionDropV2:
    """Project V2 or retain one bounded audit instead of weakening gates."""

    try:
        return project_stage_ii_cfm_view_v2(
            stage_i_view,
            trajectory_target=trajectory_target,
            reasoning_context=reasoning_context,
            decision_context=decision_context,
            review_status=review_status,
        )
    except (TypeError, ValueError) as exc:
        raw = dict(stage_i_view) if isinstance(stage_i_view, Mapping) else stage_i_view.model_dump(mode="json")
        task_type = str(raw.get("task_type") or "unknown")
        dropout_link = raw.get("dropout_link")
        link = dropout_link if isinstance(dropout_link, Mapping) else {}
        parent_sample_id = str(
            link.get("parent_full_cot_sample_id")
            or raw.get("sample_id")
            or "unknown_stage_i_parent"
        )
        requested_context_mode = {
            "full_cot": "reasoning_plus_decision",
            "reasoning_dropout": "decision_only",
        }.get(task_type, "unknown")
        return build_stage_ii_projection_drop_v2(
            parent_sample_id=parent_sample_id,
            stage_i_route_sample_id=str(raw.get("sample_id") or "unknown_stage_i_route"),
            stage_i_route_task_type=task_type,
            requested_context_mode=requested_context_mode,
            source_group_id=str(raw.get("source_group_id") or "unknown_source_group"),
            split_group_id=str(raw.get("split_group_id") or "unknown_split_group"),
            dataset_source=str(raw.get("dataset_source") or "unknown_dataset"),
            modality_profile_id=str(raw.get("modality_profile_id") or "unknown_profile"),
            reason_code=_projection_reason_code_v2(exc),
            detail=str(exc) or exc.__class__.__name__,
        )


def _projection_reason_code_v2(error: Exception) -> str:
    """Map detailed validation diagnostics to a stable bounded audit code."""

    text = str(error).lower()
    categories = (
        ("forbids both accepted and rejected pass-r", "dropout_reasoning_cache_present"),
        ("cannot create a pass-r", "dropout_reasoning_cache_present"),
        ("requires a pass-r cache", "reasoning_cache_missing"),
        ("requires accepted pass-r cache", "reasoning_cache_rejected"),
        ("reasoning requires accepted pass r", "reasoning_cache_rejected"),
        ("accepted reasoning_grounding", "reasoning_cache_rejected"),
        ("accepted decision", "decision_cache_rejected"),
        ("trajectory", "trajectory_target_invalid"),
        ("human review", "review_not_resolved"),
        ("reasoning route", "stage_i_route_invalid"),
    )
    for needle, code in categories:
        if needle in text:
            return code
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if normalized and not normalized[0].isalpha():
        normalized = f"stage_ii_{normalized}"
    return (normalized or "stage_ii_v2_projection_failed")[:96]


__all__ = [
    "build_stage_ii_projection_drop_v1",
    "build_stage_ii_projection_drop_v2",
    "project_stage_ii_cfm_view_v1",
    "project_stage_ii_cfm_view_v1_from_stage_i_v2",
    "project_stage_ii_direct_latent_view_from_stage_i_v2",
    "project_stage_ii_cfm_view_v2",
    "project_stage_ii_cfm_view_with_audit_v2",
]
