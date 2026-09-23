"""Build safety-bounded UrbanNav Stage I SFT exchange views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from urbannav_contracts.history_observation import history_image_roles, validate_history_observation

from mcp_server.agent.vqa_v2_text import render_vqa_answer_v2
from mcp_server.contracts.reasoning_dataset_v4 import (
    EvidenceCatalogEntryV4,
    InputModalitiesV4,
    ReasoningTrainingViewV4,
)
from mcp_server.contracts.stage_i_sft import (
    STAGE_I_LOSS_SPAN_POLICY_ID,
    STAGE_I_PROJECTION_DROP_SCHEMA_VERSION,
    STAGE_I_PROJECTION_POLICY_ID,
    STAGE_I_PROJECTION_POLICY_VERSION,
    STAGE_I_TRAINING_VIEW_SCHEMA_VERSION,
    StageIAnswerabilityV2,
    StageIAnswerabilityV1,
    StageIDecisionViewV1,
    StageIFullCoTViewV1,
    StageIFullCoTViewV2,
    StageIInstructionTrainingRoleV1,
    StageILossSpanV1,
    StageILossSpanV2,
    StageIModalityAvailabilityV1,
    StageIModalityAvailabilityV2,
    StageIModalityProfileIdV1,
    StageIModalityProfileIdV2,
    StageIModalitySelectionV1,
    StageIModalitySelectionV2,
    StageIModalityStateV1,
    StageIModalityStateV2,
    StageIProjectionDropV1,
    StageIVQAViewV1,
    StageIVQAViewV2,
    StageIReasoningDropoutViewV2,
    STAGE_I_LOSS_SPAN_POLICY_V2_ID,
    STAGE_I_PROJECTION_POLICY_V2_ID,
    STAGE_I_PROJECTION_POLICY_V2_VERSION,
    STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION,
    parse_stage_i_training_view_v1,
    parse_stage_i_training_view_v2,
    project_reasoning_dropout_from_full_v2,
    stage_i_projection_digest_v1,
    stage_i_projection_digest_v2,
)
from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQACapabilityConfigV2,
    VQASourceV2,
)
from mcp_server.contracts.vqa_v2 import get_vqa_task_spec_v2
from mcp_server.data.vqa.views import build_vqa_training_view_v2


@dataclass(frozen=True)
class StageIModalityProfileV1:
    """Describe one registered Stage I input combination.

    Args:
        profile_id: Stable public profile identity.
        history_fpv: Retain native history FPV images.
        instruction: Retain the native instruction.
        bev: Retain native BEV images.
    """

    profile_id: StageIModalityProfileIdV1
    history_fpv: bool
    instruction: bool
    bev: bool

    def selection(self) -> StageIModalitySelectionV1:
        """Return the exact contract selection for this profile.

        Returns:
            Frozen Stage I selection with dormant modalities disabled.
        """

        return StageIModalitySelectionV1(
            history_fpv=self.history_fpv,
            instruction=self.instruction,
            bev=self.bev,
        )


_PROFILE_ROWS = (
    ("fpv_current_v1", False, False, False),
    ("fpv_temporal_v1", True, False, False),
    ("fpv_instruction_v1", False, True, False),
    ("fpv_bev_v1", False, False, True),
    ("fpv_instruction_bev_v1", False, True, True),
    ("fpv_temporal_instruction_v1", True, True, False),
    ("fpv_temporal_bev_v1", True, False, True),
    ("fpv_temporal_instruction_bev_v1", True, True, True),
)
STAGE_I_MODALITY_PROFILES_V1 = {
    profile_id:
    StageIModalityProfileV1(
        profile_id=profile_id,  # type: ignore[arg-type]
        history_fpv=history_fpv,
        instruction=instruction,
        bev=bev,
    )
    for profile_id, history_fpv, instruction, bev in _PROFILE_ROWS
}


@dataclass(frozen=True)
class StageIModalityProfileV2:
    """Describe one V2 Stage I profile, including optional waypoint input."""

    profile_id: StageIModalityProfileIdV2
    history_fpv: bool
    instruction: bool
    bev: bool
    waypoint: bool

    def selection(self) -> StageIModalitySelectionV2:
        """Return the exact V2 retained-input selection."""

        return StageIModalitySelectionV2(
            history_fpv=self.history_fpv,
            instruction=self.instruction,
            bev=self.bev,
            waypoint=self.waypoint,
        )


_PROFILE_ROWS_V2 = tuple(
    (
        (
            "fpv_waypoint_v2"
            if profile_id == "fpv_current_v1"
            else f"{profile_id.removesuffix('_v1')}_waypoint_v2"
        ),
        history_fpv,
        instruction,
        bev,
        True,
    )
    for profile_id, history_fpv, instruction, bev in _PROFILE_ROWS
)
STAGE_I_MODALITY_PROFILES_V2 = {
    profile_id:
    StageIModalityProfileV2(
        profile_id=profile_id,  # type: ignore[arg-type]
        history_fpv=history_fpv,
        instruction=instruction,
        bev=bev,
        waypoint=False,
    )
    for profile_id, history_fpv, instruction, bev in _PROFILE_ROWS
} | {
    profile_id:
    StageIModalityProfileV2(
        profile_id=profile_id,  # type: ignore[arg-type]
        history_fpv=history_fpv,
        instruction=instruction,
        bev=bev,
        waypoint=True,
    )
    for profile_id, history_fpv, instruction, bev, _waypoint in _PROFILE_ROWS_V2
}


@dataclass(frozen=True)
class _ProjectedInputsV1:
    """Hold one internally validated prompt-visible input projection."""

    availability: StageIModalityAvailabilityV1
    selection: StageIModalitySelectionV1
    states: StageIModalityStateV1
    instruction_training_role: StageIInstructionTrainingRoleV1
    images: list[str]
    image_roles: list[str]
    instruction: str | None
    model_visible_context: dict[str, Any]
    evidence_catalog: list[EvidenceCatalogEntryV4]
    input_modalities: InputModalitiesV4


@dataclass(frozen=True)
class _ProjectedInputsV2:
    """Hold one V2 prompt-visible projection without dropping waypoints."""

    availability: StageIModalityAvailabilityV2
    selection: StageIModalitySelectionV2
    states: StageIModalityStateV2
    instruction_training_role: StageIInstructionTrainingRoleV1
    images: list[str]
    image_roles: list[str]
    instruction: str | None
    model_visible_context: dict[str, Any]
    evidence_catalog: list[EvidenceCatalogEntryV4]
    input_modalities: InputModalitiesV4


def get_stage_i_modality_profile_v1(
    profile_id: str, ) -> StageIModalityProfileV1:
    """Resolve one registered Stage I modality profile.

    Args:
        profile_id: Exact registered profile identity.

    Returns:
        Immutable profile.

    Raises:
        ValueError: If the profile is not registered.
    """

    try:
        return STAGE_I_MODALITY_PROFILES_V1[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Stage I modality profile: {profile_id}") from exc


def get_stage_i_modality_profile_v2(
    profile_id: str,
) -> StageIModalityProfileV2:
    """Resolve one V2 profile without changing the V1 registry."""

    try:
        return STAGE_I_MODALITY_PROFILES_V2[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Stage I V2 modality profile: {profile_id}") from exc


def project_stage_i_reasoning_view_v1(
    view: ReasoningTrainingViewV4 | Mapping[str, Any],
    *,
    task_type: str,
    modality_profile_id: str,
    instruction_training_role: StageIInstructionTrainingRoleV1,
    allow_fpv_only_decision: bool = False,
) -> StageIFullCoTViewV1 | StageIDecisionViewV1:
    """Project one formal V4.6 view into FullCoT or Decision-only Stage I.

    Args:
        view: Formal Reasoning V4.6 training view.
        task_type: ``full_cot`` or ``decision``.
        modality_profile_id: Registered Stage I input profile.
        instruction_training_role: Provenance of the parent instruction.
        allow_fpv_only_decision: Explicit dataset capability for FPV behavior
            cloning when no route-direction authority remains visible.

    Returns:
        Validated immutable Stage I exchange view.

    Raises:
        ValueError: If release, provenance, modality, or answerability gates fail.
    """

    typed = view if isinstance(
        view, ReasoningTrainingViewV4
    ) else ReasoningTrainingViewV4.model_validate(view)
    if typed.release_track != "formal_training" or not typed.artifact_training_eligible:
        raise ValueError(
            "Stage I accepts only formal training-ready Reasoning views")
    if typed.audit_id is None:
        raise ValueError(
            "Stage I Reasoning projection requires action audit_id")
    parent_source_fingerprint = str(
        typed.metadata.get("source_fingerprint") or "")
    if len(parent_source_fingerprint) != 64:
        raise ValueError(
            "Stage I Reasoning parent source_fingerprint is missing")

    profile = get_stage_i_modality_profile_v1(modality_profile_id)
    projected = _project_visible_inputs(
        images=typed.images,
        image_roles=typed.image_roles,
        instruction=typed.instruction,
        model_visible_context=typed.model_visible_context,
        evidence_catalog=typed.evidence_catalog,
        input_modalities=typed.input_modalities,
        profile=profile,
        instruction_training_role=instruction_training_role,
    )
    if task_type == "full_cot":
        return _project_full_cot_view(
            typed,
            projected=projected,
            parent_source_fingerprint=parent_source_fingerprint,
        )
    if task_type == "decision":
        return _project_decision_view(
            typed,
            projected=projected,
            parent_source_fingerprint=parent_source_fingerprint,
            allow_fpv_only_decision=allow_fpv_only_decision,
        )
    raise ValueError(f"unsupported Stage I Reasoning task_type: {task_type}")


def project_stage_i_vqa_view_v1(
    source: VQASourceV2 | Mapping[str, Any],
    annotation: VQAAnnotationV2 | Mapping[str, Any],
    *,
    capabilities: VQACapabilityConfigV2,
    modality_profile_id: str,
    instruction_training_role: StageIInstructionTrainingRoleV1,
) -> StageIVQAViewV1:
    """Project one accepted VQA annotation while retaining all cited evidence.

    Args:
        source: Frozen VQA source.
        annotation: Accepted typed VQA annotation.
        capabilities: Exact active-task registry.
        modality_profile_id: Registered Stage I input profile.
        instruction_training_role: Provenance of the parent instruction.

    Returns:
        Validated answer-only Stage I VQA view.

    Raises:
        ValueError: If task capability, modality, or evidence gates fail.
    """

    typed_source = source if isinstance(
        source, VQASourceV2) else VQASourceV2.model_validate(source)
    typed_annotation = (annotation if isinstance(annotation, VQAAnnotationV2)
                        else VQAAnnotationV2.model_validate(annotation))
    base_view = build_vqa_training_view_v2(typed_source, typed_annotation,
                                           capabilities)
    profile = get_stage_i_modality_profile_v1(modality_profile_id)
    projected = _project_visible_inputs(
        images=base_view.images,
        image_roles=base_view.image_roles,
        instruction=base_view.instruction,
        model_visible_context=base_view.model_visible_context,
        evidence_catalog=base_view.evidence_catalog,
        input_modalities=base_view.input_modalities,
        profile=profile,
        instruction_training_role=instruction_training_role,
    )
    task_spec = get_vqa_task_spec_v2(typed_annotation.question.task_id)
    _require_vqa_modalities(task_spec.required_modalities,
                            projected.input_modalities)
    visible_refs = {str(item.ref) for item in projected.evidence_catalog}
    missing_evidence = sorted(
        set(typed_annotation.evidence_refs) - visible_refs)
    if missing_evidence:
        raise ValueError(
            f"Stage I VQA projection removed annotation evidence: {missing_evidence}"
        )

    assistant_text = render_vqa_answer_v2(typed_annotation.answer)
    payload: dict[str, Any] = {
        **_base_payload(
            task_type="vqa",
            sample_id=f"{typed_annotation.annotation_id}:stage_i:{profile.profile_id}",
            source_group_id=typed_source.observation.source_group_id,
            split_group_id=typed_source.observation.split_group_id,
            dataset_source=typed_source.observation.dataset_source,
            profile=profile,
            projected=projected,
            assistant_text=assistant_text,
            loss_spans=[_full_loss_span(assistant_text, "vqa")],
            parent_artifact_kind="vqa_v2_0",
            parent_sample_id=base_view.sample_id,
            parent_source_fingerprint=typed_source.source_fingerprint,
            parent_annotation_id=typed_annotation.annotation_id,
            parent_audit_id=None,
            answerability=StageIAnswerabilityV1(
                vqa=True,
                reasons=("required_modalities_retained", "annotation_evidence_retained"),
            ),
        ),
        "question":
        typed_annotation.question.model_dump(mode="json"),
        "annotation_id":
        typed_annotation.annotation_id,
        "annotation_evidence_refs":
        list(typed_annotation.evidence_refs),
        "label_source":
        typed_annotation.label_source,
        "source_kind":
        str(typed_annotation.source_kind),
        "producer_id":
        typed_annotation.producer_id,
        "producer_version":
        typed_annotation.producer_version,
        "prompt_profile_id":
        base_view.prompt_profile_id,
        "prompt_policy_digest":
        base_view.prompt_policy_digest,
        "capability_config_digest":
        base_view.capability_config_digest,
        "metadata": {
            **_history_projection_metadata(
                next((item.attributes for item in typed_source.observation.evidence_catalog
                      if item.ref == "fpv.current"), {}),
                typed_source.observation.source_group_id, projected),
            "annotation_context_digest":
            typed_annotation.annotation_context_digest,
            "source_field_refs": list(typed_annotation.source_field_refs),
        },
    }
    return _finalize_view(StageIVQAViewV1, payload)


def project_stage_i_reasoning_view_with_audit_v1(
    view: ReasoningTrainingViewV4 | Mapping[str, Any],
    *,
    task_type: str,
    modality_profile_id: str,
    instruction_training_role: StageIInstructionTrainingRoleV1,
    allow_fpv_only_decision: bool = False,
) -> StageIFullCoTViewV1 | StageIDecisionViewV1 | StageIProjectionDropV1:
    """Project Reasoning or return a structured annotation-only drop audit.

    Args:
        view: Formal Reasoning V4.6 training view or raw payload.
        task_type: Requested ``full_cot`` or ``decision`` task.
        modality_profile_id: Requested Stage I profile.
        instruction_training_role: Parent instruction provenance.
        allow_fpv_only_decision: Explicit FPV behavior-cloning capability.

    Returns:
        Canonical training view on success, otherwise a deterministic drop audit.
    """

    try:
        return project_stage_i_reasoning_view_v1(
            view,
            task_type=task_type,
            modality_profile_id=modality_profile_id,
            instruction_training_role=instruction_training_role,
            allow_fpv_only_decision=allow_fpv_only_decision,
        )
    except (TypeError, ValueError) as exc:
        raw = _raw_mapping(view)
        return _build_projection_drop(
            task_type=task_type,
            parent_artifact_kind="reasoning_v4_6",
            parent_sample_id=str(
                raw.get("sample_id") or "unknown_reasoning_parent"),
            source_group_id=str(
                raw.get("source_group_id") or "unknown_source_group"),
            split_group_id=str(
                raw.get("split_group_id") or "unknown_split_group"),
            dataset_source=str(raw.get("dataset_source") or "unknown_dataset"),
            modality_profile_id=modality_profile_id,
            error=exc,
        )


def project_stage_i_vqa_view_with_audit_v1(
    source: VQASourceV2 | Mapping[str, Any],
    annotation: VQAAnnotationV2 | Mapping[str, Any],
    *,
    capabilities: VQACapabilityConfigV2,
    modality_profile_id: str,
    instruction_training_role: StageIInstructionTrainingRoleV1,
) -> StageIVQAViewV1 | StageIProjectionDropV1:
    """Project VQA or return a structured annotation-only drop audit.

    Args:
        source: Frozen VQA source or raw payload.
        annotation: Accepted VQA annotation or raw payload.
        capabilities: Exact active-task registry.
        modality_profile_id: Requested Stage I profile.
        instruction_training_role: Parent instruction provenance.

    Returns:
        Canonical VQA training view on success, otherwise a drop audit.
    """

    try:
        return project_stage_i_vqa_view_v1(
            source,
            annotation,
            capabilities=capabilities,
            modality_profile_id=modality_profile_id,
            instruction_training_role=instruction_training_role,
        )
    except (TypeError, ValueError) as exc:
        raw_source = _raw_mapping(source)
        raw_observation = raw_source.get("observation")
        observation = raw_observation if isinstance(raw_observation,
                                                    Mapping) else {}
        raw_annotation = _raw_mapping(annotation)
        return _build_projection_drop(
            task_type="vqa",
            parent_artifact_kind="vqa_v2_0",
            parent_sample_id=str(
                raw_annotation.get("annotation_id") or "unknown_vqa_parent"),
            source_group_id=str(
                observation.get("source_group_id") or "unknown_source_group"),
            split_group_id=str(
                observation.get("split_group_id") or "unknown_split_group"),
            dataset_source=str(
                observation.get("dataset_source") or "unknown_dataset"),
            modality_profile_id=modality_profile_id,
            error=exc,
        )


def _project_full_cot_view(
    view: ReasoningTrainingViewV4,
    *,
    projected: _ProjectedInputsV1,
    parent_source_fingerprint: str,
) -> StageIFullCoTViewV1:
    """Build one no-regeneration FullCoT view after strict preservation checks."""

    availability = projected.availability.model_dump(mode="python")
    selection = projected.selection.model_dump(mode="python")
    # FullCoT 没有字段级 provenance；首版只允许完整保留原始可见模态。
    for name in ("history_fpv", "instruction", "bev", "goal_image",
                 "pixel_goal", "waypoint"):
        if bool(availability[name]) != bool(selection[name]):
            raise ValueError(
                f"full_cot_answerability_failed:dropped_native_{name}")
    if view.model_visible_context:
        raise ValueError(
            "full_cot_answerability_failed:unsupported_visible_context")
    parent_evidence_refs = {str(item.ref) for item in view.evidence_catalog}
    projected_evidence_refs = {
        str(item.ref)
        for item in projected.evidence_catalog
    }
    if projected_evidence_refs != parent_evidence_refs:
        raise ValueError(
            "full_cot_answerability_failed:evidence_catalog_changed")
    action = str(view.direction_supervision.label_name)
    loss_spans = _full_cot_loss_spans(view.assistant_text, action)
    payload: dict[str, Any] = {
        **_base_payload(
            task_type="full_cot",
            sample_id=f"{view.sample_id}:stage_i:{projected_profile_id(projected)}",
            source_group_id=view.source_group_id,
            split_group_id=view.split_group_id,
            dataset_source=view.dataset_source,
            profile=get_stage_i_modality_profile_v1(
                projected_profile_id(projected)),
            projected=projected,
            assistant_text=view.assistant_text,
            loss_spans=loss_spans,
            parent_artifact_kind="reasoning_v4_6",
            parent_sample_id=view.sample_id,
            parent_source_fingerprint=parent_source_fingerprint,
            parent_annotation_id=None,
            parent_audit_id=view.audit_id,
            answerability=StageIAnswerabilityV1(
                full_cot=True,
                reasons=("all_native_modalities_retained", "canonical_full_cot_reused"),
            ),
        ),
        "reasoning_target":
        view.reasoning_target.model_dump(mode="json", exclude_none=True),
        "direction_context":
        view.direction_context.model_dump(mode="json"),
        "direction_supervision":
        view.direction_supervision.model_dump(mode="json", exclude_none=True),
        "direction_geometry": (
            view.direction_geometry.model_dump(mode="json", exclude_none=True)
            if view.direction_geometry is not None
            else None
        ),
        "prompt_profile_id":
        view.prompt_profile_id,
        "prompt_policy_digest":
        view.prompt_policy_digest,
        "metadata": {
            **_history_projection_metadata(view.metadata, view.source_group_id, projected),
            "source_view_schema_version": view.schema_version,
            "action_supervision_location": "decision.selected_action_label",
        },
    }
    return _finalize_view(StageIFullCoTViewV1, payload)


def _project_decision_view(
    view: ReasoningTrainingViewV4,
    *,
    projected: _ProjectedInputsV1,
    parent_source_fingerprint: str,
    allow_fpv_only_decision: bool,
) -> StageIDecisionViewV1:
    """Build one Decision-only row and derive its visible authority."""

    navigation = view.reasoning_target.reasoning.navigation_context_grounding.task_direction
    authority: str | None = None
    if navigation.status == "aligned" and navigation.source == "instruction" and projected.selection.instruction:
        authority = "explicit_direction_instruction"
    elif navigation.status == "aligned" and navigation.source == "bev_route" and projected.selection.bev:
        if _has_visible_bev_route_authority(projected.evidence_catalog):
            authority = "visible_bev_route"
    if authority is None and allow_fpv_only_decision:
        authority = "fpv_behavior_cloning"
    if authority is None:
        raise ValueError("decision_answerability_failed:no_visible_authority")

    decision = view.reasoning_target.reasoning.decision
    assistant_text = _render_decision(decision.model_dump(mode="json"))
    profile = get_stage_i_modality_profile_v1(projected_profile_id(projected))
    payload: dict[str, Any] = {
        **_base_payload(
            task_type="decision",
            sample_id=f"{view.sample_id}:decision:stage_i:{profile.profile_id}",
            source_group_id=view.source_group_id,
            split_group_id=view.split_group_id,
            dataset_source=view.dataset_source,
            profile=profile,
            projected=projected,
            assistant_text=assistant_text,
            loss_spans=[_full_loss_span(assistant_text, "decision")],
            parent_artifact_kind="reasoning_v4_6",
            parent_sample_id=view.sample_id,
            parent_source_fingerprint=parent_source_fingerprint,
            parent_annotation_id=None,
            parent_audit_id=view.audit_id,
            answerability=StageIAnswerabilityV1(
                decision=True,
                reasons=(str(authority), "strong_direction_stamp_reused"),
            ),
        ),
        "decision":
        decision.model_dump(mode="json"),
        "direction_context":
        view.direction_context.model_dump(mode="json"),
        "direction_supervision":
        view.direction_supervision.model_dump(mode="json", exclude_none=True),
        "direction_geometry": (
            view.direction_geometry.model_dump(mode="json", exclude_none=True)
            if view.direction_geometry is not None
            else None
        ),
        "decision_authority":
        authority,
        "metadata": {
            **_history_projection_metadata(view.metadata, view.source_group_id, projected),
            "source_view_schema_version": view.schema_version,
            "full_cot_teacher_prefix_present": False,
        },
    }
    return _finalize_view(StageIDecisionViewV1, payload)


def _history_projection_metadata(metadata, source_group_id, projected) -> dict[str, Any]:
    """Carry explicit source-clock evidence while preserving annotation eligibility.

    This projects an already admitted source view; it does not replace images,
    rewrite evidence refs, or promote a legacy annotation to a new window.
    """
    observation = metadata.get("history_observation")
    if observation is None:
        return {}
    max_lag_ns = metadata["history_max_lag_ns"]
    validate_history_observation(observation, max_lag_ns=max_lag_ns)
    if observation["source_parent_id"] != source_group_id:
        raise ValueError("History observation must belong to the annotated source parent")
    paths = list(observation["fpv_paths"])
    bev = observation["bev"]
    if bev is not None:
        paths.append(bev["bev_image_path"])
    if (projected.images != paths
            or projected.image_roles != history_image_roles(include_bev=bev is not None)):
        raise ValueError("History projection must retain the declared nine-frame observation")
    # 只投影明确的观测 metadata；教师内部字段与未来监督继续留在原始 artifact。
    return {"history_observation": observation, "history_max_lag_ns": max_lag_ns}


def _project_visible_inputs(
    *,
    images: Iterable[str],
    image_roles: Iterable[str],
    instruction: str | None,
    model_visible_context: Mapping[str, Any],
    evidence_catalog: Iterable[EvidenceCatalogEntryV4],
    input_modalities: InputModalitiesV4,
    profile: StageIModalityProfileV1,
    instruction_training_role: StageIInstructionTrainingRoleV1,
) -> _ProjectedInputsV1:
    """Remove unselected evidence without fabricating a missing-modality input."""

    availability = StageIModalityAvailabilityV1(
        history_fpv=input_modalities.history_fpv,
        instruction=input_modalities.instruction,
        bev=input_modalities.bev,
        goal_image=input_modalities.goal_image,
        pixel_goal=input_modalities.pixel_goal,
        waypoint=input_modalities.waypoint,
    )
    selection = profile.selection()
    for name in ("history_fpv", "instruction", "bev"):
        if getattr(selection, name) and not getattr(availability, name):
            raise ValueError(
                f"selected Stage I modality is not native: {name}")
    role = _normalize_instruction_role(
        available=availability.instruction,
        selected=selection.instruction,
        requested=instruction_training_role,
    )
    pairs = list(zip(images, image_roles))
    kept_pairs = [
        (path, image_role) for path, image_role in pairs
        if image_role == "fpv.current" or
        (selection.history_fpv and image_role.startswith("fpv.history_")) or (
            selection.bev and image_role.startswith("bev."))
    ]
    kept_evidence: list[EvidenceCatalogEntryV4] = []
    for item in evidence_catalog:
        ref = str(item.ref)
        if item.modality == "fpv":
            # 当前 FPV 的 object/entity refs 不等于 image role；只有 history namespace 随 history dropout。
            if not ref.startswith("fpv.history_") or selection.history_fpv:
                kept_evidence.append(item)
        elif item.modality == "bev" and selection.bev:
            kept_evidence.append(item)
        elif item.modality == "instruction" and selection.instruction:
            kept_evidence.append(item)

    unknown_context = sorted(
        set(model_visible_context) -
        {"goal", "pixel_goal", "waypoint", "waypoints"})
    if unknown_context:
        raise ValueError(
            f"unsupported Stage I visible context keys: {unknown_context}")
    states = StageIModalityStateV1(
        **{
            name:
            _modality_state(getattr(availability, name),
                            getattr(selection, name))
            for name in ("history_fpv", "instruction", "bev", "goal_image",
                         "pixel_goal", "waypoint")
        })
    projected_modalities = InputModalitiesV4(
        history_fpv=selection.history_fpv,
        instruction=selection.instruction,
        bev=selection.bev,
        goal_image=False,
        pixel_goal=False,
        waypoint=False,
    )
    return _ProjectedInputsV1(
        availability=availability,
        selection=selection,
        states=states,
        instruction_training_role=role,
        images=[path for path, _ in kept_pairs],
        image_roles=[image_role for _, image_role in kept_pairs],
        instruction=instruction if selection.instruction else None,
        model_visible_context={},
        evidence_catalog=kept_evidence,
        input_modalities=projected_modalities,
    )


def _project_visible_inputs_v2(
    *,
    images: Iterable[str],
    image_roles: Iterable[str],
    instruction: str | None,
    model_visible_context: Mapping[str, Any],
    evidence_catalog: Iterable[EvidenceCatalogEntryV4],
    input_modalities: InputModalitiesV4,
    profile: StageIModalityProfileV2,
    instruction_training_role: StageIInstructionTrainingRoleV1,
) -> _ProjectedInputsV2:
    """Project V2 inputs while retaining a declared waypoint condition."""

    availability = StageIModalityAvailabilityV2(
        history_fpv=input_modalities.history_fpv,
        instruction=input_modalities.instruction,
        bev=input_modalities.bev,
        goal_image=input_modalities.goal_image,
        pixel_goal=input_modalities.pixel_goal,
        waypoint=input_modalities.waypoint,
    )
    selection = profile.selection()
    for name in ("history_fpv", "instruction", "bev", "waypoint"):
        if getattr(selection, name) and not getattr(availability, name):
            raise ValueError(f"selected Stage I V2 modality is not native: {name}")
    role = _normalize_instruction_role(
        available=availability.instruction,
        selected=selection.instruction,
        requested=instruction_training_role,
    )
    pairs = list(zip(images, image_roles))
    kept_pairs = [
        (path, image_role) for path, image_role in pairs
        if image_role == "fpv.current"
        or (selection.history_fpv and image_role.startswith("fpv.history_"))
        or (selection.bev and image_role.startswith("bev."))
    ]
    kept_evidence: list[EvidenceCatalogEntryV4] = []
    for item in evidence_catalog:
        ref = str(item.ref)
        if item.modality == "fpv":
            if not ref.startswith("fpv.history_") or selection.history_fpv:
                kept_evidence.append(item)
        elif item.modality == "bev" and selection.bev:
            kept_evidence.append(item)
        elif item.modality == "instruction" and selection.instruction:
            kept_evidence.append(item)
        elif item.modality == "waypoint" and selection.waypoint:
            kept_evidence.append(item)

    unsupported_context = sorted(
        set(model_visible_context) - {"goal", "pixel_goal", "waypoint", "waypoints"}
    )
    if unsupported_context:
        raise ValueError(
            f"unsupported Stage I V2 visible context keys: {unsupported_context}"
        )
    waypoint_key = (
        "waypoint" if "waypoint" in model_visible_context else "waypoints"
    )
    if selection.waypoint:
        waypoint_context = model_visible_context.get(waypoint_key)
        if not isinstance(waypoint_context, Mapping):
            raise ValueError(
                "selected Stage I V2 waypoint requires visible waypoint context"
            )
        if isinstance(waypoint_context, Mapping):
            waypoint_status = waypoint_context.get("waypoint_input")
            if not isinstance(waypoint_status, Mapping):
                raise ValueError(
                    "Stage I V2 waypoint requires waypoint_input provenance"
                )
            if waypoint_status.get("available") is not True:
                raise ValueError("Stage I V2 waypoint input is not available")
            if waypoint_status.get("independent_of_future_target") is not True:
                raise ValueError(
                    "Stage I V2 waypoint input must be independent of future target"
                )
    elif "waypoint" in model_visible_context or "waypoints" in model_visible_context:
        raise ValueError("Stage I V2 waypoint context requires waypoint selection")
    if "goal" in model_visible_context or "pixel_goal" in model_visible_context:
        raise ValueError("Stage I V2 goal context is not enabled by this profile")
    visible_context = {
        key: value
        for key, value in model_visible_context.items()
        if key in {"waypoint", "waypoints"}
    }
    states = StageIModalityStateV2(
        **{
            name:
            _modality_state(getattr(availability, name), getattr(selection, name))
            for name in (
                "history_fpv",
                "instruction",
                "bev",
                "goal_image",
                "pixel_goal",
                "waypoint",
            )
        }
    )
    projected_modalities = InputModalitiesV4(
        history_fpv=selection.history_fpv,
        instruction=selection.instruction,
        bev=selection.bev,
        goal_image=False,
        pixel_goal=False,
        waypoint=selection.waypoint,
    )
    return _ProjectedInputsV2(
        availability=availability,
        selection=selection,
        states=states,
        instruction_training_role=role,
        images=[path for path, _ in kept_pairs],
        image_roles=[image_role for _, image_role in kept_pairs],
        instruction=instruction if selection.instruction else None,
        model_visible_context=visible_context,
        evidence_catalog=kept_evidence,
        input_modalities=projected_modalities,
    )


def _normalize_instruction_role(
    *,
    available: bool,
    selected: bool,
    requested: StageIInstructionTrainingRoleV1,
) -> StageIInstructionTrainingRoleV1:
    """Fail closed on target-derived or unknown selected instructions."""

    if not available:
        if selected:
            raise ValueError("selected instruction is not native")
        return "not_present"
    if selected and requested != "primary":
        raise ValueError("selected instruction is not trusted primary input")
    return requested


def _modality_state(available: bool, selected: bool) -> str:
    """Return the unique state implied by availability and selection."""

    if selected:
        return "present"
    return "training_dropout" if available else "native_missing"


def _has_visible_bev_route_authority(
        evidence_catalog: Iterable[EvidenceCatalogEntryV4]) -> bool:
    """Return whether retained BEV declares visible conditional route authority."""

    for item in evidence_catalog:
        evidence = item.bev_route_evidence
        if item.modality != "bev" or evidence is None:
            continue
        if (evidence.route_overlay_visible
                and evidence.route_direction_authority == "visual_conditional"
                and evidence.ego_forward != "unknown"):
            return True
    return False


def _require_vqa_modalities(required_modalities: Iterable[str],
                            modalities: InputModalitiesV4) -> None:
    """Require every task-level modality after projection."""

    flags = modalities.model_dump(mode="python")
    missing = sorted(name for name in required_modalities
                     if not bool(flags.get(name, False)))
    if missing:
        raise ValueError(
            f"Stage I VQA required modalities were removed: {missing}")


def _full_cot_loss_spans(assistant_text: str,
                         action_label: str) -> list[StageILossSpanV1]:
    """Mask only the final Decision action value in a FullCoT row."""

    value = json.dumps(action_label, ensure_ascii=False)
    needle = f'"selected_action_label":{value}'
    if assistant_text.count(needle) != 1:
        raise ValueError("selected_action_label span is missing or ambiguous")
    value_start = assistant_text.index(needle) + len(
        '"selected_action_label":')
    value_end = value_start + len(value)
    spans: list[StageILossSpanV1] = []
    if value_start:
        spans.append(
            StageILossSpanV1(start=0,
                             end=value_start,
                             scale=1.0,
                             channel="full_cot"))
    spans.append(
        StageILossSpanV1(start=value_start,
                         end=value_end,
                         scale=0.0,
                         channel="full_cot"))
    if value_end < len(assistant_text):
        spans.append(
            StageILossSpanV1(start=value_end,
                             end=len(assistant_text),
                             scale=1.0,
                             channel="full_cot"))
    return spans


def _build_projection_drop(
    *,
    task_type: str,
    parent_artifact_kind: str,
    parent_sample_id: str,
    source_group_id: str,
    split_group_id: str,
    dataset_source: str,
    modality_profile_id: str,
    error: Exception,
) -> StageIProjectionDropV1:
    """Build one immutable projection-drop audit from a bounded error taxonomy."""

    detail = str(error)[:1000] or error.__class__.__name__
    payload: dict[str, Any] = {
        "schema_version": STAGE_I_PROJECTION_DROP_SCHEMA_VERSION,
        "task_type": task_type,
        "parent_artifact_kind": parent_artifact_kind,
        "parent_sample_id": parent_sample_id,
        "source_group_id": source_group_id,
        "split_group_id": split_group_id,
        "dataset_source": dataset_source,
        "requested_modality_profile_id": modality_profile_id,
        "reason_code": _projection_drop_reason(detail),
        "detail": detail,
        "projection_policy_id": STAGE_I_PROJECTION_POLICY_ID,
        "projection_policy_version": STAGE_I_PROJECTION_POLICY_VERSION,
    }
    payload["drop_digest"] = stage_i_projection_digest_v1(payload)
    return StageIProjectionDropV1.model_validate(payload)


def _projection_drop_reason(detail: str) -> str:
    """Map implementation diagnostics onto a stable drop-reason registry."""

    rules = (
        ("full_cot_answerability_failed:dropped_native_",
         "full_cot_native_modality_dropout"),
        ("full_cot_answerability_failed:", "full_cot_answerability_unproven"),
        ("VQA required modalities were removed",
         "vqa_required_modality_removed"),
        ("VQA projection removed annotation evidence",
         "vqa_annotation_evidence_removed"),
        ("training_export_disabled", "vqa_capability_disabled"),
        ("decision_answerability_failed:no_visible_authority",
         "decision_no_visible_authority"),
        ("selected Stage I modality is not native", "modality_not_native"),
        ("selected instruction is not trusted primary",
         "untrusted_instruction"),
        ("formal training-ready", "parent_not_formal_training"),
    )
    for needle, reason in rules:
        if needle in detail:
            return reason
    return "invalid_parent_or_projection"


def _raw_mapping(value: Any) -> Mapping[str, Any]:
    """Return a JSON-like parent mapping without mutating typed artifacts."""

    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _full_loss_span(assistant_text: str, channel: str) -> StageILossSpanV1:
    """Return one full-target binary loss span."""

    return StageILossSpanV1(start=0,
                            end=len(assistant_text),
                            scale=1.0,
                            channel=channel)  # type: ignore[arg-type]


def _render_decision(decision: Mapping[str, Any]) -> str:
    """Render one canonical compact Decision-only assistant target."""

    payload = json.dumps(dict(decision),
                         ensure_ascii=False,
                         separators=(",", ":"))
    return f"<Decision>{payload}</Decision>"


def projected_profile_id(projected: _ProjectedInputsV1) -> str:
    """Resolve the unique registered profile for an internal selection."""

    selection = projected.selection
    matches = [
        profile.profile_id
        for profile in STAGE_I_MODALITY_PROFILES_V1.values()
        if profile.selection() == selection
    ]
    if len(matches) != 1:
        raise RuntimeError("Stage I selection does not map to one profile")
    return matches[0]


def projected_profile_id_v2(projected: _ProjectedInputsV2) -> str:
    """Resolve the unique registered V2 profile for one selection."""

    matches = [
        profile.profile_id
        for profile in STAGE_I_MODALITY_PROFILES_V2.values()
        if profile.selection() == projected.selection
    ]
    if len(matches) != 1:
        raise RuntimeError("Stage I V2 selection does not map to one profile")
    return matches[0]


def _base_payload(
    *,
    task_type: str,
    sample_id: str,
    source_group_id: str,
    split_group_id: str,
    dataset_source: str,
    profile: StageIModalityProfileV1,
    projected: _ProjectedInputsV1,
    assistant_text: str,
    loss_spans: list[StageILossSpanV1],
    parent_artifact_kind: str,
    parent_sample_id: str,
    parent_source_fingerprint: str,
    parent_annotation_id: str | None,
    parent_audit_id: str | None,
    answerability: StageIAnswerabilityV1,
) -> dict[str, Any]:
    """Build common JSON fields before the projection digest is added."""

    return {
        "schema_version":
        STAGE_I_TRAINING_VIEW_SCHEMA_VERSION,
        "task_type":
        task_type,
        "sample_id":
        sample_id,
        "source_group_id":
        source_group_id,
        "split_group_id":
        split_group_id,
        "dataset_source":
        dataset_source,
        "modality_profile_id":
        profile.profile_id,
        "native_availability":
        projected.availability.model_dump(mode="json"),
        "training_selection":
        projected.selection.model_dump(mode="json"),
        "modality_state":
        projected.states.model_dump(mode="json"),
        "instruction_training_role":
        projected.instruction_training_role,
        "answerability":
        answerability.model_dump(mode="json"),
        "images":
        list(projected.images),
        "image_roles":
        list(projected.image_roles),
        "instruction":
        projected.instruction,
        "model_visible_context":
        dict(projected.model_visible_context),
        "evidence_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in projected.evidence_catalog
        ],
        "input_modalities":
        projected.input_modalities.model_dump(mode="json"),
        "assistant_text":
        assistant_text,
        "loss_span_policy_id":
        STAGE_I_LOSS_SPAN_POLICY_ID,
        "loss_spans": [span.model_dump(mode="json") for span in loss_spans],
        "parent_artifact_kind":
        parent_artifact_kind,
        "parent_sample_id":
        parent_sample_id,
        "parent_source_fingerprint":
        parent_source_fingerprint,
        "parent_annotation_id":
        parent_annotation_id,
        "parent_audit_id":
        parent_audit_id,
        "projection_policy_id":
        STAGE_I_PROJECTION_POLICY_ID,
        "projection_policy_version":
        STAGE_I_PROJECTION_POLICY_VERSION,
    }


def _base_payload_v2(
    *,
    task_type: str,
    sample_id: str,
    source_group_id: str,
    split_group_id: str,
    dataset_source: str,
    profile: StageIModalityProfileV2,
    projected: _ProjectedInputsV2,
    assistant_text: str,
    loss_spans: list[dict[str, Any]],
    parent_artifact_kind: str,
    parent_sample_id: str,
    parent_source_fingerprint: str,
    parent_audit_id: str | None,
    answerability: StageIAnswerabilityV2,
) -> dict[str, Any]:
    """Build the V2 common envelope for the retained input projection."""

    return {
        "schema_version": STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION,
        "task_type": task_type,
        "sample_id": sample_id,
        "source_group_id": source_group_id,
        "split_group_id": split_group_id,
        "dataset_source": dataset_source,
        "modality_profile_id": profile.profile_id,
        "native_availability": projected.availability.model_dump(mode="json"),
        "training_selection": projected.selection.model_dump(mode="json"),
        "modality_state": projected.states.model_dump(mode="json"),
        "instruction_training_role": projected.instruction_training_role,
        "answerability": answerability.model_dump(mode="json"),
        "images": list(projected.images),
        "image_roles": list(projected.image_roles),
        "instruction": projected.instruction,
        "model_visible_context": dict(projected.model_visible_context),
        "evidence_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in projected.evidence_catalog
        ],
        "input_modalities": projected.input_modalities.model_dump(mode="json"),
        "assistant_text": assistant_text,
        "loss_span_policy_id": STAGE_I_LOSS_SPAN_POLICY_V2_ID,
        "loss_spans": loss_spans,
        "parent_artifact_kind": parent_artifact_kind,
        "parent_sample_id": parent_sample_id,
        "parent_source_fingerprint": parent_source_fingerprint,
        "parent_audit_id": parent_audit_id,
        "projection_policy_id": STAGE_I_PROJECTION_POLICY_V2_ID,
        "projection_policy_version": STAGE_I_PROJECTION_POLICY_V2_VERSION,
    }


def _finalize_view(model_type, payload: dict[str, Any]):
    """Attach the canonical projection digest and validate the final model."""

    normalized = {
        key: value
        for key, value in payload.items() if value is not None
    }
    normalized["projection_digest"] = stage_i_projection_digest_v1(normalized)
    return model_type.model_validate(normalized)


def _project_full_cot_view_v2(
    view: ReasoningTrainingViewV4,
    *,
    projected: _ProjectedInputsV2,
    parent_source_fingerprint: str,
) -> StageIFullCoTViewV2:
    """Build V2 FullCoT after retaining every selected native condition."""

    availability = projected.availability.model_dump(mode="python")
    selection = projected.selection.model_dump(mode="python")
    for name in (
        "history_fpv",
        "instruction",
        "bev",
        "goal_image",
        "pixel_goal",
        "waypoint",
    ):
        if bool(availability[name]) != bool(selection[name]):
            raise ValueError(f"full_cot_answerability_failed:dropped_native_{name}")
    parent_evidence_refs = {str(item.ref) for item in view.evidence_catalog}
    projected_evidence_refs = {str(item.ref) for item in projected.evidence_catalog}
    if projected_evidence_refs != parent_evidence_refs:
        raise ValueError("full_cot_answerability_failed:evidence_catalog_changed")
    profile = get_stage_i_modality_profile_v2(projected_profile_id_v2(projected))
    action = str(view.direction_supervision.label_name)
    assistant_text = view.assistant_text
    payload = {
        **_base_payload_v2(
            task_type="full_cot",
            sample_id=f"{view.sample_id}:stage_i:{profile.profile_id}",
            source_group_id=view.source_group_id,
            split_group_id=view.split_group_id,
            dataset_source=view.dataset_source,
            profile=profile,
            projected=projected,
            assistant_text=assistant_text,
            loss_spans=_full_loss_span_v2(assistant_text, "full_cot"),
            parent_artifact_kind="reasoning_v4_6",
            parent_sample_id=view.sample_id,
            parent_source_fingerprint=parent_source_fingerprint,
            parent_audit_id=view.audit_id,
            answerability=StageIAnswerabilityV2(
                full_cot=True,
                reasons=("all_native_modalities_retained", "canonical_full_cot_reused"),
            ),
        ),
        "reasoning_target": view.reasoning_target.model_dump(
            mode="json", exclude_none=True
        ),
        "direction_context": view.direction_context.model_dump(mode="json"),
        "direction_supervision": view.direction_supervision.model_dump(
            mode="json", exclude_none=True
        ),
        "direction_geometry": (
            view.direction_geometry.model_dump(mode="json", exclude_none=True)
            if view.direction_geometry is not None
            else None
        ),
        "prompt_profile_id": view.prompt_profile_id,
        "prompt_policy_digest": view.prompt_policy_digest,
        "metadata": {
            **_history_projection_metadata(view.metadata, view.source_group_id, projected),
            "source_view_schema_version": view.schema_version,
            "action_supervision_location": "decision.selected_action_label",
        },
    }
    if action != str(view.reasoning_target.reasoning.decision.selected_action_label):
        raise ValueError("V2 FullCoT Decision does not match strong direction")
    return _finalize_view_v2(StageIFullCoTViewV2, payload)


def project_stage_i_reasoning_view_v2(
    view: ReasoningTrainingViewV4 | Mapping[str, Any],
    *,
    task_type: str,
    modality_profile_id: str,
    instruction_training_role: StageIInstructionTrainingRoleV1,
) -> StageIFullCoTViewV2 | StageIReasoningDropoutViewV2:
    """Project one formal FullCoT parent into the V2 training contract.

    V2 owns its modality projection.  V1 remains available for legacy rows and
    is not used as an intermediate because its waypoint selection is disabled.
    A dropout row is derived from the same validated V2 FullCoT object.
    """

    if task_type not in {"full_cot", "reasoning_dropout"}:
        raise ValueError(
            f"unsupported Stage I V2 Reasoning task_type: {task_type}")
    typed = view if isinstance(view, ReasoningTrainingViewV4) else ReasoningTrainingViewV4.model_validate(view)
    if typed.release_track != "formal_training" or not typed.artifact_training_eligible:
        raise ValueError("Stage I accepts only formal training-ready Reasoning views")
    if typed.audit_id is None:
        raise ValueError("Stage I Reasoning projection requires action audit_id")
    parent_source_fingerprint = str(typed.metadata.get("source_fingerprint") or "")
    if len(parent_source_fingerprint) != 64:
        raise ValueError("Stage I Reasoning parent source_fingerprint is missing")
    profile = get_stage_i_modality_profile_v2(modality_profile_id)
    projected = _project_visible_inputs_v2(
        images=typed.images,
        image_roles=typed.image_roles,
        instruction=typed.instruction,
        model_visible_context=typed.model_visible_context,
        evidence_catalog=typed.evidence_catalog,
        input_modalities=typed.input_modalities,
        profile=profile,
        instruction_training_role=instruction_training_role,
    )
    full = _project_full_cot_view_v2(
        typed,
        projected=projected,
        parent_source_fingerprint=parent_source_fingerprint,
    )
    if task_type == "full_cot":
        return full
    return project_reasoning_dropout_from_full_v2(full)


def project_stage_i_reasoning_view_with_audit_v2(
    view: ReasoningTrainingViewV4 | Mapping[str, Any],
    *,
    modality_profile_id: str,
    instruction_training_role: StageIInstructionTrainingRoleV1,
) -> StageIFullCoTViewV2 | StageIProjectionDropV1:
    """Project one V2 FullCoT parent or return a structured drop audit.

    Reasoning dropout is deliberately absent from this batch-facing API.  It
    is legal only as a deterministic projection of an already accepted V2
    FullCoT parent after the epoch sampler selects the no-CoT route.

    Args:
        view: Formal Reasoning V4.6 training view or raw payload.
        modality_profile_id: Requested registered Stage I input profile.
        instruction_training_role: Provenance of the selected instruction.

    Returns:
        A validated V2 FullCoT view or a V1-compatible structured drop audit.
    """

    try:
        return project_stage_i_reasoning_view_v2(
            view,
            task_type="full_cot",
            modality_profile_id=modality_profile_id,
            instruction_training_role=instruction_training_role,
        )
    except (TypeError, ValueError) as exc:
        raw = _raw_mapping(view)
        return _build_projection_drop(
            task_type="full_cot",
            parent_artifact_kind="reasoning_v4_6",
            parent_sample_id=str(
                raw.get("sample_id") or "unknown_reasoning_parent"),
            source_group_id=str(
                raw.get("source_group_id") or "unknown_source_group"),
            split_group_id=str(
                raw.get("split_group_id") or "unknown_split_group"),
            dataset_source=str(raw.get("dataset_source") or "unknown_dataset"),
            modality_profile_id=modality_profile_id,
            error=exc,
        )


def project_stage_i_reasoning_dropout_from_full_v2(
    full: StageIFullCoTViewV2 | Mapping[str, Any],
) -> StageIReasoningDropoutViewV2:
    """Project one validated V2 FullCoT view to its direct Decision target.

    This is a deterministic projection primitive, not an instruction to
    materialize a second dataset.  Stage I V3 calls it only after the epoch
    sampler has routed a FullCoT appearance to the no-CoT path.

    Args:
        full: A validated V2 FullCoT exchange view or its JSON-compatible
            payload.

    Returns:
        The unique direct ``<Decision>`` projection with immutable lineage to
        ``full``.

    Raises:
        TypeError: If ``full`` is neither a V2 object nor a mapping.
        ValueError: If a mapping is not a V2 FullCoT view.
    """

    if isinstance(full, Mapping):
        parsed = parse_stage_i_training_view_v2(dict(full))
    elif isinstance(full, StageIFullCoTViewV2):
        parsed = full
    else:
        raise TypeError("full must be a StageIFullCoTViewV2 or mapping")
    if not isinstance(parsed, StageIFullCoTViewV2):
        raise ValueError("reasoning route requires a V2 FullCoT parent")
    return project_reasoning_dropout_from_full_v2(parsed)


def project_stage_i_vqa_view_v2(
    source: VQASourceV2 | Mapping[str, Any],
    annotation: VQAAnnotationV2 | Mapping[str, Any],
    *,
    capabilities: VQACapabilityConfigV2,
    modality_profile_id: str,
    instruction_training_role: StageIInstructionTrainingRoleV1,
) -> StageIVQAViewV2:
    """Wrap one accepted VQA V2 annotation in the V2 all-target envelope."""

    legacy = project_stage_i_vqa_view_v1(
        source,
        annotation,
        capabilities=capabilities,
        modality_profile_id=modality_profile_id,
        instruction_training_role=instruction_training_role,
    )
    payload = legacy.model_dump(mode="json", exclude_none=True)
    payload.update({
        "schema_version":
        STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION,
        "loss_span_policy_id":
        STAGE_I_LOSS_SPAN_POLICY_V2_ID,
        "loss_spans":
        _full_loss_span_v2(legacy.assistant_text, "vqa"),
        "projection_policy_id":
        STAGE_I_PROJECTION_POLICY_V2_ID,
        "projection_policy_version":
        STAGE_I_PROJECTION_POLICY_V2_VERSION,
        "answerability":
        StageIAnswerabilityV2(
            vqa=True,
            reasons=("required_modalities_retained",
                     "annotation_evidence_retained"),
        ).model_dump(mode="json"),
        "metadata": {
            **legacy.metadata,
            "source_stage_i_projection_digest_v1":
            legacy.projection_digest,
        },
    })
    return _finalize_view_v2(StageIVQAViewV2, payload)


def project_stage_i_vqa_view_with_audit_v2(
    source: VQASourceV2 | Mapping[str, Any],
    annotation: VQAAnnotationV2 | Mapping[str, Any],
    *,
    capabilities: VQACapabilityConfigV2,
    modality_profile_id: str,
    instruction_training_role: StageIInstructionTrainingRoleV1,
) -> StageIVQAViewV2 | StageIProjectionDropV1:
    """Project one V2 VQA parent or return a structured drop audit.

    Args:
        source: Frozen VQA source.
        annotation: Accepted typed VQA annotation.
        capabilities: Exact active-task registry.
        modality_profile_id: Requested registered Stage I input profile.
        instruction_training_role: Provenance of the selected instruction.

    Returns:
        A validated V2 VQA view or a V1-compatible structured drop audit.
    """

    try:
        return project_stage_i_vqa_view_v2(
            source,
            annotation,
            capabilities=capabilities,
            modality_profile_id=modality_profile_id,
            instruction_training_role=instruction_training_role,
        )
    except (TypeError, ValueError) as exc:
        raw_annotation = _raw_mapping(annotation)
        raw_source = _raw_mapping(source)
        observation = _raw_mapping(raw_source.get("observation", {}))
        return _build_projection_drop(
            task_type="vqa",
            parent_artifact_kind="vqa_v2_0",
            parent_sample_id=str(
                raw_annotation.get("annotation_id")
                or "unknown_vqa_annotation"),
            source_group_id=str(
                raw_annotation.get("source_group_id")
                or observation.get("source_group_id")
                or "unknown_source_group"),
            split_group_id=str(
                observation.get("split_group_id") or "unknown_split_group"),
            dataset_source=str(
                observation.get("dataset_source") or "unknown_dataset"),
            modality_profile_id=modality_profile_id,
            error=exc,
        )


def upgrade_stage_i_full_cot_view_v2(
    legacy: StageIFullCoTViewV1 | Mapping[str, Any], ) -> StageIFullCoTViewV2:
    """Upgrade one validated V1 FullCoT view without changing visible inputs.

    This migration is intentionally limited to FullCoT.  It cannot promote a
    V1 Decision row into reasoning dropout because that would lose the signed
    parent lineage required by V2.  The no-CoT row must still be derived from
    the returned V2 FullCoT object after epoch routing.

    Args:
        legacy: Validated V1 FullCoT object or its JSON-compatible payload.

    Returns:
        The equivalent V2 FullCoT view with all canonical tokens supervised.

    Raises:
        ValueError: If the input is not a valid V1 FullCoT view.
    """

    parsed = (parse_stage_i_training_view_v1(dict(legacy)) if isinstance(
        legacy, Mapping) else legacy)
    if not isinstance(parsed, StageIFullCoTViewV1):
        raise ValueError("Stage I V2 migration accepts only a V1 FullCoT view")

    payload = parsed.model_dump(mode="json", exclude_none=True)
    payload.update({
        "schema_version":
        STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION,
        "loss_span_policy_id":
        STAGE_I_LOSS_SPAN_POLICY_V2_ID,
        "loss_spans":
        _full_loss_span_v2(parsed.assistant_text, "full_cot"),
        "projection_policy_id":
        STAGE_I_PROJECTION_POLICY_V2_ID,
        "projection_policy_version":
        STAGE_I_PROJECTION_POLICY_V2_VERSION,
        "answerability":
        StageIAnswerabilityV2(
            full_cot=True,
            reasons=("all_native_modalities_retained",
                     "canonical_full_cot_reused"),
        ).model_dump(mode="json"),
        "metadata": {
            **parsed.metadata,
            "source_stage_i_projection_digest_v1": parsed.projection_digest,
            "decision_loss_policy": "all_canonical_tokens",
        },
    })
    return _finalize_view_v2(StageIFullCoTViewV2, payload)


def _full_loss_span_v2(assistant_text: str,
                       channel: str) -> list[dict[str, Any]]:
    """Return V2's required all-canonical-target loss partition."""

    return [{
        "start": 0,
        "end": len(assistant_text),
        "scale": 1.0,
        "channel": channel
    }]


def _finalize_view_v2(model_type, payload: dict[str, Any]):
    """Attach a V2 projection digest and validate the resulting exchange view."""

    normalized = {
        key: value
        for key, value in payload.items() if value is not None
    }
    normalized.pop("projection_digest", None)
    normalized["projection_digest"] = stage_i_projection_digest_v2(normalized)
    return model_type.model_validate(normalized)


__all__ = [
    "STAGE_I_MODALITY_PROFILES_V1",
    "STAGE_I_MODALITY_PROFILES_V2",
    "StageIModalityProfileV1",
    "StageIModalityProfileV2",
    "get_stage_i_modality_profile_v1",
    "get_stage_i_modality_profile_v2",
    "project_stage_i_reasoning_view_v1",
    "project_stage_i_reasoning_view_with_audit_v1",
    "project_stage_i_reasoning_view_v2",
    "project_stage_i_reasoning_view_with_audit_v2",
    "project_stage_i_reasoning_dropout_from_full_v2",
    "project_stage_i_vqa_view_v1",
    "project_stage_i_vqa_view_with_audit_v1",
    "project_stage_i_vqa_view_v2",
    "project_stage_i_vqa_view_with_audit_v2",
    "upgrade_stage_i_full_cot_view_v2",
]
