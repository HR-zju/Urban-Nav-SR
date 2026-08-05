"""Canonical exchange contracts for UrbanNav Stage I SFT views.

Stage I views are deterministic projections over immutable Reasoning V4.6 or
VQA V2 artifacts.  They never mutate the parent annotation and never expose
private trajectory sidecars to the student.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mcp_server.contracts.reasoning import ReasonCode
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionContextV4,
    DirectionSupervisionV4,
    EvidenceCatalogEntryV4,
    InputModalitiesV4,
)
from mcp_server.contracts.reasoning_v4 import DecisionV4, ReasoningTargetV4
from mcp_server.contracts.vqa_v2 import VQAQuestionV2


STAGE_I_TRAINING_VIEW_SCHEMA_VERSION = "urbannav_stage_i_training_view_v1"
STAGE_I_PROJECTION_DROP_SCHEMA_VERSION = "urbannav_stage_i_projection_drop_v1"
STAGE_I_PROJECTION_POLICY_ID = "urbannav_stage_i_safe_projection_v1"
STAGE_I_PROJECTION_POLICY_VERSION = "1.0"
STAGE_I_LOSS_SPAN_POLICY_ID = "urbannav_stage_i_char_spans_v1"
STAGE_I_INSTRUCTION_PROVENANCE_SCHEMA_VERSION = (
    "urbannav_stage_i_instruction_provenance_v1"
)
# V2 keeps the V1 artifacts readable, but gives FullCoT and its direct
# projection an explicit loss and lineage contract.  Do not reuse V1 digests.
STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION = "urbannav_stage_i_training_view_v2"
STAGE_I_PROJECTION_POLICY_V2_ID = "urbannav_stage_i_reasoning_dropout_v2"
STAGE_I_PROJECTION_POLICY_V2_VERSION = "2.0"
STAGE_I_LOSS_SPAN_POLICY_V2_ID = "urbannav_stage_i_all_target_spans_v2"

StageITaskTypeV1 = Literal["full_cot", "vqa", "decision"]
StageIModalityStateValueV1 = Literal[
    "present",
    "native_missing",
    "training_dropout",
]
StageIInstructionTrainingRoleV1 = Literal[
    "primary",
    "audit_only",
    "unknown",
    "not_present",
]
StageIModalityProfileIdV1 = Literal[
    "fpv_current_v1",
    "fpv_temporal_v1",
    "fpv_instruction_v1",
    "fpv_bev_v1",
    "fpv_instruction_bev_v1",
    "fpv_temporal_instruction_v1",
    "fpv_temporal_bev_v1",
    "fpv_temporal_instruction_bev_v1",
]
StageIParentArtifactKindV1 = Literal["reasoning_v4_6", "vqa_v2_0"]
StageILossChannelV1 = Literal["full_cot", "vqa", "decision"]
StageIVQALabelSourceV1 = Literal["derived_gt", "lvlm_weak", "external_gt"]
StageITaskTypeV2 = Literal["full_cot", "reasoning_dropout", "vqa"]
StageILossChannelV2 = Literal["full_cot", "reasoning_dropout", "vqa"]


class StageIInstructionProvenanceV1(BaseModel):
    """Bind one source revision to an explicit instruction training role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        STAGE_I_INSTRUCTION_PROVENANCE_SCHEMA_VERSION
    ] = STAGE_I_INSTRUCTION_PROVENANCE_SCHEMA_VERSION
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    instruction_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    training_role: StageIInstructionTrainingRoleV1
    provenance_source: ReasonCode
    policy_id: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=64)
    entry_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_provenance(self) -> "StageIInstructionProvenanceV1":
        """Require instruction presence and the immutable digest to agree."""

        if self.training_role == "not_present":
            if self.instruction_sha256 is not None:
                raise ValueError(
                    "not_present instruction provenance cannot carry a digest"
                )
        elif self.instruction_sha256 is None:
            raise ValueError(
                "present instruction provenance requires instruction_sha256"
            )
        expected = stage_i_projection_digest_v1(
            self.model_dump(mode="json", exclude={"entry_digest"})
        )
        if self.entry_digest != expected:
            raise ValueError(
                "Stage I instruction provenance digest mismatch"
            )
        return self


class StageIProjectionDropV1(BaseModel):
    """Record one deterministic Stage I projection rejection outside training.

    Args:
        schema_version: Exact drop-audit schema stamp.
        task_type: Requested physical task.
        parent_artifact_kind: Requested parent artifact family.
        parent_sample_id: Best available parent identity.
        source_group_id: Best available source-group identity.
        split_group_id: Best available leakage-safe split identity.
        dataset_source: Best available dataset family.
        requested_modality_profile_id: Requested profile before validation.
        reason_code: Stable bounded rejection category.
        detail: Bounded diagnostic detail for engineering review.
        projection_policy_id: Exact projection policy.
        projection_policy_version: Exact projection policy version.
        drop_digest: SHA-256 over all other drop fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_I_PROJECTION_DROP_SCHEMA_VERSION] = (
        STAGE_I_PROJECTION_DROP_SCHEMA_VERSION
    )
    task_type: StageITaskTypeV1
    parent_artifact_kind: StageIParentArtifactKindV1
    parent_sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    requested_modality_profile_id: str = Field(min_length=1, max_length=128)
    reason_code: ReasonCode
    detail: str = Field(min_length=1, max_length=1000)
    projection_policy_id: Literal[STAGE_I_PROJECTION_POLICY_ID] = STAGE_I_PROJECTION_POLICY_ID
    projection_policy_version: Literal[STAGE_I_PROJECTION_POLICY_VERSION] = STAGE_I_PROJECTION_POLICY_VERSION
    drop_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_drop_digest(self) -> "StageIProjectionDropV1":
        """Require the drop audit to remain immutable.

        Returns:
            Validated drop audit.

        Raises:
            ValueError: If the digest does not match the payload.
        """

        expected = stage_i_projection_digest_v1(
            self.model_dump(mode="json", exclude={"drop_digest"})
        )
        if self.drop_digest != expected:
            raise ValueError("Stage I drop_digest does not match payload")
        return self


class StageIModalityAvailabilityV1(BaseModel):
    """Record native optional-modality availability before training dropout.

    Args:
        current_fpv: Current FPV is mandatory for every Stage I view.
        history_fpv: At least one history FPV image is native input.
        instruction: A prompt-visible instruction is native input.
        bev: A prompt-visible BEV image is native input.
        goal_image: A goal image exists but is unsupported in Stage I v1.
        pixel_goal: A pixel goal exists but is unsupported in Stage I v1.
        waypoint: A waypoint exists but is dormant in Stage I v1.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    current_fpv: Literal[True] = True
    history_fpv: bool = False
    instruction: bool = False
    bev: bool = False
    goal_image: bool = False
    pixel_goal: bool = False
    waypoint: bool = False


class StageIModalitySelectionV1(BaseModel):
    """Record the optional modalities retained by one training view.

    Args:
        current_fpv: Current FPV is always retained.
        history_fpv: Retain native history FPV images.
        instruction: Retain the native instruction.
        bev: Retain native BEV images.
        goal_image: Always false in Stage I v1.
        pixel_goal: Always false in Stage I v1.
        waypoint: Always false in Stage I v1.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    current_fpv: Literal[True] = True
    history_fpv: bool = False
    instruction: bool = False
    bev: bool = False
    goal_image: Literal[False] = False
    pixel_goal: Literal[False] = False
    waypoint: Literal[False] = False


class StageIModalityStateV1(BaseModel):
    """Expose whether each optional modality is present, absent, or dropped.

    Args:
        history_fpv: State of history FPV.
        instruction: State of the instruction.
        bev: State of BEV.
        goal_image: State of goal-image input.
        pixel_goal: State of pixel-goal input.
        waypoint: State of waypoint input.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    history_fpv: StageIModalityStateValueV1
    instruction: StageIModalityStateValueV1
    bev: StageIModalityStateValueV1
    goal_image: StageIModalityStateValueV1
    pixel_goal: StageIModalityStateValueV1
    waypoint: StageIModalityStateValueV1


class StageIAnswerabilityV1(BaseModel):
    """Store task-specific safe-projection gates.

    Args:
        full_cot: The complete six-section target remains answerable.
        vqa: The atomic question retains all required visible evidence.
        decision: The Decision target has an approved visible authority.
        reasons: Deterministic admission facts, never free-form rationale.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    full_cot: bool = False
    vqa: bool = False
    decision: bool = False
    reasons: tuple[ReasonCode, ...] = Field(default_factory=tuple, max_length=16)


class StageILossSpanV1(BaseModel):
    """Assign one binary LM-loss channel to a character interval.

    The interval is half-open: ``[start, end)``.

    Args:
        start: Inclusive Unicode character offset.
        end: Exclusive Unicode character offset.
        scale: Binary Stage I token-loss scale.
        channel: Semantic loss channel for audit and metrics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    scale: Literal[0.0, 1.0]
    channel: StageILossChannelV1

    @model_validator(mode="after")
    def validate_interval(self) -> "StageILossSpanV1":
        """Require a non-empty half-open interval.

        Returns:
            Validated span.

        Raises:
            ValueError: If the end does not follow the start.
        """

        if self.end <= self.start:
            raise ValueError("loss span end must be greater than start")
        return self


class StageITrainingViewBaseV1(BaseModel):
    """Common immutable envelope for every Stage I exchange view.

    Args:
        schema_version: Exact Stage I view version.
        task_type: Discriminator for FullCoT, VQA, or Decision.
        sample_id: Unique projected-view identity.
        source_group_id: Parent source-group identity.
        split_group_id: Leakage-safe split identity.
        dataset_source: Dataset family.
        modality_profile_id: Exact registered Stage I modality profile.
        native_availability: Native input availability before dropout.
        training_selection: Modalities retained in this view.
        modality_state: Derived present/missing/dropout states.
        instruction_training_role: Provenance gate for selected instructions.
        answerability: Task-specific deterministic admission result.
        images: Ordered projected images.
        image_roles: Stable role for every projected image.
        instruction: Optional retained native instruction.
        model_visible_context: Retained non-private context.
        evidence_catalog: Evidence remaining visible after projection.
        input_modalities: Exact projected observation flags.
        assistant_text: Exact assistant target.
        loss_span_policy_id: Exact character-span policy.
        loss_spans: Complete ordered partition of assistant text.
        parent_artifact_kind: Parent canonical artifact family.
        parent_sample_id: Parent view or annotation identity.
        parent_source_fingerprint: Immutable parent source fingerprint.
        parent_annotation_id: VQA parent annotation when applicable.
        parent_audit_id: Reasoning action audit when applicable.
        projection_policy_id: Exact projection-policy identity.
        projection_policy_version: Exact projection-policy version.
        projection_digest: SHA-256 over this view excluding the digest itself.
        metadata: Non-semantic audit metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_I_TRAINING_VIEW_SCHEMA_VERSION] = (
        STAGE_I_TRAINING_VIEW_SCHEMA_VERSION
    )
    task_type: StageITaskTypeV1
    sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: StageIModalityProfileIdV1
    native_availability: StageIModalityAvailabilityV1
    training_selection: StageIModalitySelectionV1
    modality_state: StageIModalityStateV1
    instruction_training_role: StageIInstructionTrainingRoleV1
    answerability: StageIAnswerabilityV1
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, min_length=1, max_length=4096)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(min_length=1, max_length=64)
    input_modalities: InputModalitiesV4
    assistant_text: str = Field(min_length=1, max_length=20000)
    loss_span_policy_id: Literal[STAGE_I_LOSS_SPAN_POLICY_ID] = STAGE_I_LOSS_SPAN_POLICY_ID
    loss_spans: list[StageILossSpanV1] = Field(min_length=1, max_length=8)
    parent_artifact_kind: StageIParentArtifactKindV1
    parent_sample_id: str = Field(min_length=1, max_length=480)
    parent_source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    parent_annotation_id: str | None = Field(default=None, min_length=1, max_length=400)
    parent_audit_id: str | None = Field(default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    projection_policy_id: Literal[STAGE_I_PROJECTION_POLICY_ID] = STAGE_I_PROJECTION_POLICY_ID
    projection_policy_version: Literal[STAGE_I_PROJECTION_POLICY_VERSION] = STAGE_I_PROJECTION_POLICY_VERSION
    projection_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_common_view(self) -> "StageITrainingViewBaseV1":
        """Validate media, modality states, loss spans, and fingerprint.

        Returns:
            Validated Stage I view.

        Raises:
            ValueError: If any cross-field invariant is violated.
        """

        if len(self.images) != len(self.image_roles):
            raise ValueError("Stage I images and image_roles must align")
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("Stage I image_roles must be unique")
        if self.image_roles.count("fpv.current") != 1:
            raise ValueError("Stage I requires exactly one fpv.current")
        self._validate_modality_contract()
        self._validate_loss_spans()
        self._validate_parent_identity()
        expected = stage_i_projection_digest_v1(
            self.model_dump(mode="json", exclude={"projection_digest"})
        )
        if self.projection_digest != expected:
            raise ValueError("Stage I projection_digest does not match payload")
        return self

    def _validate_modality_contract(self) -> None:
        """Require availability, selection, state, and visible inputs to agree."""

        availability = self.native_availability.model_dump(mode="python")
        selection = self.training_selection.model_dump(mode="python")
        states = self.modality_state.model_dump(mode="python")
        for name in (
            "history_fpv",
            "instruction",
            "bev",
            "goal_image",
            "pixel_goal",
            "waypoint",
        ):
            if selection[name] and not availability[name]:
                raise ValueError(f"selected Stage I modality is not native: {name}")
            expected = (
                "present"
                if selection[name]
                else "training_dropout"
                if availability[name]
                else "native_missing"
            )
            if states[name] != expected:
                raise ValueError(f"Stage I modality_state mismatch: {name}={states[name]}, expected={expected}")

        # Stage I v1 明确关闭 goal/pixel/waypoint，防止 dormant 输入静默进入训练。
        if selection["goal_image"] or selection["pixel_goal"] or selection["waypoint"]:
            raise ValueError("unsupported Stage I modality was selected")
        flags = self.input_modalities.model_dump(mode="python")
        # canonical observation contract 使用 fpv，Stage I 为区分时序帧显式命名为 current_fpv。
        flag_names = {
            "current_fpv": "fpv",
            "history_fpv": "history_fpv",
            "instruction": "instruction",
            "bev": "bev",
            "goal_image": "goal_image",
            "pixel_goal": "pixel_goal",
            "waypoint": "waypoint",
        }
        for name, flag_name in flag_names.items():
            if bool(flags[flag_name]) != bool(selection[name]):
                raise ValueError(f"Stage I input_modalities mismatch: {name}")
        history_roles = [role for role in self.image_roles if role.startswith("fpv.history_")]
        bev_roles = [role for role in self.image_roles if role.startswith("bev.")]
        if bool(history_roles) != bool(selection["history_fpv"]):
            raise ValueError("Stage I history image roles do not match selection")
        if bool(bev_roles) != bool(selection["bev"]):
            raise ValueError("Stage I BEV image roles do not match selection")
        if (self.instruction is not None) != bool(selection["instruction"]):
            raise ValueError("Stage I instruction visibility mismatch")
        if selection["instruction"] and self.instruction_training_role != "primary":
            raise ValueError("selected instruction must have training_role=primary")
        if not availability["instruction"] and self.instruction_training_role != "not_present":
            raise ValueError("missing instruction requires training_role=not_present")
        if availability["instruction"] and self.instruction_training_role == "not_present":
            raise ValueError("native instruction cannot use training_role=not_present")

    def _validate_loss_spans(self) -> None:
        """Require one ordered non-overlapping partition of assistant text."""

        cursor = 0
        for span in self.loss_spans:
            if span.start != cursor:
                raise ValueError("Stage I loss spans must be contiguous and ordered")
            if span.end > len(self.assistant_text):
                raise ValueError("Stage I loss span exceeds assistant text")
            cursor = span.end
        if cursor != len(self.assistant_text):
            raise ValueError("Stage I loss spans must cover assistant text")

    def _validate_parent_identity(self) -> None:
        """Require the correct immutable parent identity for each task family."""

        if self.parent_artifact_kind == "reasoning_v4_6":
            if self.parent_audit_id is None or self.parent_annotation_id is not None:
                raise ValueError("Reasoning Stage I view requires only parent_audit_id")
        elif self.parent_annotation_id is None or self.parent_audit_id is not None:
            raise ValueError("VQA Stage I view requires only parent_annotation_id")


class StageIFullCoTViewV1(StageITrainingViewBaseV1):
    """Carry one safe complete Reasoning V4.6 FullCoT view.

    Args:
        task_type: Always ``full_cot``.
        reasoning_target: Exact canonical six-section target.
        direction_context: Public direction profiles.
        direction_supervision: Private strong stamp retained only as audit sidecar.
        prompt_profile_id: Canonical Reasoning V4.6 student prompt.
        prompt_policy_digest: Exact prompt-policy digest.
    """

    task_type: Literal["full_cot"] = "full_cot"
    reasoning_target: ReasoningTargetV4
    direction_context: DirectionContextV4
    direction_supervision: DirectionSupervisionV4
    prompt_profile_id: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_full_cot(self) -> "StageIFullCoTViewV1":
        """Require FullCoT-only answerability and action identity."""

        if not self.answerability.full_cot or self.answerability.vqa or self.answerability.decision:
            raise ValueError("FullCoT view requires only full_cot answerability")
        if str(self.reasoning_target.reasoning.decision.selected_action_label) != str(
                self.direction_supervision.label_name):
            raise ValueError("FullCoT Decision must match strong direction")
        return self


class StageIDecisionViewV1(StageITrainingViewBaseV1):
    """Carry one Decision-only LM target without a FullCoT teacher prefix.

    Args:
        task_type: Always ``decision``.
        decision: Exact canonical V4.6 Decision tuple.
        direction_context: Public direction profiles.
        direction_supervision: Complete strong direction stamp.
        decision_authority: Visible authority admitted by the projector.
    """

    task_type: Literal["decision"] = "decision"
    decision: DecisionV4
    direction_context: DirectionContextV4
    direction_supervision: DirectionSupervisionV4
    decision_authority: Literal[
        "explicit_direction_instruction",
        "visible_bev_route",
        "fpv_behavior_cloning",
    ]

    @model_validator(mode="after")
    def validate_decision(self) -> "StageIDecisionViewV1":
        """Require Decision-only answerability and strong-label equality."""

        if not self.answerability.decision or self.answerability.full_cot or self.answerability.vqa:
            raise ValueError("Decision view requires only decision answerability")
        if str(self.decision.selected_action_label) != str(self.direction_supervision.label_name):
            raise ValueError("Decision-only action must match strong direction")
        return self


class StageIVQAViewV1(StageITrainingViewBaseV1):
    """Carry one evidence-preserving answer-only VQA view.

    Args:
        task_type: Always ``vqa``.
        question: Exact answer-independent VQA V2 question.
        annotation_id: Parent VQA annotation identity.
        annotation_evidence_refs: Exact evidence used by the annotation.
        label_source: Derived, weak, or external label source.
        source_kind: Open source-kind code.
        producer_id: Label producer identity.
        producer_version: Label producer version.
        prompt_profile_id: Exact VQA V2 student prompt.
        prompt_policy_digest: Exact VQA prompt-policy digest.
        capability_config_digest: Exact VQA activation snapshot.
    """

    task_type: Literal["vqa"] = "vqa"
    question: VQAQuestionV2
    annotation_id: str = Field(min_length=1, max_length=400)
    annotation_evidence_refs: list[str] = Field(min_length=1, max_length=64)
    label_source: StageIVQALabelSourceV1
    source_kind: ReasonCode
    producer_id: str = Field(min_length=1, max_length=128)
    producer_version: str = Field(min_length=1, max_length=128)
    prompt_profile_id: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    capability_config_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_vqa(self) -> "StageIVQAViewV1":
        """Require VQA-only answerability and retained evidence."""

        if not self.answerability.vqa or self.answerability.full_cot or self.answerability.decision:
            raise ValueError("VQA view requires only vqa answerability")
        visible_refs = {str(item.ref) for item in self.evidence_catalog}
        if not set(self.annotation_evidence_refs).issubset(visible_refs):
            raise ValueError("VQA annotation evidence is absent from Stage I input")
        if self.annotation_id != self.parent_annotation_id:
            raise ValueError("VQA annotation identity mismatch")
        return self


StageITrainingViewV1 = Annotated[
    Union[StageIFullCoTViewV1, StageIVQAViewV1, StageIDecisionViewV1],
    Field(discriminator="task_type"),
]
STAGE_I_TRAINING_VIEW_ADAPTER_V1 = TypeAdapter(StageITrainingViewV1)


def stage_i_projection_digest_v1(payload: dict[str, Any]) -> str:
    """Return canonical SHA-256 for a Stage I projection payload.

    Args:
        payload: JSON-compatible view without ``projection_digest``.

    Returns:
        Lowercase SHA-256 digest.
    """

    # 投影器可省略 Optional=None，而 Pydantic 会补回它们；摘要必须对这两种等价表示稳定。
    normalized = _drop_none_values(payload)
    text = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _drop_none_values(value: Any) -> Any:
    """Remove JSON null fields recursively while preserving list order."""

    if isinstance(value, dict):
        return {
            key: _drop_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


def parse_stage_i_training_view_v1(payload: dict[str, Any]) -> StageITrainingViewV1:
    """Parse one discriminated Stage I training view.

    Args:
        payload: JSON-compatible view payload.

    Returns:
        Validated task-specific view.

    Raises:
        ValidationError: If the view violates the canonical contract.
    """

    return STAGE_I_TRAINING_VIEW_ADAPTER_V1.validate_python(payload)


class StageIAnswerabilityV2(BaseModel):
    """Record the deterministic admission facts for one V2 physical view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    full_cot: bool = False
    reasoning_dropout: bool = False
    vqa: bool = False
    reasons: tuple[ReasonCode, ...] = Field(default_factory=tuple, max_length=16)


class StageILossSpanV2(BaseModel):
    """Represent one V2 canonical assistant span with mandatory LM loss.

    V2 deliberately has no action-value exception: every canonical assistant
    character belongs to a scale-one span.  Template-only transport suffixes
    remain outside this contract in ms-swift.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    scale: Literal[1.0] = 1.0
    channel: StageILossChannelV2

    @model_validator(mode="after")
    def validate_interval(self) -> "StageILossSpanV2":
        if self.end <= self.start:
            raise ValueError("loss span end must be greater than start")
        return self


class StageIReasoningDropoutLinkV2(BaseModel):
    """Bind a direct Decision row to its immutable FullCoT parent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_full_cot_sample_id: str = Field(min_length=1, max_length=480)
    parent_full_cot_projection_digest: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    parent_full_cot_assistant_sha256: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    assistant_decision_sha256: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    dropout_reason: Literal["assistant_reasoning_prefix"] = "assistant_reasoning_prefix"


class StageITrainingViewBaseV2(BaseModel):
    """Common immutable V2 envelope shared by FullCoT, dropout and VQA."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION] = STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION
    task_type: StageITaskTypeV2
    sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: StageIModalityProfileIdV1
    native_availability: StageIModalityAvailabilityV1
    training_selection: StageIModalitySelectionV1
    modality_state: StageIModalityStateV1
    instruction_training_role: StageIInstructionTrainingRoleV1
    answerability: StageIAnswerabilityV2
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, min_length=1, max_length=4096)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(min_length=1, max_length=64)
    input_modalities: InputModalitiesV4
    assistant_text: str = Field(min_length=1, max_length=20000)
    loss_span_policy_id: Literal[STAGE_I_LOSS_SPAN_POLICY_V2_ID] = STAGE_I_LOSS_SPAN_POLICY_V2_ID
    loss_spans: list[StageILossSpanV2] = Field(min_length=1, max_length=2)
    parent_artifact_kind: StageIParentArtifactKindV1
    parent_sample_id: str = Field(min_length=1, max_length=480)
    parent_source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    parent_annotation_id: str | None = Field(default=None, min_length=1, max_length=400)
    parent_audit_id: str | None = Field(default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    projection_policy_id: Literal[STAGE_I_PROJECTION_POLICY_V2_ID] = STAGE_I_PROJECTION_POLICY_V2_ID
    projection_policy_version: Literal[STAGE_I_PROJECTION_POLICY_V2_VERSION] = STAGE_I_PROJECTION_POLICY_V2_VERSION
    projection_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_common_view(self) -> "StageITrainingViewBaseV2":
        if len(self.images) != len(self.image_roles) or len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("Stage I V2 images and image_roles must be aligned and unique")
        if self.image_roles.count("fpv.current") != 1:
            raise ValueError("Stage I V2 requires exactly one fpv.current")
        cursor = 0
        for span in self.loss_spans:
            if span.start != cursor or span.end > len(self.assistant_text):
                raise ValueError("Stage I V2 loss spans must be contiguous and ordered")
            cursor = span.end
        if cursor != len(self.assistant_text):
            raise ValueError("Stage I V2 loss spans must cover assistant text")
        expected = stage_i_projection_digest_v2(self.model_dump(mode="json", exclude={"projection_digest"}))
        if self.projection_digest != expected:
            raise ValueError("Stage I V2 projection_digest does not match payload")
        return self


class StageIFullCoTViewV2(StageITrainingViewBaseV2):
    """Carry one fully supervised six-section canonical FullCoT target."""

    task_type: Literal["full_cot"] = "full_cot"
    reasoning_target: ReasoningTargetV4
    direction_context: DirectionContextV4
    direction_supervision: DirectionSupervisionV4
    prompt_profile_id: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_full_cot(self) -> "StageIFullCoTViewV2":
        if not self.answerability.full_cot or self.answerability.reasoning_dropout or self.answerability.vqa:
            raise ValueError("V2 FullCoT requires only full_cot answerability")
        if str(self.reasoning_target.reasoning.decision.selected_action_label) != str(
                self.direction_supervision.label_name):
            raise ValueError("V2 FullCoT Decision must match strong direction")
        return self


class StageIReasoningDropoutViewV2(StageITrainingViewBaseV2):
    """Carry the direct Decision projection of one V2 FullCoT parent."""

    task_type: Literal["reasoning_dropout"] = "reasoning_dropout"
    decision: DecisionV4
    direction_context: DirectionContextV4
    direction_supervision: DirectionSupervisionV4
    dropout_link: StageIReasoningDropoutLinkV2

    @model_validator(mode="after")
    def validate_dropout(self) -> "StageIReasoningDropoutViewV2":
        if not self.answerability.reasoning_dropout or self.answerability.full_cot or self.answerability.vqa:
            raise ValueError("reasoning dropout requires only reasoning_dropout answerability")
        if str(self.decision.selected_action_label) != str(self.direction_supervision.label_name):
            raise ValueError("reasoning dropout Decision must match strong direction")
        rendered = json.dumps(self.decision.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        expected = f"<Decision>{rendered}</Decision>"
        if self.assistant_text != expected:
            raise ValueError("reasoning dropout assistant target is not canonical Decision projection")
        if self.dropout_link.assistant_decision_sha256 != hashlib.sha256(expected.encode("utf-8")).hexdigest():
            raise ValueError("reasoning dropout Decision digest does not match assistant target")
        return self


class StageIVQAViewV2(StageITrainingViewBaseV2):
    """Carry an evidence-preserving VQA V2 answer under the Stage I V2 envelope."""

    task_type: Literal["vqa"] = "vqa"
    question: VQAQuestionV2
    annotation_id: str = Field(min_length=1, max_length=400)
    annotation_evidence_refs: list[str] = Field(min_length=1, max_length=64)
    label_source: StageIVQALabelSourceV1
    source_kind: ReasonCode
    producer_id: str = Field(min_length=1, max_length=128)
    producer_version: str = Field(min_length=1, max_length=128)
    prompt_profile_id: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    capability_config_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_vqa(self) -> "StageIVQAViewV2":
        if not self.answerability.vqa or self.answerability.full_cot or self.answerability.reasoning_dropout:
            raise ValueError("V2 VQA requires only vqa answerability")
        if self.annotation_id != self.parent_annotation_id:
            raise ValueError("V2 VQA annotation identity mismatch")
        visible_refs = {str(item.ref) for item in self.evidence_catalog}
        if not set(self.annotation_evidence_refs).issubset(visible_refs):
            raise ValueError("V2 VQA annotation evidence is absent from Stage I input")
        return self


StageITrainingViewV2 = Annotated[
    Union[StageIFullCoTViewV2, StageIReasoningDropoutViewV2, StageIVQAViewV2],
    Field(discriminator="task_type"),
]
STAGE_I_TRAINING_VIEW_ADAPTER_V2 = TypeAdapter(StageITrainingViewV2)


def stage_i_projection_digest_v2(payload: dict[str, Any]) -> str:
    """Return the canonical SHA-256 digest for a V2 projection payload."""

    return stage_i_projection_digest_v1(payload)


def parse_stage_i_training_view_v2(payload: dict[str, Any]) -> StageITrainingViewV2:
    """Parse one discriminated V2 Stage I training view."""

    return STAGE_I_TRAINING_VIEW_ADAPTER_V2.validate_python(payload)


__all__ = [
    "STAGE_I_LOSS_SPAN_POLICY_ID",
    "STAGE_I_INSTRUCTION_PROVENANCE_SCHEMA_VERSION",
    "STAGE_I_PROJECTION_POLICY_ID",
    "STAGE_I_PROJECTION_POLICY_VERSION",
    "STAGE_I_PROJECTION_DROP_SCHEMA_VERSION",
    "STAGE_I_TRAINING_VIEW_ADAPTER_V1",
    "STAGE_I_TRAINING_VIEW_SCHEMA_VERSION",
    "STAGE_I_LOSS_SPAN_POLICY_V2_ID",
    "STAGE_I_PROJECTION_POLICY_V2_ID",
    "STAGE_I_PROJECTION_POLICY_V2_VERSION",
    "STAGE_I_TRAINING_VIEW_ADAPTER_V2",
    "STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION",
    "StageIAnswerabilityV1",
    "StageIAnswerabilityV2",
    "StageIDecisionViewV1",
    "StageIFullCoTViewV1",
    "StageIFullCoTViewV2",
    "StageIInstructionProvenanceV1",
    "StageIInstructionTrainingRoleV1",
    "StageILossSpanV1",
    "StageILossSpanV2",
    "StageIModalityAvailabilityV1",
    "StageIModalityProfileIdV1",
    "StageIModalitySelectionV1",
    "StageIModalityStateV1",
    "StageIProjectionDropV1",
    "StageITaskTypeV1",
    "StageITaskTypeV2",
    "StageITrainingViewV1",
    "StageITrainingViewV2",
    "StageIVQAViewV1",
    "StageIVQAViewV2",
    "StageIReasoningDropoutLinkV2",
    "StageIReasoningDropoutViewV2",
    "parse_stage_i_training_view_v1",
    "parse_stage_i_training_view_v2",
    "stage_i_projection_digest_v1",
    "stage_i_projection_digest_v2",
]
