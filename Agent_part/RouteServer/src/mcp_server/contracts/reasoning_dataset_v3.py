"""Versioned dataset wrappers for the minimal Reasoning V3.4 contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import ReasonCode
from mcp_server.contracts.reasoning_dataset import ReasoningSourceV2, reasoning_source_fingerprint
from mcp_server.contracts.reasoning_v3 import (
    LabelProvenanceEntryV3,
    ReasoningTargetV3,
    SectionLossMaskV3,
)
from mcp_server.contracts.reasoning_v3_observation import (
    DecisionRationaleObservationV3,
    ReasoningObservationV3,
)


REASONING_SOURCE_V3_SCHEMA_VERSION = "urbannav_reasoning_source_v3_4"
REASONING_ANNOTATION_V3_SCHEMA_VERSION = "urbannav_reasoning_annotation_v3_4"
REASONING_TRAINING_VIEW_V3_SCHEMA_VERSION = "urbannav_reasoning_training_view_v3_4"


class ReasoningSourceV3(BaseModel):
    """Wrap one frozen visible input with isolated teacher-only metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_SOURCE_V3_SCHEMA_VERSION] = REASONING_SOURCE_V3_SCHEMA_VERSION
    model_input: ReasoningSourceV2
    teacher_supervision_context: dict[str, Any] = Field(default_factory=dict)
    label_provenance: dict[str, LabelProvenanceEntryV3] = Field(default_factory=dict)
    section_loss_mask: SectionLossMaskV3
    warnings: list[str] = Field(default_factory=list, max_length=64)
    source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source_partition(self) -> "ReasoningSourceV3":
        dataset_source = self.model_input.dataset_source.strip().lower()
        if dataset_source.startswith("frodobots") and self.model_input.system1_enabled:
            raise ValueError("FrodoBots V3 source cannot enable continuous System1 supervision")
        payload = self.model_dump(mode="json", exclude={"source_fingerprint"})
        expected = reasoning_source_fingerprint(payload)
        if self.source_fingerprint != expected:
            raise ValueError("V3 source_fingerprint does not match frozen source payload")
        return self


class ReasoningAnnotationV3(BaseModel):
    """Record one two-pass V3.4 annotation attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_ANNOTATION_V3_SCHEMA_VERSION] = (
        REASONING_ANNOTATION_V3_SCHEMA_VERSION
    )
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    status: Literal["accepted", "observation_only", "rejected", "error"]
    observation: ReasoningObservationV3 | None = None
    decision_rationale: DecisionRationaleObservationV3 | None = None
    target: ReasoningTargetV3 | None = None
    section_loss_mask: SectionLossMaskV3 | None = None
    raw_answer: str | None = None
    errors: list[str] = Field(default_factory=list, max_length=64)
    warnings: list[str] = Field(default_factory=list, max_length=64)
    label_provenance: dict[str, LabelProvenanceEntryV3] = Field(default_factory=dict)
    attempts: int = Field(ge=0, le=32)
    teacher_provider: str | None = Field(default=None, max_length=64)
    teacher_model: str | None = Field(default=None, max_length=256)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> "ReasoningAnnotationV3":
        if self.status == "accepted":
            if (
                self.observation is None
                or self.decision_rationale is None
                or self.target is None
                or self.section_loss_mask is None
            ):
                raise ValueError(
                    "accepted V3 annotation requires observation, decision rationale, "
                    "target, and section loss mask"
                )
        elif self.status == "observation_only":
            if self.observation is None:
                raise ValueError("observation_only annotation requires observation")
            if (
                self.decision_rationale is not None
                or self.target is not None
                or self.section_loss_mask is not None
            ):
                raise ValueError(
                    "observation_only annotation cannot carry rationale, target, or section loss mask"
                )
        elif self.target is not None or self.section_loss_mask is not None:
            raise ValueError("rejected/error V3 annotation cannot carry target or section loss mask")
        return self


class ReasoningTrainingViewV3(BaseModel):
    """Cross-repository V3.4 Full-CoT view consumed by ms-swift."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_TRAINING_VIEW_V3_SCHEMA_VERSION] = (
        REASONING_TRAINING_VIEW_V3_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=320)
    source_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    travel_mode: str = Field(min_length=1, max_length=64)
    view_type: Literal["full_cot"]
    task_family: Literal["grounded_navigation"]
    task_type: str = Field(min_length=1, max_length=128)
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    available_modalities: list[str] = Field(min_length=1, max_length=24)
    instruction: str = Field(min_length=1, max_length=4096)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[dict[str, Any]] = Field(min_length=1, max_length=96)
    assistant_text: str = Field(min_length=1)
    prompt_profile_id: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    reasoning_target: ReasoningTargetV3 | None = None
    direction_profile_id: str | None = Field(default=None, min_length=1, max_length=128)
    direction_profile_version: str | None = Field(default=None, min_length=1, max_length=64)
    direction_label_index: int | None = Field(default=None, ge=0)
    direction_label_name: ReasonCode | None = None
    direction_choice: str | None = Field(default=None, min_length=1, max_length=32)
    direction_angle_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    direction_source: ReasonCode | None = None
    trajectory_ego_t: list[list[float]] | None = None
    trajectory_policy: dict[str, Any] | None = None
    action_mask: list[bool] | list[int] | None = None
    loss_mask: dict[str, bool]
    section_loss_mask: SectionLossMaskV3
    label_provenance: dict[str, LabelProvenanceEntryV3] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_view_partition(self) -> "ReasoningTrainingViewV3":
        if len(self.images) != len(self.image_roles):
            raise ValueError("images and image_roles must have equal length")
        if self.reasoning_target is None:
            raise ValueError("V3 full CoT view requires reasoning_target")

        core_direction = (
            self.direction_profile_id,
            self.direction_profile_version,
            self.direction_label_index,
            self.direction_label_name,
            self.direction_source,
        )
        sidecar_present = any(
            value is not None
            for value in (
                *core_direction,
                self.direction_choice,
                self.direction_angle_deg,
            )
        )
        if sidecar_present and any(value is None for value in core_direction):
            raise ValueError(
                "direction sidecar requires profile id/version, label index/name, and source"
            )
        if sidecar_present:
            decision = self.reasoning_target.reasoning.decision
            sidecar_identity = (
                self.direction_profile_id,
                self.direction_profile_version,
                self.direction_label_index,
                self.direction_label_name,
                self.direction_choice,
                self.direction_angle_deg,
                self.direction_source,
            )
            decision_identity = (
                decision.profile_id,
                decision.profile_version,
                decision.label_index,
                decision.label_name,
                decision.choice,
                decision.angle_deg,
                decision.source,
            )
            if sidecar_identity != decision_identity:
                raise ValueError("direction sidecar must match the canonical Decision identity")
        return self


__all__ = [
    "REASONING_ANNOTATION_V3_SCHEMA_VERSION",
    "REASONING_SOURCE_V3_SCHEMA_VERSION",
    "REASONING_TRAINING_VIEW_V3_SCHEMA_VERSION",
    "ReasoningAnnotationV3",
    "ReasoningSourceV3",
    "ReasoningTrainingViewV3",
]
