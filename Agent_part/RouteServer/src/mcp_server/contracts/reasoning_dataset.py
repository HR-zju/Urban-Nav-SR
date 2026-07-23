"""Dataset contracts for grounded ReasoningTraceV2 annotation.

RouteServer owns the auditable annotation artifacts.  A source row freezes
everything visible to the teacher/model, an annotation row records one teacher
attempt and its validation outcome, and a training view projects an accepted
annotation into Full-CoT/System1 supervision.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import EvidenceRef, ReasoningTargetV2


REASONING_SOURCE_V2_SCHEMA_VERSION = "urbannav_reasoning_source_v2"
REASONING_ANNOTATION_V2_SCHEMA_VERSION = "urbannav_reasoning_annotation_v2"
REASONING_TRAINING_VIEW_V2_SCHEMA_VERSION = "urbannav_reasoning_training_view_v2"


class EvidenceCatalogEntryV2(BaseModel):
    """Describe one addressable item visible in the model input.

    Args:
        ref: Stable reference that a reasoning target may cite.
        modality: Evidence family such as instruction, FPV, BEV, or route.
        summary: Optional input-visible description.  It must not contain
            private labels or future trajectory supervision.
        attributes: Optional structured properties already present in input.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: EvidenceRef
    modality: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    summary: str | None = Field(default=None, max_length=1024)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ReasoningSourceV2(BaseModel):
    """Freeze one multimodal input before any teacher call.

    Args:
        schema_version: Fixed source contract version.
        source_group_id: Stable id shared by Full-CoT and System1 views.
        dataset_source: Source dataset family.
        source_record_id: Original dataset record id.
        split_group_id: Leakage-safe clip/session grouping id.
        images: Ordered image paths or URLs passed to the VLM.
        image_roles: Role for every ordered image.
        available_modalities: Modalities actually visible to the model.
        instruction: Active model-visible navigation instruction.
        travel_mode: Active travel mode.
        model_visible_context: Structured goal, route, waypoint, or tool facts
            rendered into the user prompt.
        evidence_catalog: Complete set of references legal in teacher output.
        goal_geometry: Optional numeric goal encoder input.
        route_points_ego_t: Optional numeric route/waypoint condition input.
        trajectory_ego_t: Optional canonical System1 target.  It never enters
            assistant text.
        trajectory_policy: Canonical System1 coordinate contract.
        action_mask: Optional per-step System1 supervision mask.
        system1_enabled: Whether the full view may train a continuous head.
        reasoning_qa_label_validity: Frozen legacy QA audit metadata. It does
            not trigger training-view generation.
        teacher_private_context: Audit-only fields excluded from teacher and
            model prompts by default.
        metadata: Non-target provenance and conversion diagnostics.
        source_fingerprint: Hash binding the annotation to this frozen source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_SOURCE_V2_SCHEMA_VERSION] = REASONING_SOURCE_V2_SCHEMA_VERSION
    source_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=96)
    source_record_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    available_modalities: list[str] = Field(min_length=1, max_length=24)
    instruction: str = Field(min_length=1, max_length=4096)
    travel_mode: str = Field(default="pedestrian_assist", min_length=1, max_length=64)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV2] = Field(min_length=1, max_length=96)
    goal_geometry: dict[str, Any] | list[Any] | None = None
    route_points_ego_t: list[Any] | None = None
    trajectory_ego_t: list[list[float]] | None = None
    trajectory_policy: dict[str, Any] | None = None
    action_mask: list[bool] | list[int] | None = None
    system1_enabled: bool = False
    reasoning_qa_label_validity: dict[str, dict[str, Any]] = Field(default_factory=dict)
    teacher_private_context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source_contract(self) -> "ReasoningSourceV2":
        """Validate image, evidence, and System1 field partitions.

        Returns:
            Validated source.

        Raises:
            ValueError: If image roles, evidence refs, or System1 sidecars are
                inconsistent.
        """

        if len(self.images) != len(self.image_roles):
            raise ValueError("images and image_roles must have equal length")
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("image_roles must be unique and temporally addressable")
        refs = [item.ref for item in self.evidence_catalog]
        if len(set(refs)) != len(refs):
            raise ValueError("evidence_catalog refs must be unique")
        if self.system1_enabled:
            if self.trajectory_ego_t is None or self.trajectory_policy is None:
                raise ValueError("system1_enabled requires trajectory_ego_t and trajectory_policy")
        elif self.trajectory_ego_t is not None or self.trajectory_policy is not None or self.action_mask is not None:
            raise ValueError("disabled System1 source cannot carry continuous supervision")
        if self.trajectory_ego_t is not None:
            _validate_xyt(self.trajectory_ego_t, name="trajectory_ego_t")
            if self.action_mask is not None and len(self.action_mask) != len(self.trajectory_ego_t):
                raise ValueError("action_mask length must match trajectory_ego_t")
        if self.route_points_ego_t is not None:
            _validate_route_points(self.route_points_ego_t)
        expected = reasoning_source_fingerprint(self.model_dump(mode="json", exclude={"source_fingerprint"}))
        if self.source_fingerprint != expected:
            raise ValueError("source_fingerprint does not match frozen source payload")
        return self


class ReasoningAnnotationV2(BaseModel):
    """Record one validated or rejected teacher annotation.

    Args:
        schema_version: Fixed annotation contract version.
        source_group_id: Source id being annotated.
        source_fingerprint: Hash copied from the frozen source.
        status: Accepted, rejected, or provider error.
        target: Strong V2 target for accepted rows.
        raw_answer: Last raw provider response for audit and repair.
        errors: Hard parsing, grounding, or provider errors.
        warnings: Non-blocking validation warnings.
        attempts: Number of provider attempts.
        teacher_provider: Provider name without credentials.
        teacher_model: Model identifier.
        telemetry: Sanitized timing and token metadata.
        metadata: Runner and repair provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_ANNOTATION_V2_SCHEMA_VERSION] = REASONING_ANNOTATION_V2_SCHEMA_VERSION
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    status: Literal["accepted", "rejected", "error"]
    target: ReasoningTargetV2 | None = None
    raw_answer: str | None = None
    errors: list[str] = Field(default_factory=list, max_length=64)
    warnings: list[str] = Field(default_factory=list, max_length=64)
    attempts: int = Field(ge=0, le=32)
    teacher_provider: str | None = Field(default=None, max_length=64)
    teacher_model: str | None = Field(default=None, max_length=256)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_annotation_status(self) -> "ReasoningAnnotationV2":
        """Require a target only for accepted annotations.

        Returns:
            Validated annotation.

        Raises:
            ValueError: If status and target disagree.
        """

        if self.status == "accepted" and self.target is None:
            raise ValueError("accepted annotation requires target")
        if self.status != "accepted" and self.target is not None:
            raise ValueError("rejected/error annotation cannot carry target")
        return self


class ReasoningTrainingViewV2(BaseModel):
    """Canonical cross-repository view consumed by ms-swift.

    Args:
        schema_version: Fixed training-view contract version.
        sample_id: Unique row id.
        source_group_id: Parent source id for grouped splitting.
        view_type: Full CoT/System1 view.
        task_family: Main grounded-navigation reasoning family.
        task_type: Stable training task name.
        images: Ordered multimodal inputs.
        image_roles: Role for each image.
        available_modalities: Prompt-visible modality set.
        instruction: Active instruction.
        travel_mode: Active travel mode.
        model_visible_context: Context rendered into the user prompt.
        evidence_catalog: Input-visible reference catalog.
        assistant_text: Deterministic target text.
        reasoning_target: Structured full target for validation.
        goal_geometry: Optional goal encoder input.
        route_points_ego_t: Optional route condition input.
        trajectory_ego_t: Navigate-only System1 target.
        trajectory_policy: Navigate-only coordinate contract.
        action_mask: Optional System1 mask.
        loss_mask: Explicit logical losses for audits.
        metadata: Label provenance and conversion diagnostics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_TRAINING_VIEW_V2_SCHEMA_VERSION] = REASONING_TRAINING_VIEW_V2_SCHEMA_VERSION
    sample_id: str = Field(min_length=1, max_length=320)
    source_group_id: str = Field(min_length=1, max_length=240)
    view_type: Literal["full_cot"]
    task_family: Literal["grounded_navigation"]
    task_type: str = Field(min_length=1, max_length=128)
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    available_modalities: list[str] = Field(min_length=1, max_length=24)
    instruction: str = Field(min_length=1, max_length=4096)
    travel_mode: str = Field(min_length=1, max_length=64)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV2] = Field(min_length=1, max_length=96)
    assistant_text: str = Field(min_length=1)
    reasoning_target: ReasoningTargetV2 | None = None
    goal_geometry: dict[str, Any] | list[Any] | None = None
    route_points_ego_t: list[Any] | None = None
    trajectory_ego_t: list[list[float]] | None = None
    trajectory_policy: dict[str, Any] | None = None
    action_mask: list[bool] | list[int] | None = None
    loss_mask: dict[str, bool]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_view_partition(self) -> "ReasoningTrainingViewV2":
        """Enforce the Full CoT target and loss partition.

        Returns:
            Validated view.

        Raises:
            ValueError: If a full view is missing its structured target or its
                System1 labels disagree with the finish decision.
        """

        if len(self.images) != len(self.image_roles):
            raise ValueError("images and image_roles must have equal length")
        if self.reasoning_target is None:
            raise ValueError("full_cot view requires reasoning_target")
        is_navigate = self.reasoning_target.finish.decision_type == "navigate"
        has_system1 = self.trajectory_ego_t is not None
        if has_system1 and not is_navigate:
            raise ValueError("non-navigation full view cannot carry System1 supervision")
        if self.loss_mask.get("system1_action", False) != has_system1:
            raise ValueError("System1 loss mask must match trajectory target presence")
        return self


def reasoning_source_fingerprint(payload: dict[str, Any]) -> str:
    """Hash one source payload using canonical JSON serialization.

    Args:
        payload: JSON-compatible source payload without ``source_fingerprint``.

    Returns:
        Lowercase SHA-256 hex digest.
    """

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_xyt(value: list[list[float]], *, name: str) -> None:
    """Validate one finite non-empty ``[T, 3]`` pose sequence."""

    if not value:
        raise ValueError(f"{name} must be non-empty")
    for index, row in enumerate(value):
        if len(row) != 3:
            raise ValueError(f"{name}[{index}] must have width 3")
        if not all(math.isfinite(float(item)) for item in row):
            raise ValueError(f"{name}[{index}] must be finite")


def _validate_route_points(value: list[Any]) -> None:
    """Validate numeric sequence or goal-encoder dictionary route points."""

    if not value:
        raise ValueError("route_points_ego_t must be non-empty when present")
    for index, row in enumerate(value):
        if isinstance(row, dict):
            if row.get("x") is None or row.get("y") is None:
                raise ValueError(f"route_points_ego_t[{index}] requires x and y")
            numbers = [row["x"], row["y"]]
            if row.get("heading") is not None:
                numbers.append(row["heading"])
        elif isinstance(row, (list, tuple)) and len(row) in {2, 3}:
            numbers = list(row)
        else:
            raise ValueError(
                f"route_points_ego_t[{index}] must be a 2/3-vector or point object"
            )
        if not all(math.isfinite(float(item)) for item in numbers):
            raise ValueError(f"route_points_ego_t[{index}] must be finite")


__all__ = [
    "REASONING_ANNOTATION_V2_SCHEMA_VERSION",
    "REASONING_SOURCE_V2_SCHEMA_VERSION",
    "REASONING_TRAINING_VIEW_V2_SCHEMA_VERSION",
    "EvidenceCatalogEntryV2",
    "ReasoningAnnotationV2",
    "ReasoningSourceV2",
    "ReasoningTrainingViewV2",
    "reasoning_source_fingerprint",
]
