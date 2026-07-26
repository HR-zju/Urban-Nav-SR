"""Dataset and annotation contracts for UrbanNav Reasoning V4."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.direction_profile import (
    DirectionLabelStamp,
    approved_stage_a_direction_profiles,
    canonical_direction_derivation_version,
)
from mcp_server.contracts.reasoning import ReasonCode
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    ExternalEvidenceRefV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION,
    ReasoningTargetV4,
    STAGE_A_DECISION_TYPES_V4,
    SceneGroundingV4,
    TaskGroundingV4,
    coarse_motion_for_direction_label_v4,
)


REASONING_SOURCE_V4_SCHEMA_VERSION = "urbannav_reasoning_source_v4_0"
REASONING_ANNOTATION_V4_SCHEMA_VERSION = "urbannav_reasoning_annotation_v4_0"
V4_FORBIDDEN_VISIBLE_CONTEXT_KEY_PARTS = (
    "action_ego",
    "arrival",
    "bbox",
    "calibrat",
    "confidence",
    "extrinsic",
    "gnss",
    "gps",
    "ground_truth",
    "homography",
    "intrinsic",
    "latitude",
    "longitude",
    "mask",
    "odometry",
    "pixel_trajectory",
    "pose",
    "project",
    "route_theta",
    "runtime_task_phase",
    "track",
    "trajectory_ego",
    "trajectory_fpv",
    "ttc",
)

EvidenceModalityV4 = Literal[
    "instruction",
    "fpv",
    "bev",
    "goal",
    "goal_image",
    "pixel_goal",
    "waypoint",
]
GroundingSectionNameV4 = Literal[
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
]
ReasoningSectionNameV4 = Literal[
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
    "Decision",
]
TrainingViewTypeV4 = Literal[
    "full_cot_sft",
    "grounding_section_sft",
    "decision_section_sft",
]

class EvidenceCatalogEntryV4(BaseModel):
    """Describe one raw prompt-visible evidence item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: ExternalEvidenceRefV4
    modality: EvidenceModalityV4
    summary: str | None = Field(default=None, max_length=512)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_namespace(self) -> "EvidenceCatalogEntryV4":
        """Keep evidence namespaces compatible with declared modalities."""

        prefix = self.ref.split(".", 1)[0]
        expected = {
            "instruction": {"instruction"},
            "fpv": {"fpv"},
            "bev": {"bev"},
            "goal": {"goal", "goal_image", "pixel_goal"},
            "waypoint": {"waypoint"},
        }[prefix]
        if self.modality not in expected:
            raise ValueError(
                f"evidence namespace/modality mismatch: {self.ref}/{self.modality}"
            )
        return self


class InputModalitiesV4(BaseModel):
    """Record which Stage-A inputs are actually present for one sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fpv: Literal[True] = True
    history_fpv: bool = False
    instruction: bool = False
    bev: bool = False
    goal_image: bool = False
    pixel_goal: bool = False
    waypoint: bool = False


class NavObservationV4(BaseModel):
    """Freeze exactly the single-turn input visible to teacher and student."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    source_record_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, min_length=1, max_length=4096)
    travel_mode: str = Field(default="pedestrian_assist", min_length=1, max_length=64)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(min_length=1, max_length=64)
    input_modalities: InputModalitiesV4
    temporal_evidence_eligible: bool = False

    @model_validator(mode="after")
    def validate_observation(self) -> "NavObservationV4":
        """Validate image ordering, evidence identity, and modality flags."""

        if len(self.images) != len(self.image_roles):
            raise ValueError("images and image_roles must have equal length")
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("image_roles must be unique and addressable")
        if self.image_roles.count("fpv.current") != 1:
            raise ValueError("NavObservationV4 requires exactly one fpv.current")
        refs = [entry.ref for entry in self.evidence_catalog]
        if len(refs) != len(set(refs)):
            raise ValueError("evidence_catalog refs must be unique")
        ref_set = set(refs)
        for role in self.image_roles:
            if role.startswith(("fpv.", "bev.", "goal.")) and role not in ref_set:
                raise ValueError(f"image role is missing from evidence catalog: {role}")
        flags = self.input_modalities
        expected_flags = {
            "history_fpv": any(role.startswith("fpv.history_") for role in self.image_roles),
            "instruction": self.instruction is not None,
            "bev": any(role.startswith("bev.") for role in self.image_roles),
            "goal_image": any(role.startswith("goal.image_") for role in self.image_roles),
            "pixel_goal": any(entry.modality == "pixel_goal" for entry in self.evidence_catalog),
            "waypoint": any(entry.modality == "waypoint" for entry in self.evidence_catalog),
        }
        actual_flags = flags.model_dump(mode="python")
        for name, expected in expected_flags.items():
            if bool(actual_flags[name]) != bool(expected):
                raise ValueError(
                    f"input modality flag mismatch: {name}="
                    f"{actual_flags[name]}, expected={expected}"
                )
        if self.temporal_evidence_eligible and not flags.history_fpv:
            raise ValueError("temporal evidence requires at least one history FPV")
        forbidden_paths = _forbidden_prompt_visible_paths_v4(
            self.model_visible_context,
            self.evidence_catalog,
        )
        if forbidden_paths:
            raise ValueError(
                "private prompt-visible context keys are forbidden: "
                f"{forbidden_paths}"
            )
        return self


class DirectionSupervisionV4(DirectionLabelStamp):
    """Carry one complete, profile-stamped deterministic direction label."""

    @model_validator(mode="after")
    def validate_profile_stamp(self) -> "DirectionSupervisionV4":
        """Require the profile's canonical trajectory derivation."""

        expected_derivation = canonical_direction_derivation_version(
            self.profile_id
        )
        if self.derivation_version != expected_derivation:
            raise ValueError("direction derivation_version mismatch")
        coarse_motion_for_direction_label_v4(self.label_name)
        return self


class ReasoningSourceV4(BaseModel):
    """Partition prompt-visible observation from private strong supervision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_SOURCE_V4_SCHEMA_VERSION] = REASONING_SOURCE_V4_SCHEMA_VERSION
    observation: NavObservationV4
    direction_supervision: DirectionSupervisionV4 | None = None
    trajectory_ego_t: list[list[float]] | None = None
    trajectory_policy: dict[str, Any] | None = None
    action_mask: list[bool] | list[int] | None = None
    warnings: list[str] = Field(default_factory=list, max_length=64)
    source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source(self) -> "ReasoningSourceV4":
        """Validate System1 sidecars and the frozen source fingerprint."""

        has_trajectory = self.trajectory_ego_t is not None
        if has_trajectory != (self.trajectory_policy is not None):
            raise ValueError(
                "trajectory_ego_t and trajectory_policy must be present together"
            )
        if self.action_mask is not None and not has_trajectory:
            raise ValueError("action_mask requires trajectory_ego_t")
        if self.trajectory_ego_t is not None:
            _validate_xyt(self.trajectory_ego_t)
            if self.action_mask is not None and len(self.action_mask) != len(
                self.trajectory_ego_t
            ):
                raise ValueError("action_mask length must match trajectory_ego_t")
        expected = reasoning_source_fingerprint_v4(
            self.model_dump(mode="json", exclude={"source_fingerprint"})
        )
        if expected != self.source_fingerprint:
            raise ValueError("V4 source_fingerprint does not match frozen payload")
        return self


class GroundingOutputsV4(BaseModel):
    """Store independently produced grounding-section outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_grounding: TaskGroundingV4 | None = None
    scene_grounding: SceneGroundingV4 | None = None
    local_traversability: LocalTraversabilityV4 | None = None
    navigation_context_grounding: NavigationContextGroundingV4 | None = None
    interaction_constraint_grounding: InteractionConstraintGroundingV4 | None = None

    @property
    def complete(self) -> bool:
        """Return whether all five grounding outputs are available."""

        return all(
            value is not None
            for value in (
                self.task_grounding,
                self.scene_grounding,
                self.local_traversability,
                self.navigation_context_grounding,
                self.interaction_constraint_grounding,
            )
        )


class AnnotationStageRecordV4(BaseModel):
    """Record one section-specific teacher attempt stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["accepted", "rejected", "error", "skipped"]
    attempts: int = Field(ge=0, le=32)
    raw_answer: str | None = None
    errors: list[str] = Field(default_factory=list, max_length=64)
    warnings: list[str] = Field(default_factory=list, max_length=64)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    prompt_profile_id: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class ReasoningAnnotationV4(BaseModel):
    """Record partial or complete independently generated V4 supervision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_ANNOTATION_V4_SCHEMA_VERSION] = (
        REASONING_ANNOTATION_V4_SCHEMA_VERSION
    )
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    status: Literal["accepted", "partial", "rejected", "error"]
    grounding_outputs: GroundingOutputsV4
    decision: DecisionV4 | None = None
    target: ReasoningTargetV4 | None = None
    stage_records: dict[str, AnnotationStageRecordV4] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list, max_length=64)
    warnings: list[str] = Field(default_factory=list, max_length=128)
    teacher_provider: str | None = Field(default=None, max_length=64)
    teacher_model: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> "ReasoningAnnotationV4":
        """Keep overall status consistent with available section outputs."""

        if self.status == "accepted":
            if not self.grounding_outputs.complete or self.decision is None or self.target is None:
                raise ValueError(
                    "accepted V4 annotation requires five groundings, Decision, and target"
                )
        elif self.target is not None:
            raise ValueError("non-accepted V4 annotation cannot carry assembled target")
        return self


class ReasoningTrainingViewV4(BaseModel):
    """Cross-repository V4 view consumed by ms-swift."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION] = (
        REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=320)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    view_type: TrainingViewTypeV4
    section_name: ReasoningSectionNameV4 | None = None
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, min_length=1, max_length=4096)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(min_length=1, max_length=64)
    input_modalities: InputModalitiesV4
    assistant_text: str = Field(min_length=1)
    reasoning_target: ReasoningTargetV4 | None = None
    prompt_profile_id: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    direction_supervision: DirectionSupervisionV4 | None = None
    trajectory_ego_t: list[list[float]] | None = None
    trajectory_policy: dict[str, Any] | None = None
    action_mask: list[bool] | list[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_view(self) -> "ReasoningTrainingViewV4":
        """Validate view-type target and System1 sidecar partitions."""

        if len(self.images) != len(self.image_roles):
            raise ValueError("images and image_roles must have equal length")
        if self.view_type == "full_cot_sft":
            if self.section_name is not None or self.reasoning_target is None:
                raise ValueError(
                    "full_cot_sft requires reasoning_target and no section_name"
                )
            if (
                self.reasoning_target.finish.decision_type
                not in STAGE_A_DECISION_TYPES_V4
            ):
                raise ValueError(
                    "full_cot_sft requires a Stage-A navigate/stop decision"
                )
        elif self.view_type == "grounding_section_sft":
            if self.section_name not in {
                "TaskGrounding",
                "SceneGrounding",
                "LocalTraversability",
                "NavigationContextGrounding",
                "InteractionConstraintGrounding",
            }:
                raise ValueError(
                    "grounding_section_sft requires one grounding section_name"
                )
            if self.reasoning_target is not None:
                raise ValueError(
                    "grounding_section_sft cannot carry full reasoning_target"
                )
        elif self.view_type == "decision_section_sft":
            if self.section_name != "Decision":
                raise ValueError("decision_section_sft requires section_name=Decision")
            if self.reasoning_target is not None:
                raise ValueError(
                    "decision_section_sft cannot carry full reasoning_target"
                )
        has_trajectory = self.trajectory_ego_t is not None
        if has_trajectory != (self.trajectory_policy is not None):
            raise ValueError(
                "training view trajectory and policy must be present together"
            )
        if has_trajectory and self.view_type != "full_cot_sft":
            raise ValueError("System1 trajectory is legal only on full_cot_sft")
        forbidden_paths = _forbidden_prompt_visible_paths_v4(
            self.model_visible_context,
            self.evidence_catalog,
        )
        if forbidden_paths:
            raise ValueError(
                "private prompt-visible context keys are forbidden: "
                f"{forbidden_paths}"
            )
        return self


def reasoning_source_fingerprint_v4(payload: dict[str, Any]) -> str:
    """Hash one JSON-compatible V4 source payload canonically."""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def approved_direction_profiles_v4() -> dict[str, dict[str, Any]]:
    """Return a JSON-compatible snapshot of the V4 direction allowlist."""

    return approved_stage_a_direction_profiles()


def _validate_xyt(value: list[list[float]]) -> None:
    """Validate one finite, non-empty cumulative ``[T,3]`` trajectory."""

    if not value:
        raise ValueError("trajectory_ego_t must be non-empty")
    for index, row in enumerate(value):
        if len(row) != 3:
            raise ValueError(f"trajectory_ego_t[{index}] must have width 3")
        if not all(math.isfinite(float(item)) for item in row):
            raise ValueError(f"trajectory_ego_t[{index}] must be finite")


def _forbidden_visible_paths_v4(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return nested prompt-visible paths with private supervision names."""

    findings: list[str] = []
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            normalized = key.lower()
            if any(
                part in normalized
                for part in V4_FORBIDDEN_VISIBLE_CONTEXT_KEY_PARTS
            ):
                findings.append(path)
            findings.extend(
                _forbidden_visible_paths_v4(item, prefix=path)
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            findings.extend(
                _forbidden_visible_paths_v4(item, prefix=path)
            )
    return findings


def _forbidden_prompt_visible_paths_v4(
    model_visible_context: dict[str, Any],
    evidence_catalog: list[EvidenceCatalogEntryV4],
) -> list[str]:
    """Collect private paths from both visible-context containers."""

    findings = _forbidden_visible_paths_v4(model_visible_context)
    for entry in evidence_catalog:
        findings.extend(
            f"evidence_catalog.{entry.ref}.attributes.{path}"
            for path in _forbidden_visible_paths_v4(entry.attributes)
        )
    return sorted(set(findings))
