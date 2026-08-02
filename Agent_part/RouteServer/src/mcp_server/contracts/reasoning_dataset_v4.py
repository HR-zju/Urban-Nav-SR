"""Dataset, annotation, review, and training contracts for Reasoning V4.6.

The prompt-visible observation remains a reusable input envelope shared with
VQA V2.  Reasoning V4.6 itself consumes only the six compact semantic sections,
one profile-stamped strong direction, an immutable action-reconciliation audit,
an optional two-stage LVLM diagnosis, and append-only human adjudication.
Review and training eligibility are derived projections rather than mutable
annotation fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.direction_profile import (
    DirectionLabelStamp,
    affordance_probe_label_names,
    affordance_probe_profile_for_action,
    approved_stage_a_direction_profiles,
    canonical_direction_derivation_version,
    canonical_direction_source_codes,
    coarse_motion_for_direction_label,
    direction_source_requires_stop,
    direction_label_names,
    resolve_affordance_probe_profile,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning import ReasonCode
from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4,
    DecisionV4,
    ExternalEvidenceRefV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION,
    REASONING_V4_COMPACT_FULL_COT_RENDERER_ID,
    REASONING_V4_CONTRACT_SCHEMA_DIGEST,
    ReasoningTargetV4,
    SceneGroundingV4,
    TaskGroundingV4,
)


REASONING_SOURCE_V4_SCHEMA_VERSION = "urbannav_reasoning_source_v4_5"
REASONING_ANNOTATION_V4_SCHEMA_VERSION = (
    "urbannav_reasoning_annotation_v4_6"
)
ACTION_RECONCILIATION_AUDIT_V4_SCHEMA_VERSION = (
    "urbannav_action_reconciliation_audit_v3"
)
ACTION_BLIND_REVIEW_V4_SCHEMA_VERSION = (
    "urbannav_action_blind_review_v1"
)
ACTION_REVIEW_DIAGNOSIS_V4_SCHEMA_VERSION = (
    "urbannav_action_review_diagnosis_v1"
)
ACTION_REVIEW_DIAGNOSIS_RECORD_V4_SCHEMA_VERSION = (
    "urbannav_action_review_diagnosis_record_v1"
)
ACTION_ADJUDICATION_V4_SCHEMA_VERSION = (
    "urbannav_action_adjudication_v1"
)
REASONING_REPAIR_REQUEST_V4_SCHEMA_VERSION = (
    "urbannav_reasoning_repair_request_v1"
)
ACTION_REVIEW_FILTER_VERSION_V4 = "reasoning_v46_lvlm_filter_v1"
REASONING_V4_SYSTEM1_QUERY_COUNT = 16

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
AnnotationStageStatusV4 = Literal[
    "accepted",
    "rejected",
    "error",
    "skipped",
]
RecordStatusV4 = Literal["complete", "error"]
CandidateStructureStatusV4 = Literal[
    "valid",
    "invalid",
    "not_available",
]
ProjectedReviewStatusV4 = Literal[
    "not_required",
    "pending_human",
    "resolved",
    "rejected",
]
ReleaseTrackV4 = Literal["review_preview", "formal_training"]
ArtifactStatusV4 = Literal["preview_only", "training_ready"]
TrainingViewTypeV4 = Literal["full_cot_sft"]
ActionReviewVerdictV4 = Literal[
    "teacher_proposal_error",
    "grounding_error",
    "strong_label_error",
    "unresolved",
]
ActionReviewComponentV4 = Literal[
    "DecisionProposal",
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
    "StrongDirectionSidecar",
]
ActionReviewAssessmentV4 = Literal[
    "likely_correct",
    "likely_wrong",
    "uncertain",
]
ActionReviewDiagnosisDispositionV4 = Literal[
    "confirmed",
    "overridden",
    "deferred",
]
ActionReviewRepairActionV4 = Literal[
    "keep_without_change",
    "rerun_teacher_proposal",
    "relabel_grounding",
    "rebuild_strong_sidecar",
    "quarantine",
]
ActionReviewErrorSubtypeV4 = Literal[
    "proposal_direction_error",
    "proposal_stop_error",
    "proposal_move_error",
    "proposal_speed_error",
    "proposal_response_error",
    "task_grounding_error",
    "scene_grounding_error",
    "local_traversability_error",
    "navigation_context_error",
    "interaction_constraint_error",
    "direction_quantization_error",
    "direction_projection_error",
    "coordinate_convention_error",
    "stop_label_error",
    "trajectory_sidecar_error",
]
BlindEvidenceSufficiencyV4 = Literal[
    "sufficient",
    "partial",
    "insufficient",
]


class BevRouteEvidenceV4(BaseModel):
    """Describe presentation properties of one prompt-visible BEV.

    These fields are retained at the reusable observation boundary for VQA V2
    compatibility.  Reasoning V4.6 never treats them as route-progress,
    traversability, clearance, or collision-safety supervision.

    Args:
        frame: Declared BEV presentation frame.
        ego_forward: Display direction corresponding to ego forward.
        route_overlay_visible: Whether a route overlay is rendered.
        route_direction_authority: Whether clear orientation may support a
            visible task direction.
        route_progress_authority: Legacy observation metadata ignored by
            Reasoning V4.6.
        route_terminal_marker_visible: Legacy presentation metadata ignored by
            Reasoning V4.6.
        route_endpoint_semantics: Legacy endpoint provenance ignored by
            Reasoning V4.6.
        local_safety_authority: Always false in Stage-A.
        producer_contract_id: Versioned upstream producer identity.
        producer_contract_version: Exact upstream producer version.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame: Literal["ego", "north_up", "unknown"] = "unknown"
    ego_forward: Literal[
        "up",
        "right",
        "down",
        "left",
        "unknown",
    ] = "unknown"
    route_overlay_visible: bool = False
    route_direction_authority: Literal[
        "none",
        "visual_conditional",
    ] = "none"
    route_progress_authority: Literal[
        "none",
        "visual_conditional",
    ] = "none"
    route_terminal_marker_visible: bool = False
    route_endpoint_semantics: Literal[
        "planned_navigation_goal",
        "recorded_clip_endpoint",
        "crop_boundary",
        "unknown",
    ] = "unknown"
    local_safety_authority: Literal[False] = False
    producer_contract_id: str = Field(
        default="unknown",
        min_length=1,
        max_length=128,
    )
    producer_contract_version: str = Field(
        default="unknown",
        min_length=1,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_presentation(self) -> "BevRouteEvidenceV4":
        """Fail closed when a hidden overlay is assigned authority."""

        if not self.route_overlay_visible and (
            self.route_direction_authority != "none"
            or self.route_progress_authority != "none"
            or self.route_terminal_marker_visible
        ):
            raise ValueError(
                "BEV authority and terminal marker require visible overlay"
            )
        return self


class EvidenceCatalogEntryV4(BaseModel):
    """Describe one prompt-visible raw input item.

    Args:
        ref: Stable input-envelope reference.
        modality: Input modality.
        summary: Optional prompt-visible summary.
        attributes: Prompt-visible producer metadata.
        bev_route_evidence: Optional BEV presentation declaration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: ExternalEvidenceRefV4
    modality: EvidenceModalityV4
    summary: str | None = Field(default=None, max_length=512)
    attributes: dict[str, Any] = Field(default_factory=dict)
    bev_route_evidence: BevRouteEvidenceV4 | None = None

    @model_validator(mode="after")
    def validate_namespace(self) -> "EvidenceCatalogEntryV4":
        """Keep evidence namespaces compatible with modalities."""

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
                "evidence namespace/modality mismatch: "
                f"{self.ref}/{self.modality}"
            )
        if self.bev_route_evidence is not None and self.modality != "bev":
            raise ValueError(
                "bev_route_evidence is legal only for BEV entries"
            )
        return self


class InputModalitiesV4(BaseModel):
    """Record which Stage-A inputs are present.

    Args:
        fpv: Current FPV is mandatory.
        history_fpv: One or more history FPV images are input only.
        instruction: An instruction is visible.
        bev: A BEV image is visible.
        goal_image: A goal image is visible.
        pixel_goal: A pixel goal is visible.
        waypoint: A waypoint is visible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fpv: Literal[True] = True
    history_fpv: bool = False
    instruction: bool = False
    bev: bool = False
    goal_image: bool = False
    pixel_goal: bool = False
    waypoint: bool = False


class DirectionContextV4(BaseModel):
    """Select approved action and affordance-probe profiles.

    Args:
        action_profile_id: Canonical fine action-profile identifier.
        action_profile_version: Exact action-profile version.
        affordance_probe_profile_id: Canonical probe identifier.
        affordance_probe_profile_version: Exact probe version.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_profile_id: str = Field(min_length=1, max_length=128)
    action_profile_version: str = Field(min_length=1, max_length=64)
    affordance_probe_profile_id: str = Field(
        min_length=1,
        max_length=128,
    )
    affordance_probe_profile_version: str = Field(
        min_length=1,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_profiles(self) -> "DirectionContextV4":
        """Require exact versions and the canonical profile pairing."""

        action = resolve_stage_a_direction_profile(
            self.action_profile_id
        )
        if self.action_profile_version != action["profile_version"]:
            raise ValueError("direction action profile_version mismatch")
        probe = resolve_affordance_probe_profile(
            self.affordance_probe_profile_id
        )
        if (
            self.affordance_probe_profile_version
            != probe["profile_version"]
        ):
            raise ValueError("affordance probe profile_version mismatch")
        expected_probe_id = affordance_probe_profile_for_action(
            self.action_profile_id
        )
        if self.affordance_probe_profile_id != expected_probe_id:
            raise ValueError(
                "affordance probe profile does not match action profile"
            )
        return self

    @property
    def action_labels(self) -> tuple[str, ...]:
        """Return profile-owned action labels.

        Returns:
            Ordered label vocabulary.
        """

        return direction_label_names(self.action_profile_id)

    @property
    def affordance_probe_labels(self) -> tuple[str, ...]:
        """Return profile-owned affordance probes.

        Returns:
            Ordered probe vocabulary.
        """

        return affordance_probe_label_names(
            self.affordance_probe_profile_id
        )


class NavObservationV4(BaseModel):
    """Freeze the single-turn input visible to teacher and student.

    ``temporal_evidence_eligible`` is retained only to avoid changing the
    shared VQA V2 source envelope.  Reasoning V4.6 ignores it and never derives
    temporal targets or evidence links from history frames.

    Args:
        source_group_id: Stable group identity.
        dataset_source: Dataset family.
        source_record_id: Source-row identity.
        split_group_id: Leakage-safe split group.
        images: Ordered image paths.
        image_roles: Addressable image roles.
        instruction: Optional visible instruction.
        travel_mode: Runtime mode presented to the model.
        model_visible_context: Additional visible non-private context.
        evidence_catalog: Prompt-visible input catalog.
        input_modalities: Input availability flags.
        direction_context: Public action and probe vocabularies.
        temporal_evidence_eligible: VQA-only compatibility metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    source_record_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(
        default=None,
        min_length=1,
        max_length=4096,
    )
    travel_mode: str = Field(
        default="pedestrian_assist",
        min_length=1,
        max_length=64,
    )
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(
        min_length=1,
        max_length=64,
    )
    input_modalities: InputModalitiesV4
    direction_context: DirectionContextV4
    temporal_evidence_eligible: bool = False

    @model_validator(mode="after")
    def validate_observation(self) -> "NavObservationV4":
        """Validate image ordering, evidence identity, and modality flags."""

        if len(self.images) != len(self.image_roles):
            raise ValueError(
                "images and image_roles must have equal length"
            )
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError(
                "image_roles must be unique and addressable"
            )
        if self.image_roles.count("fpv.current") != 1:
            raise ValueError(
                "NavObservationV4 requires exactly one fpv.current"
            )
        refs = [str(entry.ref) for entry in self.evidence_catalog]
        if len(refs) != len(set(refs)):
            raise ValueError("evidence_catalog refs must be unique")
        ref_set = set(refs)
        for role in self.image_roles:
            if (
                role.startswith(("fpv.", "bev.", "goal."))
                and role not in ref_set
            ):
                raise ValueError(
                    "image role is missing from evidence catalog: "
                    f"{role}"
                )
        expected_flags = {
            "history_fpv": any(
                role.startswith("fpv.history_")
                for role in self.image_roles
            ),
            "instruction": self.instruction is not None,
            "bev": any(
                role.startswith("bev.")
                for role in self.image_roles
            ),
            "goal_image": any(
                role.startswith("goal.image_")
                for role in self.image_roles
            ),
            "pixel_goal": any(
                entry.modality == "pixel_goal"
                for entry in self.evidence_catalog
            ),
            "waypoint": any(
                entry.modality == "waypoint"
                for entry in self.evidence_catalog
            ),
        }
        actual_flags = self.input_modalities.model_dump(mode="python")
        for name, expected in expected_flags.items():
            if bool(actual_flags[name]) != bool(expected):
                raise ValueError(
                    f"input modality flag mismatch: {name}="
                    f"{actual_flags[name]}, expected={expected}"
                )
        if (
            self.temporal_evidence_eligible
            and not self.input_modalities.history_fpv
        ):
            raise ValueError(
                "temporal compatibility flag requires history FPV"
            )
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
    """Carry one complete deterministic direction label."""

    @model_validator(mode="after")
    def validate_profile_stamp(self) -> "DirectionSupervisionV4":
        """Require the profile's canonical trajectory derivation."""

        expected = canonical_direction_derivation_version(
            self.profile_id
        )
        if self.derivation_version != expected:
            raise ValueError("direction derivation_version mismatch")
        allowed_sources = canonical_direction_source_codes(
            self.profile_id,
            include_stop_rule=str(self.label_name) == "stop",
        )
        if str(self.source) not in allowed_sources:
            raise ValueError(
                "direction source is incompatible with selected profile: "
                f"{self.source}"
            )
        if direction_source_requires_stop(str(self.source)) and str(
            self.label_name
        ) != "stop":
            raise ValueError(
                "stop-only direction source requires label_name=stop"
            )
        coarse_motion_for_direction_label(
            self.profile_id,
            str(self.label_name),
        )
        return self


class ReasoningSourceV4(BaseModel):
    """Partition visible observation from private deterministic sidecars.

    Args:
        schema_version: Exact V4.5 source stamp.
        observation: Teacher/student-visible input.
        direction_supervision: Optional strong direction derived upstream.
        trajectory_ego_t: Optional existing System1 target.
        trajectory_policy: Versioned trajectory contract.
        action_mask: Optional existing trajectory mask.
        warnings: Source-construction diagnostics.
        source_fingerprint: SHA-256 over all other source fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_SOURCE_V4_SCHEMA_VERSION] = (
        REASONING_SOURCE_V4_SCHEMA_VERSION
    )
    observation: NavObservationV4
    direction_supervision: DirectionSupervisionV4 | None = None
    trajectory_ego_t: list[list[float]] | None = None
    trajectory_policy: dict[str, Any] | None = None
    action_mask: list[bool] | list[int] | None = None
    warnings: list[str] = Field(default_factory=list, max_length=64)
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "ReasoningSourceV4":
        """Validate System1 sidecars and the frozen fingerprint."""

        has_trajectory = self.trajectory_ego_t is not None
        if has_trajectory != (self.trajectory_policy is not None):
            raise ValueError(
                "trajectory_ego_t and trajectory_policy must co-occur"
            )
        if self.action_mask is not None and not has_trajectory:
            raise ValueError("action_mask requires trajectory_ego_t")
        if self.trajectory_ego_t is not None:
            _validate_xyt(self.trajectory_ego_t)
            if (
                self.action_mask is not None
                and len(self.action_mask) != len(self.trajectory_ego_t)
            ):
                raise ValueError(
                    "action_mask length must match trajectory_ego_t"
                )
        if self.direction_supervision is not None:
            context = self.observation.direction_context
            direction = self.direction_supervision
            if (
                direction.profile_id != context.action_profile_id
                or direction.profile_version
                != context.action_profile_version
                or str(direction.label_name)
                not in context.action_labels
            ):
                raise ValueError(
                    "direction supervision does not match public context"
                )
        expected = reasoning_source_fingerprint_v4(
            self.model_dump(
                mode="json",
                exclude={"source_fingerprint"},
            )
        )
        if expected != self.source_fingerprint:
            raise ValueError(
                "V4 source_fingerprint does not match frozen payload"
            )
        return self


class GroundingOutputsV4(BaseModel):
    """Store independently produced grounding sections.

    Args:
        task_grounding: Deterministic task taxonomy.
        scene_grounding: Coarse visible scene summary.
        local_traversability: Physical direction affordances.
        navigation_context_grounding: Visible route direction.
        interaction_constraint_grounding: Behavior constraints.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_grounding: TaskGroundingV4 | None = None
    scene_grounding: SceneGroundingV4 | None = None
    local_traversability: LocalTraversabilityV4 | None = None
    navigation_context_grounding: (
        NavigationContextGroundingV4 | None
    ) = None
    interaction_constraint_grounding: (
        InteractionConstraintGroundingV4 | None
    ) = None

    @property
    def complete(self) -> bool:
        """Return whether all five grounding sections exist.

        Returns:
            True only when every grounding section is present.
        """

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
    """Record one independent section-production attempt stream.

    Args:
        status: Final stage result.
        attempts: Number of production attempts.
        raw_answer: Optional bounded raw answer for audit.
        errors: Hard stage findings.
        warnings: Non-fatal findings.
        telemetry: Provider timing and token telemetry.
        producer_kind: Deterministic canonicalizer or LVLM teacher.
        producer_profile_id: Exact producer profile.
        derivation_version: Exact deterministic or prompt derivation version.
        prompt_policy_digest: Prompt-policy digest, absent for deterministic
            production.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AnnotationStageStatusV4
    attempts: int = Field(ge=0, le=32)
    raw_answer: str | None = None
    errors: list[str] = Field(default_factory=list, max_length=64)
    warnings: list[str] = Field(default_factory=list, max_length=64)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    producer_kind: Literal["deterministic", "lvlm"]
    producer_profile_id: str = Field(min_length=1, max_length=128)
    derivation_version: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_producer(self) -> "AnnotationStageRecordV4":
        """Keep producer kind and prompt digest consistent."""

        if (
            self.producer_kind == "lvlm"
            and self.prompt_policy_digest is None
        ):
            raise ValueError(
                "LVLM stage requires prompt_policy_digest"
            )
        if (
            self.producer_kind == "deterministic"
            and self.prompt_policy_digest is not None
        ):
            raise ValueError(
                "deterministic stage cannot carry prompt_policy_digest"
            )
        return self


class ActionComparisonFactsV4(BaseModel):
    """Store irreducible proposal-versus-strong direction facts.

    Args:
        angular_delta_deg: Absolute angle delta for two moving actions.
        stop_move_mismatch: Exactly one action is stop.
        sign_conflict: Moving actions turn to opposite sides.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    angular_delta_deg: float | None = Field(
        default=None,
        ge=0.0,
        le=360.0,
    )
    stop_move_mismatch: bool
    sign_conflict: bool


class ActionProbeProjectionV4(BaseModel):
    """Store the nearest public affordance probe for the strong action.

    Args:
        direction_label: Nearest public probe direction.
        angle_gap_deg: Absolute action-to-probe angle gap.
        affordance: Probe affordance when annotated.
        clearance: Probe clearance when annotated.
        limiting_factors: Probe physical limiting factors.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_label: ReasonCode | None = None
    angle_gap_deg: float | None = Field(
        default=None,
        ge=0.0,
        le=180.0,
    )
    affordance: str | None = Field(default=None, max_length=64)
    clearance: str | None = Field(default=None, max_length=64)
    limiting_factors: tuple[ReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_projection(self) -> "ActionProbeProjectionV4":
        """Require label and angle gap to co-occur."""

        if (self.direction_label is None) != (
            self.angle_gap_deg is None
        ):
            raise ValueError(
                "probe direction_label and angle_gap_deg must co-occur"
            )
        return self


class ActionReconciliationChecksV4(BaseModel):
    """Store the four deterministic reconciliation check outcomes.

    Args:
        proposal_strong_relation: Proposal versus strong-label relation.
        probe_support: Aggregate support across all equally near probes.
        constraint_effect_consistency: Whether the deterministic behavior tuple
            is compatible with active constraints.
        action_tuple_consistency: Whether action, speed, and response agree.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_strong_relation: Literal[
        "exact",
        "within_tolerance",
        "large_delta",
        "sign_conflict",
        "stop_move_mismatch",
    ]
    probe_support: Literal[
        "all_supported",
        "all_unsupported",
        "mixed",
        "unknown",
        "not_applicable",
    ]
    constraint_effect_consistency: Literal[
        "passed",
        "conflict",
        "not_applicable",
    ]
    action_tuple_consistency: Literal["passed", "conflict"]


class ActionReconciliationAuditV4(BaseModel):
    """Store immutable audit facts without duplicating parent payloads.

    Args:
        schema_version: Exact audit schema stamp.
        audit_id: Stable SHA-256 identity of the audit inputs and checks.
        source_fingerprint: Parent source identity.
        proposal_digest: Parent Decision proposal digest.
        groundings_digest: Parent grounding-output digest.
        comparison: Proposal-versus-strong direction facts.
        probe_projections: All equally nearest physical probe summaries.
        checks: Four deterministic consistency outcomes.
        review_trigger_codes: Sorted machine-readable review reasons.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        ACTION_RECONCILIATION_AUDIT_V4_SCHEMA_VERSION
    ] = ACTION_RECONCILIATION_AUDIT_V4_SCHEMA_VERSION
    audit_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    proposal_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    groundings_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    comparison: ActionComparisonFactsV4
    probe_projections: tuple[ActionProbeProjectionV4, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    checks: ActionReconciliationChecksV4
    review_trigger_codes: tuple[ReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )

    @model_validator(mode="after")
    def validate_triggers(self) -> "ActionReconciliationAuditV4":
        """Require deterministic projections and trigger codes."""

        projection_labels = [
            str(item.direction_label)
            for item in self.probe_projections
        ]
        if any(
            item.direction_label is None
            for item in self.probe_projections
        ):
            raise ValueError(
                "probe_projections cannot contain empty placeholders"
            )
        if projection_labels != sorted(set(projection_labels)):
            raise ValueError(
                "probe_projections must be label-sorted and unique"
            )
        values = [str(value) for value in self.review_trigger_codes]
        if values != sorted(set(values)):
            raise ValueError(
                "review_trigger_codes must be sorted and unique"
            )
        return self


class ActionReviewComponentAssessmentV4(BaseModel):
    """Record the LVLM's comparative assessment of the three error sources.

    Args:
        teacher_proposal: Assessment of the visual Decision proposal.
        grounding: Aggregate assessment of the five grounding sections.
        strong_label: Assessment of the deterministic direction sidecar.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    teacher_proposal: ActionReviewAssessmentV4
    grounding: ActionReviewAssessmentV4
    strong_label: ActionReviewAssessmentV4


class ActionBlindReviewV4(BaseModel):
    """Store one strong-label-blind visual evidence assessment.

    The blind pass cannot see the teacher proposal, private strong direction,
    trajectory sidecars, deterministic action audit, or prior human decisions.
    It records evidence quality and findings, not an action correction.

    Args:
        schema_version: Exact blind-review schema stamp.
        evidence_sufficiency: Whether the visible evidence supports comparison.
        visual_findings: Concise lower-snake-case visual findings.
        ambiguous_evidence: Concise ambiguity reason codes.
        relevant_components: Grounding responsibilities implicated by evidence.
        evidence_analysis: Short evidence-only explanation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        ACTION_BLIND_REVIEW_V4_SCHEMA_VERSION
    ] = ACTION_BLIND_REVIEW_V4_SCHEMA_VERSION
    evidence_sufficiency: BlindEvidenceSufficiencyV4
    visual_findings: tuple[ReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=24,
    )
    ambiguous_evidence: tuple[ReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    relevant_components: tuple[GroundingSectionNameV4, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    evidence_analysis: str = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def validate_blind_review(self) -> "ActionBlindReviewV4":
        """Require canonical ordering and explicit insufficient-evidence codes."""

        _require_sorted_unique_codes_v4(
            self.visual_findings,
            "visual_findings",
        )
        _require_sorted_unique_codes_v4(
            self.ambiguous_evidence,
            "ambiguous_evidence",
        )
        _require_sorted_unique_strings_v4(
            self.relevant_components,
            "relevant_components",
        )
        if (
            self.evidence_sufficiency == "insufficient"
            and not self.ambiguous_evidence
        ):
            raise ValueError(
                "insufficient blind evidence requires ambiguity codes"
            )
        return self


class ActionReviewDiagnosisPayloadV4(BaseModel):
    """Store the four-way semantic output of comparative LVLM filtering.

    Args:
        verdict: Exact four-way suspected error source.
        error_subtype: Optional verdict-specific bounded subtype.
        affected_components: Components requiring human inspection.
        component_assessment: Comparative assessment of the three sources.
        reason_codes: Sorted machine-readable diagnosis reasons.
        error_analysis: Concise comparison grounded in visible evidence.
        repair_action: Non-authoritative repair routing recommendation.
        fields_to_inspect: Canonical paths for human inspection, without values.
        confidence: Uncalibrated model self-score used only for queue ordering.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: ActionReviewVerdictV4
    error_subtype: ActionReviewErrorSubtypeV4 | None = None
    affected_components: tuple[ActionReviewComponentV4, ...] = Field(
        min_length=1,
        max_length=7,
    )
    component_assessment: ActionReviewComponentAssessmentV4
    reason_codes: tuple[ReasonCode, ...] = Field(
        min_length=1,
        max_length=32,
    )
    error_analysis: str = Field(min_length=1, max_length=4096)
    repair_action: ActionReviewRepairActionV4
    fields_to_inspect: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_diagnosis_payload(
        self,
    ) -> "ActionReviewDiagnosisPayloadV4":
        """Require a canonical four-way verdict without corrected values."""

        _validate_action_review_semantics_v4(
            verdict=self.verdict,
            error_subtype=self.error_subtype,
            affected_components=self.affected_components,
            component_assessment=self.component_assessment,
            reason_codes=self.reason_codes,
            repair_action=self.repair_action,
            fields_to_inspect=self.fields_to_inspect,
            allow_unresolved_keep=False,
        )
        return self


class ActionReviewDiagnosisV4(ActionReviewDiagnosisPayloadV4):
    """Bind one validated LVLM diagnosis to an immutable action audit.

    Args:
        schema_version: Exact diagnosis schema stamp.
        filter_version: Exact filtering-policy version.
        diagnosis_id: Content address of all other diagnosis fields.
        source_group_id: Parent source group used for review routing.
        audit_id: Exact deterministic audit being diagnosed.
    """

    schema_version: Literal[
        ACTION_REVIEW_DIAGNOSIS_V4_SCHEMA_VERSION
    ] = ACTION_REVIEW_DIAGNOSIS_V4_SCHEMA_VERSION
    filter_version: Literal[
        ACTION_REVIEW_FILTER_VERSION_V4
    ] = ACTION_REVIEW_FILTER_VERSION_V4
    diagnosis_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_group_id: str = Field(min_length=1, max_length=240)
    audit_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_diagnosis_id(self) -> "ActionReviewDiagnosisV4":
        """Require the diagnosis identity to match its canonical payload."""

        expected = action_review_diagnosis_id_v4(
            self.model_dump(
                mode="json",
                exclude={"diagnosis_id"},
                exclude_none=True,
            )
        )
        if self.diagnosis_id != expected:
            raise ValueError(
                "diagnosis_id does not match canonical diagnosis payload"
            )
        return self


class ActionReviewDiagnosisRecordV4(BaseModel):
    """Store one resumable two-stage LVLM filtering record.

    Args:
        schema_version: Exact record schema stamp.
        source_group_id: Parent source group.
        source_fingerprint: Exact parent source revision.
        audit_id: Exact deterministic audit.
        record_status: Whether both bounded LVLM stages completed.
        blind_assessment: Parsed strong-label-blind evidence assessment.
        diagnosis: Parsed comparative four-way diagnosis.
        stage_records: Blind and comparative provider attempt records.
        errors: Record-level failures.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        ACTION_REVIEW_DIAGNOSIS_RECORD_V4_SCHEMA_VERSION
    ] = ACTION_REVIEW_DIAGNOSIS_RECORD_V4_SCHEMA_VERSION
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    audit_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    record_status: RecordStatusV4
    blind_assessment: ActionBlindReviewV4 | None = None
    diagnosis: ActionReviewDiagnosisV4 | None = None
    stage_records: dict[
        Literal["blind_visual", "comparative_diagnosis"],
        AnnotationStageRecordV4,
    ] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def validate_diagnosis_record(
        self,
    ) -> "ActionReviewDiagnosisRecordV4":
        """Keep record status, stage set, and parent identities consistent."""

        expected_stages = {"blind_visual", "comparative_diagnosis"}
        if self.record_status == "complete":
            if (
                self.blind_assessment is None
                or self.diagnosis is None
                or set(self.stage_records) != expected_stages
                or any(
                    value.status != "accepted"
                    for value in self.stage_records.values()
                )
            ):
                raise ValueError(
                    "complete diagnosis record requires two accepted stages"
                )
            if (
                self.diagnosis.source_group_id != self.source_group_id
                or self.diagnosis.audit_id != self.audit_id
            ):
                raise ValueError(
                    "diagnosis record identity does not match diagnosis"
                )
        elif self.diagnosis is not None:
            raise ValueError(
                "error diagnosis record cannot carry a final diagnosis"
            )
        return self


class ActionAdjudicationPayloadV4(BaseModel):
    """Store the semantic fields exported by one human reviewer.

    Args:
        diagnosis_disposition: Whether the reviewer confirmed or overrode LVLM.
        final_verdict: Exact four-way human verdict.
        error_subtype: Optional verdict-specific subtype.
        affected_components: Human-confirmed repair scope.
        reason_codes: Sorted human reason codes.
        repair_action: Human disposition for the current audit lineage.
        fields_to_inspect: Canonical paths sent to the repair queue.
        note: Optional concise human note.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnosis_disposition: ActionReviewDiagnosisDispositionV4
    final_verdict: ActionReviewVerdictV4
    error_subtype: ActionReviewErrorSubtypeV4 | None = None
    affected_components: tuple[ActionReviewComponentV4, ...] = Field(
        min_length=1,
        max_length=7,
    )
    reason_codes: tuple[ReasonCode, ...] = Field(
        min_length=1,
        max_length=32,
    )
    repair_action: ActionReviewRepairActionV4
    fields_to_inspect: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )
    note: str | None = Field(default=None, max_length=2048)

    @model_validator(mode="after")
    def validate_adjudication_payload(
        self,
    ) -> "ActionAdjudicationPayloadV4":
        """Require a valid human verdict and fail-closed disposition."""

        assessment = ActionReviewComponentAssessmentV4(
            teacher_proposal=(
                "likely_wrong"
                if self.final_verdict == "teacher_proposal_error"
                else "uncertain"
            ),
            grounding=(
                "likely_wrong"
                if self.final_verdict == "grounding_error"
                else "uncertain"
            ),
            strong_label=(
                "likely_wrong"
                if self.final_verdict == "strong_label_error"
                else "uncertain"
            ),
        )
        _validate_action_review_semantics_v4(
            verdict=self.final_verdict,
            error_subtype=self.error_subtype,
            affected_components=self.affected_components,
            component_assessment=assessment,
            reason_codes=self.reason_codes,
            repair_action=self.repair_action,
            fields_to_inspect=self.fields_to_inspect,
            allow_unresolved_keep=True,
        )
        if (
            self.diagnosis_disposition == "deferred"
            and (
                self.final_verdict != "unresolved"
                or self.repair_action != "quarantine"
            )
        ):
            raise ValueError(
                "deferred adjudication must quarantine an unresolved sample"
            )
        return self


class ActionAdjudicationV4(ActionAdjudicationPayloadV4):
    """Bind one append-only human adjudication to a diagnosis and audit.

    Args:
        schema_version: Exact adjudication schema stamp.
        adjudication_id: Content address of all other adjudication fields.
        source_group_id: Parent source group.
        source_fingerprint: Exact parent source revision.
        audit_id: Exact deterministic audit.
        diagnosis_id: Exact LVLM diagnosis inspected by the reviewer.
        reviewer: Stable human reviewer identity.
        reviewed_at: Timezone-aware review timestamp.
    """

    schema_version: Literal[
        ACTION_ADJUDICATION_V4_SCHEMA_VERSION
    ] = ACTION_ADJUDICATION_V4_SCHEMA_VERSION
    adjudication_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    audit_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    diagnosis_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    reviewer: str = Field(min_length=1, max_length=128)
    reviewed_at: datetime

    @model_validator(mode="after")
    def validate_adjudication_id(self) -> "ActionAdjudicationV4":
        """Require timezone-aware provenance and a canonical identity."""

        if self.reviewed_at.tzinfo is None:
            raise ValueError("reviewed_at must be timezone-aware")
        expected = action_adjudication_id_v4(
            self.model_dump(
                mode="json",
                exclude={"adjudication_id"},
                exclude_none=True,
            )
        )
        if self.adjudication_id != expected:
            raise ValueError(
                "adjudication_id does not match canonical payload"
            )
        return self


class ReasoningRepairRequestV4(BaseModel):
    """Store one deterministic repair request derived from adjudication.

    Args:
        schema_version: Exact repair-request schema stamp.
        repair_request_id: Content address of all other request fields.
        source_group_id: Parent source group.
        source_fingerprint: Exact source revision requiring repair.
        audit_id: Rejected audit lineage.
        adjudication_id: Human decision authorizing the repair route.
        repair_action: Exact upstream repair family.
        affected_components: Human-confirmed repair scope.
        fields_to_inspect: Canonical paths for repair tooling.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        REASONING_REPAIR_REQUEST_V4_SCHEMA_VERSION
    ] = REASONING_REPAIR_REQUEST_V4_SCHEMA_VERSION
    repair_request_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    audit_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    adjudication_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    repair_action: Literal[
        "rerun_teacher_proposal",
        "relabel_grounding",
        "rebuild_strong_sidecar",
    ]
    affected_components: tuple[ActionReviewComponentV4, ...] = Field(
        min_length=1,
        max_length=7,
    )
    fields_to_inspect: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )

    @model_validator(mode="after")
    def validate_repair_request(self) -> "ReasoningRepairRequestV4":
        """Require canonical scope and content-addressed request identity."""

        _require_sorted_unique_strings_v4(
            self.affected_components,
            "affected_components",
        )
        _validate_review_field_paths_v4(self.fields_to_inspect)
        expected = reasoning_repair_request_id_v4(
            self.model_dump(
                mode="json",
                exclude={"repair_request_id"},
                exclude_none=True,
            )
        )
        if self.repair_request_id != expected:
            raise ValueError(
                "repair_request_id does not match canonical payload"
            )
        return self


class CandidateStructureRecordV4(BaseModel):
    """Record deterministic candidate construction independently from release.

    Args:
        composer_version: Exact composer implementation version.
        status: Candidate structural validity.
        errors: Candidate construction or validation failures.
        warnings: Non-fatal candidate findings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    composer_version: str = Field(min_length=1, max_length=128)
    status: CandidateStructureStatusV4
    errors: list[str] = Field(default_factory=list, max_length=64)
    warnings: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_record(self) -> "CandidateStructureRecordV4":
        """Keep status and errors consistent."""

        if self.status == "valid" and self.errors:
            raise ValueError(
                "valid candidate structure cannot carry errors"
            )
        if self.status == "invalid" and not self.errors:
            raise ValueError(
                "invalid candidate structure requires at least one error"
            )
        if self.status == "not_available" and not self.errors:
            raise ValueError(
                "unavailable candidate structure requires a reason"
            )
        return self


class ReasoningAnnotationV4(BaseModel):
    """Store one immutable V4.6 annotation record.

    ``record_status`` only describes whether the bounded pipeline completed.
    Section acceptance, candidate structural validity, review state, and
    formal-training eligibility remain independent axes.

    Args:
        schema_version: Exact annotation schema stamp.
        source_group_id: Parent source group.
        source_fingerprint: Exact parent source identity.
        record_status: Pipeline completion status.
        grounding_outputs: Available validated groundings.
        decision_proposal: Visual grounding-conditioned LVLM proposal.
        action_reconciliation: Immutable proposal/strong-label audit.
        composer_candidate: Deterministic final Decision when valid.
        candidate_target: Six-section candidate containing that Decision.
        candidate_structure_record: Independent candidate-construction status.
        stage_records: One stage record per attempted section.
        errors: Record-level failures.
        warnings: Record-level non-fatal findings.
        teacher_provider: Optional provider identity.
        teacher_model: Optional model identity.
        metadata: Non-semantic run metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_ANNOTATION_V4_SCHEMA_VERSION] = (
        REASONING_ANNOTATION_V4_SCHEMA_VERSION
    )
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    record_status: RecordStatusV4
    grounding_outputs: GroundingOutputsV4
    decision_proposal: DecisionProposalV4 | None = None
    action_reconciliation: ActionReconciliationAuditV4 | None = None
    composer_candidate: DecisionV4 | None = None
    candidate_target: ReasoningTargetV4 | None = None
    candidate_structure_record: CandidateStructureRecordV4
    stage_records: dict[
        ReasoningSectionNameV4,
        AnnotationStageRecordV4,
    ] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list, max_length=64)
    warnings: list[str] = Field(default_factory=list, max_length=128)
    teacher_provider: str | None = Field(default=None, max_length=64)
    teacher_model: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_annotation(self) -> "ReasoningAnnotationV4":
        """Validate independent status axes and parent-child digests."""

        expected_sections = {
            "TaskGrounding",
            "SceneGrounding",
            "LocalTraversability",
            "NavigationContextGrounding",
            "InteractionConstraintGrounding",
            "Decision",
        }
        if self.record_status == "complete" and (
            set(self.stage_records) != expected_sections
        ):
            raise ValueError(
                "complete record requires exactly six stage records"
            )
        if (
            self.action_reconciliation is not None
            and self.decision_proposal is None
        ):
            raise ValueError(
                "action reconciliation requires Decision proposal"
            )
        if self.action_reconciliation is not None:
            assert self.decision_proposal is not None
            proposal_digest = reasoning_payload_digest_v4(
                self.decision_proposal.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
            groundings_digest = reasoning_payload_digest_v4(
                self.grounding_outputs.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
            if (
                self.action_reconciliation.source_fingerprint
                != self.source_fingerprint
                or self.action_reconciliation.proposal_digest
                != proposal_digest
                or self.action_reconciliation.groundings_digest
                != groundings_digest
            ):
                raise ValueError(
                    "action reconciliation does not match parent payload"
                )
        if (self.composer_candidate is None) != (
            self.candidate_target is None
        ):
            raise ValueError(
                "composer_candidate and candidate_target must co-occur"
            )
        if (
            self.candidate_target is not None
            and self.candidate_target.reasoning.decision
            != self.composer_candidate
        ):
            raise ValueError(
                "candidate target Decision must equal composer_candidate"
            )
        has_candidate = self.composer_candidate is not None
        if (
            self.candidate_structure_record.status == "valid"
        ) != has_candidate:
            raise ValueError(
                "only a structurally valid candidate may be carried"
            )
        return self

    @property
    def all_sections_accepted(self) -> bool:
        """Return whether all six independent stages were accepted.

        Returns:
            True only for the exact accepted stage set.
        """

        expected = {
            "TaskGrounding",
            "SceneGrounding",
            "LocalTraversability",
            "NavigationContextGrounding",
            "InteractionConstraintGrounding",
            "Decision",
        }
        return (
            set(self.stage_records) == expected
            and all(
                record.status == "accepted"
                for record in self.stage_records.values()
            )
        )


class ReasoningTrainingViewV4(BaseModel):
    """Cross-repository compact FullCoT view consumed by ms-swift.

    Review previews and formal training rows share one syntax, but only formal
    rows may be adapted into LM-loss examples.  Formal rows are fail-closed on
    every eligibility axis.

    Args:
        schema_version: Exact V4.6 training-view stamp.
        reasoning_contract_schema_digest: Exact canonical contract digest.
        sample_id: Stable view identity.
        source_group_id: Parent source group.
        split_group_id: Leakage-safe split group.
        dataset_source: Dataset family.
        view_type: Compact FullCoT SFT.
        release_track: Review-only or formal-training track.
        audit_id: Exact action audit.
        review_status: Derived review state.
        artifact_status: Whether this artifact is preview-only or training-ready.
        artifact_training_eligible: Whether this exact artifact may be trained.
        formal_training_eligible: Eligibility of the joined canonical record.
        images: Ordered prompt images.
        image_roles: Addressable image roles.
        instruction: Optional visible instruction.
        model_visible_context: Additional visible context.
        evidence_catalog: Prompt-visible input catalog.
        input_modalities: Input availability flags.
        direction_context: Public direction profiles.
        assistant_text: Canonical compact six-section text.
        reasoning_target: Exact six-section typed target.
        full_cot_renderer_id: Exact compact renderer.
        prompt_profile_id: Student prompt profile.
        prompt_policy_digest: Exact student prompt-policy digest.
        direction_supervision: Complete private direction stamp.
        trajectory_ego_t: Optional existing System1 target.
        trajectory_policy: Existing System1 trajectory contract.
        action_mask: Optional existing System1 mask.
        metadata: Non-semantic export metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION
    ] = REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION
    reasoning_contract_schema_digest: Literal[
        REASONING_V4_CONTRACT_SCHEMA_DIGEST
    ] = REASONING_V4_CONTRACT_SCHEMA_DIGEST
    sample_id: str = Field(min_length=1, max_length=320)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    view_type: TrainingViewTypeV4 = "full_cot_sft"
    release_track: ReleaseTrackV4
    audit_id: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    review_status: ProjectedReviewStatusV4
    artifact_status: ArtifactStatusV4
    artifact_training_eligible: bool
    formal_training_eligible: bool
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(
        default=None,
        min_length=1,
        max_length=4096,
    )
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(
        min_length=1,
        max_length=64,
    )
    input_modalities: InputModalitiesV4
    direction_context: DirectionContextV4
    assistant_text: str = Field(min_length=1)
    reasoning_target: ReasoningTargetV4
    full_cot_renderer_id: Literal[
        REASONING_V4_COMPACT_FULL_COT_RENDERER_ID
    ] = REASONING_V4_COMPACT_FULL_COT_RENDERER_ID
    prompt_profile_id: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    direction_supervision: DirectionSupervisionV4
    trajectory_ego_t: list[list[float]] | None = None
    trajectory_policy: dict[str, Any] | None = None
    action_mask: list[bool] | list[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_view(self) -> "ReasoningTrainingViewV4":
        """Fail closed on release-track and strong-action invariants."""

        if len(self.images) != len(self.image_roles):
            raise ValueError(
                "images and image_roles must have equal length"
            )
        if self.release_track == "formal_training":
            if self.artifact_status != "training_ready":
                raise ValueError(
                    "formal_training requires artifact_status=training_ready"
                )
            if not self.artifact_training_eligible:
                raise ValueError(
                    "formal_training artifact must be training eligible"
                )
            if not self.formal_training_eligible:
                raise ValueError(
                    "formal_training requires formal eligibility"
                )
            if self.review_status not in {
                "not_required",
                "resolved",
            }:
                raise ValueError(
                    "formal_training requires completed review state"
                )
            if self.audit_id is None:
                raise ValueError(
                    "formal_training requires action audit_id"
                )
        else:
            if self.artifact_status != "preview_only":
                raise ValueError(
                    "review_preview requires artifact_status=preview_only"
                )
            if self.artifact_training_eligible:
                raise ValueError(
                    "review_preview artifact cannot be training eligible"
                )
        direction = self.direction_supervision
        if (
            direction.profile_id
            != self.direction_context.action_profile_id
            or direction.profile_version
            != self.direction_context.action_profile_version
            or str(direction.label_name)
            not in self.direction_context.action_labels
        ):
            raise ValueError(
                "direction supervision does not match public context"
            )
        if (
            str(
                self.reasoning_target.reasoning.decision
                .selected_action_label
            )
            != str(direction.label_name)
        ):
            raise ValueError(
                "Decision selected action must equal strong direction"
            )
        has_trajectory = self.trajectory_ego_t is not None
        if has_trajectory != (self.trajectory_policy is not None):
            raise ValueError(
                "trajectory and trajectory_policy must co-occur"
            )
        if self.trajectory_ego_t is not None:
            _validate_xyt(self.trajectory_ego_t)
            if (
                len(self.trajectory_ego_t)
                != REASONING_V4_SYSTEM1_QUERY_COUNT
            ):
                raise ValueError(
                    "Reasoning V4.6 System1 trajectory must have "
                    f"{REASONING_V4_SYSTEM1_QUERY_COUNT} steps"
                )
            if self.action_mask is None:
                raise ValueError(
                    "Reasoning V4.6 System1 trajectory requires action_mask"
                )
            if (
                len(self.action_mask) != len(self.trajectory_ego_t)
            ):
                raise ValueError(
                    "action_mask length must match trajectory"
                )
            mask_values = [int(value) for value in self.action_mask]
            if (
                any(value not in {0, 1} for value in mask_values)
                or sum(mask_values)
                != REASONING_V4_SYSTEM1_QUERY_COUNT
            ):
                raise ValueError(
                    "Reasoning V4.6 System1 action_mask must mark "
                    f"all {REASONING_V4_SYSTEM1_QUERY_COUNT} steps valid"
                )
        if self.action_mask is not None and not has_trajectory:
            raise ValueError(
                "action_mask requires trajectory_ego_t"
            )
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
    """Hash one JSON-compatible source payload canonically.

    Args:
        payload: Source payload excluding the fingerprint field.

    Returns:
        Lowercase SHA-256 hex digest.
    """

    return reasoning_payload_digest_v4(payload)


def reasoning_payload_digest_v4(payload: Any) -> str:
    """Hash one JSON-compatible V4 payload canonically.

    Args:
        payload: JSON-compatible value.

    Returns:
        Lowercase SHA-256 hex digest.
    """

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def action_review_diagnosis_id_v4(payload: Any) -> str:
    """Return the canonical identity of one LVLM review diagnosis.

    Args:
        payload: Diagnosis payload excluding ``diagnosis_id``.

    Returns:
        Lowercase SHA-256 hex digest.
    """

    return reasoning_payload_digest_v4(payload)


def action_adjudication_id_v4(payload: Any) -> str:
    """Return the canonical identity of one human adjudication.

    Args:
        payload: Adjudication payload excluding ``adjudication_id``.

    Returns:
        Lowercase SHA-256 hex digest.
    """

    return reasoning_payload_digest_v4(payload)


def reasoning_repair_request_id_v4(payload: Any) -> str:
    """Return the canonical identity of one derived repair request.

    Args:
        payload: Repair request payload excluding ``repair_request_id``.

    Returns:
        Lowercase SHA-256 hex digest.
    """

    return reasoning_payload_digest_v4(payload)


def _require_sorted_unique_codes_v4(
    values: tuple[ReasonCode, ...],
    field_name: str,
) -> None:
    """Require one ReasonCode tuple to be sorted and duplicate-free."""

    _require_sorted_unique_strings_v4(values, field_name)


def _require_sorted_unique_strings_v4(
    values: tuple[Any, ...],
    field_name: str,
) -> None:
    """Require one tuple to be lexically sorted and duplicate-free."""

    normalized = [str(value) for value in values]
    if normalized != sorted(set(normalized)):
        raise ValueError(
            f"{field_name} must be sorted and unique"
        )


def _validate_review_field_paths_v4(
    values: tuple[str, ...],
) -> None:
    """Require sorted inspection paths without embedded corrected values."""

    _require_sorted_unique_strings_v4(
        values,
        "fields_to_inspect",
    )
    allowed_prefixes = (
        "decision_proposal",
        "grounding_outputs.task_grounding",
        "grounding_outputs.scene_grounding",
        "grounding_outputs.local_traversability",
        "grounding_outputs.navigation_context_grounding",
        "grounding_outputs.interaction_constraint_grounding",
        "direction_supervision",
    )
    forbidden_parts = (
        "=",
        "suggested",
        "corrected",
        "replacement",
        "new_value",
    )
    invalid = [
        value
        for value in values
        if not value.startswith(allowed_prefixes)
        or any(part in value.lower() for part in forbidden_parts)
    ]
    if invalid:
        raise ValueError(
            "fields_to_inspect contains unsupported or corrective paths: "
            f"{invalid}"
        )


def _validate_action_review_semantics_v4(
    *,
    verdict: ActionReviewVerdictV4,
    error_subtype: ActionReviewErrorSubtypeV4 | None,
    affected_components: tuple[ActionReviewComponentV4, ...],
    component_assessment: ActionReviewComponentAssessmentV4,
    reason_codes: tuple[ReasonCode, ...],
    repair_action: ActionReviewRepairActionV4,
    fields_to_inspect: tuple[str, ...],
    allow_unresolved_keep: bool,
) -> None:
    """Validate one diagnosis/adjudication without expanding its taxonomy."""

    _require_sorted_unique_strings_v4(
        affected_components,
        "affected_components",
    )
    _require_sorted_unique_codes_v4(reason_codes, "reason_codes")
    _validate_review_field_paths_v4(fields_to_inspect)
    subtype_groups = {
        "teacher_proposal_error": {
            "proposal_direction_error",
            "proposal_stop_error",
            "proposal_move_error",
            "proposal_speed_error",
            "proposal_response_error",
        },
        "grounding_error": {
            "task_grounding_error",
            "scene_grounding_error",
            "local_traversability_error",
            "navigation_context_error",
            "interaction_constraint_error",
        },
        "strong_label_error": {
            "direction_quantization_error",
            "direction_projection_error",
            "coordinate_convention_error",
            "stop_label_error",
            "trajectory_sidecar_error",
        },
    }
    if verdict == "unresolved":
        if error_subtype is not None:
            raise ValueError(
                "unresolved verdict cannot carry an error_subtype"
            )
        allowed_actions = {"quarantine"}
        if allow_unresolved_keep:
            allowed_actions.add("keep_without_change")
        if repair_action not in allowed_actions:
            raise ValueError(
                "unresolved verdict must quarantine or use approved human keep"
            )
        if not any(
            value == "uncertain"
            for value in component_assessment.model_dump().values()
        ):
            raise ValueError(
                "unresolved verdict requires an uncertain component assessment"
            )
        return
    if (
        error_subtype is None
        or str(error_subtype) not in subtype_groups[verdict]
    ):
        raise ValueError(
            f"{verdict} requires a matching error_subtype"
        )
    if verdict == "teacher_proposal_error":
        if "DecisionProposal" not in affected_components:
            raise ValueError(
                "teacher_proposal_error must affect DecisionProposal"
            )
        if component_assessment.teacher_proposal != "likely_wrong":
            raise ValueError(
                "teacher_proposal_error requires likely_wrong proposal"
            )
        if repair_action not in {
            "keep_without_change",
            "rerun_teacher_proposal",
        }:
            raise ValueError(
                "teacher_proposal_error has an invalid repair action"
            )
    elif verdict == "grounding_error":
        grounding_components = {
            "TaskGrounding",
            "SceneGrounding",
            "LocalTraversability",
            "NavigationContextGrounding",
            "InteractionConstraintGrounding",
        }
        if not grounding_components.intersection(affected_components):
            raise ValueError(
                "grounding_error must affect a grounding component"
            )
        if component_assessment.grounding != "likely_wrong":
            raise ValueError(
                "grounding_error requires likely_wrong grounding"
            )
        if repair_action != "relabel_grounding":
            raise ValueError(
                "grounding_error requires relabel_grounding"
            )
    elif verdict == "strong_label_error":
        if "StrongDirectionSidecar" not in affected_components:
            raise ValueError(
                "strong_label_error must affect StrongDirectionSidecar"
            )
        if component_assessment.strong_label != "likely_wrong":
            raise ValueError(
                "strong_label_error requires likely_wrong strong label"
            )
        if repair_action != "rebuild_strong_sidecar":
            raise ValueError(
                "strong_label_error requires rebuild_strong_sidecar"
            )


def approved_direction_profiles_v4() -> dict[str, dict[str, Any]]:
    """Return the V4 direction allowlist snapshot.

    Returns:
        JSON-compatible profile registry.
    """

    return approved_stage_a_direction_profiles()


def _validate_xyt(value: list[list[float]]) -> None:
    """Validate one finite, non-empty cumulative ``[T, 3]`` trajectory."""

    if not value:
        raise ValueError("trajectory_ego_t must be non-empty")
    for index, row in enumerate(value):
        if len(row) != 3:
            raise ValueError(
                f"trajectory_ego_t[{index}] must have width 3"
            )
        if not all(math.isfinite(float(item)) for item in row):
            raise ValueError(
                f"trajectory_ego_t[{index}] must be finite"
            )


def _forbidden_visible_paths_v4(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return nested visible-context paths with private field names."""

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
    """Collect private paths from visible-context containers."""

    findings = _forbidden_visible_paths_v4(model_visible_context)
    for entry in evidence_catalog:
        findings.extend(
            f"evidence_catalog.{entry.ref}.attributes.{path}"
            for path in _forbidden_visible_paths_v4(entry.attributes)
        )
    return sorted(set(findings))


__all__ = [
    "ACTION_ADJUDICATION_V4_SCHEMA_VERSION",
    "ACTION_BLIND_REVIEW_V4_SCHEMA_VERSION",
    "ACTION_RECONCILIATION_AUDIT_V4_SCHEMA_VERSION",
    "ACTION_REVIEW_DIAGNOSIS_RECORD_V4_SCHEMA_VERSION",
    "ACTION_REVIEW_DIAGNOSIS_V4_SCHEMA_VERSION",
    "ACTION_REVIEW_FILTER_VERSION_V4",
    "ActionAdjudicationPayloadV4",
    "ActionAdjudicationV4",
    "ActionBlindReviewV4",
    "ActionComparisonFactsV4",
    "ActionProbeProjectionV4",
    "ActionReconciliationAuditV4",
    "ActionReconciliationChecksV4",
    "ActionReviewAssessmentV4",
    "ActionReviewComponentV4",
    "ActionReviewComponentAssessmentV4",
    "ActionReviewDiagnosisDispositionV4",
    "ActionReviewDiagnosisPayloadV4",
    "ActionReviewDiagnosisRecordV4",
    "ActionReviewDiagnosisV4",
    "ActionReviewErrorSubtypeV4",
    "ActionReviewRepairActionV4",
    "ActionReviewVerdictV4",
    "AnnotationStageRecordV4",
    "AnnotationStageStatusV4",
    "BevRouteEvidenceV4",
    "BlindEvidenceSufficiencyV4",
    "ArtifactStatusV4",
    "CandidateStructureRecordV4",
    "CandidateStructureStatusV4",
    "DirectionContextV4",
    "DirectionSupervisionV4",
    "EvidenceCatalogEntryV4",
    "EvidenceModalityV4",
    "GroundingOutputsV4",
    "GroundingSectionNameV4",
    "InputModalitiesV4",
    "NavObservationV4",
    "ProjectedReviewStatusV4",
    "REASONING_ANNOTATION_V4_SCHEMA_VERSION",
    "REASONING_SOURCE_V4_SCHEMA_VERSION",
    "REASONING_V4_SYSTEM1_QUERY_COUNT",
    "ReasoningAnnotationV4",
    "ReasoningRepairRequestV4",
    "ReasoningSectionNameV4",
    "ReasoningSourceV4",
    "ReasoningTrainingViewV4",
    "RecordStatusV4",
    "ReleaseTrackV4",
    "REASONING_REPAIR_REQUEST_V4_SCHEMA_VERSION",
    "TrainingViewTypeV4",
    "action_adjudication_id_v4",
    "action_review_diagnosis_id_v4",
    "approved_direction_profiles_v4",
    "reasoning_payload_digest_v4",
    "reasoning_repair_request_id_v4",
    "reasoning_source_fingerprint_v4",
]
