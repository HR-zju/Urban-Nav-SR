"""Source, annotation, capability, and training-view contracts for VQA V2."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.direction_profile import DirectionLabelStamp
from mcp_server.contracts.reasoning import ReasonCode
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionSupervisionV4,
    EvidenceCatalogEntryV4,
    InputModalitiesV4,
    NavObservationV4,
)
from mcp_server.contracts.vqa_v2 import (
    C2PixelAnswerV2,
    C2WaypointAnswerV2,
    C3AnswerV2,
    C4AnswerV2,
    C5AnswerV2,
    C6AnswerV2,
    E1AnswerV2,
    E2AnswerV2,
    E3AnswerV2,
    VQAAnswerV2,
    VQAQuestionV2,
    VQATaskIdV2,
    VQA_V2_TASK_IDS,
    get_vqa_task_spec_v2,
    validate_answer_for_question_v2,
)


VQA_SOURCE_V2_SCHEMA_VERSION = "urbannav_navigation_vqa_source_v2_0"
VQA_ANNOTATION_V2_SCHEMA_VERSION = (
    "urbannav_navigation_vqa_annotation_v2_0"
)
VQA_TRAINING_VIEW_V2_SCHEMA_VERSION = (
    "urbannav_navigation_vqa_training_view_v2_0"
)
VQA_CAPABILITY_CONFIG_SCHEMA_VERSION = (
    "urbannav_navigation_vqa_capabilities_v2_0"
)

VQALabelSourceV2 = Literal[
    "derived_gt",
    "lvlm_weak",
    "external_gt",
]


class VQATaskCapabilityV2(BaseModel):
    """Declare implementation and activation state for one atomic task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: VQATaskIdV2
    contract_ready: Literal[True] = True
    source_interface_ready: Literal[True] = True
    label_producer_ready: bool
    training_export_enabled: bool
    evaluation_enabled: bool
    producer_ids: list[str] = Field(default_factory=list, max_length=16)
    depends_on: list[VQATaskIdV2] = Field(
        default_factory=list,
        max_length=8,
    )
    disabled_reason: ReasonCode | None = None

    @model_validator(mode="after")
    def validate_activation(self) -> "VQATaskCapabilityV2":
        """Prevent enabled export without a real producer."""

        if self.training_export_enabled and not self.label_producer_ready:
            raise ValueError(
                "training export requires label_producer_ready"
            )
        if self.evaluation_enabled and not self.label_producer_ready:
            raise ValueError(
                "evaluation requires label_producer_ready"
            )
        if self.label_producer_ready and not self.producer_ids:
            raise ValueError(
                "ready label producer requires producer_ids"
            )
        if not self.label_producer_ready and self.producer_ids:
            raise ValueError(
                "dormant task cannot advertise producer_ids"
            )
        if self.training_export_enabled and self.disabled_reason is not None:
            raise ValueError(
                "enabled task cannot carry disabled_reason"
            )
        if not self.training_export_enabled and self.disabled_reason is None:
            raise ValueError(
                "disabled task requires disabled_reason"
            )
        if self.task_id in self.depends_on:
            raise ValueError("task cannot depend on itself")
        return self


class VQACapabilityConfigV2(BaseModel):
    """Freeze the task activation registry outside the wire contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        VQA_CAPABILITY_CONFIG_SCHEMA_VERSION
    ] = VQA_CAPABILITY_CONFIG_SCHEMA_VERSION
    profile_id: str = Field(min_length=1, max_length=128)
    tasks: list[VQATaskCapabilityV2] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self) -> "VQACapabilityConfigV2":
        """Require unique tasks and activated dependencies."""

        task_map = {item.task_id: item for item in self.tasks}
        if len(task_map) != len(self.tasks):
            raise ValueError("capability task ids must be unique")
        if set(task_map) != set(VQA_V2_TASK_IDS):
            raise ValueError(
                "capability registry must cover every VQA V2 task"
            )
        for item in self.tasks:
            missing = sorted(set(item.depends_on) - set(task_map))
            if missing:
                raise ValueError(
                    f"capability dependencies are missing: {missing}"
                )
            if item.training_export_enabled:
                inactive = [
                    dependency
                    for dependency in item.depends_on
                    if not task_map[
                        dependency
                    ].training_export_enabled
                ]
                if inactive:
                    raise ValueError(
                        "enabled task has inactive dependencies: "
                        f"{inactive}"
                    )
        return self

    def capability(self, task_id: str) -> VQATaskCapabilityV2:
        """Resolve one task capability.

        Args:
            task_id: Exact VQA task id.

        Returns:
            Frozen task capability.

        Raises:
            ValueError: If the registry has no such task.
        """

        for item in self.tasks:
            if item.task_id == task_id:
                return item
        raise ValueError(f"task is absent from capabilities: {task_id}")

    def digest(self) -> str:
        """Return a deterministic digest of the activation policy.

        Returns:
            Lowercase SHA-256 digest.
        """

        return _fingerprint(self.model_dump(mode="json"))


class ExternalLabelInterfaceBaseV2(BaseModel):
    """Base provenance required from a future external GT producer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_group_id: str = Field(min_length=1, max_length=240)
    producer_id: str = Field(min_length=1, max_length=128)
    producer_version: str = Field(min_length=1, max_length=128)
    input_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    qc_status: Literal["accepted"]
    evidence_refs: list[str] = Field(min_length=1, max_length=64)
    source_field_refs: list[str] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_refs(self) -> "ExternalLabelInterfaceBaseV2":
        """Require unique provenance references."""

        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("interface evidence refs must be unique")
        if len(set(self.source_field_refs)) != len(
            self.source_field_refs
        ):
            raise ValueError("source field refs must be unique")
        return self


class PixelGoalDirectionInterfaceV2(ExternalLabelInterfaceBaseV2):
    """Future validated pixel-goal direction binding."""

    source_kind: Literal["pixel_goal_direction"]
    coordinate_frame: Literal["fpv_pixel_xy_top_left"]
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    coordinate_convention_version: str = Field(
        min_length=1,
        max_length=128,
    )
    pixel_goal_source_ref: str = Field(min_length=1, max_length=240)
    direction: DirectionLabelStamp
    answer: C2PixelAnswerV2

    @model_validator(mode="after")
    def validate_pixel_binding(
        self,
    ) -> "PixelGoalDirectionInterfaceV2":
        """Bind the visible pixel goal and answer to the direction stamp."""

        if self.pixel_goal_source_ref not in self.evidence_refs:
            raise ValueError(
                "pixel goal source ref must be prompt-visible evidence"
            )
        payload = self.answer.payload
        if (
            payload.status == "aligned"
            and payload.direction_label != self.direction.label_name
        ):
            raise ValueError(
                "pixel-goal answer does not match direction stamp"
            )
        return self


class WaypointDirectionInterfaceV2(ExternalLabelInterfaceBaseV2):
    """Future validated waypoint direction binding."""

    source_kind: Literal["waypoint_direction"]
    coordinate_frame: str = Field(min_length=1, max_length=128)
    coordinate_unit: str = Field(min_length=1, max_length=64)
    transform_version: str = Field(min_length=1, max_length=128)
    waypoint_source_ref: str = Field(min_length=1, max_length=240)
    direction: DirectionLabelStamp
    answer: C2WaypointAnswerV2

    @model_validator(mode="after")
    def validate_waypoint_binding(
        self,
    ) -> "WaypointDirectionInterfaceV2":
        """Bind the visible waypoint and answer to the direction stamp."""

        if self.waypoint_source_ref not in self.evidence_refs:
            raise ValueError(
                "waypoint source ref must be prompt-visible evidence"
            )
        payload = self.answer.payload
        if (
            payload.status == "aligned"
            and payload.direction_label != self.direction.label_name
        ):
            raise ValueError(
                "waypoint answer does not match direction stamp"
            )
        return self


class BEVRouteAlignmentInterfaceV2(ExternalLabelInterfaceBaseV2):
    """Future independent BEV-route alignment binding."""

    source_kind: Literal["bev_route_direction"]
    independent_source: Literal[True]
    bev_evidence_ref: str = Field(min_length=1, max_length=240)
    fpv_evidence_ref: str = Field(min_length=1, max_length=240)
    direction: DirectionLabelStamp
    answer: C3AnswerV2

    @model_validator(mode="after")
    def validate_bev_source(self) -> "BEVRouteAlignmentInterfaceV2":
        """Reject trajectory or forbidden route-derived provenance."""

        source_text = " ".join(
            [self.direction.source, *self.source_field_refs]
        ).lower()
        forbidden = (
            "expert_trajectory",
            "gps",
            "gnss",
            "route_match",
            "route_theta",
        )
        if any(token in source_text for token in forbidden):
            raise ValueError(
                "BEV route binding cannot use trajectory, GPS, route "
                "matching, or route_theta"
            )
        if not {
            self.bev_evidence_ref,
            self.fpv_evidence_ref,
        }.issubset(set(self.evidence_refs)):
            raise ValueError(
                "BEV and FPV refs must be prompt-visible evidence"
            )
        payload = self.answer.payload
        if (
            payload.direction_label is not None
            and payload.direction_label != self.direction.label_name
        ):
            raise ValueError(
                "BEV route answer does not match direction stamp"
            )
        return self


class CrossViewEntityBindingInterfaceV2(
    ExternalLabelInterfaceBaseV2
):
    """Future validated BEV-to-FPV entity correspondence."""

    source_kind: Literal["cross_view_entity_binding"]
    bev_ref: str = Field(min_length=1, max_length=160)
    fpv_entity_ref: str = Field(min_length=1, max_length=160)
    direction: DirectionLabelStamp
    answer: C4AnswerV2

    @model_validator(mode="after")
    def validate_answer_refs(
        self,
    ) -> "CrossViewEntityBindingInterfaceV2":
        """Require the answer to preserve validated entity identities."""

        if not self.bev_ref.startswith("bev."):
            raise ValueError(
                "cross-view BEV ref must use a prompt-visible bev namespace"
            )
        if not self.fpv_entity_ref.startswith("fpv."):
            raise ValueError(
                "cross-view FPV ref must use a prompt-visible fpv namespace"
            )
        if not {
            self.bev_ref,
            self.fpv_entity_ref,
        }.issubset(set(self.evidence_refs)):
            raise ValueError(
                "cross-view entity ids must be prompt-visible evidence"
            )
        payload = self.answer.payload
        if (
            payload.bev_ref != self.bev_ref
            or payload.fpv_entity_ref != self.fpv_entity_ref
            or payload.direction_label != self.direction.label_name
        ):
            raise ValueError(
                "cross-view answer does not match validated binding"
            )
        return self


class OutsideViewDirectionInterfaceV2(
    ExternalLabelInterfaceBaseV2
):
    """Future binding that separates current and future direction."""

    source_kind: Literal["outside_view_direction"]
    current_direction: DirectionLabelStamp
    future_direction: DirectionLabelStamp
    future_horizon: float = Field(gt=0.0)
    future_horizon_unit: str = Field(min_length=1, max_length=32)
    transition_ref: str = Field(min_length=1, max_length=160)
    future_not_yet_executed: Literal[True]
    answer: C5AnswerV2

    @model_validator(mode="after")
    def validate_temporal_separation(
        self,
    ) -> "OutsideViewDirectionInterfaceV2":
        """Require answer directions to match distinct source records."""

        if self.transition_ref not in self.evidence_refs:
            raise ValueError(
                "future transition ref must be prompt-visible evidence"
            )
        if (
            self.current_direction.source
            == self.future_direction.source
            and self.current_direction.derivation_version
            == self.future_direction.derivation_version
        ):
            raise ValueError(
                "current and future direction require distinct sources"
            )
        current_profile = (
            self.current_direction.profile_id,
            self.current_direction.profile_version,
            self.current_direction.bin_mode,
        )
        future_profile = (
            self.future_direction.profile_id,
            self.future_direction.profile_version,
            self.future_direction.bin_mode,
        )
        if current_profile != future_profile:
            raise ValueError(
                "current and future directions require one profile"
            )
        payload = self.answer.payload
        if (
            payload.current_direction_label
            != self.current_direction.label_name
            or payload.future_direction_label
            != self.future_direction.label_name
        ):
            raise ValueError(
                "outside-view answer does not match direction bindings"
            )
        return self


class BEVFPVExecutabilityInterfaceV2(
    ExternalLabelInterfaceBaseV2
):
    """Future binding for route intent and local FPV executability."""

    source_kind: Literal["bev_fpv_executability"]
    bev_direction: DirectionLabelStamp
    bev_alignment_evidence_ref: str = Field(
        min_length=1,
        max_length=240,
    )
    fpv_executability_evidence_ref: str = Field(
        min_length=1,
        max_length=240,
    )
    local_executability_producer_id: str = Field(
        min_length=1,
        max_length=128,
    )
    local_executability_producer_version: str = Field(
        min_length=1,
        max_length=128,
    )
    answer: C6AnswerV2

    @model_validator(mode="after")
    def validate_executability_binding(
        self,
    ) -> "BEVFPVExecutabilityInterfaceV2":
        """Bind the route direction and local-executability evidence."""

        if not {
            self.bev_alignment_evidence_ref,
            self.fpv_executability_evidence_ref,
        }.issubset(set(self.evidence_refs)):
            raise ValueError(
                "BEV alignment and FPV executability refs must be evidence"
            )
        payload = self.answer.payload
        if (
            payload.direction_label is not None
            and payload.direction_label
            != self.bev_direction.label_name
        ):
            raise ValueError(
                "BEV-FPV answer does not match direction stamp"
            )
        return self


class PromptVisibleCandidateV2(BaseModel):
    """Describe one candidate representation actually visible to students."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1, max_length=128)
    prompt_visible_representation_ref: str = Field(
        min_length=1,
        max_length=240,
    )
    direction: DirectionLabelStamp | None = None


class CandidateSetV2(BaseModel):
    """Freeze candidate identity and presentation order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_set_id: str = Field(min_length=1, max_length=160)
    candidate_producer_id: str = Field(min_length=1, max_length=128)
    candidate_producer_version: str = Field(
        min_length=1,
        max_length=128,
    )
    candidates: list[PromptVisibleCandidateV2] = Field(
        min_length=1,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_candidates(self) -> "CandidateSetV2":
        """Require unique IDs and visible representations."""

        candidate_ids = [
            candidate.candidate_id for candidate in self.candidates
        ]
        visible_refs = [
            candidate.prompt_visible_representation_ref
            for candidate in self.candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate ids must be unique")
        if len(visible_refs) != len(set(visible_refs)):
            raise ValueError(
                "candidate visible representations must be unique"
            )
        return self


class CandidatePreferenceInterfaceV2(ExternalLabelInterfaceBaseV2):
    """Future preference label over prompt-visible candidates."""

    source_kind: Literal["candidate_trajectory_preference"]
    candidate_set: CandidateSetV2
    answer: E1AnswerV2

    @model_validator(mode="after")
    def validate_preference(self) -> "CandidatePreferenceInterfaceV2":
        """Require visible stamped candidates and one valid preference."""

        visible_refs = {
            item.prompt_visible_representation_ref
            for item in self.candidate_set.candidates
        }
        if not visible_refs.issubset(set(self.evidence_refs)):
            raise ValueError(
                "candidate representations must be prompt-visible evidence"
            )
        valid = {
            item.candidate_id for item in self.candidate_set.candidates
        }
        if any(
            item.direction is None
            for item in self.candidate_set.candidates
        ):
            raise ValueError(
                "preference candidates require direction stamps"
            )
        if self.answer.payload.preferred_candidate not in valid:
            raise ValueError(
                "preferred candidate is absent from candidate set"
            )
        return self


class CandidateDirectionInterfaceV2(ExternalLabelInterfaceBaseV2):
    """Future direction label for one prompt-visible candidate."""

    source_kind: Literal["candidate_direction"]
    candidate_set: CandidateSetV2
    target_candidate_id: str = Field(min_length=1, max_length=128)
    answer: E2AnswerV2

    @model_validator(mode="after")
    def validate_candidate_direction(
        self,
    ) -> "CandidateDirectionInterfaceV2":
        """Match answer direction to the selected candidate stamp."""

        visible_refs = {
            item.prompt_visible_representation_ref
            for item in self.candidate_set.candidates
        }
        if not visible_refs.issubset(set(self.evidence_refs)):
            raise ValueError(
                "candidate representations must be prompt-visible evidence"
            )
        candidates = {
            item.candidate_id: item
            for item in self.candidate_set.candidates
        }
        try:
            selected = candidates[self.target_candidate_id]
        except KeyError as exc:
            raise ValueError(
                "target candidate is absent from candidate set"
            ) from exc
        if selected.direction is None:
            raise ValueError(
                "candidate direction interface requires a direction stamp"
            )
        if (
            self.answer.payload.direction_label
            != selected.direction.label_name
        ):
            raise ValueError(
                "candidate direction answer does not match stamp"
            )
        return self


class TrajectoryCompatibilityInterfaceV2(
    ExternalLabelInterfaceBaseV2
):
    """Future compatibility label for one prompt-visible candidate."""

    source_kind: Literal["trajectory_compatibility"]
    candidate_set: CandidateSetV2
    target_candidate_id: str = Field(min_length=1, max_length=128)
    constraint_evidence_refs: list[str] = Field(
        default_factory=list,
        max_length=32,
    )
    answer: E3AnswerV2

    @model_validator(mode="after")
    def validate_candidate(
        self,
    ) -> "TrajectoryCompatibilityInterfaceV2":
        """Require the compatibility target to be prompt-visible."""

        visible_refs = {
            item.prompt_visible_representation_ref
            for item in self.candidate_set.candidates
        }
        if not visible_refs.issubset(set(self.evidence_refs)):
            raise ValueError(
                "candidate representations must be prompt-visible evidence"
            )
        if len(set(self.constraint_evidence_refs)) != len(
            self.constraint_evidence_refs
        ):
            raise ValueError(
                "constraint evidence refs must be unique"
            )
        if not set(self.constraint_evidence_refs).issubset(
            set(self.evidence_refs)
        ):
            raise ValueError(
                "constraint refs must be prompt-visible evidence"
            )
        valid = {
            item.candidate_id for item in self.candidate_set.candidates
        }
        if self.target_candidate_id not in valid:
            raise ValueError(
                "compatibility target is absent from candidate set"
            )
        return self


class VQASourceInterfacesV2(BaseModel):
    """Hold dormant external-label interfaces without enabling them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pixel_goal_direction: PixelGoalDirectionInterfaceV2 | None = None
    waypoint_direction: WaypointDirectionInterfaceV2 | None = None
    bev_route_alignment: BEVRouteAlignmentInterfaceV2 | None = None
    cross_view_entity_binding: (
        CrossViewEntityBindingInterfaceV2 | None
    ) = None
    outside_view_direction: OutsideViewDirectionInterfaceV2 | None = None
    bev_fpv_executability: (
        BEVFPVExecutabilityInterfaceV2 | None
    ) = None
    candidate_preference: CandidatePreferenceInterfaceV2 | None = None
    candidate_direction: CandidateDirectionInterfaceV2 | None = None
    trajectory_compatibility: (
        TrajectoryCompatibilityInterfaceV2 | None
    ) = None


VQA_EXTERNAL_INTERFACE_MODEL_BY_TASK: dict[
    str,
    type[ExternalLabelInterfaceBaseV2],
] = {
    "c2_pixel_goal_to_direction": PixelGoalDirectionInterfaceV2,
    "c2_waypoint_to_direction": WaypointDirectionInterfaceV2,
    "c3_bev_route_alignment": BEVRouteAlignmentInterfaceV2,
    "c4_bev_object_anchoring": CrossViewEntityBindingInterfaceV2,
    "c5_outside_view_direction": OutsideViewDirectionInterfaceV2,
    "c6_bev_fpv_executability": BEVFPVExecutabilityInterfaceV2,
    "e1_candidate_trajectory_preference": (
        CandidatePreferenceInterfaceV2
    ),
    "e2_candidate_direction_classification": (
        CandidateDirectionInterfaceV2
    ),
    "e3_trajectory_compatibility": (
        TrajectoryCompatibilityInterfaceV2
    ),
}


class VQASourceV2(BaseModel):
    """Partition visible VQA input from private label interfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        VQA_SOURCE_V2_SCHEMA_VERSION
    ] = VQA_SOURCE_V2_SCHEMA_VERSION
    observation: NavObservationV4
    direction_supervision: DirectionSupervisionV4 | None = None
    interfaces: VQASourceInterfacesV2 = Field(
        default_factory=VQASourceInterfacesV2
    )
    warnings: list[str] = Field(default_factory=list, max_length=128)
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_fingerprint(self) -> "VQASourceV2":
        """Verify interface identity and the complete source fingerprint."""

        available_evidence = {
            entry.ref for entry in self.observation.evidence_catalog
        }
        for interface in self.interfaces.model_dump(
            mode="python"
        ).values():
            if interface is None:
                continue
            if interface["source_group_id"] != (
                self.observation.source_group_id
            ):
                raise ValueError(
                    "external interface/source group mismatch"
                )
            missing_evidence = sorted(
                set(interface["evidence_refs"]) - available_evidence
            )
            if missing_evidence:
                raise ValueError(
                    "external interface evidence is not student-visible: "
                    f"{missing_evidence}"
                )
        expected = vqa_source_fingerprint_v2(
            self.model_dump(
                mode="json",
                exclude={"source_fingerprint"},
            )
        )
        if expected != self.source_fingerprint:
            raise ValueError(
                "VQA V2 source_fingerprint does not match payload"
            )
        return self


class VQAAnnotationV2(BaseModel):
    """Store one accepted typed answer with auditable provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        VQA_ANNOTATION_V2_SCHEMA_VERSION
    ] = VQA_ANNOTATION_V2_SCHEMA_VERSION
    annotation_id: str = Field(min_length=1, max_length=400)
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    question: VQAQuestionV2
    answer: VQAAnswerV2
    label_source: VQALabelSourceV2
    source_kind: ReasonCode
    evidence_refs: list[str] = Field(min_length=1, max_length=64)
    source_field_refs: list[str] = Field(
        default_factory=list,
        max_length=32,
    )
    producer_id: str = Field(min_length=1, max_length=128)
    producer_version: str = Field(min_length=1, max_length=128)
    input_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    annotation_context_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    prompt_profile_id: str | None = Field(
        default=None,
        max_length=128,
    )
    prompt_policy_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    teacher_provider: str | None = Field(default=None, max_length=64)
    teacher_model: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_annotation(self) -> "VQAAnnotationV2":
        """Validate answer identity and strong/weak provenance boundaries."""

        validate_answer_for_question_v2(self.question, self.answer)
        task_spec = get_vqa_task_spec_v2(self.question.task_id)
        if self.label_source not in task_spec.allowed_label_sources:
            raise ValueError(
                "label_source is not allowed for the VQA task"
            )
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("annotation evidence refs must be unique")
        if len(set(self.source_field_refs)) != len(
            self.source_field_refs
        ):
            raise ValueError(
                "annotation source field refs must be unique"
            )
        prompt_values = (
            self.prompt_profile_id,
            self.prompt_policy_digest,
            self.teacher_model,
        )
        if self.label_source == "lvlm_weak":
            if any(value is None for value in prompt_values):
                raise ValueError(
                    "LVLM weak label requires prompt and model provenance"
                )
        elif any(value is not None for value in prompt_values):
            raise ValueError(
                "non-LVLM label cannot carry teacher prompt/model fields"
            )
        if self.label_source == "derived_gt" and not self.source_field_refs:
            raise ValueError(
                "derived GT requires source_field_refs"
            )
        return self


class VQAOptionalVisualInterfacesV2(BaseModel):
    """Reserve unavailable dense visual interfaces without claiming labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seghead_available: Literal[False] = False
    sam_som_weak_adapter_available: Literal[False] = False
    seghead_ref: None = None
    sam_ref: None = None
    som_ref: None = None


class VQAOptionalMetricsV2(BaseModel):
    """Keep dense metrics explicitly unavailable in the current stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dense_metrics_available: Literal[False] = False
    miou: None = None
    harmful_recall: None = None
    forbidden_hit_rate: None = None


class VQATrainingViewV2(BaseModel):
    """Cross-repository LM-only VQA training view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        VQA_TRAINING_VIEW_V2_SCHEMA_VERSION
    ] = VQA_TRAINING_VIEW_V2_SCHEMA_VERSION
    sample_id: str = Field(min_length=1, max_length=400)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, max_length=4096)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(
        min_length=1,
        max_length=64,
    )
    input_modalities: InputModalitiesV4
    question: VQAQuestionV2
    assistant_text: str = Field(min_length=1, max_length=4000)
    annotation_id: str = Field(min_length=1, max_length=400)
    label_source: VQALabelSourceV2
    source_kind: ReasonCode
    producer_id: str = Field(min_length=1, max_length=128)
    producer_version: str = Field(min_length=1, max_length=128)
    prompt_profile_id: str = Field(min_length=1, max_length=128)
    prompt_policy_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    capability_config_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    optional_visual_interfaces: VQAOptionalVisualInterfacesV2 = Field(
        default_factory=VQAOptionalVisualInterfacesV2
    )
    optional_metrics: VQAOptionalMetricsV2 = Field(
        default_factory=VQAOptionalMetricsV2
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_view(self) -> "VQATrainingViewV2":
        """Validate visible media and keep private interfaces out."""

        if len(self.images) != len(self.image_roles):
            raise ValueError(
                "VQA training images and roles must align"
            )
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("VQA training image roles must be unique")
        if self.image_roles.count("fpv.current") != 1:
            raise ValueError(
                "VQA training view requires one fpv.current"
            )
        return self


def vqa_source_fingerprint_v2(payload: dict[str, Any]) -> str:
    """Hash one JSON-compatible VQA source payload.

    Args:
        payload: Source payload without its fingerprint.

    Returns:
        Lowercase SHA-256 digest.
    """

    return _fingerprint(payload)


def annotation_context_digest_v2(
    *,
    source: VQASourceV2,
    question: VQAQuestionV2,
    producer_id: str,
    producer_version: str,
) -> str:
    """Bind an annotation to exact visible input and producer policy.

    Args:
        source: Frozen VQA source.
        question: Exact answer-independent question.
        producer_id: Label producer identity.
        producer_version: Label producer version.

    Returns:
        Lowercase SHA-256 digest.
    """

    payload = {
        "source_group_id": source.observation.source_group_id,
        "source_fingerprint": source.source_fingerprint,
        "visible_observation": source.observation.model_dump(mode="json"),
        "question": question.model_dump(mode="json"),
        "producer_id": producer_id,
        "producer_version": producer_version,
    }
    return _fingerprint(payload)


def _fingerprint(payload: dict[str, Any]) -> str:
    """Return canonical JSON SHA-256 for an internal payload."""

    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "BEVFPVExecutabilityInterfaceV2",
    "BEVRouteAlignmentInterfaceV2",
    "CandidateDirectionInterfaceV2",
    "CandidatePreferenceInterfaceV2",
    "CandidateSetV2",
    "CrossViewEntityBindingInterfaceV2",
    "OutsideViewDirectionInterfaceV2",
    "PixelGoalDirectionInterfaceV2",
    "PromptVisibleCandidateV2",
    "TrajectoryCompatibilityInterfaceV2",
    "VQAAnnotationV2",
    "VQACapabilityConfigV2",
    "VQAOptionalMetricsV2",
    "VQAOptionalVisualInterfacesV2",
    "VQASourceInterfacesV2",
    "VQASourceV2",
    "VQATaskCapabilityV2",
    "VQATrainingViewV2",
    "VQA_ANNOTATION_V2_SCHEMA_VERSION",
    "VQA_CAPABILITY_CONFIG_SCHEMA_VERSION",
    "VQA_EXTERNAL_INTERFACE_MODEL_BY_TASK",
    "VQA_SOURCE_V2_SCHEMA_VERSION",
    "VQA_TRAINING_VIEW_V2_SCHEMA_VERSION",
    "WaypointDirectionInterfaceV2",
    "annotation_context_digest_v2",
    "vqa_source_fingerprint_v2",
]
