"""Canonical exchange contracts for UrbanNav Stage II trajectory CFM.

The RouteServer boundary owns metric trajectory truth and immutable input/cache
identity.  Model tokenization, trajectory normalization, latent-query positions,
and tensors remain ms-swift implementation details.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mcp_server.contracts.reasoning import ReasonCode
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionContextV4,
    DirectionSupervisionV4,
    EvidenceCatalogEntryV4,
    InputModalitiesV4,
)
from mcp_server.contracts.stage_i_sft import (
    StageIReasoningDropoutLinkV2,
    StageIInstructionTrainingRoleV1,
    StageIModalityAvailabilityV1,
    StageIModalityProfileIdV1,
    StageIModalitySelectionV1,
    StageIModalityStateV1,
)


STAGE_II_TRAINING_CONTRACT_ID = "stage_ii_cfm_v1"
STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION = (
    "urbannav_stage_ii_context_cache_v1"
)
STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION = (
    "urbannav_stage_ii_trajectory_target_v1"
)
STAGE_II_TRAJECTORY_SOURCE_REGISTRY_SCHEMA_VERSION = (
    "urbannav_stage_ii_trajectory_source_registry_v1"
)
STAGE_II_TRAJECTORY_SOURCE_REGISTRY_ID = (
    "urbannav_stage_ii_trajectory_sources_v1"
)
STAGE_II_TRAJECTORY_SOURCE_REGISTRY_VERSION = "1.1"
STAGE_II_SPLIT_POLICY_ID = "stage_i_group_sha256_colon_v1"
STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION = (
    "urbannav_stage_ii_trajectory_qc_v1"
)
STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION = (
    "urbannav_stage_ii_cfm_training_view_v1"
)
STAGE_II_DIRECT_LATENT_VIEW_SCHEMA_VERSION = (
    "urbannav_stage_ii_direct_latent_view_v1"
)
STAGE_II_V46_DIRECT_LATENT_PROFILE_ID = "reasoning_v4_6_direct_latent_v1"
STAGE_II_V46_DIRECT_LATENT_PROFILE_VERSION = "1.0"
STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_ID = (
    "urbannav_stage_ii_direct_latent_projection_v1"
)
STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_VERSION = "1.0"
STAGE_II_V2_TRAINING_CONTRACT_ID = "stage_ii_cfm_v2"
STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE = "cached_traj_query_v1"
STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION = (
    "urbannav_stage_ii_cached_traj_query_observation_v1"
)
STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_SCHEMA_VERSION = (
    "urbannav_stage_ii_cached_traj_query_training_view_v1"
)
STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2 = "cached_traj_query_v2"
STAGE_II_CACHED_TRAJ_QUERY_BINDING_V2_SCHEMA_VERSION = (
    "urbannav_stage_ii_cached_traj_query_binding_v2"
)
STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_V2_SCHEMA_VERSION = (
    "urbannav_stage_ii_cached_traj_query_training_view_v2"
)
STAGE_II_CACHED_TRAJ_QUERY_SOURCE_VIEW_V2_SCHEMA_VERSION = (
    "urbannav_stage_ii_cached_traj_query_source_view_v2"
)
STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION = (
    "urbannav_stage_ii_reasoning_context_v2"
)
STAGE_II_CFM_TRAINING_VIEW_V2_SCHEMA_VERSION = (
    "urbannav_stage_ii_cfm_training_view_v2"
)
STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION = (
    "urbannav_stage_ii_legacy_direct_target_v1"
)
STAGE_II_LEGACY_DIRECT_PROFILE_ID = "legacy_full_cot_direct_latent_v1"
STAGE_II_LEGACY_DIRECT_PROFILE_VERSION = "1.0"
STAGE_II_PROJECTION_DROP_SCHEMA_VERSION = (
    "urbannav_stage_ii_projection_drop_v1"
)
STAGE_II_PROJECTION_POLICY_ID = "urbannav_stage_ii_safe_projection_v1"
STAGE_II_PROJECTION_POLICY_VERSION = "1.0"
STAGE_II_PROJECTION_DROP_V2_SCHEMA_VERSION = (
    "urbannav_stage_ii_projection_drop_v2"
)
STAGE_II_PROJECTION_POLICY_V2_ID = (
    "urbannav_stage_ii_reasoning_dropout_projection_v2"
)
STAGE_II_PROJECTION_POLICY_V2_VERSION = "2.0"
STAGE_II_TRAJECTORY_POLICY_ID = "urbannav_stage_ii_metric_flu_2s_16_v1"
STAGE_II_TRAJECTORY_POLICY_VERSION = "1.0"
STAGE_II_TEMPORAL_RESAMPLING_ID = "linear_time_grid_2s_16_v1"
STAGE_II_HORIZON_STEPS = 16
STAGE_II_HORIZON_SECONDS = 2.0
STAGE_II_TARGET_FORMAT = "ego_pose_xyt_flu_v1"
STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION = (
    "urbannav_stage_ii_trajectory_target_v2_1"
)
STAGE_II_TRAJECTORY_QC_V2_SCHEMA_VERSION = (
    "urbannav_stage_ii_trajectory_qc_v2_1"
)
STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION = (
    "urbannav_stage_ii_trajectory_sidecar_v2_1"
)
STAGE_II_TRAJECTORY_POLICY_V2_ID = (
    "urbannav_stage_ii_metric_flu_4s_32_speed_audit_v2_1"
)
STAGE_II_TRAJECTORY_POLICY_V2_VERSION = "2.1"
STAGE_II_TEMPORAL_RESAMPLING_V2_ID = "linear_xy_shortest_yaw_4s_32_v2"
STAGE_II_HORIZON_STEPS_V2 = 32
STAGE_II_HORIZON_SECONDS_V2 = 4.0
STAGE_II_TARGET_DT_SEC_V2 = 0.125
STAGE_II_MAX_SOURCE_GAP_SEC_V2 = 0.5
STAGE_II_MAX_SOURCE_STEP_M_V2 = 3.0
STAGE_II_SOURCE_SPEED_AUDIT_THRESHOLD_MPS_V2 = 6.0
STAGE_II_MAX_SOURCE_YAW_RATE_RAD_S_V2 = 4.0
STAGE_II_GROUNDING_SECTION_NAMES = (
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
)

StageIIContextKindV1 = Literal["reasoning_grounding", "decision"]
StageIIContextSourceV1 = Literal["ground_truth", "model_prediction"]
StageIIContextParseStatusV1 = Literal["accepted", "rejected"]
StageIIContextModeV2 = Literal["reasoning_plus_decision", "decision_only"]
StageIIConditionModeV2 = Literal[
    "traj_query_only",
    STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE,
]
StageIIReasoningStateV2 = Literal[
    "present",
    "absent_by_reasoning_dropout",
]
StageIITrajectoryQCStatusV1 = Literal["accepted", "rejected"]
StageIITrajectoryDatasetStatusV1 = Literal["enabled", "disabled"]
StageIITrajectoryMaskPolicyV1 = Literal[
    "explicit_field_v1",
    "validated_timed_rows_v1",
]
StageIITrajectoryOriginPolicyV1 = Literal[
    "require_explicit_origin_v1",
    "prepend_current_origin_v1",
]


class StageIIContextProducerV1(BaseModel):
    """Identify the exact producer of one cached reasoning context.

    Args:
        source: Oracle or frozen-model provenance.
        base_model_id: Base LVLM identity.
        checkpoint_id: Exact Stage I checkpoint identity.
        checkpoint_digest: Immutable checkpoint digest.
        adapter_id: Optional adapter identity.
        prompt_profile_id: Pass-specific prompt profile.
        prompt_digest: Exact prompt-policy digest.
        parser_contract_id: Strict parser identity.
        parser_contract_digest: Exact parser-contract digest.
        generation_config: Frozen decoding configuration.
        generation_config_digest: SHA-256 of ``generation_config``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: StageIIContextSourceV1
    base_model_id: str = Field(min_length=1, max_length=300)
    checkpoint_id: str = Field(min_length=1, max_length=500)
    checkpoint_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    adapter_id: str | None = Field(default=None, min_length=1, max_length=500)
    prompt_profile_id: str = Field(min_length=1, max_length=160)
    prompt_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    parser_contract_id: str = Field(min_length=1, max_length=160)
    parser_contract_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    generation_config: dict[str, Any] = Field(default_factory=dict)
    generation_config_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_generation_digest(self) -> "StageIIContextProducerV1":
        """Require the generation configuration to be immutable."""

        expected = stage_ii_digest_v1(self.generation_config)
        if self.generation_config_digest != expected:
            raise ValueError("Stage II generation_config_digest mismatch")
        return self


class StageIIContextCacheV1(BaseModel):
    """Store one strict Pass-R or Pass-D result for an exact input projection.

    Args:
        schema_version: Exact cache schema stamp.
        cache_id: Stable context-kind-specific cache identity.
        sample_id: Parent Stage I FullCoT sample identity.
        source_group_id: Parent source group.
        split_group_id: Leakage-safe split group.
        dataset_source: Dataset family.
        modality_profile_id: Exact projected modality combination.
        parent_source_fingerprint: Immutable parent source digest.
        projected_input_digest: Digest of every model-visible input.
        context_kind: Independent reasoning-grounding or Decision pass.
        section_names: Exact canonical section sequence.
        producer: Frozen model/prompt/parser/generation identity.
        raw_generation: Original output retained for audit only.
        canonical_text: Strict parsed text exposed to Pass T.
        parse_status: Accepted or rejected parser state.
        errors: Bounded parser error codes.
        cache_digest: SHA-256 over all other cache fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION] = (
        STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION
    )
    cache_id: str = Field(min_length=1, max_length=480)
    sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: StageIModalityProfileIdV1
    parent_source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    projected_input_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    context_kind: StageIIContextKindV1
    section_names: tuple[str, ...] = Field(min_length=1, max_length=5)
    producer: StageIIContextProducerV1
    raw_generation: str | None = Field(default=None, max_length=30000)
    canonical_text: str | None = Field(default=None, max_length=20000)
    parse_status: StageIIContextParseStatusV1
    errors: tuple[ReasonCode, ...] = Field(default_factory=tuple, max_length=16)
    cache_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_cache(self) -> "StageIIContextCacheV1":
        """Require exact section family, parse state, and immutable identity."""

        expected_sections = (
            STAGE_II_GROUNDING_SECTION_NAMES
            if self.context_kind == "reasoning_grounding"
            else ("Decision",)
        )
        if tuple(self.section_names) != tuple(expected_sections):
            raise ValueError("Stage II context section_names mismatch")
        if self.parse_status == "accepted":
            if not self.canonical_text or self.errors:
                raise ValueError("accepted Stage II context requires text and no errors")
        elif self.canonical_text is not None or not self.errors:
            raise ValueError("rejected Stage II context requires errors and no canonical text")
        if self.producer.source == "model_prediction" and self.raw_generation is None:
            raise ValueError("predicted Stage II context requires raw_generation audit")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"cache_digest"})
        )
        if self.cache_digest != expected:
            raise ValueError("Stage II context cache_digest mismatch")
        return self


class StageIIReasoningPresentV2(BaseModel):
    """Bind V2 Pass T to one accepted reasoning-grounding cache.

    The wrapper is intentionally distinct from an optional cache field.  Its
    discriminator proves that reasoning is present and accepted; a rejected or
    empty cache therefore cannot silently become a no-CoT example.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION] = (
        STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION
    )
    reasoning_state: Literal["present"] = "present"
    reasoning_context: StageIIContextCacheV1
    state_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_state(self) -> "StageIIReasoningPresentV2":
        """Require a strict accepted Pass-R cache and immutable wrapper."""

        cache = self.reasoning_context
        if cache.context_kind != "reasoning_grounding" or cache.parse_status != "accepted":
            raise ValueError("present Stage II V2 reasoning requires accepted Pass R")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"state_digest"})
        )
        if self.state_digest != expected:
            raise ValueError("Stage II V2 reasoning state_digest mismatch")
        return self


class StageIIReasoningAbsentV2(BaseModel):
    """Prove that reasoning is absent only because Stage I routed no-CoT.

    No parser result is stored here.  Absence is valid only when it is linked
    to a validated Stage I reasoning-dropout projection of an immutable
    FullCoT parent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION] = (
        STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION
    )
    reasoning_state: Literal["absent_by_reasoning_dropout"] = (
        "absent_by_reasoning_dropout"
    )
    parent_stage_i_sample_id: str = Field(min_length=1, max_length=480)
    stage_i_dropout_sample_id: str = Field(min_length=1, max_length=480)
    stage_i_dropout_projection_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: StageIModalityProfileIdV1
    parent_source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    projected_input_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    dropout_link: StageIReasoningDropoutLinkV2
    state_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_state(self) -> "StageIIReasoningAbsentV2":
        """Require exact Stage I dropout lineage and immutable absence."""

        if self.dropout_link.parent_full_cot_sample_id != self.parent_stage_i_sample_id:
            raise ValueError("Stage II V2 absence parent identity mismatch")
        expected_dropout_id = f"{self.parent_stage_i_sample_id}:reasoning_dropout"
        if self.stage_i_dropout_sample_id != expected_dropout_id:
            raise ValueError("Stage II V2 absence dropout sample identity mismatch")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"state_digest"})
        )
        if self.state_digest != expected:
            raise ValueError("Stage II V2 absence state_digest mismatch")
        return self


StageIIReasoningContextV2 = Annotated[
    Union[StageIIReasoningPresentV2, StageIIReasoningAbsentV2],
    Field(discriminator="reasoning_state"),
]
STAGE_II_REASONING_CONTEXT_ADAPTER_V2 = TypeAdapter(StageIIReasoningContextV2)


class StageIITrajectorySplitPolicyV1(BaseModel):
    """Freeze the leakage-safe group split used by Stage I and Stage II.

    Args:
        policy_id: Exact deterministic split implementation identity.
        seed: Integer seed included in the SHA-256 input.
        train: Fraction assigned to train groups.
        validation: Fraction assigned to validation groups.
        test: Fraction assigned to held-out test groups.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: Literal[STAGE_II_SPLIT_POLICY_ID] = STAGE_II_SPLIT_POLICY_ID
    seed: int
    train: float = Field(ge=0.0, le=1.0)
    validation: float = Field(ge=0.0, le=1.0)
    test: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ratios(self) -> "StageIITrajectorySplitPolicyV1":
        """Require one complete, non-overlapping split partition."""

        if not math.isclose(
            self.train + self.validation + self.test,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Stage II split ratios must sum to one")
        return self


class StageIITrajectorySourceSpecV1(BaseModel):
    """Describe one dataset-owned trajectory source without runtime guessing.

    Args:
        dataset_source: Canonical dataset family.
        status: Whether the dataset is admitted for Stage II target building.
        sample_identity_fields: Ordered row paths forming a unique parent ID.
        split_group_fields: Ordered row paths forming the leakage-safe group.
        trajectory_field: Row path containing timed trajectory coordinates.
        source_format: Exact coordinate/time row layout.
        trajectory_source_fields: Ordered equivalent provenance paths.
        allowed_trajectory_sources: Closed source-code allowlist.
        yaw_source: Deterministic yaw construction authority.
        mask_policy: Explicit mask or registry-authorized timed-row validation.
        action_mask_field: Row path used by the explicit mask policy.
        origin_policy: Require or deterministically prepend current ego origin.
        required_values: Exact row-path stamps required before adaptation.
        disabled_reason: Stable reason when this dataset is not admitted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_source: ReasonCode
    status: StageIITrajectoryDatasetStatusV1
    sample_identity_fields: tuple[str, ...] = Field(min_length=1, max_length=8)
    split_group_fields: tuple[str, ...] = Field(min_length=1, max_length=8)
    trajectory_field: str = Field(min_length=1, max_length=160)
    source_format: Literal[
        "ego_xy_t_speed_right_front_v1",
        "ego_pose_xyt_flu_timed_v1",
    ]
    trajectory_source_fields: tuple[str, ...] = Field(min_length=1, max_length=8)
    allowed_trajectory_sources: tuple[ReasonCode, ...] = Field(max_length=16)
    yaw_source: Literal["odometry", "path_tangent"]
    mask_policy: StageIITrajectoryMaskPolicyV1
    action_mask_field: str | None = Field(default=None, min_length=1, max_length=160)
    origin_policy: StageIITrajectoryOriginPolicyV1
    required_values: dict[str, Any] = Field(default_factory=dict)
    disabled_reason: ReasonCode | None = None

    @model_validator(mode="after")
    def validate_source_spec(self) -> "StageIITrajectorySourceSpecV1":
        """Require coherent admission, mask, yaw, and provenance policies."""

        if self.status == "enabled":
            if not self.allowed_trajectory_sources or self.disabled_reason is not None:
                raise ValueError("enabled Stage II source requires an allowlist and no disabled_reason")
        elif self.allowed_trajectory_sources or self.disabled_reason is None:
            raise ValueError("disabled Stage II source requires no allowlist and one disabled_reason")
        if self.mask_policy == "explicit_field_v1" and self.action_mask_field is None:
            raise ValueError("explicit Stage II mask policy requires action_mask_field")
        if self.mask_policy == "validated_timed_rows_v1" and self.action_mask_field is not None:
            raise ValueError("timed-row Stage II mask policy cannot name action_mask_field")
        expected_yaw = (
            "path_tangent"
            if self.source_format == "ego_xy_t_speed_right_front_v1"
            else "odometry"
        )
        if self.yaw_source != expected_yaw:
            raise ValueError("Stage II source format and yaw_source disagree")
        return self


class StageIITrajectorySourceRegistryV1(BaseModel):
    """Freeze all admitted and explicitly disabled Stage II dataset sources.

    Args:
        schema_version: Exact source-registry schema stamp.
        registry_id: Stable registry family identity.
        registry_version: Exact immutable instance version.
        split_policy: Leakage-safe group split contract.
        datasets: One source specification per supported dataset family.
        registry_digest: SHA-256 over every other registry field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_TRAJECTORY_SOURCE_REGISTRY_SCHEMA_VERSION] = (
        STAGE_II_TRAJECTORY_SOURCE_REGISTRY_SCHEMA_VERSION
    )
    registry_id: Literal[STAGE_II_TRAJECTORY_SOURCE_REGISTRY_ID] = (
        STAGE_II_TRAJECTORY_SOURCE_REGISTRY_ID
    )
    registry_version: Literal[STAGE_II_TRAJECTORY_SOURCE_REGISTRY_VERSION] = (
        STAGE_II_TRAJECTORY_SOURCE_REGISTRY_VERSION
    )
    split_policy: StageIITrajectorySplitPolicyV1
    datasets: tuple[StageIITrajectorySourceSpecV1, ...] = Field(
        min_length=1,
        max_length=32,
    )
    registry_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_registry(self) -> "StageIITrajectorySourceRegistryV1":
        """Require unique complete dataset coverage and an immutable digest."""

        names = [item.dataset_source for item in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError("Stage II trajectory registry contains duplicate datasets")
        required = {"sidewalk", "foxglove", "egowalk", "frodobots"}
        if set(names) != required:
            raise ValueError(
                "Stage II trajectory registry must freeze sidewalk, foxglove, "
                "egowalk, and frodobots"
            )
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"registry_digest"})
        )
        if self.registry_digest != expected:
            raise ValueError("Stage II trajectory source registry_digest mismatch")
        return self

    def dataset(self, dataset_source: str) -> StageIITrajectorySourceSpecV1:
        """Return one exact dataset source specification.

        Args:
            dataset_source: Canonical dataset family to resolve.

        Returns:
            Frozen source specification.

        Raises:
            KeyError: If the dataset is not registered.
        """

        for item in self.datasets:
            if item.dataset_source == dataset_source:
                return item
        raise KeyError(dataset_source)


class StageIITrajectoryPolicyV1(BaseModel):
    """Freeze metric target geometry and temporal resampling semantics.

    Args:
        policy_id: Canonical trajectory-policy identity.
        policy_version: Exact policy version.
        target_field: Physical target sidecar name.
        target_format: Metric cumulative FLU pose format.
        representation: Cumulative relative-pose representation.
        coordinate_frame: Current-ego FLU frame.
        axis_names: Ordered target axes.
        units: Ordered target units.
        horizon_steps: Fixed 16-step horizon.
        horizon_seconds: Fixed two-second horizon.
        target_dt_sec: Fixed 0.125-second grid interval.
        temporal_resampling_id: Deterministic resampler stamp.
        source_format: Exact supported timed source format.
        trajectory_source: Dataset-owned source code.
        yaw_source: Odometry or deterministic path tangent.
        normalized: Always false at the canonical exchange boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: Literal[STAGE_II_TRAJECTORY_POLICY_ID] = (
        STAGE_II_TRAJECTORY_POLICY_ID
    )
    policy_version: Literal[STAGE_II_TRAJECTORY_POLICY_VERSION] = (
        STAGE_II_TRAJECTORY_POLICY_VERSION
    )
    target_field: Literal["trajectory_ego_t"] = "trajectory_ego_t"
    target_format: Literal[STAGE_II_TARGET_FORMAT] = STAGE_II_TARGET_FORMAT
    representation: Literal["cumulative_relative_pose"] = (
        "cumulative_relative_pose"
    )
    coordinate_frame: Literal["current_ego_flu"] = "current_ego_flu"
    axis_names: tuple[
        Literal["x_forward_m"],
        Literal["y_left_m"],
        Literal["yaw_rel_ccw_rad"],
    ] = ("x_forward_m", "y_left_m", "yaw_rel_ccw_rad")
    units: tuple[Literal["meter"], Literal["meter"], Literal["radian"]] = (
        "meter",
        "meter",
        "radian",
    )
    horizon_steps: Literal[STAGE_II_HORIZON_STEPS] = STAGE_II_HORIZON_STEPS
    horizon_seconds: Literal[STAGE_II_HORIZON_SECONDS] = STAGE_II_HORIZON_SECONDS
    target_dt_sec: Literal[0.125] = 0.125
    temporal_resampling_id: Literal[STAGE_II_TEMPORAL_RESAMPLING_ID] = (
        STAGE_II_TEMPORAL_RESAMPLING_ID
    )
    source_format: Literal[
        "ego_xy_t_speed_right_front_v1",
        "ego_pose_xyt_flu_timed_v1",
    ]
    trajectory_source: ReasonCode
    yaw_source: Literal["odometry", "path_tangent"]
    normalized: Literal[False] = False


class StageIITrajectoryTargetV1(BaseModel):
    """Carry one metric FLU trajectory and its prefix-contiguous valid mask.

    Args:
        schema_version: Exact target schema stamp.
        trajectory_ego_t: Metric cumulative FLU poses shaped ``[16,3]``.
        action_mask: Non-empty valid prefix followed by optional padding.
        valid_step_count: Number of true mask entries.
        policy: Frozen source/frame/time contract.
        source_target_digest: Digest of raw source rows and declarations.
        target_digest: SHA-256 over all other target fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION] = (
        STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION
    )
    trajectory_ego_t: list[list[float]] = Field(
        min_length=STAGE_II_HORIZON_STEPS,
        max_length=STAGE_II_HORIZON_STEPS,
    )
    action_mask: list[bool] = Field(
        min_length=STAGE_II_HORIZON_STEPS,
        max_length=STAGE_II_HORIZON_STEPS,
    )
    valid_step_count: int = Field(ge=1, le=STAGE_II_HORIZON_STEPS)
    policy: StageIITrajectoryPolicyV1
    source_target_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    target_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_target(self) -> "StageIITrajectoryTargetV1":
        """Reject malformed coordinates, non-prefix masks, and repeated padding."""

        seen_padding = False
        valid_count = 0
        for index, (row, valid) in enumerate(
            zip(self.trajectory_ego_t, self.action_mask)
        ):
            if len(row) != 3 or not all(math.isfinite(float(item)) for item in row):
                raise ValueError(f"Stage II trajectory row {index} must be finite [3]")
            if seen_padding and valid:
                raise ValueError("Stage II action_mask must be prefix-contiguous")
            if valid:
                valid_count += 1
            else:
                seen_padding = True
                if any(abs(float(item)) > 1e-12 for item in row):
                    raise ValueError("Stage II padded trajectory rows must be zeros")
        if valid_count != self.valid_step_count:
            raise ValueError("Stage II valid_step_count mismatch")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"target_digest"})
        )
        if self.target_digest != expected:
            raise ValueError("Stage II target_digest mismatch")
        return self


class StageIILegacyDirectTargetV1(BaseModel):
    """Bind one legacy Stage I input to canonical metric Stage II truth.

    This annotation-only exchange record intentionally contains no legacy
    assistant answer and no Pass-R/Pass-D cache.  ``legacy_visible_input_digest``
    covers only the original system/user messages and ordered images; ms-swift
    must reproduce that digest before appending isolated ``[TRAJ_i]`` queries.

    Args:
        schema_version: Exact legacy-direct target schema stamp.
        profile_id: Independent direct-latent profile identity.
        profile_version: Exact profile version.
        sample_id: Unique Stage II row identity.
        legacy_parent_id: Parent legacy Stage I sample identity.
        split: Train or validation; hidden test is forbidden.
        source_group_id: Canonical dataset-source identity.
        split_group_id: Original leakage-safe legacy group identity.
        dataset_source: Canonical dataset family.
        legacy_source_name: Original legacy experiment source name.
        legacy_source_line_no: One-based row in the frozen combined manifest.
        legacy_split_row_digest: Digest of the exact legacy split row.
        legacy_source_row_digest: Digest of the exact combined source row.
        legacy_visible_input_digest: Digest of original visible messages/images.
        trajectory_target: Canonical metric private CFM target.
        trajectory_qc_digest: Accepted deterministic trajectory QC identity.
        record_digest: SHA-256 over every other field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION] = (
        STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION
    )
    profile_id: Literal[STAGE_II_LEGACY_DIRECT_PROFILE_ID] = (
        STAGE_II_LEGACY_DIRECT_PROFILE_ID
    )
    profile_version: Literal[STAGE_II_LEGACY_DIRECT_PROFILE_VERSION] = (
        STAGE_II_LEGACY_DIRECT_PROFILE_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=480)
    legacy_parent_id: str = Field(min_length=1, max_length=480)
    split: Literal["train", "validation"]
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    legacy_source_name: str = Field(min_length=1, max_length=160)
    legacy_source_line_no: int = Field(ge=1)
    legacy_split_row_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    legacy_source_row_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    legacy_visible_input_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    trajectory_target: StageIITrajectoryTargetV1
    trajectory_qc_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    record_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_record(self) -> "StageIILegacyDirectTargetV1":
        """Require stable parent identity and immutable record content."""

        if self.sample_id != f"stage-ii-legacy-direct:{self.legacy_parent_id}":
            raise ValueError("Stage II legacy-direct sample_id mismatch")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"record_digest"})
        )
        if self.record_digest != expected:
            raise ValueError("Stage II legacy-direct record_digest mismatch")
        return self


class StageIITrajectoryQCReportV1(BaseModel):
    """Record deterministic target construction acceptance or rejection.

    Args:
        schema_version: Exact QC schema stamp.
        sample_id: Audited parent identity.
        dataset_source: Dataset family.
        trajectory_source: Declared dataset trajectory source.
        source_format: Declared timed-row format.
        status: Accepted or rejected construction state.
        reason_codes: Stable bounded QC outcomes.
        raw_step_count: Number of supplied source rows.
        source_valid_step_count: Valid source-prefix length.
        target_valid_step_count: Valid fixed-grid prefix length.
        qc_digest: SHA-256 over all other QC fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION] = (
        STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=480)
    dataset_source: str = Field(min_length=1, max_length=128)
    trajectory_source: str = Field(min_length=1, max_length=160)
    source_format: str = Field(min_length=1, max_length=160)
    status: StageIITrajectoryQCStatusV1
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1, max_length=16)
    raw_step_count: int = Field(ge=0)
    source_valid_step_count: int = Field(ge=0)
    target_valid_step_count: int = Field(ge=0, le=STAGE_II_HORIZON_STEPS)
    qc_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_qc(self) -> "StageIITrajectoryQCReportV1":
        """Require status/count coherence and immutable audit content."""

        if self.status == "accepted" and self.target_valid_step_count < 1:
            raise ValueError("accepted Stage II QC requires at least one target step")
        if self.status == "rejected" and self.target_valid_step_count != 0:
            raise ValueError("rejected Stage II QC cannot claim target steps")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"qc_digest"})
        )
        if self.qc_digest != expected:
            raise ValueError("Stage II qc_digest mismatch")
        return self


class StageIITrajectoryPolicyV2(BaseModel):
    """Freeze the strict four-second metric trajectory data policy.

    V2 is a data-only contract for the future decoupled latent-plan/CFM model.
    It intentionally does not alter the executable V1 training view or its
    16-query head.  Every accepted row contains the complete 4-second horizon;
    suffix padding is forbidden. Source speed remains an audited statistic;
    exceeding 6 m/s does not reject a row because speed alone cannot
    distinguish real fast motion from source-specific sampling behavior.

    Args:
        source_adapter_id: Versioned raw-pose adapter used for this dataset.
        trajectory_source: Dataset-owned odometry or pose provenance.
        raw_timestamp_unit: Unit of the raw pose clock before conversion.
        yaw_offset_deg: Fixed odom-to-base rotation used only for XY projection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: Literal[STAGE_II_TRAJECTORY_POLICY_V2_ID] = (
        STAGE_II_TRAJECTORY_POLICY_V2_ID
    )
    policy_version: Literal[STAGE_II_TRAJECTORY_POLICY_V2_VERSION] = (
        STAGE_II_TRAJECTORY_POLICY_V2_VERSION
    )
    target_field: Literal["trajectory_ego_t"] = "trajectory_ego_t"
    target_format: Literal[STAGE_II_TARGET_FORMAT] = STAGE_II_TARGET_FORMAT
    representation: Literal["cumulative_relative_pose"] = (
        "cumulative_relative_pose"
    )
    coordinate_frame: Literal["current_ego_flu"] = "current_ego_flu"
    axis_names: tuple[
        Literal["x_forward_m"],
        Literal["y_left_m"],
        Literal["yaw_rel_ccw_rad"],
    ] = ("x_forward_m", "y_left_m", "yaw_rel_ccw_rad")
    units: tuple[Literal["meter"], Literal["meter"], Literal["radian"]] = (
        "meter",
        "meter",
        "radian",
    )
    horizon_steps: Literal[STAGE_II_HORIZON_STEPS_V2] = (
        STAGE_II_HORIZON_STEPS_V2
    )
    horizon_seconds: Literal[STAGE_II_HORIZON_SECONDS_V2] = (
        STAGE_II_HORIZON_SECONDS_V2
    )
    target_dt_sec: Literal[STAGE_II_TARGET_DT_SEC_V2] = (
        STAGE_II_TARGET_DT_SEC_V2
    )
    temporal_resampling_id: Literal[STAGE_II_TEMPORAL_RESAMPLING_V2_ID] = (
        STAGE_II_TEMPORAL_RESAMPLING_V2_ID
    )
    source_format: Literal["ego_pose_xyt_flu_timed_v1"] = (
        "ego_pose_xyt_flu_timed_v1"
    )
    source_adapter_id: ReasonCode
    trajectory_source: ReasonCode
    raw_timestamp_unit: Literal["nanosecond", "millisecond"]
    yaw_source: Literal["odometry"] = "odometry"
    yaw_offset_deg: float
    full_horizon_required: Literal[True] = True
    max_source_gap_sec: Literal[STAGE_II_MAX_SOURCE_GAP_SEC_V2] = (
        STAGE_II_MAX_SOURCE_GAP_SEC_V2
    )
    max_source_step_m: Literal[STAGE_II_MAX_SOURCE_STEP_M_V2] = (
        STAGE_II_MAX_SOURCE_STEP_M_V2
    )
    source_speed_qc_mode: Literal["audit_only"] = "audit_only"
    source_speed_audit_threshold_mps: Literal[
        STAGE_II_SOURCE_SPEED_AUDIT_THRESHOLD_MPS_V2
    ] = (
        STAGE_II_SOURCE_SPEED_AUDIT_THRESHOLD_MPS_V2
    )
    max_source_yaw_rate_rad_s: Literal[
        STAGE_II_MAX_SOURCE_YAW_RATE_RAD_S_V2
    ] = STAGE_II_MAX_SOURCE_YAW_RATE_RAD_S_V2
    normalized: Literal[False] = False

    @model_validator(mode="after")
    def validate_policy(self) -> "StageIITrajectoryPolicyV2":
        """Require a finite odom-to-base rotation."""

        if not math.isfinite(float(self.yaw_offset_deg)):
            raise ValueError("Stage II V2 yaw_offset_deg must be finite")
        return self


class StageIITrajectoryTargetV2(BaseModel):
    """Carry one complete 4-second, 32-step metric FLU target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION] = (
        STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION
    )
    trajectory_ego_t: list[list[float]] = Field(
        min_length=STAGE_II_HORIZON_STEPS_V2,
        max_length=STAGE_II_HORIZON_STEPS_V2,
    )
    action_mask: list[bool] = Field(
        min_length=STAGE_II_HORIZON_STEPS_V2,
        max_length=STAGE_II_HORIZON_STEPS_V2,
    )
    valid_step_count: Literal[STAGE_II_HORIZON_STEPS_V2] = (
        STAGE_II_HORIZON_STEPS_V2
    )
    policy: StageIITrajectoryPolicyV2
    source_target_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    target_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_target(self) -> "StageIITrajectoryTargetV2":
        """Reject padding, non-finite metric poses, and digest drift."""

        if not all(self.action_mask):
            raise ValueError("Stage II V2 requires all 32 target steps")
        for index, row in enumerate(self.trajectory_ego_t):
            if len(row) != 3 or not all(math.isfinite(float(item)) for item in row):
                raise ValueError(
                    f"Stage II V2 trajectory row {index} must be finite [3]"
                )
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"target_digest"})
        )
        if self.target_digest != expected:
            raise ValueError("Stage II V2 target_digest mismatch")
        return self


class StageIITrajectoryQCReportV2(BaseModel):
    """Record raw-pose, temporal-coverage, and kinematic V2 QC."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_TRAJECTORY_QC_V2_SCHEMA_VERSION] = (
        STAGE_II_TRAJECTORY_QC_V2_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=480)
    dataset_source: str = Field(min_length=1, max_length=128)
    source_adapter_id: ReasonCode
    trajectory_source: ReasonCode
    clip_id: str = Field(min_length=1, max_length=300)
    frame_idx: int = Field(ge=0)
    status: StageIITrajectoryQCStatusV1
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1, max_length=16)
    raw_pose_count: int = Field(ge=0)
    source_valid_step_count: int = Field(ge=0)
    source_horizon_seconds: float | None = Field(default=None, ge=0.0)
    max_source_gap_sec: float | None = Field(default=None, ge=0.0)
    max_source_step_m: float | None = Field(default=None, ge=0.0)
    max_source_speed_mps: float | None = Field(default=None, ge=0.0)
    max_source_yaw_rate_rad_s: float | None = Field(default=None, ge=0.0)
    target_valid_step_count: int = Field(ge=0, le=STAGE_II_HORIZON_STEPS_V2)
    source_evidence_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    qc_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_qc(self) -> "StageIITrajectoryQCReportV2":
        """Require full-horizon acceptance and immutable audit content."""

        if self.status == "accepted":
            if self.target_valid_step_count != STAGE_II_HORIZON_STEPS_V2:
                raise ValueError("accepted Stage II V2 QC requires 32 target steps")
            if self.source_evidence_digest is None:
                raise ValueError("accepted Stage II V2 QC requires source evidence")
        elif self.target_valid_step_count != 0:
            raise ValueError("rejected Stage II V2 QC cannot claim target steps")
        for value in (
            self.source_horizon_seconds,
            self.max_source_gap_sec,
            self.max_source_step_m,
            self.max_source_speed_mps,
            self.max_source_yaw_rate_rad_s,
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("Stage II V2 QC statistics must be finite")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"qc_digest"})
        )
        if self.qc_digest != expected:
            raise ValueError("Stage II V2 qc_digest mismatch")
        return self


class StageIITrajectorySidecarV2(BaseModel):
    """Bind one accepted source parent to a portable V2 trajectory target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION] = (
        STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=480)
    split_group_id: str = Field(min_length=1, max_length=480)
    dataset_source: str = Field(min_length=1, max_length=128)
    split: Literal["train", "validation", "test"]
    source_manifest_name: str = Field(min_length=1, max_length=300)
    source_manifest_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_line: int = Field(ge=1)
    trajectory_target: StageIITrajectoryTargetV2
    trajectory_qc: StageIITrajectoryQCReportV2
    sidecar_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_sidecar(self) -> "StageIITrajectorySidecarV2":
        """Require one accepted target/QC identity and immutable sidecar."""

        if self.sample_id != self.source_group_id:
            raise ValueError("Stage II V2 sidecar sample/source identity mismatch")
        if self.trajectory_qc.sample_id != self.sample_id:
            raise ValueError("Stage II V2 sidecar QC identity mismatch")
        if self.trajectory_qc.dataset_source != self.dataset_source:
            raise ValueError("Stage II V2 sidecar dataset identity mismatch")
        if self.trajectory_qc.status != "accepted":
            raise ValueError("Stage II V2 sidecar requires accepted QC")
        if (
            self.trajectory_qc.source_evidence_digest
            != self.trajectory_target.source_target_digest
        ):
            raise ValueError("Stage II V2 source evidence digest mismatch")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"sidecar_digest"})
        )
        if self.sidecar_digest != expected:
            raise ValueError("Stage II V2 sidecar_digest mismatch")
        return self


class StageIILossMaskV1(BaseModel):
    """Route one trajectory row exclusively to the System1 CFM objective.

    Args:
        lm: Always false because Pass T has no assistant text.
        system1_action: Always true for the masked CFM objective.
        trajectory: Always false for the legacy pixel trajectory head.
        segmentation: Always false for the segmentation head.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lm: Literal[False] = False
    system1_action: Literal[True] = True
    trajectory: Literal[False] = False
    segmentation: Literal[False] = False


class StageIICFMTrainingViewV1(BaseModel):
    """Exchange one causal Pass-T row without token or normalized targets.

    Args:
        schema_version: Exact exchange-view schema stamp.
        training_contract: Exact Stage II training profile.
        sample_id: Unique Stage II row identity.
        source_group_id: Parent source group.
        split_group_id: Leakage-safe split group.
        dataset_source: Dataset family.
        modality_profile_id: Exact student-visible modality profile.
        native_availability: Native optional-modality availability.
        training_selection: Modalities retained by this projection.
        modality_state: Native/dropout state for every modality.
        instruction_training_role: Frozen instruction provenance.
        images: Ordered model-visible media references.
        image_roles: One semantic role per image.
        instruction: Optional model-visible instruction.
        model_visible_context: Additional safe visible context.
        evidence_catalog: Prompt-visible evidence inventory.
        input_modalities: Canonical modality flags.
        reasoning_context: Accepted independent Pass-R cache.
        decision_context: Accepted independent Pass-D cache.
        trajectory_target: Private metric CFM truth.
        direction_context: Public action vocabulary only.
        private_direction_supervision: Private sampler/evaluator stamp.
        parent_stage_i_sample_id: Formal FullCoT parent identity.
        parent_source_fingerprint: Immutable source fingerprint.
        parent_audit_id: Resolved action-reconciliation audit.
        formal_training_eligible: Always true for emitted views.
        review_status: Completed human-review state.
        loss_mask: CFM-only physical loss route.
        projection_policy_id: Exact projection policy.
        projection_policy_version: Exact projection-policy version.
        view_digest: SHA-256 over all other view fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION] = (
        STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION
    )
    training_contract: Literal[STAGE_II_TRAINING_CONTRACT_ID] = (
        STAGE_II_TRAINING_CONTRACT_ID
    )
    sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: StageIModalityProfileIdV1
    native_availability: StageIModalityAvailabilityV1
    training_selection: StageIModalitySelectionV1
    modality_state: StageIModalityStateV1
    instruction_training_role: StageIInstructionTrainingRoleV1
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, min_length=1, max_length=4096)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(min_length=1, max_length=64)
    input_modalities: InputModalitiesV4
    reasoning_context: StageIIContextCacheV1
    decision_context: StageIIContextCacheV1
    trajectory_target: StageIITrajectoryTargetV1
    direction_context: DirectionContextV4
    private_direction_supervision: DirectionSupervisionV4
    parent_stage_i_sample_id: str = Field(min_length=1, max_length=480)
    parent_source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    parent_audit_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    formal_training_eligible: Literal[True] = True
    review_status: Literal["not_required", "resolved"]
    loss_mask: StageIILossMaskV1 = Field(default_factory=StageIILossMaskV1)
    projection_policy_id: Literal[STAGE_II_PROJECTION_POLICY_ID] = (
        STAGE_II_PROJECTION_POLICY_ID
    )
    projection_policy_version: Literal[STAGE_II_PROJECTION_POLICY_VERSION] = (
        STAGE_II_PROJECTION_POLICY_VERSION
    )
    view_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_view(self) -> "StageIICFMTrainingViewV1":
        """Require exact input/cache identity and keep private targets non-textual."""

        if len(self.images) != len(self.image_roles):
            raise ValueError("Stage II images and image_roles must align")
        if self.image_roles.count("fpv.current") != 1:
            raise ValueError("Stage II requires exactly one fpv.current")
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("Stage II image_roles must be unique")
        for cache, kind in (
            (self.reasoning_context, "reasoning_grounding"),
            (self.decision_context, "decision"),
        ):
            if cache.parse_status != "accepted" or cache.context_kind != kind:
                raise ValueError(f"Stage II requires accepted {kind} cache")
            identities = (
                cache.sample_id == self.parent_stage_i_sample_id,
                cache.source_group_id == self.source_group_id,
                cache.split_group_id == self.split_group_id,
                cache.dataset_source == self.dataset_source,
                cache.modality_profile_id == self.modality_profile_id,
                cache.parent_source_fingerprint == self.parent_source_fingerprint,
            )
            if not all(identities):
                raise ValueError(f"Stage II {kind} cache identity mismatch")
        if self.reasoning_context.producer.source != self.decision_context.producer.source:
            raise ValueError("Stage II v1 forbids mixed oracle/predicted context sources")
        producer_identity_fields = (
            "base_model_id",
            "checkpoint_id",
            "checkpoint_digest",
            "adapter_id",
        )
        if any(
            getattr(self.reasoning_context.producer, name)
            != getattr(self.decision_context.producer, name)
            for name in producer_identity_fields
        ):
            raise ValueError("Stage II Pass R/D producer checkpoint identity mismatch")
        expected_parsers = {
            "reasoning_grounding": "reasoning_v4_6_full_target_strict_v1",
            "decision": "reasoning_v4_6_decision_section_strict_v1",
        }
        for cache, kind in (
            (self.reasoning_context, "reasoning_grounding"),
            (self.decision_context, "decision"),
        ):
            if cache.producer.parser_contract_id != expected_parsers[kind]:
                raise ValueError(f"Stage II {kind} parser contract mismatch")
        if (
            self.private_direction_supervision.profile_id
            != self.direction_context.action_profile_id
            or self.private_direction_supervision.profile_version
            != self.direction_context.action_profile_version
        ):
            raise ValueError("Stage II private direction/profile mismatch")
        forbidden = _find_private_prompt_keys(self.model_visible_context)
        if forbidden:
            raise ValueError(f"Stage II prompt-visible private keys: {forbidden}")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"view_digest"})
        )
        if self.view_digest != expected:
            raise ValueError("Stage II view_digest mismatch")
        return self


class StageIICFMDirectLatentViewV1(BaseModel):
    """Exchange one current-V4 direct-latent Pass-T row.

    This profile deliberately contains no Pass-R or Pass-D cache.  It is the
    direct-latent baseline for measuring whether the frozen Stage-I visual and
    language representation alone can condition the isolated trajectory
    queries.  The metric trajectory and strong direction remain private
    sidecars and never become prompt text.

    Args:
        schema_version: Exact direct-latent exchange schema stamp.
        training_contract: Shared CFM-V1 math contract.
        direct_latent_profile_id: Current-V4 cacheless profile identity.
        direct_latent_profile_version: Exact direct-latent profile version.
        sample_id: Unique direct-latent Stage-II row identity.
        source_group_id: Parent source group.
        split_group_id: Leakage-safe split group.
        dataset_source: Dataset family.
        modality_profile_id: Exact student-visible modality profile.
        native_availability: Native optional-modality availability.
        training_selection: Modalities retained by this projection.
        modality_state: Native/dropout state for every modality.
        instruction_training_role: Frozen instruction provenance.
        images: Ordered model-visible media references.
        image_roles: One semantic role per image.
        instruction: Optional model-visible instruction.
        model_visible_context: Additional safe visible context.
        evidence_catalog: Prompt-visible evidence inventory.
        input_modalities: Canonical modality flags.
        projected_input_digest: Digest of the exact visible observation.
        trajectory_target: Private metric CFM truth.
        direction_context: Public action vocabulary only.
        private_direction_supervision: Private sampler/evaluator stamp.
        parent_stage_i_sample_id: Formal V4 FullCoT parent identity.
        parent_source_fingerprint: Immutable parent source fingerprint.
        parent_audit_id: Resolved action-reconciliation audit.
        formal_training_eligible: Formal admission marker.
        review_status: Completed human-review state.
        loss_mask: CFM-only loss route.
        projection_policy_id: Direct-latent projection policy identity.
        projection_policy_version: Direct-latent projection policy version.
        view_digest: SHA-256 over all other view fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_DIRECT_LATENT_VIEW_SCHEMA_VERSION] = (
        STAGE_II_DIRECT_LATENT_VIEW_SCHEMA_VERSION
    )
    training_contract: Literal[STAGE_II_TRAINING_CONTRACT_ID] = (
        STAGE_II_TRAINING_CONTRACT_ID
    )
    direct_latent_profile_id: Literal[STAGE_II_V46_DIRECT_LATENT_PROFILE_ID] = (
        STAGE_II_V46_DIRECT_LATENT_PROFILE_ID
    )
    direct_latent_profile_version: Literal[
        STAGE_II_V46_DIRECT_LATENT_PROFILE_VERSION
    ] = STAGE_II_V46_DIRECT_LATENT_PROFILE_VERSION
    sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: StageIModalityProfileIdV1
    native_availability: StageIModalityAvailabilityV1
    training_selection: StageIModalitySelectionV1
    modality_state: StageIModalityStateV1
    instruction_training_role: StageIInstructionTrainingRoleV1
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, min_length=1, max_length=4096)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(min_length=1, max_length=64)
    input_modalities: InputModalitiesV4
    projected_input_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    trajectory_target: StageIITrajectoryTargetV1
    direction_context: DirectionContextV4
    private_direction_supervision: DirectionSupervisionV4
    parent_stage_i_sample_id: str = Field(min_length=1, max_length=480)
    parent_source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    parent_audit_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    formal_training_eligible: Literal[True] = True
    review_status: Literal["not_required", "resolved"]
    loss_mask: StageIILossMaskV1 = Field(default_factory=StageIILossMaskV1)
    projection_policy_id: Literal[STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_ID] = (
        STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_ID
    )
    projection_policy_version: Literal[
        STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_VERSION
    ] = STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_VERSION
    view_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_view(self) -> "StageIICFMDirectLatentViewV1":
        """Require a cacheless visible input and stable private sidecars."""

        if len(self.images) != len(self.image_roles):
            raise ValueError("Stage II direct-latent images and image_roles must align")
        if self.image_roles.count("fpv.current") != 1:
            raise ValueError("Stage II direct-latent requires exactly one fpv.current")
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("Stage II direct-latent image_roles must be unique")
        if self.sample_id != f"{self.parent_stage_i_sample_id}:stage_ii_direct_latent":
            raise ValueError("Stage II direct-latent sample identity mismatch")
        if (
            self.private_direction_supervision.profile_id
            != self.direction_context.action_profile_id
            or self.private_direction_supervision.profile_version
            != self.direction_context.action_profile_version
        ):
            raise ValueError("Stage II direct-latent private direction/profile mismatch")
        forbidden = _find_private_prompt_keys(self.model_visible_context)
        if forbidden:
            raise ValueError(f"Stage II direct-latent private visible keys: {forbidden}")
        expected_input_digest = stage_ii_projected_input_digest_v1(
            images=self.images,
            image_roles=self.image_roles,
            instruction=self.instruction,
            model_visible_context=self.model_visible_context,
            evidence_catalog=[
                item.model_dump(mode="json", exclude_none=True)
                for item in self.evidence_catalog
            ],
            input_modalities=self.input_modalities.model_dump(mode="json"),
        )
        if self.projected_input_digest != expected_input_digest:
            raise ValueError("Stage II direct-latent projected_input_digest mismatch")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"view_digest"})
        )
        if self.view_digest != expected:
            raise ValueError("Stage II direct-latent view_digest mismatch")
        return self


class StageIICachedTrajQueryBindingV1(BaseModel):
    """Bind an existing ``[TRAJ_i]`` cache to a later action observation.

    This is a timing contract only.  The query tensor and fast visual memory
    remain ms-swift runtime tensors; no new semantic token is introduced.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_mode: Literal[STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE] = (
        STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE
    )
    query_source_timestamp_sec: float
    action_observation_timestamp_sec: float
    query_age_sec: float
    modality_profile_id: StageIModalityProfileIdV1

    @model_validator(mode="after")
    def validate_causal_timestamps(self) -> "StageIICachedTrajQueryBindingV1":
        """Require a causal query and an internally consistent age."""

        values = (
            self.query_source_timestamp_sec,
            self.action_observation_timestamp_sec,
            self.query_age_sec,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("cached trajectory-query timestamps must be finite")
        if self.query_source_timestamp_sec > self.action_observation_timestamp_sec:
            raise ValueError("cached trajectory query cannot come from the future")
        expected_age = (
            self.action_observation_timestamp_sec
            - self.query_source_timestamp_sec
        )
        if abs(self.query_age_sec - expected_age) > 1e-4:
            raise ValueError("cached trajectory-query age does not match timestamps")
        if self.query_age_sec < 0.0:
            raise ValueError("cached trajectory-query age must be non-negative")
        return self


class StageIICFMTrainingViewV2(BaseModel):
    """Exchange one four-second CFM row under an explicit reasoning route.

    V2 keeps the observation and accepted standalone Decision visible on both
    routes.  The reasoning field is a discriminated union: either an accepted
    Pass-R cache or a cryptographically linked Stage I dropout absence.  Query
    count is deliberately not part of this data view because ms-swift owns the
    independent ``Q`` profile while the target remains fixed at ``T=32``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_CFM_TRAINING_VIEW_V2_SCHEMA_VERSION] = (
        STAGE_II_CFM_TRAINING_VIEW_V2_SCHEMA_VERSION
    )
    training_contract: Literal[STAGE_II_V2_TRAINING_CONTRACT_ID] = (
        STAGE_II_V2_TRAINING_CONTRACT_ID
    )
    sample_id: str = Field(min_length=1, max_length=560)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: StageIModalityProfileIdV1
    native_availability: StageIModalityAvailabilityV1
    training_selection: StageIModalitySelectionV1
    modality_state: StageIModalityStateV1
    instruction_training_role: StageIInstructionTrainingRoleV1
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, min_length=1, max_length=4096)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(min_length=1, max_length=64)
    input_modalities: InputModalitiesV4
    projected_input_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    context_mode: StageIIContextModeV2
    reasoning: StageIIReasoningContextV2
    decision_context: StageIIContextCacheV1
    pass_t_input_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    system1_condition_mode: StageIIConditionModeV2 = "traj_query_only"
    cached_traj_query: StageIICachedTrajQueryBindingV1 | None = None
    trajectory_target: StageIITrajectoryTargetV2
    direction_context: DirectionContextV4
    private_direction_supervision: DirectionSupervisionV4
    parent_stage_i_sample_id: str = Field(min_length=1, max_length=480)
    stage_i_route_sample_id: str = Field(min_length=1, max_length=480)
    stage_i_route_task_type: Literal["full_cot", "reasoning_dropout"]
    stage_i_route_projection_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    parent_source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    parent_audit_id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    formal_training_eligible: Literal[True] = True
    review_status: Literal["not_required", "resolved"]
    loss_mask: StageIILossMaskV1 = Field(default_factory=StageIILossMaskV1)
    projection_policy_id: Literal[STAGE_II_PROJECTION_POLICY_V2_ID] = (
        STAGE_II_PROJECTION_POLICY_V2_ID
    )
    projection_policy_version: Literal[STAGE_II_PROJECTION_POLICY_V2_VERSION] = (
        STAGE_II_PROJECTION_POLICY_V2_VERSION
    )
    view_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_view(self) -> "StageIICFMTrainingViewV2":
        """Close route, cache, identity, target, and prompt-visible invariants."""

        if self.system1_condition_mode == STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE:
            if self.cached_traj_query is None:
                raise ValueError(
                    "cached_traj_query_v1 requires a timing and modality binding"
                )
            if self.cached_traj_query.modality_profile_id != self.modality_profile_id:
                raise ValueError("cached trajectory-query modality profile mismatch")
        elif self.cached_traj_query is not None:
            raise ValueError(
                "cached_traj_query is only valid with cached_traj_query_v1"
            )
        if len(self.images) != len(self.image_roles):
            raise ValueError("Stage II V2 images and image_roles must align")
        if self.image_roles.count("fpv.current") != 1:
            raise ValueError("Stage II V2 requires exactly one fpv.current")
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("Stage II V2 image_roles must be unique")
        expected_route = {
            "reasoning_plus_decision": ("present", "full_cot"),
            "decision_only": (
                "absent_by_reasoning_dropout",
                "reasoning_dropout",
            ),
        }[self.context_mode]
        actual_route = (self.reasoning.reasoning_state, self.stage_i_route_task_type)
        if actual_route != expected_route:
            raise ValueError("Stage II V2 context mode and reasoning route disagree")
        expected_sample_id = (
            f"{self.parent_stage_i_sample_id}:stage_ii_cfm_v2:{self.context_mode}"
        )
        if self.sample_id != expected_sample_id:
            raise ValueError("Stage II V2 sample_id does not encode the context route")
        if self.stage_i_route_task_type == "full_cot":
            if self.stage_i_route_sample_id != self.parent_stage_i_sample_id:
                raise ValueError("Stage II V2 FullCoT route identity mismatch")
        elif not isinstance(self.reasoning, StageIIReasoningAbsentV2):
            raise ValueError("Stage II V2 dropout route requires an absence record")
        self._validate_cache_identity(self.decision_context, "decision")
        if isinstance(self.reasoning, StageIIReasoningPresentV2):
            reasoning_cache = self.reasoning.reasoning_context
            self._validate_cache_identity(reasoning_cache, "reasoning_grounding")
            if reasoning_cache.producer.source != self.decision_context.producer.source:
                raise ValueError("Stage II V2 forbids mixed R/D context sources")
            producer_identity_fields = (
                "base_model_id",
                "checkpoint_id",
                "checkpoint_digest",
                "adapter_id",
            )
            if any(
                getattr(reasoning_cache.producer, name)
                != getattr(self.decision_context.producer, name)
                for name in producer_identity_fields
            ):
                raise ValueError("Stage II V2 Pass R/D producer identity mismatch")
        else:
            absence = self.reasoning
            absence_identity = (
                absence.parent_stage_i_sample_id == self.parent_stage_i_sample_id,
                absence.stage_i_dropout_sample_id == self.stage_i_route_sample_id,
                absence.stage_i_dropout_projection_digest
                == self.stage_i_route_projection_digest,
                absence.source_group_id == self.source_group_id,
                absence.split_group_id == self.split_group_id,
                absence.dataset_source == self.dataset_source,
                absence.modality_profile_id == self.modality_profile_id,
                absence.parent_source_fingerprint
                == self.parent_source_fingerprint,
                absence.projected_input_digest == self.projected_input_digest,
            )
            if not all(absence_identity):
                raise ValueError("Stage II V2 reasoning absence identity mismatch")
        expected_parsers = {
            "reasoning_grounding": "reasoning_v4_6_full_target_strict_v1",
            "decision": "reasoning_v4_6_decision_section_strict_v1",
        }
        if self.decision_context.producer.parser_contract_id != expected_parsers["decision"]:
            raise ValueError("Stage II V2 Decision parser contract mismatch")
        if isinstance(self.reasoning, StageIIReasoningPresentV2):
            parser_id = self.reasoning.reasoning_context.producer.parser_contract_id
            if parser_id != expected_parsers["reasoning_grounding"]:
                raise ValueError("Stage II V2 reasoning parser contract mismatch")
        if (
            self.private_direction_supervision.profile_id
            != self.direction_context.action_profile_id
            or self.private_direction_supervision.profile_version
            != self.direction_context.action_profile_version
        ):
            raise ValueError("Stage II V2 private direction/profile mismatch")
        forbidden = _find_private_prompt_keys(self.model_visible_context)
        if forbidden:
            raise ValueError(f"Stage II V2 prompt-visible private keys: {forbidden}")
        expected_pass_t = _stage_ii_pass_t_digest_from_view_v2(self)
        if self.pass_t_input_digest != expected_pass_t:
            raise ValueError("Stage II V2 pass_t_input_digest mismatch")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"view_digest"})
        )
        if self.view_digest != expected:
            raise ValueError("Stage II V2 view_digest mismatch")
        return self

    def _validate_cache_identity(
        self,
        cache: StageIIContextCacheV1,
        kind: StageIIContextKindV1,
    ) -> None:
        """Require one accepted cache over the exact V2 observation."""

        if cache.parse_status != "accepted" or cache.context_kind != kind:
            raise ValueError(f"Stage II V2 requires accepted {kind} cache")
        identities = (
            cache.sample_id == self.parent_stage_i_sample_id,
            cache.source_group_id == self.source_group_id,
            cache.split_group_id == self.split_group_id,
            cache.dataset_source == self.dataset_source,
            cache.modality_profile_id == self.modality_profile_id,
            cache.parent_source_fingerprint == self.parent_source_fingerprint,
            cache.projected_input_digest == self.projected_input_digest,
        )
        if not all(identities):
            raise ValueError(f"Stage II V2 {kind} cache identity mismatch")


class StageIICachedTrajQueryObservationV1(BaseModel):
    """Model-visible observation used to produce one cached query state.

    This record intentionally contains no trajectory target or private action
    supervision. It is paired with a later action view by the training-view
    contract below.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION] = (
        STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: StageIModalityProfileIdV1
    images: list[str] = Field(min_length=1, max_length=16)
    image_roles: list[str] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, min_length=1, max_length=4096)
    model_visible_context: dict[str, Any] = Field(default_factory=dict)
    evidence_catalog: list[EvidenceCatalogEntryV4] = Field(min_length=1, max_length=64)
    input_modalities: InputModalitiesV4
    frame_timestamps_sec: list[float] = Field(min_length=1, max_length=16)
    observation_timestamp_sec: float
    bev_timestamp_sec: float | None = None
    view_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_observation(self) -> "StageIICachedTrajQueryObservationV1":
        """Require causal frame timestamps and a target-free immutable digest."""

        if (
            len(self.images) != len(self.image_roles)
            or len(self.images) != len(self.frame_timestamps_sec)
        ):
            raise ValueError("cached query observation media and timestamps must align")
        if self.image_roles.count("fpv.current") != 1:
            raise ValueError("cached query observation requires exactly one fpv.current")
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("cached query observation image_roles must be unique")
        values = [*self.frame_timestamps_sec, self.observation_timestamp_sec]
        if self.bev_timestamp_sec is not None:
            values.append(self.bev_timestamp_sec)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("cached query observation timestamps must be finite")
        fpv_timestamps = [
            timestamp
            for role, timestamp in zip(self.image_roles, self.frame_timestamps_sec)
            if role.startswith("fpv.")
        ]
        if any(left > right for left, right in zip(fpv_timestamps, fpv_timestamps[1:])):
            raise ValueError("cached query observation FPV timestamps must be ordered")
        bev_indices = [
            index for index, role in enumerate(self.image_roles) if role == "bev.current"
        ]
        if bev_indices:
            if self.bev_timestamp_sec is None:
                raise ValueError("cached query observation bev.current requires bev_timestamp_sec")
            if abs(self.frame_timestamps_sec[bev_indices[0]] - self.bev_timestamp_sec) > 1e-6:
                raise ValueError("cached query observation BEV timestamp does not match its media slot")
        elif self.bev_timestamp_sec is not None:
            raise ValueError("cached query observation bev_timestamp_sec requires bev.current")
        if any(value > self.observation_timestamp_sec + 1e-6 for value in self.frame_timestamps_sec):
            raise ValueError("cached query observation cannot contain future FPV frames")
        if (
            self.bev_timestamp_sec is not None
            and self.bev_timestamp_sec > self.observation_timestamp_sec + 1e-6
        ):
            raise ValueError("cached query observation cannot contain a future BEV frame")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"view_digest"})
        )
        if self.view_digest != expected:
            raise ValueError("cached query observation view_digest mismatch")
        return self


class StageIICachedTrajQueryTrainingViewV1(BaseModel):
    """Pair a slow target-free observation with a later CFM action view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_SCHEMA_VERSION] = (
        STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_SCHEMA_VERSION
    )
    training_contract: Literal[STAGE_II_V2_TRAINING_CONTRACT_ID] = STAGE_II_V2_TRAINING_CONTRACT_ID
    sample_id: str = Field(min_length=1, max_length=560)
    query_source: StageIICachedTrajQueryObservationV1
    action_observation: StageIICachedTrajQueryObservationV1
    action_view: StageIICFMTrainingViewV2
    cached_traj_query: StageIICachedTrajQueryBindingV1
    formal_training_eligible: Literal[True] = True
    view_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_pair(self) -> "StageIICachedTrajQueryTrainingViewV1":
        """Close identity, causal-time and target-isolation invariants."""

        query = self.query_source
        action = self.action_observation
        target_view = self.action_view
        if target_view.system1_condition_mode != "traj_query_only":
            raise ValueError("cached pair action_view must start from traj_query_only V2 view")
        identities = (
            action.source_group_id == target_view.source_group_id,
            query.split_group_id == action.split_group_id == target_view.split_group_id,
            query.dataset_source == action.dataset_source == target_view.dataset_source,
            query.modality_profile_id == action.modality_profile_id == target_view.modality_profile_id,
            action.images == target_view.images,
            query.image_roles == action.image_roles == target_view.image_roles,
            query.instruction == action.instruction == target_view.instruction,
            query.model_visible_context == action.model_visible_context == target_view.model_visible_context,
            query.input_modalities == action.input_modalities == target_view.input_modalities,
        )
        if not all(identities):
            raise ValueError("cached pair source/action identity or visible inputs disagree")
        if self.cached_traj_query.modality_profile_id != action.modality_profile_id:
            raise ValueError("cached pair modality profile mismatch")
        if abs(self.cached_traj_query.query_source_timestamp_sec - query.observation_timestamp_sec) > 1e-4:
            raise ValueError("cached pair query timestamp does not match query source")
        if abs(self.cached_traj_query.action_observation_timestamp_sec - action.observation_timestamp_sec) > 1e-4:
            raise ValueError("cached pair action timestamp does not match action observation")
        if query.observation_timestamp_sec > action.observation_timestamp_sec:
            raise ValueError("cached pair query source cannot be newer than action observation")
        expected_sample_id = f"{target_view.sample_id}:cached_traj_query_v1"
        if self.sample_id != expected_sample_id:
            raise ValueError("cached pair sample_id does not encode its condition mode")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"view_digest"})
        )
        if self.view_digest != expected:
            raise ValueError("cached pair view_digest mismatch")
        return self


class StageIICachedTrajQueryBindingV2(BaseModel):
    """Bind a target-free slow observation to a later action observation.

    The two observations may use different modality profiles and different
    physical source groups.  The binding carries only the causal timing and
    stable-condition identity; it never contains a trajectory target.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_CACHED_TRAJ_QUERY_BINDING_V2_SCHEMA_VERSION] = (
        STAGE_II_CACHED_TRAJ_QUERY_BINDING_V2_SCHEMA_VERSION
    )
    condition_mode: Literal[STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2] = (
        STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2
    )
    query_source_timestamp_sec: float
    action_observation_timestamp_sec: float
    query_age_sec: float
    query_source_group_id: str = Field(min_length=1, max_length=240)
    action_source_group_id: str = Field(min_length=1, max_length=240)
    query_modality_profile_id: StageIModalityProfileIdV1
    action_modality_profile_id: StageIModalityProfileIdV1
    condition_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_causal_binding(self) -> "StageIICachedTrajQueryBindingV2":
        """Require causal timestamps and an internally consistent query age."""

        values = (
            self.query_source_timestamp_sec,
            self.action_observation_timestamp_sec,
            self.query_age_sec,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("cached trajectory-query V2 timestamps must be finite")
        if self.query_source_timestamp_sec > self.action_observation_timestamp_sec:
            raise ValueError("cached trajectory-query V2 cannot come from the future")
        expected_age = (
            self.action_observation_timestamp_sec
            - self.query_source_timestamp_sec
        )
        if abs(self.query_age_sec - expected_age) > 1e-4:
            raise ValueError("cached trajectory-query V2 age does not match timestamps")
        if self.query_age_sec < 0.0:
            raise ValueError("cached trajectory-query V2 age must be non-negative")
        return self


class StageIICachedTrajQueryTrainingViewV2(BaseModel):
    """Pair target-free query input with current action input and a 4s/32 target.

    This view deliberately bypasses Pass-R and Pass-D.  Those caches belong to
    the legacy explicit-text reasoning route; the V2 action condition is built
    from the query hidden states and current visual memory in ms-swift.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_V2_SCHEMA_VERSION] = (
        STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_V2_SCHEMA_VERSION
    )
    training_contract: Literal[STAGE_II_V2_TRAINING_CONTRACT_ID] = (
        STAGE_II_V2_TRAINING_CONTRACT_ID
    )
    condition_mode: Literal[STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2] = (
        STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2
    )
    sample_id: str = Field(min_length=1, max_length=560)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    query_source: StageIICachedTrajQueryObservationV1
    action_observation: StageIICachedTrajQueryObservationV1
    temporal_binding: StageIICachedTrajQueryBindingV2
    trajectory_target: StageIITrajectoryTargetV2
    provenance: dict[str, Any] = Field(default_factory=dict)
    formal_training_eligible: Literal[True] = True
    loss_mask: StageIILossMaskV1 = Field(default_factory=StageIILossMaskV1)
    view_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_training_view(self) -> "StageIICachedTrajQueryTrainingViewV2":
        """Close pair identity, target isolation, timing and digest invariants."""

        query = self.query_source
        action = self.action_observation
        binding = self.temporal_binding
        if query.split_group_id != self.split_group_id or action.split_group_id != self.split_group_id:
            raise ValueError("cached trajectory-query V2 split identity mismatch")
        if query.dataset_source != self.dataset_source or action.dataset_source != self.dataset_source:
            raise ValueError("cached trajectory-query V2 dataset identity mismatch")
        if action.source_group_id != self.source_group_id:
            raise ValueError("cached trajectory-query V2 action source identity mismatch")
        if binding.query_source_group_id != query.source_group_id:
            raise ValueError("cached trajectory-query V2 query source binding mismatch")
        if binding.action_source_group_id != action.source_group_id:
            raise ValueError("cached trajectory-query V2 action source binding mismatch")
        if binding.query_modality_profile_id != query.modality_profile_id:
            raise ValueError("cached trajectory-query V2 query profile mismatch")
        if binding.action_modality_profile_id != action.modality_profile_id:
            raise ValueError("cached trajectory-query V2 action profile mismatch")
        if abs(binding.query_source_timestamp_sec - query.observation_timestamp_sec) > 1e-4:
            raise ValueError("cached trajectory-query V2 query timestamp mismatch")
        if abs(binding.action_observation_timestamp_sec - action.observation_timestamp_sec) > 1e-4:
            raise ValueError("cached trajectory-query V2 action timestamp mismatch")
        if self.sample_id != f"{self.source_group_id}:cached_traj_query_v2":
            raise ValueError("cached trajectory-query V2 sample_id mismatch")
        if self.loss_mask != StageIILossMaskV1():
            raise ValueError("cached trajectory-query V2 requires the CFM-only loss mask")
        target_source = str(self.provenance.get("trajectory_source_group_id") or "")
        if target_source and target_source != self.source_group_id:
            raise ValueError("cached trajectory-query V2 target provenance mismatch")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"view_digest"})
        )
        if self.view_digest != expected:
            raise ValueError("cached trajectory-query V2 view_digest mismatch")
        return self


class StageIICachedTrajQuerySourceViewV2(BaseModel):
    """Carry visible Stage-I input and one accepted private trajectory target.

    This is the materializer input, not a model prompt row.  It deliberately
    contains no reasoning or Decision cache; the paired view is formed later
    from two target-free observations and the action-side target.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_CACHED_TRAJ_QUERY_SOURCE_VIEW_V2_SCHEMA_VERSION] = (
        STAGE_II_CACHED_TRAJ_QUERY_SOURCE_VIEW_V2_SCHEMA_VERSION
    )
    training_contract: Literal[STAGE_II_V2_TRAINING_CONTRACT_ID] = (
        STAGE_II_V2_TRAINING_CONTRACT_ID
    )
    sample_id: str = Field(min_length=1, max_length=560)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    observation: StageIICachedTrajQueryObservationV1
    trajectory_target: StageIITrajectoryTargetV2
    provenance: dict[str, Any] = Field(default_factory=dict)
    formal_training_eligible: Literal[True] = True
    view_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_source_view(self) -> "StageIICachedTrajQuerySourceViewV2":
        """Close source identity without admitting hidden text caches."""

        if self.source_group_id != self.observation.source_group_id:
            raise ValueError("cached-query V2 source identity mismatch")
        if self.split_group_id != self.observation.split_group_id:
            raise ValueError("cached-query V2 source split identity mismatch")
        if self.dataset_source != self.observation.dataset_source:
            raise ValueError("cached-query V2 source dataset identity mismatch")
        target_source = str(self.provenance.get("trajectory_source_group_id") or "")
        if target_source and target_source != self.source_group_id:
            raise ValueError("cached-query V2 source target provenance mismatch")
        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"view_digest"})
        )
        if self.view_digest != expected:
            raise ValueError("cached-query V2 source view_digest mismatch")
        return self


class StageIIProjectionDropV2(BaseModel):
    """Record one fail-closed V2 projection rejection without training content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_PROJECTION_DROP_V2_SCHEMA_VERSION] = (
        STAGE_II_PROJECTION_DROP_V2_SCHEMA_VERSION
    )
    parent_sample_id: str = Field(min_length=1, max_length=480)
    stage_i_route_sample_id: str = Field(min_length=1, max_length=480)
    stage_i_route_task_type: str = Field(min_length=1, max_length=96)
    requested_context_mode: str = Field(min_length=1, max_length=96)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: str = Field(min_length=1, max_length=160)
    reason_code: ReasonCode
    detail: str = Field(min_length=1, max_length=1000)
    projection_policy_id: Literal[STAGE_II_PROJECTION_POLICY_V2_ID] = (
        STAGE_II_PROJECTION_POLICY_V2_ID
    )
    projection_policy_version: Literal[STAGE_II_PROJECTION_POLICY_V2_VERSION] = (
        STAGE_II_PROJECTION_POLICY_V2_VERSION
    )
    drop_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_drop(self) -> "StageIIProjectionDropV2":
        """Require immutable rejection evidence."""

        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"drop_digest"})
        )
        if self.drop_digest != expected:
            raise ValueError("Stage II V2 drop_digest mismatch")
        return self


class StageIIProjectionDropV1(BaseModel):
    """Record one fail-closed Stage II projection rejection.

    Args:
        schema_version: Exact drop schema stamp.
        parent_sample_id: Best available parent identity.
        source_group_id: Best available source group.
        split_group_id: Best available leakage-safe split group.
        dataset_source: Dataset family.
        modality_profile_id: Requested modality profile.
        reason_code: Stable bounded rejection category.
        detail: Bounded engineering diagnostic.
        projection_policy_id: Exact projection policy.
        projection_policy_version: Exact projection-policy version.
        drop_digest: SHA-256 over all other drop fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STAGE_II_PROJECTION_DROP_SCHEMA_VERSION] = (
        STAGE_II_PROJECTION_DROP_SCHEMA_VERSION
    )
    parent_sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    modality_profile_id: str = Field(min_length=1, max_length=128)
    reason_code: ReasonCode
    detail: str = Field(min_length=1, max_length=1000)
    projection_policy_id: Literal[STAGE_II_PROJECTION_POLICY_ID] = (
        STAGE_II_PROJECTION_POLICY_ID
    )
    projection_policy_version: Literal[STAGE_II_PROJECTION_POLICY_VERSION] = (
        STAGE_II_PROJECTION_POLICY_VERSION
    )
    drop_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_drop(self) -> "StageIIProjectionDropV1":
        """Require immutable rejection evidence."""

        expected = stage_ii_digest_v1(
            self.model_dump(mode="json", exclude={"drop_digest"})
        )
        if self.drop_digest != expected:
            raise ValueError("Stage II drop_digest mismatch")
        return self


def stage_ii_pass_t_input_digest_v2(
    *,
    projected_input_digest: str,
    context_mode: StageIIContextModeV2,
    reasoning: StageIIReasoningContextV2 | dict[str, Any],
    decision_context: StageIIContextCacheV1 | dict[str, Any],
) -> str:
    """Fingerprint only the content visible to the V2 Pass-T prompt.

    Producer metadata and private targets are deliberately excluded.  The
    digest binds the common observation, explicit reasoning state, optional
    canonical grounding text, and accepted standalone Decision text.
    """

    typed_reasoning = (
        STAGE_II_REASONING_CONTEXT_ADAPTER_V2.validate_python(reasoning)
        if isinstance(reasoning, dict)
        else reasoning
    )
    typed_decision = (
        StageIIContextCacheV1.model_validate(decision_context)
        if isinstance(decision_context, dict)
        else decision_context
    )
    reasoning_visible: dict[str, Any] = {
        "state": typed_reasoning.reasoning_state,
    }
    if isinstance(typed_reasoning, StageIIReasoningPresentV2):
        reasoning_visible["canonical_text"] = (
            typed_reasoning.reasoning_context.canonical_text
        )
    return stage_ii_digest_v1(
        {
            "projected_input_digest": projected_input_digest,
            "context_mode": context_mode,
            "reasoning": reasoning_visible,
            "decision": typed_decision.canonical_text,
        }
    )


def _stage_ii_pass_t_digest_from_view_v2(view: StageIICFMTrainingViewV2) -> str:
    """Compute the public Pass-T digest from one typed V2 view."""

    return stage_ii_pass_t_input_digest_v2(
        projected_input_digest=view.projected_input_digest,
        context_mode=view.context_mode,
        reasoning=view.reasoning,
        decision_context=view.decision_context,
    )


def stage_ii_digest_v1(payload: Any) -> str:
    """Return canonical SHA-256 after recursively removing JSON nulls.

    Args:
        payload: JSON-compatible value to canonicalize.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    normalized = _drop_none_values(payload)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage_ii_projected_input_digest_v1(
    *,
    images: list[str],
    image_roles: list[str],
    instruction: str | None,
    model_visible_context: Mapping[str, Any],
    evidence_catalog: list[Mapping[str, Any]],
    input_modalities: Mapping[str, Any],
) -> str:
    """Fingerprint the exact prompt-visible observation before any Stage II pass.

    Args:
        images: Ordered projected media references.
        image_roles: Semantic role aligned with every image.
        instruction: Optional model-visible instruction.
        model_visible_context: Additional safe visible context.
        evidence_catalog: Prompt-visible evidence inventory.
        input_modalities: Canonical projected modality flags.

    Returns:
        Canonical SHA-256 digest shared by cache-bearing and direct-latent
        Stage II profiles.
    """

    return stage_ii_digest_v1({
        "images": images,
        "image_roles": image_roles,
        "instruction": instruction,
        "model_visible_context": dict(model_visible_context),
        "evidence_catalog": [dict(item) for item in evidence_catalog],
        "input_modalities": dict(input_modalities),
    })


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


def _find_private_prompt_keys(value: Any, *, path: str = "") -> list[str]:
    """Find target-like keys that must never enter model-visible context."""

    forbidden_names = {
        "action_mask",
        "decision_target",
        "direction_supervision",
        "trajectory_ego_t",
        "trajectory_target",
    }
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key).lower() in forbidden_names:
                found.append(child)
            found.extend(_find_private_prompt_keys(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_private_prompt_keys(item, path=f"{path}[{index}]"))
    return found


__all__ = [
    "STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_ID",
    "STAGE_II_DIRECT_LATENT_PROJECTION_POLICY_VERSION",
    "STAGE_II_DIRECT_LATENT_VIEW_SCHEMA_VERSION",
    "STAGE_II_CFM_TRAINING_VIEW_V2_SCHEMA_VERSION",
    "STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION",
    "STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION",
    "STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE",
    "STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2",
    "STAGE_II_CACHED_TRAJ_QUERY_BINDING_V2_SCHEMA_VERSION",
    "STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION",
    "STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_SCHEMA_VERSION",
    "STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_V2_SCHEMA_VERSION",
    "STAGE_II_CACHED_TRAJ_QUERY_SOURCE_VIEW_V2_SCHEMA_VERSION",
    "STAGE_II_GROUNDING_SECTION_NAMES",
    "STAGE_II_HORIZON_SECONDS",
    "STAGE_II_HORIZON_SECONDS_V2",
    "STAGE_II_HORIZON_STEPS",
    "STAGE_II_HORIZON_STEPS_V2",
    "STAGE_II_LEGACY_DIRECT_PROFILE_ID",
    "STAGE_II_LEGACY_DIRECT_PROFILE_VERSION",
    "STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION",
    "STAGE_II_PROJECTION_DROP_SCHEMA_VERSION",
    "STAGE_II_PROJECTION_DROP_V2_SCHEMA_VERSION",
    "STAGE_II_PROJECTION_POLICY_ID",
    "STAGE_II_PROJECTION_POLICY_VERSION",
    "STAGE_II_PROJECTION_POLICY_V2_ID",
    "STAGE_II_PROJECTION_POLICY_V2_VERSION",
    "STAGE_II_REASONING_CONTEXT_ADAPTER_V2",
    "STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION",
    "STAGE_II_TARGET_FORMAT",
    "STAGE_II_TARGET_DT_SEC_V2",
    "STAGE_II_TEMPORAL_RESAMPLING_ID",
    "STAGE_II_TEMPORAL_RESAMPLING_V2_ID",
    "STAGE_II_TRAINING_CONTRACT_ID",
    "STAGE_II_V2_TRAINING_CONTRACT_ID",
    "STAGE_II_V46_DIRECT_LATENT_PROFILE_ID",
    "STAGE_II_V46_DIRECT_LATENT_PROFILE_VERSION",
    "STAGE_II_TRAJECTORY_SOURCE_REGISTRY_ID",
    "STAGE_II_TRAJECTORY_SOURCE_REGISTRY_SCHEMA_VERSION",
    "STAGE_II_TRAJECTORY_SOURCE_REGISTRY_VERSION",
    "STAGE_II_TRAJECTORY_POLICY_ID",
    "STAGE_II_TRAJECTORY_POLICY_VERSION",
    "STAGE_II_TRAJECTORY_POLICY_V2_ID",
    "STAGE_II_TRAJECTORY_POLICY_V2_VERSION",
    "STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION",
    "STAGE_II_TRAJECTORY_QC_V2_SCHEMA_VERSION",
    "STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION",
    "STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION",
    "STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION",
    "STAGE_II_MAX_SOURCE_GAP_SEC_V2",
    "STAGE_II_SOURCE_SPEED_AUDIT_THRESHOLD_MPS_V2",
    "STAGE_II_MAX_SOURCE_STEP_M_V2",
    "STAGE_II_MAX_SOURCE_YAW_RATE_RAD_S_V2",
    "StageIICFMTrainingViewV1",
    "StageIICFMDirectLatentViewV1",
    "StageIICFMTrainingViewV2",
    "StageIICachedTrajQueryBindingV1",
    "StageIICachedTrajQueryBindingV2",
    "StageIICachedTrajQueryObservationV1",
    "StageIICachedTrajQueryTrainingViewV1",
    "StageIICachedTrajQueryTrainingViewV2",
    "StageIICachedTrajQuerySourceViewV2",
    "StageIIContextCacheV1",
    "StageIIContextModeV2",
    "StageIIConditionModeV2",
    "StageIIContextKindV1",
    "StageIIContextProducerV1",
    "StageIILossMaskV1",
    "StageIILegacyDirectTargetV1",
    "StageIIProjectionDropV1",
    "StageIIProjectionDropV2",
    "StageIIReasoningAbsentV2",
    "StageIIReasoningContextV2",
    "StageIIReasoningPresentV2",
    "StageIIReasoningStateV2",
    "StageIITrajectoryPolicyV1",
    "StageIITrajectoryPolicyV2",
    "StageIITrajectoryQCReportV1",
    "StageIITrajectoryQCReportV2",
    "StageIITrajectorySidecarV2",
    "StageIITrajectorySourceRegistryV1",
    "StageIITrajectorySourceSpecV1",
    "StageIITrajectorySplitPolicyV1",
    "StageIITrajectoryTargetV1",
    "StageIITrajectoryTargetV2",
    "stage_ii_digest_v1",
    "stage_ii_projected_input_digest_v1",
    "stage_ii_pass_t_input_digest_v2",
]
