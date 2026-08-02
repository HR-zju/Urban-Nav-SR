"""Canonical exchange contracts for the UrbanNav open-loop benchmark.

The benchmark deliberately separates model-visible inputs, hidden references,
model outputs, and failures.  Task-specific semantic validation remains in the
Stage I and Stage II evaluators; this module owns artifact identity, provenance,
renderer policy, and run lineage only.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.direction_profile import DirectionLabelStamp


OPENLOOP_BENCHMARK_SCHEMA_VERSION = "urbannav_openloop_benchmark_v2_0"
OPENLOOP_INPUT_SCHEMA_VERSION = "urbannav_openloop_input_v2_0"
OPENLOOP_REFERENCE_SCHEMA_VERSION = "urbannav_openloop_reference_v2_0"
OPENLOOP_PREDICTION_SCHEMA_VERSION = "urbannav_openloop_prediction_v2_0"
OPENLOOP_FAILURE_SCHEMA_VERSION = "urbannav_openloop_failure_v2_0"
OPENLOOP_RUN_SCHEMA_VERSION = "urbannav_openloop_run_v2_0"
OPENLOOP_METRIC_POLICY_VERSION = "urbannav_openloop_metric_policy_v2_0"
OPENLOOP_RENDERER_POLICY_VERSION = "urbannav_openloop_renderer_policy_v2_0"
OPENLOOP_EXPOSURE_REGISTRY_VERSION = (
    "urbannav_checkpoint_exposure_registry_v2_0"
)
OPENLOOP_SPLIT_LINEAGE_ENTRY_VERSION = (
    "urbannav_openloop_split_lineage_entry_v2_0"
)

OpenLoopStageV1 = Literal["stage_i", "stage_ii"]
OpenLoopTaskTypeV1 = Literal[
    "full_cot",
    "vqa",
    "decision",
    "trajectory",
]
ReferenceSemanticV1 = Literal[
    "deterministic_gt",
    "human_gold",
    "lvlm_weak",
    "reference_annotation",
]
AttemptedPassV1 = Literal[
    "stage_i_generation",
    "pass_r",
    "pass_d",
    "pass_t",
    "evaluation",
]
RunContextPolicyV1 = Literal[
    "not_applicable",
    "shared_current_checkpoint_v1",
    "frozen_context_ablation",
]

_PRIVATE_INPUT_KEYS = {
    "action_mask",
    "answer",
    "assistant_target",
    "direction_supervision",
    "human_adjudication",
    "loss_scale",
    "loss_spans",
    "reasoning_target",
    "reference",
    "strong_direction_label",
    "teacher_private_rationale",
    "trajectory_ego_t",
}


class OpenLoopMessageV1(BaseModel):
    """Represent one model-visible prompt message.

    Args:
        role: Prompt role accepted by the inference adapter.
        content: Exact model-visible content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant", "tool", "tool_response"]
    content: str = Field(min_length=1)


class OpenLoopRendererPolicyV1(BaseModel):
    """Freeze canonical context rendering and overflow behavior.

    Args:
        policy_version: Exact renderer-policy protocol version.
        reasoning_renderer_id: Canonical Reasoning renderer implementation.
        reasoning_renderer_digest: Digest of the Reasoning renderer contract.
        decision_renderer_id: Canonical Decision renderer implementation.
        decision_renderer_digest: Digest of the Decision renderer contract.
        field_order_policy: Frozen mapping-field ordering policy.
        list_normalization_policy: Frozen list normalization policy.
        null_normalization_policy: Frozen optional/null normalization policy.
        max_context_tokens: Hard maximum context length.
        overflow_policy: Overflow handling; V2.0 fails closed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: Literal[OPENLOOP_RENDERER_POLICY_VERSION] = (
        OPENLOOP_RENDERER_POLICY_VERSION
    )
    reasoning_renderer_id: str = Field(min_length=1, max_length=160)
    reasoning_renderer_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_renderer_id: str = Field(min_length=1, max_length=160)
    decision_renderer_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    field_order_policy: Literal["schema_order_v1"] = "schema_order_v1"
    list_normalization_policy: Literal[
        "preserve_semantic_order_sort_unordered_v1"
    ] = "preserve_semantic_order_sort_unordered_v1"
    null_normalization_policy: Literal["omit_optional_none_v1"] = (
        "omit_optional_none_v1"
    )
    max_context_tokens: int = Field(gt=0)
    overflow_policy: Literal["fail_closed"] = "fail_closed"


class OpenLoopMetricPolicyV1(BaseModel):
    """Freeze primary aggregations and metric-domain conventions.

    Args:
        policy_version: Exact metric-policy protocol version.
        full_cot_reliability_primary: FullCoT reliability aggregation.
        full_cot_semantics_primary: FullCoT semantic aggregation.
        decision_primary: Standalone Decision aggregation.
        vqa_primary: VQA aggregation order.
        stage_ii_geometry_primary: Trajectory aggregation order.
        direction_range_policy: Treatment of 50plus direction ranges.
        stage_ii_horizons_s: Fixed evaluation horizons in seconds.
        stage_ii_step_s: Time interval between target points.
        stage_ii_motion_policy: Physical motion policy availability.
        semantic_failure_denominator: Denominator used for strict semantics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: Literal[OPENLOOP_METRIC_POLICY_VERSION] = (
        OPENLOOP_METRIC_POLICY_VERSION
    )
    full_cot_reliability_primary: Literal["parent_natural_micro"] = (
        "parent_natural_micro"
    )
    full_cot_semantics_primary: Literal["human_gold_field_macro"] = (
        "human_gold_field_macro"
    )
    decision_primary: Literal["native_profile_macro"] = (
        "native_profile_macro"
    )
    vqa_primary: Literal[
        "task_parent_balanced_then_provenance_task_macro"
    ] = "task_parent_balanced_then_provenance_task_macro"
    stage_ii_geometry_primary: Literal[
        "eligible_dataset_macro_per_horizon"
    ] = "eligible_dataset_macro_per_horizon"
    direction_range_policy: Literal[
        "exclude_50plus_from_point_angle_report_coverage"
    ] = "exclude_50plus_from_point_angle_report_coverage"
    stage_ii_horizons_s: tuple[float, ...] = (0.5, 1.0, 2.0)
    stage_ii_step_s: float = 0.125
    stage_ii_motion_policy: Literal["not_frozen", "frozen"] = "not_frozen"
    semantic_failure_denominator: Literal["full_reference"] = (
        "full_reference"
    )

    @model_validator(mode="after")
    def validate_time_grid(self) -> "OpenLoopMetricPolicyV1":
        """Require the exact Stage II V2.0 time grid.

        Returns:
            Validated policy.

        Raises:
            ValueError: If horizons or step duration drift.
        """

        if self.stage_ii_horizons_s != (0.5, 1.0, 2.0):
            raise ValueError("Stage II V2.0 horizons must be 0.5/1.0/2.0 s")
        if self.stage_ii_step_s != 0.125:
            raise ValueError("Stage II V2.0 step must be 0.125 s")
        return self


class OpenLoopInputV1(BaseModel):
    """Store one strictly model-visible benchmark input.

    Args:
        sample_id: Stable task-row identity.
        parent_sample_id: Stable physical-sample identity.
        source_group_id: Stable source/session group.
        split_group_id: Atomic train/validation/test split group.
        split: Frozen benchmark split.
        stage: Benchmark stage.
        task_type: Atomic task family.
        dataset_source: Dataset identity used for grouped metrics.
        messages: Model-visible prompt messages only.
        images: Ordered model-visible image paths.
        image_roles: Semantic role for every image.
        public_context: Additional prompt-visible structured context.
        task_id: VQA task identifier when applicable.
        modality_profile_id: Frozen input-modality profile.
        direction_profile_id: Public output vocabulary when applicable.
        renderer_policy_digest: Digest of the benchmark renderer policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[OPENLOOP_INPUT_SCHEMA_VERSION] = (
        OPENLOOP_INPUT_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=256)
    source_group_id: str = Field(min_length=1, max_length=256)
    split_group_id: str = Field(min_length=1, max_length=256)
    split: Literal["validation", "test"]
    stage: OpenLoopStageV1
    task_type: OpenLoopTaskTypeV1
    dataset_source: str = Field(min_length=1, max_length=128)
    messages: list[OpenLoopMessageV1] = Field(min_length=1)
    images: list[str] = Field(default_factory=list)
    image_roles: list[str] = Field(default_factory=list)
    public_context: dict[str, Any] = Field(default_factory=dict)
    task_id: str | None = Field(default=None, max_length=128)
    modality_profile_id: str = Field(min_length=1, max_length=160)
    direction_profile_id: str | None = Field(default=None, max_length=160)
    renderer_policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_visible_boundary(self) -> "OpenLoopInputV1":
        """Reject hidden targets and inconsistent image metadata.

        Returns:
            Validated model-visible input.

        Raises:
            ValueError: If private supervision leaks into the input.
        """

        if len(self.images) != len(self.image_roles):
            raise ValueError("images and image_roles must have equal length")
        if any(message.role == "assistant" for message in self.messages):
            raise ValueError("open-loop input cannot contain assistant turns")
        leaked = sorted(_find_private_input_keys(self.model_dump(mode="json")))
        if leaked:
            raise ValueError(
                "model-visible input contains private supervision: "
                + ",".join(leaked)
            )
        if self.task_type == "vqa" and not self.task_id:
            raise ValueError("VQA input requires task_id")
        if self.stage == "stage_ii" and self.task_type != "trajectory":
            raise ValueError("Stage II input must use task_type=trajectory")
        if self.stage == "stage_i" and self.task_type == "trajectory":
            raise ValueError("trajectory task belongs to Stage II")
        return self


class OpenLoopReferenceV1(BaseModel):
    """Store one hidden reference row aligned to a model-visible input.

    Args:
        sample_id: Exact input identity.
        parent_sample_id: Exact physical-sample identity.
        source_group_id: Exact source/session group.
        split_group_id: Exact split group.
        stage: Benchmark stage.
        task_type: Atomic task family.
        dataset_source: Dataset identity used for grouped metrics.
        target: Hidden task-specific target payload.
        label_provenance: Default semantic provenance of the target.
        field_provenance: Optional dotted field to provenance mapping.
        direction_stamp: Deterministic direction reference when applicable.
        trajectory_valid_mask: Hidden valid future-point mask for Stage II.
        trajectory_policy_id: Frozen target policy identity.
        subgroup_metadata: Hidden evaluation subgroup properties.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[OPENLOOP_REFERENCE_SCHEMA_VERSION] = (
        OPENLOOP_REFERENCE_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=256)
    source_group_id: str = Field(min_length=1, max_length=256)
    split_group_id: str = Field(min_length=1, max_length=256)
    stage: OpenLoopStageV1
    task_type: OpenLoopTaskTypeV1
    dataset_source: str = Field(min_length=1, max_length=128)
    target: dict[str, Any]
    label_provenance: ReferenceSemanticV1
    field_provenance: dict[str, ReferenceSemanticV1] = Field(
        default_factory=dict
    )
    direction_stamp: DirectionLabelStamp | None = None
    trajectory_valid_mask: list[bool] | None = None
    trajectory_policy_id: str | None = Field(default=None, max_length=160)
    subgroup_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference_shape(self) -> "OpenLoopReferenceV1":
        """Require task-specific hidden metadata only where meaningful.

        Returns:
            Validated reference row.

        Raises:
            ValueError: If Stage II target metadata are absent or misplaced.
        """

        if self.stage == "stage_ii":
            if self.task_type != "trajectory":
                raise ValueError("Stage II reference must be trajectory")
            if self.trajectory_valid_mask is None:
                raise ValueError("Stage II reference requires valid mask")
            if not self.trajectory_policy_id:
                raise ValueError("Stage II reference requires trajectory policy")
        elif self.trajectory_valid_mask is not None:
            raise ValueError("Stage I reference cannot carry trajectory mask")
        return self


class OpenLoopPredictionV1(BaseModel):
    """Store one successful model output and its inference identity.

    Args:
        sample_id: Exact benchmark row identity.
        parent_sample_id: Exact physical-sample identity.
        stage: Benchmark stage.
        task_type: Atomic task family.
        response: Raw text generation for Stage I tasks.
        trajectory_ego_t: Strict Stage II trajectory output.
        metadata: Non-secret generation and latency diagnostics.
        input_digest: Digest of the exact model-visible input.
        context_identity: Stage II R/D/T producer and renderer identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[OPENLOOP_PREDICTION_SCHEMA_VERSION] = (
        OPENLOOP_PREDICTION_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=256)
    stage: OpenLoopStageV1
    task_type: OpenLoopTaskTypeV1
    response: str | None = None
    trajectory_ego_t: list[list[float]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_identity: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_output_channel(self) -> "OpenLoopPredictionV1":
        """Require text for Stage I and trajectory values for Stage II.

        Returns:
            Validated prediction.

        Raises:
            ValueError: If the output channel does not match the task.
        """

        if self.stage == "stage_i":
            if self.response is None or self.trajectory_ego_t is not None:
                raise ValueError("Stage I prediction requires response only")
        else:
            if self.task_type != "trajectory":
                raise ValueError("Stage II prediction must be trajectory")
            if self.trajectory_ego_t is None or self.response is not None:
                raise ValueError("Stage II prediction requires trajectory only")
        return self


class OpenLoopFailureV1(BaseModel):
    """Record one benchmark attempt that did not produce a prediction.

    Args:
        sample_id: Exact benchmark row identity.
        parent_sample_id: Exact physical-sample identity.
        stage: Benchmark stage.
        task_type: Atomic task family.
        attempted_pass: Pass at which the attempt failed.
        failure_code: Stable bounded failure category.
        detail: Optional bounded diagnostic, never a traceback dump.
        retry_attempted: Whether the run retried the sample.
        input_digest: Digest of the attempted model-visible input.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[OPENLOOP_FAILURE_SCHEMA_VERSION] = (
        OPENLOOP_FAILURE_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=256)
    stage: OpenLoopStageV1
    task_type: OpenLoopTaskTypeV1
    attempted_pass: AttemptedPassV1
    failure_code: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    detail: str | None = Field(default=None, max_length=512)
    retry_attempted: bool = False
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class OpenLoopRepositoryStateV1(BaseModel):
    """Freeze one repository revision used by a benchmark run.

    Args:
        name: Repository role/name.
        commit: Exact Git commit.
        dirty: Whether uncommitted changes were present.
        dirty_diff_digest: Optional digest of the captured diff.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    dirty: bool
    dirty_diff_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_dirty_evidence(self) -> "OpenLoopRepositoryStateV1":
        """Require a diff digest whenever a run uses dirty code.

        Returns:
            Validated repository state.

        Raises:
            ValueError: If dirty code has no captured diff digest.
        """

        if self.dirty and self.dirty_diff_digest is None:
            raise ValueError("dirty repository requires dirty_diff_digest")
        return self


class OpenLoopCheckpointIdentityV1(BaseModel):
    """Freeze model/checkpoint identity for one run.

    Args:
        model_id: Base model identity.
        checkpoint_id: Human-readable checkpoint/run identity.
        checkpoint_digest: Content or immutable artifact digest.
        adapter_id: Optional adapter identity.
        tokenizer_id: Tokenizer identity.
        processor_id: Multimodal processor identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=256)
    checkpoint_id: str = Field(min_length=1, max_length=512)
    checkpoint_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_id: str | None = Field(default=None, max_length=512)
    tokenizer_id: str = Field(min_length=1, max_length=256)
    processor_id: str = Field(min_length=1, max_length=256)


class OpenLoopContextIdentityV1(BaseModel):
    """Freeze Pass R/D/T lineage and inference context policy.

    Args:
        policy: Production or explicit frozen-context ablation policy.
        reasoning_context_source: Benchmark provenance of the Pass R context.
        decision_context_source: Benchmark provenance of the Pass D context.
        reasoning_producer_checkpoint_digest: Pass R producer backbone digest.
        decision_producer_checkpoint_digest: Pass D producer backbone digest.
        pass_t_backbone_checkpoint_digest: Pass T backbone digest.
        reasoning_renderer_digest: Canonical Pass R renderer digest.
        decision_renderer_digest: Canonical Pass D renderer digest.
        input_manifest_digest: Exact canonical benchmark input artifact digest.
        adapter_input_manifest_digest: Exact physical ms-swift row artifact.
        reasoning_cache_digest: Exact Pass R cache digest.
        decision_cache_digest: Exact Pass D cache digest.
        normalization_profile_digest: Exact trajectory normalization digest.
        flow_steps: CFM sampling steps.
        seed: Run seed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: RunContextPolicyV1
    reasoning_context_source: Literal[
        "not_applicable",
        "reference_annotation",
        "model_prediction",
    ]
    decision_context_source: Literal[
        "not_applicable",
        "reference_annotation",
        "model_prediction",
    ]
    reasoning_producer_checkpoint_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    decision_producer_checkpoint_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    pass_t_backbone_checkpoint_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reasoning_renderer_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    decision_renderer_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    input_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_input_manifest_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reasoning_cache_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    decision_cache_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    normalization_profile_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    flow_steps: int | None = Field(default=None, gt=0)
    seed: int

    @model_validator(mode="after")
    def validate_checkpoint_lineage(self) -> "OpenLoopContextIdentityV1":
        """Enforce shared-current-checkpoint identity in production.

        Returns:
            Validated context identity.

        Raises:
            ValueError: If production R/D/T checkpoint digests differ.
        """

        if self.policy == "not_applicable":
            if (
                self.reasoning_context_source != "not_applicable"
                or self.decision_context_source != "not_applicable"
            ):
                raise ValueError(
                    "not-applicable context policy requires no R/D source"
                )
            return self
        required = (
            self.reasoning_producer_checkpoint_digest,
            self.decision_producer_checkpoint_digest,
            self.pass_t_backbone_checkpoint_digest,
            self.reasoning_renderer_digest,
            self.decision_renderer_digest,
            self.adapter_input_manifest_digest,
            self.reasoning_cache_digest,
            self.decision_cache_digest,
            self.normalization_profile_digest,
            self.flow_steps,
        )
        if any(value is None for value in required):
            raise ValueError("Stage II context identity is incomplete")
        if self.policy == "shared_current_checkpoint_v1":
            if (
                self.reasoning_context_source != "model_prediction"
                or self.decision_context_source != "model_prediction"
            ):
                raise ValueError(
                    "production requires model_prediction R/D context"
                )
            digests = {
                self.reasoning_producer_checkpoint_digest,
                self.decision_producer_checkpoint_digest,
                self.pass_t_backbone_checkpoint_digest,
            }
            if len(digests) != 1:
                raise ValueError(
                    "production requires equal Pass R/D/T checkpoint digests"
                )
        return self


class OpenLoopRunManifestV1(BaseModel):
    """Freeze one model run and all reproducibility-critical identities.

    Args:
        run_id: Stable run identity.
        benchmark_id: Frozen benchmark package identity.
        benchmark_digest: Digest of the benchmark package manifest.
        stage: Run stage.
        repositories: Exact repository states.
        checkpoint: Model/checkpoint identity.
        renderer_policy: Frozen rendering policy.
        metric_policy_digest: Digest of the metric policy.
        context_identity: Pass R/D/T lineage and sampling identity.
        generation_parameters: Decoding or sampling parameters.
        hardware: Non-secret hardware/runtime description.
        batch_size: Physical inference batch size.
        precision: Inference precision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[OPENLOOP_RUN_SCHEMA_VERSION] = (
        OPENLOOP_RUN_SCHEMA_VERSION
    )
    run_id: str = Field(min_length=1, max_length=256)
    benchmark_id: str = Field(min_length=1, max_length=256)
    benchmark_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage: OpenLoopStageV1
    repositories: list[OpenLoopRepositoryStateV1] = Field(min_length=2)
    checkpoint: OpenLoopCheckpointIdentityV1
    renderer_policy: OpenLoopRendererPolicyV1
    metric_policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_identity: OpenLoopContextIdentityV1
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(gt=0)
    precision: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_stage_identity(self) -> "OpenLoopRunManifestV1":
        """Require Stage II context identity only for Stage II runs.

        Returns:
            Validated run manifest.

        Raises:
            ValueError: If context identity contradicts the run stage.
        """

        policy = self.context_identity.policy
        if self.stage == "stage_i" and policy != "not_applicable":
            raise ValueError("Stage I run context policy must be not_applicable")
        if self.stage == "stage_ii" and policy == "not_applicable":
            raise ValueError("Stage II run requires a context policy")
        return self


class CheckpointExposureEntryV1(BaseModel):
    """Record one group exposed to a checkpoint lineage.

    Args:
        parent_sample_id: Physical-sample identity seen in training.
        split_group_id: Atomic exposed group identity.
        source_fingerprint: Optional immutable source fingerprint.
        manifest_digest: Train/replay manifest containing the exposure.
        stage: Training stage or replay source.
        annotation_version: Optional annotation protocol version.
        cache_version: Optional Stage II context-cache version.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_sample_id: str = Field(min_length=1, max_length=256)
    split_group_id: str = Field(min_length=1, max_length=256)
    source_fingerprint: str | None = Field(default=None, max_length=256)
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage: str = Field(min_length=1, max_length=96)
    annotation_version: str | None = Field(default=None, max_length=160)
    cache_version: str | None = Field(default=None, max_length=160)


class OpenLoopSplitLineageEntryV1(BaseModel):
    """Freeze one explicitly mapped parent and atomic split group.

    Args:
        split: Dataset role assigned by the source split manifest.
        dataset_source: Canonical dataset identity after an explicit alias map.
        parent_sample_id: Stable source-row identity.
        split_group_id: Atomic leakage-safe clip/session identity.
        source_manifest_id: Human-readable immutable manifest identity.
        source_manifest_digest: SHA256 of the source split manifest.
        source_row_number: One-based source JSONL row number.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[OPENLOOP_SPLIT_LINEAGE_ENTRY_VERSION] = (
        OPENLOOP_SPLIT_LINEAGE_ENTRY_VERSION
    )
    split: Literal["train", "validation", "test"]
    dataset_source: str = Field(min_length=1, max_length=128)
    parent_sample_id: str = Field(min_length=1, max_length=256)
    split_group_id: str = Field(min_length=1, max_length=256)
    source_manifest_id: str = Field(min_length=1, max_length=256)
    source_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_row_number: int = Field(gt=0)


class CheckpointExposureRegistryV1(BaseModel):
    """Store all known training exposure for one checkpoint lineage.

    Args:
        checkpoint_digest: Checkpoint lineage being audited.
        entries: Parent/group exposure records from all train manifests.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[OPENLOOP_EXPOSURE_REGISTRY_VERSION] = (
        OPENLOOP_EXPOSURE_REGISTRY_VERSION
    )
    checkpoint_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: list[CheckpointExposureEntryV1] = Field(default_factory=list)


class OpenLoopPackageManifestV1(BaseModel):
    """Freeze file identities for one input/reference benchmark package.

    Args:
        benchmark_id: Human-readable versioned benchmark identity.
        split: Frozen validation or test split.
        input_path: Relative model-visible input JSONL path.
        input_digest: SHA256 of the input artifact.
        reference_path: Relative hidden-reference JSONL path.
        reference_digest: SHA256 of the reference artifact.
        metric_policy: Frozen metric policy.
        renderer_policy: Frozen renderer policy.
        reference_count: Exact reference denominator.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[OPENLOOP_BENCHMARK_SCHEMA_VERSION] = (
        OPENLOOP_BENCHMARK_SCHEMA_VERSION
    )
    benchmark_id: str = Field(min_length=1, max_length=256)
    split: Literal["validation", "test"]
    input_path: str = Field(min_length=1)
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_path: str = Field(min_length=1)
    reference_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_policy: OpenLoopMetricPolicyV1
    renderer_policy: OpenLoopRendererPolicyV1
    reference_count: int = Field(gt=0)


def openloop_schema_digest_v1() -> str:
    """Return a semantic digest covering every public exchange model.

    Returns:
        SHA256 over normalized JSON Schemas and version constants.
    """

    payload = {
        "versions": {
            "benchmark": OPENLOOP_BENCHMARK_SCHEMA_VERSION,
            "input": OPENLOOP_INPUT_SCHEMA_VERSION,
            "reference": OPENLOOP_REFERENCE_SCHEMA_VERSION,
            "prediction": OPENLOOP_PREDICTION_SCHEMA_VERSION,
            "failure": OPENLOOP_FAILURE_SCHEMA_VERSION,
            "run": OPENLOOP_RUN_SCHEMA_VERSION,
        },
        "schemas": {
            model.__name__: _semantic_schema(model.model_json_schema())
            for model in (
                OpenLoopInputV1,
                OpenLoopReferenceV1,
                OpenLoopPredictionV1,
                OpenLoopFailureV1,
                OpenLoopRunManifestV1,
                OpenLoopPackageManifestV1,
                CheckpointExposureRegistryV1,
                OpenLoopSplitLineageEntryV1,
            )
        },
    }
    return sha256_json_v1(payload)


def sha256_json_v1(value: Any) -> str:
    """Return SHA256 over stable compact JSON.

    Args:
        value: JSON-compatible value.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_private_input_keys(value: Any, prefix: str = "") -> set[str]:
    """Find forbidden supervision keys recursively in an input payload."""

    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in _PRIVATE_INPUT_KEYS:
                found.add(path)
            found.update(_find_private_input_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_find_private_input_keys(item, f"{prefix}[{index}]"))
    return found


def _semantic_schema(value: Any) -> Any:
    """Remove unstable JSON Schema decoration before hashing."""

    if isinstance(value, dict):
        return {
            key: _semantic_schema(item)
            for key, item in sorted(value.items())
            if key not in {"description", "title"}
        }
    if isinstance(value, list):
        return [_semantic_schema(item) for item in value]
    return value


OPENLOOP_BENCHMARK_SCHEMA_DIGEST = openloop_schema_digest_v1()


__all__ = [
    "AttemptedPassV1",
    "CheckpointExposureEntryV1",
    "CheckpointExposureRegistryV1",
    "OPENLOOP_BENCHMARK_SCHEMA_DIGEST",
    "OPENLOOP_BENCHMARK_SCHEMA_VERSION",
    "OPENLOOP_EXPOSURE_REGISTRY_VERSION",
    "OPENLOOP_FAILURE_SCHEMA_VERSION",
    "OPENLOOP_INPUT_SCHEMA_VERSION",
    "OPENLOOP_METRIC_POLICY_VERSION",
    "OPENLOOP_PREDICTION_SCHEMA_VERSION",
    "OPENLOOP_REFERENCE_SCHEMA_VERSION",
    "OPENLOOP_RENDERER_POLICY_VERSION",
    "OPENLOOP_RUN_SCHEMA_VERSION",
    "OPENLOOP_SPLIT_LINEAGE_ENTRY_VERSION",
    "OpenLoopCheckpointIdentityV1",
    "OpenLoopContextIdentityV1",
    "OpenLoopFailureV1",
    "OpenLoopInputV1",
    "OpenLoopMessageV1",
    "OpenLoopMetricPolicyV1",
    "OpenLoopPackageManifestV1",
    "OpenLoopPredictionV1",
    "OpenLoopRendererPolicyV1",
    "OpenLoopRepositoryStateV1",
    "OpenLoopRunManifestV1",
    "OpenLoopSplitLineageEntryV1",
    "OpenLoopStageV1",
    "OpenLoopTaskTypeV1",
    "ReferenceSemanticV1",
    "RunContextPolicyV1",
    "openloop_schema_digest_v1",
    "sha256_json_v1",
]
