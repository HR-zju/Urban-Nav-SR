"""Canonical contracts for pluggable, frozen open-loop benchmarks.

This contract is deliberately independent from the legacy UrbanNav Stage I/II
open-loop exchange.  It defines the stable boundary shared by a benchmark
dataset plugin, a model-side prediction adapter, and one or more metric
suites.  Legacy contracts remain valid for existing Stage I/II experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


BENCHMARK_PLUGIN_SCHEMA_VERSION = "urbannav_benchmark_plugin_v1"
BENCHMARK_INPUT_SCHEMA_VERSION = "urbannav_benchmark_input_v1"
BENCHMARK_REFERENCE_SCHEMA_VERSION = "urbannav_benchmark_reference_v1"
BENCHMARK_PREDICTION_SCHEMA_VERSION = "urbannav_benchmark_prediction_v1"
BENCHMARK_FAILURE_SCHEMA_VERSION = "urbannav_benchmark_failure_v1"
BENCHMARK_MEDIA_MANIFEST_SCHEMA_VERSION = "urbannav_benchmark_media_manifest_v1"
BENCHMARK_PACKAGE_SCHEMA_VERSION = "urbannav_benchmark_package_v1"
BENCHMARK_PROFILE_SCHEMA_VERSION = "urbannav_benchmark_profile_v1"
BENCHMARK_CAPABILITIES_SCHEMA_VERSION = "urbannav_benchmark_capabilities_v1"

BenchmarkSplitV1 = Literal["train", "validation", "test"]
PredictionTimeBasisV1 = Literal["explicit", "reference_schedule"]
AdaptationModeV1 = Literal[
    "zero_shot",
    "prompt_only",
    "head_finetune",
    "full_finetune",
    "reference",
]
LabelProvenanceV1 = Literal[
    "deterministic_gt",
    "human_gold",
    "reference_annotation",
]

_PRIVATE_INPUT_KEYS = frozenset(
    {
        "action_mask",
        "answer",
        "arrival_label",
        "assistant_target",
        "future_trajectory",
        "ground_truth",
        "label",
        "reference",
        "scenario_labels",
        "target_trajectory",
        "trajectory_ego_t",
    }
)


class BenchmarkMessageV1(BaseModel):
    """One model-visible benchmark prompt message.

    Args:
        role: Prompt role accepted by an inference adapter.
        content: Exact visible message content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "tool", "tool_response"]
    content: str = Field(min_length=1)


class BenchmarkTimePolicyV1(BaseModel):
    """Freeze physical-time alignment and angular eligibility rules.

    Args:
        policy_id: Stable policy identity.
        horizons_s: Ordered positive physical horizons used for comparison.
        interpolation: Curve interpolation used by every metric suite.
        min_angle_reference_distance_m: Minimum GT radial displacement for
            an angle to be mathematically meaningful.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1, max_length=160)
    horizons_s: tuple[float, ...] = Field(min_length=1)
    interpolation: Literal["linear_xy_v1"] = "linear_xy_v1"
    min_angle_reference_distance_m: float = Field(ge=0.0, default=1e-3)

    @model_validator(mode="after")
    def validate_horizons(self) -> "BenchmarkTimePolicyV1":
        """Require finite, strictly increasing physical horizons."""

        prior = 0.0
        for horizon in self.horizons_s:
            if not math.isfinite(horizon) or horizon <= prior:
                raise ValueError("horizons_s must be finite and strictly increasing")
            prior = horizon
        return self


class BenchmarkProfileV1(BaseModel):
    """Bind one legal dataset, prediction geometry, and metric-suite set.

    Args:
        profile_id: Stable benchmark protocol identity.
        dataset_plugin_id: Dataset plugin that materializes frozen cases.
        model_input_profile_id: Exact public input rendering contract.
        prediction_geometry_id: Expected output geometry semantics.
        metric_suite_ids: Ordered metric suites evaluated by the runner.
        time_policy: Shared physical-time alignment contract.
        minimum_success_coverage: Minimum full-denominator successful-prediction
            fraction required for a formal runner to report ``passed``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[BENCHMARK_PROFILE_SCHEMA_VERSION] = BENCHMARK_PROFILE_SCHEMA_VERSION
    profile_id: str = Field(min_length=1, max_length=160)
    dataset_plugin_id: str = Field(min_length=1, max_length=160)
    model_input_profile_id: str = Field(min_length=1, max_length=160)
    prediction_geometry_id: str = Field(min_length=1, max_length=160)
    metric_suite_ids: tuple[str, ...] = Field(min_length=1)
    time_policy: BenchmarkTimePolicyV1
    minimum_success_coverage: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_metric_suites(self) -> "BenchmarkProfileV1":
        """Reject duplicate or empty metric suite identities."""

        if any(not item for item in self.metric_suite_ids):
            raise ValueError("metric_suite_ids cannot contain an empty value")
        if len(set(self.metric_suite_ids)) != len(self.metric_suite_ids):
            raise ValueError("metric_suite_ids must be unique")
        if not math.isfinite(self.minimum_success_coverage):
            raise ValueError("minimum_success_coverage must be finite")
        return self


class BenchmarkInputV1(BaseModel):
    """Store one model-visible, frozen benchmark case.

    Args:
        sample_id: Stable task identity, including any frozen point-goal draw.
        parent_sample_id: Underlying physical-frame identity.
        source_group_id: Stable source/session identity.
        split_group_id: Atomic split identity.
        split: Frozen train/validation/test assignment.
        dataset_plugin_id: Plugin that produced the case.
        benchmark_profile_id: Legal benchmark profile used for this row.
        dataset_source: Dataset family for grouped reports.
        messages: Exact model-visible prompt messages.
        images: Ordered model-visible image paths.
        image_roles: Semantic role for every image.
        public_context: Typed-by-profile public numeric/context payload.
        modality_profile_id: Exact modality rendering identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[BENCHMARK_INPUT_SCHEMA_VERSION] = BENCHMARK_INPUT_SCHEMA_VERSION
    sample_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=256)
    source_group_id: str = Field(min_length=1, max_length=256)
    split_group_id: str = Field(min_length=1, max_length=256)
    split: BenchmarkSplitV1
    dataset_plugin_id: str = Field(min_length=1, max_length=160)
    benchmark_profile_id: str = Field(min_length=1, max_length=160)
    dataset_source: str = Field(min_length=1, max_length=128)
    messages: list[BenchmarkMessageV1] = Field(min_length=1)
    images: list[str] = Field(default_factory=list)
    image_roles: list[str] = Field(default_factory=list)
    public_context: dict[str, Any] = Field(default_factory=dict)
    modality_profile_id: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_visible_boundary(self) -> "BenchmarkInputV1":
        """Reject assistant targets and recursively named hidden supervision."""

        if len(self.images) != len(self.image_roles):
            raise ValueError("images and image_roles must have equal length")
        leaked = sorted(_find_private_keys(self.public_context))
        if leaked:
            raise ValueError("benchmark input contains private supervision: " + ",".join(leaked))
        return self


class BenchmarkReferenceV1(BaseModel):
    """Store hidden geometry, task labels, and subgroup evidence for one case.

    Args:
        sample_id: Exact input identity.
        parent_sample_id: Underlying physical-frame identity.
        source_group_id: Stable source/session identity.
        split_group_id: Atomic split identity.
        split: Frozen split assignment.
        dataset_plugin_id: Plugin that produced the reference.
        benchmark_profile_id: Legal benchmark profile used for this row.
        dataset_source: Dataset family for grouped reports.
        trajectory: Cumulative local future trajectory with width two or three.
        trajectory_timestamps_s: Strictly increasing relative timestamps.
        trajectory_valid_mask: Valid future-point mask.
        coordinate_frame: Versioned local coordinate convention.
        arrival_label: Optional point-goal arrival classification label.
        scenario_labels: Overlap-permitted external scenario memberships.
        label_provenance: Provenance of the hidden labels.
        field_provenance: Optional field-level provenance for mixed labels.
        subgroup_metadata: Additional evaluator-only group keys.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[BENCHMARK_REFERENCE_SCHEMA_VERSION] = BENCHMARK_REFERENCE_SCHEMA_VERSION
    sample_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=256)
    source_group_id: str = Field(min_length=1, max_length=256)
    split_group_id: str = Field(min_length=1, max_length=256)
    split: BenchmarkSplitV1
    dataset_plugin_id: str = Field(min_length=1, max_length=160)
    benchmark_profile_id: str = Field(min_length=1, max_length=160)
    dataset_source: str = Field(min_length=1, max_length=128)
    trajectory: list[list[float]] = Field(min_length=1)
    trajectory_timestamps_s: list[float] = Field(min_length=1)
    trajectory_valid_mask: list[bool] = Field(min_length=1)
    coordinate_frame: str = Field(min_length=1, max_length=160)
    arrival_label: bool | None = None
    scenario_labels: tuple[str, ...] = ()
    label_provenance: LabelProvenanceV1 = "deterministic_gt"
    field_provenance: dict[str, LabelProvenanceV1] = Field(default_factory=dict)
    subgroup_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_geometry(self) -> "BenchmarkReferenceV1":
        """Require finite aligned geometry with a prefix-contiguous mask."""

        _validate_trajectory(
            self.trajectory,
            self.trajectory_timestamps_s,
            self.trajectory_valid_mask,
            name="reference",
        )
        if len(set(self.scenario_labels)) != len(self.scenario_labels):
            raise ValueError("scenario_labels must be unique")
        return self


class BenchmarkPredictionV1(BaseModel):
    """Store one successful model prediction independent of a dataset plugin.

    Args:
        sample_id: Exact frozen input identity.
        parent_sample_id: Underlying physical-frame identity.
        benchmark_profile_id: Benchmark profile accepted by the adapter.
        model_adapter_id: Adapter implementation identity.
        adaptation_mode: Training/adaptation arm reported with the result.
        prediction_geometry_id: Output coordinate geometry emitted by adapter.
        trajectory: Cumulative local predicted trajectory with width two or three.
        time_basis: Whether timestamps are explicit or reuse the reference grid.
        trajectory_timestamps_s: Explicit timestamps when ``time_basis`` is
            ``explicit``.
        arrival_score: Optional calibrated probability in ``[0, 1]``.
        input_digest: Digest of the exact model-visible input.
        metadata: Non-secret latency and adapter diagnostics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[BENCHMARK_PREDICTION_SCHEMA_VERSION] = BENCHMARK_PREDICTION_SCHEMA_VERSION
    sample_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=256)
    benchmark_profile_id: str = Field(min_length=1, max_length=160)
    model_adapter_id: str = Field(min_length=1, max_length=160)
    adaptation_mode: AdaptationModeV1
    prediction_geometry_id: str = Field(min_length=1, max_length=160)
    trajectory: list[list[float]] = Field(min_length=1)
    time_basis: PredictionTimeBasisV1
    trajectory_timestamps_s: list[float] | None = None
    arrival_score: float | None = Field(default=None, ge=0.0, le=1.0)
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prediction(self) -> "BenchmarkPredictionV1":
        """Validate explicit/reference time semantics and finite geometry."""

        if self.time_basis == "explicit":
            if self.trajectory_timestamps_s is None:
                raise ValueError("explicit prediction time basis requires timestamps")
            _validate_trajectory(
                self.trajectory,
                self.trajectory_timestamps_s,
                [True] * len(self.trajectory),
                name="prediction",
            )
        elif self.trajectory_timestamps_s is not None:
            raise ValueError("reference_schedule prediction must not carry timestamps")
        else:
            _validate_trajectory_rows(self.trajectory, name="prediction")
        return self


class BenchmarkFailureV1(BaseModel):
    """Record one model adapter attempt that did not yield a prediction.

    Args:
        sample_id: Exact frozen input identity.
        parent_sample_id: Underlying physical-frame identity.
        benchmark_profile_id: Benchmark profile attempted by the adapter.
        model_adapter_id: Adapter implementation identity.
        failure_code: Stable bounded failure category.
        detail: Bounded non-secret diagnostic.
        input_digest: Digest of the attempted model-visible input.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[BENCHMARK_FAILURE_SCHEMA_VERSION] = BENCHMARK_FAILURE_SCHEMA_VERSION
    sample_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=256)
    benchmark_profile_id: str = Field(min_length=1, max_length=160)
    model_adapter_id: str = Field(min_length=1, max_length=160)
    failure_code: str = Field(min_length=1, max_length=96, pattern=r"^[a-z][a-z0-9_]*$")
    detail: str | None = Field(default=None, max_length=512)
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkMediaManifestEntryV1(BaseModel):
    """Bind one frozen logical image path to portable verified media bytes.

    The logical path remains part of :class:`BenchmarkInputV1` and therefore
    its input digest.  ``relative_path`` is only a runtime locator below an
    explicitly supplied read-only media root; it never changes the canonical
    input or prediction identity.

    Args:
        logical_path: Exact ordered image string appearing in frozen inputs.
        relative_path: Safe POSIX path below a runtime-supplied media root.
        byte_size: Exact file size in bytes.
        sha256: SHA256 of the resolved file bytes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[BENCHMARK_MEDIA_MANIFEST_SCHEMA_VERSION] = BENCHMARK_MEDIA_MANIFEST_SCHEMA_VERSION
    logical_path: str = Field(min_length=1, max_length=2048)
    relative_path: str = Field(min_length=1, max_length=1024)
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_relative_path(self) -> "BenchmarkMediaManifestEntryV1":
        """Reject absolute, traversal, and platform-ambiguous media paths."""

        if "\\" in self.relative_path:
            raise ValueError("media relative_path must use POSIX separators")
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("media relative_path must be a safe non-empty relative path")
        return self


class BenchmarkPackageManifestV1(BaseModel):
    """Bind one immutable input/reference pair to a benchmark profile.

    Args:
        package_id: Human-readable frozen package identity.
        benchmark_profile: Full profile, including metric and time policies.
        split: Package split.
        input_path: Input JSONL path relative to this manifest.
        reference_path: Hidden reference JSONL path relative to this manifest.
        input_digest: SHA256 of the input artifact.
        reference_digest: SHA256 of the reference artifact.
        input_count: Input row count.
        reference_count: Reference row count.
        media_manifest_path: Optional manifest JSONL path relative to this
            package.  Its entries bind logical image paths to portable bytes.
        media_manifest_digest: SHA256 of the optional manifest JSONL.
        media_manifest_count: Unique media entry count in the optional
            manifest.
        source_fingerprints: Dataset/source evidence retained with the package.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[BENCHMARK_PACKAGE_SCHEMA_VERSION] = BENCHMARK_PACKAGE_SCHEMA_VERSION
    package_id: str = Field(min_length=1, max_length=200)
    benchmark_profile: BenchmarkProfileV1
    split: BenchmarkSplitV1
    input_path: str = Field(min_length=1, max_length=1024)
    reference_path: str = Field(min_length=1, max_length=1024)
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_count: int = Field(ge=1)
    reference_count: int = Field(ge=1)
    media_manifest_path: str | None = Field(default=None, min_length=1, max_length=1024)
    media_manifest_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    media_manifest_count: int | None = Field(default=None, ge=1)
    source_fingerprints: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_count(self) -> "BenchmarkPackageManifestV1":
        """Require one visible case for every hidden reference."""

        if self.input_count != self.reference_count:
            raise ValueError("benchmark package input/reference counts must match")
        media_fields = (
            self.media_manifest_path,
            self.media_manifest_digest,
            self.media_manifest_count,
        )
        if any(item is None for item in media_fields) and any(item is not None for item in media_fields):
            raise ValueError("benchmark package media manifest fields must be all present or all absent")
        return self


class BenchmarkModelCapabilitiesV1(BaseModel):
    """Declare a model adapter's legal profiles and output capabilities.

    Args:
        adapter_id: Stable adapter implementation identity.
        supported_input_profiles: Accepted input-rendering profile ids.
        prediction_geometry_ids: Output geometry semantics supported by adapter.
        supports_arrival: Whether a calibrated arrival score can be emitted.
        adaptation_modes: Legal training/adaptation identities for the adapter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[BENCHMARK_CAPABILITIES_SCHEMA_VERSION] = BENCHMARK_CAPABILITIES_SCHEMA_VERSION
    adapter_id: str = Field(min_length=1, max_length=160)
    supported_input_profiles: tuple[str, ...] = Field(min_length=1)
    prediction_geometry_ids: tuple[str, ...] = Field(min_length=1)
    supports_arrival: bool
    adaptation_modes: tuple[AdaptationModeV1, ...] = Field(min_length=1)


def benchmark_plugin_digest_v1(value: Any) -> str:
    """Return the canonical SHA256 digest of one JSON-compatible value.

    Args:
        value: JSON-compatible object or Pydantic model.

    Returns:
        Lowercase SHA256 digest.
    """

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _find_private_keys(value: Any, *, prefix: str = "") -> set[str]:
    """Return recursively named private keys in a public context payload."""

    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if name in _PRIVATE_INPUT_KEYS:
                found.add(path)
            found.update(_find_private_keys(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_find_private_keys(item, prefix=f"{prefix}[{index}]"))
    return found


def _validate_trajectory(
    trajectory: list[list[float]],
    timestamps: list[float],
    mask: list[bool],
    *,
    name: str,
) -> None:
    """Require aligned finite trajectory, time, and prefix mask arrays."""

    if len(trajectory) != len(timestamps) or len(trajectory) != len(mask):
        raise ValueError(f"{name} trajectory/timestamps/mask lengths must match")
    _validate_trajectory_rows(trajectory, name=name)
    prior = 0.0
    seen_invalid = False
    for timestamp, valid in zip(timestamps, mask):
        if not math.isfinite(float(timestamp)) or float(timestamp) <= prior:
            raise ValueError(f"{name} timestamps must be finite and strictly increasing")
        prior = float(timestamp)
        if seen_invalid and valid:
            raise ValueError(f"{name} trajectory mask must be prefix-contiguous")
        seen_invalid = seen_invalid or not valid


def _validate_trajectory_rows(trajectory: list[list[float]], *, name: str) -> None:
    """Require a finite, homogeneous local trajectory with width two or three."""

    width: int | None = None
    for index, row in enumerate(trajectory):
        if not isinstance(row, list) or len(row) not in {2, 3}:
            raise ValueError(f"{name} trajectory row {index} must have width two or three")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{name} trajectory rows must have equal width")
        if not all(math.isfinite(float(item)) for item in row):
            raise ValueError(f"{name} trajectory contains NaN or Inf")


BENCHMARK_PLUGIN_CONTRACT_DIGEST = benchmark_plugin_digest_v1(
    {
        "schema_version": BENCHMARK_PLUGIN_SCHEMA_VERSION,
        "input": BENCHMARK_INPUT_SCHEMA_VERSION,
        "reference": BENCHMARK_REFERENCE_SCHEMA_VERSION,
        "prediction": BENCHMARK_PREDICTION_SCHEMA_VERSION,
        "failure": BENCHMARK_FAILURE_SCHEMA_VERSION,
        "media_manifest": BENCHMARK_MEDIA_MANIFEST_SCHEMA_VERSION,
        "package": BENCHMARK_PACKAGE_SCHEMA_VERSION,
        "profile": BENCHMARK_PROFILE_SCHEMA_VERSION,
    }
)


__all__ = [
    "AdaptationModeV1",
    "BENCHMARK_CAPABILITIES_SCHEMA_VERSION",
    "BENCHMARK_FAILURE_SCHEMA_VERSION",
    "BENCHMARK_INPUT_SCHEMA_VERSION",
    "BENCHMARK_MEDIA_MANIFEST_SCHEMA_VERSION",
    "BENCHMARK_PACKAGE_SCHEMA_VERSION",
    "BENCHMARK_PLUGIN_CONTRACT_DIGEST",
    "BENCHMARK_PLUGIN_SCHEMA_VERSION",
    "BENCHMARK_PREDICTION_SCHEMA_VERSION",
    "BENCHMARK_PROFILE_SCHEMA_VERSION",
    "BENCHMARK_REFERENCE_SCHEMA_VERSION",
    "BenchmarkFailureV1",
    "BenchmarkInputV1",
    "BenchmarkMessageV1",
    "BenchmarkMediaManifestEntryV1",
    "BenchmarkModelCapabilitiesV1",
    "BenchmarkPackageManifestV1",
    "BenchmarkPredictionV1",
    "BenchmarkProfileV1",
    "BenchmarkReferenceV1",
    "BenchmarkSplitV1",
    "BenchmarkTimePolicyV1",
    "benchmark_plugin_digest_v1",
]
