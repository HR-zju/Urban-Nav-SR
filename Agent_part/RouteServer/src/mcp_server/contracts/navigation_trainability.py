"""Typed post-annotation navigation dataset suitability contracts.

The filter in this module is not an interestingness selector and does not
modify source media.  It records scene applicability and visual consistency of
recorded motion, then projects explicit eligibility for Stage I, Stage II, and
VQA views.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


NAVIGATION_TRAINABILITY_FILTER_SCHEMA_VERSION_V1 = "urbannav_navigation_trainability_filter_v1"
MOTION_EPISODE_SUMMARY_SCHEMA_VERSION_V1 = "urbannav_motion_episode_summary_v1"

TrainabilityFilterStatusV1 = Literal["accepted", "quarantine", "rejected", "pending_retry"]
SceneSuitabilityClassV1 = Literal[
    "outdoor_pedestrian_supported",
    "outdoor_but_unsuitable",
    "indoor_transit",
    "indoor_narrow_corridor",
    "indoor_other",
    "vehicle_interior",
    "camera_occluded",
    "non_pedestrian_or_irrelevant",
    "unknown",
]
RecordedMotionStateV1 = Literal["moving", "stationary", "decelerating", "turning", "unknown"]
MotionSceneConsistencyV1 = Literal[
    "consistent",
    "likely_unexplained_stop",
    "likely_unsafe_dense_crowd_motion",
    "likely_clearance_conflict",
    "visual_evidence_insufficient",
    "unknown",
]
CrowdDensityV1 = Literal["none", "sparse", "moderate", "dense", "unknown"]
CorridorOccupancyV1 = Literal["clear", "shared", "blocked", "unknown"]
VisibleStopCueV1 = Literal[
    "red_light",
    "crossing_wait",
    "pedestrian_yield",
    "obstacle",
    "goal_arrival",
    "none",
    "unknown",
]
AssessmentConfidenceV1 = Literal["high", "medium", "low"]


class MotionEpisodeSummaryV1(BaseModel):
    """Deterministic kinematic summary kept outside student-visible prompts.

    Args:
        source_kind: Origin of the trajectory sequence used for QC.
        source_artifact_digest: Digest of an immutable external trajectory
            sidecar when one supplies the motion sequence.  This keeps motion
            cache invalidation explicit without adding trajectory values back
            into the canonical V4 source row.
        available: Whether finite time-aligned trajectory data was available.
        dt_s: Sampling interval, if available.
        point_count: Number of cumulative ego-frame points summarized.
        duration_s: Total observed duration.
        initial_speed_mps: First finite speed estimate.
        mean_speed_mps: Mean finite inter-point speed.
        max_speed_mps: Maximum finite inter-point speed.
        max_acceleration_mps2: Maximum absolute speed derivative.
        max_yaw_rate_radps: Maximum yaw-rate estimate when headings exist.
        max_step_m: Largest adjacent XY displacement.
        stationary_duration_s: Longest continuous near-zero-speed duration.
        recorded_motion_state: Coarse deterministic episode state.
        qc_flags: Bounded machine-generated motion findings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[MOTION_EPISODE_SUMMARY_SCHEMA_VERSION_V1] = (
        MOTION_EPISODE_SUMMARY_SCHEMA_VERSION_V1
    )
    source_kind: Literal["reasoning_source_trajectory", "external_sidecar", "unavailable"]
    source_artifact_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    available: bool
    dt_s: float | None = Field(default=None, gt=0.0, le=10.0)
    point_count: int = Field(ge=0, le=4096)
    duration_s: float | None = Field(default=None, ge=0.0, le=3600.0)
    initial_speed_mps: float | None = Field(default=None, ge=0.0, le=100.0)
    mean_speed_mps: float | None = Field(default=None, ge=0.0, le=100.0)
    max_speed_mps: float | None = Field(default=None, ge=0.0, le=100.0)
    max_acceleration_mps2: float | None = Field(default=None, ge=0.0, le=200.0)
    max_yaw_rate_radps: float | None = Field(default=None, ge=0.0, le=100.0)
    max_step_m: float | None = Field(default=None, ge=0.0, le=100.0)
    stationary_duration_s: float | None = Field(default=None, ge=0.0, le=3600.0)
    recorded_motion_state: RecordedMotionStateV1
    qc_flags: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_availability(self) -> "MotionEpisodeSummaryV1":
        """Require complete core statistics only for available motion."""

        statistics = (
            self.dt_s,
            self.duration_s,
            self.initial_speed_mps,
            self.mean_speed_mps,
            self.max_speed_mps,
            self.max_acceleration_mps2,
            self.max_step_m,
            self.stationary_duration_s,
        )
        if self.available:
            if self.source_kind == "unavailable" or self.point_count < 2 or any(value is None for value in statistics):
                raise ValueError("available_motion_summary_requires_complete_core_statistics")
        elif any(value is not None for value in statistics) or self.point_count != 0:
            raise ValueError("unavailable_motion_summary_must_not_carry_statistics")
        if not self.available and self.source_artifact_digest is not None:
            raise ValueError("unavailable_motion_summary_must_not_reference_sidecar")
        for value in statistics:
            if value is not None and not math.isfinite(value):
                raise ValueError("motion_summary_statistics_must_be_finite")
        return self


class SceneSuitabilityAssessmentV1(BaseModel):
    """LVLM assessment of whether the current scene fits the outdoor corpus.

    Args:
        scene_class: Visible scene category, not an action label.
        confidence: Teacher confidence in visible scene evidence.
        evidence_refs: Current-frame references that support the decision.
        reason_codes: Compact visible scene findings.
        teacher_model: Provider model identity.
        prompt_profile_id: Frozen scene-profile identity.
        prompt_profile_digest: Digest of scene prompt and schema.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_class: SceneSuitabilityClassV1
    confidence: AssessmentConfidenceV1
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=5)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=8)
    teacher_model: str = Field(min_length=1, max_length=256)
    prompt_profile_id: str = Field(min_length=1, max_length=160)
    prompt_profile_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_current_fpv_grounding(self) -> "SceneSuitabilityAssessmentV1":
        """Keep scene rejection grounded in current visible evidence."""

        if "fpv.current" not in self.evidence_refs:
            raise ValueError("scene_suitability_requires_fpv_current_evidence")
        return self


class MotionSceneAssessmentV1(BaseModel):
    """LVLM comparison between temporal FPV evidence and recorded motion.

    Args:
        consistency: Whether visible evidence explains the recorded motion.
        crowd_density_near_path: Visible local pedestrian density near travel.
        corridor_occupancy: Visible occupancy of the near-field corridor.
        visible_stop_cue: Visible explanation for a stationary episode.
        confidence: Teacher confidence in the temporal visual assessment.
        evidence_refs: Ordered FPV evidence references supporting the result.
        reason_codes: Compact discrepancy or confirmation codes.
        teacher_model: Provider model identity.
        prompt_profile_id: Frozen temporal-profile identity.
        prompt_profile_digest: Digest of temporal prompt and schema.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    consistency: MotionSceneConsistencyV1
    crowd_density_near_path: CrowdDensityV1
    corridor_occupancy: CorridorOccupancyV1
    visible_stop_cue: VisibleStopCueV1
    confidence: AssessmentConfidenceV1
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=8)
    teacher_model: str = Field(min_length=1, max_length=256)
    prompt_profile_id: str = Field(min_length=1, max_length=160)
    prompt_profile_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_temporal_grounding(self) -> "MotionSceneAssessmentV1":
        """Require current FPV evidence even when history frames are visible."""

        if "fpv.current" not in self.evidence_refs:
            raise ValueError("motion_scene_assessment_requires_fpv_current_evidence")
        return self


class NavigationTrainingEligibilityV1(BaseModel):
    """Explicit downstream eligibility projection without source mutation.

    Args:
        stage_i_decision: Eligibility for language/reasoning decision views.
        stage_ii_cfm: Eligibility for continuous trajectory supervision.
        vqa: Eligibility for non-action VQA views.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_i_decision: bool
    stage_ii_cfm: bool
    vqa: bool


class NavigationTrainabilityFilterV1(BaseModel):
    """One immutable post-annotation data-suitability decision sidecar.

    Args:
        decision_id: Content-addressed filter result identity.
        parent_sample_id: Stable physical source parent.
        source_fingerprint: Exact source revision.
        dataset_source: Dataset family.
        motion_summary: Deterministic teacher-only kinematic summary.
        scene_assessment: Current-frame LVLM scene decision.
        motion_scene_assessment: Temporal LVLM consistency decision.
        status: Accepted, quarantined, rejected, or retry-needed disposition.
        eligibility: Deterministic Stage I/II/VQA view projection.
        reason_codes: Final machine-composed reasons for the disposition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[NAVIGATION_TRAINABILITY_FILTER_SCHEMA_VERSION_V1] = (
        NAVIGATION_TRAINABILITY_FILTER_SCHEMA_VERSION_V1
    )
    decision_id: str = Field(min_length=1, max_length=160)
    parent_sample_id: str = Field(min_length=1, max_length=480)
    source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    dataset_source: str = Field(min_length=1, max_length=128)
    motion_summary: MotionEpisodeSummaryV1
    scene_assessment: SceneSuitabilityAssessmentV1 | None = None
    motion_scene_assessment: MotionSceneAssessmentV1 | None = None
    status: TrainabilityFilterStatusV1
    eligibility: NavigationTrainingEligibilityV1
    reason_codes: tuple[str, ...] = Field(default=(), max_length=16)
    warnings: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_disposition(self) -> "NavigationTrainabilityFilterV1":
        """Prevent retry/reject rows from silently entering a training view."""

        any_eligible = any(
            (self.eligibility.stage_i_decision, self.eligibility.stage_ii_cfm, self.eligibility.vqa)
        )
        if self.status in {"rejected", "pending_retry"} and any_eligible:
            raise ValueError("rejected_or_retry_filter_row_must_not_be_training_eligible")
        if self.status == "accepted":
            if self.scene_assessment is None or self.motion_scene_assessment is None:
                raise ValueError("accepted_filter_row_requires_both_lvlm_assessments")
            if not (self.eligibility.stage_i_decision and self.eligibility.stage_ii_cfm):
                raise ValueError("accepted_filter_row_requires_stage_i_and_stage_ii_eligibility")
        if self.status == "pending_retry" and (
            self.scene_assessment is not None or self.motion_scene_assessment is not None
        ):
            raise ValueError("pending_retry_filter_row_must_not_mix_partial_assessments")
        return self


def navigation_trainability_decision_id_v1(
    *,
    parent_sample_id: str,
    source_fingerprint: str,
    scene_profile_digest: str,
    motion_profile_digest: str,
    motion_summary: MotionEpisodeSummaryV1,
) -> str:
    """Return a cache-safe identity for one trainability decision.

    Args:
        parent_sample_id: Stable physical parent identity.
        source_fingerprint: Exact source revision.
        scene_profile_digest: Scene teacher prompt/schema digest.
        motion_profile_digest: Motion teacher prompt/schema digest.
        motion_summary: Deterministic summary used by the temporal LVLM pass.

    Returns:
        Content-addressed decision identity.  The motion summary is included so
        corrected pose data cannot accidentally reuse an old filter outcome.
    """

    payload = {
        "parent_sample_id": parent_sample_id,
        "source_fingerprint": source_fingerprint,
        "scene_profile_digest": scene_profile_digest,
        "motion_profile_digest": motion_profile_digest,
        "motion_summary": motion_summary.model_dump(mode="json"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"navigation_trainability:{digest}"


__all__ = [
    "AssessmentConfidenceV1",
    "CorridorOccupancyV1",
    "CrowdDensityV1",
    "MOTION_EPISODE_SUMMARY_SCHEMA_VERSION_V1",
    "MotionEpisodeSummaryV1",
    "MotionSceneAssessmentV1",
    "MotionSceneConsistencyV1",
    "NAVIGATION_TRAINABILITY_FILTER_SCHEMA_VERSION_V1",
    "NavigationTrainabilityFilterV1",
    "NavigationTrainingEligibilityV1",
    "RecordedMotionStateV1",
    "SceneSuitabilityAssessmentV1",
    "SceneSuitabilityClassV1",
    "TrainabilityFilterStatusV1",
    "VisibleStopCueV1",
    "navigation_trainability_decision_id_v1",
]
