"""Deterministic motion QC and view eligibility for trainability filtering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from mcp_server.contracts.navigation_trainability import (
    MotionEpisodeSummaryV1,
    MotionSceneAssessmentV1,
    NavigationTrainabilityFilterV1,
    NavigationTrainingEligibilityV1,
    SceneSuitabilityAssessmentV1,
    navigation_trainability_decision_id_v1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.contracts.stage_ii_cfm import StageIITrajectorySidecarV2


@dataclass(frozen=True)
class NavigationTrainabilityPolicyV1:
    """Freeze deterministic suitability thresholds and stage projections.

    Args:
        stationary_speed_mps: Maximum speed treated as stationary.
        turning_yaw_rate_radps: Yaw-rate threshold for a turning episode.
        hard_max_speed_mps: Kinematic speed above which all training views are
            rejected as a likely pose/unit artifact.
        hard_max_acceleration_mps2: Acceleration above which all views are
            rejected as a likely pose discontinuity.
        hard_max_step_m: Adjacent pose jump above which all views are rejected.
        hard_max_yaw_rate_radps: Yaw-rate above which all views are rejected.
    """

    stationary_speed_mps: float = 0.12
    turning_yaw_rate_radps: float = 0.8
    hard_max_speed_mps: float = 20.0
    hard_max_acceleration_mps2: float = 40.0
    hard_max_step_m: float = 8.0
    hard_max_yaw_rate_radps: float = 12.0


def summarize_reasoning_source_motion_v1(
    source: ReasoningSourceV4,
    *,
    policy: NavigationTrainabilityPolicyV1 | None = None,
) -> MotionEpisodeSummaryV1:
    r"""Summarize a source-owned cumulative trajectory without exposing it to students.

    Given cumulative ego points \(p_i=(x_i,y_i,\psi_i)\) and sampling interval
    \(\Delta t\), this function computes

    \[
    v_i=\frac{\|p_i^{xy}-p_{i-1}^{xy}\|_2}{\Delta t},\qquad
    a_i=\frac{|v_i-v_{i-1}|}{\Delta t}.
    \]

    The summary is a teacher-only data-QC input.  It is not copied into any
    student-facing annotation or prompt.

    Args:
        source: Canonical source with an optional existing trajectory sidecar.
        policy: Fixed kinematic thresholds used only for summary state flags.

    Returns:
        Finite motion summary, or an unavailable summary when source trajectory
        data cannot be interpreted safely.
    """

    effective_policy = policy or NavigationTrainabilityPolicyV1()
    rows = source.trajectory_ego_t
    dt_s = _trajectory_dt_s(source.trajectory_policy)
    if rows is None or len(rows) < 2 or dt_s is None:
        return _unavailable_motion_summary_v1(
            "reasoning_source_trajectory" if rows is not None else "unavailable",
            "trajectory_missing_or_dt_unavailable",
        )
    return _summarize_motion_points_v1(
        rows,
        dt_s=dt_s,
        source_kind="reasoning_source_trajectory",
        source_artifact_digest=None,
        policy=effective_policy,
    )


def summarize_stage_ii_sidecar_motion_v1(
    sidecar: StageIITrajectorySidecarV2,
    *,
    policy: NavigationTrainabilityPolicyV1 | None = None,
) -> MotionEpisodeSummaryV1:
    r"""Summarize a trusted Stage-II V2 sidecar without changing V4 source rows.

    The V2 target contains future cumulative poses at fixed interval
    ``target_dt_sec``.  The current origin is deterministically prepended so
    the first velocity is measured on \([0, \Delta t]\) rather than silently
    dropping the first target interval.

    Args:
        sidecar: Immutable accepted Stage-II V2 trajectory sidecar.
        policy: Fixed kinematic thresholds used only for summary state flags.

    Returns:
        Teacher-only external-sidecar motion summary carrying the immutable
        sidecar digest for cache invalidation and audit.
    """

    target = sidecar.trajectory_target
    rows: list[list[float]] = [[0.0, 0.0, 0.0]]
    rows.extend([list(row) for row in target.trajectory_ego_t])
    return _summarize_motion_points_v1(
        rows,
        dt_s=float(target.policy.target_dt_sec),
        source_kind="external_sidecar",
        source_artifact_digest=sidecar.sidecar_digest,
        policy=policy or NavigationTrainabilityPolicyV1(),
    )


def _summarize_motion_points_v1(
    rows: Iterable[Sequence[float]],
    *,
    dt_s: float,
    source_kind: str,
    source_artifact_digest: str | None,
    policy: NavigationTrainabilityPolicyV1,
) -> MotionEpisodeSummaryV1:
    """Compute deterministic kinematics from one trusted cumulative pose sequence."""

    try:
        points = [(float(row[0]), float(row[1]), float(row[2])) for row in rows]
    except (IndexError, TypeError, ValueError):
        return _unavailable_motion_summary_v1(source_kind, "trajectory_row_invalid")
    if len(points) < 2 or not all(all(math.isfinite(value) for value in point) for point in points):
        return _unavailable_motion_summary_v1(source_kind, "trajectory_nonfinite")

    speeds: list[float] = []
    steps: list[float] = []
    yaw_rates: list[float] = []
    for previous, current in zip(points, points[1:]):
        step = math.hypot(current[0] - previous[0], current[1] - previous[1])
        steps.append(step)
        speeds.append(step / dt_s)
        yaw_rates.append(abs(_wrapped_angle(current[2] - previous[2])) / dt_s)
    accelerations = [abs(current - previous) / dt_s for previous, current in zip(speeds, speeds[1:])]
    stationary_duration_s = _longest_stationary_duration_s(
        speeds,
        dt_s=dt_s,
        stationary_speed_mps=policy.stationary_speed_mps,
    )
    max_speed = max(speeds)
    max_acceleration = max(accelerations, default=0.0)
    max_yaw_rate = max(yaw_rates, default=0.0)
    max_step = max(steps)
    flags: list[str] = []
    if max_speed > policy.hard_max_speed_mps:
        flags.append("kinematic_speed_extreme")
    if max_acceleration > policy.hard_max_acceleration_mps2:
        flags.append("kinematic_acceleration_extreme")
    if max_step > policy.hard_max_step_m:
        flags.append("kinematic_step_extreme")
    if max_yaw_rate > policy.hard_max_yaw_rate_radps:
        flags.append("kinematic_yaw_rate_extreme")
    state = _recorded_motion_state_v1(
        speeds=speeds,
        max_yaw_rate=max_yaw_rate,
        policy=policy,
    )
    return MotionEpisodeSummaryV1(
        source_kind=source_kind,  # type: ignore[arg-type]
        source_artifact_digest=source_artifact_digest,
        available=True,
        dt_s=dt_s,
        point_count=len(points),
        duration_s=dt_s * float(len(points) - 1),
        initial_speed_mps=speeds[0],
        mean_speed_mps=sum(speeds) / len(speeds),
        max_speed_mps=max_speed,
        max_acceleration_mps2=max_acceleration,
        max_yaw_rate_radps=max_yaw_rate,
        max_step_m=max_step,
        stationary_duration_s=stationary_duration_s,
        recorded_motion_state=state,
        qc_flags=tuple(flags),
    )


def derive_navigation_trainability_v1(
    *,
    source: ReasoningSourceV4,
    motion_summary: MotionEpisodeSummaryV1,
    scene_assessment: SceneSuitabilityAssessmentV1 | None,
    motion_scene_assessment: MotionSceneAssessmentV1 | None,
    scene_profile_digest: str,
    motion_profile_digest: str,
    policy: NavigationTrainabilityPolicyV1 | None = None,
    retry_required: bool = False,
) -> NavigationTrainabilityFilterV1:
    """Compose a fail-closed training disposition from fixed policy and LVLM facts.

    LVLM outputs scene and temporal visual facts.  This deterministic composer
    alone decides which downstream view is allowed.  In particular, a model
    cannot mark a visually unsafe dense-crowd motion sequence as Stage-II
    eligible by merely emitting a permissive boolean.

    Args:
        source: Canonical source whose media remains untouched.
        motion_summary: Teacher-only deterministic kinematic summary.
        scene_assessment: Parsed current-FPV scene assessment, if successful.
        motion_scene_assessment: Parsed temporal consistency assessment, if successful.
        scene_profile_digest: Exact scene profile cache identity.
        motion_profile_digest: Exact temporal profile cache identity.
        policy: Deterministic eligibility thresholds.
        retry_required: Whether any provider/parser stage failed.

    Returns:
        Immutable accepted, quarantined, rejected, or retry-required sidecar.
    """

    effective_policy = policy or NavigationTrainabilityPolicyV1()
    if retry_required or scene_assessment is None:
        return _filter_row_v1(
            source=source,
            motion_summary=motion_summary,
            scene_assessment=None,
            motion_scene_assessment=None,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            status="pending_retry",
            eligibility=NavigationTrainingEligibilityV1(
                stage_i_decision=False,
                stage_ii_cfm=False,
                vqa=False,
            ),
            reason_codes=("lvlm_assessment_unavailable",),
        )

    reasons = list(motion_summary.qc_flags)
    if _has_hard_motion_failure_v1(motion_summary, effective_policy):
        reasons.append("hard_kinematic_qc_failure")
        return _filter_row_v1(
            source=source,
            motion_summary=motion_summary,
            scene_assessment=scene_assessment,
            motion_scene_assessment=motion_scene_assessment,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            status="rejected",
            eligibility=NavigationTrainingEligibilityV1(
                stage_i_decision=False,
                stage_ii_cfm=False,
                vqa=False,
            ),
            reason_codes=tuple(_unique_codes_v1(reasons)),
        )

    excluded_scenes = {
        "indoor_transit",
        "indoor_narrow_corridor",
        "indoor_other",
        "vehicle_interior",
        "camera_occluded",
        "non_pedestrian_or_irrelevant",
        "outdoor_but_unsuitable",
    }
    if scene_assessment.scene_class in excluded_scenes:
        reasons.extend(("scene_not_in_outdoor_pedestrian_core", scene_assessment.scene_class))
        return _filter_row_v1(
            source=source,
            motion_summary=motion_summary,
            scene_assessment=scene_assessment,
            motion_scene_assessment=motion_scene_assessment,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            status="rejected",
            eligibility=NavigationTrainingEligibilityV1(
                stage_i_decision=False,
                stage_ii_cfm=False,
                vqa=False,
            ),
            reason_codes=tuple(_unique_codes_v1(reasons)),
        )

    if scene_assessment.scene_class == "unknown":
        reasons.append("visual_scene_suitability_unknown")
        return _filter_row_v1(
            source=source,
            motion_summary=motion_summary,
            scene_assessment=scene_assessment,
            motion_scene_assessment=None,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            status="quarantine",
            eligibility=NavigationTrainingEligibilityV1(
                stage_i_decision=False,
                stage_ii_cfm=False,
                vqa=False,
            ),
            reason_codes=tuple(_unique_codes_v1(reasons)),
        )

    if motion_scene_assessment is None and not motion_summary.available:
        reasons.append("trajectory_motion_unavailable_stage_ii_disabled")
        return _filter_row_v1(
            source=source,
            motion_summary=motion_summary,
            scene_assessment=scene_assessment,
            motion_scene_assessment=None,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            status="quarantine",
            eligibility=NavigationTrainingEligibilityV1(
                stage_i_decision=True,
                stage_ii_cfm=False,
                vqa=True,
            ),
            reason_codes=tuple(_unique_codes_v1(reasons)),
        )

    if motion_scene_assessment is None:
        return _filter_row_v1(
            source=source,
            motion_summary=motion_summary,
            scene_assessment=scene_assessment,
            motion_scene_assessment=None,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            status="pending_retry",
            eligibility=NavigationTrainingEligibilityV1(
                stage_i_decision=False,
                stage_ii_cfm=False,
                vqa=False,
            ),
            reason_codes=("motion_scene_assessment_unavailable",),
        )

    if motion_scene_assessment.consistency in {
        "visual_evidence_insufficient",
        "unknown",
    }:
        reasons.extend(("visual_suitability_insufficient", motion_scene_assessment.consistency))
        return _filter_row_v1(
            source=source,
            motion_summary=motion_summary,
            scene_assessment=scene_assessment,
            motion_scene_assessment=motion_scene_assessment,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            status="quarantine",
            eligibility=NavigationTrainingEligibilityV1(
                stage_i_decision=False,
                stage_ii_cfm=False,
                vqa=False,
            ),
            reason_codes=tuple(_unique_codes_v1(reasons)),
        )

    inconsistent = {
        "likely_unexplained_stop",
        "likely_unsafe_dense_crowd_motion",
        "likely_clearance_conflict",
    }
    if motion_scene_assessment.consistency in inconsistent:
        # 社会交互视觉问答仍可能有价值，但动作和 CFM 不能学习未经解释的运动。
        reasons.extend(("motion_scene_inconsistency", motion_scene_assessment.consistency))
        return _filter_row_v1(
            source=source,
            motion_summary=motion_summary,
            scene_assessment=scene_assessment,
            motion_scene_assessment=motion_scene_assessment,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            status="quarantine",
            eligibility=NavigationTrainingEligibilityV1(
                stage_i_decision=False,
                stage_ii_cfm=False,
                vqa=True,
            ),
            reason_codes=tuple(_unique_codes_v1(reasons)),
        )

    if not motion_summary.available:
        reasons.append("trajectory_motion_unavailable_stage_ii_disabled")
        return _filter_row_v1(
            source=source,
            motion_summary=motion_summary,
            scene_assessment=scene_assessment,
            motion_scene_assessment=motion_scene_assessment,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            status="quarantine",
            eligibility=NavigationTrainingEligibilityV1(
                stage_i_decision=True,
                stage_ii_cfm=False,
                vqa=True,
            ),
            reason_codes=tuple(_unique_codes_v1(reasons)),
        )

    return _filter_row_v1(
        source=source,
        motion_summary=motion_summary,
        scene_assessment=scene_assessment,
        motion_scene_assessment=motion_scene_assessment,
        scene_profile_digest=scene_profile_digest,
        motion_profile_digest=motion_profile_digest,
        status="accepted",
        eligibility=NavigationTrainingEligibilityV1(
            stage_i_decision=True,
            stage_ii_cfm=True,
            vqa=True,
        ),
        reason_codes=tuple(_unique_codes_v1(reasons or ["outdoor_scene_motion_consistent"])),
    )


def _filter_row_v1(
    *,
    source: ReasoningSourceV4,
    motion_summary: MotionEpisodeSummaryV1,
    scene_assessment: SceneSuitabilityAssessmentV1 | None,
    motion_scene_assessment: MotionSceneAssessmentV1 | None,
    scene_profile_digest: str,
    motion_profile_digest: str,
    status: str,
    eligibility: NavigationTrainingEligibilityV1,
    reason_codes: tuple[str, ...],
) -> NavigationTrainabilityFilterV1:
    """Create one identity-stamped filter row after policy projection."""

    return NavigationTrainabilityFilterV1(
        decision_id=navigation_trainability_decision_id_v1(
            parent_sample_id=source.observation.source_group_id,
            source_fingerprint=source.source_fingerprint,
            scene_profile_digest=scene_profile_digest,
            motion_profile_digest=motion_profile_digest,
            motion_summary=motion_summary,
        ),
        parent_sample_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        dataset_source=source.observation.dataset_source,
        motion_summary=motion_summary,
        scene_assessment=scene_assessment,
        motion_scene_assessment=motion_scene_assessment,
        status=status,  # type: ignore[arg-type]
        eligibility=eligibility,
        reason_codes=reason_codes,
    )


def _trajectory_dt_s(policy: dict[str, object] | None) -> float | None:
    """Extract one safe interval from legacy and current trajectory policies."""

    if not isinstance(policy, dict):
        return None
    for key in ("dt_sec", "dt_s", "frame_dt_sec", "sample_interval_s"):
        value = policy.get(key)
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and 0.0 < numeric <= 10.0:
            return numeric
    return None


def _unavailable_motion_summary_v1(source_kind: str, flag: str) -> MotionEpisodeSummaryV1:
    """Create the unique canonical representation for unavailable kinematics."""

    return MotionEpisodeSummaryV1(
        source_kind=source_kind,  # type: ignore[arg-type]
        source_artifact_digest=None,
        available=False,
        dt_s=None,
        point_count=0,
        duration_s=None,
        initial_speed_mps=None,
        mean_speed_mps=None,
        max_speed_mps=None,
        max_acceleration_mps2=None,
        max_yaw_rate_radps=None,
        max_step_m=None,
        stationary_duration_s=None,
        recorded_motion_state="unknown",
        qc_flags=(flag,),
    )


def _wrapped_angle(value: float) -> float:
    """Normalize one yaw difference to ``[-pi, pi]``."""

    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _longest_stationary_duration_s(
    speeds: Iterable[float],
    *,
    dt_s: float,
    stationary_speed_mps: float,
) -> float:
    """Return the longest continuous low-speed interval."""

    longest = 0
    current = 0
    for speed in speeds:
        if speed <= stationary_speed_mps:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest) * dt_s


def _recorded_motion_state_v1(
    *,
    speeds: list[float],
    max_yaw_rate: float,
    policy: NavigationTrainabilityPolicyV1,
) -> str:
    """Derive one coarse state without using LVLM scene interpretation."""

    if max(speeds, default=0.0) <= policy.stationary_speed_mps:
        return "stationary"
    if max_yaw_rate >= policy.turning_yaw_rate_radps:
        return "turning"
    if speeds and speeds[-1] <= policy.stationary_speed_mps < speeds[0]:
        return "decelerating"
    return "moving"


def _has_hard_motion_failure_v1(
    summary: MotionEpisodeSummaryV1,
    policy: NavigationTrainabilityPolicyV1,
) -> bool:
    """Return whether deterministic kinematics prove a broken source row."""

    return bool(
        summary.available
        and (
            (summary.max_speed_mps or 0.0) > policy.hard_max_speed_mps
            or (summary.max_acceleration_mps2 or 0.0) > policy.hard_max_acceleration_mps2
            or (summary.max_step_m or 0.0) > policy.hard_max_step_m
            or (summary.max_yaw_rate_radps or 0.0) > policy.hard_max_yaw_rate_radps
        )
    )


def _unique_codes_v1(values: Iterable[str]) -> list[str]:
    """Return deterministic lower-snake-code order without duplicates."""

    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "NavigationTrainabilityPolicyV1",
    "derive_navigation_trainability_v1",
    "summarize_reasoning_source_motion_v1",
    "summarize_stage_ii_sidecar_motion_v1",
]
