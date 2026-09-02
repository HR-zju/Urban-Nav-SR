"""Deterministic Stage II trajectory construction and quality control."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_HORIZON_SECONDS,
    STAGE_II_HORIZON_SECONDS_V2,
    STAGE_II_HORIZON_STEPS,
    STAGE_II_HORIZON_STEPS_V2,
    STAGE_II_MAX_SOURCE_GAP_SEC_V2,
    STAGE_II_SOURCE_SPEED_AUDIT_THRESHOLD_MPS_V2,
    STAGE_II_MAX_SOURCE_STEP_M_V2,
    STAGE_II_MAX_SOURCE_YAW_RATE_RAD_S_V2,
    STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_QC_V2_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION,
    StageIITrajectoryPolicyV1,
    StageIITrajectoryPolicyV2,
    StageIITrajectoryQCReportV1,
    StageIITrajectoryQCReportV2,
    StageIITrajectoryTargetV1,
    StageIITrajectoryTargetV2,
    stage_ii_digest_v1,
)


@dataclass(frozen=True)
class StageIITrajectoryBuildResultV1:
    """Return either a validated target or one immutable rejection report.

    Args:
        target: Accepted metric target, otherwise ``None``.
        qc: Immutable accepted/rejected construction report.
    """

    target: StageIITrajectoryTargetV1 | None
    qc: StageIITrajectoryQCReportV1

    @property
    def accepted(self) -> bool:
        """Return whether deterministic target construction succeeded."""

        return self.target is not None and self.qc.status == "accepted"


@dataclass(frozen=True)
class StageIITrajectoryBuildResultV2:
    """Return one strict 4-second target or its fail-closed QC report."""

    target: StageIITrajectoryTargetV2 | None
    qc: StageIITrajectoryQCReportV2

    @property
    def accepted(self) -> bool:
        """Return whether complete-horizon V2 construction succeeded."""

        return self.target is not None and self.qc.status == "accepted"


class _TrajectoryBuildError(ValueError):
    """Carry one stable fail-closed QC reason."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def build_stage_ii_trajectory_target_v1(
    *,
    sample_id: str,
    dataset_source: str,
    raw_trajectory: Sequence[Sequence[Any]],
    action_mask: Sequence[bool | int] | None,
    source_format: str,
    trajectory_source: str,
    allowed_trajectory_sources: Iterable[str],
    yaw_source: str,
    max_source_step_m: float | None = None,
) -> StageIITrajectoryBuildResultV1:
    """Build one 2-second, 16-step metric FLU target from existing dataset GT.

    Args:
        sample_id: Stable parent sample identity.
        dataset_source: Dataset family for audit.
        raw_trajectory: Existing timed trajectory rows.  Legacy rows use
            ``[x_right, y_front, time_sec, speed]``.  Canonical timed rows use
            ``[x_forward, y_left, yaw_rel_ccw, time_sec]``.
        action_mask: Explicit raw-step validity mask; missing masks are rejected.
        source_format: Exact supported source-format stamp.
        trajectory_source: Exact dataset source code.
        allowed_trajectory_sources: Frozen caller-owned source registry.
        yaw_source: ``path_tangent`` for legacy XY or ``odometry`` for timed FLU.
        max_source_step_m: Optional caller-stamped adjacent-pose discontinuity
            gate.  The generic V1 contract keeps its historical behavior when
            this is ``None``; versioned admission profiles may enable it.

    Returns:
        Accepted metric target plus QC, or a rejection QC with no target.

    Raises:
        ValueError: If ``max_source_step_m`` is not finite and positive.
        ValidationError: If a programmer-created contract payload violates
            the canonical schema after deterministic construction.

    Notes:
        Temporal resampling occurs before legacy axis conversion so the original
        timestamps are never discarded.  GPS, route, BEV, or LVLM fallbacks are
        deliberately absent from this API.
    """

    if max_source_step_m is not None and (
        not math.isfinite(float(max_source_step_m))
        or float(max_source_step_m) <= 0.0
    ):
        raise ValueError("max_source_step_m must be finite and positive")
    raw_rows = list(raw_trajectory or [])
    source_mask = list(action_mask) if action_mask is not None else None
    source_valid_count = 0
    try:
        allowed = {str(item).strip() for item in allowed_trajectory_sources}
        if trajectory_source not in allowed:
            raise _TrajectoryBuildError(
                "trajectory_source_not_allowed",
                f"trajectory_source={trajectory_source!r} is not in the frozen registry",
            )
        lowered_source = trajectory_source.lower()
        if any(marker in lowered_source for marker in ("gps", "gnss", "route_theta", "route_match", "bev_route", "lvlm")):
            raise _TrajectoryBuildError(
                "trajectory_source_forbidden",
                "GPS/route/BEV/LVLM trajectory sources are forbidden",
            )
        if source_mask is None:
            raise _TrajectoryBuildError(
                "trajectory_mask_missing",
                "Stage II requires an explicit raw action_mask",
            )
        if len(source_mask) != len(raw_rows):
            raise _TrajectoryBuildError(
                "trajectory_mask_length_mismatch",
                "raw action_mask length must equal raw trajectory length",
            )
        valid_mask = _validate_prefix_mask(source_mask)
        source_valid_count = sum(valid_mask)
        if source_valid_count < 2:
            raise _TrajectoryBuildError(
                "trajectory_insufficient_timed_steps",
                "Stage II temporal interpolation requires at least two valid source steps",
            )
        parsed = _parse_timed_rows(
            raw_rows[:source_valid_count],
            source_format=source_format,
            yaw_source=yaw_source,
        )
        if max_source_step_m is not None:
            _validate_source_step_bound_v1(
                parsed,
                max_source_step_m=float(max_source_step_m),
            )
        target_rows, target_mask = _resample_rows(parsed, yaw_source=yaw_source)
        target_valid_count = sum(target_mask)
        source_evidence = {
            "raw_trajectory": raw_rows,
            "action_mask": valid_mask,
            "source_format": source_format,
            "trajectory_source": trajectory_source,
            "yaw_source": yaw_source,
        }
        if max_source_step_m is not None:
            source_evidence["max_source_step_m"] = float(max_source_step_m)
        source_digest = stage_ii_digest_v1(source_evidence)
        policy = StageIITrajectoryPolicyV1(
            source_format=source_format,
            trajectory_source=trajectory_source,
            yaw_source=yaw_source,
        )
        target_payload = {
            "schema_version": STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION,
            "trajectory_ego_t": target_rows,
            "action_mask": target_mask,
            "valid_step_count": target_valid_count,
            "policy": policy.model_dump(mode="json"),
            "source_target_digest": source_digest,
        }
        target_payload["target_digest"] = stage_ii_digest_v1(target_payload)
        target = StageIITrajectoryTargetV1.model_validate(target_payload)
        qc = _build_qc(
            sample_id=sample_id,
            dataset_source=dataset_source,
            trajectory_source=trajectory_source,
            source_format=source_format,
            status="accepted",
            reason_codes=("trajectory_build_accepted",),
            raw_step_count=len(raw_rows),
            source_valid_step_count=source_valid_count,
            target_valid_step_count=target_valid_count,
        )
        return StageIITrajectoryBuildResultV1(target=target, qc=qc)
    except _TrajectoryBuildError as exc:
        qc = _build_qc(
            sample_id=sample_id,
            dataset_source=dataset_source,
            trajectory_source=trajectory_source or "unknown",
            source_format=source_format or "unknown",
            status="rejected",
            reason_codes=(exc.code,),
            raw_step_count=len(raw_rows),
            source_valid_step_count=source_valid_count,
            target_valid_step_count=0,
        )
        return StageIITrajectoryBuildResultV1(target=None, qc=qc)


def build_stage_ii_trajectory_target_v2(
    *,
    sample_id: str,
    dataset_source: str,
    clip_id: str,
    frame_idx: int,
    raw_trajectory: Sequence[Sequence[Any]],
    action_mask: Sequence[bool | int] | None,
    source_adapter_id: str,
    trajectory_source: str,
    allowed_trajectory_sources: Iterable[str],
    raw_timestamp_unit: str,
    yaw_offset_deg: float,
    full_horizon_required: bool = True,
) -> StageIITrajectoryBuildResultV2:
    """Build one 4-second, 32-step metric FLU target.

    Args:
        sample_id: Stable dataset parent identity.
        dataset_source: Canonical dataset family.
        clip_id: Raw clip or trajectory identity.
        frame_idx: Current source frame index.
        raw_trajectory: Current origin plus future rows shaped
            ``[x_forward, y_left, yaw_rel_ccw, time_sec]``.
        action_mask: Explicit validity mask aligned with ``raw_trajectory``.
        source_adapter_id: Versioned raw-pose adapter identity.
        trajectory_source: Dataset-owned odometry provenance.
        allowed_trajectory_sources: Frozen caller allowlist.
        raw_timestamp_unit: Raw clock unit before conversion to seconds.
        yaw_offset_deg: Odom-to-base rotation used by the raw adapter for XY.
        full_horizon_required: Require source coverage through 4 seconds when
            true. When false, emit a zero-padded suffix and its prefix mask.

    Returns:
        Metric target plus QC, or one rejection report.

    Notes:
        V2 never extrapolates or repeats a terminal pose. The partial-mask
        capability keeps source-covered target times and zero-pads the suffix;
        the default policy still requires every target time.
    """

    raw_rows = list(raw_trajectory or [])
    source_mask = list(action_mask) if action_mask is not None else None
    source_valid_count = 0
    metrics: dict[str, float | None] = {
        "source_horizon_seconds": None,
        "max_source_gap_sec": None,
        "max_source_step_m": None,
        "max_source_speed_mps": None,
        "max_source_yaw_rate_rad_s": None,
    }
    evidence_digest: str | None = None
    try:
        allowed = {str(item).strip() for item in allowed_trajectory_sources}
        if trajectory_source not in allowed:
            raise _TrajectoryBuildError(
                "trajectory_source_not_allowed",
                f"trajectory_source={trajectory_source!r} is not in the V2 allowlist",
            )
        lowered_source = trajectory_source.lower()
        if any(
            marker in lowered_source
            for marker in (
                "gps",
                "gnss",
                "route_theta",
                "route_match",
                "bev_route",
                "lvlm",
            )
        ):
            raise _TrajectoryBuildError(
                "trajectory_source_forbidden",
                "GPS/route/BEV/LVLM trajectory sources are forbidden",
            )
        if source_mask is None:
            raise _TrajectoryBuildError(
                "trajectory_mask_missing",
                "Stage II V2 requires an explicit raw action_mask",
            )
        if len(source_mask) != len(raw_rows):
            raise _TrajectoryBuildError(
                "trajectory_mask_length_mismatch",
                "raw action_mask length must equal raw trajectory length",
            )
        valid_mask = _validate_prefix_mask(source_mask)
        source_valid_count = sum(valid_mask)
        if source_valid_count != len(raw_rows):
            raise _TrajectoryBuildError(
                "trajectory_partial_source_mask_forbidden",
                "Stage II V2 raw windows cannot contain padded source rows",
            )
        if source_valid_count < 2:
            raise _TrajectoryBuildError(
                "trajectory_insufficient_timed_steps",
                "Stage II V2 interpolation requires at least two source poses",
            )
        parsed = _parse_timed_rows(
            raw_rows,
            source_format="ego_pose_xyt_flu_timed_v1",
            yaw_source="odometry",
        )
        _validate_v2_current_origin(parsed)
        metrics = _source_motion_metrics(parsed)
        _validate_v2_source_metrics(metrics)
        target_rows, target_mask = _resample_rows_on_grid(
            parsed,
            yaw_source="odometry",
            horizon_seconds=STAGE_II_HORIZON_SECONDS_V2,
            horizon_steps=STAGE_II_HORIZON_STEPS_V2,
            full_horizon_required=bool(full_horizon_required),
        )
        evidence_digest = stage_ii_digest_v1(
            {
                "raw_trajectory": raw_rows,
                "action_mask": valid_mask,
                "source_adapter_id": source_adapter_id,
                "trajectory_source": trajectory_source,
                "raw_timestamp_unit": raw_timestamp_unit,
                "yaw_offset_deg": float(yaw_offset_deg),
            }
        )
        policy = StageIITrajectoryPolicyV2(
            source_adapter_id=source_adapter_id,
            trajectory_source=trajectory_source,
            raw_timestamp_unit=raw_timestamp_unit,
            yaw_offset_deg=float(yaw_offset_deg),
            full_horizon_required=bool(full_horizon_required),
        )
        target_payload = {
            "schema_version": STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION,
            "trajectory_ego_t": target_rows,
            "action_mask": target_mask,
            "valid_step_count": sum(target_mask),
            "policy": policy.model_dump(mode="json"),
            "source_target_digest": evidence_digest,
        }
        target_payload["target_digest"] = stage_ii_digest_v1(target_payload)
        target = StageIITrajectoryTargetV2.model_validate(target_payload)
        reason_codes = ["trajectory_build_accepted"]
        if (
            float(metrics["max_source_speed_mps"])
            > STAGE_II_SOURCE_SPEED_AUDIT_THRESHOLD_MPS_V2 + 1e-9
        ):
            reason_codes.append("trajectory_source_speed_audit_exceeded")
        qc = _build_qc_v2(
            sample_id=sample_id,
            dataset_source=dataset_source,
            source_adapter_id=source_adapter_id,
            trajectory_source=trajectory_source,
            clip_id=clip_id,
            frame_idx=frame_idx,
            status="accepted",
            reason_codes=tuple(reason_codes),
            raw_pose_count=len(raw_rows),
            source_valid_step_count=source_valid_count,
            target_valid_step_count=sum(target_mask),
            source_evidence_digest=evidence_digest,
            **metrics,
        )
        return StageIITrajectoryBuildResultV2(target=target, qc=qc)
    except _TrajectoryBuildError as exc:
        qc = _build_qc_v2(
            sample_id=sample_id,
            dataset_source=dataset_source,
            source_adapter_id=source_adapter_id or "unknown_adapter",
            trajectory_source=trajectory_source or "unknown",
            clip_id=clip_id or "unknown",
            frame_idx=max(int(frame_idx), 0),
            status="rejected",
            reason_codes=(exc.code,),
            raw_pose_count=len(raw_rows),
            source_valid_step_count=source_valid_count,
            target_valid_step_count=0,
            source_evidence_digest=evidence_digest,
            **metrics,
        )
        return StageIITrajectoryBuildResultV2(target=None, qc=qc)


def _validate_prefix_mask(values: Sequence[bool | int]) -> list[bool]:
    """Parse an explicit binary prefix mask without silently resizing it."""

    parsed: list[bool] = []
    seen_padding = False
    for index, value in enumerate(values):
        if isinstance(value, bool):
            valid = value
        elif isinstance(value, int) and value in {0, 1}:
            valid = bool(value)
        else:
            raise _TrajectoryBuildError(
                "trajectory_mask_not_binary",
                f"action_mask[{index}] must be bool or 0/1",
            )
        if seen_padding and valid:
            raise _TrajectoryBuildError(
                "trajectory_mask_not_prefix_contiguous",
                "raw action_mask contains a valid step after padding",
            )
        seen_padding = seen_padding or not valid
        parsed.append(valid)
    return parsed


def _parse_timed_rows(
    rows: Sequence[Sequence[Any]],
    *,
    source_format: str,
    yaw_source: str,
) -> list[tuple[float, float, float | None, float]]:
    """Parse supported timed rows into FLU coordinates plus source time."""

    if source_format not in {
        "ego_xy_t_speed_right_front_v1",
        "ego_pose_xyt_flu_timed_v1",
    }:
        raise _TrajectoryBuildError(
            "trajectory_source_format_unsupported",
            f"unsupported source_format={source_format!r}",
        )
    expected_yaw = "path_tangent" if source_format == "ego_xy_t_speed_right_front_v1" else "odometry"
    if yaw_source != expected_yaw:
        raise _TrajectoryBuildError(
            "trajectory_yaw_source_mismatch",
            f"{source_format} requires yaw_source={expected_yaw}",
        )
    parsed: list[tuple[float, float, float | None, float]] = []
    previous_time = -math.inf
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise _TrajectoryBuildError(
                "trajectory_row_invalid",
                f"raw trajectory row {index} must have width 4",
            )
        numbers = [_finite_float(item, index=index) for item in row]
        if source_format == "ego_xy_t_speed_right_front_v1":
            x_right, y_front, time_sec, _speed = numbers
            x_forward, y_left, yaw = y_front, -x_right, None
        else:
            x_forward, y_left, yaw, time_sec = numbers
            yaw = _wrap_angle(yaw)
        if time_sec < 0.0 or time_sec <= previous_time:
            raise _TrajectoryBuildError(
                "trajectory_time_not_strictly_increasing",
                "valid source timestamps must be finite, non-negative, and strictly increasing",
            )
        parsed.append((x_forward, y_left, yaw, time_sec))
        previous_time = time_sec
    first_target = STAGE_II_HORIZON_SECONDS / STAGE_II_HORIZON_STEPS
    if parsed[0][3] > first_target + 1e-9:
        raise _TrajectoryBuildError(
            "trajectory_initial_horizon_missing",
            "source trajectory starts after the first Stage II target timestamp",
        )
    return parsed


def _resample_rows(
    rows: Sequence[tuple[float, float, float | None, float]],
    *,
    yaw_source: str,
) -> tuple[list[list[float]], list[bool]]:
    """Interpolate onto the fixed future grid and zero-pad the invalid suffix."""

    return _resample_rows_on_grid(
        rows,
        yaw_source=yaw_source,
        horizon_seconds=STAGE_II_HORIZON_SECONDS,
        horizon_steps=STAGE_II_HORIZON_STEPS,
        full_horizon_required=False,
    )


def _resample_rows_on_grid(
    rows: Sequence[tuple[float, float, float | None, float]],
    *,
    yaw_source: str,
    horizon_seconds: float,
    horizon_steps: int,
    full_horizon_required: bool,
) -> tuple[list[list[float]], list[bool]]:
    """Interpolate onto one explicit grid without temporal extrapolation."""

    times = [row[3] for row in rows]
    target_times = [
        float(horizon_seconds) * (index + 1) / int(horizon_steps)
        for index in range(int(horizon_steps))
    ]
    if full_horizon_required and times[-1] + 1e-9 < target_times[-1]:
        raise _TrajectoryBuildError(
            "trajectory_full_horizon_missing",
            "source trajectory does not cover the complete target horizon",
        )
    positions: list[tuple[float, float]] = []
    interpolated_yaws: list[float] = []
    for target_time in target_times:
        if target_time > times[-1] + 1e-9:
            break
        right = bisect.bisect_left(times, target_time)
        if right < len(times) and abs(times[right] - target_time) <= 1e-9:
            left = right
        else:
            left = max(0, right - 1)
        if right >= len(times):
            break
        if left == right:
            alpha = 0.0
        else:
            span = times[right] - times[left]
            alpha = (target_time - times[left]) / span
        x = _lerp(rows[left][0], rows[right][0], alpha)
        y = _lerp(rows[left][1], rows[right][1], alpha)
        positions.append((x, y))
        if yaw_source == "odometry":
            left_yaw = float(rows[left][2])
            right_yaw = float(rows[right][2])
            delta = _wrap_angle(right_yaw - left_yaw)
            interpolated_yaws.append(_wrap_angle(left_yaw + alpha * delta))
    if not positions:
        raise _TrajectoryBuildError(
            "trajectory_horizon_empty",
            "source trajectory does not cover any Stage II target timestamp",
        )
    if yaw_source == "path_tangent":
        interpolated_yaws = _path_tangent_yaws(positions)
    target = [
        [float(x), float(y), float(yaw)]
        for (x, y), yaw in zip(positions, interpolated_yaws)
    ]
    valid_count = len(target)
    if full_horizon_required and valid_count != int(horizon_steps):
        raise _TrajectoryBuildError(
            "trajectory_full_horizon_missing",
            "interpolation did not produce every required target step",
        )
    target.extend([[0.0, 0.0, 0.0] for _ in range(int(horizon_steps) - valid_count)])
    mask = [True] * valid_count + [False] * (int(horizon_steps) - valid_count)
    return target, mask


def _source_motion_metrics(
    rows: Sequence[tuple[float, float, float | None, float]],
) -> dict[str, float]:
    """Measure adjacent raw-pose gaps, speed, and yaw rate."""

    gaps: list[float] = []
    steps: list[float] = []
    speeds: list[float] = []
    yaw_rates: list[float] = []
    for left, right in zip(rows, rows[1:]):
        dt = float(right[3] - left[3])
        distance = math.hypot(float(right[0] - left[0]), float(right[1] - left[1]))
        yaw_delta = abs(_wrap_angle(float(right[2]) - float(left[2])))
        gaps.append(dt)
        steps.append(distance)
        speeds.append(distance / dt)
        yaw_rates.append(yaw_delta / dt)
    return {
        "source_horizon_seconds": float(rows[-1][3]),
        "max_source_gap_sec": max(gaps, default=0.0),
        "max_source_step_m": max(steps, default=0.0),
        "max_source_speed_mps": max(speeds, default=0.0),
        "max_source_yaw_rate_rad_s": max(yaw_rates, default=0.0),
    }


def _validate_source_step_bound_v1(
    rows: Sequence[tuple[float, float, float | None, float]],
    *,
    max_source_step_m: float,
) -> None:
    """Reject one V1 source window containing an impossible pose jump."""

    max_step = max(
        (
            math.hypot(float(right[0] - left[0]), float(right[1] - left[1]))
            for left, right in zip(rows, rows[1:])
        ),
        default=0.0,
    )
    if max_step > float(max_source_step_m) + 1e-9:
        raise _TrajectoryBuildError(
            "trajectory_source_step_exceeded",
            f"max_source_step_m={max_step:.6f} exceeds admission threshold="
            f"{float(max_source_step_m):.6f}",
        )


def _validate_v2_current_origin(
    rows: Sequence[tuple[float, float, float | None, float]],
) -> None:
    """Require the raw window to start at the current FLU pose and time zero."""

    x_forward, y_left, yaw_relative, time_sec = rows[0]
    if abs(float(time_sec)) > 1e-9:
        raise _TrajectoryBuildError(
            "trajectory_current_origin_time_missing",
            "Stage II V2 raw windows must include the current pose at t=0",
        )
    if (
        abs(float(x_forward)) > 1e-6
        or abs(float(y_left)) > 1e-6
        or abs(float(yaw_relative)) > 1e-6
    ):
        raise _TrajectoryBuildError(
            "trajectory_current_origin_pose_invalid",
            "Stage II V2 raw windows must begin at FLU pose [0,0,0]",
        )


def _validate_v2_source_metrics(metrics: dict[str, float | None]) -> None:
    """Apply discontinuity gates while leaving source speed audit-only."""

    checks = (
        (
            "max_source_gap_sec",
            STAGE_II_MAX_SOURCE_GAP_SEC_V2,
            "trajectory_source_gap_exceeded",
        ),
        (
            "max_source_step_m",
            STAGE_II_MAX_SOURCE_STEP_M_V2,
            "trajectory_source_step_exceeded",
        ),
        (
            "max_source_yaw_rate_rad_s",
            STAGE_II_MAX_SOURCE_YAW_RATE_RAD_S_V2,
            "trajectory_source_yaw_rate_exceeded",
        ),
    )
    for field, threshold, reason in checks:
        value = float(metrics[field])
        if value > float(threshold) + 1e-9:
            raise _TrajectoryBuildError(
                reason,
                f"{field}={value:.6f} exceeds frozen threshold={threshold:.6f}",
            )


def _path_tangent_yaws(positions: Sequence[tuple[float, float]]) -> list[float]:
    """Construct deterministic path-tangent yaw relative to the current origin."""

    result: list[float] = []
    previous_x = 0.0
    previous_y = 0.0
    previous_yaw = 0.0
    for x, y in positions:
        dx = x - previous_x
        dy = y - previous_y
        if abs(dx) + abs(dy) > 1e-8:
            previous_yaw = math.atan2(dy, dx)
        result.append(_wrap_angle(previous_yaw))
        previous_x, previous_y = x, y
    return result


def _finite_float(value: Any, *, index: int) -> float:
    """Convert one raw coordinate while preserving a bounded QC error."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _TrajectoryBuildError(
            "trajectory_row_invalid",
            f"raw trajectory row {index} contains a non-numeric value",
        ) from exc
    if not math.isfinite(number):
        raise _TrajectoryBuildError(
            "trajectory_row_non_finite",
            f"raw trajectory row {index} contains NaN or Inf",
        )
    return number


def _lerp(left: float, right: float, alpha: float) -> float:
    """Return deterministic scalar linear interpolation."""

    return float(left + (right - left) * alpha)


def _wrap_angle(angle: float) -> float:
    """Wrap one angle into ``[-pi, pi)``."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _build_qc(**payload: Any) -> StageIITrajectoryQCReportV1:
    """Attach the deterministic QC digest and validate the report."""

    normalized = {
        "schema_version": STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION,
        **payload,
    }
    normalized["qc_digest"] = stage_ii_digest_v1(normalized)
    return StageIITrajectoryQCReportV1.model_validate(normalized)


def build_stage_ii_trajectory_qc_v2(**payload: Any) -> StageIITrajectoryQCReportV2:
    """Attach the canonical digest to a V2 raw-source QC payload."""

    normalized = {
        "schema_version": STAGE_II_TRAJECTORY_QC_V2_SCHEMA_VERSION,
        **payload,
    }
    normalized["qc_digest"] = stage_ii_digest_v1(normalized)
    return StageIITrajectoryQCReportV2.model_validate(normalized)


def _build_qc_v2(**payload: Any) -> StageIITrajectoryQCReportV2:
    """Build one internal V2 QC report."""

    return build_stage_ii_trajectory_qc_v2(**payload)


__all__ = [
    "StageIITrajectoryBuildResultV1",
    "StageIITrajectoryBuildResultV2",
    "build_stage_ii_trajectory_qc_v2",
    "build_stage_ii_trajectory_target_v1",
    "build_stage_ii_trajectory_target_v2",
]
