"""Public-only contract for running a native CityWalker model on UrbanNav.

The reference CityWalker dataset estimates ``step_scale`` from a complete
source trajectory.  That estimator is not legal when CityWalker is evaluated
on UrbanNav because it can include future positions.  This contract therefore
defines the only admitted reverse-adapter input: five explicitly associated
public FPV/history pairs, one public local goal, and a robust scale computed
from those historical positions alone.

This file is mirrored verbatim into ``ms-swift``.  Keep it dependency-light so
both repositories use exactly the same coordinate and admission rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1 = "citywalker_to_urbannav_adapter_contract_v1"
CITYWALKER_TO_URBANNAV_ADAPTER_ID_V1 = "citywalker_to_urbannav_native_v1"
CITYWALKER_TO_URBANNAV_ADAPTER_VERSION_V1 = "1.0"
URBANNAV_CITYWALKER_NATIVE_INPUT_PROFILE_ID_V1 = "urbannav_citywalker_native_5fpv_public_history_goal_v1"
URBANNAV_CURRENT_EGO_FLU_COORDINATE_FRAME_V1 = "current_ego_x_forward_y_left_v1"
URBANNAV_CURRENT_EGO_FLU_PREDICTION_GEOMETRY_ID_V1 = "urbannav_current_ego_flu_xy_v1"
CITYWALKER_CURRENT_POSE_XY_COORDINATE_FRAME_V1 = "citywalker_current_pose_x_right_y_front_v1"
CITYWALKER_TO_URBANNAV_PUBLIC_HISTORY_TRANSFORM_ID_V1 = "citywalker_input_target_history_rotation_v1"
CITYWALKER_TO_URBANNAV_PUBLIC_SCALE_ESTIMATOR_ID_V1 = "public_history_median_step_v1"
CITYWALKER_TO_URBANNAV_FLU_TO_NATIVE_PROJECTION_ID_V1 = "urbannav_flu_to_citywalker_xright_yfront_v1"
CITYWALKER_TO_URBANNAV_NATIVE_TO_FLU_PROJECTION_ID_V1 = "citywalker_xright_yfront_to_urbannav_flu_v1"
CITYWALKER_TO_URBANNAV_NATIVE_OUTPUT_SCHEDULE_ID_V1 = "citywalker_native_five_waypoint_1hz_v1"
CITYWALKER_TO_URBANNAV_NATIVE_OUTPUT_TIMESTAMPS_S_V1 = (1.0, 2.0, 3.0, 4.0,
                                                        5.0)
CITYWALKER_TO_URBANNAV_REQUIRED_FPV_COUNT_V1 = 5
CITYWALKER_TO_URBANNAV_MIN_PUBLIC_STEP_M_V1 = 1e-2

CityWalkerToUrbanNavEligibilityStatusV1 = Literal[
    "eligible",
    "input_ineligible",
    "native_model_error",
    "output_contract_error",
]
CityWalkerToUrbanNavRejectReasonV1 = Literal[
    "insufficient_public_fpv",
    "missing_public_history",
    "missing_public_goal",
    "invalid_scale",
    "unsupported_public_input",
    "native_model_error",
    "output_contract_error",
]

_INPUT_REJECT_REASONS = frozenset({
    "insufficient_public_fpv",
    "missing_public_history",
    "missing_public_goal",
    "invalid_scale",
    "unsupported_public_input",
})


class CityWalkerToUrbanNavInputErrorV1(ValueError):
    """Report one closed public-input rejection reason.

    Args:
        reason: Closed eligibility-funnel reason.
    """

    def __init__(self, reason: CityWalkerToUrbanNavRejectReasonV1) -> None:
        """Store the closed reason alongside a readable exception message."""

        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class CityWalkerToUrbanNavPreparedInputV1:
    """One native CityWalker request derived only from public UrbanNav input.

    Args:
        selected_image_indices: Five original image positions in chronological order.
        selected_image_roles: Public roles aligned to the selected images.
        selected_images: Five public FPV paths passed to the native model.
        history_flu_xy_m: Five corresponding public local-history positions.
        target_flu_xy_m: Public local point goal in UrbanNav FLU.
        normalized_native_input_xy: Five rotated history rows plus an unrotated
            native target, all divided by ``public_scale_value``.
        public_scale_value: Median non-zero historical step length in metres.
    """

    selected_image_indices: tuple[int, ...]
    selected_image_roles: tuple[str, ...]
    selected_images: tuple[str, ...]
    history_flu_xy_m: tuple[tuple[float, float], ...]
    target_flu_xy_m: tuple[float, float]
    normalized_native_input_xy: tuple[tuple[float, float], ...]
    public_scale_value: float


class CityWalkerToUrbanNavAdapterTraceV1(BaseModel):
    """Public-only trace sidecar for a native CityWalker reverse-adapter run.

    The sidecar is intentionally independent of canonical scorecards.  It has
    no free-form metadata, hidden reference timestamps, future trajectory, or
    evaluator loss fields.

    Args:
        sample_id: Frozen benchmark input identity.
        benchmark_input_digest: Digest of the exact public input row.
        native_adapter_id: Adapter implementation stamp.
        native_adapter_version: Adapter behavior version.
        native_checkpoint_digest: SHA256 of the native checkpoint.
        selected_image_indices: Original input indices passed to CityWalker.
        selected_image_roles: Roles aligned to those indices.
        public_history_transform_id: Public history transform stamp.
        public_scale_estimator_id: Public-only scale estimator stamp.
        public_scale_value: Estimated scale in metres.
        input_coordinate_projection_id: UrbanNav-FLU to CityWalker-xy stamp.
        output_coordinate_projection_id: CityWalker-xy to UrbanNav-FLU stamp.
        native_output_schedule_id: Explicit five-waypoint time schedule stamp.
        native_output_count: Native point count actually produced.
        eligibility_status: Closed adapter result status.
        reject_reason: Closed first rejection reason when not eligible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1] = (
            CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1)
    sample_id: str = Field(min_length=1, max_length=256)
    benchmark_input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_adapter_id: str = Field(min_length=1, max_length=160)
    native_adapter_version: str = Field(min_length=1, max_length=80)
    native_checkpoint_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_image_indices: tuple[int, ...] = ()
    selected_image_roles: tuple[str, ...] = ()
    public_history_transform_id: str | None = Field(default=None,
                                                    max_length=160)
    public_scale_estimator_id: str | None = Field(default=None, max_length=160)
    public_scale_value: float | None = None
    input_coordinate_projection_id: str | None = Field(default=None,
                                                       max_length=160)
    output_coordinate_projection_id: str | None = Field(default=None,
                                                        max_length=160)
    native_output_schedule_id: str | None = Field(default=None, max_length=160)
    native_output_count: int = Field(ge=0, le=256)
    eligibility_status: CityWalkerToUrbanNavEligibilityStatusV1
    reject_reason: CityWalkerToUrbanNavRejectReasonV1 | None = None

    @model_validator(mode="after")
    def validate_trace(self) -> "CityWalkerToUrbanNavAdapterTraceV1":
        """Require complete successful evidence and closed rejected states."""

        if len(self.selected_image_indices) != len(self.selected_image_roles):
            raise ValueError("selected image indices and roles must align")
        if any(index < 0 for index in self.selected_image_indices) or len(
                set(self.selected_image_indices)) != len(
                    self.selected_image_indices):
            raise ValueError(
                "selected image indices must be unique and non-negative")
        if self.public_scale_value is not None and (not math.isfinite(
                self.public_scale_value) or self.public_scale_value <= 0.0):
            raise ValueError("public_scale_value must be finite and positive")
        if self.eligibility_status == "eligible":
            required = (
                self.public_history_transform_id,
                self.public_scale_estimator_id,
                self.public_scale_value,
                self.input_coordinate_projection_id,
                self.output_coordinate_projection_id,
                self.native_output_schedule_id,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "eligible trace is missing public adapter evidence")
            if not self.selected_image_indices or self.native_output_count <= 0:
                raise ValueError(
                    "eligible trace must record selected public media and native output"
                )
            if self.reject_reason is not None:
                raise ValueError("eligible trace cannot carry a reject reason")
        elif self.reject_reason is None:
            raise ValueError("rejected trace requires a closed reject reason")
        elif self.eligibility_status == "input_ineligible" and self.reject_reason not in _INPUT_REJECT_REASONS:
            raise ValueError(
                "input_ineligible trace requires a public-input reject_reason")
        elif self.eligibility_status == "native_model_error" and self.reject_reason != "native_model_error":
            raise ValueError(
                "native_model_error trace requires native_model_error reject_reason"
            )
        elif self.eligibility_status == "output_contract_error" and self.reject_reason != "output_contract_error":
            raise ValueError(
                "output_contract_error trace requires output_contract_error reject_reason"
            )
        return self


def prepare_citywalker_to_urbannav_public_input_v1(
    *,
    images: Sequence[str],
    image_roles: Sequence[str],
    public_context: Mapping[str, Any],
) -> CityWalkerToUrbanNavPreparedInputV1:
    r"""Derive a native input exclusively from frozen public UrbanNav fields.

    The source frame is UrbanNav FLU ``(f,l)``.  Its public conversion to
    CityWalker local xy is

    \[
      (x_{right}, y_{front})=(-l, f).
    \]

    The five history points are translated to their current point and rotated
    exactly as CityWalker's ``input_target`` path does.  The target is *not*
    rotated because the native dataset concatenates its independently local
    target after the rotated history.  All six rows are normalized by the
    median positive step length of the public five-point history.

    Args:
        images: Ordered model-visible image paths.
        image_roles: Roles aligned to ``images``.
        public_context: Must contain ``coordinate_frame``, ``history_xy_m``,
            ``history_image_indices``, and ``target_xy_m``.

    Returns:
        Validated public-native request.

    Raises:
        CityWalkerToUrbanNavInputErrorV1: If the public contract is incomplete
            or unsafe to adapt.
    """

    if len(images) != len(image_roles):
        raise CityWalkerToUrbanNavInputErrorV1("unsupported_public_input")
    if public_context.get("coordinate_frame"
                          ) != URBANNAV_CURRENT_EGO_FLU_COORDINATE_FRAME_V1:
        raise CityWalkerToUrbanNavInputErrorV1("unsupported_public_input")
    raw_history = public_context.get("history_xy_m")
    raw_indices = public_context.get("history_image_indices")
    if not isinstance(raw_history, Sequence) or isinstance(
            raw_history, (str, bytes)):
        raise CityWalkerToUrbanNavInputErrorV1("missing_public_history")
    if not isinstance(raw_indices, Sequence) or isinstance(
            raw_indices, (str, bytes)):
        raise CityWalkerToUrbanNavInputErrorV1("missing_public_history")
    if len(raw_history) != len(raw_indices):
        raise CityWalkerToUrbanNavInputErrorV1("missing_public_history")
    history = tuple(
        _finite_xy_v1(row, reason="missing_public_history")
        for row in raw_history)
    try:
        indices = tuple(int(index) for index in raw_indices)
    except (TypeError, ValueError) as exc:
        raise CityWalkerToUrbanNavInputErrorV1(
            "missing_public_history") from exc
    if (len(indices) < CITYWALKER_TO_URBANNAV_REQUIRED_FPV_COUNT_V1
            or any(index < 0 or index >= len(images)
                   for index in indices) or len(set(indices)) != len(indices)
            or tuple(sorted(indices)) != indices):
        raise CityWalkerToUrbanNavInputErrorV1("insufficient_public_fpv")
    selected_indices = indices[-CITYWALKER_TO_URBANNAV_REQUIRED_FPV_COUNT_V1:]
    history = history[-CITYWALKER_TO_URBANNAV_REQUIRED_FPV_COUNT_V1:]
    roles = tuple(str(image_roles[index]) for index in selected_indices)
    if not all(_is_fpv_role_v1(role) for role in roles):
        raise CityWalkerToUrbanNavInputErrorV1("insufficient_public_fpv")
    target = public_context.get("target_xy_m")
    if target is None:
        raise CityWalkerToUrbanNavInputErrorV1("missing_public_goal")
    target_flu = _finite_xy_v1(target, reason="missing_public_goal")
    scale = public_history_step_scale_v1(history)
    native_history = _rotate_native_history_v1(
        tuple(flu_to_citywalker_xy_v1(point) for point in history))
    native_target = flu_to_citywalker_xy_v1(target_flu)
    normalized = tuple((point[0] / scale, point[1] / scale)
                       for point in (*native_history, native_target))
    return CityWalkerToUrbanNavPreparedInputV1(
        selected_image_indices=selected_indices,
        selected_image_roles=roles,
        selected_images=tuple(
            str(images[index]) for index in selected_indices),
        history_flu_xy_m=history,
        target_flu_xy_m=target_flu,
        normalized_native_input_xy=normalized,
        public_scale_value=scale,
    )


def public_history_step_scale_v1(
        history_flu_xy_m: Sequence[Sequence[float]]) -> float:
    r"""Estimate a native scale solely from public history motion.

    \[
      s = \operatorname{median}\{\|p_i-p_{i-1}\|_2: \|p_i-p_{i-1}\|_2\ge 0.01\}.
    \]

    Args:
        history_flu_xy_m: At least two public ``(forward,left)`` positions.

    Returns:
        Positive robust step scale in metres.

    Raises:
        CityWalkerToUrbanNavInputErrorV1: If no stable public motion exists.
    """

    points = tuple(
        _finite_xy_v1(point, reason="invalid_scale")
        for point in history_flu_xy_m)
    steps = [
        math.hypot(current[0] - prior[0], current[1] - prior[1])
        for prior, current in zip(points, points[1:])
    ]
    positive_steps = [
        step for step in steps if math.isfinite(step)
        and step >= CITYWALKER_TO_URBANNAV_MIN_PUBLIC_STEP_M_V1
    ]
    if not positive_steps:
        raise CityWalkerToUrbanNavInputErrorV1("invalid_scale")
    value = float(median(positive_steps))
    if not math.isfinite(
            value) or value < CITYWALKER_TO_URBANNAV_MIN_PUBLIC_STEP_M_V1:
        raise CityWalkerToUrbanNavInputErrorV1("invalid_scale")
    return value


def flu_to_citywalker_xy_v1(
        point_flu_xy: Sequence[float]) -> tuple[float, float]:
    """Project UrbanNav FLU ``(forward,left)`` into CityWalker ``(right,front)``.

    Args:
        point_flu_xy: Finite UrbanNav local point.

    Returns:
        Finite CityWalker local point.
    """

    forward, left = _finite_xy_v1(point_flu_xy,
                                  reason="unsupported_public_input")
    return (-left, forward)


def citywalker_xy_to_flu_v1(
        point_citywalker_xy: Sequence[float]) -> tuple[float, float]:
    """Project CityWalker ``(right,front)`` output into UrbanNav FLU.

    Args:
        point_citywalker_xy: Finite native CityWalker local point.

    Returns:
        Finite UrbanNav ``(forward,left)`` point.
    """

    right, front = _finite_xy_v1(point_citywalker_xy,
                                 reason="output_contract_error")
    return (front, -right)


def _rotate_native_history_v1(
    history_xy: Sequence[tuple[float,
                               float]]) -> tuple[tuple[float, float], ...]:
    """Translate and rotate public native xy history as CityWalker input-target does."""

    if len(history_xy) != CITYWALKER_TO_URBANNAV_REQUIRED_FPV_COUNT_V1:
        raise CityWalkerToUrbanNavInputErrorV1("insufficient_public_fpv")
    current_x, current_y = history_xy[-1]
    translated = tuple((x - current_x, y - current_y) for x, y in history_xy)
    penultimate_x, penultimate_y = translated[-2]
    if math.hypot(penultimate_x,
                  penultimate_y) < CITYWALKER_TO_URBANNAV_MIN_PUBLIC_STEP_M_V1:
        raise CityWalkerToUrbanNavInputErrorV1("invalid_scale")
    angle = -math.pi / 2.0 - math.atan2(penultimate_y, penultimate_x)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return tuple((cos_angle * x - sin_angle * y, sin_angle * x + cos_angle * y)
                 for x, y in translated)


def _finite_xy_v1(
        value: Any, *,
        reason: CityWalkerToUrbanNavRejectReasonV1) -> tuple[float, float]:
    """Parse one finite two-dimensional public coordinate or reject it."""

    if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes)) or len(value) != 2:
        raise CityWalkerToUrbanNavInputErrorV1(reason)
    try:
        first, second = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise CityWalkerToUrbanNavInputErrorV1(reason) from exc
    if not math.isfinite(first) or not math.isfinite(second):
        raise CityWalkerToUrbanNavInputErrorV1(reason)
    return first, second


def _is_fpv_role_v1(role: str) -> bool:
    """Return whether one role explicitly names an FPV observation."""

    return role in {"fpv_history", "fpv_current"} or role.startswith("fpv.")


__all__ = [
    "CITYWALKER_CURRENT_POSE_XY_COORDINATE_FRAME_V1",
    "CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1",
    "CITYWALKER_TO_URBANNAV_ADAPTER_ID_V1",
    "CITYWALKER_TO_URBANNAV_ADAPTER_VERSION_V1",
    "CITYWALKER_TO_URBANNAV_FLU_TO_NATIVE_PROJECTION_ID_V1",
    "CITYWALKER_TO_URBANNAV_MIN_PUBLIC_STEP_M_V1",
    "CITYWALKER_TO_URBANNAV_NATIVE_OUTPUT_SCHEDULE_ID_V1",
    "CITYWALKER_TO_URBANNAV_NATIVE_OUTPUT_TIMESTAMPS_S_V1",
    "CITYWALKER_TO_URBANNAV_NATIVE_TO_FLU_PROJECTION_ID_V1",
    "CITYWALKER_TO_URBANNAV_PUBLIC_HISTORY_TRANSFORM_ID_V1",
    "CITYWALKER_TO_URBANNAV_PUBLIC_SCALE_ESTIMATOR_ID_V1",
    "CITYWALKER_TO_URBANNAV_REQUIRED_FPV_COUNT_V1",
    "CityWalkerToUrbanNavAdapterTraceV1",
    "CityWalkerToUrbanNavEligibilityStatusV1",
    "CityWalkerToUrbanNavInputErrorV1",
    "CityWalkerToUrbanNavPreparedInputV1",
    "CityWalkerToUrbanNavRejectReasonV1",
    "URBANNAV_CITYWALKER_NATIVE_INPUT_PROFILE_ID_V1",
    "URBANNAV_CURRENT_EGO_FLU_COORDINATE_FRAME_V1",
    "URBANNAV_CURRENT_EGO_FLU_PREDICTION_GEOMETRY_ID_V1",
    "citywalker_xy_to_flu_v1",
    "flu_to_citywalker_xy_v1",
    "prepare_citywalker_to_urbannav_public_input_v1",
    "public_history_step_scale_v1",
]
