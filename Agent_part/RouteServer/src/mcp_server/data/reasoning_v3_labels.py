"""Deterministic teacher-only labels and dataset supervision policy for V3.3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mcp_server.contracts.reasoning_v3 import LabelProvenanceEntryV3, SectionLossMaskV3
from mcp_server.data.reasoning_v3_conditions import condition_source_directions_from_row_v3
from mcp_server.runtime.system1_contract import canonicalize_trajectory_ego_t


@dataclass(frozen=True)
class DatasetSupervisionPolicyV3:
    """Dataset-specific trust boundary for V3.3 label construction.

    Args:
        dataset_source: Canonical dataset family.
        continuous_trajectory_trusted: Whether System1 trajectory is trainable.
        trajectory_direction_trusted: Whether trajectory derives action labels.
        fpv_projection_trusted: Whether projected FPV corridors are strong GT.
        bev_direction_trusted: Whether validated BEV geometry is a condition.
    """

    dataset_source: str
    continuous_trajectory_trusted: bool
    trajectory_direction_trusted: bool
    fpv_projection_trusted: bool
    bev_direction_trusted: bool


@dataclass(frozen=True)
class DirectionDerivationConfigV3:
    """Thresholds for deterministic coarse direction labels.

    Args:
        stop_distance_m: Endpoint distance at or below which motion is stop.
        turn_threshold_deg: Absolute endpoint bearing that starts a turn bin.
        positive_y_is_left: Whether positive ego lateral displacement is left.
        generator_version: Provenance version for derived labels.
    """

    stop_distance_m: float = 0.20
    turn_threshold_deg: float = 10.0
    positive_y_is_left: bool = True
    generator_version: str = "trajectory_direction_v1"


@dataclass(frozen=True)
class ObjectiveDerivationConfigV3:
    """Thresholds for authoritative runtime task-phase facts.

    Args:
        goal_stop_distance_m: Distance at or below which the goal is reached.
        goal_approach_distance_m: Distance defining the goal-approach phase.
        generator_version: Provenance version for derived objectives.
    """

    goal_stop_distance_m: float = 0.50
    goal_approach_distance_m: float = 3.00
    generator_version: str = "runtime_task_state_v1"


@dataclass(frozen=True)
class DeterministicSupervisionV3:
    """Teacher-only labels constructed without an LVLM call.

    Args:
        dataset_source: Canonical dataset family.
        demonstrated_direction: Trusted demonstrated action direction.
        condition_direction: Direction implied by real model-visible conditions.
        authoritative_local_objective: Objective derived only from authoritative
            runtime or goal state, never from demonstrated motion.
        demonstrated_corridor_ref: Trusted demonstrated FPV corridor projection.
        system1_enabled: Whether continuous trajectory supervision is trusted.
        teacher_supervision_context: Private composer inputs and policy audit.
        label_provenance: Per-field supervision provenance.
        section_loss_mask: Preliminary source-level section eligibility.
        warnings: Deterministic conversion warnings.
    """

    dataset_source: str
    demonstrated_direction: str
    condition_direction: str
    authoritative_local_objective: str
    demonstrated_corridor_ref: str | None
    system1_enabled: bool
    teacher_supervision_context: dict[str, Any]
    label_provenance: dict[str, LabelProvenanceEntryV3]
    section_loss_mask: SectionLossMaskV3
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-compatible audit payload.

        Returns:
            Serialized deterministic supervision and provenance.
        """

        return {
            "dataset_source": self.dataset_source,
            "demonstrated_direction": self.demonstrated_direction,
            "condition_direction": self.condition_direction,
            "authoritative_local_objective": self.authoritative_local_objective,
            "demonstrated_corridor_ref": self.demonstrated_corridor_ref,
            "system1_enabled": self.system1_enabled,
            "teacher_supervision_context": self.teacher_supervision_context,
            "label_provenance": {
                key: value.model_dump(mode="json") for key, value in self.label_provenance.items()
            },
            "section_loss_mask": self.section_loss_mask.model_dump(mode="json"),
            "warnings": list(self.warnings),
        }


_POLICIES = {
    "sidewalk": DatasetSupervisionPolicyV3("sidewalk", True, True, True, True),
    "foxglove": DatasetSupervisionPolicyV3("foxglove", True, True, True, True),
    "egowalk": DatasetSupervisionPolicyV3("egowalk", True, True, False, False),
    "frodobots": DatasetSupervisionPolicyV3("frodobots", False, False, False, True),
}

_DATASET_ALIASES = {
    "sidewalk-main": "sidewalk",
    "sidewalk-extra": "sidewalk",
    "sidewalk_main": "sidewalk",
    "sidewalk_extra": "sidewalk",
    "foxglove-yaw-p90": "foxglove",
    "foxglove_yaw_p90": "foxglove",
    "egowalk-fpv-only": "egowalk",
    "egowalk_fpv_only": "egowalk",
    "frodobots-qc": "frodobots",
    "frodobots_qc": "frodobots",
}


def dataset_supervision_policy_v3(dataset_source: str) -> DatasetSupervisionPolicyV3:
    """Return the explicit trust policy for one canonical dataset family.

    Args:
        dataset_source: Dataset name or supported alias.

    Returns:
        Immutable supervision trust policy.

    Raises:
        ValueError: If no policy exists for the dataset family.
    """

    normalized = _normalize_dataset_source(dataset_source)
    try:
        return _POLICIES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported V3 dataset supervision policy: {dataset_source}") from exc


def derive_direction_from_trajectory_v3(
    trajectory: Sequence[Sequence[float]],
    *,
    config: DirectionDerivationConfigV3 | None = None,
) -> tuple[str, dict[str, float]]:
    r"""Derive stop/left/straight/right from a trusted ego trajectory.

    For endpoint \((x_T, y_T)\), the rule computes

    \[
    d_T = \sqrt{x_T^2 + y_T^2}, \qquad
    \theta_T = \operatorname{atan2}(y_T, x_T) \cdot 180 / \pi.
    \]

    ``stop`` is selected when \(d_T\) is below the stop threshold; otherwise
    the sign and magnitude of \(\theta_T\) select left, straight, or right.

    Args:
        trajectory: Trusted ego-frame points containing at least x and y.
        config: Optional direction thresholds and axis convention.

    Returns:
        Coarse direction and numerical derivation audit fields.

    Raises:
        ValueError: If no finite endpoint exists.
    """

    cfg = config or DirectionDerivationConfigV3()
    points = _finite_xy_points(trajectory)
    if not points:
        raise ValueError("trusted trajectory has no finite xy point")
    x_forward, y_lateral = points[-1]
    if not cfg.positive_y_is_left:
        y_lateral = -y_lateral
    distance_m = math.hypot(x_forward, y_lateral)
    theta_deg = math.degrees(math.atan2(y_lateral, x_forward))
    if distance_m <= cfg.stop_distance_m:
        direction = "stop"
    elif theta_deg >= cfg.turn_threshold_deg:
        direction = "left"
    elif theta_deg <= -cfg.turn_threshold_deg:
        direction = "right"
    else:
        direction = "straight"
    return direction, {
        "endpoint_x_forward_m": x_forward,
        "endpoint_y_left_m": y_lateral,
        "endpoint_distance_m": distance_m,
        "endpoint_theta_deg": theta_deg,
        "stop_distance_m": cfg.stop_distance_m,
        "turn_threshold_deg": cfg.turn_threshold_deg,
    }


def derive_corridor_from_fpv_trajectory_v3(
    trajectory: Sequence[Sequence[float]],
    *,
    coord_frame: str,
    image_width: float | None = None,
) -> tuple[str, dict[str, float | str]]:
    r"""Map a trusted FPV endpoint to one coarse corridor address.

    Pixel endpoints use \(u=x_T/W\); normalized endpoints provide \(u\)
    directly. The image is partitioned into thirds: left for \(u<1/3\),
    center for \(1/3\leq u\leq2/3\), and right for \(u>2/3\).

    Args:
        trajectory: Trusted FPV points containing horizontal coordinates.
        coord_frame: Normalized, norm1000, or pixel coordinate convention.
        image_width: Required positive width for pixel coordinates.

    Returns:
        Canonical corridor reference and projection audit fields.

    Raises:
        ValueError: If points, coordinate frame, width, or normalized endpoint
            are invalid.
    """

    points = _finite_xy_points(trajectory)
    if not points:
        raise ValueError("trusted FPV trajectory has no finite xy point")
    x_value = points[-1][0]
    normalized_frame = str(coord_frame).strip().lower()
    if normalized_frame in {"normalized", "normalized_0_1", "norm01", "fpv_norm"}:
        x_norm = x_value
    elif normalized_frame in {"norm1000", "grid1000"}:
        x_norm = x_value / 1000.0
    elif normalized_frame in {"pixel", "pixels", "fpv_pixel"}:
        if image_width is None or float(image_width) <= 0:
            raise ValueError("pixel FPV trajectory requires positive image_width")
        x_norm = x_value / float(image_width)
    else:
        raise ValueError(f"unsupported FPV trajectory coordinate frame: {coord_frame}")
    if not math.isfinite(x_norm) or not 0.0 <= x_norm <= 1.0:
        raise ValueError("FPV trajectory endpoint is outside normalized image width")
    if x_norm < 1.0 / 3.0:
        corridor = "corridor.left"
    elif x_norm > 2.0 / 3.0:
        corridor = "corridor.right"
    else:
        corridor = "corridor.center"
    return corridor, {"endpoint_x": x_value, "endpoint_x_norm": x_norm, "coord_frame": normalized_frame}


def derive_local_objective_v3(
    row: Mapping[str, Any],
    *,
    config: ObjectiveDerivationConfigV3 | None = None,
) -> tuple[str, dict[str, Any]]:
    """Derive an objective only from authoritative structured task state.

    Demonstrated motion is deliberately excluded. A stop trajectory can mean
    yielding, blockage, recording delay, or goal arrival and therefore cannot
    define what the user requested.

    Args:
        row: Original row containing task, goal, waypoint, or route state.
        config: Optional goal-distance phase thresholds.

    Returns:
        Controlled local objective and rule audit fields.
    """

    cfg = config or ObjectiveDerivationConfigV3()
    goal_reached = bool(_first_present(row, "goal_reached", "arrived", "is_goal_reached"))
    goal_distance = _finite_number(_first_present(row, "goal_distance_m", "distance_to_goal_m"))
    has_goal = _first_present(row, "goal_spec", "goal", "destination", "goal_text") is not None
    active_waypoint = _first_present(row, "active_waypoint", "active_waypoint_ref", "active_waypoint_index")
    route_state = _first_present(row, "route_state", "active_route_state")
    explicit_phase = str(_first_present(row, "runtime_task_phase", "task_phase") or "").strip().lower()
    allowed_phases = {
        "continue_route",
        "track_waypoint",
        "approach_goal",
        "enter_crosswalk",
        "exit_crosswalk",
        "hold_position",
        "stop_at_goal",
    }

    if explicit_phase in allowed_phases:
        objective = explicit_phase
    elif goal_reached or (has_goal and goal_distance is not None and goal_distance <= cfg.goal_stop_distance_m):
        objective = "stop_at_goal"
    elif has_goal and goal_distance is not None and goal_distance <= cfg.goal_approach_distance_m:
        objective = "approach_goal"
    elif active_waypoint is not None:
        objective = "track_waypoint"
    elif route_state is not None:
        objective = "continue_route"
    else:
        objective = "unknown"
    return objective, {
        "authoritative": objective != "unknown",
        "explicit_phase": explicit_phase or None,
        "goal_reached": goal_reached,
        "goal_distance_m": goal_distance,
        "has_goal": has_goal,
        "active_waypoint_present": active_waypoint is not None,
        "route_state_present": route_state is not None,
    }


def build_deterministic_supervision_v3(
    row: Mapping[str, Any],
    *,
    dataset_source: str,
    direction_config: DirectionDerivationConfigV3 | None = None,
    objective_config: ObjectiveDerivationConfigV3 | None = None,
) -> DeterministicSupervisionV3:
    """Build fixed supervision while enforcing the dataset trust matrix.

    Args:
        row: Original heterogeneous manifest row.
        dataset_source: Dataset family or supported alias.
        direction_config: Optional trajectory direction thresholds.
        objective_config: Optional task-phase thresholds.

    Returns:
        Isolated deterministic labels, provenance, masks, and warnings.

    Raises:
        ValueError: If the dataset is unsupported or trusted geometry is
            malformed.
    """

    policy = dataset_supervision_policy_v3(dataset_source)
    warnings: list[str] = []
    provenance: dict[str, LabelProvenanceEntryV3] = {}
    teacher_context: dict[str, Any] = {
        "dataset_policy": {
            "dataset_source": policy.dataset_source,
            "continuous_trajectory_trusted": policy.continuous_trajectory_trusted,
            "trajectory_direction_trusted": policy.trajectory_direction_trusted,
            "fpv_projection_trusted": policy.fpv_projection_trusted,
            "bev_direction_trusted": policy.bev_direction_trusted,
        }
    }

    trajectory = _trajectory_value(row)
    has_trusted_trajectory = bool(trajectory) and policy.trajectory_direction_trusted
    demonstrated_direction = "unknown"
    if has_trusted_trajectory:
        demonstrated_direction, details = derive_direction_from_trajectory_v3(
            trajectory,
            config=direction_config,
        )
        teacher_context["trusted_trajectory_summary"] = details
        provenance["demonstrated_direction"] = LabelProvenanceEntryV3(
            source="gt_trajectory" if policy.dataset_source != "egowalk" else "odometry",
            strength="strong",
            generator_version=(direction_config or DirectionDerivationConfigV3()).generator_version,
            details=details,
        )
    elif policy.dataset_source == "frodobots" and trajectory:
        warnings.append("frodobots_trajectory_ignored_by_policy")
    elif not trajectory:
        warnings.append("trusted_demonstration_direction_missing")

    condition_source_directions = condition_source_directions_from_row_v3(row)
    if policy.dataset_source == "frodobots":
        bev_direction, details = _derive_frodobots_bev_direction(row)
        if bev_direction != "unknown":
            condition_source_directions["bev.current"] = bev_direction
            teacher_context["validated_bev_direction"] = details
        else:
            warnings.append("frodobots_missing_validated_bev_direction")
    condition_values = set(condition_source_directions.values())
    condition_direction = next(iter(condition_values), "unknown") if len(condition_values) == 1 else "unknown"
    if len(condition_values) > 1:
        warnings.append("condition_direction_conflict")
    if condition_direction != "unknown":
        condition_sources = sorted(condition_source_directions)
        source_name = "bev_geometry" if condition_sources == ["bev.current"] else "visible_condition_rules"
        provenance["condition_direction"] = LabelProvenanceEntryV3(
            source=source_name,
            strength="strong",
            generator_version="condition_direction_v1",
            details={"source_directions": dict(sorted(condition_source_directions.items()))},
        )

    demonstrated_corridor_ref: str | None = None
    fpv_trajectory, coord_frame, image_width = _fpv_trajectory_value(row)
    fpv_projection_qc_passed = policy.fpv_projection_trusted and _fpv_projection_qc_passed(row)
    if fpv_trajectory and fpv_projection_qc_passed and demonstrated_direction != "stop":
        try:
            demonstrated_corridor_ref, details = derive_corridor_from_fpv_trajectory_v3(
                fpv_trajectory,
                coord_frame=coord_frame,
                image_width=image_width,
            )
        except ValueError as exc:
            warnings.append(f"fpv_projection_rejected:{exc}")
        else:
            teacher_context["trusted_fpv_corridor"] = details
            provenance["demonstrated_corridor_ref"] = LabelProvenanceEntryV3(
                source="validated_fpv_projection",
                strength="strong",
                generator_version="fpv_corridor_v1",
                details=details,
            )
    elif fpv_trajectory and policy.fpv_projection_trusted:
        warnings.append("fpv_projection_ignored_without_explicit_qc")

    authoritative_local_objective, objective_details = derive_local_objective_v3(
        row,
        config=objective_config,
    )
    teacher_context["authoritative_task_state"] = objective_details
    if authoritative_local_objective != "unknown":
        provenance["authoritative_local_objective"] = LabelProvenanceEntryV3(
            source="runtime_task_state",
            strength="strong",
            generator_version=(objective_config or ObjectiveDerivationConfigV3()).generator_version,
            details=objective_details,
        )

    system1_enabled = policy.continuous_trajectory_trusted and bool(trajectory)
    instruction_present = bool(str(_first_present(row, "instruction", "navigation_instruction", "caption") or "").strip())
    section_loss_mask = SectionLossMaskV3(
        task_grounding=instruction_present,
        scene_grounding=True,
        local_traversability=True,
        navigation_context_grounding=_has_navigation_context_input(row),
        interaction_and_rule_grounding=True,
        # Decision 可由强 GT outcome 或经过验证的可见 task/context 形成弱监督；
        # 是否真正训练仍由 composer 的 rationale provenance 二次门控。
        decision=(
            demonstrated_direction != "unknown"
            or condition_direction != "unknown"
            or instruction_present
        ),
    )
    teacher_context["fixed_supervision_version"] = "reasoning_v3_3_fixed_supervision_v1"
    teacher_context["fixed_labels"] = {
        "demonstrated_direction": demonstrated_direction,
        "condition_direction": condition_direction,
        "condition_source_directions": dict(sorted(condition_source_directions.items())),
        "authoritative_local_objective": authoritative_local_objective,
        "goal_reached": bool(objective_details.get("goal_reached")),
        "demonstrated_corridor_ref": demonstrated_corridor_ref,
        "system1_enabled": system1_enabled,
    }
    return DeterministicSupervisionV3(
        dataset_source=policy.dataset_source,
        demonstrated_direction=demonstrated_direction,
        condition_direction=condition_direction,
        authoritative_local_objective=authoritative_local_objective,
        demonstrated_corridor_ref=demonstrated_corridor_ref,
        system1_enabled=system1_enabled,
        teacher_supervision_context=teacher_context,
        label_provenance=provenance,
        section_loss_mask=section_loss_mask,
        warnings=tuple(warnings),
    )


def _has_navigation_context_input(row: Mapping[str, Any]) -> bool:
    """Return whether a row exposes any route-specific model input.

    Args:
        row: Original heterogeneous manifest row.

    Returns:
        ``True`` for directional/route instructions or optional BEV, goal,
        waypoint, route, and tool-route inputs.
    """

    # 这里只声明输入源存在；它是不是导航 instruction 由 LVLM task semantic teacher 判断。
    instruction = str(row.get("instruction") or row.get("navigation_instruction") or row.get("caption") or "")
    if instruction.strip():
        return True
    if any(row.get(key) is not None for key in (
        "bev_image_path",
        "bev_path",
        "goal",
        "goal_spec",
        "goal_image_path",
        "waypoints",
        "coarse_waypoints",
        "route_context",
        "tool_response_context",
    )):
        return True
    return any(str(role).startswith(("bev.", "goal.image_")) for role in row.get("image_roles") or ())


def _derive_frodobots_bev_direction(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    containers: list[tuple[str, Mapping[str, Any]]] = [("", row)]
    for container_name in ("sidewalk_meta", "route_context", "bev_context"):
        value = row.get(container_name)
        if isinstance(value, Mapping):
            containers.append((container_name, value))
    for container_name, container in containers:
        for key in (
            "bev_direction",
            "direction_label",
            "intent_gt",
            "intent",
            "direction_choice_label",
        ):
            value = container.get(key)
            direction = _normalize_direction(value)
            if direction != "unknown":
                field = f"{container_name}.{key}" if container_name else key
                return direction, {"source_field": field, "source_value": value}
    for container_name, container in containers:
        theta = _finite_number(_first_present(container, "route_theta_deg", "bev_route_theta_deg"))
        if theta is None:
            continue
        threshold = DirectionDerivationConfigV3().turn_threshold_deg
        # FrodoBots BEV 坐标为 x 向右、y 向前，导出器约定负 theta 表示左转。
        direction = "left" if theta <= -threshold else ("right" if theta >= threshold else "straight")
        field = f"{container_name}.route_theta_deg" if container_name else "route_theta_deg"
        return direction, {
            "source_field": field,
            "source_value": theta,
            "turn_threshold_deg": threshold,
            "theta_convention": "negative_left_positive_right",
        }
    return "unknown", {}


def _normalize_direction(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return "unknown"
    if normalized == "stop" or normalized.startswith("stop_"):
        return "stop"
    if normalized in {"straight", "go_straight", "forward", "center"}:
        return "straight"
    if "left" in normalized:
        return "left"
    if "right" in normalized:
        return "right"
    return "unknown"


def _trajectory_value(row: Mapping[str, Any]) -> list[list[float]]:
    """Return a canonical metric FLU trajectory or an empty list.

    Bare historical rows are converted through the same System1 coordinate
    contract used by the source builder. Invalid or unknown declared formats
    are never used to derive a strong demonstrated action.
    """

    value = _first_present(row, "trajectory_ego_t", "odometry_trajectory_ego_t")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    policy = row.get("trajectory_policy") if isinstance(row.get("trajectory_policy"), Mapping) else {}
    source_format = str(policy.get("system1_target_format") or row.get("system1_target_format") or "").strip()
    try:
        return canonicalize_trajectory_ego_t(value, source_format=source_format or None)
    except (TypeError, ValueError):
        return []


def _fpv_projection_qc_passed(row: Mapping[str, Any]) -> bool:
    """Return whether a projected FPV trajectory has explicit trusted provenance.

    An image-plane polyline is strong supervision only when a row explicitly
    marks projection QC as passed or identifies a calibrated pose/camera
    projection source. Merely finding ``pixel_trajectory`` is insufficient.
    """

    sidewalk_meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), Mapping) else {}
    for key in ("fpv_projection_qc_passed", "trajectory_projection_qc_passed", "projection_qc_passed"):
        if key in row:
            return _qc_truth_value(row.get(key))
        if key in sidewalk_meta:
            return _qc_truth_value(sidewalk_meta.get(key))
    trajectory_policy = row.get("trajectory_policy") if isinstance(row.get("trajectory_policy"), Mapping) else {}
    source = str(
        _first_present(row, "trajectory_source", "fpv_trajectory_source")
        or trajectory_policy.get("trajectory_source")
        or sidewalk_meta.get("trajectory_source")
        or ""
    ).strip().lower()
    return source in {
        "calibrated_pose_projection",
        "calibrated_camera_projection",
        "anchor_fixed_calibrated_pose_projection",
        "foxglove_anchor_fixed_projection",
    }


def _qc_truth_value(value: Any) -> bool:
    """Normalize explicit projection-QC booleans and status strings."""

    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "pass", "passed", "valid", "ok"}


def _fpv_trajectory_value(row: Mapping[str, Any]) -> tuple[list[list[float]], str, float | None]:
    value = _first_present(row, "trajectory_fpv_t", "pixel_trajectory", "trajectory_grid1000")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [], "", None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            return [], "", None
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return [], "", None
    coord_frame = str(
        _first_present(row, "trajectory_fpv_coord_frame", "trajectory_coord_frame", "pixel_trajectory_coord_frame")
        or ""
    )
    image_width = _finite_number(_first_present(row, "fpv_image_width", "image_width"))
    if not coord_frame and points:
        max_x = max(abs(point[0]) for point in points)
        if max_x <= 1.0:
            coord_frame = "normalized_0_1"
    return points, coord_frame, image_width


def _finite_xy_points(value: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for point in value:
        if len(point) < 2:
            continue
        try:
            x_value = float(point[0])
            y_value = float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            result.append((x_value, y_value))
    return result


def _normalize_dataset_source(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    return _DATASET_ALIASES.get(normalized, normalized)


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "DatasetSupervisionPolicyV3",
    "DeterministicSupervisionV3",
    "DirectionDerivationConfigV3",
    "ObjectiveDerivationConfigV3",
    "build_deterministic_supervision_v3",
    "dataset_supervision_policy_v3",
    "derive_corridor_from_fpv_trajectory_v3",
    "derive_direction_from_trajectory_v3",
    "derive_local_objective_v3",
]
