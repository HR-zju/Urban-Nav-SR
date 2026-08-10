"""CityWalker-specific metrics on top of the generic plugin contract."""

from __future__ import annotations

import math
from typing import Any, Sequence

from mcp_server.contracts.benchmark_plugin import BenchmarkProfileV1
from mcp_server.contracts.scene_event_tags import CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1
from mcp_server.eval.openloop.benchmark_plugin_base import (
    BenchmarkJoinedRowV1,
    BenchmarkMetricPluginV1,
)
from mcp_server.eval.openloop.benchmark_plugin_metrics import (
    evaluate_binary_arrival_v1,
    evaluate_trajectory_geometry_v1,
    interpolate_prediction_xy_v1,
    interpolate_reference_xy_v1,
    select_rows_v1,
)


# This benchmark still reads original CityWalker reference labels.  The shared
# taxonomy only guarantees stable slice names, not equal annotation provenance.
CITYWALKER_SCENARIO_LABELS_V1 = CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1


class CityWalkerMetricPluginV1(BenchmarkMetricPluginV1):
    """Evaluate CityWalker trajectory, angle, arrival, and overlapping slices."""

    metric_suite_id = "citywalker_openloop_metrics_v1"

    def evaluate(
        self,
        rows: Sequence[BenchmarkJoinedRowV1],
        *,
        profile: BenchmarkProfileV1,
    ) -> dict[str, Any]:
        """Evaluate the capability-gated CityWalker benchmark suite.

        Args:
            rows: Full CityWalker reference denominator joined by sample id.
            profile: Frozen benchmark profile and physical-time policy.

        Returns:
            CityWalker scorecard.  Unsupported arrival metrics are explicit
            ``N/A`` rather than zero-valued pseudo-scores.
        """

        geometry = evaluate_trajectory_geometry_v1(rows, profile=profile)
        angle = _evaluate_angle_v1(rows, profile=profile)
        slices = {}
        for label in CITYWALKER_SCENARIO_LABELS_V1:
            scenario_rows = select_rows_v1(
                rows,
                lambda row, item=label: item in row.reference.scenario_labels,
            )
            slices[label] = _slice_report_v1(scenario_rows, profile)
        return {
            "metric_id": self.metric_suite_id,
            "status": "ok" if rows else "N/A",
            "coordinate_frame_required": "citywalker_current_pose_x_right_y_front_v1",
            "geometry": geometry,
            "angle": angle,
            "arrival": evaluate_binary_arrival_v1(rows),
            "overlap_permitted_slices": slices,
        }


def _slice_report_v1(rows: Sequence[BenchmarkJoinedRowV1], profile: BenchmarkProfileV1) -> dict[str, Any]:
    """Return a compact single-scenario report without pretending exclusivity."""

    if not rows:
        return {"status": "N/A", "reason": "no_reference_rows", "reference_count": 0}
    geometry = evaluate_trajectory_geometry_v1(rows, profile=profile)
    angle = _evaluate_angle_v1(rows, profile=profile)
    return {
        "status": "ok",
        "reference_count": len(rows),
        "amde_m": geometry["amde_m"],
        "success_coverage": geometry["success_coverage"],
        "maoe_deg": angle["maoe_deg"],
        "angle_coverage": angle["angle_coverage"],
    }


def _evaluate_angle_v1(rows: Sequence[BenchmarkJoinedRowV1], *, profile: BenchmarkProfileV1) -> dict[str, Any]:
    """Evaluate ray-angle error only for non-degenerate reference displacements."""

    per_horizon: dict[str, Any] = {}
    all_angles: list[float] = []
    eligible_total = 0
    valid_total = 0
    for horizon in profile.time_policy.horizons_s:
        eligible = 0
        valid = 0
        angles: list[float] = []
        for row in rows:
            target = interpolate_reference_xy_v1(row, horizon)
            if target is None or math.hypot(*target) < profile.time_policy.min_angle_reference_distance_m:
                continue
            eligible += 1
            prediction = interpolate_prediction_xy_v1(row, horizon)
            if prediction is None:
                continue
            valid += 1
            angles.append(_angle_deg(target, prediction))
        eligible_total += eligible
        valid_total += valid
        all_angles.extend(angles)
        per_horizon[f"{horizon:g}s"] = {
            "horizon_s": horizon,
            "reference_eligible_count": eligible,
            "valid_prediction_count": valid,
            "angle_coverage": None if eligible == 0 else valid / eligible,
            "conditional_aoe_deg": None if not angles else sum(angles) / len(angles),
        }
    return {
        "status": "N/A" if eligible_total == 0 else "ok",
        "reason": "no_non_degenerate_reference_displacement" if eligible_total == 0 else None,
        "angle_eligible_count": eligible_total,
        "valid_prediction_count": valid_total,
        "angle_coverage": None if eligible_total == 0 else valid_total / eligible_total,
        "maoe_deg": None if not all_angles else sum(all_angles) / len(all_angles),
        "horizons": per_horizon,
    }


def _angle_deg(target: tuple[float, float], prediction: tuple[float, float]) -> float:
    """Return absolute ray-angle error in degrees for two local xy vectors."""

    target_angle = math.atan2(target[1], target[0])
    prediction_angle = math.atan2(prediction[1], prediction[0])
    delta = (prediction_angle - target_angle + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


__all__ = ["CITYWALKER_SCENARIO_LABELS_V1", "CityWalkerMetricPluginV1"]
