"""Deterministic geometry and capability-gated metrics for benchmark plugins.

The functions in this module are intentionally dataset neutral.  Dataset
plugins own how hidden references are constructed; metric plugins only consume
the frozen reference/prediction ledgers and a profile-owned physical time grid.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Callable, Sequence

from mcp_server.contracts.benchmark_plugin import BenchmarkProfileV1
from mcp_server.eval.openloop.benchmark_plugin_base import (
    BenchmarkJoinedRowV1,
    BenchmarkMetricPluginV1,
)


def evaluate_trajectory_geometry_v1(
    rows: Sequence[BenchmarkJoinedRowV1],
    *,
    profile: BenchmarkProfileV1,
) -> dict[str, Any]:
    r"""Evaluate fixed-time local-trajectory error and coverage.

    At each profile horizon ``t``, a row is geometry-eligible iff GT covers
    ``t`` and prediction-valid iff a successful output also covers ``t``.
    Conditional error is evaluated only over valid outputs, while coverage is
    always reported against the complete reference denominator:

    \[
    C_t = \frac{N_{\mathrm{prediction\ covers\ }t}}
    {N_{\mathrm{reference\ covers\ }t}}.
    \]

    Args:
        rows: Full-reference rows joined by stable sample id.
        profile: Frozen physical-time and geometry contract.

    Returns:
        Aggregate fixed-time error, axes, ADE, and coverage report.
    """

    horizon_rows: dict[float, list[tuple[tuple[float, float], tuple[float, float]]]] = defaultdict(list)
    reference_eligible: dict[float, int] = defaultdict(int)
    output_covered: dict[float, int] = defaultdict(int)
    failures = sum(1 for row in rows if row.failure is not None)
    successful_outputs = sum(1 for row in rows if row.prediction is not None)
    latency_s: list[float] = []

    for row in rows:
        if row.prediction is not None:
            latency = row.prediction.metadata.get("latency_sec")
            if isinstance(latency, (int, float)) and math.isfinite(float(latency)) and float(latency) >= 0.0:
                latency_s.append(float(latency))
        for horizon in profile.time_policy.horizons_s:
            target = interpolate_reference_xy_v1(row, horizon)
            if target is None:
                continue
            reference_eligible[horizon] += 1
            predicted = interpolate_prediction_xy_v1(row, horizon)
            if predicted is None:
                continue
            output_covered[horizon] += 1
            horizon_rows[horizon].append((target, predicted))

    horizon_report: dict[str, Any] = {}
    all_errors: list[float] = []
    for horizon in profile.time_policy.horizons_s:
        pairs = horizon_rows[horizon]
        errors = [_l2(target, predicted) for target, predicted in pairs]
        lateral_errors = [abs(target[0] - predicted[0]) for target, predicted in pairs]
        longitudinal_errors = [abs(target[1] - predicted[1]) for target, predicted in pairs]
        all_errors.extend(errors)
        eligible = reference_eligible[horizon]
        valid = output_covered[horizon]
        horizon_report[_horizon_key(horizon)] = {
            "horizon_s": horizon,
            "reference_eligible_count": eligible,
            "valid_prediction_count": valid,
            "geometry_coverage": _ratio(valid, eligible),
            "conditional_l2_m": _mean_or_none(errors),
            "conditional_lateral_abs_m": _mean_or_none(lateral_errors),
            "conditional_longitudinal_abs_m": _mean_or_none(longitudinal_errors),
        }

    return {
        "metric_id": "trajectory_geometry_core_v1",
        "status": "ok" if rows else "N/A",
        "reference_count": len(rows),
        "successful_prediction_count": successful_outputs,
        "explicit_failure_count": failures,
        "attempt_coverage": _ratio(successful_outputs + failures, len(rows)),
        "success_coverage": _ratio(successful_outputs, len(rows)),
        "amde_m": _mean_or_none(all_errors),
        "horizons": horizon_report,
        "latency_sec": {
            "count": len(latency_s),
            "mean": _mean_or_none(latency_s),
            "p50": _quantile_or_none(latency_s, 0.5),
            "p95": _quantile_or_none(latency_s, 0.95),
        },
    }


class TrajectoryGeometryMetricPluginV1(BenchmarkMetricPluginV1):
    """Expose the common fixed-time local-trajectory metric suite."""

    metric_suite_id = "trajectory_geometry_core_v1"

    def evaluate(
        self,
        rows: Sequence[BenchmarkJoinedRowV1],
        *,
        profile: BenchmarkProfileV1,
    ) -> dict[str, Any]:
        """Evaluate shared fixed-time geometry.

        Args:
            rows: Complete joined output ledger.
            profile: Frozen common geometry profile.

        Returns:
            Deterministic geometry scorecard.
        """

        return evaluate_trajectory_geometry_v1(rows, profile=profile)


def interpolate_reference_xy_v1(
    row: BenchmarkJoinedRowV1,
    horizon_s: float,
) -> tuple[float, float] | None:
    """Interpolate a hidden reference at a physical horizon without extrapolation.

    Args:
        row: One joined benchmark row.
        horizon_s: Positive requested physical horizon.

    Returns:
        ``(x, y)`` when the valid GT prefix covers the horizon, otherwise
        ``None``.
    """

    reference = row.reference
    valid_length = _valid_prefix_length(reference.trajectory_valid_mask)
    if valid_length == 0:
        return None
    return _interpolate_xy(
        reference.trajectory[:valid_length],
        reference.trajectory_timestamps_s[:valid_length],
        horizon_s,
    )


def interpolate_prediction_xy_v1(
    row: BenchmarkJoinedRowV1,
    horizon_s: float,
) -> tuple[float, float] | None:
    """Interpolate a model prediction at a physical horizon without extrapolation.

    A ``reference_schedule`` prediction is legal only while its output length
    fits the valid hidden reference schedule.  This supports CityWalker-native
    decoders without leaking GT timestamps to the model adapter.

    Args:
        row: One joined benchmark row.
        horizon_s: Positive requested physical horizon.

    Returns:
        ``(x, y)`` when prediction geometry covers the horizon, otherwise
        ``None``.
    """

    prediction = row.prediction
    if prediction is None:
        return None
    if prediction.time_basis == "explicit":
        assert prediction.trajectory_timestamps_s is not None
        return _interpolate_xy(prediction.trajectory, prediction.trajectory_timestamps_s, horizon_s)
    valid_length = _valid_prefix_length(row.reference.trajectory_valid_mask)
    if len(prediction.trajectory) > valid_length:
        return None
    return _interpolate_xy(
        prediction.trajectory,
        row.reference.trajectory_timestamps_s[: len(prediction.trajectory)],
        horizon_s,
    )


def evaluate_binary_arrival_v1(rows: Sequence[BenchmarkJoinedRowV1]) -> dict[str, Any]:
    """Evaluate arrival scores only when every required label/score is present.

    Args:
        rows: Full joined benchmark output ledger.

    Returns:
        Accuracy, precision, recall, F1, and AUROC or an explicit ``N/A``
        capability/coverage reason.
    """

    labeled = [row for row in rows if row.reference.arrival_label is not None]
    scored = [row for row in labeled if row.prediction is not None and row.prediction.arrival_score is not None]
    if not labeled:
        return {"status": "N/A", "reason": "arrival_label_absent", "reference_count": len(rows)}
    if not scored:
        return {
            "status": "N/A",
            "reason": "adapter_did_not_emit_arrival_score",
            "reference_count": len(rows),
            "labeled_reference_count": len(labeled),
            "scored_count": 0,
        }
    threshold = 0.5
    labels = [bool(row.reference.arrival_label) for row in scored]
    scores = [float(row.prediction.arrival_score) for row in scored if row.prediction is not None]
    predicted = [score >= threshold for score in scores]
    true_positive = sum(1 for label, item in zip(labels, predicted) if label and item)
    false_positive = sum(1 for label, item in zip(labels, predicted) if not label and item)
    false_negative = sum(1 for label, item in zip(labels, predicted) if label and not item)
    accuracy = _ratio(sum(label == item for label, item in zip(labels, predicted)), len(labels))
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0.0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "status": "ok",
        "threshold": threshold,
        "reference_count": len(rows),
        "labeled_reference_count": len(labeled),
        "scored_count": len(scored),
        "score_coverage": _ratio(len(scored), len(labeled)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _binary_auroc_or_none(labels, scores),
    }


def select_rows_v1(
    rows: Sequence[BenchmarkJoinedRowV1],
    predicate: Callable[[BenchmarkJoinedRowV1], bool],
) -> list[BenchmarkJoinedRowV1]:
    """Return a stable-order subgroup from a full benchmark denominator.

    Args:
        rows: Source joined rows.
        predicate: Deterministic group-membership predicate.

    Returns:
        Rows satisfying the predicate, preserving frozen package order.
    """

    return [row for row in rows if predicate(row)]


def _valid_prefix_length(mask: Sequence[bool]) -> int:
    """Return the prefix-contiguous valid trajectory length."""

    return next((index for index, item in enumerate(mask) if not item), len(mask))


def _interpolate_xy(
    trajectory: Sequence[Sequence[float]],
    timestamps_s: Sequence[float],
    horizon_s: float,
) -> tuple[float, float] | None:
    """Linearly interpolate x/y at a covered physical horizon."""

    if not trajectory or horizon_s < timestamps_s[0] or horizon_s > timestamps_s[-1]:
        return None
    for index, timestamp in enumerate(timestamps_s):
        if math.isclose(horizon_s, timestamp, rel_tol=0.0, abs_tol=1e-9):
            return float(trajectory[index][0]), float(trajectory[index][1])
        if horizon_s < timestamp:
            prior_time = timestamps_s[index - 1]
            alpha = (horizon_s - prior_time) / (timestamp - prior_time)
            prior = trajectory[index - 1]
            current = trajectory[index]
            return (
                float(prior[0]) + alpha * (float(current[0]) - float(prior[0])),
                float(prior[1]) + alpha * (float(current[1]) - float(prior[1])),
            )
    return float(trajectory[-1][0]), float(trajectory[-1][1])


def _l2(left: tuple[float, float], right: tuple[float, float]) -> float:
    """Return Euclidean error in the profile-owned local frame."""

    return math.hypot(left[0] - right[0], left[1] - right[1])


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return a safe denominator-aware ratio."""

    return None if denominator == 0 else numerator / denominator


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Return an arithmetic mean only for a non-empty finite sequence."""

    return None if not values else mean(values)


def _quantile_or_none(values: Sequence[float], probability: float) -> float | None:
    """Return deterministic linear-interpolated quantile for a non-empty list."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _binary_auroc_or_none(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Compute tie-aware AUROC, or ``None`` without both classes."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    wins = 0.0
    for label, score in zip(labels, scores):
        if not label:
            continue
        for other_label, other_score in zip(labels, scores):
            if other_label:
                continue
            if score > other_score:
                wins += 1.0
            elif math.isclose(score, other_score, rel_tol=0.0, abs_tol=1e-12):
                wins += 0.5
    return wins / (positives * negatives)


def _horizon_key(horizon_s: float) -> str:
    """Return a JSON-stable physical-horizon key."""

    return f"{horizon_s:g}s"


__all__ = [
    "TrajectoryGeometryMetricPluginV1",
    "evaluate_binary_arrival_v1",
    "evaluate_trajectory_geometry_v1",
    "interpolate_prediction_xy_v1",
    "interpolate_reference_xy_v1",
    "select_rows_v1",
]
