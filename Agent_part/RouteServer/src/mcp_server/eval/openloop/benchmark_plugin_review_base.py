"""Shared review-only contracts and geometry helpers for plugin benchmarks.

This module deliberately keeps native-adapter diagnostics outside the frozen
input, reference, prediction, and scorecard contracts.  A model-side adapter
may write a public-only trace sidecar; RouteServer validates and renders that
sidecar only after canonical scoring has completed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    benchmark_plugin_digest_v1,
)
from mcp_server.contracts.citywalker_to_urbannav_adapter import (
    CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1,
    CityWalkerToUrbanNavAdapterTraceV1,
    CityWalkerToUrbanNavEligibilityStatusV1,
    CityWalkerToUrbanNavRejectReasonV1,
)
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkJoinedRowV1
from mcp_server.eval.openloop.benchmark_plugin_metrics import (
    interpolate_prediction_xy_v1,
    interpolate_reference_xy_v1,
)

BENCHMARK_NATIVE_ADAPTER_TRACE_SCHEMA_VERSION = CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1
BenchmarkNativeAdapterTraceV1 = CityWalkerToUrbanNavAdapterTraceV1
NativeAdapterTraceStatusV1 = CityWalkerToUrbanNavEligibilityStatusV1
NativeAdapterRejectReasonV1 = CityWalkerToUrbanNavRejectReasonV1


@dataclass(frozen=True)
class BenchmarkReviewTimeSplitV1:
    """Split one time-stamped curve at an evaluator score horizon.

    Args:
        scored_prefix: Curve points from its first timestamp through the scored
            horizon, including a linearly interpolated final point when needed.
        future_suffix: Curve points after the scored horizon.  It begins at the
            same interpolated boundary point when a continuous dashed tail is
            available.
        at_horizon: Evaluator comparable point or ``None`` when uncovered.
        terminal: Final available source point.
        initial_time_s: First timestamp of the displayed curve.
        terminal_time_s: Final timestamp of the displayed curve.
    """

    scored_prefix: tuple[tuple[float, float], ...]
    future_suffix: tuple[tuple[float, float], ...]
    at_horizon: tuple[float, float] | None
    terminal: tuple[float, float] | None
    initial_time_s: float | None
    terminal_time_s: float | None


@dataclass(frozen=True)
class BenchmarkReviewCaseGeometryV1:
    """Evaluator-only review geometry shared by generic and native renderers.

    Args:
        reference_curve: Hidden reference curve partitioned at score horizon.
        prediction_curve: Prediction curve partitioned at score horizon.
        l2_at_horizon_m: Euclidean error only when both curves cover horizon.
    """

    reference_curve: BenchmarkReviewTimeSplitV1
    prediction_curve: BenchmarkReviewTimeSplitV1
    l2_at_horizon_m: float | None


class BenchmarkReviewRendererBaseV1:
    """Prepare invariant review geometry for specialized HTML renderers.

    This small base class owns time-split and horizon-L2 semantics only.  A
    future native CityWalker renderer can inherit it or use it by composition,
    while keeping its panel layout separate from canonical metrics and
    model-side inference.

    Args:
        horizon_s: Positive physical score horizon rendered by a subclass.
    """

    def __init__(self, *, horizon_s: float) -> None:
        """Validate the physical review horizon.

        Args:
            horizon_s: Positive finite metric/review horizon.

        Raises:
            ValueError: If ``horizon_s`` is not finite and positive.
        """

        if not math.isfinite(horizon_s) or horizon_s <= 0.0:
            raise ValueError("review horizon_s must be finite and positive")
        self._horizon_s = float(horizon_s)

    @property
    def horizon_s(self) -> float:
        """Return the immutable physical horizon shared by all review cards.

        Returns:
            Positive horizon in seconds.
        """

        return self._horizon_s

    def prepare_case(
        self,
        row: BenchmarkJoinedRowV1,
    ) -> BenchmarkReviewCaseGeometryV1:
        """Compute evaluator-owned geometry once for one review card.

        Args:
            row: Frozen reference plus optional prediction/failure ledger row.

        Returns:
            Time-split hidden reference, prediction, and conditional L2.
        """

        reference_curve = reference_review_curve_v1(row,
                                                    horizon_s=self._horizon_s)
        prediction_curve = prediction_review_curve_v1(
            row, horizon_s=self._horizon_s)
        target = reference_curve.at_horizon
        prediction = prediction_curve.at_horizon
        l2_at_horizon_m = (None if target is None or prediction is None else
                           math.hypot(target[0] - prediction[0], target[1] -
                                      prediction[1]))
        return BenchmarkReviewCaseGeometryV1(
            reference_curve=reference_curve,
            prediction_curve=prediction_curve,
            l2_at_horizon_m=l2_at_horizon_m,
        )


def split_benchmark_review_curve_at_horizon_v1(
    trajectory: Sequence[Sequence[float]],
    timestamps_s: Sequence[float],
    *,
    horizon_s: float,
) -> BenchmarkReviewTimeSplitV1:
    """Split a finite xy curve at one physical score horizon.

    Args:
        trajectory: Ordered local two- or three-dimensional trajectory rows.
        timestamps_s: Strictly increasing timestamps aligned to trajectory.
        horizon_s: Positive physical score horizon.

    Returns:
        Solid-prefix, dashed-tail, interpolation, and endpoint information for
        a review renderer.  It does not extrapolate outside observed time.

    Raises:
        ValueError: If series lengths, coordinates, timestamps, or horizon are
            malformed.
    """

    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("review horizon_s must be finite and positive")
    if len(trajectory) != len(timestamps_s):
        raise ValueError("review trajectory/timestamp lengths must match")
    if not trajectory:
        return BenchmarkReviewTimeSplitV1((), (), None, None, None, None)
    points = tuple(_finite_xy_v1(item) for item in trajectory)
    times = tuple(float(item) for item in timestamps_s)
    if any(not math.isfinite(item) for item in times):
        raise ValueError("review timestamps must be finite")
    if any(current <= prior for prior, current in zip(times, times[1:])):
        raise ValueError("review timestamps must be strictly increasing")
    if horizon_s < times[0]:
        return BenchmarkReviewTimeSplitV1((), points, None, points[-1],
                                          times[0], times[-1])
    if horizon_s > times[-1]:
        return BenchmarkReviewTimeSplitV1(points, (), None, points[-1],
                                          times[0], times[-1])
    for index, current_time in enumerate(times):
        if math.isclose(horizon_s, current_time, rel_tol=0.0, abs_tol=1e-9):
            boundary = points[index]
            tail = () if index == len(points) - 1 else (
                boundary,
                *points[index + 1:],
            )
            return BenchmarkReviewTimeSplitV1(
                points[:index + 1],
                tail,
                boundary,
                points[-1],
                times[0],
                times[-1],
            )
        if horizon_s < current_time:
            prior_time = times[index - 1]
            alpha = (horizon_s - prior_time) / (current_time - prior_time)
            prior = points[index - 1]
            current = points[index]
            boundary = (
                prior[0] + alpha * (current[0] - prior[0]),
                prior[1] + alpha * (current[1] - prior[1]),
            )
            return BenchmarkReviewTimeSplitV1(
                (*points[:index], boundary),
                (boundary, *points[index:]),
                boundary,
                points[-1],
                times[0],
                times[-1],
            )
    raise AssertionError("validated review horizon split did not resolve")


def reference_review_curve_v1(
    row: BenchmarkJoinedRowV1,
    *,
    horizon_s: float,
) -> BenchmarkReviewTimeSplitV1:
    """Return the valid hidden reference curve split at a score horizon.

    Args:
        row: Joined benchmark row containing the hidden reference.
        horizon_s: Positive physical score horizon.

    Returns:
        Review-only reference geometry split without extrapolation.
    """

    valid_length = _valid_reference_prefix_length_v1(
        row.reference.trajectory_valid_mask)
    return split_benchmark_review_curve_at_horizon_v1(
        row.reference.trajectory[:valid_length],
        row.reference.trajectory_timestamps_s[:valid_length],
        horizon_s=horizon_s,
    )


def prediction_review_curve_v1(
    row: BenchmarkJoinedRowV1,
    *,
    horizon_s: float,
) -> BenchmarkReviewTimeSplitV1:
    """Return a prediction curve split at a score horizon without GT leakage.

    A legal ``reference_schedule`` prediction remains aligned only by the
    already-approved runner semantics.  The renderer receives no raw hidden
    timestamp from a model adapter.

    Args:
        row: Joined benchmark row with an optional successful prediction.
        horizon_s: Positive physical score horizon.

    Returns:
        Review-only prediction geometry split or an empty split on failure.
    """

    prediction = row.prediction
    if prediction is None:
        return BenchmarkReviewTimeSplitV1((), (), None, None, None, None)
    if prediction.time_basis == "explicit":
        assert prediction.trajectory_timestamps_s is not None
        timestamps_s = prediction.trajectory_timestamps_s
    else:
        valid_length = _valid_reference_prefix_length_v1(
            row.reference.trajectory_valid_mask)
        timestamps_s = row.reference.trajectory_timestamps_s[:len(prediction.
                                                                  trajectory)]
        if len(prediction.trajectory) > valid_length:
            raise ValueError(
                "reference-schedule prediction exceeds valid reference prefix")
    return split_benchmark_review_curve_at_horizon_v1(prediction.trajectory,
                                                      timestamps_s,
                                                      horizon_s=horizon_s)


def load_benchmark_native_adapter_traces_v1(
    path: str | Path,
    *,
    inputs: Mapping[str, BenchmarkInputV1],
    require_complete: bool = False,
) -> dict[str, BenchmarkNativeAdapterTraceV1]:
    """Load and bind public native-adapter traces to frozen benchmark inputs.

    Args:
        path: JSONL trace path emitted by a model-side native adapter.
        inputs: Frozen public inputs keyed by stable sample id.
        require_complete: Whether every frozen input must have one trace row.

    Returns:
        Strict trace mapping keyed by sample id.

    Raises:
        ValueError: If an id/digest drifts, rows duplicate, or complete trace
            coverage was requested but not achieved.
    """

    traces: dict[str, BenchmarkNativeAdapterTraceV1] = {}
    for payload in iter_jsonl(path):
        trace = BenchmarkNativeAdapterTraceV1.model_validate(payload)
        input_row = inputs.get(trace.sample_id)
        if input_row is None:
            raise ValueError(
                f"adapter trace sample is absent from frozen inputs: {trace.sample_id}"
            )
        expected_digest = benchmark_plugin_digest_v1(input_row)
        if trace.benchmark_input_digest != expected_digest:
            raise ValueError(
                f"adapter trace input digest mismatch: {trace.sample_id}")
        if any(index >= len(input_row.images)
               for index in trace.selected_image_indices):
            raise ValueError(
                f"adapter trace selected image index is out of range: {trace.sample_id}"
            )
        expected_roles = tuple(input_row.image_roles[index]
                               for index in trace.selected_image_indices)
        if trace.selected_image_roles != expected_roles:
            raise ValueError(
                f"adapter trace selected image roles drift from input: {trace.sample_id}"
            )
        if trace.sample_id in traces:
            raise ValueError(
                f"duplicate adapter trace sample id: {trace.sample_id}")
        traces[trace.sample_id] = trace
    if require_complete and set(traces) != set(inputs):
        missing = sorted(set(inputs) - set(traces))
        unexpected = sorted(set(traces) - set(inputs))
        raise ValueError(
            "adapter trace coverage is incomplete: "
            f"missing={len(missing)}, unexpected={len(unexpected)}")
    return traces


def summarize_benchmark_native_adapter_funnel_v1(
    *,
    rows: Sequence[BenchmarkJoinedRowV1],
    inputs: Mapping[str, BenchmarkInputV1],
    traces: Mapping[str, BenchmarkNativeAdapterTraceV1],
    horizons_s: Sequence[float],
) -> dict[str, object]:
    """Summarize public adapter admission and evaluator coverage separately.

    Args:
        rows: Joined rows in the explicit review/report denominator.
        inputs: Frozen public inputs keyed by stable sample id.
        traces: Bound adapter trace sidecars keyed by stable sample id.
        horizons_s: Positive score horizons whose joint coverage is reported.

    Returns:
        JSON-compatible funnel.  Native-adapter admission and evaluator
        geometry coverage remain separate to avoid hiding rejected inputs.

    Raises:
        ValueError: If rows, traces, or requested horizons drift from inputs.
    """

    row_ids = [row.reference.sample_id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("review rows must have unique sample ids")
    if any(sample_id not in inputs for sample_id in row_ids):
        raise ValueError("review row is absent from frozen public inputs")
    if any(sample_id not in inputs for sample_id in traces):
        raise ValueError("adapter trace is absent from frozen public inputs")
    ordered_horizons = tuple(float(item) for item in horizons_s)
    if (not ordered_horizons or any(not math.isfinite(item) or item <= 0.0
                                    for item in ordered_horizons)
            or tuple(sorted(set(ordered_horizons))) != ordered_horizons):
        raise ValueError(
            "adapter funnel horizons_s must be unique positive ascending values"
        )
    review_traces = tuple(traces[sample_id] for sample_id in row_ids
                          if sample_id in traces)
    status_counts = {
        status: sum(trace.eligibility_status == status
                    for trace in review_traces)
        for status in (
            "eligible",
            "input_ineligible",
            "native_model_error",
            "output_contract_error",
        )
    }
    reject_reason_counts = {
        reason: sum(trace.reject_reason == reason for trace in review_traces)
        for reason in (
            "insufficient_public_fpv",
            "missing_public_history",
            "missing_public_goal",
            "invalid_scale",
            "unsupported_public_input",
            "native_model_error",
            "output_contract_error",
        )
    }
    trace_present_count = len(review_traces)
    public_fpv_available_count = (
        trace_present_count - reject_reason_counts["insufficient_public_fpv"])
    public_context_available_count = (
        public_fpv_available_count -
        reject_reason_counts["missing_public_history"] -
        reject_reason_counts["missing_public_goal"])
    public_scale_valid_count = (public_context_available_count -
                                reject_reason_counts["invalid_scale"])
    horizon_coverage: dict[str, dict[str, int]] = {}
    for horizon_s in ordered_horizons:
        reference_covered = 0
        prediction_covered = 0
        joint_covered = 0
        for row in rows:
            reference = interpolate_reference_xy_v1(row, horizon_s)
            prediction = interpolate_prediction_xy_v1(row, horizon_s)
            reference_covered += reference is not None
            prediction_covered += prediction is not None
            joint_covered += reference is not None and prediction is not None
        horizon_coverage[f"{horizon_s:g}s"] = {
            "horizon_s": horizon_s,
            "reference_covered_count": reference_covered,
            "prediction_covered_count": prediction_covered,
            "joint_covered_count": joint_covered,
            "denominator_count": len(rows),
        }
    return {
        "schema_version":
        "urbannav_benchmark_native_adapter_funnel_v1",
        "denominator_count":
        len(row_ids),
        "trace_present_count":
        trace_present_count,
        "trace_missing_count":
        sum(sample_id not in traces for sample_id in row_ids),
        "adapter_status_counts":
        status_counts,
        "adapter_reject_reason_counts":
        reject_reason_counts,
        "public_input_gate_counts": {
            "public_fpv_available_count": public_fpv_available_count,
            "public_history_goal_available_count":
            public_context_available_count,
            "public_scale_valid_count": public_scale_valid_count,
        },
        "successful_prediction_count":
        sum(row.prediction is not None for row in rows),
        "horizon_coverage":
        horizon_coverage,
    }


def _valid_reference_prefix_length_v1(mask: Sequence[bool]) -> int:
    """Return the prefix-contiguous hidden reference length for review only."""

    return next((index for index, valid in enumerate(mask) if not valid),
                len(mask))


def _finite_xy_v1(value: Sequence[float]) -> tuple[float, float]:
    """Return a finite two-dimensional local point from a trajectory row."""

    if len(value) < 2:
        raise ValueError("review trajectory row must have at least two values")
    point = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in point):
        raise ValueError("review trajectory contains non-finite xy")
    return point


__all__ = [
    "BENCHMARK_NATIVE_ADAPTER_TRACE_SCHEMA_VERSION",
    "BenchmarkNativeAdapterTraceV1",
    "BenchmarkReviewCaseGeometryV1",
    "BenchmarkReviewRendererBaseV1",
    "BenchmarkReviewTimeSplitV1",
    "NativeAdapterRejectReasonV1",
    "NativeAdapterTraceStatusV1",
    "load_benchmark_native_adapter_traces_v1",
    "prediction_review_curve_v1",
    "reference_review_curve_v1",
    "split_benchmark_review_curve_at_horizon_v1",
    "summarize_benchmark_native_adapter_funnel_v1",
]
