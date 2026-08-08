"""Deterministic post-hoc slice review and coordinate diagnostics.

The functions here consume a *completed* benchmark ledger.  They are never
called by a model adapter and must not be used to construct model-visible
inputs: scenario labels, hidden reference geometry, and L2 errors are all
evaluator-only information.  Their purpose is to make error diagnosis
repeatable before changing an input adapter or attributing a result to domain
shift.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from statistics import median
from typing import Any, Sequence

from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkJoinedRowV1
from mcp_server.eval.openloop.benchmark_plugin_metrics import (
    interpolate_prediction_xy_v1,
    interpolate_reference_xy_v1,
)

BENCHMARK_SLICE_REVIEW_SELECTION_SCHEMA_VERSION = "urbannav_benchmark_slice_review_selection_v1"
BENCHMARK_SLICE_REVIEW_DIAGNOSIS_SCHEMA_VERSION = "urbannav_benchmark_slice_review_diagnosis_v1"
BENCHMARK_COORDINATE_DIAGNOSTIC_SCHEMA_VERSION = "urbannav_benchmark_coordinate_diagnostic_v1"
BENCHMARK_REFERENCE_MOTION_DIAGNOSTIC_SCHEMA_VERSION = "urbannav_benchmark_reference_motion_diagnostic_v1"
BENCHMARK_SLICE_REVIEW_POLICY_ID_V1 = "stratified_l2_tercile_slice_review_v1"
GLOBAL_COORDINATE_ABSOLUTE_IMPROVEMENT_THRESHOLD_M_V1 = 0.05
GLOBAL_COORDINATE_RELATIVE_IMPROVEMENT_THRESHOLD_V1 = 0.01


def build_benchmark_slice_review_selection_v1(
    rows: Sequence[BenchmarkJoinedRowV1],
    *,
    slice_labels: Sequence[str],
    horizon_s: float,
    per_slice: int = 20,
    seed: int = 20260808,
) -> dict[str, Any]:
    r"""Build a stable low/mid/high-L2 post-hoc review selection.

    For every overlapping scenario slice, the eligible prediction/reference
    pairs are sorted by their fixed-horizon error

    \[
    e_i(t) = \lVert \hat p_i(t) - p_i(t) \rVert_2.
    \]

    The requested count is taken from the lower rank band, the centered rank
    band, and the upper rank band.  Ties use ``SHA256(seed, slice, sample_id)``
    rather than process-local randomness.  One physical sample may belong to
    multiple slice selections, but appears once in the top-level ``entries``
    ledger with all its selection reasons.

    Args:
        rows: Complete frozen reference denominator joined to output ledgers.
        slice_labels: Ordered, overlap-permitted scenario labels to audit.
        horizon_s: Physical horizon used only for post-hoc L2 strata.
        per_slice: Maximum requested review examples for every slice.
        seed: Stable tie-break seed recorded in the selection artifact.

    Returns:
        JSON-compatible evaluator-only selection ledger.  It includes no
        model input construction and explicitly records insufficient support.

    Raises:
        ValueError: If labels, horizon, or requested review count are invalid.
    """

    labels = _validate_selection_args_v1(slice_labels,
                                         horizon_s=horizon_s,
                                         per_slice=per_slice)
    selected: dict[str, dict[str, Any]] = {}
    selected_order: list[str] = []
    slice_summaries: dict[str, dict[str, Any]] = {}
    for label in labels:
        candidates = [
            row for row in rows if label in row.reference.scenario_labels
        ]
        comparable, unavailable = _score_slice_candidates_v1(
            candidates, horizon_s=horizon_s, seed=seed, label=label)
        chosen = _select_rank_strata_v1(comparable, requested_count=per_slice)
        slice_summaries[label] = {
            "reference_count":
            len(candidates),
            "comparable_count":
            len(comparable),
            "unavailable_count":
            len(unavailable),
            "selected_count":
            len(chosen),
            "requested_count":
            per_slice,
            "selection_shortfall":
            max(0, per_slice - len(chosen)),
            "selected_l2_m":
            _summarize_values_v1([item["l2_m"] for item in chosen]),
        }
        for item in chosen:
            sample_id = item["row"].reference.sample_id
            entry = selected.get(sample_id)
            reason = {
                "slice_label": label,
                "stratum": item["stratum"],
                "l2_m": item["l2_m"],
                "rank": item["rank"],
                "comparable_count": len(comparable),
            }
            if entry is None:
                row = item["row"]
                entry = _selection_entry_v1(row, horizon_s=horizon_s)
                entry["selection_reasons"] = []
                selected[sample_id] = entry
                selected_order.append(sample_id)
            entry["selection_reasons"].append(reason)
    return {
        "schema_version": BENCHMARK_SLICE_REVIEW_SELECTION_SCHEMA_VERSION,
        "policy_id": BENCHMARK_SLICE_REVIEW_POLICY_ID_V1,
        "evaluator_only": True,
        "not_model_visible": True,
        "seed": seed,
        "horizon_s": horizon_s,
        "requested_per_slice": per_slice,
        "slice_labels": list(labels),
        "slice_summaries": slice_summaries,
        "selected_unique_count": len(selected_order),
        "selected_sample_ids": selected_order,
        "entries": [selected[sample_id] for sample_id in selected_order],
    }


def build_benchmark_slice_review_diagnosis_template_v1(
    selection: dict[str, Any], ) -> list[dict[str, Any]]:
    """Create human-fillable diagnosis rows for a fixed selection ledger.

    Args:
        selection: Output from :func:`build_benchmark_slice_review_selection_v1`.

    Returns:
        One JSON-compatible pending-review row per unique selected sample.

    Raises:
        ValueError: If the expected selection schema or entries are absent.
    """

    if selection.get("schema_version"
                     ) != BENCHMARK_SLICE_REVIEW_SELECTION_SCHEMA_VERSION:
        raise ValueError(
            "slice-review diagnosis requires a v1 selection ledger")
    entries = selection.get("entries")
    if not isinstance(entries, list):
        raise ValueError("slice-review selection entries must be a list")
    template: list[dict[str, Any]] = []
    for entry in entries:
        template.append({
            "schema_version": BENCHMARK_SLICE_REVIEW_DIAGNOSIS_SCHEMA_VERSION,
            "sample_id": entry["sample_id"],
            "parent_sample_id": entry["parent_sample_id"],
            "selected_for": entry["selection_reasons"],
            "fixed_horizon_l2_m": entry["fixed_horizon_l2_m"],
            "review_status": "pending",
            "input_media_readable": None,
            "point_goal_prompt_consistent": None,
            "gt_trajectory_consistent": None,
            "prediction_coordinate_frame_consistent": None,
            "global_axis_or_sign_issue": None,
            "visible_domain_shift_plausible": None,
            "final_diagnosis": None,
            "reviewer_note": None,
        })
    return template


def select_benchmark_slice_review_rows_v1(
    rows: Sequence[BenchmarkJoinedRowV1],
    *,
    selection: dict[str, Any],
) -> list[BenchmarkJoinedRowV1]:
    """Resolve a fixed selection artifact back to joined rows in artifact order.

    Args:
        rows: Full joined benchmark ledger.
        selection: Output from :func:`build_benchmark_slice_review_selection_v1`.

    Returns:
        Unique selected joined rows in the selection's deterministic order.

    Raises:
        ValueError: If the selection schema or selected stable IDs drift.
    """

    if selection.get("schema_version"
                     ) != BENCHMARK_SLICE_REVIEW_SELECTION_SCHEMA_VERSION:
        raise ValueError(
            "slice-review row resolution requires a v1 selection ledger")
    by_id = {row.reference.sample_id: row for row in rows}
    resolved: list[BenchmarkJoinedRowV1] = []
    for sample_id in selection.get("selected_sample_ids", []):
        row = by_id.get(sample_id)
        if row is None:
            raise ValueError(
                f"slice-review selection sample is absent from ledger: {sample_id}"
            )
        resolved.append(row)
    return resolved


def evaluate_benchmark_coordinate_diagnostic_v1(
    rows: Sequence[BenchmarkJoinedRowV1],
    *,
    horizon_s: float,
) -> dict[str, Any]:
    r"""Diagnose a global xy axis/sign mismatch without altering predictions.

    Every valid prediction/reference pair is compared after each signed
    permutation in the two-dimensional dihedral family.  Identity is the
    adapter-declared mapping.  If another global transform materially lowers
    L2, the result is a *diagnostic candidate* for adapter/coordinate audit;
    it does not authorize retrospectively changing an official score.

    Args:
        rows: Full or explicitly selected joined benchmark ledger.
        horizon_s: Physical horizon used for the post-hoc comparison.

    Returns:
        Observed coordinate stamps and L2 reports for all eight transforms.

    Raises:
        ValueError: If the requested horizon is not positive and finite.
    """

    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError(
            "coordinate diagnostic horizon_s must be positive and finite")
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row in rows:
        reference = interpolate_reference_xy_v1(row, horizon_s)
        prediction = interpolate_prediction_xy_v1(row, horizon_s)
        if reference is not None and prediction is not None:
            pairs.append((reference, prediction))
    reports: list[dict[str, Any]] = []
    for transform_id, swap_axes, x_sign, y_sign in _SIGNED_PERMUTATIONS_V1:
        errors = [
            math.hypot(
                reference[0] - transformed[0],
                reference[1] - transformed[1],
            ) for reference, prediction in pairs
            for transformed in (_apply_signed_permutation_v1(
                prediction, swap_axes, x_sign, y_sign), )
        ]
        reports.append({
            "transform_id": transform_id,
            **_summarize_values_v1(errors)
        })
    ordered = sorted(
        reports,
        key=lambda item:
        (math.inf
         if item["mean"] is None else item["mean"], item["transform_id"]))
    identity = next(item for item in reports
                    if item["transform_id"] == "identity")
    best = ordered[0]
    identity_mean = identity["mean"]
    best_mean = best["mean"]
    improvement = None if identity_mean is None or best_mean is None else identity_mean - best_mean
    return {
        "schema_version":
        BENCHMARK_COORDINATE_DIAGNOSTIC_SCHEMA_VERSION,
        "evaluator_only":
        True,
        "not_an_official_metric":
        True,
        "horizon_s":
        horizon_s,
        "reference_count":
        len(rows),
        "comparable_count":
        len(pairs),
        "observed_contract_stamps":
        _observed_coordinate_stamps_v1(rows),
        "transforms":
        reports,
        "best_transform_id":
        best["transform_id"] if pairs else None,
        "identity_mean_l2_m":
        identity_mean,
        "best_mean_l2_m":
        best_mean,
        "identity_minus_best_mean_l2_m":
        improvement,
        "material_improvement_threshold_m":
        GLOBAL_COORDINATE_ABSOLUTE_IMPROVEMENT_THRESHOLD_M_V1,
        "material_relative_improvement_threshold":
        GLOBAL_COORDINATE_RELATIVE_IMPROVEMENT_THRESHOLD_V1,
        "material_global_transform_candidate":
        _is_material_coordinate_improvement_v1(
            identity_mean=identity_mean,
            improvement=improvement,
        ),
        "interpretation":
        _coordinate_interpretation_v1(
            comparable_count=len(pairs),
            best_transform_id=best["transform_id"] if pairs else None,
            identity_mean=identity_mean,
            improvement=improvement,
        ),
    }


def evaluate_benchmark_reference_motion_diagnostic_v1(
    rows: Sequence[BenchmarkJoinedRowV1],
    *,
    horizon_s: float,
) -> dict[str, Any]:
    """Group fixed-horizon errors by hidden-reference displacement magnitude.

    This is a post-hoc diagnostic, not a new benchmark metric and not a
    predicted stop label.  Its bins help distinguish an all-regime coordinate
    failure from a transfer failure concentrated in nearly-static or moving
    reference states.

    Args:
        rows: Full or explicitly selected joined benchmark ledger.
        horizon_s: Physical horizon used for hidden-reference interpolation.

    Returns:
        Evaluator-only radial-displacement bucket report with model/reference
        displacement and L2 summaries.

    Raises:
        ValueError: If the requested horizon is not positive and finite.
    """

    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError(
            "reference-motion diagnostic horizon_s must be positive and finite"
        )
    buckets: dict[str, list[tuple[float, float, float]]] = {
        key: []
        for key, _, _ in _REFERENCE_MOTION_BINS_V1
    }
    for row in rows:
        reference = interpolate_reference_xy_v1(row, horizon_s)
        prediction = interpolate_prediction_xy_v1(row, horizon_s)
        if reference is None or prediction is None:
            continue
        reference_radius = math.hypot(*reference)
        prediction_radius = math.hypot(*prediction)
        error = math.hypot(reference[0] - prediction[0],
                           reference[1] - prediction[1])
        bucket = _reference_motion_bucket_v1(reference_radius)
        buckets[bucket].append((reference_radius, prediction_radius, error))
    return {
        "schema_version": BENCHMARK_REFERENCE_MOTION_DIAGNOSTIC_SCHEMA_VERSION,
        "evaluator_only": True,
        "not_an_official_metric": True,
        "horizon_s": horizon_s,
        "reference_count": len(rows),
        "comparable_count": sum(len(values) for values in buckets.values()),
        "bins": {
            key: {
                "lower_inclusive_m":
                lower,
                "upper_exclusive_m":
                upper,
                "count":
                len(values),
                "reference_displacement_m":
                _summarize_values_v1([item[0] for item in values]),
                "prediction_displacement_m":
                _summarize_values_v1([item[1] for item in values]),
                "l2_m":
                _summarize_values_v1([item[2] for item in values]),
            }
            for key, lower, upper in _REFERENCE_MOTION_BINS_V1
            for values in (buckets[key], )
        },
    }


_SIGNED_PERMUTATIONS_V1 = (
    ("identity", False, 1.0, 1.0),
    ("x_sign_flip", False, -1.0, 1.0),
    ("y_sign_flip", False, 1.0, -1.0),
    ("rotate_180", False, -1.0, -1.0),
    ("swap_xy", True, 1.0, 1.0),
    ("swap_xy_x_sign_flip", True, -1.0, 1.0),
    ("swap_xy_y_sign_flip", True, 1.0, -1.0),
    ("swap_xy_rotate_180", True, -1.0, -1.0),
)

_REFERENCE_MOTION_BINS_V1 = (
    ("reference_displacement_lt_0_25m", 0.0, 0.25),
    ("reference_displacement_0_25_to_1m", 0.25, 1.0),
    ("reference_displacement_1_to_3m", 1.0, 3.0),
    ("reference_displacement_ge_3m", 3.0, math.inf),
)


def _validate_selection_args_v1(
    slice_labels: Sequence[str],
    *,
    horizon_s: float,
    per_slice: int,
) -> tuple[str, ...]:
    """Validate public selection parameters before touching an evaluator ledger."""

    labels = tuple(slice_labels)
    if not labels or any(not isinstance(item, str) or not item
                         for item in labels):
        raise ValueError(
            "slice-review requires one or more non-empty slice labels")
    if len(set(labels)) != len(labels):
        raise ValueError("slice-review labels must be unique")
    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("slice-review horizon_s must be positive and finite")
    if per_slice <= 0:
        raise ValueError("slice-review per_slice must be positive")
    return labels


def _score_slice_candidates_v1(
    rows: Sequence[BenchmarkJoinedRowV1],
    *,
    horizon_s: float,
    seed: int,
    label: str,
) -> tuple[list[dict[str, Any]], list[BenchmarkJoinedRowV1]]:
    """Compute a stable post-hoc L2 order and retain unavailable rows separately."""

    scored: list[dict[str, Any]] = []
    unavailable: list[BenchmarkJoinedRowV1] = []
    for row in rows:
        reference = interpolate_reference_xy_v1(row, horizon_s)
        prediction = interpolate_prediction_xy_v1(row, horizon_s)
        if reference is None or prediction is None:
            unavailable.append(row)
            continue
        score = math.hypot(reference[0] - prediction[0],
                           reference[1] - prediction[1])
        scored.append({
            "row":
            row,
            "l2_m":
            score,
            "tie_break":
            _tie_break_v1(seed=seed,
                          label=label,
                          sample_id=row.reference.sample_id),
        })
    scored.sort(key=lambda item: (item["l2_m"], item["tie_break"], item["row"].
                                  reference.sample_id))
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank
    return scored, unavailable


def _select_rank_strata_v1(
    scored: Sequence[dict[str, Any]],
    *,
    requested_count: int,
) -> list[dict[str, Any]]:
    """Select lower, centered, and upper rank bands without duplicate cases."""

    if not scored:
        return []
    count = min(requested_count, len(scored))
    base_count, remainder = divmod(count, 3)
    lower_count = base_count + (1 if remainder >= 1 else 0)
    upper_count = base_count + (1 if remainder >= 2 else 0)
    middle_count = base_count
    lower = list(scored[:lower_count])
    upper_start = max(lower_count, len(scored) - upper_count)
    upper = list(scored[upper_start:])
    used_indices = set(range(lower_count)) | set(
        range(upper_start, len(scored)))
    remaining = [
        item for index, item in enumerate(scored) if index not in used_indices
    ]
    middle_start = max(0, (len(remaining) - middle_count) // 2)
    middle = remaining[middle_start:middle_start + middle_count]
    selected: list[dict[str, Any]] = []
    for stratum, values in (("low_l2", lower), ("mid_l2", middle), ("high_l2",
                                                                    upper)):
        for item in values:
            selected.append({**item, "stratum": stratum})
    return selected


def _selection_entry_v1(row: BenchmarkJoinedRowV1, *,
                        horizon_s: float) -> dict[str, Any]:
    """Render only reviewer-needed post-hoc identity and contract metadata."""

    reference = row.reference
    prediction = row.prediction
    target = interpolate_reference_xy_v1(row, horizon_s)
    predicted = interpolate_prediction_xy_v1(row, horizon_s)
    error = None if target is None or predicted is None else math.hypot(
        target[0] - predicted[0], target[1] - predicted[1])
    return {
        "sample_id":
        reference.sample_id,
        "parent_sample_id":
        reference.parent_sample_id,
        "dataset_source":
        reference.dataset_source,
        "scenario_labels":
        list(reference.scenario_labels),
        "fixed_horizon_l2_m":
        error,
        "reference_xy_at_horizon_m":
        None if target is None else [target[0], target[1]],
        "prediction_xy_at_horizon_m":
        None if predicted is None else [predicted[0], predicted[1]],
        "reference_displacement_m":
        None if target is None else math.hypot(*target),
        "prediction_displacement_m":
        None if predicted is None else math.hypot(*predicted),
        "reference_coordinate_frame":
        reference.coordinate_frame,
        "prediction_geometry_id":
        None if prediction is None else prediction.prediction_geometry_id,
        "prediction_time_basis":
        None if prediction is None else prediction.time_basis,
        "coordinate_projection_id":
        None if prediction is None else
        prediction.metadata.get("coordinate_projection_id"),
    }


def _observed_coordinate_stamps_v1(
        rows: Sequence[BenchmarkJoinedRowV1]) -> dict[str, dict[str, int]]:
    """Count versioned coordinate stamps exactly as emitted by the ledger."""

    def count(values: Sequence[str | None]) -> dict[str, int]:
        return dict(
            sorted(
                Counter("<absent>" if value is None else str(value)
                        for value in values).items()))

    predictions = [
        row.prediction for row in rows if row.prediction is not None
    ]
    return {
        "reference_coordinate_frame":
        count([row.reference.coordinate_frame for row in rows]),
        "prediction_geometry_id":
        count([item.prediction_geometry_id for item in predictions]),
        "prediction_time_basis":
        count([item.time_basis for item in predictions]),
        "coordinate_projection_id":
        count([
            item.metadata.get("coordinate_projection_id")
            for item in predictions
        ]),
    }


def _apply_signed_permutation_v1(
    point: tuple[float, float],
    swap_axes: bool,
    x_sign: float,
    y_sign: float,
) -> tuple[float, float]:
    """Apply one signed coordinate permutation for a diagnostic counterfactual."""

    x_value, y_value = (point[1], point[0]) if swap_axes else point
    return x_sign * x_value, y_sign * y_value


def _reference_motion_bucket_v1(radius_m: float) -> str:
    """Return the closed-open radial displacement bin for a finite reference."""

    for key, _lower, upper in _REFERENCE_MOTION_BINS_V1:
        if radius_m < upper:
            return key
    raise ValueError("reference displacement must be finite")


def _coordinate_interpretation_v1(
    *,
    comparable_count: int,
    best_transform_id: str | None,
    identity_mean: float | None,
    improvement: float | None,
) -> str:
    """Return a bounded interpretation without silently changing official scores."""

    if comparable_count == 0:
        return "no_comparable_pairs"
    if best_transform_id == "identity" or not _is_material_coordinate_improvement_v1(
            identity_mean=identity_mean,
            improvement=improvement,
    ):
        return "no_material_global_axis_or_sign_transform_improves_identity"
    return "global_axis_or_sign_transform_improves_l2_candidate"


def _is_material_coordinate_improvement_v1(
    *,
    identity_mean: float | None,
    improvement: float | None,
) -> bool:
    """Reject incidental asymmetric-slice gains before flagging a global bug."""

    if identity_mean is None or improvement is None or identity_mean <= 0.0:
        return False
    return (improvement
            >= GLOBAL_COORDINATE_ABSOLUTE_IMPROVEMENT_THRESHOLD_M_V1
            and improvement / identity_mean
            >= GLOBAL_COORDINATE_RELATIVE_IMPROVEMENT_THRESHOLD_V1)


def _summarize_values_v1(values: Sequence[float]) -> dict[str, float | None]:
    """Summarize finite values without fabricating a zero for missing support."""

    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": median(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _tie_break_v1(*, seed: int, label: str, sample_id: str) -> str:
    """Return a deterministic digest used only to order equal-error candidates."""

    payload = f"{seed}:{label}:{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BENCHMARK_COORDINATE_DIAGNOSTIC_SCHEMA_VERSION",
    "BENCHMARK_REFERENCE_MOTION_DIAGNOSTIC_SCHEMA_VERSION",
    "BENCHMARK_SLICE_REVIEW_DIAGNOSIS_SCHEMA_VERSION",
    "BENCHMARK_SLICE_REVIEW_POLICY_ID_V1",
    "BENCHMARK_SLICE_REVIEW_SELECTION_SCHEMA_VERSION",
    "GLOBAL_COORDINATE_ABSOLUTE_IMPROVEMENT_THRESHOLD_M_V1",
    "GLOBAL_COORDINATE_RELATIVE_IMPROVEMENT_THRESHOLD_V1",
    "build_benchmark_slice_review_diagnosis_template_v1",
    "build_benchmark_slice_review_selection_v1",
    "evaluate_benchmark_coordinate_diagnostic_v1",
    "evaluate_benchmark_reference_motion_diagnostic_v1",
    "select_benchmark_slice_review_rows_v1",
]
