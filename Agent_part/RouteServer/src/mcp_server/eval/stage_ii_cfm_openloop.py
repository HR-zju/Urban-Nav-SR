"""Strict fixed-horizon Stage II CFM open-loop evaluation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from mcp_server.contracts.openloop_benchmark import (
    OpenLoopContextIdentityV1,
    OpenLoopMetricPolicyV1,
    OpenLoopRunManifestV1,
)
from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_HORIZON_STEPS,
    StageIITrajectoryTargetV1,
)
from mcp_server.eval.openloop_common import (
    OpenLoopJoinedRowV1,
    coverage_metrics_v1,
    macro_mean_v1,
)


STAGE_II_OPENLOOP_EVALUATOR_VERSION = (
    "urbannav_stage_ii_cfm_openloop_evaluator_v2_1"
)


def evaluate_stage_ii_cfm_openloop_v1(
    rows: Iterable[OpenLoopJoinedRowV1],
    *,
    run_manifest: OpenLoopRunManifestV1,
    metric_policy: OpenLoopMetricPolicyV1,
) -> dict[str, Any]:
    """Evaluate strict 16-step CFM outputs at fixed physical horizons.

    Predictions must contain exactly 16 finite ``[x_forward, y_left, yaw]``
    rows.  Short, long, or non-finite outputs fail for every horizon instead of
    receiving prefix metrics.  Geometry metrics are conditional on valid output
    while Geometry Coverage retains the complete reference-eligible denominator.

    Args:
        rows: Full-reference Stage II joined rows.
        run_manifest: Frozen model, renderer, context, and sampling identity.
        metric_policy: Frozen horizon and aggregation policy.

    Returns:
        Reliability, context identity, per-horizon geometry, and diagnostics.

    Raises:
        ValueError: If task types, run identity, or prediction context drift.
    """

    materialized = list(rows)
    if run_manifest.stage != "stage_ii":
        raise ValueError("Stage II evaluator requires a Stage II run manifest")
    for joined in materialized:
        if (
            joined.reference.stage != "stage_ii"
            or joined.reference.task_type != "trajectory"
        ):
            raise ValueError(
                "Stage II evaluator received non-trajectory row: "
                f"{joined.reference.sample_id}"
            )

    context_errors: list[dict[str, Any]] = []
    valid_predictions: dict[str, list[list[float]]] = {}
    output_errors: list[dict[str, Any]] = []
    targets: dict[str, StageIITrajectoryTargetV1] = {}
    for joined in materialized:
        reference = joined.reference
        target = _reference_target(reference.target)
        if reference.trajectory_valid_mask != target.action_mask:
            raise ValueError(
                "reference trajectory_valid_mask differs from target: "
                f"{reference.sample_id}"
            )
        targets[reference.sample_id] = target
        prediction = joined.prediction
        if prediction is None:
            continue
        try:
            identity = OpenLoopContextIdentityV1.model_validate(
                prediction.context_identity
            )
            if identity != run_manifest.context_identity:
                raise ValueError("prediction context identity differs from run")
            valid_predictions[reference.sample_id] = _strict_prediction(
                prediction.trajectory_ego_t
            )
        except (TypeError, ValueError) as exc:
            record = {
                "sample_id": reference.sample_id,
                "error": str(exc),
            }
            output_errors.append(record)
            if "context" in str(exc).lower() or "checkpoint" in str(exc).lower():
                context_errors.append(record)

    if context_errors:
        raise ValueError(
            "prediction/run context identity mismatch: "
            f"{context_errors[:8]}"
        )

    horizon_metrics = {
        _horizon_key(horizon): _evaluate_horizon(
            materialized,
            targets,
            valid_predictions,
            horizon_s=horizon,
            step_s=metric_policy.stage_ii_step_s,
        )
        for horizon in metric_policy.stage_ii_horizons_s
    }
    return {
        "evaluator_version": STAGE_II_OPENLOOP_EVALUATOR_VERSION,
        "run_id": run_manifest.run_id,
        "coverage": coverage_metrics_v1(materialized),
        "strict_output": {
            "valid_prediction_count": len(valid_predictions),
            "valid_prediction_rate_full_reference": (
                len(valid_predictions) / len(materialized)
                if materialized
                else 0.0
            ),
            "invalid_output_count": len(output_errors),
            "errors": output_errors[:100],
        },
        "three_pass_reliability": _three_pass_reliability(
            materialized,
            valid_prediction_ids=set(valid_predictions),
        ),
        "context_identity": {
            "policy": run_manifest.context_identity.policy,
            "reasoning_source": (
                run_manifest.context_identity.reasoning_context_source
            ),
            "decision_source": (
                run_manifest.context_identity.decision_context_source
            ),
            "checkpoint_lineage_valid": True,
        },
        "horizons": horizon_metrics,
        "selected_l2": _explicit_selected_l2_report(
            horizon_metrics,
            metric_policy=metric_policy,
        ),
        "decision_stop_drift": _decision_stop_drift(materialized),
        "physical_motion": (
            {
                "status": "N/A",
                "false_move": None,
                "false_freeze": None,
                "reason": "Stage II physical motion threshold policy is not frozen",
            }
            if metric_policy.stage_ii_motion_policy == "not_frozen"
            else {
                "status": "not_implemented",
                "false_move": None,
                "false_freeze": None,
                "reason": "frozen motion policy evaluator is unavailable",
            }
        ),
        "unsupported_claims": {
            "collision": "N/A",
            "route_progress": "N/A",
            "closed_loop_success": "N/A",
        },
    }


def _evaluate_horizon(
    rows: list[OpenLoopJoinedRowV1],
    targets: dict[str, StageIITrajectoryTargetV1],
    predictions: dict[str, list[list[float]]],
    *,
    horizon_s: float,
    step_s: float,
) -> dict[str, Any]:
    """Evaluate one fixed horizon with a reference-owned denominator."""

    step_count = int(round(horizon_s / step_s))
    if step_count < 1 or step_count > STAGE_II_HORIZON_STEPS:
        raise ValueError(f"unsupported Stage II horizon: {horizon_s}")
    if not math.isclose(step_count * step_s, horizon_s, abs_tol=1e-9):
        raise ValueError(f"horizon is off the fixed Stage II grid: {horizon_s}")

    eligible_by_dataset: dict[str, list[OpenLoopJoinedRowV1]] = defaultdict(list)
    for joined in rows:
        target = targets[joined.reference.sample_id]
        if all(target.action_mask[:step_count]):
            eligible_by_dataset[joined.reference.dataset_source].append(joined)

    dataset_metrics: dict[str, dict[str, Any]] = {}
    micro_errors: list[dict[str, float]] = []
    eligible_total = 0
    predicted_total = 0
    for dataset, eligible_rows in sorted(eligible_by_dataset.items()):
        eligible_total += len(eligible_rows)
        errors: list[dict[str, float]] = []
        for joined in eligible_rows:
            sample_id = joined.reference.sample_id
            prediction = predictions.get(sample_id)
            if prediction is None:
                continue
            target = targets[sample_id]
            errors.append(
                _trajectory_errors(
                    target.trajectory_ego_t[:step_count],
                    prediction[:step_count],
                    yaw_eligible=_moving_yaw_eligible(
                        target,
                        step_count=step_count,
                        subgroup=joined.reference.subgroup_metadata,
                    ),
                )
            )
        predicted_total += len(errors)
        micro_errors.extend(errors)
        dataset_metrics[dataset] = _summarize_geometry(
            errors,
            eligible_count=len(eligible_rows),
        )

    active_datasets = [
        value for value in dataset_metrics.values() if value["eligible_count"]
    ]
    return {
        "horizon_s": horizon_s,
        "step_count": step_count,
        "eligible_reference_count": eligible_total,
        "valid_prediction_count": predicted_total,
        "geometry_coverage": (
            predicted_total / eligible_total if eligible_total else None
        ),
        "eligible_dataset_count": len(active_datasets),
        "primary_eligible_dataset_macro": {
            name: macro_mean_v1(
                value[name] for value in active_datasets
            )
            for name in (
                "conditional_ade_m",
                "conditional_fde_m",
                "conditional_l2_m",
                "conditional_lateral_endpoint_error_m",
                "conditional_longitudinal_endpoint_error_m",
                "conditional_yaw_mae_rad",
            )
        },
        "secondary_natural_micro": _summarize_geometry(
            micro_errors,
            eligible_count=eligible_total,
        ),
        "per_dataset": dataset_metrics,
    }


def _trajectory_errors(
    target: list[list[float]],
    prediction: list[list[float]],
    *,
    yaw_eligible: bool,
) -> dict[str, float]:
    """Compute conditional fixed-grid errors for one valid prediction."""

    displacement_errors = [
        math.hypot(pred[0] - ref[0], pred[1] - ref[1])
        for ref, pred in zip(target, prediction)
    ]
    endpoint_ref = target[-1]
    endpoint_pred = prediction[-1]
    result = {
        "ade_m": sum(displacement_errors) / len(displacement_errors),
        "fde_m": displacement_errors[-1],
        "lateral_endpoint_error_m": abs(endpoint_pred[1] - endpoint_ref[1]),
        "longitudinal_endpoint_error_m": abs(endpoint_pred[0] - endpoint_ref[0]),
    }
    if yaw_eligible:
        yaw_errors = [
            abs(_wrap_angle(pred[2] - ref[2]))
            for ref, pred in zip(target, prediction)
        ]
        result["yaw_mae_rad"] = sum(yaw_errors) / len(yaw_errors)
    return result


def _summarize_geometry(
    errors: list[dict[str, float]],
    *,
    eligible_count: int,
) -> dict[str, Any]:
    """Summarize conditional geometry without hiding missing outputs."""

    mapping = {
        "conditional_ade_m": "ade_m",
        "conditional_fde_m": "fde_m",
        # 单候选合同下，固定 horizon 的 L2 就是该截断轨迹的 FDE。
        "conditional_l2_m": "fde_m",
        "conditional_lateral_endpoint_error_m": "lateral_endpoint_error_m",
        "conditional_longitudinal_endpoint_error_m": "longitudinal_endpoint_error_m",
        "conditional_yaw_mae_rad": "yaw_mae_rad",
    }
    result: dict[str, Any] = {
        "eligible_count": eligible_count,
        "valid_prediction_count": len(errors),
        "geometry_coverage": len(errors) / eligible_count if eligible_count else None,
    }
    for output_name, source_name in mapping.items():
        values = [item[source_name] for item in errors if source_name in item]
        result[output_name] = sum(values) / len(values) if values else None
        if source_name == "yaw_mae_rad":
            result["moving_yaw_count"] = len(values)
    return result


def _explicit_selected_l2_report(
    horizon_metrics: dict[str, dict[str, Any]],
    *,
    metric_policy: OpenLoopMetricPolicyV1,
) -> dict[str, Any]:
    """Publish selected-trajectory endpoint L2 at one and two seconds.

    For the current single-trajectory contract, fixed-horizon L2 is exactly
    the endpoint displacement error of the trajectory prefix:

    ``L2@tau = ||p_hat(tau) - p(tau)||_2 = FDE@tau``.

    Args:
        horizon_metrics: Per-horizon evaluator outputs keyed by physical time.
        metric_policy: Frozen aggregation and temporal-grid policy.

    Returns:
        Explicit primary dataset-macro and secondary natural-micro L2 values.

    Raises:
        ValueError: If the frozen evaluator policy does not expose 1 s and 2 s.
    """

    required = {"l2_1s_m": _horizon_key(1.0), "l2_2s_m": _horizon_key(2.0)}
    missing = [key for key in required.values() if key not in horizon_metrics]
    if missing:
        raise ValueError(
            "Stage II selected L2 requires fixed 1 s and 2 s horizons: "
            f"missing={missing}"
        )

    def collect(section: str) -> dict[str, float | None]:
        return {
            output_name: horizon_metrics[horizon_key][section][
                "conditional_l2_m"
            ]
            for output_name, horizon_key in required.items()
        }

    return {
        "definition": "selected_trajectory_endpoint_l2_v1",
        "unit": "meter",
        "primary_aggregation": metric_policy.stage_ii_geometry_primary,
        "primary_eligible_dataset_macro": collect(
            "primary_eligible_dataset_macro"
        ),
        "secondary_natural_micro": collect("secondary_natural_micro"),
    }


def _strict_prediction(value: Any) -> list[list[float]]:
    """Validate exactly 16 finite width-three trajectory rows."""

    if not isinstance(value, list) or len(value) != STAGE_II_HORIZON_STEPS:
        raise ValueError("Stage II prediction must contain exactly 16 rows")
    result: list[list[float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"trajectory row {index} must have width 3")
        parsed = [float(item) for item in row]
        if not all(math.isfinite(item) for item in parsed):
            raise ValueError(f"trajectory row {index} contains NaN/Inf")
        result.append(parsed)
    return result


def _reference_target(payload: dict[str, Any]) -> StageIITrajectoryTargetV1:
    """Load the canonical metric target from a hidden reference payload."""

    target = payload.get("trajectory_target", payload)
    return StageIITrajectoryTargetV1.model_validate(target)


def _moving_yaw_eligible(
    target: StageIITrajectoryTargetV1,
    *,
    step_count: int,
    subgroup: dict[str, Any],
) -> bool:
    """Gate yaw by explicit metadata or deterministic path variation."""

    by_horizon = subgroup.get("moving_yaw_eligible_by_horizon")
    if isinstance(by_horizon, dict):
        key = _horizon_key(step_count * target.policy.target_dt_sec)
        if key in by_horizon:
            return bool(by_horizon[key])
    points = target.trajectory_ego_t[:step_count]
    return any(
        math.hypot(second[0] - first[0], second[1] - first[1]) > 1e-9
        for first, second in zip(points, points[1:])
    )


def _decision_stop_drift(rows: list[OpenLoopJoinedRowV1]) -> dict[str, Any]:
    """Compare predicted Pass-D stop/move state with reference Decision."""

    reference_count = 0
    observed_count = 0
    disagreement = 0
    predicted_stop = 0
    reference_stop = 0
    for joined in rows:
        stamp = joined.reference.direction_stamp
        if stamp is None:
            continue
        reference_count += 1
        expected_stop = str(stamp.label_name) == "stop"
        reference_stop += expected_stop
        prediction = joined.prediction
        if prediction is None:
            continue
        predicted_label = prediction.metadata.get("decision_context_label")
        if predicted_label is None:
            continue
        observed_count += 1
        actual_stop = str(predicted_label) == "stop"
        predicted_stop += actual_stop
        disagreement += actual_stop != expected_stop
    return {
        "reference_count": reference_count,
        "observed_count": observed_count,
        "coverage": observed_count / reference_count if reference_count else None,
        "stop_move_disagreement_rate": (
            disagreement / observed_count if observed_count else None
        ),
        "reference_stop_count": reference_stop,
        "predicted_stop_count": predicted_stop,
        "interpretation": "categorical Pass-D drift, not physical motion error",
    }


def _three_pass_reliability(
    rows: list[OpenLoopJoinedRowV1],
    *,
    valid_prediction_ids: set[str],
) -> dict[str, Any]:
    """Reconstruct Pass R/D/T attempts from the failure ledger."""

    failure_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    explicit_failures = 0
    missing = 0
    for row in rows:
        if row.failure is not None:
            explicit_failures += 1
            failure_counts[row.failure.attempted_pass] += 1
            code_counts[row.failure.failure_code] += 1
        elif row.prediction is None:
            missing += 1
    total = len(rows)
    pass_r_attempts = total - missing
    pass_r_success = pass_r_attempts - failure_counts["pass_r"]
    pass_d_attempts = pass_r_success
    pass_d_success = pass_d_attempts - failure_counts["pass_d"]
    pass_t_attempts = pass_d_success
    pass_t_success = len(valid_prediction_ids)
    invalid_pass_t_outputs = sum(
        row.prediction is not None
        and row.reference.sample_id not in valid_prediction_ids
        for row in rows
    )
    if invalid_pass_t_outputs:
        code_counts["invalid_pass_t_output"] += invalid_pass_t_outputs
    if pass_t_success > pass_t_attempts:
        raise ValueError("valid Pass-T outputs exceed reconstructed attempts")
    return {
        "reference_count": total,
        "explicit_failure_count": explicit_failures,
        "unattempted_count": missing,
        "passes": {
            "pass_r": {"attempted": pass_r_attempts, "succeeded": pass_r_success},
            "pass_d": {"attempted": pass_d_attempts, "succeeded": pass_d_success},
            "pass_t": {"attempted": pass_t_attempts, "succeeded": pass_t_success},
        },
        "failure_codes": dict(sorted(code_counts.items())),
    }


def _wrap_angle(value: float) -> float:
    """Wrap radians to the closed-open interval [-pi, pi)."""

    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _horizon_key(horizon_s: float) -> str:
    """Return a stable JSON key for one horizon."""

    return f"{float(horizon_s):.3f}s"


__all__ = [
    "STAGE_II_OPENLOOP_EVALUATOR_VERSION",
    "evaluate_stage_ii_cfm_openloop_v1",
]
