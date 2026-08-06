"""Offline UrbanNav v2 metrics with fixed-denominator ranking scores.

The v2 evaluator preserves the v1 destination and route matching semantics but
adds ranking-sensitive metrics and task-aware success aggregation. It consumes
only frozen ground truth and saved predictions; it never invokes a model or map
provider.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from mcp_server.eval.urbannav.urbannav_metrics import (
    UrbanNavEvalConfig,
    evaluate_case,
    evaluate_extended_case_metrics,
    match_destination,
)
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPrediction


URBANNAV_EVAL_V2_VERSION = "urbannav_eval_v2"


@dataclass(frozen=True)
class UrbanNavEvalV2Config:
    """Configure fixed-denominator ranking and strict task aggregation.

    Args:
        destination_tolerance_m: Coordinate fallback tolerance used after
            stable-id matching.
        endpoint_tolerance_m: Maximum route endpoint error accepted by v1 route
            evaluation.
        multi_stop_continuity_tolerance_m: Maximum adjacent route-leg gap.
        hit_cutoffs: Candidate ranking cutoffs written to each v2 row.
        mrr_cutoffs: Reciprocal-rank cutoffs written to each v2 row.
    """

    destination_tolerance_m: float = 30.0
    endpoint_tolerance_m: float = 10.0
    multi_stop_continuity_tolerance_m: float = 5.0
    hit_cutoffs: tuple[int, ...] = (1, 3, 5)
    mrr_cutoffs: tuple[int, ...] = (3, 5)

    def base_config(self) -> UrbanNavEvalConfig:
        """Build the compatible v1 geometry configuration.

        Returns:
            Base evaluator configuration used for destination and route checks.
        """

        return UrbanNavEvalConfig(
            destination_tolerance_m=self.destination_tolerance_m,
            endpoint_tolerance_m=self.endpoint_tolerance_m,
            multi_stop_continuity_tolerance_m=self.multi_stop_continuity_tolerance_m,
        )


def evaluate_predictions_v2(
    cases: Iterable[UrbanNavCase],
    predictions: Iterable[UrbanNavPrediction],
    *,
    config: UrbanNavEvalV2Config | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    r"""Evaluate saved predictions with fixed-denominator ranking metrics.

    For a required destination whose first acceptable candidate occurs at rank
    :math:`r`, the per-target metrics are

    \[
    Hit@k = \mathbb{1}[r \leq k], \qquad
    MRR@k = \mathbb{1}[r \leq k] / r.
    \]

    Missing predictions, empty candidate lists, and unmatched candidates all
    contribute zero. Multi-stop cases average these values over required stops,
    then every case receives equal weight in the dataset aggregate.

    Args:
        cases: Frozen UrbanNav cases.
        predictions: Existing model predictions.
        config: Optional v2 metric configuration.

    Returns:
        Tuple of per-case rows, aggregate summary, and sensitivity report.
    """

    cfg = config or UrbanNavEvalV2Config()
    case_rows = list(cases)
    predictions_by_id = {prediction.case_id: prediction for prediction in predictions}
    rows = [
        _evaluate_case_v2(case, predictions_by_id.get(case.case_id), config=cfg)
        for case in case_rows
    ]
    return rows, summarize_v2_rows(rows), build_metric_sensitivity(rows)


def write_eval_v2_artifacts(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    sensitivity: dict[str, Any],
    output_dir: str | Path,
    run_id: str,
    dataset_path: str | Path,
    predictions_path: str | Path,
    config: UrbanNavEvalV2Config,
) -> dict[str, str]:
    """Write deterministic v2 evaluation artifacts.

    Args:
        rows: Per-case v2 rows.
        summary: Aggregate v2 summary.
        sensitivity: Metric-threshold sensitivity report.
        output_dir: Destination directory, conventionally ``eval_v2``.
        run_id: Stable evaluation run identifier.
        dataset_path: Frozen GT path used for evaluation.
        predictions_path: Existing prediction artifact used for evaluation.
        config: Effective v2 metric configuration.

    Returns:
        Mapping from artifact name to written path.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows_path = root / "eval_rows.jsonl"
    summary_path = root / "summary.json"
    sensitivity_path = root / "metric_sensitivity.json"
    manifest_path = root / "manifest.json"

    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    sensitivity_path.write_text(
        json.dumps(sensitivity, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": URBANNAV_EVAL_V2_VERSION,
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "predictions_path": str(predictions_path),
        "case_count": len(rows),
        "config": {
            "destination_tolerance_m": config.destination_tolerance_m,
            "endpoint_tolerance_m": config.endpoint_tolerance_m,
            "multi_stop_continuity_tolerance_m": config.multi_stop_continuity_tolerance_m,
            "hit_cutoffs": list(config.hit_cutoffs),
            "mrr_cutoffs": list(config.mrr_cutoffs),
            "fixed_denominator": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "rows": str(rows_path),
        "summary": str(summary_path),
        "metric_sensitivity": str(sensitivity_path),
        "manifest": str(manifest_path),
    }


def summarize_v2_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate v2 rows overall and by task category.

    Args:
        rows: Per-case v2 evaluation rows.

    Returns:
        JSON-ready summary with all cases retained in every denominator.
    """

    return {
        "schema_version": URBANNAV_EVAL_V2_VERSION,
        "summary": _summarize_group(rows),
        "by_category": {
            category: _summarize_group(group)
            for category, group in _group_rows(rows, "category").items()
        },
    }


def build_metric_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build cutoff and geometry-threshold sensitivity diagnostics.

    Route diagnostics are conditioned on destination success and are separated
    by category. They are not promoted to main-table metrics because replay
    routes are deterministic after destination selection.

    Args:
        rows: Per-case v2 rows.

    Returns:
        JSON-ready sensitivity report.
    """

    return {
        "schema_version": URBANNAV_EVAL_V2_VERSION,
        "ranking": {
            "task_hit@1": _mean(rows, "task_hit@1"),
            "task_hit@3": _mean(rows, "task_hit@3"),
            "task_hit@5": _mean(rows, "task_hit@5"),
            "task_mrr@3": _mean(rows, "task_mrr@3"),
            "task_mrr@5": _mean(rows, "task_mrr@5"),
            "candidate_coverage": _mean(rows, "candidate_coverage"),
            "fixed_denominator_cases": len(rows),
        },
        "route_conditioned_on_destination_success": {
            "overall": _route_sensitivity(rows),
            "by_category": {
                category: _route_sensitivity(group)
                for category, group in _group_rows(rows, "category").items()
            },
        },
    }


def _evaluate_case_v2(
    case: UrbanNavCase,
    prediction: UrbanNavPrediction | None,
    *,
    config: UrbanNavEvalV2Config,
) -> dict[str, Any]:
    """Evaluate one case while preserving the v1 audit fields."""

    base_row = evaluate_case(case, prediction, config=config.base_config())
    ranks = (
        [None for _ in _required_candidate_ranks(case, prediction, tolerance_m=config.destination_tolerance_m)]
        if base_row.prediction_exception
        else _required_candidate_ranks(case, prediction, tolerance_m=config.destination_tolerance_m)
    )
    if not ranks:
        ranks = [None]
    ranking: dict[str, float] = {}
    for cutoff in config.hit_cutoffs:
        ranking[f"task_hit@{cutoff}"] = sum(
            1.0 if rank is not None and rank <= cutoff else 0.0 for rank in ranks
        ) / len(ranks)
    for cutoff in config.mrr_cutoffs:
        ranking[f"task_mrr@{cutoff}"] = sum(
            (1.0 / rank) if rank is not None and rank <= cutoff else 0.0 for rank in ranks
        ) / len(ranks)

    row = base_row.to_dict()
    if prediction is None or base_row.prediction_exception:
        # 缺失或异常 prediction 仍保留任务类型，并生成显式的扩展失败指标。
        row["extended_metrics"] = evaluate_extended_case_metrics(
            case,
            None,
            config=config.base_config(),
        )
    task_success = _task_success(row)
    task_spl = _task_spl(row, task_success=task_success)
    if base_row.prediction_exception:
        row.update(
            {
                "destination_success": False,
                "route_success": False,
                "route_ratio": None,
                "spl": 0.0,
            }
        )
        task_success = False
        task_spl = 0.0
    row.update(
        {
            "schema_version": URBANNAV_EVAL_V2_VERSION,
            "required_destination_count": len(ranks),
            "candidate_ranks": ranks,
            "candidate_coverage": sum(rank is not None for rank in ranks) / len(ranks),
            **ranking,
            "task_success_v2": task_success,
            "task_spl_v2": task_spl,
        }
    )
    return row


def _required_candidate_ranks(
    case: UrbanNavCase,
    prediction: UrbanNavPrediction | None,
    *,
    tolerance_m: float,
) -> list[int | None]:
    """Return first acceptable rank for each required destination or stop."""

    if prediction is None:
        if case.category == "multi_stop_route":
            return [None for _ in _gt_route_legs(case)] or [None]
        return [None]
    if case.category != "multi_stop_route":
        acceptable = list(case.acceptable_destinations)
        if not acceptable and case.gt_destination is not None:
            acceptable = [case.gt_destination]
        return [_first_acceptable_rank(prediction.candidate_destinations, acceptable, tolerance_m=tolerance_m)]

    pred_legs = list(prediction.route_legs or [])
    ranks: list[int | None] = []
    for index, gt_leg in enumerate(_gt_route_legs(case)):
        acceptable = _parse_destinations(
            gt_leg.get("acceptable_destinations") or [gt_leg.get("gt_destination")]
        )
        pred_leg = pred_legs[index] if index < len(pred_legs) and isinstance(pred_legs[index], dict) else {}
        candidates = _parse_destinations(pred_leg.get("candidate_destinations"))
        ranks.append(_first_acceptable_rank(candidates, acceptable, tolerance_m=tolerance_m))
    return ranks or [None]


def _first_acceptable_rank(
    candidates: Iterable[UrbanNavDestination],
    acceptable: list[UrbanNavDestination],
    *,
    tolerance_m: float,
) -> int | None:
    """Return the one-based rank of the first acceptable candidate."""

    for rank, candidate in enumerate(candidates, start=1):
        if match_destination(candidate, acceptable, tolerance_m=tolerance_m) is not None:
            return rank
    return None


def _task_success(row: dict[str, Any]) -> bool:
    """Return strict task-aware end-to-end success."""

    extended = row.get("extended_metrics") if isinstance(row.get("extended_metrics"), dict) else {}
    if row.get("category") == "multi_stop_route":
        metrics = extended.get("multi_stop") if isinstance(extended.get("multi_stop"), dict) else {}
        return bool(metrics.get("sequence_success") and metrics.get("leg_route_success"))
    if str(row.get("category") or "").startswith("route_update"):
        metrics = extended.get("route_update") if isinstance(extended.get("route_update"), dict) else {}
        return bool(metrics.get("route_update_success"))
    return bool(row.get("route_success"))


def _task_spl(row: dict[str, Any], *, task_success: bool) -> float:
    """Return task-aware SPL gated by strict v2 success."""

    if not task_success:
        return 0.0
    extended = row.get("extended_metrics") if isinstance(row.get("extended_metrics"), dict) else {}
    if row.get("category") == "multi_stop_route":
        metrics = extended.get("multi_stop") if isinstance(extended.get("multi_stop"), dict) else {}
        return _number(metrics.get("multi_stop_spl"))
    return _number(row.get("spl"))


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one fixed-denominator row group."""

    return {
        "total_cases": len(rows),
        "destination_success": _mean_bool(rows, "destination_success"),
        "task_hit@1": _mean(rows, "task_hit@1"),
        "task_hit@3": _mean(rows, "task_hit@3"),
        "task_hit@5": _mean(rows, "task_hit@5"),
        "task_mrr@3": _mean(rows, "task_mrr@3"),
        "task_mrr@5": _mean(rows, "task_mrr@5"),
        "candidate_coverage": _mean(rows, "candidate_coverage"),
        "task_success": _mean_bool(rows, "task_success_v2"),
        "task_spl": _mean(rows, "task_spl_v2"),
        "multi_stop_sequence_success": _extended_mean(rows, "multi_stop", "sequence_success"),
        "multi_stop_stop_recall": _extended_mean(rows, "multi_stop", "stop_recall"),
        "multi_stop_leg_continuity_success": _extended_mean(rows, "multi_stop", "leg_continuity_success"),
        "route_update_success": _extended_mean(rows, "route_update", "route_update_success"),
        "prediction_exception_count": sum(bool(row.get("prediction_exception")) for row in rows),
        "missing_prediction_count": sum("missing_prediction" in (row.get("errors") or []) for row in rows),
    }


def _route_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize endpoint and route-ratio thresholds after DS succeeds."""

    successful = [row for row in rows if row.get("destination_success")]
    endpoints = [
        float(row["endpoint_error_m"])
        for row in successful
        if isinstance(row.get("endpoint_error_m"), (int, float))
    ]
    ratios = [
        float(row["route_ratio"])
        for row in successful
        if isinstance(row.get("route_ratio"), (int, float))
    ]
    return {
        "destination_success_cases": len(successful),
        "endpoint_observations": len(endpoints),
        "endpoint_le_1m": _fraction(endpoints, lambda value: value <= 1.0),
        "endpoint_le_5m": _fraction(endpoints, lambda value: value <= 5.0),
        "endpoint_le_10m": _fraction(endpoints, lambda value: value <= 10.0),
        "endpoint_le_30m": _fraction(endpoints, lambda value: value <= 30.0),
        "route_ratio_observations": len(ratios),
        "route_ratio_le_1.05": _fraction(ratios, lambda value: value <= 1.05),
        "route_ratio_le_1.10": _fraction(ratios, lambda value: value <= 1.10),
        "route_ratio_le_1.20": _fraction(ratios, lambda value: value <= 1.20),
    }


def _gt_route_legs(case: UrbanNavCase) -> list[dict[str, Any]]:
    """Return ordered frozen GT route legs."""

    value = case.raw.get("gt_route_legs")
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _parse_destinations(value: Any) -> list[UrbanNavDestination]:
    """Parse destination dictionaries from a list-like value."""

    values = value if isinstance(value, list) else [value]
    return [UrbanNavDestination.from_dict(item) for item in values if isinstance(item, dict)]


def _group_rows(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    """Group rows by one non-empty string field."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field) or "unknown")
        groups[key].append(row)
    return dict(sorted(groups.items()))


def _extended_mean(rows: list[dict[str, Any]], group: str, field: str) -> float | None:
    """Average one extended metric only over applicable task rows."""

    values: list[float] = []
    for row in rows:
        extended = row.get("extended_metrics") if isinstance(row.get("extended_metrics"), dict) else {}
        metrics = extended.get(group) if isinstance(extended.get(group), dict) else None
        if metrics is not None and field in metrics:
            values.append(_number(metrics.get(field)))
    return sum(values) / len(values) if values else None


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    """Average a numeric field over every row in the group."""

    return sum(_number(row.get(field)) for row in rows) / len(rows) if rows else 0.0


def _mean_bool(rows: list[dict[str, Any]], field: str) -> float:
    """Average a boolean field over every row in the group."""

    return sum(1.0 if row.get(field) else 0.0 for row in rows) / len(rows) if rows else 0.0


def _number(value: Any) -> float:
    """Convert finite numeric values to float and all other values to zero."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _fraction(values: list[float], predicate: Any) -> float | None:
    """Return the fraction satisfying a predicate, or ``None`` if empty."""

    return sum(1 for value in values if predicate(value)) / len(values) if values else None
