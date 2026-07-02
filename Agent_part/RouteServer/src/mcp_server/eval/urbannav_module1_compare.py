"""Relative comparison utilities for UrbanNav Module 1 benchmark runs.

The absolute Module 1 report keeps the primary benchmark metrics unchanged.
This module adds an internal comparison layer for model groups: it turns each
run's metrics into a bounded quality score, normalizes that score against a
chosen baseline, and computes paired win/draw/loss over shared cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


MODULE1_COMPARISON_VERSION = "urbannav_module1_relative_comparison_v1"
DEFAULT_SCORE_WEIGHTS = {
    "destination_success": 0.30,
    "hit@5": 0.10,
    "route_success": 0.25,
    "spl": 0.30,
    "route_efficiency": 0.05,
}


@dataclass(frozen=True)
class Module1RunSpec:
    """A labeled Module 1 run directory or summary file.

    Args:
        label: Human-readable run label, for example ``base`` or
            ``doubao_replay``.
        path: Run directory containing ``summary.json`` and ``eval_rows.jsonl``,
            or a direct path to ``summary.json``.
    """

    label: str
    path: str | Path


@dataclass(frozen=True)
class Module1ComparisonConfig:
    """Configuration for relative Module 1 comparison.

    Args:
        runs: Labeled run specs.
        baseline_label: Label used as the denominator for relative indices. If
            absent, the first run is used.
        weights: Metric weights for the scalar internal score. Weights are
            normalized as supplied and should sum to one.
        win_epsilon: Minimum per-case score difference counted as a paired win
            or loss. Smaller differences are draws.
    """

    runs: list[Module1RunSpec]
    baseline_label: str | None = None
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    win_epsilon: float = 1e-9


@dataclass(frozen=True)
class Module1RunData:
    """Loaded Module 1 run data.

    Args:
        label: Run label.
        path: Original input path.
        summary_path: Resolved summary JSON path.
        rows_path: Resolved per-case rows JSONL path, if present.
        report: Parsed summary report.
        rows: Parsed per-case rows.
    """

    label: str
    path: str
    summary_path: str
    rows_path: str | None
    report: dict[str, Any]
    rows: list[dict[str, Any]]


def compare_module1_runs(config: Module1ComparisonConfig) -> dict[str, Any]:
    r"""Compare Module 1 runs by absolute and relative scores.

    The aggregate internal score is:

    \[
    Q = w_D D + w_H H + w_R R + w_S SPL + w_E E
    \]

    where \(D\) is destination success, \(H\) is Hit@5, \(R\) is route
    success, and \(E=\min(1,1/\mathrm{median\_route\_ratio})\).  The relative
    index against baseline \(b\) is:

    \[
    I_m = 100 \cdot Q_m / Q_b
    \]

    Args:
        config: Comparison configuration.

    Returns:
        JSON-serializable comparison report.

    Raises:
        ValueError: If no runs are provided, labels are duplicated, or the
            baseline label cannot be found.
    """

    if not config.runs:
        raise ValueError("At least one run is required")
    labels = [run.label for run in config.runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Run labels must be unique: {labels}")
    runs = [load_module1_run(run) for run in config.runs]
    baseline_label = config.baseline_label or runs[0].label
    baseline = _run_by_label(runs, baseline_label)
    baseline_score = score_summary(_main_metrics(baseline.report), weights=config.weights)

    ranking = [
        _ranking_row(run, baseline_score=baseline_score, weights=config.weights)
        for run in runs
    ]
    ranking.sort(key=lambda item: (item["score"] is not None, item["score"] or -1.0), reverse=True)

    return {
        "schema_version": MODULE1_COMPARISON_VERSION,
        "baseline_label": baseline_label,
        "weights": dict(config.weights),
        "win_epsilon": config.win_epsilon,
        "run_count": len(runs),
        "ranking": ranking,
        "paired_vs_baseline": {
            run.label: paired_compare(run, baseline, weights=config.weights, win_epsilon=config.win_epsilon)
            for run in runs
            if run.label != baseline_label
        },
        "slice_rankings": _slice_rankings(runs, baseline=baseline, weights=config.weights),
    }


def load_module1_run(spec: Module1RunSpec) -> Module1RunData:
    """Load one Module 1 run report.

    Args:
        spec: Labeled run spec.

    Returns:
        Loaded run data.

    Raises:
        FileNotFoundError: If ``summary.json`` is missing.
    """

    input_path = Path(spec.path)
    summary_path = input_path / "summary.json" if input_path.is_dir() else input_path
    if not summary_path.exists():
        raise FileNotFoundError(str(summary_path))
    rows_path = summary_path.with_name("eval_rows.jsonl")
    return Module1RunData(
        label=spec.label,
        path=str(input_path),
        summary_path=str(summary_path),
        rows_path=str(rows_path) if rows_path.exists() else None,
        report=json.loads(summary_path.read_text(encoding="utf-8")),
        rows=_read_jsonl(rows_path) if rows_path.exists() else [],
    )


def score_summary(metrics: dict[str, Any], *, weights: dict[str, float] | None = None) -> float:
    """Compute the scalar internal score from aggregate metrics.

    Args:
        metrics: Metric dictionary containing Module 1 summary fields.
        weights: Optional metric weights.

    Returns:
        Score in ``[0, 1]`` for ordinary metric values.
    """

    active_weights = weights or DEFAULT_SCORE_WEIGHTS
    destination_success = _float(metrics.get("destination_success"))
    hit_at_5 = _float(metrics.get("task_hit@5", metrics.get("hit@5")))
    route_success = _float(metrics.get("route_success"))
    spl = _float(metrics.get("task_spl", metrics.get("spl")))
    route_efficiency = _route_efficiency(metrics.get("median_route_ratio"), fallback=route_success)
    return (
        active_weights.get("destination_success", 0.0) * destination_success
        + active_weights.get("hit@5", 0.0) * hit_at_5
        + active_weights.get("route_success", 0.0) * route_success
        + active_weights.get("spl", 0.0) * spl
        + active_weights.get("route_efficiency", 0.0) * route_efficiency
    )


def score_case(row: dict[str, Any], *, weights: dict[str, float] | None = None) -> float:
    """Compute per-case internal score for paired comparison.

    Args:
        row: One ``eval_rows.jsonl`` item.
        weights: Optional metric weights.

    Returns:
        Per-case score.
    """

    active_weights = weights or DEFAULT_SCORE_WEIGHTS
    route_success = _bool_float(row.get("route_success"))
    route_efficiency = _case_route_efficiency(row, fallback=route_success)
    return (
        active_weights.get("destination_success", 0.0) * _bool_float(row.get("destination_success"))
        + active_weights.get("hit@5", 0.0) * _case_hit_at_5(row)
        + active_weights.get("route_success", 0.0) * route_success
        + active_weights.get("spl", 0.0) * _case_spl(row)
        + active_weights.get("route_efficiency", 0.0) * route_efficiency
    )


def paired_compare(
    run: Module1RunData,
    baseline: Module1RunData,
    *,
    weights: dict[str, float] | None = None,
    win_epsilon: float = 1e-9,
) -> dict[str, Any]:
    """Compute paired win/draw/loss against baseline over shared cases.

    Args:
        run: Run being compared.
        baseline: Baseline run.
        weights: Optional metric weights.
        win_epsilon: Score margin required for a win/loss.

    Returns:
        Paired comparison summary.
    """

    run_rows = {str(row.get("case_id")): row for row in run.rows if row.get("case_id")}
    base_rows = {str(row.get("case_id")): row for row in baseline.rows if row.get("case_id")}
    common = sorted(set(run_rows) & set(base_rows))
    wins = draws = losses = 0
    deltas: list[float] = []
    for case_id in common:
        delta = score_case(run_rows[case_id], weights=weights) - score_case(base_rows[case_id], weights=weights)
        deltas.append(delta)
        if delta > win_epsilon:
            wins += 1
        elif delta < -win_epsilon:
            losses += 1
        else:
            draws += 1
    total = len(common)
    return {
        "baseline": baseline.label,
        "common_cases": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total if total else None,
        "loss_rate": losses / total if total else None,
        "mean_delta": sum(deltas) / total if total else None,
    }


def comparison_to_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report.

    Args:
        report: Output of ``compare_module1_runs``.

    Returns:
        Markdown summary.
    """

    lines = [
        "# UrbanNav Module 1 Relative Comparison",
        "",
        f"- Baseline: `{report.get('baseline_label')}`",
        f"- Runs: {report.get('run_count')}",
        f"- Weights: `{json.dumps(report.get('weights') or {}, sort_keys=True)}`",
        "",
        "## Ranking",
        "",
        "| Rank | Run | Score | Relative Index | DS | Task Hit@5 | RS | Task SPL | Median Ratio | Cases |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(report.get("ranking") or [], start=1):
        metrics = row.get("metrics") or {}
        lines.append(
            "| {rank} | `{label}` | {score} | {relative} | {ds} | {hit} | {rs} | {spl} | {ratio} | {cases} |".format(
                rank=index,
                label=row.get("label"),
                score=_fmt(row.get("score")),
                relative=_fmt(row.get("relative_index")),
                ds=_fmt(metrics.get("destination_success")),
                hit=_fmt(metrics.get("task_hit@5", metrics.get("hit@5"))),
                rs=_fmt(metrics.get("route_success")),
                spl=_fmt(metrics.get("task_spl", metrics.get("spl"))),
                ratio=_fmt(metrics.get("median_route_ratio")),
                cases=metrics.get("total_cases"),
            )
        )
    paired = report.get("paired_vs_baseline") or {}
    if paired:
        lines.extend(
            [
                "",
                "## Paired Comparison",
                "",
                "| Run | Common Cases | Wins | Draws | Losses | Win Rate | Mean Delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, row in paired.items():
            lines.append(
                "| `{label}` | {common} | {wins} | {draws} | {losses} | {win_rate} | {mean_delta} |".format(
                    label=label,
                    common=row.get("common_cases"),
                    wins=row.get("wins"),
                    draws=row.get("draws"),
                    losses=row.get("losses"),
                    win_rate=_fmt(row.get("win_rate")),
                    mean_delta=_fmt(row.get("mean_delta")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _ranking_row(
    run: Module1RunData,
    *,
    baseline_score: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    metrics = _main_metrics(run.report)
    score = score_summary(metrics, weights=weights)
    return {
        "label": run.label,
        "path": run.path,
        "summary_path": run.summary_path,
        "rows_path": run.rows_path,
        "score": score,
        "relative_index": 100.0 * score / baseline_score if baseline_score > 0 else None,
        "delta_vs_baseline": score - baseline_score,
        "metrics": metrics,
    }


def _slice_rankings(
    runs: list[Module1RunData],
    *,
    baseline: Module1RunData,
    weights: dict[str, float],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for section in ("by_category", "by_subcategory", "by_city", "by_scene_id", "by_difficulty"):
        slice_names = sorted(
            {
                name
                for run in runs
                for name in ((run.report.get(section) or {}).keys())
            }
        )
        section_result: dict[str, list[dict[str, Any]]] = {}
        for name in slice_names:
            baseline_metrics = (baseline.report.get(section) or {}).get(name) or {}
            baseline_score = score_summary(baseline_metrics, weights=weights)
            rows = []
            for run in runs:
                metrics = (run.report.get(section) or {}).get(name)
                if not isinstance(metrics, dict):
                    continue
                score = score_summary(metrics, weights=weights)
                rows.append(
                    {
                        "label": run.label,
                        "score": score,
                        "relative_index": 100.0 * score / baseline_score if baseline_score > 0 else None,
                        "metrics": metrics,
                    }
                )
            rows.sort(key=lambda item: (item["score"] is not None, item["score"] or -1.0), reverse=True)
            if rows:
                section_result[name] = rows
        if section_result:
            result[section] = section_result
    return result


def _run_by_label(runs: Iterable[Module1RunData], label: str) -> Module1RunData:
    for run in runs:
        if run.label == label:
            return run
    raise ValueError(f"Unknown baseline label: {label}")


def _main_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("summary") if isinstance(report.get("summary"), dict) else report
    # Module1 summary.json wraps metrics as summary.summary.  Older evaluator
    # outputs use summary directly.
    if isinstance(metrics.get("summary"), dict):
        metrics = metrics["summary"]
    return dict(metrics)


def _route_efficiency(value: Any, *, fallback: float = 0.0) -> float:
    ratio = _float(value)
    if ratio <= 0:
        return max(0.0, min(1.0, fallback))
    return max(0.0, min(1.0, 1.0 / max(1.0, ratio)))


def _case_hit_at_5(row: dict[str, Any]) -> float:
    if str(row.get("category") or "") == "multi_stop_route":
        value = ((row.get("extended_metrics") or {}).get("multi_stop") or {}).get("stop_hit@5")
        if value is not None:
            return _float(value)
    return _bool_float(row.get("hit@5"))


def _case_spl(row: dict[str, Any]) -> float:
    if str(row.get("category") or "") == "multi_stop_route":
        value = ((row.get("extended_metrics") or {}).get("multi_stop") or {}).get("multi_stop_spl")
        if value is not None:
            return _float(value)
    return _float(row.get("spl"))


def _case_route_efficiency(row: dict[str, Any], *, fallback: float = 0.0) -> float:
    if str(row.get("category") or "") == "multi_stop_route":
        multi_stop = (row.get("extended_metrics") or {}).get("multi_stop") or {}
        pred_total = _float(multi_stop.get("pred_total_distance_m"))
        ref_total = _float(multi_stop.get("reference_total_distance_m"))
        if pred_total > 0 and ref_total > 0:
            return max(0.0, min(1.0, ref_total / max(ref_total, pred_total)))
    return _route_efficiency(row.get("route_ratio"), fallback=fallback)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _float(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows
