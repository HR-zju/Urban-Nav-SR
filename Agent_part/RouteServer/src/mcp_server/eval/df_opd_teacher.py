"""Offline, paired DF-OPD Teacher-certification metrics and gates.

This evaluator consumes frozen per-condition score rows.  It deliberately does
not invoke a model, regenerate media, or choose training rows.  Its only
promotion decision is whether the supplied evidence satisfies the necessary
paired behavioural phenomena for Student OPD.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mcp_server.contracts.df_opd_certification import (
    DFOPDTeacherCertificationScoreV1,
)
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic


DF_OPD_TEACHER_CERTIFICATION_REPORT_SCHEMA_V1 = "urbannav_df_opd_teacher_certification_report_v1"
_CONDITIONS = (
    "same_input_t0",
    "correct_privilege_tc",
    "shuffled_privilege_ts",
    "exact_future_oracle_to",
)
_COMMON_RUN_FIELDS = (
    "profile_id",
    "profile_digest",
    "dataset_fingerprint",
    "checkpoint_digest",
    "seed",
)


def certify_df_opd_teacher_v1(
    score_rows: Iterable[DFOPDTeacherCertificationScoreV1 | Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int = 1000,
    high_conflict_limit: int = 50,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute condition metrics, paired confidence intervals, and promotion gates.

    A passing result requires paired evidence for both final Decision accuracy
    and trajectory FDE: TC must strictly improve over T0 and TS, and TO must
    not be worse than TC.  Decision CE and recoverability remain diagnostics;
    neither can substitute for behavioural evidence.

    Args:
        score_rows: Frozen T0/TC/TS/TO score rows from one certification run.
        bootstrap_seed: Deterministic resampling seed recorded in the report.
        bootstrap_replicates: Number of paired bootstrap resamples.
        high_conflict_limit: Maximum selected review candidates.

    Returns:
        A non-sensitive certification report and an ordered review-selection
        list keyed by canonical sample ID.

    Raises:
        ValueError: If rows are empty, duplicate, or drift across required run
            identities, or if bootstrap configuration is invalid.
    """

    if int(bootstrap_replicates) <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if int(high_conflict_limit) < 0:
        raise ValueError("high_conflict_limit cannot be negative")
    rows = [_parse_score_row(row) for row in score_rows]
    if not rows:
        raise ValueError("DF-OPD Teacher certification requires at least one score row")
    run_identity = _validate_run_identity(rows)
    grouped = _group_paired_rows(rows)
    complete_sample_ids = sorted(
        sample_id
        for sample_id, conditions in grouped.items()
        if set(conditions) == set(_CONDITIONS)
    )
    incomplete = [
        {
            "sample_id": sample_id,
            "conditions": sorted(conditions),
        }
        for sample_id, conditions in sorted(grouped.items())
        if set(conditions) != set(_CONDITIONS)
    ]
    by_condition = {
        condition: [row for row in rows if row.condition == condition]
        for condition in _CONDITIONS
    }
    comparison_specs = (
        ("tc_minus_t0_decision_accuracy", "correct_privilege_tc", "same_input_t0", "decision_correct", True),
        ("tc_minus_t0_fde_improvement_m", "correct_privilege_tc", "same_input_t0", "trajectory_fde_m", False),
        ("tc_minus_ts_decision_accuracy", "correct_privilege_tc", "shuffled_privilege_ts", "decision_correct", True),
        ("tc_minus_ts_fde_improvement_m", "correct_privilege_tc", "shuffled_privilege_ts", "trajectory_fde_m", False),
        ("to_minus_tc_decision_accuracy", "exact_future_oracle_to", "correct_privilege_tc", "decision_correct", True),
        ("to_minus_tc_fde_improvement_m", "exact_future_oracle_to", "correct_privilege_tc", "trajectory_fde_m", False),
    )
    comparisons = {
        name: _paired_comparison(
            grouped,
            complete_sample_ids,
            left_condition=left,
            right_condition=right,
            metric_name=metric_name,
            higher_is_better=higher_is_better,
            bootstrap_seed=int(bootstrap_seed),
            bootstrap_replicates=int(bootstrap_replicates),
            comparison_name=name,
        )
        for name, left, right, metric_name, higher_is_better in comparison_specs
    }
    gate = _certification_gate(
        comparisons,
        complete_sample_count=len(complete_sample_ids),
        incomplete_count=len(incomplete),
    )
    selection = _high_conflict_selection(grouped, complete_sample_ids, int(high_conflict_limit))
    report = {
        "schema_version": DF_OPD_TEACHER_CERTIFICATION_REPORT_SCHEMA_V1,
        "run_identity": run_identity,
        "bootstrap": {
            "seed": int(bootstrap_seed),
            "replicates": int(bootstrap_replicates),
            "confidence_interval": "percentile_95",
            "paired_resampling_unit": "sample_id",
        },
        "coverage": {
            "score_rows": len(rows),
            "paired_complete_samples": len(complete_sample_ids),
            "incomplete_samples": incomplete,
            "condition_rows": {condition: len(by_condition[condition]) for condition in _CONDITIONS},
        },
        "condition_metrics": {
            condition: _condition_metrics(by_condition[condition])
            for condition in _CONDITIONS
        },
        "paired_comparisons": comparisons,
        "promotion_gate": gate,
        "review_selection_count": len(selection),
    }
    return report, selection


def certify_df_opd_teacher_jsonl_v1(
    score_jsonl: str | Path,
    report_json: str | Path,
    *,
    high_conflict_jsonl: str | Path | None = None,
    bootstrap_seed: int,
    bootstrap_replicates: int = 1000,
    high_conflict_limit: int = 50,
) -> dict[str, Any]:
    """Load frozen score JSONL and persist its certification receipt offline.

    Args:
        score_jsonl: Input score rows from a completed local inference run.
        report_json: Destination certification report JSON.
        high_conflict_jsonl: Optional review-selection JSONL destination.
        bootstrap_seed: Deterministic paired-resampling seed.
        bootstrap_replicates: Number of paired bootstrap resamples.
        high_conflict_limit: Maximum high-conflict candidates to select.

    Returns:
        The persisted certification report.

    Raises:
        ValueError: If score rows violate certification invariants.
    """

    report, selection = certify_df_opd_teacher_v1(
        iter_jsonl(score_jsonl),
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        high_conflict_limit=high_conflict_limit,
    )
    report = dict(report)
    report["score_jsonl"] = str(Path(score_jsonl).expanduser().resolve())
    report["score_jsonl_sha256"] = file_sha256(score_jsonl)
    report["report_json"] = str(Path(report_json).expanduser().resolve())
    if high_conflict_jsonl is not None:
        selection_path = Path(high_conflict_jsonl).expanduser().resolve()
        write_jsonl_atomic(selection_path, selection)
        report["high_conflict_jsonl"] = str(selection_path)
        report["high_conflict_jsonl_sha256"] = file_sha256(selection_path)
    _write_json_atomic(Path(report_json).expanduser().resolve(), report)
    return report


def _parse_score_row(value: DFOPDTeacherCertificationScoreV1 | Mapping[str, Any]) -> DFOPDTeacherCertificationScoreV1:
    """Accept an immutable score object or validate its JSON-safe payload."""

    if isinstance(value, DFOPDTeacherCertificationScoreV1):
        return value
    return DFOPDTeacherCertificationScoreV1.model_validate(value)


def _validate_run_identity(rows: Sequence[DFOPDTeacherCertificationScoreV1]) -> dict[str, Any]:
    """Require all score rows to come from one frozen certification setting."""

    identity = {field: getattr(rows[0], field) for field in _COMMON_RUN_FIELDS}
    for row in rows[1:]:
        for field, expected in identity.items():
            if getattr(row, field) != expected:
                raise ValueError(
                    f"DF-OPD certification run identity drift for {field}: "
                    f"expected {expected!r}, got {getattr(row, field)!r}"
                )
    return identity


def _group_paired_rows(
    rows: Sequence[DFOPDTeacherCertificationScoreV1],
) -> dict[str, dict[str, DFOPDTeacherCertificationScoreV1]]:
    """Group condition rows by sample while rejecting duplicate or drifting identity."""

    grouped: dict[str, dict[str, DFOPDTeacherCertificationScoreV1]] = defaultdict(dict)
    for row in rows:
        conditions = grouped[row.sample_id]
        if row.condition in conditions:
            raise ValueError(
                f"duplicate DF-OPD certification score for sample/condition "
                f"{row.sample_id!r}/{row.condition!r}"
            )
        if conditions:
            reference = next(iter(conditions.values()))
            if row.canonical_digest != reference.canonical_digest:
                raise ValueError(f"canonical_digest drift across conditions for {row.sample_id!r}")
            if row.source_group_id != reference.source_group_id:
                raise ValueError(f"source_group_id drift across conditions for {row.sample_id!r}")
        conditions[row.condition] = row
    return dict(grouped)


def _condition_metrics(rows: Sequence[DFOPDTeacherCertificationScoreV1]) -> dict[str, Any]:
    """Summarize one condition without erasing missing/invalid denominators."""

    return {
        "rows": len(rows),
        "strict_parse_rate": _mean_metric(rows, "decision_strict_parse"),
        "decision_accuracy": _mean_metric(rows, "decision_correct"),
        "decision_ce": _mean_metric(rows, "decision_ce"),
        "trajectory_valid_rate": _mean_metric(rows, "trajectory_valid"),
        "trajectory_ade_m": _mean_metric(rows, "trajectory_ade_m"),
        "trajectory_fde_m": _mean_metric(rows, "trajectory_fde_m"),
        "direction_error_deg": _mean_metric(rows, "direction_error_deg"),
        "turn_fde_m": _mean_metric([row for row in rows if row.is_turn], "trajectory_fde_m"),
        "stop_fde_m": _mean_metric([row for row in rows if row.is_stop], "trajectory_fde_m"),
        "recoverability_probe_accuracy": _mean_metric(rows, "recoverability_probe_correct"),
        "context_conflict_score": _mean_metric(rows, "context_conflict_score"),
    }


def _mean_metric(rows: Sequence[DFOPDTeacherCertificationScoreV1], field_name: str) -> dict[str, Any]:
    """Return an explicit numerator denominator summary for an optional metric."""

    values = [float(value) for row in rows if (value := getattr(row, field_name)) is not None]
    return {
        "mean": sum(values) / len(values) if values else None,
        "numerator": len(values),
        "denominator": len(rows),
    }


def _paired_comparison(
    grouped: Mapping[str, Mapping[str, DFOPDTeacherCertificationScoreV1]],
    complete_sample_ids: Sequence[str],
    *,
    left_condition: str,
    right_condition: str,
    metric_name: str,
    higher_is_better: bool,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    comparison_name: str,
) -> dict[str, Any]:
    """Compute a signed paired gain and deterministic percentile interval."""

    deltas: list[float] = []
    for sample_id in complete_sample_ids:
        left_value = getattr(grouped[sample_id][left_condition], metric_name)
        right_value = getattr(grouped[sample_id][right_condition], metric_name)
        if left_value is None or right_value is None:
            continue
        difference = float(left_value) - float(right_value)
        deltas.append(difference if higher_is_better else -difference)
    interval = _bootstrap_mean_interval(
        deltas,
        seed=_derived_seed(bootstrap_seed, comparison_name),
        replicates=bootstrap_replicates,
    )
    return {
        "left_condition": left_condition,
        "right_condition": right_condition,
        "metric": metric_name,
        "positive_means_left_better": True,
        "expected_pairs": len(complete_sample_ids),
        "metric_pairs": len(deltas),
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "ci95": interval,
    }


def _bootstrap_mean_interval(values: Sequence[float], *, seed: int, replicates: int) -> dict[str, float] | None:
    """Calculate a deterministic paired-bootstrap percentile confidence interval."""

    if not values:
        return None
    random_state = random.Random(seed)
    size = len(values)
    estimates = []
    for _ in range(replicates):
        estimates.append(sum(values[random_state.randrange(size)] for _ in range(size)) / size)
    estimates.sort()
    lower_index = int((len(estimates) - 1) * 0.025)
    upper_index = int((len(estimates) - 1) * 0.975)
    return {"lower": estimates[lower_index], "upper": estimates[upper_index]}


def _derived_seed(seed: int, comparison_name: str) -> int:
    """Derive an order-independent bootstrap stream for one named comparison."""

    digest = hashlib.sha256(f"{seed}:{comparison_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _certification_gate(
    comparisons: Mapping[str, Mapping[str, Any]],
    *,
    complete_sample_count: int,
    incomplete_count: int,
) -> dict[str, Any]:
    """Apply the documented TC/T0/TS/TO necessary behavioural phenomena."""

    positive_keys = (
        "tc_minus_t0_decision_accuracy",
        "tc_minus_t0_fde_improvement_m",
        "tc_minus_ts_decision_accuracy",
        "tc_minus_ts_fde_improvement_m",
    )
    nonnegative_keys = (
        "to_minus_tc_decision_accuracy",
        "to_minus_tc_fde_improvement_m",
    )
    if complete_sample_count == 0 or incomplete_count:
        return {
            "status": "insufficient_evidence",
            "student_opd_eligible": False,
            "reasons": ["certification requires complete paired T0/TC/TS/TO rows for every sample"],
        }
    coverage_failures = [
        name
        for name, comparison in comparisons.items()
        if comparison["metric_pairs"] != comparison["expected_pairs"] or comparison["ci95"] is None
    ]
    if coverage_failures:
        return {
            "status": "insufficient_evidence",
            "student_opd_eligible": False,
            "reasons": [f"missing paired behavioural metric coverage: {', '.join(coverage_failures)}"],
        }
    failed_positive = [
        name
        for name in positive_keys
        if float(comparisons[name]["ci95"]["lower"]) <= 0.0
    ]
    failed_nonnegative = [
        name
        for name in nonnegative_keys
        if float(comparisons[name]["mean_delta"]) < 0.0
    ]
    if failed_positive or failed_nonnegative:
        return {
            "status": "failed",
            "student_opd_eligible": False,
            "reasons": [
                *(f"TC behavioural advantage is not positive at CI95: {name}" for name in failed_positive),
                *(f"TO oracle upper bound is worse than TC: {name}" for name in failed_nonnegative),
            ],
        }
    return {
        "status": "passed",
        "student_opd_eligible": True,
        "reasons": [
            "TC beats T0 and TS on paired Decision accuracy and FDE improvement",
            "TO is not worse than TC on the same paired behavioural metrics",
        ],
    }


def _high_conflict_selection(
    grouped: Mapping[str, Mapping[str, DFOPDTeacherCertificationScoreV1]],
    complete_sample_ids: Sequence[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Select review IDs by maximum reported conflict without changing training data."""

    candidates = []
    for sample_id in complete_sample_ids:
        conditions = grouped[sample_id]
        scores = [row.context_conflict_score for row in conditions.values() if row.context_conflict_score is not None]
        if not scores:
            continue
        reference = conditions[_CONDITIONS[0]]
        candidates.append(
            {
                "sample_id": sample_id,
                "source_group_id": reference.source_group_id,
                "canonical_digest": reference.canonical_digest,
                "max_context_conflict_score": max(scores),
                "conditions": list(_CONDITIONS),
            }
        )
    candidates.sort(key=lambda item: (-float(item["max_context_conflict_score"]), str(item["sample_id"])))
    return candidates[:limit]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a compact certification receipt without partial-file exposure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "DF_OPD_TEACHER_CERTIFICATION_REPORT_SCHEMA_V1",
    "certify_df_opd_teacher_jsonl_v1",
    "certify_df_opd_teacher_v1",
]
