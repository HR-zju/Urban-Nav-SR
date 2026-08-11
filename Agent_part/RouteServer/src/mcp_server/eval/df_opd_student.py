"""Evaluate Student-only DF-OPD inference ledgers against canonical GT offline."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.df_opd import DFOPDCanonicalSampleV1, df_opd_digest_v1
from mcp_server.contracts.reasoning_v4 import DecisionV4
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl

DF_OPD_STUDENT_EVALUATION_SCHEMA_V1 = "urbannav_df_opd_student_evaluation_v1"
_FORBIDDEN_PREDICTION_FIELDS = {
    "messages",
    "images",
    "image_roles",
    "teacher_images",
    "teacher_image_roles",
    "teacher_messages",
    "teacher_prompt",
    "decision_target",
    "trajectory_target",
    "trajectory_ego_t_target",
    "action_mask",
    "canonical_full_cot",
    "df_opd_teacher_condition",
    "df_opd_privilege_policy_digest",
    "df_opd_teacher_cue_sha256",
}


def evaluate_df_opd_student_predictions_v1(
    canonical_jsonl: str | Path,
    inference_jsonls: Sequence[str | Path],
    report_json: str | Path,
    *,
    dataset_fingerprint: str,
    checkpoint_digest: str,
    seed: int,
) -> dict[str, Any]:
    """Join one complete Student ledger to a single-profile canonical split.

    Args:
        canonical_jsonl: Frozen canonical samples containing hidden GT.
        inference_jsonls: Student prediction and failure JSONLs whose union has
            exactly one row per canonical sample.
        report_json: Destination metrics-only evaluation report.
        dataset_fingerprint: Frozen benchmark split SHA-256 identity.
        checkpoint_digest: Evaluated Student checkpoint SHA-256 identity.
        seed: Deterministic inference seed.

    Returns:
        Metrics-only report with no prompt, media, private cue, Decision text,
        raw trajectory, or GT value.

    Raises:
        ValueError: If profile/run identity, complete output coverage, or
            Student-only no-leak guarantees are violated.
    """

    _require_digest(dataset_fingerprint, "dataset_fingerprint")
    _require_digest(checkpoint_digest, "checkpoint_digest")
    canonical_path = Path(canonical_jsonl).expanduser().resolve()
    paths = _resolve_paths(inference_jsonls)
    canonical = _read_canonical(canonical_path)
    profile_ids = {sample.profile.profile_id for sample in canonical.values()}
    if len(profile_ids) != 1:
        raise ValueError("DF-OPD Student evaluation requires one profile per invocation")
    predictions = _read_student_ledgers(paths, canonical)
    if set(predictions) != set(canonical):
        missing = sorted(set(canonical) - set(predictions))
        raise ValueError("DF-OPD Student evaluation lacks prediction/failure rows: " + ", ".join(missing[:8]))
    profile = next(iter(canonical.values())).profile
    rows = [
        _row_metrics(sample, predictions[sample_id])
        for sample_id, sample in canonical.items()
    ]
    report = {
        "schema_version": DF_OPD_STUDENT_EVALUATION_SCHEMA_V1,
        "student_only": True,
        "canonical_jsonl": str(canonical_path),
        "canonical_jsonl_sha256": file_sha256(canonical_path),
        "inference_jsonls": [str(path) for path in paths],
        "inference_jsonl_sha256": [file_sha256(path) for path in paths],
        "prediction_manifest_digest": df_opd_digest_v1({
            "inference_jsonl_sha256": sorted(file_sha256(path) for path in paths),
        }),
        "dataset_fingerprint": dataset_fingerprint,
        "checkpoint_digest": checkpoint_digest,
        "seed": int(seed),
        "profile_id": profile.profile_id,
        "profile_digest": profile.profile_digest,
        "sample_count": len(rows),
        "decision": _decision_report(rows),
        "trajectory": _trajectory_report(rows, dt_sec=profile.dt_sec, horizon_steps=profile.target_horizon_steps),
        "latency_ms": _latency_report(rows),
    }
    _write_json_atomic(Path(report_json).expanduser().resolve(), report)
    return report


def _resolve_paths(values: Sequence[str | Path]) -> list[Path]:
    """Resolve a non-empty, unique sequence of Student prediction ledgers."""

    paths = [Path(value).expanduser().resolve() for value in values]
    if not paths:
        raise ValueError("DF-OPD Student evaluation requires inference JSONLs")
    if len(set(paths)) != len(paths):
        raise ValueError("DF-OPD Student evaluation inference JSONLs must be unique")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _read_canonical(path: Path) -> dict[str, DFOPDCanonicalSampleV1]:
    """Load unique immutable canonical rows and validate their Decision targets."""

    result: dict[str, DFOPDCanonicalSampleV1] = {}
    for line_number, payload in enumerate(iter_jsonl(path), start=1):
        try:
            sample = DFOPDCanonicalSampleV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid canonical DF-OPD row at line {line_number}: {exc}") from exc
        if sample.sample_id in result:
            raise ValueError(f"duplicate canonical sample_id: {sample.sample_id!r}")
        if _parse_strict_decision(sample.decision_target) is None:
            raise ValueError(f"canonical sample {sample.sample_id!r} has non-canonical decision_target")
        result[sample.sample_id] = sample
    if not result:
        raise ValueError("DF-OPD Student evaluation requires canonical samples")
    return result


def _read_student_ledgers(
    paths: Sequence[Path],
    canonical: Mapping[str, DFOPDCanonicalSampleV1],
) -> dict[str, Mapping[str, Any]]:
    """Validate public Student outputs and index exactly one row per sample."""

    result: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        for line_number, row in enumerate(iter_jsonl(path), start=1):
            forbidden = _find_forbidden_field(row)
            if forbidden is not None:
                raise ValueError(f"Student inference ledger leaked forbidden field {forbidden!r}: {path}:{line_number}")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in canonical:
                raise ValueError(f"Student inference row has unknown sample_id: {path}:{line_number}")
            sample = canonical[sample_id]
            if (
                row.get("canonical_digest") != sample.canonical_digest
                or row.get("df_opd_profile_id") != sample.profile.profile_id
                or row.get("df_opd_profile_digest") != sample.profile.profile_digest
            ):
                raise ValueError(f"Student inference identity/profile drift: {path}:{line_number}")
            if sample_id in result:
                raise ValueError(f"duplicate Student inference row for {sample_id!r}")
            result[sample_id] = row
    return result


def _row_metrics(sample: DFOPDCanonicalSampleV1, row: Mapping[str, Any]) -> dict[str, Any]:
    """Compute one private-free metric accumulator without retaining values."""

    expected = _parse_strict_decision(sample.decision_target)
    assert expected is not None
    raw_decision = row.get("decision")
    raw_trajectory = row.get("trajectory_ego_t")
    if row.get("failure_code") is not None:
        if raw_decision is not None or raw_trajectory is not None:
            raise ValueError("failed Student inference ledger must not retain Decision or trajectory output")
        raw_decision = raw_trajectory = None
    predicted = _parse_strict_decision(raw_decision)
    trajectory = _parse_valid_trajectory(raw_trajectory, sample)
    field_correct = {
        "selected_action_label": bool(predicted is not None and
                                      predicted.selected_action_label == expected.selected_action_label),
        "speed": bool(predicted is not None and predicted.speed == expected.speed),
        "response": bool(predicted is not None and predicted.response == expected.response),
    }
    return {
        "strict_parse": predicted is not None,
        "tuple_correct": bool(predicted is not None and predicted == expected),
        "field_correct": field_correct,
        "is_turn": _is_turn_label(str(expected.selected_action_label)),
        "is_stop": str(expected.selected_action_label) == "stop",
        "trajectory": trajectory,
        "target": sample,
        "latency_ms": _extract_latency(row),
    }


def _decision_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate strict Decision accuracy with explicit full-reference denominators."""

    count = len(rows)
    return {
        "strict_parse_rate": _ratio(sum(bool(row["strict_parse"]) for row in rows), count),
        "tuple_accuracy": _ratio(sum(bool(row["tuple_correct"]) for row in rows), count),
        "selected_action_accuracy": _ratio(sum(row["field_correct"]["selected_action_label"] for row in rows), count),
        "speed_accuracy": _ratio(sum(row["field_correct"]["speed"] for row in rows), count),
        "response_accuracy": _ratio(sum(row["field_correct"]["response"] for row in rows), count),
        "turn": _subset_decision_accuracy(rows, "is_turn"),
        "stop": _subset_decision_accuracy(rows, "is_stop"),
    }


def _subset_decision_accuracy(rows: Sequence[Mapping[str, Any]], flag: str) -> dict[str, Any]:
    """Return tuple accuracy over one named canonical subgroup."""

    selected = [row for row in rows if row[flag]]
    return {
        "reference_count": len(selected),
        "tuple_accuracy": _ratio(sum(bool(row["tuple_correct"]) for row in selected), len(selected)),
    }


def _trajectory_report(
    rows: Sequence[Mapping[str, Any],
    ],
    *,
    dt_sec: float,
    horizon_steps: int,
) -> dict[str, Any]:
    """Aggregate fixed-grid ADE/FDE and coverage at all reportable horizons."""

    horizons = [0.5, 1.0, 2.0]
    max_horizon = float(horizon_steps) * float(dt_sec)
    if max_horizon > 2.0:
        horizons.append(max_horizon)
    result: dict[str, Any] = {}
    for horizon_s in horizons:
        step_count = int(round(float(horizon_s) / float(dt_sec)))
        if step_count < 1 or step_count > horizon_steps:
            continue
        eligible = []
        valid = []
        for row in rows:
            sample = row["target"]
            if all(sample.action_mask[:step_count]):
                eligible.append(row)
                if row["trajectory"] is not None:
                    valid.append(row)
        ade_values = [_prefix_ade(row["target"], row["trajectory"], step_count) for row in valid]
        fde_values = [_prefix_fde(row["target"], row["trajectory"], step_count) for row in valid]
        result[_horizon_key(horizon_s)] = {
            "horizon_s": horizon_s,
            "step_count": step_count,
            "eligible_reference_count": len(eligible),
            "valid_prediction_count": len(valid),
            "geometry_coverage": _ratio(len(valid), len(eligible)),
            "conditional_ade_m": _mean(ade_values),
            "conditional_fde_m": _mean(fde_values),
            "conditional_l2_m": _mean(fde_values),
        }
    return result


def _latency_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only finite non-negative latency fields from output ledgers."""

    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row["latency_ms"].items():
            values[name].append(value)
    return {
        name: {
            "count": len(samples),
            "mean": _mean(samples),
            "p50": _quantile(samples, 0.5),
            "p95": _quantile(samples, 0.95),
        }
        for name, samples in sorted(values.items())
    }


def _extract_latency(row: Mapping[str, Any]) -> dict[str, float]:
    """Read output timing diagnostics without permitting arbitrary metadata."""

    result: dict[str, float] = {}
    total = row.get("latency_ms")
    if isinstance(total, (int, float)) and math.isfinite(float(total)) and float(total) >= 0.0:
        result["wall_total"] = float(total)
    metadata = row.get("action_metadata")
    nested = metadata.get("latency_ms") if isinstance(metadata, Mapping) else None
    if isinstance(nested, Mapping):
        for name, value in nested.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0:
                result[str(name)] = float(value)
    return result


def _parse_strict_decision(value: Any) -> DecisionV4 | None:
    """Accept only exact canonical Decision wire text."""

    if not isinstance(value, str) or not value.startswith("<Decision>") or not value.endswith("</Decision>"):
        return None
    try:
        decision = DecisionV4.model_validate(json.loads(value[len("<Decision>"):-len("</Decision>")]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    canonical = "<Decision>" + json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + "</Decision>"
    return decision if canonical == value else None


def _parse_valid_trajectory(value: Any, sample: DFOPDCanonicalSampleV1) -> tuple[tuple[float, float, float], ...] | None:
    """Return only complete finite trajectory output matching the profile."""

    if not isinstance(value, list) or len(value) != sample.profile.target_horizon_steps:
        return None
    points: list[tuple[float, float, float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != sample.profile.action_dim:
            return None
        try:
            parsed = tuple(float(item) for item in point)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in parsed):
            return None
        points.append(parsed)
    return tuple(points)


def _prefix_ade(sample: DFOPDCanonicalSampleV1, prediction: Sequence[tuple[float, float, float]], steps: int) -> float:
    """Compute XY ADE over one eligible fixed-grid prefix."""

    return _mean([
        math.hypot(prediction[index][0] - sample.trajectory_ego_t[index][0],
                   prediction[index][1] - sample.trajectory_ego_t[index][1])
        for index in range(steps)
    ])


def _prefix_fde(sample: DFOPDCanonicalSampleV1, prediction: Sequence[tuple[float, float, float]], steps: int) -> float:
    """Compute XY FDE at the terminal point of one fixed-grid prefix."""

    index = steps - 1
    return math.hypot(prediction[index][0] - sample.trajectory_ego_t[index][0],
                      prediction[index][1] - sample.trajectory_ego_t[index][1])


def _find_forbidden_field(value: Any) -> str | None:
    """Return the first prompt, media, GT, or Teacher field in Student output."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_PREDICTION_FIELDS:
                return str(key)
            nested = _find_forbidden_field(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_forbidden_field(item)
            if nested is not None:
                return nested
    return None


def _is_turn_label(value: str) -> bool:
    """Classify explicit left/right/U-turn actions as the turn subset."""

    return str(value).strip().lower() in {"left", "right", "u_turn", "uturn", "turn_left", "turn_right"}


def _horizon_key(value: float) -> str:
    """Render a stable report key for a positive physical horizon."""

    return f"{int(value) if float(value).is_integer() else value:g}s"


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return an explicit nullable fraction."""

    return numerator / denominator if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    """Return a nullable arithmetic mean."""

    return sum(values) / len(values) if values else None


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    """Return a deterministic linear-interpolated percentile."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _require_digest(value: str, field_name: str) -> None:
    """Require one lower-case SHA-256 identity."""

    if len(str(value)) != 64 or any(character not in "0123456789abcdef" for character in str(value)):
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a metrics-only report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
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
    "DF_OPD_STUDENT_EVALUATION_SCHEMA_V1",
    "evaluate_df_opd_student_predictions_v1",
]

