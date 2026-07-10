#!/usr/bin/env python3
"""Filter SideWalk Phase1 rows by ego-state motion saliency.

For each manifest row, this script estimates acceleration and turning saliency
from the available future ego trajectory.  It can fuse the score with a
behavior-filter VLM decision through the paper-style priority weight:

    S' = w_p * S

where ``w_p`` is larger for visually interesting segments and smaller for
boring segments.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "sidewalk_phase1_motion_saliency_filter_v1"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Read a JSONL file.

    Args:
        path: JSONL path.

    Returns:
        Iterator of decoded dictionary rows.

    Raises:
        ValueError: If any line is invalid JSON.
    """

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file.

    Args:
        path: Output JSONL path.
        row: JSON-serializable row.

    Returns:
        ``None``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON object.

    Args:
        path: Output JSON path.
        payload: JSON-serializable payload.

    Returns:
        ``None``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def row_id(row: dict[str, Any], index: int | None = None) -> str:
    """Return a stable manifest row id.

    Args:
        row: Source manifest row.
        index: Optional source line index for id-less rows.

    Returns:
        Stable id string.
    """

    raw = row.get("id")
    if raw:
        return str(raw)
    clip_id = row.get("clip_id") or (row.get("sidewalk_meta") or {}).get("clip_id")
    frame_idx = row.get("frame_idx") or (row.get("sidewalk_meta") or {}).get("frame_idx")
    if clip_id is not None and frame_idx is not None:
        return f"{clip_id}_{frame_idx}"
    return f"__index_{index}" if index is not None else "__missing_id"


def row_clip_id(row: dict[str, Any]) -> str:
    """Extract a clip id for temporal diversity filtering.

    Args:
        row: Manifest row.

    Returns:
        Clip id string.
    """

    return str(row.get("clip_id") or (row.get("sidewalk_meta") or {}).get("clip_id") or "unknown_clip")


def row_frame_idx(row: dict[str, Any], fallback: int) -> int:
    """Extract a frame index for temporal diversity filtering.

    Args:
        row: Manifest row.
        fallback: Fallback index.

    Returns:
        Integer frame index.
    """

    value = row.get("frame_idx")
    if value is None and isinstance(row.get("sidewalk_meta"), dict):
        value = row["sidewalk_meta"].get("frame_idx")
    try:
        return int(value)
    except Exception:
        return int(fallback)


def row_intent(row: dict[str, Any]) -> str:
    """Extract intent from common SideWalk manifest fields.

    Args:
        row: Manifest row.

    Returns:
        Intent string or ``unknown``.
    """

    route_context = row.get("route_context")
    if isinstance(route_context, dict) and route_context.get("intent_gt"):
        return str(route_context["intent_gt"])
    sidewalk_meta = row.get("sidewalk_meta")
    if isinstance(sidewalk_meta, dict) and sidewalk_meta.get("intent_gt"):
        return str(sidewalk_meta["intent_gt"])
    return str(row.get("intent_gt") or "unknown")


def safe_float(value: Any) -> float | None:
    """Parse a finite float.

    Args:
        value: Raw value.

    Returns:
        Parsed finite float or ``None``.
    """

    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def trajectory_array(row: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Extract a trajectory array for motion saliency.

    The preferred schema is ``trajectory_ego_t`` or ``gt_trajectory_ego_t`` with
    columns ``x_right_m, y_forward_m, t_sec, speed_mps``.  If it is unavailable,
    normalized FPV trajectory points are used as a lower-quality fallback.

    Args:
        row: Manifest row.

    Returns:
        ``(array, source_name)``.  The array has shape ``N x C``.
    """

    for key in ("trajectory_ego_t", "gt_trajectory_ego_t"):
        value = row.get(key)
        if isinstance(value, list) and value:
            arr = np.asarray(value, dtype=np.float64).reshape(-1, len(value[0]))
            if arr.shape[1] >= 2:
                return arr, key
    for key in ("pixel_trajectory", "gt_pixel_trajectory"):
        value = row.get(key)
        if isinstance(value, list) and value:
            arr = np.asarray(value, dtype=np.float64).reshape(-1, len(value[0]))
            if arr.shape[1] >= 2:
                return arr[:, :2], key
    return np.zeros((0, 2), dtype=np.float64), "missing"


def delta_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Return the wrapped signed angle difference between two segment vectors.

    Args:
        a: Previous vector ``[x_right, y_forward]``.
        b: Next vector ``[x_right, y_forward]``.

    Returns:
        Signed angle difference in degrees in ``[-180, 180]``.
    """

    angle_a = math.degrees(math.atan2(float(a[0]), float(a[1])))
    angle_b = math.degrees(math.atan2(float(b[0]), float(b[1])))
    return ((angle_b - angle_a + 180.0) % 360.0) - 180.0


def robust_variance(values: np.ndarray) -> float:
    """Return population variance for finite values.

    Args:
        values: Numeric vector.

    Returns:
        Variance as a float, or ``0`` for empty inputs.
    """

    if values.size == 0:
        return 0.0
    return float(np.var(values.astype(np.float64)))


def compute_motion_features(row: dict[str, Any], *, default_dt_sec: float) -> dict[str, Any]:
    r"""Compute acceleration and turning saliency features.

    Acceleration saliency components are computed from absolute speed changes:

    \[
    \alpha_i = |v_i - v_{i-1}| / \max(t_i - t_{i-1}, \epsilon)
    \]

    Turning components are absolute changes between consecutive local segment
    headings:

    \[
    \theta_i = |\operatorname{wrap}(\psi_i - \psi_{i-1})|
    \]

    Args:
        row: Manifest row.
        default_dt_sec: Fallback time delta for rows without timestamps.

    Returns:
        Feature dictionary with raw component values.
    """

    arr, source = trajectory_array(row)
    points = arr[:, :2] if arr.shape[0] else np.zeros((0, 2), dtype=np.float64)
    if points.shape[0] < 2:
        return {
            "trajectory_source": source,
            "num_points": int(points.shape[0]),
            "valid": False,
            "error": "too_few_points",
            "accel_avg": 0.0,
            "accel_max": 0.0,
            "accel_var": 0.0,
            "turn_avg_deg": 0.0,
            "turn_max_deg": 0.0,
            "turn_var_deg2": 0.0,
            "mean_speed": 0.0,
            "max_speed": 0.0,
            "path_length": 0.0,
        }

    if arr.shape[1] >= 3:
        times = arr[:, 2].astype(np.float64)
    else:
        times = np.arange(points.shape[0], dtype=np.float64) * max(float(default_dt_sec), 1e-6)
    dt = np.diff(times)
    fallback_dt = max(float(default_dt_sec), 1e-6)
    dt = np.where(dt > 1e-6, dt, fallback_dt)

    deltas = np.diff(points, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    segment_speeds = distances / dt
    if arr.shape[1] >= 4:
        speed_profile = arr[:, 3].astype(np.float64)
        finite_mask = np.isfinite(speed_profile)
        if finite_mask.sum() >= 2:
            speed_for_accel = speed_profile[finite_mask]
            time_for_accel = times[finite_mask]
            dt_accel = np.diff(time_for_accel)
            dt_accel = np.where(dt_accel > 1e-6, dt_accel, fallback_dt)
            accel = np.abs(np.diff(speed_for_accel)) / dt_accel
        else:
            accel = np.abs(np.diff(segment_speeds)) / np.maximum(dt[1:], fallback_dt)
    else:
        accel = np.abs(np.diff(segment_speeds)) / np.maximum(dt[1:], fallback_dt)

    headings_delta: list[float] = []
    for i in range(1, deltas.shape[0]):
        if np.linalg.norm(deltas[i - 1]) <= 1e-9 or np.linalg.norm(deltas[i]) <= 1e-9:
            continue
        headings_delta.append(abs(delta_angle_deg(deltas[i - 1], deltas[i])))
    turns = np.asarray(headings_delta, dtype=np.float64)

    return {
        "trajectory_source": source,
        "num_points": int(points.shape[0]),
        "valid": True,
        "accel_avg": float(np.mean(accel)) if accel.size else 0.0,
        "accel_max": float(np.max(accel)) if accel.size else 0.0,
        "accel_var": robust_variance(accel),
        "turn_avg_deg": float(np.mean(turns)) if turns.size else 0.0,
        "turn_max_deg": float(np.max(turns)) if turns.size else 0.0,
        "turn_var_deg2": robust_variance(turns),
        "mean_speed": float(np.mean(segment_speeds)) if segment_speeds.size else 0.0,
        "max_speed": float(np.max(segment_speeds)) if segment_speeds.size else 0.0,
        "path_length": float(np.sum(distances)) if distances.size else 0.0,
    }


def weighted_sum(values: list[float], weights: list[float]) -> float:
    """Compute a weighted sum.

    Args:
        values: Numeric values.
        weights: Matching weights.

    Returns:
        Weighted sum.
    """

    return float(sum(float(v) * float(w) for v, w in zip(values, weights)))


def parse_weights(value: str, *, expected: int, name: str) -> list[float]:
    """Parse comma-separated scoring weights.

    Args:
        value: Raw CLI value.
        expected: Expected number of weights.
        name: Name used in error messages.

    Returns:
        Parsed weight list.

    Raises:
        ValueError: If the number of weights is wrong.
    """

    weights = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(weights) != expected:
        raise ValueError(f"{name} requires {expected} comma-separated weights, got {len(weights)}")
    return weights


def robust_scale(values: list[float], *, percentile: float) -> list[float]:
    """Scale values by a robust positive percentile.

    Args:
        values: Raw values.
        percentile: Percentile denominator.

    Returns:
        Values divided by the selected percentile and clipped at ``1``.
    """

    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.0 for _ in values]
    denom = float(np.percentile(finite, float(percentile)))
    if denom <= 1e-12:
        denom = float(np.max(finite)) if finite.size else 1.0
    if denom <= 1e-12:
        return [0.0 for _ in values]
    return [float(max(0.0, min(float(v) / denom, 1.0))) if math.isfinite(float(v)) else 0.0 for v in values]


def load_behavior_decisions(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load behavior decisions by id.

    Args:
        path: Optional decisions JSONL path.

    Returns:
        Mapping from sample id to behavior filter dictionary.
    """

    decisions: dict[str, dict[str, Any]] = {}
    if path is None or not path.is_file():
        return decisions
    for row in read_jsonl(path):
        sample_id = str(row.get("id") or "")
        behavior = row.get("behavior_filter")
        if sample_id and isinstance(behavior, dict):
            decisions[sample_id] = behavior
    return decisions


def behavior_label(row: dict[str, Any], behavior_by_id: dict[str, dict[str, Any]], sample_id: str) -> int | None:
    """Resolve a row's behavior label.

    Args:
        row: Manifest row.
        behavior_by_id: Optional external decision map.
        sample_id: Stable sample id.

    Returns:
        Integer label, or ``None`` when missing/unparseable.
    """

    behavior = row.get("behavior_filter")
    if not isinstance(behavior, dict):
        behavior = behavior_by_id.get(sample_id)
    if not isinstance(behavior, dict):
        return None
    try:
        return int(behavior.get("label"))
    except Exception:
        return None


def behavior_priority_weight(label: int | None, *, interesting_weight: float, boring_weight: float) -> float:
    """Return the VLM priority multiplier for a behavior label.

    Args:
        label: ``1`` interesting, ``2`` boring, or ``None``.
        interesting_weight: Weight for interesting rows.
        boring_weight: Weight for boring rows.

    Returns:
        Priority multiplier.
    """

    if label == 1:
        return float(interesting_weight)
    if label == 2:
        return float(boring_weight)
    return 1.0


def score_rows(
    rows: list[dict[str, Any]],
    *,
    behavior_by_id: dict[str, dict[str, Any]],
    default_dt_sec: float,
    accel_weights: list[float],
    turn_weights: list[float],
    combine_mode: str,
    scale_percentile: float,
    interesting_weight: float,
    boring_weight: float,
) -> list[dict[str, Any]]:
    """Compute saliency records for all rows.

    Args:
        rows: Source manifest rows.
        behavior_by_id: Optional VLM behavior decisions by id.
        default_dt_sec: Fallback time delta.
        accel_weights: Weights for average, max, and variance acceleration.
        turn_weights: Weights for average, max, and variance turning.
        combine_mode: ``max`` or ``sum``.
        scale_percentile: Percentile used for robust score scaling.
        interesting_weight: VLM interesting multiplier.
        boring_weight: VLM boring multiplier.

    Returns:
        List of scored record dictionaries in source order.
    """

    records: list[dict[str, Any]] = []
    raw_accel_scores: list[float] = []
    raw_turn_scores: list[float] = []

    for index, row in enumerate(rows, start=1):
        sample_id = row_id(row, index=index)
        features = compute_motion_features(row, default_dt_sec=default_dt_sec)
        accel_score = weighted_sum(
            [features["accel_avg"], features["accel_max"], features["accel_var"]],
            accel_weights,
        )
        turn_score = weighted_sum(
            [features["turn_avg_deg"], features["turn_max_deg"], features["turn_var_deg2"]],
            turn_weights,
        )
        raw_accel_scores.append(accel_score)
        raw_turn_scores.append(turn_score)
        records.append(
            {
                "source_index": index,
                "id": sample_id,
                "clip_id": row_clip_id(row),
                "frame_idx": row_frame_idx(row, fallback=index),
                "intent_gt": row_intent(row),
                "row": row,
                "features": features,
                "raw_accel_saliency": accel_score,
                "raw_turn_saliency": turn_score,
            }
        )

    scaled_accel = robust_scale(raw_accel_scores, percentile=scale_percentile)
    scaled_turn = robust_scale(raw_turn_scores, percentile=scale_percentile)
    for record, accel_norm, turn_norm in zip(records, scaled_accel, scaled_turn):
        sample_id = str(record["id"])
        label = behavior_label(record["row"], behavior_by_id, sample_id)
        if combine_mode == "sum":
            combined = float(accel_norm) + float(turn_norm)
        else:
            combined = max(float(accel_norm), float(turn_norm))
        priority = behavior_priority_weight(
            label,
            interesting_weight=interesting_weight,
            boring_weight=boring_weight,
        )
        record["motion_saliency_filter"] = {
            "schema_version": SCHEMA_VERSION,
            "acceleration_saliency": {
                "avg": record["features"]["accel_avg"],
                "max": record["features"]["accel_max"],
                "variance": record["features"]["accel_var"],
                "raw_score": record["raw_accel_saliency"],
                "normalized_score": accel_norm,
                "weights": accel_weights,
            },
            "turning_saliency": {
                "avg_deg": record["features"]["turn_avg_deg"],
                "max_deg": record["features"]["turn_max_deg"],
                "variance_deg2": record["features"]["turn_var_deg2"],
                "raw_score": record["raw_turn_saliency"],
                "normalized_score": turn_norm,
                "weights": turn_weights,
            },
            "motion_features": record["features"],
            "behavior_label": label,
            "priority_weight": priority,
            "combine_mode": combine_mode,
            "base_score": combined,
            "priority_score": combined * priority,
            "selected": False,
            "selection_reason": "not_selected",
        }
    return records


def selection_target_count(total: int, *, top_k: int | None, top_fraction: float, keep_all: bool) -> int:
    """Compute the requested number of selected rows.

    Args:
        total: Total row count.
        top_k: Optional exact target count.
        top_fraction: Optional fraction in ``(0, 1]``.
        keep_all: Whether all rows should be kept.

    Returns:
        Target selected count.
    """

    if keep_all:
        return total
    if top_k is not None and top_k > 0:
        return min(total, int(top_k))
    if top_fraction > 0:
        return min(total, max(1, int(round(total * float(top_fraction)))))
    return total


def select_records(
    records: list[dict[str, Any]],
    *,
    top_k: int | None,
    top_fraction: float,
    min_score: float | None,
    min_frame_gap: int,
    keep_all: bool,
) -> set[str]:
    """Select high-value rows by score and temporal diversity.

    Args:
        records: Scored records.
        top_k: Optional exact selected count.
        top_fraction: Optional selected fraction.
        min_score: Optional priority-score floor.
        min_frame_gap: Minimum same-clip frame distance between selected rows.
        keep_all: Select all rows.

    Returns:
        Set of selected sample ids.
    """

    if keep_all:
        return {str(record["id"]) for record in records}
    target = selection_target_count(len(records), top_k=top_k, top_fraction=top_fraction, keep_all=False)
    ranked = sorted(
        records,
        key=lambda record: (
            float(record["motion_saliency_filter"]["priority_score"]),
            float(record["motion_saliency_filter"]["base_score"]),
        ),
        reverse=True,
    )
    selected: set[str] = set()
    selected_frames_by_clip: dict[str, list[int]] = {}
    gap = max(0, int(min_frame_gap))
    for record in ranked:
        score = float(record["motion_saliency_filter"]["priority_score"])
        if min_score is not None and score < float(min_score):
            continue
        clip_id = str(record["clip_id"])
        frame_idx = int(record["frame_idx"])
        if gap > 0 and any(abs(frame_idx - prev) < gap for prev in selected_frames_by_clip.get(clip_id, [])):
            continue
        selected.add(str(record["id"]))
        selected_frames_by_clip.setdefault(clip_id, []).append(frame_idx)
        if len(selected) >= target:
            break
    return selected


def write_scored_outputs(
    *,
    records: list[dict[str, Any]],
    selected_ids: set[str],
    output_jsonl: Path,
    rejected_jsonl: Path,
    decisions_jsonl: Path,
) -> dict[str, int]:
    """Write kept, rejected, and scored decision JSONL files.

    Args:
        records: Scored records.
        selected_ids: Selected sample ids.
        output_jsonl: Kept-row JSONL path.
        rejected_jsonl: Rejected-row JSONL path.
        decisions_jsonl: Per-row saliency JSONL path.

    Returns:
        Count dictionary.
    """

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
    decisions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8")
    rejected_jsonl.write_text("", encoding="utf-8")
    decisions_jsonl.write_text("", encoding="utf-8")

    counts: Counter[str] = Counter()
    ranked_ids = {
        str(record["id"]): rank
        for rank, record in enumerate(
            sorted(records, key=lambda item: float(item["motion_saliency_filter"]["priority_score"]), reverse=True),
            start=1,
        )
    }
    for record in records:
        selected = str(record["id"]) in selected_ids
        saliency = dict(record["motion_saliency_filter"])
        saliency["selected"] = bool(selected)
        saliency["selection_rank"] = ranked_ids.get(str(record["id"]))
        saliency["selection_reason"] = "high_motion_saliency" if selected else "below_selection_cut"
        row = dict(record["row"])
        row["motion_saliency_filter"] = saliency
        decision_row = {
            "id": record["id"],
            "clip_id": record["clip_id"],
            "frame_idx": record["frame_idx"],
            "intent_gt": record["intent_gt"],
            "motion_saliency_filter": saliency,
        }
        append_jsonl(decisions_jsonl, decision_row)
        if selected:
            append_jsonl(output_jsonl, row)
            counts["kept"] += 1
        else:
            append_jsonl(rejected_jsonl, row)
            counts["rejected"] += 1
    return dict(counts)


def filter_motion_saliency_samples(
    *,
    source_jsonl: Path,
    output_jsonl: Path,
    rejected_jsonl: Path,
    decisions_jsonl: Path,
    report_json: Path,
    behavior_decisions_jsonl: Path | None,
    limit: int | None,
    default_dt_sec: float,
    accel_weights: list[float],
    turn_weights: list[float],
    combine_mode: str,
    scale_percentile: float,
    interesting_weight: float,
    boring_weight: float,
    top_k: int | None,
    top_fraction: float,
    min_score: float | None,
    min_frame_gap: int,
    keep_all: bool,
) -> dict[str, Any]:
    """Run motion-saliency filtering over a source manifest.

    Args:
        source_jsonl: Source manifest JSONL.
        output_jsonl: Kept-row output JSONL.
        rejected_jsonl: Rejected-row output JSONL.
        decisions_jsonl: Scored decisions JSONL.
        report_json: Summary report path.
        behavior_decisions_jsonl: Optional behavior-filter decision JSONL.
        limit: Optional input row cap.
        default_dt_sec: Fallback time delta.
        accel_weights: Acceleration formula weights.
        turn_weights: Turning formula weights.
        combine_mode: Score fusion mode.
        scale_percentile: Robust score scaling percentile.
        interesting_weight: Priority multiplier for label ``1``.
        boring_weight: Priority multiplier for label ``2``.
        top_k: Optional exact selection target.
        top_fraction: Optional selection fraction.
        min_score: Optional priority-score floor.
        min_frame_gap: Same-clip temporal diversity gap.
        keep_all: Keep all rows while still writing scores.

    Returns:
        Summary report dictionary.
    """

    rows = list(read_jsonl(source_jsonl))
    if limit is not None:
        rows = rows[: int(limit)]
    behavior_by_id = load_behavior_decisions(behavior_decisions_jsonl)
    records = score_rows(
        rows,
        behavior_by_id=behavior_by_id,
        default_dt_sec=default_dt_sec,
        accel_weights=accel_weights,
        turn_weights=turn_weights,
        combine_mode=combine_mode,
        scale_percentile=scale_percentile,
        interesting_weight=interesting_weight,
        boring_weight=boring_weight,
    )
    selected_ids = select_records(
        records,
        top_k=top_k,
        top_fraction=top_fraction,
        min_score=min_score,
        min_frame_gap=min_frame_gap,
        keep_all=keep_all,
    )
    counts = write_scored_outputs(
        records=records,
        selected_ids=selected_ids,
        output_jsonl=output_jsonl,
        rejected_jsonl=rejected_jsonl,
        decisions_jsonl=decisions_jsonl,
    )

    intent_counts: Counter[str] = Counter()
    kept_intent_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    kept_behavior_counts: Counter[str] = Counter()
    for record in records:
        intent = str(record["intent_gt"])
        intent_counts[intent] += 1
        label = record["motion_saliency_filter"].get("behavior_label")
        behavior_counts[str(label if label is not None else "missing")] += 1
        if str(record["id"]) in selected_ids:
            kept_intent_counts[intent] += 1
            kept_behavior_counts[str(label if label is not None else "missing")] += 1

    scores = [float(record["motion_saliency_filter"]["priority_score"]) for record in records]
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_jsonl": str(source_jsonl),
        "output_jsonl": str(output_jsonl),
        "rejected_jsonl": str(rejected_jsonl),
        "decisions_jsonl": str(decisions_jsonl),
        "behavior_decisions_jsonl": str(behavior_decisions_jsonl) if behavior_decisions_jsonl else None,
        "input_rows": len(rows),
        "limit": limit,
        "counts": counts,
        "intent_counts": dict(sorted(intent_counts.items())),
        "kept_intent_counts": dict(sorted(kept_intent_counts.items())),
        "behavior_label_counts": dict(sorted(behavior_counts.items())),
        "kept_behavior_label_counts": dict(sorted(kept_behavior_counts.items())),
        "selection": {
            "top_k": top_k,
            "top_fraction": top_fraction,
            "min_score": min_score,
            "min_frame_gap": min_frame_gap,
            "keep_all": keep_all,
            "selected_rows": len(selected_ids),
        },
        "scoring": {
            "default_dt_sec": default_dt_sec,
            "accel_weights": accel_weights,
            "turn_weights": turn_weights,
            "combine_mode": combine_mode,
            "scale_percentile": scale_percentile,
            "interesting_weight": interesting_weight,
            "boring_weight": boring_weight,
        },
        "score_summary": {
            "min": float(np.min(scores)) if scores else 0.0,
            "p50": float(np.percentile(scores, 50)) if scores else 0.0,
            "p90": float(np.percentile(scores, 90)) if scores else 0.0,
            "p95": float(np.percentile(scores, 95)) if scores else 0.0,
            "max": float(np.max(scores)) if scores else 0.0,
        },
    }
    write_json(report_json, report)
    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--rejected-jsonl", required=True)
    parser.add_argument("--decisions-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--behavior-decisions-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--default-dt-sec", type=float, default=0.2)
    parser.add_argument("--accel-weights", default="0.35,0.45,0.20")
    parser.add_argument("--turn-weights", default="0.35,0.45,0.20")
    parser.add_argument("--combine-mode", default="max", choices=["max", "sum"])
    parser.add_argument("--scale-percentile", type=float, default=95.0)
    parser.add_argument("--interesting-weight", type=float, default=1.5)
    parser.add_argument("--boring-weight", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-fraction", type=float, default=0.35)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--min-frame-gap", type=int, default=5)
    parser.add_argument("--keep-all", action="store_true", help="Write every row to kept output while still scoring decisions.")
    return parser.parse_args()


def main() -> None:
    """Run the motion-saliency filter CLI."""

    args = parse_args()
    try:
        accel_weights = parse_weights(args.accel_weights, expected=3, name="--accel-weights")
        turn_weights = parse_weights(args.turn_weights, expected=3, name="--turn-weights")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not (0.0 < float(args.scale_percentile) <= 100.0):
        raise SystemExit("--scale-percentile must be in (0, 100]")
    if float(args.top_fraction) < 0:
        raise SystemExit("--top-fraction must be >= 0")
    report = filter_motion_saliency_samples(
        source_jsonl=Path(args.source_jsonl),
        output_jsonl=Path(args.output_jsonl),
        rejected_jsonl=Path(args.rejected_jsonl),
        decisions_jsonl=Path(args.decisions_jsonl),
        report_json=Path(args.report_json),
        behavior_decisions_jsonl=Path(args.behavior_decisions_jsonl) if args.behavior_decisions_jsonl else None,
        limit=args.limit,
        default_dt_sec=float(args.default_dt_sec),
        accel_weights=accel_weights,
        turn_weights=turn_weights,
        combine_mode=str(args.combine_mode),
        scale_percentile=float(args.scale_percentile),
        interesting_weight=float(args.interesting_weight),
        boring_weight=float(args.boring_weight),
        top_k=args.top_k,
        top_fraction=float(args.top_fraction),
        min_score=args.min_score,
        min_frame_gap=int(args.min_frame_gap),
        keep_all=bool(args.keep_all),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
