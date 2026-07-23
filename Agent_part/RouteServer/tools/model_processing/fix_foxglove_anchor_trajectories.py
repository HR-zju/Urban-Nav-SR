#!/usr/bin/env python3
"""Anchor Foxglove Phase1 FPV trajectories at the robot-side bottom center.

This is a post-processing repair for already-labeled Phase1 manifests.  It
does not regenerate CoT text or touch media files.  For Foxglove rows only, it
updates the normalized FPV trajectory targets so the first point is always
``[0.5, 0.9]`` and subsequent points move monotonically upward in image space.
The same x/y coordinates are synchronized across ``pixel_trajectory``,
``trajectory_fpv_t``, ``trajectory_fpv_tv``, and the assistant ``<finish>`` JSON
when present.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


FINISH_RE = re.compile(r"<finish>\s*(\{.*?\})\s*</finish>", flags=re.DOTALL)
ANCHOR = [0.5, 0.9]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clamp01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.0
    return min(1.0, max(0.0, x))


def round5(value: Any) -> float:
    return round(float(value), 5)


def row_id(row: dict[str, Any], line_no: int) -> str:
    return str(
        row.get("id")
        or f"{row.get('clip_id') or (row.get('sidewalk_meta') or {}).get('clip_id') or 'row'}_{line_no}"
    )


def intent_of(row: dict[str, Any]) -> str:
    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    return str(row.get("intent") or row.get("intent_gt") or meta.get("intent_gt") or meta.get("intent") or "unknown")


def clip_id_of(row: dict[str, Any]) -> str:
    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    return str(row.get("clip_id") or meta.get("clip_id") or "")


def frame_idx_of(row: dict[str, Any]) -> int | None:
    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    value = row.get("frame_idx", meta.get("frame_idx"))
    try:
        return int(value)
    except Exception:
        return None


def path_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "fpv_image_path",
        "bev_image_path",
        "bev_patch_meta_path",
        "trajectory_label_path",
    ):
        value = row.get(key)
        if value:
            values.append(str(value))
    for key in ("fpv_image_paths", "history_fpv_image_paths", "bev_image_paths", "images"):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    for key in ("source_dataset", "bev_patch_meta_path", "trajectory_label_path"):
        value = meta.get(key)
        if value:
            values.append(str(value))
    return values


def is_foxglove_row(row: dict[str, Any]) -> bool:
    text = " ".join(path_values(row)).replace("\\", "/").lower()
    return "foxglove" in text


def normalize_points(points: Any) -> list[list[float]]:
    out: list[list[float]] = []
    if not isinstance(points, list):
        return out
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        out.append([clamp01(item[0]), clamp01(item[1])])
    return out


def anchor_and_smooth(points: list[list[float]], *, min_y_step: float) -> list[list[float]]:
    """Force a bottom-center anchor and monotonic upward y progression."""

    if not points:
        return []
    arr = [[clamp01(x), clamp01(y)] for x, y in points]
    arr[0] = list(ANCHOR)
    step = max(float(min_y_step), 0.0)
    for i in range(1, len(arr)):
        prev = list(arr[i - 1])
        max_allowed_y = max(0.02, float(prev[1]) - step)
        if float(arr[i][1]) <= max_allowed_y:
            arr[i] = [clamp01(arr[i][0]), clamp01(arr[i][1])]
            continue

        replacement = list(arr[i])
        for j in range(i + 1, len(arr)):
            if float(arr[j][1]) <= max_allowed_y:
                alpha = 1.0 / float(j - i + 1)
                replacement = [
                    prev[0] * (1.0 - alpha) + float(arr[j][0]) * alpha,
                    prev[1] * (1.0 - alpha) + float(arr[j][1]) * alpha,
                ]
                break
        replacement[0] = clamp01(replacement[0])
        replacement[1] = clamp01(min(float(replacement[1]), max_allowed_y))
        arr[i] = replacement
    return [[round5(x), round5(y)] for x, y in arr]


def trajectory_region_and_type(points: list[list[float]]) -> dict[str, str]:
    if not points:
        return {
            "fpv_route_region": "unknown",
            "route_to_fpv_direction": "unknown",
            "safe_corridor": "unknown",
            "lateral_preference": "center",
            "trajectory_type": "unknown",
            "speed_policy": "normal",
        }
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    mean_x = sum(xs) / float(len(xs))
    delta_x = xs[-1] - xs[0] if len(xs) > 1 else 0.0
    y_span = max(ys) - min(ys) if len(ys) > 1 else 0.0
    if mean_x < 0.42:
        region = "left"
    elif mean_x > 0.58:
        region = "right"
    else:
        region = "center"
    if abs(delta_x) < 0.035:
        turn = "straight"
    elif delta_x < 0.0:
        turn = "slight_left"
    else:
        turn = "slight_right"
    if y_span < 0.08:
        trajectory_type = "stop_hold"
        speed_policy = "stop"
    else:
        trajectory_type = f"{turn}_{region}"
        speed_policy = "normal" if y_span > 0.18 else "slow"
    route_to_fpv = "forward_center" if region == "center" else f"forward_{region}"
    return {
        "fpv_route_region": region,
        "route_to_fpv_direction": route_to_fpv,
        "safe_corridor": f"{region}_corridor",
        "lateral_preference": region,
        "trajectory_type": trajectory_type,
        "speed_policy": speed_policy,
    }


def update_fpv_t(value: Any, points: list[list[float]]) -> list[list[float]]:
    old = value if isinstance(value, list) else []
    out: list[list[float]] = []
    for i, point in enumerate(points):
        t = 0.0
        if i < len(old) and isinstance(old[i], (list, tuple)) and len(old[i]) >= 3:
            t = float(old[i][2])
        out.append([point[0], point[1], round5(t)])
    return out


def update_fpv_tv(value: Any, points: list[list[float]]) -> list[list[float]]:
    old = value if isinstance(value, list) else []
    out: list[list[float]] = []
    for i, point in enumerate(points):
        t = 0.0
        speed = 0.0
        if i < len(old) and isinstance(old[i], (list, tuple)):
            if len(old[i]) >= 3:
                t = float(old[i][2])
            if len(old[i]) >= 4:
                speed = float(old[i][3])
        out.append([point[0], point[1], round5(t), round5(speed)])
    return out


def update_finish_response(response: str, points: list[list[float]]) -> tuple[str, bool, str | None]:
    if not response:
        return response, False, "missing_response"
    match = FINISH_RE.search(response)
    if not match:
        return response, False, "missing_finish"
    try:
        finish = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return response, False, f"invalid_finish_json:{exc.msg}"
    if not isinstance(finish, dict):
        return response, False, "finish_not_object"
    finish["trajectory"] = points
    finish["trajectory_type"] = trajectory_region_and_type(points)["trajectory_type"]
    finish["trajectory_validity"] = finish.get("trajectory_validity") or "valid"
    replacement = f"<finish>{json.dumps(finish, ensure_ascii=False, separators=(',', ':'))}</finish>"
    return response[: match.start()] + replacement + response[match.end() :], True, None


def start_is_anchor(points: list[list[float]], *, eps: float) -> bool:
    if not points:
        return False
    return abs(float(points[0][0]) - ANCHOR[0]) <= eps and abs(float(points[0][1]) - ANCHOR[1]) <= eps


def start_is_edge(points: list[list[float]], *, threshold: float) -> bool:
    if not points:
        return False
    x = float(points[0][0])
    return x <= float(threshold) or x >= 1.0 - float(threshold)


def make_qc_row(row: dict[str, Any], *, line_no: int, old_points: list[list[float]], new_points: list[list[float]]) -> dict[str, Any]:
    qc = deepcopy(row)
    qc["anchor_fix_qc"] = {
        "reason": "foxglove_start_x_on_left_or_right_edge_before_anchor_fix",
        "line_no": line_no,
        "id": row_id(row, line_no),
        "clip_id": clip_id_of(row),
        "frame_idx": frame_idx_of(row),
        "intent": intent_of(row),
        "old_start": old_points[0] if old_points else None,
        "new_start": new_points[0] if new_points else None,
        "old_pixel_trajectory": old_points,
        "new_pixel_trajectory": new_points,
    }
    return qc


def repair_row(
    row: dict[str, Any],
    *,
    line_no: int,
    min_y_step: float,
    anchor_eps: float,
    edge_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    info: dict[str, Any] = {
        "is_foxglove": False,
        "changed": False,
        "strong_qc": False,
        "response_updated": False,
        "response_error": None,
    }
    if not is_foxglove_row(row):
        return row, info
    info["is_foxglove"] = True
    old_points = normalize_points(row.get("pixel_trajectory"))
    if not old_points:
        info["response_error"] = "missing_pixel_trajectory"
        return row, info

    info["already_anchor"] = start_is_anchor(old_points, eps=anchor_eps)
    info["strong_qc"] = start_is_edge(old_points, threshold=edge_threshold)
    new_points = anchor_and_smooth(old_points, min_y_step=min_y_step)
    changed = new_points != [[round5(x), round5(y)] for x, y in old_points]
    if not changed:
        return row, info

    out = deepcopy(row)
    out["pixel_trajectory"] = new_points
    out["trajectory_fpv_t"] = update_fpv_t(row.get("trajectory_fpv_t"), new_points)
    out["trajectory_fpv_tv"] = update_fpv_tv(row.get("trajectory_fpv_tv"), new_points)
    if "response" in out:
        response, response_updated, response_error = update_finish_response(str(out.get("response") or ""), new_points)
        out["response"] = response
        info["response_updated"] = response_updated
        info["response_error"] = response_error
    if isinstance(out.get("cot_teacher"), dict):
        post = dict(out["cot_teacher"].get("postprocess") or {})
        post["foxglove_anchor_fix"] = {
            "schema_version": "foxglove_anchor_fix_v1",
            "old_start": [round5(old_points[0][0]), round5(old_points[0][1])],
            "new_start": list(ANCHOR),
            "min_y_step": float(min_y_step),
        }
        out["cot_teacher"]["postprocess"] = post
    info["changed"] = True
    info["old_points"] = old_points
    info["new_points"] = new_points
    return out, info


def process_manifest(
    path: Path,
    *,
    backup_suffix: str,
    min_y_step: float,
    anchor_eps: float,
    edge_threshold: float,
    qc_jsonl: Path | None,
    report_qc_seen: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    response_errors: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    tmp_path = path.with_name(f".{path.name}.tmp_anchor_fix_{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    with tmp_path.open("w", encoding="utf-8") as out_f:
        for line_no, row in read_jsonl(path):
            counters["input_rows"] += 1
            repaired, info = repair_row(
                row,
                line_no=line_no,
                min_y_step=min_y_step,
                anchor_eps=anchor_eps,
                edge_threshold=edge_threshold,
            )
            if info.get("is_foxglove"):
                counters["foxglove_rows"] += 1
                if info.get("already_anchor"):
                    counters["foxglove_already_anchor"] += 1
                else:
                    counters["foxglove_non_anchor_start"] += 1
                if info.get("strong_qc"):
                    counters["foxglove_strong_qc_edge_start"] += 1
                    key = row_id(row, line_no)
                    if qc_jsonl is not None and key not in report_qc_seen and info.get("new_points"):
                        append_jsonl(
                            qc_jsonl,
                            make_qc_row(
                                row,
                                line_no=line_no,
                                old_points=info["old_points"],
                                new_points=info["new_points"],
                            ),
                        )
                        report_qc_seen.add(key)
                if info.get("changed"):
                    counters["foxglove_changed"] += 1
                if info.get("response_updated"):
                    counters["response_finish_updated"] += 1
                if info.get("response_error"):
                    response_errors[str(info["response_error"])] += 1
                if info.get("changed") and len(examples) < 10:
                    old_points = info.get("old_points") or []
                    new_points = info.get("new_points") or []
                    examples.append(
                        {
                            "line_no": line_no,
                            "id": row_id(row, line_no),
                            "intent": intent_of(row),
                            "old_start": old_points[0] if old_points else None,
                            "new_start": new_points[0] if new_points else None,
                            "old_end": old_points[-1] if old_points else None,
                            "new_end": new_points[-1] if new_points else None,
                        }
                    )
            out_f.write(json.dumps(repaired, ensure_ascii=False, separators=(",", ":")) + "\n")

    counters["output_rows"] = counters["input_rows"]
    backup_path = path.with_name(path.name + backup_suffix)
    if not dry_run:
        if backup_path.exists():
            raise FileExistsError(f"Backup already exists: {backup_path}")
        shutil.copy2(path, backup_path)
        os.replace(tmp_path, path)
    else:
        tmp_path.unlink(missing_ok=True)
    return {
        "path": str(path),
        "backup_path": str(backup_path) if not dry_run else None,
        "counts": dict(counters),
        "response_errors": dict(response_errors),
        "examples": examples,
    }


def verify_manifest(path: Path, *, anchor_eps: float, edge_threshold: float) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    for _line_no, row in read_jsonl(path):
        counters["rows"] += 1
        if not is_foxglove_row(row):
            continue
        counters["foxglove_rows"] += 1
        points = normalize_points(row.get("pixel_trajectory"))
        if start_is_anchor(points, eps=anchor_eps):
            counters["foxglove_anchor_start"] += 1
        else:
            counters["foxglove_non_anchor_start"] += 1
        if start_is_edge(points, threshold=edge_threshold):
            counters["foxglove_edge_start_after"] += 1
        bad_monotonic = False
        for prev, cur in zip(points, points[1:]):
            if float(cur[1]) > float(prev[1]) - 1e-8:
                bad_monotonic = True
                break
        if bad_monotonic:
            counters["foxglove_non_monotonic_y"] += 1
    return dict(counters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True, help="Manifest JSONL to repair. Can be repeated.")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--strong-qc-jsonl", required=True)
    parser.add_argument("--backup-suffix", default=None)
    parser.add_argument("--min-y-step", type=float, default=0.005)
    parser.add_argument("--anchor-eps", type=float, default=1e-5)
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_suffix = args.backup_suffix or f".bak_anchor_fix_{timestamp}"
    report_path = Path(args.report_json)
    qc_path = Path(args.strong_qc_jsonl)
    if qc_path.exists():
        qc_path.unlink()
    qc_seen: set[str] = set()

    reports = []
    for raw in args.jsonl:
        reports.append(
            process_manifest(
                Path(raw),
                backup_suffix=backup_suffix,
                min_y_step=float(args.min_y_step),
                anchor_eps=float(args.anchor_eps),
                edge_threshold=float(args.edge_threshold),
                qc_jsonl=qc_path,
                report_qc_seen=qc_seen,
                dry_run=bool(args.dry_run),
            )
        )

    verifications = [
        {
            "path": str(path),
            "verify": verify_manifest(Path(path), anchor_eps=float(args.anchor_eps), edge_threshold=float(args.edge_threshold)),
        }
        for path in args.jsonl
    ]
    report = {
        "schema_version": "foxglove_anchor_fix_v1_report",
        "backup_suffix": backup_suffix,
        "min_y_step": float(args.min_y_step),
        "anchor": ANCHOR,
        "edge_threshold": float(args.edge_threshold),
        "dry_run": bool(args.dry_run),
        "manifests": reports,
        "strong_qc_jsonl": str(qc_path),
        "strong_qc_rows": len(qc_seen),
        "verifications": verifications,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
