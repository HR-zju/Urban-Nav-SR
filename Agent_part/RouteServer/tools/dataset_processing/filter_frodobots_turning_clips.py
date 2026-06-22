#!/usr/bin/env python3
"""Build a turn-heavy FrodoBots clip subset before Phase1/CoT conversion.

The input is the BEV sampling batch report produced by
``run_frodobots_bev_sampling_batch.py``.  This script reads each completed
clip's ``bev_sample_trajectory.json``, scores local GPS bearing changes, and
creates a symlink subset sorted toward clips with more turns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any


def angle_delta_deg(a: float, b: float) -> float:
    return abs((b - a + 180.0) % 360.0 - 180.0)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def load_points(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("samples") or data.get("trajectory") or data.get("frames") or []
    else:
        rows = []
    points = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        lat = row.get("lat")
        lon = row.get("lng", row.get("lon"))
        if lat is None or lon is None:
            continue
        try:
            points.append({"lat": float(lat), "lon": float(lon)})
        except (TypeError, ValueError):
            continue
    return points


def score_clip(
    clip_dir: Path,
    *,
    window_samples: int,
    min_leg_distance_m: float,
    min_turn_deg: float,
    max_turn_deg: float,
) -> dict[str, Any]:
    traj_path = clip_dir / "bev_sample_trajectory.json"
    points = load_points(traj_path)
    changes: list[float] = []
    turn_changes: list[float] = []
    total_distance_m = 0.0

    for a, b in zip(points, points[1:]):
        total_distance_m += haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])

    for i in range(window_samples, len(points) - window_samples):
        prev = points[i - window_samples]
        mid = points[i]
        nxt = points[i + window_samples]
        before_dist = haversine_m(prev["lat"], prev["lon"], mid["lat"], mid["lon"])
        after_dist = haversine_m(mid["lat"], mid["lon"], nxt["lat"], nxt["lon"])
        if before_dist < min_leg_distance_m or after_dist < min_leg_distance_m:
            continue
        before_bearing = bearing_deg(prev["lat"], prev["lon"], mid["lat"], mid["lon"])
        after_bearing = bearing_deg(mid["lat"], mid["lon"], nxt["lat"], nxt["lon"])
        change = angle_delta_deg(before_bearing, after_bearing)
        changes.append(change)
        if min_turn_deg <= change <= max_turn_deg:
            turn_changes.append(change)

    return {
        "clip_dir": str(clip_dir),
        "trajectory_path": str(traj_path),
        "trajectory_samples": len(points),
        "total_distance_m": round(total_distance_m, 3),
        "usable_windows": len(changes),
        "turn_windows": len(turn_changes),
        "turn_window_ratio": round((len(turn_changes) / len(changes)) if changes else 0.0, 6),
        "median_turn_deg": round(median(turn_changes), 3) if turn_changes else 0.0,
        "p90_turn_deg": round(percentile(turn_changes, 0.90), 3),
        "p95_change_deg": round(percentile(changes, 0.95), 3),
        "max_change_deg": round(max(changes), 3) if changes else 0.0,
        "has_trajectory": traj_path.exists(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter sampled FrodoBots clips toward turning sequences")
    parser.add_argument("--sampling-report", required=True, help="sampling_report_top*.json from BEV sampling batch")
    parser.add_argument("--output-root", required=True, help="Directory for metrics/report/symlink subset")
    parser.add_argument("--keep-percent", type=float, default=50.0, help="Keep top N percent after thresholding")
    parser.add_argument("--min-turn-windows", type=int, default=3, help="Minimum local turn windows required")
    parser.add_argument("--min-turn-deg", type=float, default=20.0)
    parser.add_argument("--max-turn-deg", type=float, default=140.0)
    parser.add_argument("--window-samples", type=int, default=5, help="BEV samples per side for bearing change")
    parser.add_argument("--min-leg-distance-m", type=float, default=1.5)
    parser.add_argument("--symlink-dir-name", default="clips_turning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.sampling_report)
    output_root = Path(args.output_root)
    symlink_root = output_root / args.symlink_dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    symlink_root.mkdir(parents=True, exist_ok=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in report.get("results", [])
        if isinstance(row, dict) and str(row.get("status")) in {"done", "skipped_existing", "exists"}
    ]

    metrics: list[dict[str, Any]] = []
    for row in rows:
        clip_dir = Path(str(row.get("clip_dir") or ""))
        metric = score_clip(
            clip_dir,
            window_samples=max(1, int(args.window_samples)),
            min_leg_distance_m=float(args.min_leg_distance_m),
            min_turn_deg=float(args.min_turn_deg),
            max_turn_deg=float(args.max_turn_deg),
        )
        metric["sequence_name"] = row.get("sequence_name") or clip_dir.name
        metric["frame_count"] = row.get("frame_count")
        metrics.append(metric)

    candidates = [m for m in metrics if int(m["turn_windows"]) >= int(args.min_turn_windows)]
    candidates.sort(
        key=lambda m: (
            int(m["turn_windows"]),
            float(m["turn_window_ratio"]),
            float(m["p90_turn_deg"]),
            int(m.get("frame_count") or 0),
        ),
        reverse=True,
    )
    keep_n = int(math.ceil(len(candidates) * max(0.0, min(float(args.keep_percent), 100.0)) / 100.0))
    selected = candidates[:keep_n]

    for rank, metric in enumerate(selected):
        src = Path(str(metric["clip_dir"]))
        dst = symlink_root / src.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(src, dst, target_is_directory=True)
        metric["subset_rank"] = rank

    csv_path = output_root / "turning_clip_metrics.csv"
    fieldnames = [
        "subset_rank",
        "sequence_name",
        "clip_dir",
        "frame_count",
        "trajectory_samples",
        "total_distance_m",
        "usable_windows",
        "turn_windows",
        "turn_window_ratio",
        "median_turn_deg",
        "p90_turn_deg",
        "p95_change_deg",
        "max_change_deg",
        "has_trajectory",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for metric in selected + [m for m in metrics if m not in selected]:
            writer.writerow(metric)

    out_report = {
        "schema_version": "frodobots_turning_clip_filter_v1",
        "sampling_report": str(report_path),
        "output_root": str(output_root),
        "symlink_root": str(symlink_root),
        "input_clips": len(rows),
        "candidate_clips": len(candidates),
        "selected_clips": len(selected),
        "keep_percent": float(args.keep_percent),
        "min_turn_windows": int(args.min_turn_windows),
        "min_turn_deg": float(args.min_turn_deg),
        "max_turn_deg": float(args.max_turn_deg),
        "window_samples": int(args.window_samples),
        "min_leg_distance_m": float(args.min_leg_distance_m),
        "metrics_csv": str(csv_path),
        "selected": selected,
    }
    report_out = output_root / "turning_clip_filter_report.json"
    report_out.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out_report.items() if k != "selected"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
