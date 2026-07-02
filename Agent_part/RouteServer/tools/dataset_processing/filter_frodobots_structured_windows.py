#!/usr/bin/env python3
"""Select structured FrodoBots trajectory windows from sampled prepared clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def angle_delta_deg(a: float, b: float) -> float:
    return abs((b - a + 180.0) % 360.0 - 180.0)


def segment_bearings(points_xy: np.ndarray) -> np.ndarray:
    deltas = np.diff(points_xy[:, :2], axis=0)
    return (np.degrees(np.arctan2(deltas[:, 0], deltas[:, 1])) + 360.0) % 360.0


def path_metrics(points_xy: np.ndarray, speeds: np.ndarray | None = None) -> dict[str, Any]:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return {}
    deltas = np.diff(points, axis=0)
    step_dist = np.linalg.norm(deltas, axis=1)
    path_length = float(np.sum(step_dist))
    net_displacement = float(np.linalg.norm(points[-1] - points[0]))
    tortuosity = float(path_length / max(net_displacement, 1e-6))

    turn_changes: list[float] = []
    turn_indices: list[int] = []
    lookback = 20
    for idx in range(lookback, len(points) - lookback):
        before_vec = points[idx] - points[idx - lookback]
        after_vec = points[idx + lookback] - points[idx]
        before_dist = float(np.linalg.norm(before_vec))
        after_dist = float(np.linalg.norm(after_vec))
        if before_dist < 1.5 or after_dist < 1.5:
            continue
        before = (math.degrees(math.atan2(float(before_vec[0]), float(before_vec[1]))) + 360.0) % 360.0
        after = (math.degrees(math.atan2(float(after_vec[0]), float(after_vec[1]))) + 360.0) % 360.0
        change = angle_delta_deg(float(before), float(after))
        if 20.0 <= change <= 140.0:
            turn_changes.append(change)
            turn_indices.append(idx)

    turn_events = 0
    if turn_indices:
        turn_events = 1
        previous = turn_indices[0]
        for idx in turn_indices[1:]:
            if idx - previous > 20:
                turn_events += 1
            previous = idx

    # Count revisits on a decimated path to avoid treating slow movement inside
    # the same grid cell as a loop.
    revisit_points = points[::20] if len(points) >= 20 else points
    grid = np.floor((revisit_points - revisit_points[0]) / 2.0).astype(np.int64)
    first_seen: dict[tuple[int, int], int] = {}
    revisits = 0
    for idx, cell_arr in enumerate(grid):
        cell = (int(cell_arr[0]), int(cell_arr[1]))
        previous = first_seen.get(cell)
        if previous is None:
            first_seen[cell] = idx
        elif idx - previous > 2:
            revisits += 1
    revisit_rate = float(revisits / max(len(revisit_points), 1))

    speed_max = None
    speed_p95 = None
    if speeds is not None and len(speeds):
        finite = np.asarray(speeds, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            speed_max = float(np.max(finite))
            speed_p95 = float(np.percentile(finite, 95))

    return {
        "path_length_m": round(path_length, 3),
        "net_displacement_m": round(net_displacement, 3),
        "tortuosity": round(tortuosity, 4),
        "turn_count": int(turn_events),
        "turn_window_count": int(len(turn_changes)),
        "turn_changes_deg": [round(float(v), 3) for v in turn_changes],
        "revisit_rate": round(revisit_rate, 6),
        "speed_max_mps": round(speed_max, 3) if speed_max is not None else None,
        "speed_p95_mps": round(speed_p95, 3) if speed_p95 is not None else None,
    }


def valid_video(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def load_optional_array(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_window_clip(src_clip: Path, dst_clip: Path, start: int, end: int, metrics: dict[str, Any]) -> None:
    dst_clip.mkdir(parents=True, exist_ok=True)
    link_or_copy((src_clip / "front.mp4").resolve(), dst_clip / "front.mp4")
    for name in ("camera_params.json", "frame_orientation_order.json"):
        src = src_clip / name
        if src.is_file():
            shutil.copy2(src, dst_clip / name)

    array_names = [
        "frame_positions.npy",
        "frame_orientations.npy",
        "frame_times.npy",
        "frame_speeds.npy",
        "frame_video_indices.npy",
        "frame_gps.npy",
    ]
    for name in array_names:
        src = src_clip / name
        if not src.is_file():
            continue
        arr = np.load(src)
        sliced = arr[start:end].copy()
        np.save(dst_clip / name, sliced)

    source_report = src_clip / "conversion_report.json"
    source_meta: dict[str, Any] = {}
    if source_report.is_file():
        try:
            source_meta = json.loads(source_report.read_text(encoding="utf-8"))
        except Exception:
            source_meta = {}
    report = {
        "schema_version": "frodobots_structured_window_clip_v1",
        "source_clip_dir": str(src_clip),
        "source_conversion_report": str(source_report) if source_report.is_file() else None,
        "front_video_fps": source_meta.get("front_video_fps"),
        "frame_orientations_order": source_meta.get("frame_orientations_order"),
        "fps": source_meta.get("fps"),
        "window_start_frame": int(start),
        "window_end_frame": int(end),
        "frame_count": int(end - start),
        **metrics,
    }
    (dst_clip / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (dst_clip / "mcap_extract_meta.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_windows(frame_count: int, window_frames: int, stride_frames: int) -> range:
    if frame_count < window_frames:
        return range(0)
    return range(0, frame_count - window_frames + 1, max(1, stride_frames))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select structured FrodoBots windows")
    parser.add_argument("--sampling-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--window-sec", type=float, default=45.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--min-path-length-m", type=float, default=10.0)
    parser.add_argument("--max-path-length-m", type=float, default=80.0)
    parser.add_argument("--min-net-displacement-m", type=float, default=8.0)
    parser.add_argument("--max-tortuosity", type=float, default=3.0)
    parser.add_argument("--min-turn-count", type=int, default=1)
    parser.add_argument("--max-turn-count", type=int, default=3)
    parser.add_argument("--max-revisit-rate", type=float, default=0.05)
    parser.add_argument("--max-speed-mps", type=float, default=4.0)
    parser.add_argument("--max-speed-p95-mps", type=float, default=2.5)
    parser.add_argument("--max-windows", type=int, default=300)
    parser.add_argument("--max-windows-per-source", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampling_report = Path(args.sampling_report)
    output_root = Path(args.output_root)
    clips_root = output_root / "clips_structured"
    output_root.mkdir(parents=True, exist_ok=True)
    clips_root.mkdir(parents=True, exist_ok=True)

    report = json.loads(sampling_report.read_text(encoding="utf-8"))
    window_frames = int(round(float(args.window_sec) * float(args.fps)))
    stride_frames = int(round(float(args.stride_sec) * float(args.fps)))

    candidates: list[dict[str, Any]] = []
    rejects: dict[str, int] = {}
    for item in report.get("results", []):
        if not isinstance(item, dict) or str(item.get("status")) not in {"done", "skipped_existing", "exists"}:
            continue
        src_clip = Path(str(item.get("clip_dir") or ""))
        video = src_clip / "front.mp4"
        if not valid_video(video):
            rejects["bad_video"] = rejects.get("bad_video", 0) + 1
            continue
        positions = load_optional_array(src_clip / "frame_positions.npy")
        if positions is None or len(positions) < window_frames:
            rejects["too_short"] = rejects.get("too_short", 0) + 1
            continue
        speeds = load_optional_array(src_clip / "frame_speeds.npy")
        source_candidates = []
        for start in iter_windows(len(positions), window_frames, stride_frames):
            end = start + window_frames
            metrics = path_metrics(positions[start:end], None if speeds is None else speeds[start:end])
            if not metrics:
                continue
            reason = None
            if metrics["path_length_m"] < float(args.min_path_length_m):
                reason = "path_too_short"
            elif metrics["path_length_m"] > float(args.max_path_length_m):
                reason = "path_too_long"
            elif metrics["net_displacement_m"] < float(args.min_net_displacement_m):
                reason = "net_displacement_too_small"
            elif metrics["tortuosity"] > float(args.max_tortuosity):
                reason = "tortuosity_too_high"
            elif not (int(args.min_turn_count) <= int(metrics["turn_count"]) <= int(args.max_turn_count)):
                reason = "turn_count_out_of_range"
            elif metrics["revisit_rate"] > float(args.max_revisit_rate):
                reason = "revisit_rate_too_high"
            elif metrics.get("speed_max_mps") is not None and float(metrics["speed_max_mps"]) > float(args.max_speed_mps):
                reason = "speed_max_too_high"
            elif metrics.get("speed_p95_mps") is not None and float(metrics["speed_p95_mps"]) > float(args.max_speed_p95_mps):
                reason = "speed_p95_too_high"
            if reason:
                rejects[reason] = rejects.get(reason, 0) + 1
                continue
            turn_bonus = 6.0 * min(float(metrics["turn_count"]), 5.0)
            turn_density_bonus = 0.02 * min(float(metrics.get("turn_window_count") or 0.0), 120.0)
            score = (
                0.35 * float(metrics["net_displacement_m"])
                + 0.15 * float(metrics["path_length_m"])
                + turn_bonus
                + turn_density_bonus
                - 5.0 * max(0.0, float(metrics["tortuosity"]) - 1.0)
                - 30.0 * float(metrics["revisit_rate"])
            )
            source_candidates.append(
                {
                    "source_clip_dir": str(src_clip),
                    "sequence_name": item.get("sequence_name") or src_clip.name,
                    "source_frame_count": int(item.get("frame_count") or len(positions)),
                    "window_start": int(start),
                    "window_end": int(end),
                    "window_sec": float(args.window_sec),
                    "score": round(float(score), 4),
                    **metrics,
                }
            )
        source_candidates.sort(key=lambda row: float(row["score"]), reverse=True)
        candidates.extend(source_candidates[: max(1, int(args.max_windows_per_source))])

    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = candidates[: max(0, int(args.max_windows))]
    for idx, row in enumerate(selected):
        src_clip = Path(str(row["source_clip_dir"]))
        dst_name = f"{src_clip.name}_win{int(row['window_start']):06d}_{int(row['window_end']):06d}"
        dst_clip = clips_root / dst_name
        write_window_clip(src_clip, dst_clip, int(row["window_start"]), int(row["window_end"]), row)
        row["clip_dir"] = str(dst_clip)
        row["window_clip_id"] = dst_name
        row["rank"] = idx

    out_report = {
        "schema_version": "frodobots_structured_windows_v1",
        "sampling_report": str(sampling_report),
        "output_root": str(output_root),
        "clips_root": str(clips_root),
        "window_sec": float(args.window_sec),
        "stride_sec": float(args.stride_sec),
        "fps": float(args.fps),
        "criteria": {
            "path_length_m": [float(args.min_path_length_m), float(args.max_path_length_m)],
            "min_net_displacement_m": float(args.min_net_displacement_m),
            "max_tortuosity": float(args.max_tortuosity),
            "turn_count": [int(args.min_turn_count), int(args.max_turn_count)],
            "max_revisit_rate": float(args.max_revisit_rate),
            "max_speed_mps": float(args.max_speed_mps),
            "max_speed_p95_mps": float(args.max_speed_p95_mps),
        },
        "candidate_windows": len(candidates),
        "selected_windows": len(selected),
        "reject_counts": dict(sorted(rejects.items())),
        "selected": selected,
    }
    report_path = output_root / "structured_windows_report.json"
    report_path.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out_report.items() if k != "selected"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
