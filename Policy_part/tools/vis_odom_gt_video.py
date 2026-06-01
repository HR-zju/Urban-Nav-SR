#!/usr/bin/env python3
"""Render GT odometry trajectories on dataset videos.

The vint fine-tune data format stores every source video as multiple clip
directories. Each clip has:

  - frame_data.npy
  - front_pinhole_crop_resize.mp4

This script groups clips by the directory name before ``__clip_XXXX`` and
writes one overlay video per source video.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


CLIP_RE = re.compile(r"^(?P<base>.+)__clip_(?P<clip>\d+)$")


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    base: str
    clip_index: int


def load_projection_function() -> Callable[..., np.ndarray]:
    vis_path = Path(__file__).resolve().parent / "vis_gt_single_gt.py"
    func_name = "project_waypoints_to_fisheye_image_with_polygon_new"
    source = vis_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(vis_path))
    nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == func_name:
            nodes.append(node)
            break
    else:
        raise ImportError(f"Function {func_name} not found in {vis_path}")

    module = ast.Module(body=nodes, type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(module), str(vis_path), "exec"), namespace)
    return namespace[func_name]  # type: ignore[return-value]


def discover_clips(data_root: Path, include: str | None) -> dict[str, list[ClipInfo]]:
    groups: dict[str, list[ClipInfo]] = {}
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        match = CLIP_RE.match(child.name)
        if not match:
            continue
        base = match.group("base")
        if include and include not in child.name and include not in base:
            continue
        if not (child / "frame_data.npy").exists():
            continue
        groups.setdefault(base, []).append(
            ClipInfo(path=child, base=base, clip_index=int(match.group("clip")))
        )

    for clips in groups.values():
        clips.sort(key=lambda item: item.clip_index)
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def load_clip_positions(clip: ClipInfo) -> np.ndarray:
    data = np.load(clip.path / "frame_data.npy", allow_pickle=False)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected frame_data.npy with shape (N, >=2), got {data.shape}")
    xy = np.asarray(data[:, :2], dtype=np.float32)
    finite = np.isfinite(xy).all(axis=1)
    if not finite.all():
        xy = xy.copy()
        bad = np.where(~finite)[0]
        good = np.where(finite)[0]
        if good.size == 0:
            raise ValueError(f"No finite xy values in {clip.path / 'frame_data.npy'}")
        nearest_good = good[np.searchsorted(good, bad, side="left").clip(0, good.size - 1)]
        xy[bad] = xy[nearest_good]
    return xy


def derive_headings(xy: np.ndarray, smooth_window: int) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(xy) == 1:
        return np.zeros((1,), dtype=np.float32)

    window = max(1, int(smooth_window))
    headings = np.zeros((len(xy),), dtype=np.float32)
    valid_heading = 0.0
    for i in range(len(xy)):
        left = max(0, i - window)
        right = min(len(xy) - 1, i + window)
        delta = xy[right] - xy[left]
        if float(np.linalg.norm(delta)) > 1e-4:
            valid_heading = math.atan2(float(delta[1]), float(delta[0]))
        headings[i] = valid_heading
    return headings


def local_future_waypoints(
    xy: np.ndarray,
    headings: np.ndarray,
    frame_index: int,
    future_frames: int,
    stride: int,
    max_forward_m: float,
) -> np.ndarray:
    start = int(frame_index)
    end = min(len(xy), start + max(2, int(future_frames)))
    if end - start < 2:
        return np.zeros((0, 2), dtype=np.float32)

    indices = np.arange(start, end, max(1, int(stride)), dtype=np.int64)
    if indices[-1] != end - 1:
        indices = np.append(indices, end - 1)

    delta = xy[indices] - xy[start]
    theta = float(headings[start])
    c = math.cos(theta)
    s = math.sin(theta)

    # World to robot frame: +x forward, +y left.
    forward = c * delta[:, 0] + s * delta[:, 1]
    left = -s * delta[:, 0] + c * delta[:, 1]
    waypoints = np.column_stack([forward, left]).astype(np.float32)
    waypoints = waypoints[waypoints[:, 0] >= -0.05]
    if waypoints.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    waypoints[:, 0] = np.clip(waypoints[:, 0], 0.0, float(max_forward_m))
    if waypoints.shape[0] > 1:
        keep = np.ones((waypoints.shape[0],), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) > 1e-4
        waypoints = waypoints[keep]
    if waypoints.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return waypoints


def make_intrinsic(width: int, height: int, base_f: float, base_width: float) -> list[float]:
    scale = float(width) / float(base_width)
    return [float(base_f) * scale, float(width) / 2.0, float(height) / 2.0, 0.0]


def open_writer(out_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")
    return writer


def probe_video(video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read first frame from: {video_path}")
            height, width = frame.shape[:2]
        return float(fps), frames, width, height
    finally:
        cap.release()


def render_group(
    base: str,
    clips: list[ClipInfo],
    args: argparse.Namespace,
    project_func: Callable[..., np.ndarray],
) -> Path:
    clip_positions = [load_clip_positions(clip) for clip in clips]
    all_xy = np.concatenate(clip_positions, axis=0)
    headings = derive_headings(all_xy, args.heading_smooth_window)

    first_video = clips[0].path / args.video_name
    fps, _frames, width, height = probe_video(first_video)
    if args.fps is not None:
        fps = float(args.fps)
    intrinsic = make_intrinsic(width, height, args.intrinsic_f, args.intrinsic_base_width)

    out_path = Path(args.output_dir) / f"{base}__gt_odom_overlay.mp4"
    writer = open_writer(out_path, fps, width, height)

    global_frame_index = 0
    total_written = 0
    try:
        for clip, xy in zip(clips, clip_positions):
            video_path = clip.path / args.video_name
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            local_frame = 0
            try:
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    if local_frame >= len(xy):
                        break
                    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

                    waypoints = local_future_waypoints(
                        all_xy,
                        headings,
                        global_frame_index,
                        args.future_frames,
                        args.stride,
                        args.max_forward_m,
                    )
                    if waypoints.shape[0] >= 2:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        frame_bgr = project_func(
                            waypoints,
                            intrinsic,
                            frame_rgb,
                            save_fig=False,
                            camera_height=args.camera_height,
                            base_alpha=args.alpha,
                        )

                    if args.draw_status:
                        label = f"clip {clip.clip_index:04d} frame {local_frame:04d}"
                        cv2.putText(
                            frame_bgr,
                            label,
                            (8, max(18, height - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    writer.write(frame_bgr)
                    local_frame += 1
                    global_frame_index += 1
                    total_written += 1
            finally:
                cap.release()

            if local_frame < len(xy):
                print(
                    f"[warn] {clip.path.name}: video ended at {local_frame} frames, "
                    f"odom has {len(xy)} frames"
                )
    finally:
        writer.release()

    print(
        f"[done] {base}: clips={len(clips)} frames={total_written} "
        f"fps={fps:.3f} out={out_path}"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay future GT odometry trajectories on data_format videos."
    )
    parser.add_argument("--data-root", default="data_format", help="Dataset root symlink/path")
    parser.add_argument(
        "--output-dir",
        default="Policy_part/vis/gt_odom_overlay",
        help="Directory for merged overlay videos",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Only process groups whose clip/base name contains this substring",
    )
    parser.add_argument(
        "--video-name",
        default="front_pinhole_crop_resize.mp4",
        help="Video file inside every clip directory",
    )
    parser.add_argument("--future-frames", type=int, default=120, help="Future odom horizon in frames")
    parser.add_argument("--stride", type=int, default=5, help="Waypoint sampling stride in frames")
    parser.add_argument("--max-forward-m", type=float, default=20.0, help="Clamp forward distance")
    parser.add_argument("--heading-smooth-window", type=int, default=8, help="Frames used to infer heading")
    parser.add_argument("--camera-height", type=float, default=0.41)
    parser.add_argument("--intrinsic-f", type=float, default=790.0)
    parser.add_argument("--intrinsic-base-width", type=float, default=1920.0)
    parser.add_argument("--alpha", type=float, default=0.15, help="Base overlay alpha")
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS")
    parser.add_argument("--max-groups", type=int, default=None, help="Debug limit")
    parser.add_argument("--dry-run", action="store_true", help="List grouped clips without rendering")
    parser.add_argument("--draw-status", action="store_true", help="Draw clip/frame index text")
    return parser.parse_args()


def iter_limited_groups(
    groups: dict[str, list[ClipInfo]], max_groups: int | None
) -> Iterable[tuple[str, list[ClipInfo]]]:
    count = 0
    for base, clips in groups.items():
        if max_groups is not None and count >= max_groups:
            break
        count += 1
        yield base, clips


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    groups = discover_clips(data_root, args.include)
    if not groups:
        raise RuntimeError(f"No clip groups found under {data_root}")

    print(f"[info] found {len(groups)} video groups under {data_root}")
    for base, clips in iter_limited_groups(groups, args.max_groups):
        clip_ids = ",".join(f"{clip.clip_index:04d}" for clip in clips)
        print(f"[group] {base}: {len(clips)} clips [{clip_ids}]")

    if args.dry_run:
        return

    project_func = load_projection_function()
    for base, clips in iter_limited_groups(groups, args.max_groups):
        render_group(base, clips, args, project_func)


if __name__ == "__main__":
    main()
