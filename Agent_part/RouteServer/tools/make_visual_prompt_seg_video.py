#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameItem:
    frame_name: str
    image_path: Path
    json_path: Path
    seg_vis_path: Path | None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return int(default)


def _load_manifest_frames(run_dir: Path) -> list[FrameItem]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"manifest.json not found under: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames") or []
    if not isinstance(frames, list):
        raise RuntimeError("manifest.json frames must be a list")

    seg_vis_dir = run_dir / "seg_vis"
    out: list[FrameItem] = []
    for item in frames:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ok":
            continue
        frame_name = str(item.get("frame_name") or "").strip()
        if not frame_name:
            continue
        image_path = Path(str(item.get("image_path") or "")).expanduser()
        json_path = Path(str(item.get("json_path") or "")).expanduser()
        seg_vis = seg_vis_dir / f"{frame_name}_visualization.jpg"
        out.append(
            FrameItem(
                frame_name=frame_name,
                image_path=image_path,
                json_path=json_path,
                seg_vis_path=seg_vis if seg_vis.exists() else None,
            )
        )
    if not out:
        raise RuntimeError(f"No ok frames found in manifest: {manifest_path}")
    return out


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return img


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    scale = float(height) / float(h)
    new_w = max(int(round(w * scale)), 1)
    return cv2.resize(img, (new_w, int(height)), interpolation=cv2.INTER_AREA)


def _draw_polyline(img: np.ndarray, pts: list[tuple[int, int]], color_bgr: tuple[int, int, int], thickness: int) -> None:
    if len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [arr], isClosed=False, color=color_bgr, thickness=int(thickness), lineType=cv2.LINE_AA)


def _draw_prompt_overlay(
    *,
    rgb_bgr: np.ndarray,
    prompt: dict[str, Any] | None,
    alpha: float = 0.6,
) -> np.ndarray:
    if not isinstance(prompt, dict):
        return rgb_bgr

    overlay = rgb_bgr.copy()
    prompt_type = str(prompt.get("type") or "")
    if prompt_type != "path_corridor":
        return rgb_bgr

    samples = prompt.get("path_samples") if isinstance(prompt.get("path_samples"), list) else []
    centers: list[tuple[int, int]] = []
    lefts: list[tuple[int, int]] = []
    rights: list[tuple[int, int]] = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        c = s.get("center_uv")
        l = s.get("left_edge_uv")
        r = s.get("right_edge_uv")
        if isinstance(c, list) and len(c) == 2:
            centers.append((_safe_int(c[0]), _safe_int(c[1])))
        if isinstance(l, list) and len(l) == 2:
            lefts.append((_safe_int(l[0]), _safe_int(l[1])))
        if isinstance(r, list) and len(r) == 2:
            rights.append((_safe_int(r[0]), _safe_int(r[1])))

    # Corridor fill (best-effort).
    if len(lefts) == len(rights) and len(lefts) >= 2:
        poly = lefts + list(reversed(rights))
        cv2.fillPoly(overlay, [np.array(poly, dtype=np.int32).reshape((-1, 1, 2))], color=(0, 255, 255))

    # Center / edges.
    _draw_polyline(overlay, centers, color_bgr=(0, 255, 255), thickness=3)   # yellow
    _draw_polyline(overlay, lefts, color_bgr=(255, 255, 0), thickness=2)     # cyan
    _draw_polyline(overlay, rights, color_bgr=(255, 0, 255), thickness=2)    # magenta

    for pt in centers:
        cv2.circle(overlay, pt, radius=4, color=(0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(overlay, float(alpha), rgb_bgr, float(1.0 - alpha), 0.0)


def _prompt_for_frame(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    arb = payload.get("arbiter_decision") if isinstance(payload.get("arbiter_decision"), dict) else {}
    dbg = arb.get("debug") if isinstance((arb or {}).get("debug"), dict) else {}
    # Prefer pre-constraint prompt when available.
    rejected = dbg.get("rejected_rendered_prompt") if isinstance(dbg.get("rejected_rendered_prompt"), dict) else None
    if rejected:
        return rejected, "pre_constraint(rejected_rendered_prompt)"
    rendered = arb.get("rendered_prompt") if isinstance(arb.get("rendered_prompt"), dict) else None
    if rendered:
        return rendered, "final(rendered_prompt)"
    return None, "none"


def _compose_frame(
    *,
    rgb_bgr: np.ndarray,
    seg_vis_bgr: np.ndarray | None,
    frame_name: str,
    prompt_source: str,
) -> np.ndarray:
    left = rgb_bgr
    right = seg_vis_bgr if seg_vis_bgr is not None else np.zeros_like(rgb_bgr)
    right = _resize_to_height(right, left.shape[0])
    left = _resize_to_height(left, right.shape[0])
    combined = np.hstack([left, right])

    # Header bar
    bar_h = 34
    bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)
    cv2.putText(
        bar,
        f"frame={frame_name}  prompt={prompt_source}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, combined])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose pre-constraint visual prompt overlay with SAM3 visualization and export an mp4 video."
    )
    parser.add_argument("--run-dir", required=True, help="Output directory from run_*_visual_prompt_pipeline (contains manifest.json)")
    parser.add_argument("--output-video", required=True, help="Output mp4 path")
    parser.add_argument("--fps", type=float, default=5.0, help="Video FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    frames = _load_manifest_frames(run_dir)
    if args.max_frames is not None:
        frames = frames[: max(int(args.max_frames), 0)]

    first_payload = json.loads(frames[0].json_path.read_text(encoding="utf-8"))
    first_rgb = _read_bgr(frames[0].image_path)
    prompt, prompt_source = _prompt_for_frame(first_payload)
    left = _draw_prompt_overlay(rgb_bgr=first_rgb, prompt=prompt)
    seg_vis = _read_bgr(frames[0].seg_vis_path) if frames[0].seg_vis_path else None
    first_frame = _compose_frame(
        rgb_bgr=left,
        seg_vis_bgr=seg_vis,
        frame_name=frames[0].frame_name,
        prompt_source=prompt_source,
    )

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, float(args.fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_video}")

    try:
        writer.write(first_frame)
        for item in frames[1:]:
            payload = json.loads(item.json_path.read_text(encoding="utf-8"))
            rgb = _read_bgr(item.image_path)
            prompt, prompt_source = _prompt_for_frame(payload)
            left = _draw_prompt_overlay(rgb_bgr=rgb, prompt=prompt)
            seg_vis = _read_bgr(item.seg_vis_path) if item.seg_vis_path else None
            frame = _compose_frame(
                rgb_bgr=left,
                seg_vis_bgr=seg_vis,
                frame_name=item.frame_name,
                prompt_source=prompt_source,
            )
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    print(f"[OK] Wrote video: {output_video}")


if __name__ == "__main__":
    main()

