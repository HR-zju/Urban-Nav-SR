#!/usr/bin/env python3
"""Multi-stage FPS benchmark suite for ONNX navigation model.

One run benchmarks multiple incremental stages and writes all FPS numbers to txt:
1) pure_infer                  : only session.run with cached observation
2) infer_with_preprocess       : + image preprocess + fisheye->pinhole each iter
3) infer_with_postprocess      : + trajectory_nms + waypoint decode
4) infer_with_topic_pack       : + waypoint payload packing (publish-like formatting)
5) infer_with_overlay          : + waypoint drawing overlay

This is intended for quick A/B performance profiling on RTX 3090 and also
for deployment sanity checks.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch

from pd_controller_ros2_new2 import pd_controller
from run_onnx_from_mp4 import (
    choose_providers,
    convert_fisheye_to_pinhole,
    project_waypoints_to_fisheye_image_with_polygon_new,
    trajectory_nms,
)


@dataclass
class BenchContext:
    session: ort.InferenceSession
    device: torch.device
    T: int
    intrinsic_params: List[float]
    K: np.ndarray
    dist: np.ndarray
    fov_x: float
    fov_y: float
    metric_spacing: np.ndarray
    embodiment_id: np.ndarray
    # Cached fast-path observation (B, T, C, H, W)
    cached_observation: np.ndarray
    # Source RGB frames, each 1920x1080
    rgb_frames: List[np.ndarray]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-stage ONNX FPS benchmark suite")
    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument("--img", default=None, help="Single image path")
    p.add_argument("--img_dir", default=None, help="Image folder path")
    p.add_argument("--frames", type=int, default=21, help="Temporal length T")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--repeats", type=int, default=10, help="Repeat count per stage")
    p.add_argument("--iters", type=int, default=20, help="Iterations per repeat")
    p.add_argument("--warmup_iters", type=int, default=3, help="Warmup iters per repeat")
    p.add_argument("--intrinsic_f", type=float, default=790.0, help="Projection focal")
    p.add_argument("--fov_x", type=float, default=None, help="Optional fov_x override")
    p.add_argument("--fov_y", type=float, default=None, help="Optional fov_y override")
    p.add_argument(
        "--txt",
        type=str,
        default=None,
        help="Output txt path (default: auto timestamped under logs/)",
    )
    return p.parse_args()


def _load_rgb_1920x1080(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if (w, h) != (1920, 1080):
        img_rgb = cv2.resize(img_rgb, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    return img_rgb


def load_frames(args: argparse.Namespace) -> List[np.ndarray]:
    T = int(args.frames)
    if args.img_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [
            f
            for f in sorted(os.listdir(args.img_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if len(files) < T:
            raise RuntimeError(
                f"Not enough images in {args.img_dir}: found {len(files)}, need >= {T}"
            )
        chosen = files[-T:]
        return [_load_rgb_1920x1080(os.path.join(args.img_dir, f)) for f in chosen]

    if args.img:
        one = _load_rgb_1920x1080(args.img)
        return [one.copy() for _ in range(T)]

    raise RuntimeError("You must provide --img or --img_dir")


def build_observation_from_frames(
    rgb_frames: List[np.ndarray],
    device: torch.device,
    K: np.ndarray,
    dist: np.ndarray,
    fov_x: float,
    fov_y: float,
) -> np.ndarray:
    frame_ts = []
    for rgb in rgb_frames:
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        frame_ts.append(t)
    frames_t = torch.stack(frame_ts, dim=0).to(device)  # (T, C, H, W)

    small = torch.nn.functional.interpolate(
        frames_t,
        size=(288, 512),
        mode="bilinear",
        align_corners=False,
    )

    pinhole = convert_fisheye_to_pinhole(
        K,
        dist,
        device,
        small,
        fov_x=fov_x,
        fov_y=fov_y,
    )

    # (T, C, H, W) -> (1, T, C, H, W)
    return pinhole.unsqueeze(0).cpu().numpy().astype(np.float32)


def run_inference(ctx: BenchContext, observation: np.ndarray):
    return ctx.session.run(
        None,
        {
            "observation": observation,
            "embodiment_id": ctx.embodiment_id,
            "metric_spacing": ctx.metric_spacing,
        },
    )


def decode_waypoints(outputs) -> np.ndarray:
    traj_raw = outputs[0][:, 1:]
    score_raw = outputs[1][:, 1:]
    traj_t = torch.from_numpy(traj_raw)
    score_t = torch.from_numpy(score_raw)

    traj_sel, score_sel = trajectory_nms(traj_t, score_t)
    traj_sel = traj_sel[0]
    score_sel = torch.softmax(score_sel[0], dim=0)
    traj_final = traj_sel[torch.argmax(score_sel)]

    wp_pred = traj_final[..., :2].unsqueeze(0)
    wp_pred = torch.cat(
        [
            torch.zeros_like(wp_pred[..., 0:1, :]),
            wp_pred[..., :-1, :],
        ],
        dim=-2,
    )
    wp_pred[..., 0] = wp_pred[..., 0].clamp(0.0, 20.0)
    return wp_pred.detach()[0, :, :2].cpu().numpy()


def pack_waypoint_topic_payload(wp_np: np.ndarray) -> List[float]:
    # Match navflow publish-like payload construction overhead (without ROS publish)
    wp_cam = np.stack([-wp_np[:, 1], wp_np[:, 0]], axis=1)
    payload: List[float] = []
    if wp_cam[-1][1] > 0.1:
        for i in range(0, len(wp_cam), 2):
            payload += [float(wp_cam[i][1]), float(-wp_cam[i][0])]
    else:
        for _ in range(0, len(wp_cam), 2):
            payload += [0.0, 0.0]
    return payload


def stage_pure_infer(ctx: BenchContext) -> None:
    _ = run_inference(ctx, ctx.cached_observation)


def stage_with_preprocess(ctx: BenchContext) -> None:
    obs = build_observation_from_frames(
        ctx.rgb_frames,
        ctx.device,
        ctx.K,
        ctx.dist,
        ctx.fov_x,
        ctx.fov_y,
    )
    _ = run_inference(ctx, obs)


def stage_with_postprocess(ctx: BenchContext) -> None:
    outputs = run_inference(ctx, ctx.cached_observation)
    _ = decode_waypoints(outputs)


def stage_with_topic_pack(ctx: BenchContext) -> None:
    outputs = run_inference(ctx, ctx.cached_observation)
    wp = decode_waypoints(outputs)
    payload = pack_waypoint_topic_payload(wp)
    # Extra tiny op to prevent potential dead-store simplification
    _ = len(payload)


def stage_with_overlay(ctx: BenchContext) -> None:
    outputs = run_inference(ctx, ctx.cached_observation)
    wp = decode_waypoints(outputs)

    # Include lightweight control op as optional module overhead
    _v, _w = pd_controller(wp, 0.0, 0.0)
    _ = (_v, _w)

    _ = project_waypoints_to_fisheye_image_with_polygon_new(
        wp,
        ctx.intrinsic_params,
        ctx.rgb_frames[-1],
        save_fig=False,
    )


def measure_stage(
    name: str,
    fn: Callable[[BenchContext], None],
    ctx: BenchContext,
    repeats: int,
    iters: int,
    warmup_iters: int,
) -> Dict[str, object]:
    repeat_fps: List[float] = []

    for _ in range(repeats):
        for _ in range(warmup_iters):
            fn(ctx)

        t0 = time.perf_counter()
        for _ in range(iters):
            fn(ctx)
        t1 = time.perf_counter()

        dt = max(t1 - t0, 1e-9)
        repeat_fps.append(iters / dt)

    arr = np.array(repeat_fps, dtype=np.float64)
    return {
        "name": name,
        "repeats": repeats,
        "iters": iters,
        "fps_values": repeat_fps,
        "fps_mean": float(arr.mean()),
        "fps_median": float(np.median(arr)),
        "fps_min": float(arr.min()),
        "fps_max": float(arr.max()),
        "fps_std": float(arr.std(ddof=0)),
    }


def make_output_txt_path(args: argparse.Namespace) -> str:
    if args.txt:
        return args.txt
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"fps_suite_{ts}.txt")


def main() -> None:
    args = parse_args()

    providers = choose_providers(args.device == "cuda")
    print(f"Using ONNX providers: {providers}")
    session = ort.InferenceSession(args.onnx, providers=providers)

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    rgb_frames = load_frames(args)

    fov_x = (
        args.fov_x
        if args.fov_x is not None
        else math.degrees(2 * math.atan(1920 / (2 * args.intrinsic_f)))
    )
    fov_y = (
        args.fov_y
        if args.fov_y is not None
        else math.degrees(2 * math.atan(1080 / (2 * args.intrinsic_f)))
    )

    K = np.array(
        [
            [args.intrinsic_f * 512 / 1920, 0.0, 512 / 2.0],
            [0.0, args.intrinsic_f * 288 / 1080, 288 / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist = np.array(
        [-0.03054249, 0.00974967, -0.00679687, 0.00238449, 0.0],
        dtype=np.float32,
    )

    cached_observation = build_observation_from_frames(
        rgb_frames,
        device,
        K,
        dist,
        fov_x,
        fov_y,
    )

    ctx = BenchContext(
        session=session,
        device=device,
        T=int(args.frames),
        intrinsic_params=[args.intrinsic_f, 1920.0 / 2.0, 1080.0 / 2.0, 0.0],
        K=K,
        dist=dist,
        fov_x=fov_x,
        fov_y=fov_y,
        metric_spacing=np.array([[0.0, 0.51, -0.26, 0.26]], dtype=np.float32),
        embodiment_id=np.array([[0]], dtype=np.int64),
        cached_observation=cached_observation,
        rgb_frames=rgb_frames,
    )

    stages: List[Tuple[str, Callable[[BenchContext], None]]] = [
        ("pure_infer", stage_pure_infer),
        ("infer_with_preprocess", stage_with_preprocess),
        ("infer_with_postprocess", stage_with_postprocess),
        ("infer_with_topic_pack", stage_with_topic_pack),
        ("infer_with_overlay", stage_with_overlay),
    ]

    results = []
    for stage_name, stage_fn in stages:
        print(f"\n[benchmark] running stage: {stage_name}")
        result = measure_stage(
            name=stage_name,
            fn=stage_fn,
            ctx=ctx,
            repeats=int(args.repeats),
            iters=int(args.iters),
            warmup_iters=int(args.warmup_iters),
        )
        results.append(result)
        print(
            f"  mean={result['fps_mean']:.2f} | median={result['fps_median']:.2f} "
            f"| min={result['fps_min']:.2f} | max={result['fps_max']:.2f} "
            f"| std={result['fps_std']:.2f}"
        )

    out_txt = make_output_txt_path(args)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("ONNX FPS Suite Result\n")
        f.write(f"time: {datetime.datetime.now().isoformat()}\n")
        f.write(f"onnx: {args.onnx}\n")
        f.write(f"device: {device}\n")
        f.write(f"providers: {session.get_providers()}\n")
        f.write(
            f"repeats={args.repeats}, iters={args.iters}, warmup_iters={args.warmup_iters}\n"
        )
        f.write("\n")

        for result in results:
            f.write(f"[stage] {result['name']}\n")
            f.write(
                "mean={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}, std={:.4f}\n".format(
                    result["fps_mean"],
                    result["fps_median"],
                    result["fps_min"],
                    result["fps_max"],
                    result["fps_std"],
                )
            )
            values = ", ".join(f"{x:.4f}" for x in result["fps_values"])
            f.write(f"values: {values}\n\n")

        if len(results) >= 2:
            base = results[0]["fps_mean"]
            f.write("relative_to_pure_infer:\n")
            for result in results:
                ratio = result["fps_mean"] / base if base > 1e-12 else 0.0
                f.write(f"  {result['name']}: {ratio:.4f}x\n")

    print(f"\nSaved detailed FPS report to: {out_txt}")


if __name__ == "__main__":
    main()
