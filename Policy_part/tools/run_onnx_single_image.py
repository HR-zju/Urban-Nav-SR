#!/usr/bin/env python3
"""Single-image ONNX runner.

Repeatedly run the model on the same front image, overlay predicted
waypoints, display in real-time, and print the first 10 waypoint
predictions.

Usage example:

        python tools/run_onnx_single_image.py \
        --img_dir ./test_images_frame \
        --onnx onnx_ckpts/s1_ours_coco_sft_stage2_goal_less.onnx \
        --frames 21 \
        --device cuda

This script mirrors the preprocessing / postprocessing used in
``run_onnx_from_mp4.py``, so results should be consistent with the
video pipeline.
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

# Reuse helpers from the video runner to keep behavior consistent
from run_onnx_from_mp4 import (
    project_waypoints_to_fisheye_image_with_polygon_new,
    convert_fisheye_to_pinhole,
    trajectory_nms,
    choose_providers,
)

from Policy_part.tools.pd_controller_ros2_new2_agilex import pd_controller


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--img",
        required=False,
        default=None,
        help=(
            "Path to input image (front camera); "
            "optional if --img_dir is used"
        ),
    )
    p.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help=(
            "Directory containing a sequence of images; when set, the last "
            "T (= --frames) images are used as the temporal input"
        ),
    )
    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument(
        "--frames",
        type=int,
        default=21,
        help="Temporal length T used by model (default: 21)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Device for preprocessing tensors (cuda or cpu)",
    )
    p.add_argument(
        "--fov_x",
        type=float,
        default=None,
        help="Optional fov_x for fisheye->pinhole conversion (deg)",
    )
    p.add_argument(
        "--fov_y",
        type=float,
        default=None,
        help="Optional fov_y for fisheye->pinhole conversion (deg)",
    )
    p.add_argument(
        "--intrinsic_f",
        type=float,
        default=790.0,
        help="Focal length f for intrinsic (default 790)",
    )
    p.add_argument(
        "--loop_fps",
        type=int,
        default=5,
        help="Overlay refresh FPS for display loop (default 5Hz)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Require at least one of --img or --img_dir
    if args.img is None and args.img_dir is None:
        raise RuntimeError("You must provide either --img or --img_dir")

    providers = choose_providers(args.device == "cuda")
    print(f"Using ONNX providers: {providers}")
    session = ort.InferenceSession(args.onnx, providers=providers)

    # Intrinsics consistent with repo defaults (same as run_onnx_from_mp4.py)
    # intrinsic for projection: f, cx, cy, k
    intrinsic_params = [args.intrinsic_f, 1920.0 / 2.0, 1080.0 / 2.0, 0.0]

    # K and dist used by convert_fisheye_to_pinhole
    # (scaled for 512x288 pinhole)
    K = np.array(
        [
            [790 * 512 / 1920, 0.0, 512 / 2.0],
            [0.0, 790 * 288 / 1080, 288 / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist = np.array(
        [-0.03054249, 0.00974967, -0.00679687, 0.00238449, 0.0],
        dtype=np.float32,
    )

    # Device for preprocessing
    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device == "cuda"
        else "cpu"
    )

    # Build temporal stack either from a directory of images (preferred)
    # or by repeating a single image, for backward compatibility.
    T = int(args.frames)

    img_rgb_last = None

    if hasattr(args, "img_dir") and args.img_dir is not None:
        # Folder mode: read a sequence of images as the T frames
        img_dir = args.img_dir
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_files = [
            f
            for f in sorted(os.listdir(img_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if len(all_files) < T:
            raise RuntimeError(
                f"Not enough images in {img_dir}: found {len(all_files)}, "
                f"but need at least {T}."
            )

        selected_files = all_files[-T:]
        frames = []
        for fname in selected_files:
            path = os.path.join(img_dir, fname)
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to read image: {path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h0, w0 = img_rgb.shape[:2]
            if (w0, h0) != (1920, 1080):
                img_rgb = cv2.resize(
                    img_rgb,
                    (1920, 1080),
                    interpolation=cv2.INTER_LINEAR,
                )
            frame_t = (
                torch.from_numpy(img_rgb)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
            frames.append(frame_t)
            img_rgb_last = img_rgb

        frames_t = torch.stack(frames, dim=0).to(device)  # (T, C, H, W)
        video_frames_small = torch.nn.functional.interpolate(
            frames_t,
            size=(288, 512),
            mode="bilinear",
            align_corners=False,
        )
    else:
        # Single-image mode: keep previous behavior (repeat the same
        # frame T times).
        img_bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {args.img}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]
        if (w0, h0) != (1920, 1080):
            img_rgb = cv2.resize(
                img_rgb,
                (1920, 1080),
                interpolation=cv2.INTER_LINEAR,
            )

        frame_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        frame_t = frame_t.to(device)

        frame_small = torch.nn.functional.interpolate(
            frame_t.unsqueeze(0),  # (1, C, H, W)
            size=(288, 512),
            mode="bilinear",
            align_corners=False,
        )  # -> (1, C, 288, 512)

        video_frames_small = frame_small.repeat(T, 1, 1, 1)  # (T, C, H, W)
        img_rgb_last = img_rgb

    # prepare metric_spacing and embodiment_id as in video script
    metric_spacing = np.array(
        [[0.0, 0.51, -0.26, 0.26]],
        dtype=np.float32,
    ).reshape(1, 4)
    embodiment_id = np.array([[0]], dtype=np.int64)

    # fisheye -> pinhole
    fov_x = (
        args.fov_x
        if args.fov_x is not None
        else math.degrees(2 * math.atan(1920 / (2 * 790)))
    )
    fov_y = (
        args.fov_y
        if args.fov_y is not None
        else math.degrees(2 * math.atan(1080 / (2 * 790)))
    )

    pinhole = convert_fisheye_to_pinhole(
        K,
        dist,
        device,
        video_frames_small,
        fov_x=fov_x,
        fov_y=fov_y,
    )

    # onnx expects (B, T, C, H, W) with B=1, T=T
    observation = pinhole.unsqueeze(0).cpu().numpy().astype(np.float32)

    print("Starting repeated inference on image sequence. Press 'q' to quit.")

    best_infer_fps = 0.0
    printed_count = 0
    current_v = 0.0
    current_w = 0.0
    # output txt for first 10 waypoint predictions
    if hasattr(args, "img_dir") and args.img_dir is not None:
        img_base = os.path.basename(os.path.normpath(args.img_dir))
    else:
        img_base = os.path.splitext(os.path.basename(args.img))[0]
    out_txt = f"logs/{img_base}_waypoints.txt"
    print(f"Will save first 10 waypoint predictions to: {out_txt}")

    with open(out_txt, "w", encoding="utf-8") as f:
        while True:
            # pure ONNX inference timing
            t0 = time.perf_counter()
            outputs = session.run(
                None,
                {
                    "observation": observation,
                    "embodiment_id": embodiment_id,
                    "metric_spacing": metric_spacing,
                },
            )
            t1 = time.perf_counter()
            infer_dt = max(t1 - t0, 1e-8)
            infer_fps = 1.0 / infer_dt
            best_infer_fps = max(best_infer_fps, infer_fps)
            print(
                f"[perf] onnx inference: {infer_dt*1000:.2f} ms, "
                f"fps={infer_fps:.2f}, best_fps={best_infer_fps:.2f}"
            )

            # parse outputs (same logic as in run_onnx_from_mp4.py)
            traj_raw = outputs[0][:, 1:]
            score_raw = outputs[1][:, 1:]
            traj_t = torch.from_numpy(traj_raw)
            score_t = torch.from_numpy(score_raw)
            traj_sel, score_sel = trajectory_nms(traj_t, score_t)
            traj_sel = traj_sel[0]
            score_sel = score_sel[0]
            score_sel = torch.softmax(score_sel, dim=0)
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
            wp_np = wp_pred.detach()[0, :, :2].cpu().numpy()

            # run PD controller and print/save first 10 waypoint predictions
            if printed_count < 10:
                v_cmd, w_cmd = pd_controller(wp_np, current_v, current_w)
                current_v = v_cmd
                current_w = w_cmd

                header = (
                    "================ Waypoints prediction #{} "
                    "================".format(printed_count + 1)
                )
                print(header)
                print(wp_np)
                print(
                    f"[pd_controller] v={v_cmd:.4f} m/s, "
                    f"w={w_cmd:.4f} rad/s",
                )

                f.write(header + "\n")
                np.savetxt(f, wp_np, fmt="%.6f")
                f.write(
                    "v {:.6f} w {:.6f}\n\n".format(
                        float(v_cmd), float(w_cmd)
                    )
                )

                printed_count += 1

            # overlay on original fisheye image (use RGB image but
            # function handles color)
            try:
                vis_img = project_waypoints_to_fisheye_image_with_polygon_new(
                    wp_np,
                    intrinsic_params,
                    img_rgb_last,
                    save_fig=False,
                )
            except Exception as e:
                print(f"[overlay] draw error: {e}")
                vis_img = cv2.cvtColor(img_rgb_last, cv2.COLOR_RGB2BGR)

            # ensure BGR for display
            if vis_img.shape[-1] == 3:
                vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            else:
                vis_bgr = vis_img

            cv2.imshow("single_image_overlay", vis_bgr)

            # control display loop frequency
            if (
                cv2.waitKey(int(1000 / max(1, args.loop_fps))) & 0xFF
                == ord("q")
            ):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
