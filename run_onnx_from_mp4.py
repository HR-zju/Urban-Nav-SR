#!/usr/bin/env python3
"""
Lightweight runner: read a front mp4, run ONNX model, overlay predicted waypoints, and display.

This script reuses helper code from the repository:
- `convert_fisheye_to_pinhole` and `trajectory_nms` from `test_nav_flow_il_go2.py`
- `project_waypoints_to_fisheye_image_with_polygon_new` from `vis_gt_single_gt.py`

Usage:
  python run_onnx_from_mp4.py --mp4 front/your.mp4 --onnx onnx_ckpts/s1_ours_coco_sft_stage2_goal_less.onnx

Note: requires onnxruntime, torch, numpy, opencv-python. GPU provider used if available.
"""
import argparse
import math
import os
from collections import deque
import numpy as np
import cv2
import onnxruntime as ort
import torch
import threading
import time

import ast

# Dynamically load only the visualization function from vis_gt_single_gt.py
def load_function_from_file(filepath, func_name):
    """Extract imports and a single function definition from a file and load it into a namespace.

    This avoids executing top-level demo code in the source file.
    """
    with open(filepath, 'r') as f:
        src = f.read()
    tree = ast.parse(src, filepath)

    new_nodes = []
    func_node = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            new_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_node = node
            new_nodes.append(node)
            break

    if func_node is None:
        raise ImportError(f"Function {func_name} not found in {filepath}")

    mod = ast.Module(body=new_nodes, type_ignores=[])
    code = compile(ast.fix_missing_locations(mod), filepath, 'exec')
    ns = {}
    exec(code, ns)
    return ns[func_name]


# load the vis function without running demo code
project_waypoints_to_fisheye_image_with_polygon_new = load_function_from_file(
    os.path.join(os.path.dirname(__file__), 'vis_gt_single_gt.py'),
    'project_waypoints_to_fisheye_image_with_polygon_new'
)

# import heavy helpers from test_nav_flow_il_go2 (this file is mostly defs so import is ok)
from test_nav_flow_il_go2 import convert_fisheye_to_pinhole, trajectory_nms


def draw_simple_traj(wp_xy: np.ndarray, size: int = 400) -> np.ndarray:
    """Draw a simple 2D top-down trajectory for debugging.

    wp_xy: (N, 2) array in robot frame, x forward, y left (meters).
    Returns a BGR image showing the path.
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # origin at bottom center
    origin_x = size // 2
    origin_y = size - 20
    origin = (origin_x, origin_y)

    # scale: 1 m = size/10 pixels (so ~10 m fits vertically)
    scale = size / 10.0

    pts = []
    for x, y in wp_xy:
        px = int(origin_x - y * scale)   # left is +y → right in image
        py = int(origin_y - x * scale)   # forward is -y in image coordinates
        pts.append([px, py])

    if len(pts) >= 2:
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_arr], False, (0, 255, 0), 2, cv2.LINE_AA)

    # draw robot at origin
    cv2.circle(canvas, origin, 4, (255, 255, 255), -1)
    cv2.line(canvas, (origin_x, origin_y), (origin_x, origin_y - int(2 * scale)), (0, 0, 255), 2)

    return canvas


def choose_providers(use_cuda: bool):
    if use_cuda:
        try:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            return ["CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mp4", required=True, help="Path to input mp4 (front camera)")
    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument("--frames", type=int, default=21, help="Number of frames in input buffer (default: 21)")
    p.add_argument("--device", default="cuda", help="Device for preprocessing tensors (cuda or cpu)")
    p.add_argument("--fov_x", type=float, default=None, help="Optional fov_x for fisheye->pinhole conversion (deg)")
    p.add_argument("--fov_y", type=float, default=None, help="Optional fov_y for fisheye->pinhole conversion (deg)")
    p.add_argument("--intrinsic_f", type=float, default=790.0, help="Focal length f for intrinsic (default 790)")
    p.add_argument(
        "--saveOverlay",
        action='store_true',
        help="Save overlay video to overlay/<input_mp4_basename>_overlay.mp4",
    )
    p.add_argument("--simple_vis", action='store_true', help="Show simple 2D trajectory debug window")
    p.add_argument("--sample_fps", type=int, default=5, help="Number of frames per second to sample into the model buffer (default 5)")
    return p.parse_args()


def main():
    args = parse_args()

    providers = choose_providers(args.device == 'cuda')
    print(f"Using ONNX providers: {providers}")
    session = ort.InferenceSession(args.onnx, providers=providers)

    # Intrinsics consistent with repo defaults (modify if your mp4 differs)
    # intrinsic for projection: f, cx, cy, k
    # assume mp4 is 1920x1080; if not, pass adjusted values or let script resize
    intrinsic_params = [args.intrinsic_f, 1920.0 / 2.0, 1080.0 / 2.0, 0.0]

    # K and dist used by convert_fisheye_to_pinhole (scaled later if needed)
    K = np.array([
        [790 * 512 / 1920, 0.0, 512 / 2.],
        [0.0, 790 * 288 / 1080, 288 / 2.],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist = np.array([-0.03054249, 0.00974967, -0.00679687, 0.00238449, 0.0], dtype=np.float32)

    cap = cv2.VideoCapture(args.mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open mp4: {args.mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    print(f"Opened {args.mp4}, fps={fps}")

    # manipulate buffer length based on desired sample_fps (similar to ROS: image_callback ~ sample_fps)
    sample_stride = max(1, int(round(fps / args.sample_fps)))
    print(f"Original FPS: {fps}, Sampling for {args.sample_fps}Hz every {sample_stride} frames.")

    buffer_len = args.frames
    frame_buffer = deque(maxlen=buffer_len)

    # video writer (single-threaded, writes one frame per inference batch)
    writer = None
    out_path = None
    frames_written = 0
    if args.saveOverlay:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("[overlay] saveOverlay enabled, will write video when first overlay frame is ready")

    # prepare metric_spacing and embodiment id as in test script
    metric_spacing = np.array([[0., 0.51, -0.26, 0.26]], dtype=np.float32).reshape(1, 4)
    embodiment_id = np.array([[0]], dtype=np.int64)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    frame_count = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("End of video or cannot fetch frame")
                break

            frame_count += 1

            # 仅按 sample_stride 采样一部分帧，模拟 ROS 中较低频率的 image_callback
            if frame_count % sample_stride != 0:
                continue

            # Convert BGR -> RGB and resize to 1920x1080 if needed (same as ROS pipeline)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h0, w0 = frame_rgb.shape[:2]
            if (w0, h0) != (1920, 1080):
                frame_rgb = cv2.resize(frame_rgb, (1920, 1080), interpolation=cv2.INTER_LINEAR)

            # to torch (C,H,W) float [0,1]
            frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_buffer.append(frame_t)

            # 如果缓冲未满，就先显示原始帧（或简单跳过显示）
            if len(frame_buffer) < buffer_len:
                vis_img = frame_bgr
            else:
                # 构建 batch: (N, C, H, W)
                video_frames = torch.stack(list(frame_buffer), dim=0).to(device)

                # resize frames to model input (288,512)
                video_frames_small = torch.nn.functional.interpolate(
                    video_frames, size=(288, 512), mode='bilinear', align_corners=False
                )

                # convert fisheye->pinhole (expects N,C,H,W on device)
                fov_x = args.fov_x if args.fov_x is not None else math.degrees(2 * math.atan(1920 / (2 * 790)))
                fov_y = args.fov_y if args.fov_y is not None else math.degrees(2 * math.atan(1080 / (2 * 790)))
                pinhole = convert_fisheye_to_pinhole(
                    K,
                    dist,
                    device,
                    video_frames_small,
                    fov_x=fov_x,
                    fov_y=fov_y,
                )

                # onnx expects (B, T, C, H, W) with B=1, T=buffer_len
                observation = pinhole.unsqueeze(0).cpu().numpy().astype(np.float32)

                outputs = session.run(
                    None,
                    {"observation": observation, "embodiment_id": embodiment_id, "metric_spacing": metric_spacing},
                )

                # parse outputs (copying logic from test script)
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
                    [torch.zeros_like(wp_pred[..., 0:1, :]), wp_pred[..., :-1, :]], dim=-2
                )
                wp_pred[..., 0] = wp_pred[..., 0].clamp(0.0, 20.0)
                wp_np = wp_pred.detach()[0, :, :2].cpu().numpy()

                # 使用当前采样帧（对应 batch 的最后一帧）进行可视化，仿照 ROS 的 process_batch
                try:
                    vis_img = project_waypoints_to_fisheye_image_with_polygon_new(
                        wp_np, intrinsic_params, frame_rgb, save_fig=False
                    )
                except Exception as e:
                    print(f"[overlay] draw error: {e}")
                    vis_img = frame_bgr

            # optional simple 2D debug visualization
            if args.simple_vis and len(frame_buffer) == buffer_len:
                try:
                    traj_canvas = draw_simple_traj(wp_np)
                    cv2.imshow("traj_2d", traj_canvas)
                except Exception:
                    pass

            # show (single-threaded, approximate sample_fps display)
            cv2.imshow("overlay", vis_img)

            # lazy-init video writer when we know frame size
            if args.saveOverlay:
                if writer is None:
                    h, w = vis_img.shape[:2]
                    base = os.path.splitext(os.path.basename(args.mp4))[0]
                    out_dir = os.path.join(os.getcwd(), "overlay")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{base}_overlay.mp4")
                    out_fps = float(args.sample_fps)
                    print(
                        f"[overlay] initializing VideoWriter: path={out_path}, size=({w}x{h}), fps={out_fps}"
                    )
                    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
                    if not writer.isOpened():
                        print(f"[overlay] ERROR: failed to open VideoWriter at {out_path}")

                if writer is not None and writer.isOpened():
                    writer.write(vis_img)
                    frames_written += 1
                    if frames_written % 30 == 0:
                        print(f"[overlay] frames written: {frames_written}")

            # 适当等待以接近真实的 sample_fps 播放速度
            if cv2.waitKey(int(1000 / max(1, args.sample_fps))) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
