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
import os
from collections import deque
import numpy as np
import cv2
import onnxruntime as ort
import torch

import ast
import inspect

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
    p.add_argument("--save", action='store_true', help="Save overlay video to ./overlay_out.mp4")
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

    buffer_len = args.frames
    frame_buffer = deque(maxlen=buffer_len)

    # video writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = None  # init later when we know frame size

    # prepare metric_spacing and embodiment id as in test script
    metric_spacing = np.array([[0., 0.51, -0.26, 0.26]], dtype=np.float32).reshape(1, 4)
    embodiment_id = np.array([[0]], dtype=np.int64)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("End of video or cannot fetch frame")
                break

            # Convert BGR -> RGB and to float tensor
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Resize to 1920x1080 if source differs (keeps original aspect if already 1920x1080)
            H0, W0 = frame_rgb.shape[:2]
            if (W0, H0) != (1920, 1080):
                frame_rgb = cv2.resize(frame_rgb, (1920, 1080), interpolation=cv2.INTER_LINEAR)

            # to torch (C,H,W) float [0,1]
            frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_buffer.append(frame_t)

            if len(frame_buffer) < buffer_len:
                # show raw frame until buffer fills
                cv2.imshow('overlay', frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # build batch: (N, C, H, W)
            video_frames = torch.stack(list(frame_buffer), dim=0).to(device)

            # resize frames to model input (288,512)
            video_frames_small = torch.nn.functional.interpolate(video_frames, size=(288, 512), mode='bilinear', align_corners=False)

            # convert fisheye->pinhole (expects N,C,H,W on device)
            pinhole = convert_fisheye_to_pinhole(K, dist, device, video_frames_small, fov_x=args.fov_x, fov_y=args.fov_y)
            # pinhole returned is (N,C,H,W) on device

            observation = pinhole.unsqueeze(0).cpu().numpy().astype(np.float32)

            # run onnx
            outputs = session.run(None, {"observation": observation, "embodiment_id": embodiment_id, 'metric_spacing': metric_spacing})

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
            wp_pred = torch.cat([torch.zeros_like(wp_pred[..., 0:1, :]), wp_pred[..., :-1, :]], dim=-2)
            wp_pred[..., 0] = wp_pred[..., 0].clamp(0., 20.)
            wp = wp_pred.detach()[0, :, :2]
            # reorder to [x,y] expected by vis: same transform as in test script
            wp = torch.cat([ -wp[:, 1:2], wp[:, 0:1]], dim=1).cpu().numpy()

            # call vis function (expects RGB numpy image input)
            vis_img = project_waypoints_to_fisheye_image_with_polygon_new(wp, intrinsic_params, frame_rgb, save_fig=False)
            # project returns BGR image
            vis_bgr = vis_img

            # show
            cv2.imshow('overlay', vis_bgr)
            if writer is None and args.save:
                h, w = vis_bgr.shape[:2]
                writer = cv2.VideoWriter('overlay_out.mp4', fourcc, fps, (w, h))
            if writer is not None:
                writer.write(vis_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
