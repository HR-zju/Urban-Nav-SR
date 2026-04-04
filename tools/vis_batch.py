#!/usr/bin/env python3
"""
Batch visualizer (clean) for saved samples in vis/ folders.
Run this instead of the messy tools/vis_sample.py created earlier.

Usage:
  mamba activate foundation_policy
  python tools/vis_batch.py --folder vis/navflow20260404_122553 --fps 20 --out vis/navflow20260404_122553/overlay_video.mp4
"""
import argparse
import ast
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import cv2
import torch

from test_nav_flow_il_go2 import trajectory_nms


def load_vis_function(filepath, func_name):
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


def ensure_traj_tensor(traj_np):
    arr = np.asarray(traj_np)
    if arr.ndim == 4:
        B, N, T, D = arr.shape
        if D == 2:
            z = np.zeros((B, N, T, 1), dtype=arr.dtype)
            arr = np.concatenate([arr, z], axis=-1)
        return torch.from_numpy(arr).float()
    elif arr.ndim == 3:
        N, T, D = arr.shape
        if D == 2:
            arr = arr[None, ...]
            z = np.zeros((1, N, T, 1), dtype=arr.dtype)
            arr = np.concatenate([arr, z], axis=-1)
            return torch.from_numpy(arr).float()
    raise ValueError(f"Unsupported traj shape: {arr.shape}")


def ensure_score_tensor(score_np, N):
    s = np.asarray(score_np)
    if s.ndim == 1 and s.shape[0] == N:
        return torch.from_numpy(s[None, :]).float()
    s = s.ravel()
    if s.size >= N:
        return torch.from_numpy(s[:N][None, :]).float()
    return torch.zeros((1, N), dtype=torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--intrinsic_f', type=float, default=790.0)
    p.add_argument('--out', default=None)
    p.add_argument('--keep_tmp', action='store_true')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(folder)

    vis_file = Path(__file__).resolve().parent.parent / 'vis_gt_single_gt.py'
    vis_func = load_vis_function(str(vis_file), 'project_waypoints_to_fisheye_image_with_polygon_new')

    intrinsic = [args.intrinsic_f, 1920.0 / 2.0, 1080.0 / 2.0, 0.0]

    overlays = []
    i = 0
    while True:
        img_path = folder / f'sample_{i}.png'
        traj_path = folder / f'sample_{i}_traj.npy'
        score_path = folder / f'sample_{i}_score.npy'
        if not img_path.exists():
            break
        if not traj_path.exists():
            print(f"skip {i}: traj missing")
            i += 1
            continue
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        traj_np = np.load(str(traj_path), allow_pickle=False)
        try:
            traj_t = ensure_traj_tensor(traj_np)
        except Exception as e:
            print(f"bad traj {i}: {e}")
            i += 1
            continue
        B, N, T, D = traj_t.shape
        if score_path.exists():
            score_np = np.load(str(score_path), allow_pickle=False)
            score_t = ensure_score_tensor(score_np, N)
        else:
            score_t = torch.zeros((B, N), dtype=torch.float32)
        try:
            selected_trajs, _ = trajectory_nms(traj_t, score_t)
        except Exception as e:
            print(f"trajectory_nms failed {i}: {e}")
            selected_trajs = traj_t[:, :1, :, :]
        chosen = selected_trajs[0, 0].cpu().numpy()
        wp = chosen[:, :2]
        try:
            out_img = vis_func(wp.astype(np.float32), intrinsic, img_rgb, save_fig=False)
        except Exception as e:
            print(f"projection failed {i}: {e}")
            out_img = img_bgr
        out_path = folder / f'sample_{i}_overlay.png'
        cv2.imwrite(str(out_path), out_img)
        overlays.append(out_path)
        print(f"wrote {out_path}")
        if args.show:
            cv2.imshow('overlay', out_img)
            cv2.waitKey(1)
        i += 1
    if args.show:
        cv2.destroyAllWindows()

    if not overlays:
        print('no overlays')
        return

    tmpdir = Path(tempfile.mkdtemp(prefix='vis_seq_'))
    for idx, p in enumerate(overlays):
        dst = tmpdir / f'{idx:06d}.png'
        shutil.copy(str(p), str(dst))
    out_mp4 = Path(args.out) if args.out else folder / 'overlay_video.mp4'
    cmd = ['ffmpeg', '-y', '-framerate', str(args.fps), '-i', str(tmpdir / '%06d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_mp4)]
    print('running:', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print('wrote', out_mp4)
    except subprocess.CalledProcessError as e:
        print('ffmpeg failed', e)
    if not args.keep_tmp:
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
