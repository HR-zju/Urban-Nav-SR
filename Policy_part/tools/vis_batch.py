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
from datetime import datetime

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
    vis_func = load_vis_function(
        str(vis_file),
        'project_waypoints_to_fisheye_image_with_polygon_new',
    )

    intrinsic = [args.intrinsic_f, 1920.0 / 2.0, 1080.0 / 2.0, 0.0]

    # collect messages for a run-level summary
    summary_lines = []

    overlays = []
    i = 0
    while True:
        img_path = folder / f'sample_{i}.png'
        traj_path = folder / f'sample_{i}_traj.npy'
        score_path = folder / f'sample_{i}_score.npy'
        if not img_path.exists():
            break
        if not traj_path.exists():
            msg = (
                f"[skip] sample_{i}: trajectory file missing. "
                f"img={img_path}, traj={traj_path}"
            )
            print(msg)
            summary_lines.append(msg)
            i += 1
            continue
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        traj_np = np.load(str(traj_path), allow_pickle=False)
        try:
            traj_t = ensure_traj_tensor(traj_np)
        except Exception as e:
            shape = getattr(traj_np, 'shape', None)
            msg = (
                f"[error] bad traj for sample_{i}: cannot convert to "
                f"tensor. path={traj_path}, shape={shape}, error={e}"
            )
            print(msg)
            summary_lines.append(msg)
            i += 1
            continue
        B, N, T, D = traj_t.shape
        if score_path.exists():
            score_np = np.load(str(score_path), allow_pickle=False)
            score_t = ensure_score_tensor(score_np, N)
        else:
            msg = (
                f"[warn] sample_{i}: score file missing, using zeros. "
                f"expected path={score_path}"
            )
            print(msg)
            summary_lines.append(msg)
            score_t = torch.zeros((B, N), dtype=torch.float32)
        try:
            selected_trajs, _ = trajectory_nms(traj_t, score_t)
        except Exception as e:
            msg = (
                f"[error] trajectory_nms failed for sample_{i}: {e}. "
                f"traj_shape={traj_t.shape}, score_shape={score_t.shape}. "
                "Falling back to first trajectory without NMS."
            )
            print(msg)
            summary_lines.append(msg)
            selected_trajs = traj_t[:, :1, :, :]
        chosen = selected_trajs[0, 0].cpu().numpy()
        wp = chosen[:, :2]
        try:
            out_img = vis_func(
                wp.astype(np.float32),
                intrinsic,
                img_rgb,
                save_fig=False,
            )
        except Exception as e:
            msg = (
                f"[error] projection failed for sample_{i}: {e}. "
                f"Will save raw image without trajectory overlay."
            )
            print(msg)
            summary_lines.append(msg)
            out_img = img_bgr
        out_path = folder / f'sample_{i}_overlay.png'
        cv2.imwrite(str(out_path), out_img)
        overlays.append(out_path)
        msg = f"wrote {out_path}"
        print(msg)
        summary_lines.append(msg)
        if args.show:
            cv2.imshow('overlay', out_img)
            cv2.waitKey(1)
        i += 1
    if args.show:
        cv2.destroyAllWindows()

    if not overlays:
        msg = 'no overlays'
        print(msg)
        summary_lines.append(msg)
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix='vis_seq_'))
        for idx, p in enumerate(overlays):
            dst = tmpdir / f'{idx:06d}.png'
            shutil.copy(str(p), str(dst))
        out_mp4 = Path(args.out) if args.out else folder / 'overlay_video.mp4'
        cmd = [
            'ffmpeg',
            '-y',
            '-framerate',
            str(args.fps),
            '-i',
            str(tmpdir / '%06d.png'),
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            str(out_mp4),
        ]
        print('running:', ' '.join(cmd))
        try:
            subprocess.run(cmd, check=True)
            msg = f'wrote {out_mp4}'
            print(msg)
            summary_lines.append(msg)
        except subprocess.CalledProcessError as e:
            msg = f'ffmpeg failed {e}'
            print(msg)
            summary_lines.append(msg)
        if not args.keep_tmp:
            shutil.rmtree(tmpdir)

    # write run summary to logs/summary file
    try:
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = folder.name
        summary_path = log_dir / f'vis_batch_{base}_{stamp}_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f'folder: {folder}\n')
            for line in summary_lines:
                f.write(line + '\n')
        print(f'[summary] wrote {summary_path}')
    except Exception as e:
        print(f'[summary] failed to write summary file: {e}')


if __name__ == '__main__':
    main()
