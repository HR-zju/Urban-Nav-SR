#!/usr/bin/env python3
"""Diagnostic: compare trajectory headings vs EKF and render BEV side-by-side.

Outputs:
 - trajectory_vs_ekf.csv in ride dir
 - bev_images/compare_{idx:06d}.jpg for first N frames

Usage:
 python3 tools/diag_traj_vs_ekf.py /path/to/ride_dir --frames 5
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import math
import csv
import argparse
import numpy as np
from PIL import Image

# ensure imports find local modules
TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR / "bev"))
sys.path.insert(0, str(TOOLS_DIR / "math_tools"))
from frodobot_bev_enhanced import render_bev_map, MAP_PROVIDERS, ProcessOptions
from pose_ekf import PoseEKF


def _quaternion_to_euler_deg(*, x: float, y: float, z: float, w: float):
    sinr_cosp = 2.0 * (float(w) * float(x) + float(y) * float(z))
    cosr_cosp = 1.0 - 2.0 * (float(x) * float(x) + float(y) * float(y))
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (float(w) * float(y) - float(z) * float(x))))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (float(w) * float(z) + float(x) * float(y))
    cosy_cosp = 1.0 - 2.0 * (float(y) * float(y) + float(z) * float(z))
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _calculate_heading_from_gps(lat1, lng1, lat2, lng2):
    if abs(lat1 - lat2) < 1e-7 and abs(lng1 - lng2) < 1e-7:
        return 0.0
    d_lon = math.radians(lng2 - lng1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    y = math.sin(d_lon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def run_diagnostic(ride_dir: Path, frames: int = 5, sample_fps: float = 20.0):
    ride_dir = Path(ride_dir)
    if not ride_dir.exists():
        raise RuntimeError(f"Ride dir not found: {ride_dir}")

    traj_path = ride_dir / 'trajectory.json'
    if not traj_path.exists():
        raise RuntimeError(f"trajectory.json missing: {traj_path}")
    trajectory = json.loads(traj_path.read_text())

    gps_arr = np.load(ride_dir / 'frame_gps.npy')
    ori_arr = np.load(ride_dir / 'frame_orientations.npy')
    times_arr = np.load(ride_dir / 'frame_times.npy')
    imu_path = ride_dir / 'frame_angular_velocities.npy'
    imu_arr = None
    if imu_path.exists():
        imu_arr = np.load(imu_path)

    # Build list of data indices and selected frames (use trajectory order)
    selected = []
    for entry in trajectory:
        img = Path(entry['image_path'])
        try:
            idx = int(img.stem)
        except Exception:
            # try basename
            try:
                idx = int(Path(entry['image_path']).name.split('.')[0])
            except Exception:
                idx = None
        selected.append((idx, entry))

    N = min(frames, len(selected))
    selected = selected[:N]

    # Build yaw_list from quaternions
    yaw_list = []
    for idx, ent in selected:
        if idx is None:
            yaw_list.append(None)
            continue
        qx, qy, qz, qw = map(float, ori_arr[idx])
        _, _, yaw_deg = _quaternion_to_euler_deg(x=qx, y=qy, z=qz, w=qw)
        compass_yaw = (90.0 - float(yaw_deg)) % 360.0
        yaw_list.append(compass_yaw)

    # Build smoothed_positions from trajectory (reusing trajectory lat/lng)
    smoothed_positions = [(float(e['lat']), float(e['lng'])) for _, e in selected]
    raw_headings = []
    for i in range(len(smoothed_positions)):
        if i < len(smoothed_positions) - 1:
            next_lat, next_lng = smoothed_positions[i+1]
            lat, lng = smoothed_positions[i]
            raw_headings.append(_calculate_heading_from_gps(lat, lng, next_lat, next_lng))
        elif i > 0:
            prev_lat, prev_lng = smoothed_positions[i-1]
            lat, lng = smoothed_positions[i]
            raw_headings.append(_calculate_heading_from_gps(prev_lat, prev_lng, lat, lng))
        else:
            raw_headings.append(0.0)

    # Prepare EKF
    first_idx = selected[0][0]
    yaw_vals = [y for y in yaw_list if y is not None]
    yaw_range = max(yaw_vals) - min(yaw_vals) if yaw_vals else 0.0
    use_yaw_obs = True
    if yaw_range < 0.5:
        use_yaw_obs = False
        if len(smoothed_positions) >= 2:
            init_heading = _calculate_heading_from_gps(smoothed_positions[0][0], smoothed_positions[0][1], smoothed_positions[1][0], smoothed_positions[1][1])
        else:
            init_heading = raw_headings[0]
        ekf = PoseEKF.init_from_latlng_heading(float(gps_arr[first_idx,0]), float(gps_arr[first_idx,1]), init_heading, pos_var=25.0, theta_var=0.5)
    else:
        ekf = PoseEKF.init_from_latlng_heading(float(gps_arr[first_idx,0]), float(gps_arr[first_idx,1]), yaw_vals[0], pos_var=25.0, theta_var=0.5)

    prev_time_ns = int(times_arr[first_idx])
    ekf_headings = []
    for i, (idx, ent) in enumerate(selected):
        t_ns = int(times_arr[idx])
        dt = (t_ns - prev_time_ns) / 1e9 if i > 0 else 0.0
        if imu_arr is not None and idx < len(imu_arr):
            omega = float(imu_arr[idx,2])
        else:
            cur_yaw = yaw_list[i] if yaw_list[i] is not None else raw_headings[i]
            prev_yaw = yaw_list[i-1] if i>0 and yaw_list[i-1] is not None else cur_yaw
            diff_rad = math.radians(cur_yaw - prev_yaw)
            diff_rad = (diff_rad + math.pi) % (2*math.pi) - math.pi
            omega = diff_rad / max(dt, 1e-6)
        ekf.predict(omega, dt, q_pos=1e-3, q_theta=1e-2)
        if use_yaw_obs and yaw_list[i] is not None:
            ekf.update_gps_and_yaw(float(gps_arr[idx,0]), float(gps_arr[idx,1]), yaw_deg=yaw_list[i], r_xy=9.0, r_theta=0.5)
        else:
            ekf.update_gps(float(gps_arr[idx,0]), float(gps_arr[idx,1]), r_xy=9.0)
        ekf_headings.append(ekf.get_heading_deg())
        prev_time_ns = t_ns

    # Write CSV
    csv_path = ride_dir / 'trajectory_vs_ekf.csv'
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['data_idx','timestamp','quat_yaw_deg','compass_from_quat_deg','gps_bearing_deg','imu_omega_z','trajectory_heading_deg','ekf_heading_deg','attitude_source'])
        for i, (idx, ent) in enumerate(selected):
            ts = int(ent['timestamp'])
            quat_yaw = None
            compass = None
            if idx is not None:
                    qx,qy,qz,qw = map(float, ori_arr[idx])
                    _,_,qy = _quaternion_to_euler_deg(x=qx,y=qy,z=qz,w=qw)
                    quat_yaw = float(qy)
                    compass = (90.0 - qy) % 360.0
            gps_b = raw_headings[i]
            imu_z = float(imu_arr[idx,2]) if (imu_arr is not None and idx is not None) else ''
            traj_h = float(ent.get('heading_deg', ''))
            ekf_h = float(ekf_headings[i])
            writer.writerow([idx, ts, quat_yaw, compass, gps_b, imu_z, traj_h, ekf_h, ent.get('attitude_source','')])

    print(f"Wrote CSV: {csv_path}")

    # Render compare images
    out_dir = ride_dir / 'bev_images'
    out_dir.mkdir(parents=True, exist_ok=True)
    provider_config = MAP_PROVIDERS.get('osm')
    options = ProcessOptions(map_provider='osm', output_size=1024, zoom=19, cache_dir=Path('/tmp/bev_cache'), draw_trajectory=True, add_attribution=True)

    for i, (idx, ent) in enumerate(selected):
        lat = float(ent['lat'])
        lng = float(ent['lng'])
        traj_h = float(ent['heading_deg'])
        ekf_h = float(ekf_headings[i])
        try:
            img_l = render_bev_map(lat=lat, lng=lng, heading_deg=traj_h, navigation_path_gps=smoothed_positions, options=options, provider_config=provider_config)
            img_r = render_bev_map(lat=lat, lng=lng, heading_deg=ekf_h, navigation_path_gps=smoothed_positions, options=options, provider_config=provider_config)
            # combine side-by-side
            w,h = img_l.size
            combo = Image.new('RGB', (w*2, h))
            combo.paste(img_l, (0,0))
            combo.paste(img_r, (w,0))
            out_path = out_dir / f'compare_{idx:06d}.jpg'
            combo.save(out_path, quality=90)
            # write meta
            meta = {'left_heading_deg': traj_h, 'right_heading_deg': ekf_h, 'data_idx': idx}
            (out_path.with_suffix('.compare.meta.json')).write_text(json.dumps(meta), encoding='utf-8')
            print(f"Saved compare image: {out_path}")
        except Exception as e:
            print(f"Failed render for idx {idx}: {e}")

    print('Done diagnostics')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ride_dir', type=Path)
    p.add_argument('--frames', type=int, default=5)
    args = p.parse_args()
    run_diagnostic(args.ride_dir, frames=args.frames)
