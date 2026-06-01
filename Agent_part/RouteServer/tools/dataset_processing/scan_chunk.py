#!/usr/bin/env python3
"""Scan a sidewalk chunk directory for IMU/quaternion/heading issues.

Outputs a CSV summary and prints a short suspicious list.

Usage:
  python3 tools/scan_chunk.py --sidewalk-root "/media/.../sidewalk/chunk_26"
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import csv


def _quat_to_compass_yaw(qx, qy, qz, qw):
    # reuse euler extraction
    siny_cosp = 2.0 * (float(qw) * float(qz) + float(qx) * float(qy))
    cosy_cosp = 1.0 - 2.0 * (float(qy) * float(qy) + float(qz) * float(qz))
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    compass = (90.0 - yaw) % 360.0
    return compass


def _calc_gps_bearings(positions):
    # positions: list of (lat,lng)
    bearings = []
    L = len(positions)
    for i in range(L):
        if i < L - 1:
            lat1, lng1 = positions[i]
            lat2, lng2 = positions[i + 1]
        elif i > 0:
            lat1, lng1 = positions[i - 1]
            lat2, lng2 = positions[i]
        else:
            bearings.append(0.0)
            continue
        dlon = math.radians(lng2 - lng1)
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        y = math.sin(dlon) * math.cos(lat2_r)
        x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        bearing = math.degrees(math.atan2(y, x))
        compass = (bearing + 360.0) % 360.0
        bearings.append(compass)
    return bearings


def _ang_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def scan_chunk(sidewalk_chunk_dir: Path, out_csv: Path):
    sidewalk_chunk_dir = Path(sidewalk_chunk_dir)
    rides = sorted([d for d in sidewalk_chunk_dir.iterdir() if d.is_dir()])
    rows = []
    for ride in rides:
        try:
            traj_path = ride / 'trajectory.json'
            if not traj_path.exists():
                rows.append([ride.name, 'no_trajectory', '', '', '', '', ''])
                continue
            traj = json.loads(traj_path.read_text())
            # sample attitude_source
            attitude = traj[0].get('attitude_source','') if len(traj)>0 else ''
            # build trajectory headings list and positions
            traj_headings = [float(f.get('heading_deg', 0.0)) for f in traj]
            positions = [(float(f.get('lat', 0.0)), float(f.get('lng', 0.0))) for f in traj]
            gps_bearings = _calc_gps_bearings(positions)
            # load quaternions if present
            ori_path = ride / 'frame_orientations.npy'
            yaw_list = []
            yaw_range = ''
            if ori_path.exists():
                ori = np.load(ori_path)
                for q in ori:
                    qx,qy,qz,qw = map(float, q)
                    yaw = _quat_to_compass_yaw(qx,qy,qz,qw)
                    yaw_list.append(yaw)
                if len(yaw_list)>0:
                    yaw_range = float(max(yaw_list)-min(yaw_list))
            # imu
            imu_path = ride / 'frame_angular_velocities.npy'
            imu_ratio = ''
            imu_nonzero_frac = ''
            if imu_path.exists():
                imu = np.load(imu_path)
                if imu.ndim==2 and imu.shape[1]>=3:
                    z = np.abs(imu[:,2])
                    imu_nonzero_frac = float((z>1e-4).sum())/len(z)
                    imu_ratio = imu_nonzero_frac
            # compare traj vs gps
            mean_abs_diff = ''
            frac_large = ''
            if len(traj_headings)>0 and len(gps_bearings)==len(traj_headings):
                diffs = [abs(_ang_diff(traj_headings[i], gps_bearings[i])) for i in range(len(traj_headings))]
                mean_abs_diff = float(sum(diffs)/len(diffs))
                frac_large = float(sum(1 for d in diffs if d>30.0)/len(diffs))
            has_meta = (ride / 'mcap_extract_meta.json').exists()
            rows.append([ride.name, imu_ratio, yaw_range, mean_abs_diff, frac_large, has_meta, attitude])
        except Exception as e:
            rows.append([ride.name, 'error', str(e)])
    # write CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ride','imu_nonzero_frac','yaw_range_deg','mean_abs_diff_deg','frac_diff_gt30','has_meta','attitude_sample'])
        for r in rows:
            writer.writerow(r)
    # print suspicious
    suspicious = []
    for r in rows:
        try:
            imu_frac = float(r[1]) if r[1] != '' else 0.0
            yaw_r = float(r[2]) if r[2] != '' else 9999.0
            frac_large = float(r[4]) if r[4] != '' else 0.0
            if imu_frac < 0.3 or yaw_r < 0.5 or frac_large > 0.2:
                suspicious.append(r)
        except Exception:
            continue
    print(f"Wrote summary to {out_csv}. Found {len(suspicious)} suspicious rides (criteria: imu_frac<0.3 or yaw_range<0.5 or frac_diff_gt30>0.2):")
    for s in suspicious[:200]:
        print(' -', s)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sidewalk-root', type=Path, required=True)
    p.add_argument('--out-csv', type=Path, default=None)
    args = p.parse_args()
    out = args.out_csv if args.out_csv else Path(args.sidewalk_root)/'scan_summary.csv'
    scan_chunk(args.sidewalk_root, out)
