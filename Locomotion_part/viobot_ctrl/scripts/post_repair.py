import os
import argparse
import csv
import json
import traceback
from datetime import datetime

import numpy as np
from mcap_ros2.reader import read_ros2_messages
from mcap_ros2.writer import Writer

ODOM_TOPIC = '/baton/stereo3/odometry'
ODOM_ORIG_TOPIC = '/baton/stereo3/odometry_orig'
ODOM_REPAIRED_TOPIC = '/baton/stereo3/odometry_repaired'
SPP_FIX_TOPIC = '/baton/gnss/spp_fix'
SPP_FIX_ORIG_TOPIC = '/baton/gnss/spp_fix_orig'
SPP_FIX_REPAIRED_TOPIC = '/baton/gnss/spp_fix_repaired'
SPP_FIX_RAW_TOPIC = '/baton/gnss/spp_fix_raw'

ODOM_MAX_STEP_2D_M = 0.8
ODOM_MAX_DT_FOR_STEP_JUMP_S = 0.20
ODOM_MAX_SPEED_2D_MPS = 8.0
ODOM_TWIST_SPEED_RATIO = 4.0
ODOM_TWIST_SPEED_MARGIN_MPS = 3.0
ODOM_HISTORY_SIZE = 30

OK_SKIP_STATUSES = {
    'skipped_existing_output',
    'skipped_output_exists',
}

QA_MAX_REPAIRED_STEP_2D_M = 1.0
QA_MAX_REPAIRED_SPEED_2D_MPS = 50.0
QA_MAX_JUMPS_GT_1M = 0
QA_MAX_JUMPS_GT_2M = 0
QA_MIN_ALIGNMENT_INLIER_RATIO = 0.20
QA_MAX_ALIGNMENT_MEAN_ERR_M = 3.0
QA_MAX_ALIGNMENT_MAX_ERR_M = 10.0


def ecef2geo(ecef):
    x, y, z = ecef
    a = 6378137.0
    b = 6356752.31424518
    e2 = (a**2 - b**2) / (a**2)
    ep2 = (a**2 - b**2) / (b**2)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + ep2 * b * np.sin(th)**3), (p - e2 * a * np.cos(th)**3))
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    return np.array([np.degrees(lat), np.degrees(lon), alt])

def geo2ecef(lla):
    lat = np.radians(lla[0])
    lon = np.radians(lla[1])
    alt = lla[2]
    a = 6378137.0
    e2 = 0.00669437999014
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    return np.array([x, y, z])

def geo2rotation(lla):
    lat = np.radians(lla[0])
    lon = np.radians(lla[1])
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    return np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [     0.0,            cos_lat,           sin_lat]
    ])


def discover_mcap_files(path):
    """Discover .mcap files from a path (file or directory).
    Excludes files ending with '_repaired.mcap' or '.download'.
    """
    if os.path.isfile(path):
        if path.endswith('.mcap') and not path.endswith('.download'):
            return [path]
        else:
            print(f"[WARN] Not an mcap file: {path}")
            return []
    elif os.path.isdir(path):
        files = []
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.endswith('.mcap') and not f.endswith('_repaired.mcap') and not f.endswith('.download'):
                    files.append(os.path.join(root, f))
        files.sort()
        return files
    else:
        print(f"[ERROR] Path does not exist: {path}")
        return []


def make_output_path(input_path, output_dir=None, input_root=None):
    """Generate output path: <basename>_repaired.mcap"""
    base_name = os.path.basename(input_path)
    base, ext = os.path.splitext(base_name)
    repaired_name = f"{base}_repaired{ext}"

    if output_dir is None:
        return os.path.join(os.path.dirname(input_path), repaired_name)

    if input_root is not None:
        rel_dir = os.path.dirname(os.path.relpath(os.path.abspath(input_path), os.path.abspath(input_root)))
        if rel_dir == '.':
            rel_dir = ''
    else:
        rel_dir = ''
    return os.path.join(output_dir, rel_dir, repaired_name)

def infer_input_root(path):
    if os.path.isdir(path):
        return os.path.abspath(path)
    return os.path.dirname(os.path.abspath(path))

def odom_msg_to_enu(ros_msg):
    """Map Stereo3 odometry pose into the pseudo-ENU frame used by the node."""
    pos = ros_msg.pose.pose.position
    return np.array([-pos.y, pos.x, pos.z], dtype=float)

def odom_twist_to_enu(ros_msg):
    """Map odometry linear twist into the same pseudo-ENU frame as pose."""
    twist = ros_msg.twist.twist.linear
    return np.array([-twist.y, twist.x, twist.z], dtype=float)

def set_odom_msg_from_enu(ros_msg, enu_pos):
    """Write a pseudo-ENU position back into the original odometry axes."""
    ros_msg.pose.pose.position.x = float(enu_pos[1])
    ros_msg.pose.pose.position.y = float(-enu_pos[0])
    ros_msg.pose.pose.position.z = float(enu_pos[2])

def clone_ros_msg(value):
    """Clone mcap_ros2 dynamic messages whose fields are declared in __slots__."""
    if hasattr(value, '__slots__'):
        cloned = type(value)()
        for field in value.__slots__:
            setattr(cloned, field, clone_ros_msg(getattr(value, field)))
        return cloned
    if isinstance(value, list):
        return [clone_ros_msg(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_ros_msg(item) for item in value)
    return value

def compute_step_metrics(t, xyz):
    t = np.asarray(t, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    metrics = {
        'count': int(len(xyz)),
        'duration_s': 0.0,
        'freq_hz': 0.0,
        'step2d_mean_m': 0.0,
        'step2d_max_m': 0.0,
        'step2d_p95_m': 0.0,
        'step2d_p99_m': 0.0,
        'step3d_max_m': 0.0,
        'alt_step_abs_max_m': 0.0,
        'speed2d_max_mps': 0.0,
        'jumps_gt_0_5m': 0,
        'jumps_gt_1m': 0,
        'jumps_gt_2m': 0,
        'jumps_gt_5m': 0,
        'nonpositive_dt_count': 0,
        'nonfinite_count': 0,
    }

    if len(xyz) < 2:
        return metrics

    dt = np.diff(t)
    dxyz = np.diff(xyz, axis=0)
    finite_steps = np.isfinite(dt) & np.all(np.isfinite(dxyz), axis=1)
    metrics['nonpositive_dt_count'] = int(np.sum(dt <= 0))
    metrics['nonfinite_count'] = int(len(dt) - np.sum(finite_steps))

    valid = finite_steps & (dt > 0)
    if not np.any(valid):
        return metrics

    dt_valid = dt[valid]
    dxyz_valid = dxyz[valid]
    dist_2d = np.linalg.norm(dxyz[:, :2], axis=1)
    dist_3d = np.linalg.norm(dxyz_valid, axis=1)
    dist_2d_valid = np.linalg.norm(dxyz_valid[:, :2], axis=1)
    speed_2d = dist_2d_valid / dt_valid

    metrics.update({
        'duration_s': float(t[-1] - t[0]),
        'freq_hz': float(1.0 / np.mean(dt_valid)),
        'step2d_mean_m': float(np.mean(dist_2d_valid)),
        'step2d_max_m': float(np.max(dist_2d_valid)),
        'step2d_p95_m': float(np.percentile(dist_2d_valid, 95)),
        'step2d_p99_m': float(np.percentile(dist_2d_valid, 99)),
        'step3d_max_m': float(np.max(dist_3d)),
        'alt_step_abs_max_m': float(np.max(np.abs(dxyz_valid[:, 2]))),
        'speed2d_max_mps': float(np.max(speed_2d)),
        'jumps_gt_0_5m': int(np.sum(dist_2d > 0.5)),
        'jumps_gt_1m': int(np.sum(dist_2d > 1.0)),
        'jumps_gt_2m': int(np.sum(dist_2d > 2.0)),
        'jumps_gt_5m': int(np.sum(dist_2d > 5.0)),
    })
    return metrics

def compute_navsat_metrics(records):
    if len(records) < 2:
        return compute_step_metrics([], [])

    lat = np.array([r['lat'] for r in records], dtype=float)
    lon = np.array([r['lon'] for r in records], dtype=float)
    alt = np.array([r['alt'] for r in records], dtype=float)
    t = np.array([r['t'] for r in records], dtype=float)

    mean_lat_rad = np.radians(np.mean(lat))
    R_earth = 6378137.0
    x = np.radians(lon - lon[0]) * R_earth * np.cos(mean_lat_rad)
    y = np.radians(lat - lat[0]) * R_earth
    z = alt - alt[0]
    return compute_step_metrics(t, np.column_stack([x, y, z]))

def print_step_stats(label, t, xyz, metrics=None):
    metrics = metrics or compute_step_metrics(t, xyz)
    if metrics['count'] < 2:
        return
    print(f"\n[{label}] ({metrics['count']} msgs, {metrics['freq_hz']:.1f} Hz):")
    print(f"  2D Step Dist (m) => Mean: {metrics['step2d_mean_m']:.4f} | Max: {metrics['step2d_max_m']:.4f} | 95%: {metrics['step2d_p95_m']:.4f} | 99%: {metrics['step2d_p99_m']:.4f}")
    print(f"  Max Speed (m/s)  => {metrics['speed2d_max_mps']:.2f}")
    print(f"  Jump Counts      => >0.5m: {metrics['jumps_gt_0_5m']} | >1.0m: {metrics['jumps_gt_1m']} | >2.0m: {metrics['jumps_gt_2m']} | >5.0m: {metrics['jumps_gt_5m']}")

def repair_odom_data(odom_data):
    """Remove single-frame pose jumps while preserving the odometry trend.

    The input pose occasionally jumps several meters in one 24 Hz frame, while
    twist remains reasonable. This keeps a cumulative offset so future raw pose
    samples remain usable after a rejected transition.
    """
    if not odom_data:
        return [], {}

    raw_pos = np.array([[p['x'], p['y'], p['z']] for p in odom_data], dtype=float)
    t = np.array([p['t'] for p in odom_data], dtype=float)
    repaired = np.zeros_like(raw_pos)
    repaired[0] = raw_pos[0]

    offset = np.zeros(3, dtype=float)
    good_deltas = []
    bad_flags = np.zeros(len(odom_data), dtype=bool)
    bad_reasons = []

    def estimate_expected_delta(i, dt):
        prev_twist = odom_data[i - 1]['twist']
        cur_twist = odom_data[i]['twist']
        twist_delta = 0.5 * (prev_twist + cur_twist) * dt

        if np.all(np.isfinite(twist_delta)) and np.linalg.norm(twist_delta[:2]) <= ODOM_MAX_SPEED_2D_MPS * max(dt, 1e-3):
            return twist_delta

        if good_deltas:
            recent = np.array(good_deltas[-ODOM_HISTORY_SIZE:], dtype=float)
            return np.median(recent, axis=0)

        return np.zeros(3, dtype=float)

    for i in range(1, len(odom_data)):
        dt = t[i] - t[i - 1]
        candidate = raw_pos[i] + offset
        delta = candidate - repaired[i - 1]
        step_2d = np.linalg.norm(delta[:2])
        pose_speed = step_2d / dt if dt > 0 else np.inf

        prev_twist = odom_data[i - 1]['twist']
        cur_twist = odom_data[i]['twist']
        twist_speed = np.linalg.norm(0.5 * (prev_twist[:2] + cur_twist[:2]))
        speed_limit = max(ODOM_MAX_SPEED_2D_MPS, ODOM_TWIST_SPEED_RATIO * twist_speed + ODOM_TWIST_SPEED_MARGIN_MPS)

        bad = False
        reason = ""
        if dt <= 0 or not np.isfinite(dt):
            bad = True
            reason = "bad_dt"
        elif not np.all(np.isfinite(delta)):
            bad = True
            reason = "non_finite_pose"
        elif step_2d > ODOM_MAX_STEP_2D_M and dt < ODOM_MAX_DT_FOR_STEP_JUMP_S:
            bad = True
            reason = "large_short_step"
        elif pose_speed > speed_limit:
            bad = True
            reason = "speed_outlier"

        if bad:
            expected_delta = estimate_expected_delta(i, max(dt, 0.0))
            repaired[i] = repaired[i - 1] + expected_delta
            offset = repaired[i] - raw_pos[i]
            bad_flags[i] = True
            bad_reasons.append((i - 1, i, reason, dt, step_2d, pose_speed, twist_speed))
        else:
            repaired[i] = candidate
            good_deltas.append(delta)
            if len(good_deltas) > ODOM_HISTORY_SIZE:
                good_deltas.pop(0)

    stats = {
        'bad_flags': bad_flags,
        'bad_reasons': bad_reasons,
        'raw_pos': raw_pos,
        'repaired_pos': repaired,
        't': t,
    }
    return repaired, stats

def make_repaired_odom_msg(src_ros_msg, repaired_enu):
    rep_msg = clone_ros_msg(src_ros_msg)
    set_odom_msg_from_enu(rep_msg, repaired_enu)
    return rep_msg

def build_quality_report(input_bag, output_bag, status, metrics=None, error=None):
    report = {
        'input_bag': input_bag,
        'output_bag': output_bag,
        'status': status,
        'ok': False,
        'reasons': [],
        'metrics': metrics or {},
        'error': error,
    }

    if status in OK_SKIP_STATUSES:
        report['ok'] = True
        return report

    if status != 'success':
        report['reasons'].append(status)
        return report

    repaired_odom = report['metrics'].get('repaired_odom', {})
    repaired_gnss = report['metrics'].get('repaired_gnss', {})
    alignment = report['metrics'].get('alignment', {})

    def check_metric(scope, data):
        reasons = []
        if data.get('count', 0) < 2:
            reasons.append(f'{scope}:not_enough_messages')
            return reasons
        if data.get('nonpositive_dt_count', 0) > 0:
            reasons.append(f'{scope}:nonpositive_dt={data["nonpositive_dt_count"]}')
        if data.get('nonfinite_count', 0) > 0:
            reasons.append(f'{scope}:nonfinite={data["nonfinite_count"]}')
        if data.get('jumps_gt_1m', 0) > QA_MAX_JUMPS_GT_1M:
            reasons.append(f'{scope}:jumps_gt_1m={data["jumps_gt_1m"]}')
        if data.get('jumps_gt_2m', 0) > QA_MAX_JUMPS_GT_2M:
            reasons.append(f'{scope}:jumps_gt_2m={data["jumps_gt_2m"]}')
        if data.get('step2d_max_m', 0.0) > QA_MAX_REPAIRED_STEP_2D_M:
            reasons.append(f'{scope}:max_step={data["step2d_max_m"]:.3f}m')
        if data.get('speed2d_max_mps', 0.0) > QA_MAX_REPAIRED_SPEED_2D_MPS:
            reasons.append(f'{scope}:max_speed={data["speed2d_max_mps"]:.1f}mps')
        return reasons

    report['reasons'].extend(check_metric('repaired_odom', repaired_odom))
    report['reasons'].extend(check_metric('repaired_gnss', repaired_gnss))

    matched_pairs = alignment.get('matched_pairs', 0)
    inlier_count = alignment.get('inlier_count', 0)
    inlier_ratio = alignment.get('inlier_ratio', 0.0)
    if matched_pairs < 2:
        report['reasons'].append(f'alignment:matched_pairs={matched_pairs}')
    if inlier_count < 2:
        report['reasons'].append(f'alignment:inliers={inlier_count}')
    if inlier_ratio < QA_MIN_ALIGNMENT_INLIER_RATIO:
        report['reasons'].append(f'alignment:inlier_ratio={inlier_ratio:.3f}')
    if alignment.get('mean_err_m', 0.0) > QA_MAX_ALIGNMENT_MEAN_ERR_M:
        report['reasons'].append(f'alignment:mean_err={alignment["mean_err_m"]:.3f}m')
    if alignment.get('max_err_m', 0.0) > QA_MAX_ALIGNMENT_MAX_ERR_M:
        report['reasons'].append(f'alignment:max_err={alignment["max_err_m"]:.3f}m')

    report['ok'] = len(report['reasons']) == 0
    return report

def flatten_report_for_csv(report):
    metrics = report.get('metrics', {})
    raw_odom = metrics.get('raw_odom', {})
    repaired_odom = metrics.get('repaired_odom', {})
    repaired_gnss = metrics.get('repaired_gnss', {})
    alignment = metrics.get('alignment', {})
    return {
        'ok': report.get('ok', False),
        'status': report.get('status', ''),
        'input_bag': report.get('input_bag', ''),
        'output_bag': report.get('output_bag', ''),
        'reasons': ';'.join(report.get('reasons', [])),
        'error': report.get('error') or '',
        'raw_odom_max_step_m': raw_odom.get('step2d_max_m', ''),
        'raw_odom_jumps_gt_2m': raw_odom.get('jumps_gt_2m', ''),
        'repaired_odom_max_step_m': repaired_odom.get('step2d_max_m', ''),
        'repaired_odom_jumps_gt_1m': repaired_odom.get('jumps_gt_1m', ''),
        'repaired_odom_jumps_gt_2m': repaired_odom.get('jumps_gt_2m', ''),
        'repaired_odom_max_speed_mps': repaired_odom.get('speed2d_max_mps', ''),
        'odom_repaired_transitions': metrics.get('odom_repaired_transitions', ''),
        'repaired_gnss_max_step_m': repaired_gnss.get('step2d_max_m', ''),
        'repaired_gnss_jumps_gt_1m': repaired_gnss.get('jumps_gt_1m', ''),
        'repaired_gnss_jumps_gt_2m': repaired_gnss.get('jumps_gt_2m', ''),
        'repaired_gnss_max_speed_mps': repaired_gnss.get('speed2d_max_mps', ''),
        'alignment_matched_pairs': alignment.get('matched_pairs', ''),
        'alignment_inlier_count': alignment.get('inlier_count', ''),
        'alignment_inlier_ratio': alignment.get('inlier_ratio', ''),
        'alignment_mean_err_m': alignment.get('mean_err_m', ''),
        'alignment_max_err_m': alignment.get('max_err_m', ''),
    }

def read_jsonl(path):
    records = []
    if not path or not os.path.exists(path):
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Ignoring malformed JSONL line {line_no} in {path}: {e}")
    return records

def append_jsonl(path, record):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()

def find_latest_manifest(report_dir):
    if not os.path.isdir(report_dir):
        return None
    candidates = [
        os.path.join(report_dir, name)
        for name in os.listdir(report_dir)
        if name.startswith('repair_manifest_') and name.endswith('.jsonl')
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def load_completed_inputs(manifest_path):
    completed = set()
    for record in read_jsonl(manifest_path):
        status = record.get('status')
        if status in {'success', 'skipped_output_exists', 'skipped_existing_output'}:
            input_bag = record.get('input_bag')
            if input_bag:
                completed.add(os.path.abspath(input_bag))
    return completed

def write_batch_reports(reports, output_dir, timestamp=None):
    timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'repair_quality_report_{timestamp}.json')
    bad_json_path = os.path.join(output_dir, f'repair_quality_bad_{timestamp}.json')
    bad_csv_path = os.path.join(output_dir, f'repair_quality_bad_{timestamp}.csv')

    bad_reports = [r for r in reports if not r.get('ok', False)]

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    with open(bad_json_path, 'w', encoding='utf-8') as f:
        json.dump(bad_reports, f, ensure_ascii=False, indent=2)

    rows = [flatten_report_for_csv(r) for r in bad_reports]
    fieldnames = list(flatten_report_for_csv({}).keys())
    with open(bad_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return report_path, bad_json_path, bad_csv_path, bad_reports

def write_batch_reports_from_jsonl(report_jsonl_path, output_dir, timestamp=None):
    reports = read_jsonl(report_jsonl_path)
    return write_batch_reports(reports, output_dir, timestamp)


def repair_single(input_bag, output_bag=None, overwrite=False):
    """Run post-processing repair on a single mcap bag file."""
    output_bag = output_bag or make_output_path(input_bag)

    print(f"\n{'=' * 87}")
    print(f"  Repairing: {input_bag}")
    print(f"  Output:    {output_bag}")
    print(f"{'=' * 87}")

    if os.path.exists(output_bag) and not overwrite:
        print("[SKIP] Output already exists. Use --overwrite to regenerate it.")
        return build_quality_report(input_bag, output_bag, 'skipped_output_exists')

    raw_gnss = []
    odom_data = []
    total_msg_count = 0

    navsat_schema_info = None
    navsat_template_msg = None

    for msg in read_ros2_messages(input_bag):
        total_msg_count += 1
        topic = msg.channel.topic
        t_sec = msg.publish_time.timestamp()
        ros_msg = msg.ros_msg

        if topic == SPP_FIX_RAW_TOPIC:
            raw_gnss.append({'t': t_sec, 'lat': ros_msg.latitude, 'lon': ros_msg.longitude, 'alt': ros_msg.altitude})
            if navsat_schema_info is None:
                navsat_schema_info = (msg.schema.name, msg.schema.data)
                navsat_template_msg = clone_ros_msg(ros_msg)
        elif topic == ODOM_TOPIC:
            enu_pos = odom_msg_to_enu(ros_msg)
            enu_twist = odom_twist_to_enu(ros_msg)
            odom_data.append({'t': t_sec, 'x': enu_pos[0], 'y': enu_pos[1], 'z': enu_pos[2], 'twist': enu_twist})

    print(f"Loaded {total_msg_count} total messages ({len(raw_gnss)} raw GNSS, {len(odom_data)} odometry).")

    if not raw_gnss:
        print("[WARN] No raw GNSS data found, skipping repair.")
        return build_quality_report(input_bag, output_bag, 'skipped_no_raw_gnss')
    if not odom_data:
        print("[WARN] No odometry data found, skipping repair.")
        return build_quality_report(input_bag, output_bag, 'skipped_no_odometry')

    repaired_odom_enu, odom_repair_stats = repair_odom_data(odom_data)
    odom_t_stats = odom_repair_stats['t']
    raw_odom_enu = odom_repair_stats['raw_pos']
    raw_odom_metrics = compute_step_metrics(odom_t_stats, raw_odom_enu)
    repaired_odom_metrics = compute_step_metrics(odom_t_stats, repaired_odom_enu)
    print_step_stats("Original Odometry", odom_t_stats, raw_odom_enu, raw_odom_metrics)
    print_step_stats("Repaired Odometry", odom_t_stats, repaired_odom_enu, repaired_odom_metrics)
    print(f"  Odom repaired transitions: {len(odom_repair_stats['bad_reasons'])}")
    if odom_repair_stats['bad_reasons']:
        print("  Top repaired odom transitions:")
        for idx0, idx1, reason, dt, step_2d, pose_speed, twist_speed in sorted(
            odom_repair_stats['bad_reasons'], key=lambda item: item[4], reverse=True
        )[:10]:
            print(f"    [{idx0}->{idx1}] {reason}: dt={dt:.4f}s step={step_2d:.3f}m pose_speed={pose_speed:.1f}m/s twist={twist_speed:.2f}m/s")

    odom_z0 = repaired_odom_enu[0, 2]

    # 1. Establish origin from first 10 stable GNSS points
    init_ecef = [geo2ecef([p['lat'], p['lon'], p['alt']]) for p in raw_gnss[:10]]
    origin_ecef = np.mean(init_ecef, axis=0)
    origin_lla = ecef2geo(origin_ecef)
    R_enu2ecef = geo2rotation(origin_lla)

    # Convert raw_gnss to local ENU
    raw_enu = []
    for p in raw_gnss:
        ecef = geo2ecef([p['lat'], p['lon'], p['alt']])
        enu = R_enu2ecef.T @ (ecef - origin_ecef)
        raw_enu.append({'t': p['t'], 'x': enu[0], 'y': enu[1], 'z': enu[2]})

    # 2. Match raw_gnss and odometry by timestamp
    odom_t = np.array([p['t'] for p in odom_data])
    odom_xy = repaired_odom_enu[:, :2]

    matched_odom = []
    matched_gnss = []
    matched_t = []
    for g in raw_enu:
        idx = np.argmin(np.abs(odom_t - g['t']))
        if np.abs(odom_t[idx] - g['t']) < 0.05:
            matched_odom.append(odom_xy[idx])
            matched_gnss.append([g['x'], g['y']])
            matched_t.append(g['t'])

    matched_odom = np.array(matched_odom).T
    matched_gnss = np.array(matched_gnss).T
    matched_t = np.array(matched_t)
    N = matched_odom.shape[1]

    print(f"Matched {N} odom-GNSS pairs for alignment.")
    if N < 2:
        print("[ERROR] Not enough matched odom-GNSS pairs for alignment.")
        return build_quality_report(
            input_bag,
            output_bag,
            'failed_not_enough_matches',
            {
                'raw_odom': raw_odom_metrics,
                'repaired_odom': repaired_odom_metrics,
                'odom_repaired_transitions': len(odom_repair_stats['bad_reasons']),
                'alignment': {'matched_pairs': int(N), 'inlier_count': 0, 'inlier_ratio': 0.0},
            },
        )

    # 3. Global RANSAC alignment (< 3.0m inlier threshold)
    #    Find the best overall rigid transform from odometry to GNSS ENU.
    #    This preserves the odometry's intrinsic smoothness.
    best_inliers = []
    np.random.seed(42)
    for _ in range(2000):
        idx = np.random.choice(N, 2, replace=False)
        p1_o, p2_o = matched_odom[:, idx[0]], matched_odom[:, idx[1]]
        p1_g, p2_g = matched_gnss[:, idx[0]], matched_gnss[:, idx[1]]

        do = p2_o - p1_o
        if np.linalg.norm(do) < 5.0:
            continue

        dg = p2_g - p1_g
        ang_diff = np.arctan2(dg[1], dg[0]) - np.arctan2(do[1], do[0])
        R = np.array([[np.cos(ang_diff), -np.sin(ang_diff)],
                      [np.sin(ang_diff),  np.cos(ang_diff)]])
        t = p1_g - R @ p1_o

        pred = R @ matched_odom + t.reshape(2, 1)
        err = np.linalg.norm(pred - matched_gnss, axis=0)
        inliers = np.where(err < 3.0)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    print(f"RANSAC identified {len(best_inliers)} / {N} global inliers.")
    if len(best_inliers) < 2:
        print("[ERROR] Not enough RANSAC inliers for alignment.")
        return build_quality_report(
            input_bag,
            output_bag,
            'failed_not_enough_inliers',
            {
                'raw_odom': raw_odom_metrics,
                'repaired_odom': repaired_odom_metrics,
                'odom_repaired_transitions': len(odom_repair_stats['bad_reasons']),
                'alignment': {
                    'matched_pairs': int(N),
                    'inlier_count': int(len(best_inliers)),
                    'inlier_ratio': float(len(best_inliers) / N if N else 0.0),
                },
            },
        )

    # Compute Umeyama similarity transform on inliers
    mean_o = np.mean(matched_odom[:, best_inliers], axis=1, keepdims=True)
    mean_g = np.mean(matched_gnss[:, best_inliers], axis=1, keepdims=True)
    centered_o = matched_odom[:, best_inliers] - mean_o
    centered_g = matched_gnss[:, best_inliers] - mean_g

    H = centered_o @ centered_g.T
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T
    t_opt = (mean_g - R_opt @ mean_o).flatten()

    print(f"Global transform: angle={np.degrees(np.arctan2(R_opt[1,0], R_opt[0,0])):.2f}°, "
          f"t=({t_opt[0]:.1f}, {t_opt[1]:.1f})")

    # 4. Compute time-varying residual correction
    #    The global transform gives good smoothness but has position bias that
    #    varies over time (due to odometry drift). We compute the residual at
    #    each matched GNSS point and smooth it to create a slowly-varying
    #    correction signal.
    global_pred = R_opt @ matched_odom + t_opt.reshape(2, 1)
    residuals_x = matched_gnss[0] - global_pred[0]  # GNSS - predicted
    residuals_y = matched_gnss[1] - global_pred[1]

    # Step 1: Median filter to reject GNSS outlier spikes
    def median_filter(arr, kernel=11):
        result = np.copy(arr)
        pad = kernel // 2
        for i in range(len(arr)):
            lo = max(0, i - pad)
            hi = min(len(arr), i + pad + 1)
            result[i] = np.median(arr[lo:hi])
        return result

    res_x_med = median_filter(residuals_x, kernel=21)
    res_y_med = median_filter(residuals_y, kernel=21)

    # Step 2: Moving average for additional smoothing
    def moving_average(arr, window=31):
        result = np.copy(arr)
        pad = window // 2
        for i in range(len(arr)):
            lo = max(0, i - pad)
            hi = min(len(arr), i + pad + 1)
            result[i] = np.mean(arr[lo:hi])
        return result

    res_x_smooth = moving_average(res_x_med, window=51)
    res_y_smooth = moving_average(res_y_med, window=51)

    # Build correction lookup: (time, correction_x, correction_y)
    correction_t = matched_t
    correction_x = res_x_smooth
    correction_y = res_y_smooth

    # Verify correction quality
    corrected_pred = global_pred + np.array([correction_x, correction_y])
    final_err = np.linalg.norm(corrected_pred - matched_gnss, axis=0)
    print(f"Residual correction: mean={np.mean(np.sqrt(correction_x**2 + correction_y**2)):.2f}m")
    print(f"After correction: mean_err={np.mean(final_err):.2f}m, max_err={np.max(final_err):.2f}m")

    def get_correction(t_query):
        """Interpolate the smoothed residual correction at a given time."""
        if t_query <= correction_t[0]:
            return np.array([correction_x[0], correction_y[0]])
        if t_query >= correction_t[-1]:
            return np.array([correction_x[-1], correction_y[-1]])

        idx = np.searchsorted(correction_t, t_query) - 1
        idx = max(0, min(idx, len(correction_t) - 2))
        alpha = (t_query - correction_t[idx]) / (correction_t[idx + 1] - correction_t[idx])
        cx = (1 - alpha) * correction_x[idx] + alpha * correction_x[idx + 1]
        cy = (1 - alpha) * correction_y[idx] + alpha * correction_y[idx + 1]
        return np.array([cx, cy])

    print("Hybrid alignment (global + residual correction) computed successfully.")
    print(f"Writing repaired bag to: {output_bag}")

    temp_output_bag = f"{output_bag}.download"
    if os.path.exists(temp_output_bag):
        os.remove(temp_output_bag)
    if os.path.exists(output_bag):
        os.remove(output_bag)
    os.makedirs(os.path.dirname(os.path.abspath(output_bag)), exist_ok=True)

    writer = Writer(temp_output_bag)
    schemas = {}

    try:
        if navsat_schema_info is not None and navsat_schema_info[0] not in schemas:
            schemas[navsat_schema_info[0]] = writer.register_msgdef(navsat_schema_info[0], navsat_schema_info[1].decode())

        repaired_trajectory = []
        odom_write_idx = 0
        odom_fix_idx = 0

        for msg in read_ros2_messages(input_bag):
            topic = msg.channel.topic
            s_name = msg.schema.name
            if s_name not in schemas:
                schemas[s_name] = writer.register_msgdef(s_name, msg.schema.data.decode())

            ros_msg = msg.ros_msg
            log_ns = msg.log_time_ns
            pub_ns = msg.publish_time_ns

            # If original spp_fix, rename to _orig so user can compare
            if topic == SPP_FIX_TOPIC:
                writer.write_message(SPP_FIX_ORIG_TOPIC, schemas[s_name], ros_msg, log_ns, pub_ns)
            elif topic == ODOM_TOPIC:
                if odom_write_idx < len(repaired_odom_enu):
                    repaired_odom_msg = make_repaired_odom_msg(ros_msg, repaired_odom_enu[odom_write_idx])
                    writer.write_message(ODOM_ORIG_TOPIC, schemas[s_name], ros_msg, log_ns, pub_ns)
                    writer.write_message(ODOM_TOPIC, schemas[s_name], repaired_odom_msg, log_ns, pub_ns)
                    writer.write_message(ODOM_REPAIRED_TOPIC, schemas[s_name], repaired_odom_msg, log_ns, pub_ns)
                    odom_write_idx += 1
                else:
                    writer.write_message(topic, schemas[s_name], ros_msg, log_ns, pub_ns)
            else:
                # All other topics (including non-target topics) are written as-is
                writer.write_message(topic, schemas[s_name], ros_msg, log_ns, pub_ns)

            # When we encounter odometry, generate and publish the repaired spp_fix
            if topic == ODOM_TOPIC and navsat_template_msg is not None and odom_fix_idx < len(repaired_odom_enu):
                t_sec = pub_ns / 1e9
                oxy = repaired_odom_enu[odom_fix_idx, :2]
                oz = repaired_odom_enu[odom_fix_idx, 2]
                odom_fix_idx += 1

                # Apply global transform + smoothed residual correction
                gxy = R_opt @ oxy + t_opt + get_correction(t_sec)
                fused_enu = np.array([gxy[0], gxy[1], oz - odom_z0])

                cur_ecef = origin_ecef + R_enu2ecef @ fused_enu
                cur_lla = ecef2geo(cur_ecef)

                repaired_trajectory.append({'t': t_sec, 'lat': cur_lla[0], 'lon': cur_lla[1], 'alt': cur_lla[2]})

                # Construct repaired NavSatFix message
                rep_msg = type(navsat_template_msg)()
                for field in navsat_template_msg.__slots__:
                    setattr(rep_msg, field, getattr(navsat_template_msg, field))

                rep_msg.latitude = float(cur_lla[0])
                rep_msg.longitude = float(cur_lla[1])
                rep_msg.altitude = float(cur_lla[2])
                rep_msg.header = ros_msg.header

                navsat_schema = schemas[navsat_schema_info[0]]
                writer.write_message(SPP_FIX_TOPIC, navsat_schema, rep_msg, log_ns, pub_ns)
                writer.write_message(SPP_FIX_REPAIRED_TOPIC, navsat_schema, rep_msg, log_ns, pub_ns)

        writer.finish()
    except BaseException:
        if os.path.exists(temp_output_bag):
            try:
                os.remove(temp_output_bag)
            except OSError:
                pass
        raise
    else:
        if os.path.exists(output_bag):
            os.remove(output_bag)
        os.rename(temp_output_bag, output_bag)
    print("=======================================================================================")
    print("                     POST-PROCESSING REPAIR SUCCESSFULLY EXPORTED                      ")
    print("=======================================================================================")
    print(f"Output bag file: {output_bag}")
    print(f"Topics in repaired bag:")
    print(f"  - /baton/gnss/spp_fix          : Repaired smooth GPS trajectory generated from repaired odometry")
    print(f"  - /baton/gnss/spp_fix_repaired : Alias of repaired trajectory")
    print(f"  - /baton/gnss/spp_fix_orig     : Original un-repaired online trajectory (for comparison)")
    print(f"  - /baton/gnss/spp_fix_raw      : Original raw SPP定位数据")
    print(f"  - /baton/stereo3/odometry      : Repaired odometry")
    print(f"  - /baton/stereo3/odometry_repaired : Alias of repaired odometry")
    print(f"  - /baton/stereo3/odometry_orig : Original odometry (for comparison)")
    print(f"  - (all other original topics)  : Preserved as-is")

    # Verify statistics of repaired trajectory
    t = np.array([r['t'] for r in repaired_trajectory])
    lat = np.array([r['lat'] for r in repaired_trajectory])
    lon = np.array([r['lon'] for r in repaired_trajectory])
    mean_lat_rad = np.radians(np.mean(lat))
    R_earth = 6378137.0
    dx = np.radians(np.diff(lon)) * R_earth * np.cos(mean_lat_rad)
    dy = np.radians(np.diff(lat)) * R_earth
    dist_2d = np.sqrt(dx**2 + dy**2)

    print(f"\n[Repaired Trajectory Verification] ({len(repaired_trajectory)} msgs, {1.0/np.mean(np.diff(t)):.1f} Hz):")
    print(f"  2D Step Dist (m) => Mean: {np.mean(dist_2d):.4f} | Max: {np.max(dist_2d):.4f} | 95%: {np.percentile(dist_2d, 95):.4f}")
    print(f"  Jump Counts      => >0.5m: {np.sum(dist_2d > 0.5)} | >1.0m: {np.sum(dist_2d > 1.0)} | >2.0m: {np.sum(dist_2d > 2.0)} | >5.0m: {np.sum(dist_2d > 5.0)}")

    repaired_gnss_metrics = compute_navsat_metrics(repaired_trajectory)
    metrics = {
        'raw_odom': raw_odom_metrics,
        'repaired_odom': repaired_odom_metrics,
        'repaired_gnss': repaired_gnss_metrics,
        'odom_repaired_transitions': len(odom_repair_stats['bad_reasons']),
        'alignment': {
            'matched_pairs': int(N),
            'inlier_count': int(len(best_inliers)),
            'inlier_ratio': float(len(best_inliers) / N if N else 0.0),
            'mean_err_m': float(np.mean(final_err)),
            'max_err_m': float(np.max(final_err)),
            'residual_correction_mean_m': float(np.mean(np.sqrt(correction_x**2 + correction_y**2))),
        },
    }
    report = build_quality_report(input_bag, output_bag, 'success', metrics)
    if report['ok']:
        print("\n[Quality Check] OK")
    else:
        print("\n[Quality Check] NEEDS REVIEW")
        for reason in report['reasons']:
            print(f"  - {reason}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Post-process MCAP bag files to repair GNSS trajectory using odometry. '
                    'Supports single file or batch directory processing.'
    )
    parser.add_argument(
        'path', nargs='?', default='.',
        help='Path to a single .mcap file or a directory containing .mcap files. '
             'Defaults to current directory. Files ending with _repaired.mcap are excluded.'
    )
    parser.add_argument(
        '--report-dir',
        default=None,
        help='Directory for batch quality reports. Defaults to the input directory.'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for repaired MCAP files. For directory input, relative subdirectories are preserved.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing repaired MCAP files. By default existing outputs are skipped.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the manifest JSONL by skipping inputs previously marked success or skipped.'
    )
    parser.add_argument(
        '--manifest',
        default=None,
        help='Path to progress manifest JSONL. Defaults to <report-dir>/repair_manifest_<timestamp>.jsonl.'
    )
    parser.add_argument(
        '--report-jsonl',
        default=None,
        help='Path to per-file quality report JSONL. Defaults to <report-dir>/repair_quality_report_<timestamp>.jsonl.'
    )
    args = parser.parse_args()

    mcap_files = discover_mcap_files(args.path)
    if not mcap_files:
        print("[ERROR] No .mcap files found.")
        return

    input_root = infer_input_root(args.path)
    if args.report_dir is not None:
        report_dir = args.report_dir
    elif args.output_dir is not None:
        report_dir = args.output_dir
    elif os.path.isdir(args.path):
        report_dir = args.path
    else:
        report_dir = os.path.dirname(os.path.abspath(mcap_files[0])) or '.'
    os.makedirs(report_dir, exist_ok=True)

    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.manifest:
        manifest_path = args.manifest
    elif args.resume:
        manifest_path = find_latest_manifest(report_dir) or os.path.join(report_dir, f'repair_manifest_{batch_timestamp}.jsonl')
    else:
        manifest_path = os.path.join(report_dir, f'repair_manifest_{batch_timestamp}.jsonl')
    report_jsonl_path = args.report_jsonl or os.path.join(report_dir, f'repair_quality_report_{batch_timestamp}.jsonl')

    print(f"Found {len(mcap_files)} mcap file(s) to process:")
    for f in mcap_files:
        print(f"  - {f}")
    print(f"Output dir: {args.output_dir or '[input file directory]'}")
    print(f"Report dir: {report_dir}")
    print(f"Manifest JSONL: {manifest_path}")
    print(f"Quality JSONL : {report_jsonl_path}")

    completed_inputs = load_completed_inputs(manifest_path) if args.resume else set()
    if completed_inputs:
        print(f"Resume enabled: {len(completed_inputs)} completed input(s) loaded from manifest.")

    processed_count = 0
    skipped_count = 0
    ok_count = 0
    bad_count = 0
    failed_count = 0

    for idx, bag_path in enumerate(mcap_files, 1):
        output_bag = make_output_path(bag_path, args.output_dir, input_root)
        print(f"\n[{idx}/{len(mcap_files)}] {bag_path}")

        if args.resume and os.path.abspath(bag_path) in completed_inputs:
            print("[SKIP] Already completed according to manifest.")
            skipped_count += 1
            continue

        if os.path.exists(output_bag) and not args.overwrite:
            print("[SKIP] Output already exists. Use --overwrite to regenerate it.")
            report = build_quality_report(bag_path, output_bag, 'skipped_existing_output')
            append_jsonl(manifest_path, report)
            append_jsonl(report_jsonl_path, report)
            processed_count += 1
            skipped_count += 1
            ok_count += 1
            continue

        try:
            report = repair_single(bag_path, output_bag, overwrite=args.overwrite)
        except Exception as e:
            print(f"\n[ERROR] Failed to repair {bag_path}: {e}")
            traceback.print_exc()
            report = build_quality_report(
                bag_path,
                output_bag,
                'exception',
                error=f'{type(e).__name__}: {e}\n{traceback.format_exc()}',
            )

        append_jsonl(manifest_path, report)
        append_jsonl(report_jsonl_path, report)
        processed_count += 1
        if report.get('ok', False):
            ok_count += 1
        else:
            bad_count += 1
            if report.get('status') == 'exception':
                failed_count += 1

        print(f"[Progress] processed={processed_count} ok={ok_count} needs_review={bad_count} skipped={skipped_count} failed={failed_count}")

    report_path, bad_json_path, bad_csv_path, bad_reports = write_batch_reports_from_jsonl(
        report_jsonl_path,
        report_dir,
        batch_timestamp,
    )

    print(f"\n{'=' * 87}")
    print(f"  Batch processing complete. {len(mcap_files)} file(s) discovered.")
    print(f"  Processed this run: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  OK: {ok_count}")
    print(f"  Needs review: {bad_count}")
    print(f"  Failed exceptions: {failed_count}")
    print(f"  Manifest JSONL: {manifest_path}")
    print(f"  Quality JSONL : {report_jsonl_path}")
    print(f"  Quality report: {report_path}")
    print(f"  Needs-review JSON: {bad_json_path}")
    print(f"  Needs-review CSV : {bad_csv_path}")
    print(f"  Needs-review bags: {len(bad_reports)}")
    if bad_reports:
        print("  Files needing review:")
        for report in bad_reports:
            print(f"    - {report['input_bag']}")
            for reason in report.get('reasons', []):
                print(f"      * {reason}")
    print(f"{'=' * 87}")


if __name__ == '__main__':
    main()
