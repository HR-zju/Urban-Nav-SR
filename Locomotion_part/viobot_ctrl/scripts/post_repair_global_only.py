import argparse
import csv
import json
import os
import traceback
from datetime import datetime

import numpy as np
from mcap_ros2.reader import read_ros2_messages
from mcap_ros2.writer import Writer

from post_repair import (
    ODOM_ORIG_TOPIC,
    ODOM_REPAIRED_TOPIC,
    ODOM_TOPIC,
    SPP_FIX_ORIG_TOPIC,
    SPP_FIX_RAW_TOPIC,
    SPP_FIX_REPAIRED_TOPIC,
    SPP_FIX_TOPIC,
    append_jsonl,
    build_quality_report,
    clone_ros_msg,
    compute_navsat_metrics,
    compute_step_metrics,
    discover_mcap_files,
    ecef2geo,
    geo2ecef,
    geo2rotation,
    infer_input_root,
    load_completed_inputs,
    make_repaired_odom_msg,
    odom_msg_to_enu,
    odom_twist_to_enu,
    print_step_stats,
    read_jsonl,
    repair_odom_data,
)

STATIC_ODOM_SPEED_MPS = 0.05


def make_global_output_path(input_path, output_dir=None, input_root=None):
    base_name = os.path.basename(input_path)
    base, ext = os.path.splitext(base_name)
    repaired_name = f"{base}_global_repaired{ext}"

    if output_dir is None:
        return os.path.join(os.path.dirname(input_path), repaired_name)

    if input_root is not None:
        rel_dir = os.path.dirname(os.path.relpath(os.path.abspath(input_path), os.path.abspath(input_root)))
        if rel_dir == ".":
            rel_dir = ""
    else:
        rel_dir = ""
    return os.path.join(output_dir, rel_dir, repaired_name)


def find_latest_global_manifest(search_dir):
    if not os.path.isdir(search_dir):
        return None
    candidates = []
    for root, _, filenames in os.walk(search_dir):
        for name in filenames:
            if name.startswith("global_only_manifest_") and name.endswith(".jsonl"):
                candidates.append(os.path.join(root, name))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def find_latest_global_report_jsonl(search_dir):
    if not os.path.isdir(search_dir):
        return None
    candidates = [
        os.path.join(search_dir, name)
        for name in os.listdir(search_dir)
        if name.startswith("global_only_repair_report_") and name.endswith(".jsonl")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def flatten_global_report_for_csv(report):
    metrics = report.get("metrics", {})
    raw_odom = metrics.get("raw_odom", {})
    repaired_odom = metrics.get("repaired_odom", {})
    repaired_gnss = metrics.get("repaired_gnss", {})
    alignment = metrics.get("alignment", {})
    return {
        "ok": report.get("ok", False),
        "status": report.get("status", ""),
        "input_bag": report.get("input_bag", ""),
        "output_bag": report.get("output_bag", ""),
        "reasons": ";".join(report.get("reasons", [])),
        "error": report.get("error") or "",
        "odom_source": alignment.get("odom_source", ""),
        "raw_odom_max_step_m": raw_odom.get("step2d_max_m", ""),
        "raw_odom_jumps_gt_1m": raw_odom.get("jumps_gt_1m", ""),
        "raw_odom_jumps_gt_2m": raw_odom.get("jumps_gt_2m", ""),
        "raw_odom_max_speed_mps": raw_odom.get("speed2d_max_mps", ""),
        "output_odom_max_step_m": repaired_odom.get("step2d_max_m", ""),
        "output_odom_jumps_gt_1m": repaired_odom.get("jumps_gt_1m", ""),
        "output_odom_jumps_gt_2m": repaired_odom.get("jumps_gt_2m", ""),
        "output_odom_max_speed_mps": repaired_odom.get("speed2d_max_mps", ""),
        "repaired_gnss_max_step_m": repaired_gnss.get("step2d_max_m", ""),
        "repaired_gnss_jumps_gt_1m": repaired_gnss.get("jumps_gt_1m", ""),
        "repaired_gnss_jumps_gt_2m": repaired_gnss.get("jumps_gt_2m", ""),
        "repaired_gnss_max_speed_mps": repaired_gnss.get("speed2d_max_mps", ""),
        "alignment_total_matched_pairs": alignment.get("total_matched_pairs", ""),
        "alignment_static_filtered_pairs": alignment.get("static_filtered_pairs", ""),
        "alignment_static_speed_threshold_mps": alignment.get("static_speed_threshold_mps", ""),
        "alignment_matched_pairs": alignment.get("matched_pairs", ""),
        "alignment_inlier_count": alignment.get("inlier_count", ""),
        "alignment_inlier_ratio": alignment.get("inlier_ratio", ""),
        "alignment_mean_err_m": alignment.get("mean_err_m", ""),
        "alignment_max_err_m": alignment.get("max_err_m", ""),
        "alignment_all_mean_err_m": alignment.get("all_mean_err_m", ""),
        "alignment_all_max_err_m": alignment.get("all_max_err_m", ""),
    }


def write_global_reports_from_jsonl(report_jsonl_path, output_dir, timestamp):
    reports = read_jsonl(report_jsonl_path)
    report_path = os.path.join(output_dir, f"global_only_repair_report_{timestamp}.json")
    recheck_json_path = os.path.join(output_dir, f"global_only_recheck_{timestamp}.json")
    recheck_list_path = os.path.join(output_dir, f"global_only_recheck_{timestamp}.txt")
    recheck_csv_path = os.path.join(output_dir, f"global_only_recheck_{timestamp}.csv")

    recheck_reports = [report for report in reports if not report.get("ok", False)]

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    with open(recheck_json_path, "w", encoding="utf-8") as f:
        json.dump(recheck_reports, f, ensure_ascii=False, indent=2)
    with open(recheck_list_path, "w", encoding="utf-8") as f:
        for report in recheck_reports:
            reasons = ";".join(report.get("reasons", []))
            f.write(f"{report.get('input_bag', '')}\t{reasons}\n")

    rows = [flatten_global_report_for_csv(report) for report in recheck_reports]
    fieldnames = list(flatten_global_report_for_csv({}).keys())
    with open(recheck_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return report_path, recheck_json_path, recheck_list_path, recheck_csv_path, recheck_reports


def compute_odom_motion_speed(t, xyz):
    t = np.asarray(t, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    n = len(t)
    motion_speed = np.zeros(n, dtype=float)
    if n < 2:
        return motion_speed

    dt = np.diff(t)
    dxy = np.diff(xyz[:, :2], axis=0)
    valid = np.isfinite(dt) & (dt > 0.0) & np.all(np.isfinite(dxy), axis=1)
    segment_speed = np.zeros(n - 1, dtype=float)
    segment_speed[valid] = np.linalg.norm(dxy[valid], axis=1) / dt[valid]

    motion_speed[:-1] = np.maximum(motion_speed[:-1], segment_speed)
    motion_speed[1:] = np.maximum(motion_speed[1:], segment_speed)

    return motion_speed


def compute_global_alignment(matched_odom, matched_gnss):
    n = matched_odom.shape[1]
    best_inliers = []
    np.random.seed(42)

    for _ in range(2000):
        idx = np.random.choice(n, 2, replace=False)
        p1_o, p2_o = matched_odom[:, idx[0]], matched_odom[:, idx[1]]
        p1_g, p2_g = matched_gnss[:, idx[0]], matched_gnss[:, idx[1]]

        do = p2_o - p1_o
        if np.linalg.norm(do) < 5.0:
            continue

        dg = p2_g - p1_g
        ang_diff = np.arctan2(dg[1], dg[0]) - np.arctan2(do[1], do[0])
        r = np.array(
            [
                [np.cos(ang_diff), -np.sin(ang_diff)],
                [np.sin(ang_diff), np.cos(ang_diff)],
            ]
        )
        t = p1_g - r @ p1_o

        pred = r @ matched_odom + t.reshape(2, 1)
        err = np.linalg.norm(pred - matched_gnss, axis=0)
        inliers = np.where(err < 3.0)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    best_inliers = np.asarray(best_inliers, dtype=int)
    if len(best_inliers) < 2:
        return None, None, best_inliers, None, None

    mean_o = np.mean(matched_odom[:, best_inliers], axis=1, keepdims=True)
    mean_g = np.mean(matched_gnss[:, best_inliers], axis=1, keepdims=True)
    centered_o = matched_odom[:, best_inliers] - mean_o
    centered_g = matched_gnss[:, best_inliers] - mean_g

    h = centered_o @ centered_g.T
    u, _, vt = np.linalg.svd(h)
    r_opt = vt.T @ u.T
    if np.linalg.det(r_opt) < 0:
        vt[-1, :] *= -1
        r_opt = vt.T @ u.T
    t_opt = (mean_g - r_opt @ mean_o).flatten()

    pred = r_opt @ matched_odom + t_opt.reshape(2, 1)
    err_all = np.linalg.norm(pred - matched_gnss, axis=0)
    err_inliers = err_all[best_inliers]
    return r_opt, t_opt, best_inliers, err_all, err_inliers


def repair_single_global(input_bag, output_bag=None, overwrite=False, odom_source="raw"):
    output_bag = output_bag or make_global_output_path(input_bag)

    print(f"\n{'=' * 87}")
    print(f"  Global-only repairing: {input_bag}")
    print(f"  Output:                {output_bag}")
    print(f"{'=' * 87}")

    if os.path.exists(output_bag) and not overwrite:
        print("[SKIP] Output already exists. Use --overwrite to regenerate it.")
        return build_quality_report(input_bag, output_bag, "skipped_output_exists")

    raw_gnss = []
    odom_data = []
    total_msg_count = 0
    navsat_schema_info = None
    navsat_template_msg = None

    for msg in read_ros2_messages(input_bag):
        total_msg_count += 1
        topic = msg.channel.topic
        t_sec = msg.publish_time_ns / 1e9
        ros_msg = msg.ros_msg

        if topic == SPP_FIX_RAW_TOPIC:
            raw_gnss.append({"t": t_sec, "lat": ros_msg.latitude, "lon": ros_msg.longitude, "alt": ros_msg.altitude})
            if navsat_schema_info is None:
                navsat_schema_info = (msg.schema.name, msg.schema.data)
                navsat_template_msg = clone_ros_msg(ros_msg)
        elif topic == ODOM_TOPIC:
            enu_pos = odom_msg_to_enu(ros_msg)
            enu_twist = odom_twist_to_enu(ros_msg)
            odom_data.append(
                {"t": t_sec, "x": enu_pos[0], "y": enu_pos[1], "z": enu_pos[2], "twist": enu_twist}
            )

    print(f"Loaded {total_msg_count} total messages ({len(raw_gnss)} raw GNSS, {len(odom_data)} odometry).")

    if not raw_gnss:
        print("[WARN] No raw GNSS data found, skipping repair.")
        return build_quality_report(input_bag, output_bag, "skipped_no_raw_gnss")
    if not odom_data:
        print("[WARN] No odometry data found, skipping repair.")
        return build_quality_report(input_bag, output_bag, "skipped_no_odometry")

    repaired_odom_enu, odom_repair_stats = repair_odom_data(odom_data)
    odom_t_stats = odom_repair_stats["t"]
    raw_odom_enu = odom_repair_stats["raw_pos"]
    raw_odom_metrics = compute_step_metrics(odom_t_stats, raw_odom_enu)
    repaired_odom_metrics = compute_step_metrics(odom_t_stats, repaired_odom_enu)
    print_step_stats("Original Odometry", odom_t_stats, raw_odom_enu, raw_odom_metrics)
    print_step_stats("Repaired Odometry", odom_t_stats, repaired_odom_enu, repaired_odom_metrics)
    print(f"  Odom repaired transitions: {len(odom_repair_stats['bad_reasons'])}")

    if odom_source == "repaired":
        output_odom_enu = repaired_odom_enu
        output_odom_metrics = repaired_odom_metrics
        print("Odom shape source: repaired odometry from repair_odom_data().")
    else:
        output_odom_enu = raw_odom_enu
        output_odom_metrics = raw_odom_metrics
        print("Odom shape source: raw odometry, preserving original shape.")

    init_ecef = [geo2ecef([p["lat"], p["lon"], p["alt"]]) for p in raw_gnss[:10]]
    origin_ecef = np.mean(init_ecef, axis=0)
    origin_lla = ecef2geo(origin_ecef)
    r_enu2ecef = geo2rotation(origin_lla)

    raw_enu = []
    for p in raw_gnss:
        ecef = geo2ecef([p["lat"], p["lon"], p["alt"]])
        enu = r_enu2ecef.T @ (ecef - origin_ecef)
        raw_enu.append({"t": p["t"], "x": enu[0], "y": enu[1], "z": enu[2]})

    odom_t = np.array([p["t"] for p in odom_data])
    odom_xy = output_odom_enu[:, :2]
    odom_motion_speed = compute_odom_motion_speed(odom_t, output_odom_enu)
    matched_odom = []
    matched_gnss = []
    matched_odom_indices = []

    for g in raw_enu:
        idx = np.argmin(np.abs(odom_t - g["t"]))
        if np.abs(odom_t[idx] - g["t"]) < 0.05:
            matched_odom.append(odom_xy[idx])
            matched_gnss.append([g["x"], g["y"]])
            matched_odom_indices.append(idx)

    total_matched_pairs = len(matched_odom)
    matched_odom = np.asarray(matched_odom, dtype=float)
    matched_gnss = np.asarray(matched_gnss, dtype=float)
    matched_odom_indices = np.asarray(matched_odom_indices, dtype=int)

    if total_matched_pairs:
        moving_mask = odom_motion_speed[matched_odom_indices] >= STATIC_ODOM_SPEED_MPS
        static_filtered_pairs = int(total_matched_pairs - np.sum(moving_mask))
        matched_odom = matched_odom[moving_mask].T
        matched_gnss = matched_gnss[moving_mask].T
    else:
        static_filtered_pairs = 0
        matched_odom = np.empty((2, 0), dtype=float)
        matched_gnss = np.empty((2, 0), dtype=float)
    n = matched_odom.shape[1]

    print(
        f"Matched {total_matched_pairs} odom-GNSS pairs; kept {n} moving pairs for global alignment "
        f"(filtered {static_filtered_pairs} static, speed < {STATIC_ODOM_SPEED_MPS:.2f} m/s)."
    )
    if n < 2:
        print("[ERROR] Not enough moving matched odom-GNSS pairs for alignment.")
        return build_quality_report(
            input_bag,
            output_bag,
            "failed_not_enough_moving_matches",
            {
                "raw_odom": raw_odom_metrics,
                "repaired_odom": repaired_odom_metrics,
                "odom_repaired_transitions": len(odom_repair_stats["bad_reasons"]),
                "alignment": {
                    "total_matched_pairs": int(total_matched_pairs),
                    "static_filtered_pairs": int(static_filtered_pairs),
                    "static_speed_threshold_mps": float(STATIC_ODOM_SPEED_MPS),
                    "matched_pairs": int(n),
                    "inlier_count": 0,
                    "inlier_ratio": 0.0,
                    "mode": "global_only",
                    "odom_source": odom_source,
                },
            },
        )

    r_opt, t_opt, best_inliers, err_all, err_inliers = compute_global_alignment(matched_odom, matched_gnss)
    print(f"RANSAC identified {len(best_inliers)} / {n} global inliers.")
    if r_opt is None:
        print("[ERROR] Not enough RANSAC inliers for alignment.")
        return build_quality_report(
            input_bag,
            output_bag,
            "failed_not_enough_inliers",
            {
                "raw_odom": raw_odom_metrics,
                "repaired_odom": repaired_odom_metrics,
                "odom_repaired_transitions": len(odom_repair_stats["bad_reasons"]),
                "alignment": {
                    "total_matched_pairs": int(total_matched_pairs),
                    "static_filtered_pairs": int(static_filtered_pairs),
                    "static_speed_threshold_mps": float(STATIC_ODOM_SPEED_MPS),
                    "matched_pairs": int(n),
                    "inlier_count": int(len(best_inliers)),
                    "inlier_ratio": float(len(best_inliers) / n if n else 0.0),
                    "mode": "global_only",
                    "odom_source": odom_source,
                },
            },
        )

    print(
        f"Global transform: angle={np.degrees(np.arctan2(r_opt[1, 0], r_opt[0, 0])):.2f} deg, "
        f"t=({t_opt[0]:.1f}, {t_opt[1]:.1f})"
    )
    print(f"Global-only alignment error on inliers: mean={np.mean(err_inliers):.2f}m, max={np.max(err_inliers):.2f}m")
    print(f"Global-only alignment error on all matches: mean={np.mean(err_all):.2f}m, max={np.max(err_all):.2f}m")
    print("Altitude is intentionally held at local ENU z=0 in this global-only script.")

    print(f"Writing global-only repaired bag to: {output_bag}")
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

            if topic == SPP_FIX_TOPIC:
                writer.write_message(SPP_FIX_ORIG_TOPIC, schemas[s_name], ros_msg, log_ns, pub_ns)
            elif topic == ODOM_TOPIC:
                if odom_write_idx < len(output_odom_enu):
                    repaired_odom_msg = make_repaired_odom_msg(ros_msg, output_odom_enu[odom_write_idx])
                    writer.write_message(ODOM_ORIG_TOPIC, schemas[s_name], ros_msg, log_ns, pub_ns)
                    writer.write_message(ODOM_TOPIC, schemas[s_name], repaired_odom_msg, log_ns, pub_ns)
                    writer.write_message(ODOM_REPAIRED_TOPIC, schemas[s_name], repaired_odom_msg, log_ns, pub_ns)
                    odom_write_idx += 1
                else:
                    writer.write_message(topic, schemas[s_name], ros_msg, log_ns, pub_ns)
            else:
                writer.write_message(topic, schemas[s_name], ros_msg, log_ns, pub_ns)

            if topic == ODOM_TOPIC and navsat_template_msg is not None and odom_fix_idx < len(output_odom_enu):
                t_sec = pub_ns / 1e9
                oxy = output_odom_enu[odom_fix_idx, :2]
                odom_fix_idx += 1

                gxy = r_opt @ oxy + t_opt
                fused_enu = np.array([gxy[0], gxy[1], 0.0])
                cur_ecef = origin_ecef + r_enu2ecef @ fused_enu
                cur_lla = ecef2geo(cur_ecef)
                repaired_trajectory.append({"t": t_sec, "lat": cur_lla[0], "lon": cur_lla[1], "alt": cur_lla[2]})

                rep_msg = clone_ros_msg(navsat_template_msg)
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
    print("                         GLOBAL-ONLY REPAIR SUCCESSFULLY EXPORTED                      ")
    print("=======================================================================================")
    print(f"Output bag file: {output_bag}")
    print("Topics in repaired bag:")
    print("  - /baton/gnss/spp_fix              : Global-only trajectory from repaired odometry")
    print("  - /baton/gnss/spp_fix_repaired     : Alias of global-only trajectory")
    print("  - /baton/gnss/spp_fix_orig         : Original online trajectory")
    print("  - /baton/gnss/spp_fix_raw          : Original raw SPP data")
    print("  - /baton/stereo3/odometry          : Repaired odometry")
    print("  - /baton/stereo3/odometry_repaired : Alias of repaired odometry")
    print("  - /baton/stereo3/odometry_orig     : Original odometry")

    repaired_gnss_metrics = compute_navsat_metrics(repaired_trajectory)
    print_step_stats(
        "Global-only Repaired Trajectory",
        [r["t"] for r in repaired_trajectory],
        np.column_stack(
            [
                np.array([r["lon"] for r in repaired_trajectory]),
                np.array([r["lat"] for r in repaired_trajectory]),
                np.array([r["alt"] for r in repaired_trajectory]),
            ]
        ),
        repaired_gnss_metrics,
    )

    metrics = {
        "raw_odom": raw_odom_metrics,
        "repaired_odom": output_odom_metrics,
        "repaired_gnss": repaired_gnss_metrics,
        "odom_repaired_transitions": len(odom_repair_stats["bad_reasons"]),
        "alignment": {
            "total_matched_pairs": int(total_matched_pairs),
            "static_filtered_pairs": int(static_filtered_pairs),
            "static_speed_threshold_mps": float(STATIC_ODOM_SPEED_MPS),
            "matched_pairs": int(n),
            "inlier_count": int(len(best_inliers)),
            "inlier_ratio": float(len(best_inliers) / n if n else 0.0),
            "mean_err_m": float(np.mean(err_inliers)),
            "max_err_m": float(np.max(err_inliers)),
            "all_mean_err_m": float(np.mean(err_all)),
            "all_max_err_m": float(np.max(err_all)),
            "mode": "global_only",
            "odom_source": odom_source,
        },
    }
    report = build_quality_report(input_bag, output_bag, "success", metrics)
    if report["ok"]:
        print("\n[Quality Check] OK")
    else:
        print("\n[Quality Check] NEEDS REVIEW")
        for reason in report["reasons"]:
            print(f"  - {reason}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Create global-only repaired MCAP files. GNSS is used only for one rigid XY transform."
    )
    parser.add_argument("path", nargs="?", default=".", help="Path to a single .mcap file or a directory.")
    parser.add_argument("--output-dir", default=None, help="Directory for global-only repaired MCAP files.")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Base directory for timestamped report folders. Defaults to output/input dir.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing global-only output files.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the global-only manifest JSONL by skipping inputs previously marked success or skipped.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to progress manifest JSONL. Defaults to <report-dir>/global_only_manifest_<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--report-jsonl",
        default=None,
        help="Path to per-file quality report JSONL. Defaults to <report-dir>/global_only_repair_report_<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--odom-source",
        choices=("raw", "repaired"),
        default="raw",
        help="Odometry shape source for global-only output. Defaults to raw to preserve the original odom shape.",
    )
    args = parser.parse_args()

    mcap_files = discover_mcap_files(args.path)
    if not mcap_files:
        print("[ERROR] No .mcap files found.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_root = infer_input_root(args.path)
    if args.output_dir is not None:
        output_base_dir = args.output_dir
    elif os.path.isdir(args.path):
        output_base_dir = args.path
    else:
        output_base_dir = os.path.dirname(os.path.abspath(mcap_files[0])) or "."

    if args.report_dir is not None:
        report_base_dir = args.report_dir
    elif args.output_dir is not None:
        report_base_dir = args.output_dir
    elif os.path.isdir(args.path):
        report_base_dir = args.path
    else:
        report_base_dir = os.path.dirname(os.path.abspath(mcap_files[0])) or "."

    if args.manifest:
        manifest_path = args.manifest
        run_dir = os.path.dirname(os.path.abspath(manifest_path)) or "."
    elif args.resume:
        latest_manifest = find_latest_global_manifest(report_base_dir)
        if latest_manifest:
            manifest_path = latest_manifest
            run_dir = os.path.dirname(os.path.abspath(manifest_path)) or "."
        else:
            run_dir = os.path.join(report_base_dir, f"global_only_{timestamp}")
            manifest_path = os.path.join(run_dir, f"global_only_manifest_{timestamp}.jsonl")
    elif args.report_jsonl:
        report_jsonl_path = args.report_jsonl
        run_dir = os.path.dirname(os.path.abspath(report_jsonl_path)) or "."
        manifest_path = os.path.join(run_dir, f"global_only_manifest_{timestamp}.jsonl")
    else:
        run_dir = os.path.join(report_base_dir, f"global_only_{timestamp}")
        manifest_path = os.path.join(run_dir, f"global_only_manifest_{timestamp}.jsonl")

    if not args.report_jsonl:
        reuse_report_jsonl = find_latest_global_report_jsonl(run_dir) if (args.resume or args.manifest) else None
        report_jsonl_path = reuse_report_jsonl or os.path.join(run_dir, f"global_only_repair_report_{timestamp}.jsonl")

    output_run_dir = output_base_dir
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(output_run_dir, exist_ok=True)

    print(f"Found {len(mcap_files)} mcap file(s) to process:")
    for path in mcap_files:
        print(f"  - {path}")
    print(f"Output dir: {output_run_dir}")
    print(f"Report dir: {run_dir}")
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
        output_bag = make_global_output_path(bag_path, output_run_dir, input_root)
        print(f"\n[{idx}/{len(mcap_files)}] {bag_path}")

        if args.resume and os.path.abspath(bag_path) in completed_inputs:
            print("[SKIP] Already completed according to manifest.")
            skipped_count += 1
            continue

        try:
            report = repair_single_global(bag_path, output_bag, overwrite=args.overwrite, odom_source=args.odom_source)
        except Exception as exc:
            print(f"\n[ERROR] Failed to repair {bag_path}: {exc}")
            traceback.print_exc()
            report = build_quality_report(
                bag_path,
                output_bag,
                "exception",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )

        append_jsonl(manifest_path, report)
        append_jsonl(report_jsonl_path, report)
        processed_count += 1
        if report.get("status", "").startswith("skipped"):
            skipped_count += 1
        if report.get("ok", False):
            ok_count += 1
        else:
            bad_count += 1
            if report.get("status") == "exception":
                failed_count += 1
        print(
            f"[Progress] processed={processed_count} ok={ok_count} "
            f"needs_review={bad_count} skipped={skipped_count} failed={failed_count}"
        )

    report_path, recheck_json_path, recheck_list_path, recheck_csv_path, recheck_reports = (
        write_global_reports_from_jsonl(report_jsonl_path, run_dir, timestamp)
    )

    print(f"\n{'=' * 87}")
    print(f"  Global-only batch complete. {len(mcap_files)} file(s) discovered.")
    print(f"  Processed this run: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  OK: {ok_count}")
    print(f"  Needs review: {bad_count}")
    print(f"  Failed exceptions: {failed_count}")
    print(f"  Manifest JSONL: {manifest_path}")
    print(f"  Quality JSONL : {report_jsonl_path}")
    print(f"  Report JSON   : {report_path}")
    print(f"  Recheck JSON  : {recheck_json_path}")
    print(f"  Recheck list  : {recheck_list_path}")
    print(f"  Recheck CSV   : {recheck_csv_path}")
    print(f"  Recheck bags  : {len(recheck_reports)}")
    if recheck_reports:
        print("Bags needing recheck:")
        for report in recheck_reports:
            print(f"  - {report.get('input_bag', '')}")
            for reason in report.get("reasons", []):
                print(f"    * {reason}")
    print(f"{'=' * 87}")


if __name__ == "__main__":
    main()
