#!/usr/bin/env python3
"""Generate a standalone HTML report for a PoseStamped trajectory in a ROS 2 MCAP bag."""

import argparse
import html
import math
from pathlib import Path

import numpy as np
from mcap_ros2.reader import read_ros2_messages


DEFAULT_TOPIC = "/insight/vio_20hz"
COLORS = ("#2563eb", "#ef4444", "#16a34a", "#9333ea")


def resolve_bags(path):
    """Return the MCAP files represented by *path* in recording order."""
    path = Path(path).expanduser()
    if path.is_file():
        if path.suffix.lower() != ".mcap":
            raise ValueError(f"Not an MCAP file: {path}")
        return [path]
    if path.is_dir():
        bags = sorted(path.glob("*.mcap"))
        if bags:
            return bags
        raise ValueError(f"No .mcap files found in: {path}")
    raise ValueError(f"Input does not exist: {path}")


def quaternion_to_rpy(x, y, z, w):
    """Convert a quaternion to roll, pitch and yaw in radians."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return math.nan, math.nan, math.nan
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def load_pose_records(paths, topic):
    records = []
    frame_ids = set()
    for path in paths:
        for message in read_ros2_messages(str(path)):
            if message.channel.topic != topic:
                continue
            pose_stamped = message.ros_msg
            position = pose_stamped.pose.position
            orientation = pose_stamped.pose.orientation
            roll, pitch, yaw = quaternion_to_rpy(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
            frame_id = str(pose_stamped.header.frame_id)
            if frame_id:
                frame_ids.add(frame_id)
            records.append(
                (
                    message.publish_time_ns * 1e-9,
                    float(position.x),
                    float(position.y),
                    float(position.z),
                    roll,
                    pitch,
                    yaw,
                )
            )

    records.sort(key=lambda item: item[0])
    if not records:
        raise ValueError(f"No PoseStamped messages found on topic {topic!r}")
    return np.asarray(records, dtype=float), sorted(frame_ids)


def evenly_sample_indices(length, maximum):
    if length <= maximum:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def finite_range(values, pad_ratio=0.06):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return -1.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    span = high - low
    pad = (span if span > 1e-9 else max(abs(low), 1.0)) * pad_ratio
    return low - pad, high + pad


def svg_polyline(x, y, x_range, y_range, width, height, margin):
    x0, x1 = x_range
    y0, y1 = y_range
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    points = []
    for xv, yv in zip(x, y):
        if not (math.isfinite(xv) and math.isfinite(yv)):
            continue
        px = margin + (xv - x0) / max(x1 - x0, 1e-12) * plot_w
        py = height - margin - (yv - y0) / max(y1 - y0, 1e-12) * plot_h
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def chart_svg(series, x_range, y_range, y_label, width=1100, height=310):
    margin = 58
    x0, x1 = x_range
    y0, y1 = y_range
    grid = []
    labels = []
    for i in range(6):
        ratio = i / 5
        px = margin + ratio * (width - 2 * margin)
        py = height - margin - ratio * (height - 2 * margin)
        xv = x0 + ratio * (x1 - x0)
        yv = y0 + ratio * (y1 - y0)
        grid.append(f'<line x1="{px:.1f}" y1="{margin}" x2="{px:.1f}" y2="{height-margin}" />')
        grid.append(f'<line x1="{margin}" y1="{py:.1f}" x2="{width-margin}" y2="{py:.1f}" />')
        labels.append(f'<text x="{px:.1f}" y="{height-20}" text-anchor="middle">{xv:.1f}</text>')
        labels.append(f'<text x="{margin-9}" y="{py+4:.1f}" text-anchor="end">{yv:.2f}</text>')

    lines = []
    legend = []
    for index, (name, x, y) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        points = svg_polyline(x, y, x_range, y_range, width, height, margin)
        lines.append(f'<polyline points="{points}" style="stroke:{color}"><title>{html.escape(name)}</title></polyline>')
        legend.append(f'<span><i style="background:{color}"></i>{html.escape(name)}</span>')

    return f"""
<div class="legend">{''.join(legend)}</div>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(y_label)}">
  <g class="grid">{''.join(grid)}</g>
  <g class="tick-labels">{''.join(labels)}</g>
  <text class="axis-label" x="{width/2:.1f}" y="{height-2}" text-anchor="middle">Time (s)</text>
  <text class="axis-label" transform="translate(15 {height/2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>
  <g class="series">{''.join(lines)}</g>
</svg>"""


def trajectory_svg(x, y, width=820, height=680):
    margin = 65
    x0, x1 = finite_range(x)
    y0, y1 = finite_range(y)
    # Use the same metre-per-pixel scale on both axes so the shape is not distorted.
    plot_ratio = (width - 2 * margin) / (height - 2 * margin)
    dx, dy = x1 - x0, y1 - y0
    if dx / dy > plot_ratio:
        wanted_dy = dx / plot_ratio
        center = (y0 + y1) / 2
        y0, y1 = center - wanted_dy / 2, center + wanted_dy / 2
    else:
        wanted_dx = dy * plot_ratio
        center = (x0 + x1) / 2
        x0, x1 = center - wanted_dx / 2, center + wanted_dx / 2

    grid = []
    labels = []
    for i in range(6):
        ratio = i / 5
        px = margin + ratio * (width - 2 * margin)
        py = height - margin - ratio * (height - 2 * margin)
        xv, yv = x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0)
        grid.extend((
            f'<line x1="{px:.1f}" y1="{margin}" x2="{px:.1f}" y2="{height-margin}" />',
            f'<line x1="{margin}" y1="{py:.1f}" x2="{width-margin}" y2="{py:.1f}" />',
        ))
        labels.extend((
            f'<text x="{px:.1f}" y="{height-29}" text-anchor="middle">{xv:.2f}</text>',
            f'<text x="{margin-9}" y="{py+4:.1f}" text-anchor="end">{yv:.2f}</text>',
        ))

    points = svg_polyline(x, y, (x0, x1), (y0, y1), width, height, margin)
    first = svg_polyline(x[:1], y[:1], (x0, x1), (y0, y1), width, height, margin)
    last = svg_polyline(x[-1:], y[-1:], (x0, x1), (y0, y1), width, height, margin)
    fx, fy = first.split(",")
    lx, ly = last.split(",")
    return f"""
<div class="legend"><span><i style="background:{COLORS[0]}"></i>XY trajectory</span><span class="dot start"></span>Start<span class="dot end"></span>End</div>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="XY trajectory">
  <g class="grid">{''.join(grid)}</g>
  <g class="tick-labels">{''.join(labels)}</g>
  <text class="axis-label" x="{width/2:.1f}" y="{height-3}" text-anchor="middle">X (m)</text>
  <text class="axis-label" transform="translate(15 {height/2:.1f}) rotate(-90)" text-anchor="middle">Y (m)</text>
  <g class="series"><polyline points="{points}" style="stroke:{COLORS[0]}" /></g>
  <circle cx="{fx}" cy="{fy}" r="6" fill="#16a34a"><title>Start</title></circle>
  <circle cx="{lx}" cy="{ly}" r="6" fill="#ef4444"><title>End</title></circle>
</svg>"""


def summary(records):
    t = records[:, 0]
    xyz = records[:, 1:4]
    dt = np.diff(t)
    distance_steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    valid_dt = dt > 0
    speeds = distance_steps[valid_dt] / dt[valid_dt]
    return {
        "count": len(records),
        "duration": float(t[-1] - t[0]),
        "rate": float((len(records) - 1) / (t[-1] - t[0])) if t[-1] > t[0] else 0.0,
        "distance": float(np.sum(distance_steps)),
        "displacement": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "max_speed": float(np.max(speeds)) if len(speeds) else 0.0,
        "mean_speed": float(np.mean(speeds)) if len(speeds) else 0.0,
    }


def write_report(output, source_paths, topic, records, frame_ids, max_points):
    stats = summary(records)
    relative_t = records[:, 0] - records[0, 0]
    position = records[:, 1:4] - records[0, 1:4]
    angles = np.rad2deg(records[:, 4:7])
    angles[:, 2] = np.rad2deg(np.unwrap(records[:, 6]))

    dt = np.diff(relative_t)
    step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    valid = dt > 0
    speed_t = relative_t[1:][valid]
    speed = step[valid] / dt[valid]

    sample = evenly_sample_indices(len(records), max_points)
    speed_sample = evenly_sample_indices(len(speed), max_points)
    time_range = finite_range(relative_t, pad_ratio=0.0)
    if time_range[0] == time_range[1]:
        time_range = (0.0, 1.0)

    position_chart = chart_svg(
        [("X", relative_t[sample], position[sample, 0]),
         ("Y", relative_t[sample], position[sample, 1]),
         ("Z", relative_t[sample], position[sample, 2])],
        time_range,
        finite_range(position),
        "Relative position (m)",
    )
    angle_chart = chart_svg(
        [("Roll", relative_t[sample], angles[sample, 0]),
         ("Pitch", relative_t[sample], angles[sample, 1]),
         ("Yaw (unwrapped)", relative_t[sample], angles[sample, 2])],
        time_range,
        finite_range(angles),
        "Euler angle (deg)",
    )
    speed_chart = chart_svg(
        [("3D speed", speed_t[speed_sample], speed[speed_sample])],
        time_range,
        finite_range(speed),
        "Speed (m/s)",
    )
    trajectory = trajectory_svg(position[sample, 0], position[sample, 1])

    sources = "".join(f"<li>{html.escape(str(path))}</li>" for path in source_paths)
    frames = ", ".join(html.escape(frame) for frame in frame_ids) or "(empty)"
    cards = (
        ("Messages", f'{stats["count"]:,}'),
        ("Duration", f'{stats["duration"]:.2f} s'),
        ("Mean rate", f'{stats["rate"]:.2f} Hz'),
        ("Travel distance", f'{stats["distance"]:.2f} m'),
        ("Displacement", f'{stats["displacement"]:.2f} m'),
        ("Mean / max speed", f'{stats["mean_speed"]:.2f} / {stats["max_speed"]:.2f} m/s'),
    )
    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in cards
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VIO Pose Visualization</title>
<style>
:root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; color:#172033; background:#f3f5f8; }}
body {{ margin:0; }} main {{ max-width:1220px; margin:auto; padding:24px; }}
h1 {{ margin:0 0 6px; font-size:26px; }} h2 {{ font-size:18px; margin:0 0 10px; }}
.muted {{ color:#667085; font-size:13px; overflow-wrap:anywhere; }}
.panel {{ background:white; border:1px solid #dfe3ea; border-radius:10px; padding:18px; margin-top:18px; box-shadow:0 1px 2px #1018280a; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:10px; margin-top:18px; }}
.card {{ background:white; border:1px solid #dfe3ea; border-radius:9px; padding:13px; }}
.card span {{ display:block; color:#667085; font-size:12px; margin-bottom:5px; }} .card strong {{ font-size:18px; }}
svg {{ display:block; width:100%; height:auto; background:#fcfdff; border:1px solid #eaecf0; border-radius:5px; }}
.grid line {{ stroke:#e5e9f0; stroke-width:1; }} .tick-labels {{ fill:#667085; font-size:11px; }} .axis-label {{ fill:#344054; font-size:12px; }}
.series polyline {{ fill:none; stroke-width:1.8; vector-effect:non-scaling-stroke; stroke-linejoin:round; stroke-linecap:round; }}
.legend {{ min-height:22px; display:flex; align-items:center; flex-wrap:wrap; gap:7px 16px; font-size:12px; color:#475467; margin-bottom:8px; }}
.legend span {{ display:inline-flex; align-items:center; gap:5px; }} .legend i {{ display:inline-block; width:18px; height:3px; }}
.dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:-12px; }} .start {{ background:#16a34a; }} .end {{ background:#ef4444; }}
ul {{ margin:7px 0 0; padding-left:20px; }}
@media (max-width:650px) {{ main {{ padding:12px; }} .panel {{ padding:10px; }} }}
</style></head><body><main>
<h1>VIO Pose Visualization</h1>
<div class="muted">Topic: <code>{html.escape(topic)}</code> · frame_id: {frames}</div>
<div class="cards">{card_html}</div>
<section class="panel"><h2>XY trajectory (relative to first pose)</h2>{trajectory}</section>
<section class="panel"><h2>Position over time</h2>{position_chart}</section>
<section class="panel"><h2>Orientation over time</h2>{angle_chart}</section>
<section class="panel"><h2>Translational speed</h2>{speed_chart}</section>
<section class="panel"><h2>Sources</h2><ul class="muted">{sources}</ul></section>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def default_output_path(input_path):
    path = Path(input_path).expanduser()
    if path.is_dir():
        return path / "vio_pose_visualization.html"
    return path.with_name(f"{path.stem}_vio_pose_visualization.html")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a geometry_msgs/PoseStamped trajectory from a ROS 2 MCAP bag."
    )
    parser.add_argument("bag", help="An .mcap file or a rosbag2 directory containing .mcap files.")
    parser.add_argument("-t", "--topic", default=DEFAULT_TOPIC, help=f"Pose topic (default: {DEFAULT_TOPIC}).")
    parser.add_argument("-o", "--output", type=Path, help="Output standalone HTML file.")
    parser.add_argument("--max-points", type=int, default=5000, help="Maximum SVG points per curve (default: 5000).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_points < 2:
        raise SystemExit("--max-points must be at least 2")
    try:
        paths = resolve_bags(args.bag)
        records, frame_ids = load_pose_records(paths, args.topic)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    output = args.output or default_output_path(args.bag)
    write_report(output, paths, args.topic, records, frame_ids, args.max_points)
    print(f"Loaded {len(records):,} poses from {len(paths)} MCAP file(s).")
    print(f"Wrote visualization: {output}")


if __name__ == "__main__":
    main()
