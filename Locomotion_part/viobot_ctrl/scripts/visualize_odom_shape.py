import argparse
import html
import os

import numpy as np
from mcap_ros2.reader import read_ros2_messages

from post_repair import (
    ODOM_ORIG_TOPIC,
    ODOM_REPAIRED_TOPIC,
    ODOM_TOPIC,
    odom_msg_to_enu,
    odom_twist_to_enu,
    repair_odom_data,
)


ODOM_TOPICS = {
    ODOM_ORIG_TOPIC: "odom_orig",
    ODOM_REPAIRED_TOPIC: "odom_repaired",
    ODOM_TOPIC: "odom_current",
}


def load_odom_records(path, prefix=""):
    records = {topic: [] for topic in ODOM_TOPICS}
    odom_data_for_compute = []

    for msg in read_ros2_messages(path):
        topic = msg.channel.topic
        if topic not in ODOM_TOPICS:
            continue

        t = msg.publish_time_ns / 1e9
        pos = odom_msg_to_enu(msg.ros_msg)
        records[topic].append((t, pos[0], pos[1], pos[2]))

        if topic == ODOM_TOPIC:
            twist = odom_twist_to_enu(msg.ros_msg)
            odom_data_for_compute.append(
                {"t": t, "x": pos[0], "y": pos[1], "z": pos[2], "twist": twist}
            )

    datasets = {}
    base = prefix or os.path.basename(path)

    if records[ODOM_ORIG_TOPIC]:
        datasets[f"{base}: odom_orig"] = records[ODOM_ORIG_TOPIC]
    elif records[ODOM_TOPIC]:
        datasets[f"{base}: odom_raw"] = records[ODOM_TOPIC]

    if records[ODOM_REPAIRED_TOPIC]:
        datasets[f"{base}: odom_repaired"] = records[ODOM_REPAIRED_TOPIC]
    elif odom_data_for_compute:
        repaired, _ = repair_odom_data(odom_data_for_compute)
        times = [p["t"] for p in odom_data_for_compute]
        datasets[f"{base}: odom_repair_computed"] = [
            (times[i], repaired[i, 0], repaired[i, 1], repaired[i, 2])
            for i in range(len(times))
        ]

    return datasets


def normalize(records):
    if not records:
        return []
    x0, y0, z0 = records[0][1], records[0][2], records[0][3]
    return [(t, x - x0, y - y0, z - z0) for t, x, y, z in records]


def step_stats(records):
    if len(records) < 2:
        return None
    t = np.array([r[0] for r in records], dtype=float)
    xy = np.array([[r[1], r[2]] for r in records], dtype=float)
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
    return {
        "count": len(records),
        "duration": float(t[-1] - t[0]),
        "mean_step": float(np.mean(step)),
        "p95_step": float(np.percentile(step, 95)),
        "p99_step": float(np.percentile(step, 99)),
        "max_step": float(np.max(step)),
        "max_speed": float(np.max(speed)),
        "j05": int(np.sum(step > 0.5)),
        "j10": int(np.sum(step > 1.0)),
        "j20": int(np.sum(step > 2.0)),
        "step_series": [(t[i + 1] - t[0], float(step[i])) for i in range(len(step))],
    }


def shape_delta(base_records, other_records):
    if len(base_records) < 2 or len(other_records) < 2:
        return None

    base = normalize(base_records)
    other = normalize(other_records)
    base_t = np.array([r[0] for r in base], dtype=float)
    base_xy = np.array([[r[1], r[2]] for r in base], dtype=float)

    deltas = []
    for t, x, y, _ in other:
        idx = int(np.argmin(np.abs(base_t - t)))
        if abs(base_t[idx] - t) <= 0.05:
            deltas.append(np.linalg.norm(np.array([x, y]) - base_xy[idx]))

    if not deltas:
        return None
    deltas = np.array(deltas, dtype=float)
    return {
        "count": int(len(deltas)),
        "mean": float(np.mean(deltas)),
        "p95": float(np.percentile(deltas, 95)),
        "p99": float(np.percentile(deltas, 99)),
        "max": float(np.max(deltas)),
    }


def bounds_for(datasets):
    xs = []
    ys = []
    for records in datasets.values():
        xs.extend([r[1] for r in records])
        ys.extend([r[2] for r in records])
    if not xs:
        return -1.0, 1.0, -1.0, 1.0
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = max(max_x - min_x, max_y - min_y, 1.0) * 0.06
    return min_x - pad, max_x + pad, min_y - pad, max_y + pad


def scale_points(records, bounds, width, height):
    min_x, max_x, min_y, max_y = bounds
    sx = width / max(max_x - min_x, 1e-9)
    sy = height / max(max_y - min_y, 1e-9)
    scale = min(sx, sy)
    used_w = (max_x - min_x) * scale
    used_h = (max_y - min_y) * scale
    ox = (width - used_w) * 0.5
    oy = (height - used_h) * 0.5
    return [
        (ox + (x - min_x) * scale, height - (oy + (y - min_y) * scale))
        for _, x, y, _ in records
    ]


def polyline(points):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def step_path(series, width, height, max_step):
    if not series:
        return ""
    max_t = max(t for t, _ in series) or 1.0
    pts = []
    for t, value in series:
        x = t / max_t * width
        y = height - min(value / max_step, 1.0) * height
        pts.append((x, y))
    return polyline(pts)


def marker_points(points):
    if not points:
        return ""
    idx = np.linspace(0, len(points) - 1, min(12, len(points))).astype(int)
    return "\n".join(
        f'<circle cx="{points[i][0]:.1f}" cy="{points[i][1]:.1f}" r="2.4" />'
        for i in idx
    )


def write_html(output_path, input_paths, datasets):
    normalized = {name: normalize(records) for name, records in datasets.items() if records}
    stats = {name: step_stats(records) for name, records in normalized.items()}
    bounds = bounds_for(normalized)
    width = 1080
    height = 720
    colors = ["#1769aa", "#c62828", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f"]

    lines = []
    legend = []
    for i, (name, records) in enumerate(normalized.items()):
        color = colors[i % len(colors)]
        points = scale_points(records, bounds, width, height)
        label = html.escape(name)
        lines.append(
            f'<g class="series" style="--c:{color}">'
            f'<polyline points="{polyline(points)}" />'
            f'{marker_points(points)}'
            f'<title>{label}</title>'
            f'</g>'
        )
        legend.append(f'<span><i style="background:{color}"></i>{label}</span>')

    max_step = max((s["max_step"] for s in stats.values() if s), default=1.0)
    step_lines = []
    for i, (name, stat) in enumerate(stats.items()):
        if not stat:
            continue
        color = colors[i % len(colors)]
        step_lines.append(
            f'<polyline style="stroke:{color}" points="{step_path(stat["step_series"], width, 220, max_step)}">'
            f'<title>{html.escape(name)}</title>'
            '</polyline>'
        )

    stat_rows = []
    for name, stat in stats.items():
        if not stat:
            continue
        stat_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{stat['count']}</td>"
            f"<td>{stat['duration']:.2f}</td>"
            f"<td>{stat['mean_step']:.4f}</td>"
            f"<td>{stat['p95_step']:.4f}</td>"
            f"<td>{stat['p99_step']:.4f}</td>"
            f"<td>{stat['max_step']:.4f}</td>"
            f"<td>{stat['max_speed']:.2f}</td>"
            f"<td>{stat['j05']}</td>"
            f"<td>{stat['j10']}</td>"
            f"<td>{stat['j20']}</td>"
            "</tr>"
        )

    delta_rows = []
    names = list(datasets.keys())
    if names:
        base_name = names[0]
        base_records = datasets[base_name]
        for name in names[1:]:
            delta = shape_delta(base_records, datasets[name])
            if not delta:
                continue
            delta_rows.append(
                "<tr>"
                f"<td>{html.escape(base_name)}</td>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{delta['count']}</td>"
                f"<td>{delta['mean']:.4f}</td>"
                f"<td>{delta['p95']:.4f}</td>"
                f"<td>{delta['p99']:.4f}</td>"
                f"<td>{delta['max']:.4f}</td>"
                "</tr>"
            )

    source_items = "".join(f"<li>{html.escape(path)}</li>" for path in input_paths)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Odometry Shape Comparison</title>
<style>
body {{
  margin: 0;
  font-family: Arial, sans-serif;
  color: #1f2933;
  background: #f7f8fa;
}}
main {{
  max-width: 1240px;
  margin: 0 auto;
  padding: 24px;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 24px;
}}
h2 {{
  margin: 24px 0 12px;
  font-size: 18px;
}}
.panel {{
  background: #fff;
  border: 1px solid #d7dde5;
  border-radius: 6px;
  padding: 16px;
  margin-top: 16px;
}}
.legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  margin: 12px 0;
  font-size: 13px;
}}
.legend span {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.legend i {{
  width: 18px;
  height: 3px;
  display: inline-block;
}}
svg {{
  width: 100%;
  height: auto;
  background: #fbfcfd;
  border: 1px solid #e1e6ed;
}}
.axis {{
  stroke: #c7d0da;
  stroke-width: 1;
}}
.series polyline {{
  fill: none;
  stroke: var(--c);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}}
.series circle {{
  fill: var(--c);
  opacity: 0.75;
}}
.steps polyline {{
  fill: none;
  stroke-width: 1.8;
  vector-effect: non-scaling-stroke;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}}
th, td {{
  border-bottom: 1px solid #e5e9ef;
  padding: 7px 8px;
  text-align: right;
}}
th:first-child, td:first-child {{
  text-align: left;
}}
</style>
</head>
<body>
<main>
<h1>Odometry Shape Comparison</h1>
<div class="panel">
<strong>Sources</strong>
<ul>{source_items}</ul>
<div class="legend">{''.join(legend)}</div>
</div>

<section class="panel">
<h2>Normalized XY Shape</h2>
<svg viewBox="0 0 {width} {height}" role="img">
<line class="axis" x1="0" y1="{height / 2:.1f}" x2="{width}" y2="{height / 2:.1f}" />
<line class="axis" x1="{width / 2:.1f}" y1="0" x2="{width / 2:.1f}" y2="{height}" />
{''.join(lines)}
</svg>
</section>

<section class="panel">
<h2>2D Step Distance</h2>
<svg class="steps" viewBox="0 0 {width} 220" role="img">
<line class="axis" x1="0" y1="220" x2="{width}" y2="220" />
<line class="axis" x1="0" y1="0" x2="0" y2="220" />
{''.join(step_lines)}
</svg>
<p>Step chart vertical range: 0 to {max_step:.4f} m.</p>
</section>

<section class="panel">
<h2>Step Statistics</h2>
<table>
<thead>
<tr><th>Series</th><th>Count</th><th>Duration s</th><th>Mean m</th><th>P95 m</th><th>P99 m</th><th>Max m</th><th>Max m/s</th><th>&gt;0.5m</th><th>&gt;1m</th><th>&gt;2m</th></tr>
</thead>
<tbody>{''.join(stat_rows)}</tbody>
</table>
</section>

<section class="panel">
<h2>Shape Delta From First Series</h2>
<table>
<thead>
<tr><th>Base</th><th>Compared</th><th>Matched</th><th>Mean m</th><th>P95 m</th><th>P99 m</th><th>Max m</th></tr>
</thead>
<tbody>{''.join(delta_rows)}</tbody>
</table>
</section>
</main>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)


def default_output_path(path):
    base, _ = os.path.splitext(path)
    return f"{base}_odom_shape_compare.html"


def main():
    parser = argparse.ArgumentParser(description="Plot original and repaired odometry shapes from MCAP files.")
    parser.add_argument("bags", nargs="+", help="One or more MCAP bags to visualize.")
    parser.add_argument("-o", "--output", help="Output HTML path. Defaults to <last_bag>_odom_shape_compare.html.")
    args = parser.parse_args()

    datasets = {}
    for idx, bag in enumerate(args.bags, 1):
        prefix = os.path.basename(bag) if len(args.bags) > 1 else ""
        loaded = load_odom_records(bag, prefix=prefix)
        datasets.update(loaded)

    if not datasets:
        raise SystemExit("No odometry records found.")

    output = args.output or default_output_path(args.bags[-1])
    write_html(output, args.bags, datasets)
    print(f"Wrote odometry shape visualization: {output}")


if __name__ == "__main__":
    main()
