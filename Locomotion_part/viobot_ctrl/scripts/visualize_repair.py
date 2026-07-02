import argparse
import html
import os
from math import cos, radians

import numpy as np
from mcap_ros2.reader import read_ros2_messages


ODOM_TOPIC = '/baton/stereo3/odometry'
ODOM_ORIG_TOPIC = '/baton/stereo3/odometry_orig'
ODOM_REPAIRED_TOPIC = '/baton/stereo3/odometry_repaired'
SPP_FIX_TOPIC = '/baton/gnss/spp_fix'
SPP_FIX_ORIG_TOPIC = '/baton/gnss/spp_fix_orig'
SPP_FIX_REPAIRED_TOPIC = '/baton/gnss/spp_fix_repaired'
SPP_FIX_RAW_TOPIC = '/baton/gnss/spp_fix_raw'


def odom_to_xy(ros_msg):
    pos = ros_msg.pose.pose.position
    return -float(pos.y), float(pos.x), float(pos.z)


def lla_to_xy(records, origin=None):
    if not records:
        return [], None

    if origin is None:
        origin = records[0][1], records[0][2], records[0][3]

    lat0, lon0, alt0 = origin
    scale_x = 6378137.0 * cos(radians(lat0))
    scale_y = 6378137.0
    xy = []
    for t, lat, lon, alt in records:
        x = radians(lon - lon0) * scale_x
        y = radians(lat - lat0) * scale_y
        xy.append((t, x, y, alt - alt0))
    return xy, origin


def load_bag(path, prefix):
    data = {}
    wanted = {
        ODOM_TOPIC,
        ODOM_ORIG_TOPIC,
        ODOM_REPAIRED_TOPIC,
        SPP_FIX_TOPIC,
        SPP_FIX_ORIG_TOPIC,
        SPP_FIX_REPAIRED_TOPIC,
        SPP_FIX_RAW_TOPIC,
    }

    for msg in read_ros2_messages(path):
        topic = msg.channel.topic
        if topic not in wanted:
            continue

        t = msg.publish_time_ns / 1e9
        ros_msg = msg.ros_msg
        if topic in {ODOM_TOPIC, ODOM_ORIG_TOPIC, ODOM_REPAIRED_TOPIC}:
            x, y, z = odom_to_xy(ros_msg)
            data.setdefault(f'{prefix}{topic}', []).append((t, x, y, z))
        else:
            data.setdefault(f'{prefix}{topic}', []).append((
                t,
                float(ros_msg.latitude),
                float(ros_msg.longitude),
                float(ros_msg.altitude),
            ))

    return data


def normalize_datasets(raw_data):
    origin = None
    datasets = {}
    for name, records in raw_data.items():
        if not records:
            continue

        is_fix = any(topic in name for topic in (
            SPP_FIX_TOPIC,
            SPP_FIX_ORIG_TOPIC,
            SPP_FIX_REPAIRED_TOPIC,
            SPP_FIX_RAW_TOPIC,
        ))
        is_odom = any(topic in name for topic in (
            ODOM_TOPIC,
            ODOM_ORIG_TOPIC,
            ODOM_REPAIRED_TOPIC,
        ))

        if is_fix and not is_odom:
            xy, origin = lla_to_xy(records, origin)
            datasets[name] = xy
        else:
            x0 = records[0][1]
            y0 = records[0][2]
            z0 = records[0][3]
            datasets[name] = [(t, x - x0, y - y0, z - z0) for t, x, y, z in records]

    return datasets


def step_stats(records):
    if len(records) < 2:
        return None
    t = np.array([r[0] for r in records], dtype=float)
    xy = np.array([[r[1], r[2]] for r in records], dtype=float)
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
    return {
        'count': len(records),
        'duration': t[-1] - t[0],
        'mean_step': float(np.mean(step)),
        'max_step': float(np.max(step)),
        'p95_step': float(np.percentile(step, 95)),
        'p99_step': float(np.percentile(step, 99)),
        'max_speed': float(np.max(speed)),
        'j05': int(np.sum(step > 0.5)),
        'j10': int(np.sum(step > 1.0)),
        'j20': int(np.sum(step > 2.0)),
        'step_series': [(t[i + 1] - t[0], float(step[i])) for i in range(len(step))],
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
    s = min(sx, sy)
    used_w = (max_x - min_x) * s
    used_h = (max_y - min_y) * s
    ox = (width - used_w) * 0.5
    oy = (height - used_h) * 0.5

    pts = []
    for _, x, y, _ in records:
        px = ox + (x - min_x) * s
        py = height - (oy + (y - min_y) * s)
        pts.append((px, py))
    return pts


def polyline(points):
    return ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)


def sample_markers(points, count=10):
    if not points:
        return ''
    idx = np.linspace(0, len(points) - 1, min(count, len(points))).astype(int)
    return '\n'.join(
        f'<circle cx="{points[i][0]:.1f}" cy="{points[i][1]:.1f}" r="2.5" />'
        for i in idx
    )


def step_path(series, width, height, max_step=None):
    if not series:
        return ''
    max_t = max(t for t, _ in series) or 1.0
    max_v = max_step if max_step is not None else max(v for _, v in series) or 1.0
    pts = []
    for t, v in series:
        x = t / max_t * width
        y = height - min(v / max_v, 1.0) * height
        pts.append((x, y))
    return polyline(pts)


def display_name(name):
    replacements = [
        ('original:', 'original '),
        ('repaired:', 'repaired '),
        ('/baton/stereo3/', ''),
        ('/baton/gnss/', ''),
        ('odometry_repaired', 'odom repaired'),
        ('odometry_orig', 'odom original'),
        ('odometry', 'odom current'),
        ('spp_fix_repaired', 'spp repaired'),
        ('spp_fix_orig', 'spp original'),
        ('spp_fix_raw', 'spp raw'),
        ('spp_fix', 'spp current'),
    ]
    out = name
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def write_html(output_path, input_paths, datasets):
    width = 1080
    height = 720
    bounds = bounds_for(datasets)
    stats = {name: step_stats(records) for name, records in datasets.items()}
    colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#9467bd',
        '#ff7f0e', '#17becf', '#8c564b', '#7f7f7f',
    ]

    lines = []
    legend = []
    table_rows = []
    max_step = max((s['max_step'] for s in stats.values() if s), default=1.0)

    for i, (name, records) in enumerate(datasets.items()):
        color = colors[i % len(colors)]
        pts = scale_points(records, bounds, width, height)
        label = display_name(name)
        escaped_label = html.escape(label)
        lines.append(
            f'<g class="series" style="--c:{color}">'
            f'<polyline points="{polyline(pts)}" />'
            f'{sample_markers(pts)}'
            f'<title>{escaped_label}</title>'
            f'</g>'
        )
        legend.append(f'<span><i style="background:{color}"></i>{escaped_label}</span>')
        s = stats[name]
        if s:
            table_rows.append(
                '<tr>'
                f'<td>{escaped_label}</td>'
                f'<td>{s["count"]}</td>'
                f'<td>{s["duration"]:.2f}</td>'
                f'<td>{s["mean_step"]:.3f}</td>'
                f'<td>{s["p95_step"]:.3f}</td>'
                f'<td>{s["p99_step"]:.3f}</td>'
                f'<td>{s["max_step"]:.3f}</td>'
                f'<td>{s["max_speed"]:.1f}</td>'
                f'<td>{s["j05"]}</td>'
                f'<td>{s["j10"]}</td>'
                f'<td>{s["j20"]}</td>'
                '</tr>'
            )

    step_lines = []
    for i, (name, s) in enumerate(stats.items()):
        if not s:
            continue
        color = colors[i % len(colors)]
        step_lines.append(
            f'<polyline style="stroke:{color}" points="{step_path(s["step_series"], width, 220, max_step)}">'
            f'<title>{html.escape(display_name(name))}</title>'
            '</polyline>'
        )

    source_list = ''.join(f'<li>{html.escape(path)}</li>' for path in input_paths)
    html_text = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MCAP Repair Visualization</title>
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
  background: #ffffff;
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
code {{
  background: #eef2f6;
  padding: 1px 4px;
  border-radius: 3px;
}}
</style>
</head>
<body>
<main>
<h1>MCAP Repair Visualization</h1>
<div class="panel">
<strong>Sources</strong>
<ul>{source_list}</ul>
<div class="legend">{''.join(legend)}</div>
</div>

<section class="panel">
<h2>XY Trajectory</h2>
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
<p>Step chart vertical range: 0 to {max_step:.3f} m.</p>
</section>

<section class="panel">
<h2>Step Statistics</h2>
<table>
<thead>
<tr>
<th>Series</th><th>Count</th><th>Duration s</th><th>Mean m</th><th>P95 m</th><th>P99 m</th><th>Max m</th><th>Max m/s</th><th>&gt;0.5m</th><th>&gt;1m</th><th>&gt;2m</th>
</tr>
</thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</section>
</main>
</body>
</html>
'''
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_text)


def default_output_path(path):
    base, _ = os.path.splitext(path)
    return f'{base}_repair_compare.html'


def main():
    parser = argparse.ArgumentParser(description='Create a temporary HTML/SVG comparison of odom and GNSS repair outputs.')
    parser.add_argument('bags', nargs='+', help='One bag, or original bag followed by repaired bag.')
    parser.add_argument('-o', '--output', help='Output HTML path. Defaults to <last_bag>_repair_compare.html.')
    args = parser.parse_args()

    raw_data = {}
    if len(args.bags) == 1:
        raw_data.update(load_bag(args.bags[0], ''))
    else:
        raw_data.update(load_bag(args.bags[0], 'original:'))
        raw_data.update(load_bag(args.bags[1], 'repaired:'))

    datasets = normalize_datasets(raw_data)
    output = args.output or default_output_path(args.bags[-1])
    write_html(output, args.bags, datasets)
    print(f'Wrote visualization: {output}')


if __name__ == '__main__':
    main()
