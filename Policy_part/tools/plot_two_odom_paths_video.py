#!/usr/bin/env python3
import argparse
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from threading import Lock

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


Trajectory = list[tuple[float, float, float]]


@dataclass(frozen=True)
class PanelConfig:
    title: str
    raw_topic: str
    fusion_topic: str


@dataclass
class PanelState:
    config: PanelConfig
    raw: Trajectory = field(default_factory=list)
    fusion: Trajectory = field(default_factory=list)


@dataclass(frozen=True)
class FusionVariant:
    name: str
    title: str
    params: dict[str, object]


FUSION_VARIANTS: dict[str, FusionVariant] = {
    'alignment4': FusionVariant(
        name='alignment4',
        title='alignment freeze 4',
        params={
            'optimization_solver': 'alignment',
            'alignment_min_gps_constraints': 4,
            'freeze_alignment_after_init': True,
            'publish_before_global_alignment': False,
            'sync_tolerance_sec': 0.01,
        },
    ),
    'alignment10': FusionVariant(
        name='alignment10',
        title='alignment freeze 10',
        params={
            'optimization_solver': 'alignment',
            'alignment_min_gps_constraints': 10,
            'freeze_alignment_after_init': True,
            'publish_before_global_alignment': False,
            'sync_tolerance_sec': 0.01,
        },
    ),
    'alignment_live': FusionVariant(
        name='alignment_live',
        title='alignment live',
        params={
            'optimization_solver': 'alignment',
            'alignment_min_gps_constraints': 4,
            'freeze_alignment_after_init': False,
            'publish_before_global_alignment': False,
            'sync_tolerance_sec': 0.01,
        },
    ),
    'pose_graph': FusionVariant(
        name='pose_graph',
        title='pose graph',
        params={
            'optimization_solver': 'pose_graph',
            'sync_tolerance_sec': 0.01,
            'publish_before_global_alignment': False,
        },
    ),
    'old_python': FusionVariant(
        name='old_python',
        title='old python style',
        params={
            'optimization_solver': 'pose_graph',
            'sync_tolerance_sec': 0.2,
            'publish_before_global_alignment': True,
        },
    ),
}


class MultiOdomPathVideoNode(Node):
    def __init__(self, args: argparse.Namespace, panels: list[PanelConfig]):
        super().__init__('multi_odom_path_video')
        self.args = args
        self.panels = [PanelState(config=panel) for panel in panels]
        self.lock = Lock()
        self.last_write_mono = 0.0
        self.frame_count = 0
        self.start_mono = time.monotonic()
        self.last_msg_mono = self.start_mono
        self.stop_requested = False
        self.raw_color = parse_color(args.raw_color)
        self.fusion_color = parse_color(args.fusion_color)

        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        self.writer = cv2.VideoWriter(
            args.output,
            fourcc,
            float(args.fps),
            (int(args.width), int(args.height)),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f'Failed to open video writer: {args.output}')

        topic_destinations: dict[str, list[tuple[int, str]]] = {}
        for panel_index, panel in enumerate(self.panels):
            topic_destinations.setdefault(panel.config.raw_topic, []).append((panel_index, 'raw'))
            topic_destinations.setdefault(panel.config.fusion_topic, []).append((panel_index, 'fusion'))

        self._topic_subscriptions = []
        for topic, destinations in topic_destinations.items():
            self._topic_subscriptions.append(
                self.create_subscription(
                    Odometry,
                    topic,
                    self.make_odom_callback(destinations),
                    100,
                )
            )

        if not args.frame_on_message:
            self.timer = self.create_timer(1.0 / float(args.fps), self.timer_callback)
        else:
            self.timer = None

        panel_desc = '; '.join(
            f'{p.title}: raw={p.raw_topic}, fusion={p.fusion_topic}'
            for p in panels
        )
        self.get_logger().info(
            'Recording multi-panel odom path video: %s -> %s'
            % (panel_desc, args.output)
        )

    def make_odom_callback(self, destinations: list[tuple[int, str]]):
        def callback(msg: Odometry):
            for panel_index, kind in destinations:
                self.append_pose(panel_index, kind, msg)
            if self.args.frame_on_message:
                self.write_frame_throttled()

        return callback

    def append_pose(self, panel_index: int, kind: str, msg: Odometry):
        p = msg.pose.pose.position
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(stamp)):
            return

        with self.lock:
            panel = self.panels[panel_index]
            trajectory = panel.raw if kind == 'raw' else panel.fusion
            trajectory.append((float(stamp), float(p.x), float(p.y)))
            if self.args.max_points > 0 and len(trajectory) > self.args.max_points:
                del trajectory[:len(trajectory) - self.args.max_points]
            self.last_msg_mono = time.monotonic()

    def timer_callback(self):
        self.write_frame()

    def write_frame_throttled(self):
        now = time.monotonic()
        if now - self.last_write_mono < 1.0 / float(self.args.fps):
            return
        self.write_frame()

    def snapshot_panels(self) -> list[PanelState]:
        with self.lock:
            return [
                PanelState(
                    config=panel.config,
                    raw=list(panel.raw),
                    fusion=list(panel.fusion),
                )
                for panel in self.panels
            ]

    def write_frame(self):
        frame = self.render_frame(self.snapshot_panels())
        self.writer.write(frame)
        self.frame_count += 1
        self.last_write_mono = time.monotonic()

    def render_frame(self, panels: list[PanelState]) -> np.ndarray:
        width = int(self.args.width)
        height = int(self.args.height)
        background = parse_color(self.args.background)
        frame = np.full((height, width, 3), background, dtype=np.uint8)

        panel_count = max(1, len(panels))
        cols = int(self.args.columns) if int(self.args.columns) > 0 else panel_count
        cols = max(1, min(cols, panel_count))
        rows = int(math.ceil(panel_count / cols))
        pane_w = width // cols
        pane_h = height // rows

        for index, panel in enumerate(panels):
            row = index // cols
            col = index % cols
            x0 = col * pane_w
            y0 = row * pane_h
            x1 = width if col == cols - 1 else (col + 1) * pane_w
            y1 = height if row == rows - 1 else (row + 1) * pane_h
            pane = frame[y0:y1, x0:x1]
            self.render_panel(pane, panel)
            cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), parse_color(self.args.panel_border), 1)

        return frame

    def render_panel(self, frame: np.ndarray, panel: PanelState):
        width = frame.shape[1]
        height = frame.shape[0]
        margin = int(self.args.margin)
        raw_points_abs = trajectory_to_points(panel.raw)
        fusion_points_abs = trajectory_to_points(panel.fusion)
        raw_points, fusion_points = normalize_panel_points(
            raw_points_abs,
            fusion_points_abs,
            self.args.origin_mode,
        )
        all_points = np.concatenate(
            [points for points in (raw_points, fusion_points) if points.size > 0],
            axis=0,
        ) if raw_points.size or fusion_points.size else np.zeros((0, 2), dtype=np.float32)

        if all_points.shape[0] == 0:
            draw_centered_text(frame, 'Waiting for odometry...', (190, 190, 190))
            self.draw_panel_header(frame, panel)
            return

        min_xy = np.min(all_points, axis=0)
        max_xy = np.max(all_points, axis=0)
        center = 0.5 * (min_xy + max_xy)
        span = np.maximum(max_xy - min_xy, np.array([1.0, 1.0], dtype=np.float32))
        usable_w = max(1, width - 2 * margin)
        usable_h = max(1, height - 2 * margin)
        scale = min(float(usable_w) / float(span[0]), float(usable_h) / float(span[1]))
        if self.args.min_meter_span > 0.0:
            scale = min(scale, float(usable_w) / float(self.args.min_meter_span))
            scale = min(scale, float(usable_h) / float(self.args.min_meter_span))
        if self.args.meters_per_pixel > 0.0:
            scale = 1.0 / float(self.args.meters_per_pixel)
        scale *= float(self.args.zoom)

        def world_to_px(points: np.ndarray) -> np.ndarray:
            px = np.empty_like(points, dtype=np.int32)
            px[:, 0] = np.round((points[:, 0] - center[0]) * scale + width * 0.5).astype(np.int32)
            px[:, 1] = np.round(height * 0.5 - (points[:, 1] - center[1]) * scale).astype(np.int32)
            return px

        self.draw_grid(frame, center, scale, margin)
        if raw_points.shape[0] > 0:
            self.draw_trajectory(frame, raw_points, world_to_px(raw_points), self.raw_color)
        if fusion_points.shape[0] > 0:
            self.draw_trajectory(frame, fusion_points, world_to_px(fusion_points), self.fusion_color)
        self.draw_scale_bar(frame, scale)
        self.draw_panel_header(frame, panel)

    def draw_grid(self, frame: np.ndarray, center: np.ndarray, scale: float, margin: int):
        if scale <= 1e-9:
            return
        height, width = frame.shape[:2]
        grid_color = parse_color(self.args.grid_color)
        axis_color = parse_color(self.args.axis_color)
        meters_per_grid = nice_grid_spacing(float(self.args.grid_px) / scale)
        if meters_per_grid <= 0.0:
            return
        x_min = center[0] - width * 0.5 / scale
        x_max = center[0] + width * 0.5 / scale
        y_min = center[1] - height * 0.5 / scale
        y_max = center[1] + height * 0.5 / scale

        x = math.floor(x_min / meters_per_grid) * meters_per_grid
        while x <= x_max:
            px = int(round((x - center[0]) * scale + width * 0.5))
            color = axis_color if abs(x) < 1e-6 else grid_color
            cv2.line(frame, (px, margin), (px, height - margin), color, 1, cv2.LINE_AA)
            x += meters_per_grid

        y = math.floor(y_min / meters_per_grid) * meters_per_grid
        while y <= y_max:
            py = int(round(height * 0.5 - (y - center[1]) * scale))
            color = axis_color if abs(y) < 1e-6 else grid_color
            cv2.line(frame, (margin, py), (width - margin, py), color, 1, cv2.LINE_AA)
            y += meters_per_grid

    def draw_trajectory(
        self,
        frame: np.ndarray,
        points_world: np.ndarray,
        points_px: np.ndarray,
        color: tuple[int, int, int],
    ):
        if points_px.shape[0] >= 2:
            jump_m = float(self.args.break_jump_meters)
            if jump_m > 0.0:
                segment_start = 0
                for idx in range(1, points_px.shape[0]):
                    if float(np.linalg.norm(points_world[idx] - points_world[idx - 1])) > jump_m:
                        if idx - segment_start >= 2:
                            cv2.polylines(frame, [points_px[segment_start:idx]], False, color, int(self.args.line_width), cv2.LINE_AA)
                        segment_start = idx
                if points_px.shape[0] - segment_start >= 2:
                    cv2.polylines(frame, [points_px[segment_start:]], False, color, int(self.args.line_width), cv2.LINE_AA)
            else:
                cv2.polylines(frame, [points_px], False, color, int(self.args.line_width), cv2.LINE_AA)
        current = tuple(points_px[-1].tolist())
        cv2.circle(frame, current, int(self.args.dot_radius), color, -1, cv2.LINE_AA)
        cv2.circle(frame, current, int(self.args.dot_radius) + 2, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_scale_bar(self, frame: np.ndarray, scale: float):
        if scale <= 1e-9:
            return
        height, width = frame.shape[:2]
        bar_m = nice_grid_spacing(90.0 / scale)
        bar_px = int(round(bar_m * scale))
        x0 = width - int(self.args.margin) - bar_px
        x1 = x0 + bar_px
        y = height - 22
        cv2.line(frame, (x0, y), (x1, y), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.line(frame, (x0, y - 5), (x0, y + 5), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y - 5), (x1, y + 5), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(frame, f'{bar_m:g} m', (x0, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

    def draw_panel_header(self, frame: np.ndarray, panel: PanelState):
        x = 14
        y = 24
        cv2.putText(frame, panel.config.title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, f'raw: {len(panel.raw)} pts', (x, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.raw_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f'fusion: {len(panel.fusion)} pts', (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.fusion_color, 2, cv2.LINE_AA)
        if self.args.show_topics:
            cv2.putText(frame, panel.config.raw_topic, (x, y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.raw_color, 1, cv2.LINE_AA)
            cv2.putText(frame, panel.config.fusion_topic, (x, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.fusion_color, 1, cv2.LINE_AA)

    def close(self):
        final_panels = self.snapshot_panels()
        for panel in final_panels:
            self.get_logger().info(
                '%s received raw=%d fusion=%d points'
                % (panel.config.title, len(panel.raw), len(panel.fusion))
            )
        frame = self.render_frame(final_panels)
        self.writer.write(frame)
        self.frame_count += 1
        self.writer.release()
        self.get_logger().info(f'Wrote {self.frame_count} frames to {self.args.output}')


def trajectory_to_points(trajectory: Trajectory) -> np.ndarray:
    if not trajectory:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([(x, y) for _, x, y in trajectory], dtype=np.float32)


def normalize_panel_points(
    raw_points: np.ndarray,
    fusion_points: np.ndarray,
    origin_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw = raw_points.copy()
    fusion = fusion_points.copy()

    def first_point(points: np.ndarray) -> np.ndarray | None:
        if points.shape[0] == 0:
            return None
        return points[0].copy()

    if origin_mode == 'absolute':
        return raw, fusion
    if origin_mode == 'each_first':
        raw_origin = first_point(raw)
        fusion_origin = first_point(fusion)
        if raw_origin is not None:
            raw -= raw_origin
        if fusion_origin is not None:
            fusion -= fusion_origin
        return raw, fusion
    if origin_mode == 'raw_first':
        origin = first_point(raw)
    elif origin_mode == 'fusion_first':
        origin = first_point(fusion)
    else:
        raise ValueError(f'Unknown origin mode: {origin_mode}')

    if origin is not None:
        raw -= origin
        fusion -= origin
    return raw, fusion


def split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def build_panels(args: argparse.Namespace) -> list[PanelConfig]:
    if args.run_fusions:
        raw_topic = args.raw_odom_topic or args.odom_a
        variants = resolve_variants(args.fusion_variants)
        labels = split_csv(args.panel_labels)
        if labels and len(labels) != len(variants):
            raise ValueError('--panel-labels must match the number of --fusion-variants when --run-fusions is used.')
        return [
            PanelConfig(
                title=labels[i] if labels else variant.title,
                raw_topic=raw_topic,
                fusion_topic=fusion_output_topic(args.fusion_namespace_prefix, variant.name),
            )
            for i, variant in enumerate(variants)
        ]

    raw_topics = split_csv(args.raw_topics)
    fusion_topics = split_csv(args.fusion_topics)
    labels = split_csv(args.panel_labels)

    if not raw_topics and not fusion_topics:
        raw_topics = [args.odom_a]
        fusion_topics = [args.odom_b]
        labels = [args.panel_labels or 'odom comparison']

    if not raw_topics:
        raw_topics = [args.odom_a]
    if not fusion_topics:
        raise ValueError('At least one fusion topic is required.')
    if len(raw_topics) == 1 and len(fusion_topics) > 1:
        raw_topics = raw_topics * len(fusion_topics)
    if len(raw_topics) != len(fusion_topics):
        raise ValueError('--raw-topics must have length 1 or the same length as --fusion-topics.')
    if not labels:
        labels = [f'fusion {idx + 1}' for idx in range(len(fusion_topics))]
    if len(labels) == 1 and len(fusion_topics) > 1:
        labels = [f'{labels[0]} {idx + 1}' for idx in range(len(fusion_topics))]
    if len(labels) != len(fusion_topics):
        raise ValueError('--panel-labels must have length 1 or the same length as --fusion-topics.')

    return [
        PanelConfig(title=labels[i], raw_topic=raw_topics[i], fusion_topic=fusion_topics[i])
        for i in range(len(fusion_topics))
    ]


def resolve_variants(value: str) -> list[FusionVariant]:
    names = split_csv(value)
    if not names:
        names = ['alignment4', 'alignment10', 'pose_graph']
    variants = []
    for name in names:
        if name not in FUSION_VARIANTS:
            valid = ', '.join(sorted(FUSION_VARIANTS))
            raise ValueError(f'Unknown fusion variant {name!r}. Valid variants: {valid}')
        variants.append(FUSION_VARIANTS[name])
    return variants


def fusion_output_topic(namespace_prefix: str, variant_name: str) -> str:
    namespace = fusion_namespace(namespace_prefix, variant_name)
    return f'/{namespace}/globalEstimator/global_odometry'


def fusion_namespace(namespace_prefix: str, variant_name: str) -> str:
    prefix = namespace_prefix.strip('/_')
    return f'{prefix}_{variant_name}' if prefix else variant_name


def param_value(value: object) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def start_fusion_processes(args: argparse.Namespace) -> list[subprocess.Popen]:
    if not args.run_fusions:
        return []

    raw_topic = args.raw_odom_topic or args.odom_a
    gps_topic = args.gps_topic
    variants = resolve_variants(args.fusion_variants)
    script_path = os.path.abspath(args.fusion_node_script)
    output_dir = os.path.dirname(os.path.abspath(args.output)) or '/tmp'
    processes: list[subprocess.Popen] = []

    for variant in variants:
        namespace = fusion_namespace(args.fusion_namespace_prefix, variant.name)
        csv_path = os.path.join(output_dir, f'{namespace}_vio_global.csv')
        cmd = [
            sys.executable,
            script_path,
            '--ros-args',
            '-r',
            f'__ns:=/{namespace}',
            '-p',
            f'vio_topic:={raw_topic}',
            '-p',
            f'gps_topic:={gps_topic}',
            '-p',
            f'vio_global_csv_path:={csv_path}',
            '-p',
            'route_poll_sec:=0.0',
            '-p',
            'path_publish_period_sec:=0.0',
        ]
        if args.fusion_use_sim_time:
            cmd.extend(['-p', 'use_sim_time:=true'])
        if args.fusion_optimization_period_sec > 0.0:
            cmd.extend(['-p', f'optimization_period_sec:={args.fusion_optimization_period_sec}'])
        for key, value in variant.params.items():
            cmd.extend(['-p', f'{key}:={param_value(value)}'])

        log_path = os.path.join(output_dir, f'{namespace}.log')
        log_file = open(log_path, 'w', encoding='utf-8')
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        process._codex_log_file = log_file  # type: ignore[attr-defined]
        processes.append(process)
        print(
            f'Started fusion variant {variant.name}: namespace=/{namespace}, '
            f'output={fusion_output_topic(args.fusion_namespace_prefix, variant.name)}, log={log_path}',
            flush=True,
        )

    if args.fusion_startup_delay_sec > 0.0:
        time.sleep(float(args.fusion_startup_delay_sec))
    return processes


def stop_fusion_processes(processes: list[subprocess.Popen]):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 3.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
        log_file = getattr(process, '_codex_log_file', None)
        if log_file is not None:
            log_file.close()


def parse_color(value: str) -> tuple[int, int, int]:
    named = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'red': (40, 40, 230),
        'green': (70, 210, 90),
        'blue': (230, 120, 40),
        'yellow': (40, 220, 235),
        'cyan': (230, 210, 40),
        'magenta': (220, 80, 220),
        'gray': (80, 80, 80),
        'dark': (22, 24, 28),
    }
    value = value.strip().lower()
    if value in named:
        return named[value]
    if value.startswith('#') and len(value) == 7:
        r = int(value[1:3], 16)
        g = int(value[3:5], 16)
        b = int(value[5:7], 16)
        return (b, g, r)
    parts = [int(part) for part in value.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f'Invalid color: {value}')
    return (parts[0], parts[1], parts[2])


def nice_grid_spacing(target_m: float) -> float:
    if target_m <= 0.0 or not math.isfinite(target_m):
        return 1.0
    exponent = math.floor(math.log10(target_m))
    base = 10.0 ** exponent
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        spacing = multiplier * base
        if spacing >= target_m:
            return spacing
    return 10.0 * base


def draw_centered_text(frame: np.ndarray, text: str, color: tuple[int, int, int]):
    height, width = frame.shape[:2]
    scale = 0.65
    thickness = 2
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (width - size[0]) // 2
    y = (height + size[1]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Subscribe to raw/fusion odometry topic pairs and render XY paths into one multi-panel video.',
    )
    parser.add_argument('--odom-a', default='/aft_mapped_to_init', help='Legacy single-panel raw odom topic.')
    parser.add_argument('--odom-b', default='/globalEstimator/global_odometry', help='Legacy single-panel fusion odom topic.')
    parser.add_argument('--raw-topics', default=None, help='Comma-separated raw odom topics. One topic can be reused for all fusion topics.')
    parser.add_argument('--fusion-topics', default=None, help='Comma-separated fusion odom topics, one per panel.')
    parser.add_argument('--panel-labels', default=None, help='Comma-separated panel titles.')
    parser.add_argument('--run-fusions', action='store_true', help='Start fusion variants inside this script and subscribe to their outputs.')
    parser.add_argument('--raw-odom-topic', default=None, help='Raw odom topic used by internal fusion nodes. Defaults to --odom-a.')
    parser.add_argument('--gps-topic', default='/gps/fix', help='GPS topic used by internal fusion nodes.')
    parser.add_argument('--fusion-node-script', default=None, help='Path to ros2_vins_fusion_node.py. Defaults to this script directory.')
    parser.add_argument('--fusion-variants', default='alignment4,alignment10,pose_graph', help='Comma-separated variants: alignment4, alignment10, alignment_live, pose_graph, old_python.')
    parser.add_argument('--fusion-namespace-prefix', default='vf', help='Namespace prefix for internal fusion nodes.')
    parser.add_argument('--fusion-use-sim-time', action='store_true', help='Pass use_sim_time:=true to internal fusion nodes.')
    parser.add_argument('--fusion-optimization-period-sec', type=float, default=0.5, help='Optimization period for internal fusion nodes.')
    parser.add_argument('--fusion-startup-delay-sec', type=float, default=2.0, help='Delay after starting internal fusion nodes before recording video.')
    parser.add_argument('--label-a', default=None, help='Deprecated alias; use --raw-topics and panel labels.')
    parser.add_argument('--label-b', default=None, help='Deprecated alias; use --fusion-topics and panel labels.')
    parser.add_argument('--output', default='/tmp/multi_odom_paths.mp4', help='Output video path.')
    parser.add_argument('--fps', type=float, default=20.0, help='Output video FPS.')
    parser.add_argument('--width', type=int, default=1920, help='Video width.')
    parser.add_argument('--height', type=int, default=720, help='Video height.')
    parser.add_argument('--columns', type=int, default=0, help='Panel columns. 0 means all panels in one row.')
    parser.add_argument('--fourcc', default='mp4v', help='OpenCV fourcc, e.g. mp4v or XVID.')
    parser.add_argument('--line-width', type=int, default=3, help='Trajectory line width.')
    parser.add_argument('--dot-radius', type=int, default=5, help='Current pose dot radius.')
    parser.add_argument('--margin', type=int, default=44, help='Pane margin in pixels.')
    parser.add_argument('--grid-px', type=float, default=80.0, help='Approximate grid spacing in pixels.')
    parser.add_argument('--zoom', type=float, default=0.90, help='Auto-fit zoom factor.')
    parser.add_argument('--meters-per-pixel', type=float, default=0.0, help='Fixed scale override. 0 means auto fit.')
    parser.add_argument('--min-meter-span', type=float, default=10.0, help='Minimum visible world span in meters.')
    parser.add_argument('--max-points', type=int, default=0, help='Maximum points per path. 0 keeps all.')
    parser.add_argument('--frame-on-message', action='store_true', help='Render on odom callbacks, throttled by FPS.')
    parser.add_argument(
        '--origin-mode',
        choices=('absolute', 'each_first', 'raw_first', 'fusion_first'),
        default='each_first',
        help='Coordinate origin for drawing. each_first overlays raw/fusion shapes even when frames differ.',
    )
    parser.add_argument(
        '--break-jump-meters',
        type=float,
        default=20.0,
        help='Do not draw connecting line segments longer than this many meters. 0 disables.',
    )
    parser.add_argument('--raw-color', '--color-a', dest='raw_color', default='blue', help='Raw odom path color.')
    parser.add_argument('--fusion-color', '--color-b', dest='fusion_color', default='red', help='Fusion odom path color.')
    parser.add_argument('--background', default='dark', help='Background color.')
    parser.add_argument('--grid-color', default='45,48,54', help='Grid color in B,G,R.')
    parser.add_argument('--axis-color', default='85,90,100', help='Axis color in B,G,R.')
    parser.add_argument('--panel-border', default='70,75,84', help='Panel border color in B,G,R.')
    parser.add_argument('--show-topics', action='store_true', help='Draw topic names in each panel.')
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args, ros_args = parser.parse_known_args(argv)
    if args.fusion_node_script is None:
        args.fusion_node_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ros2_vins_fusion_node.py')
    try:
        panels = build_panels(args)
    except ValueError as exc:
        parser.error(str(exc))

    fusion_processes = start_fusion_processes(args)
    rclpy.init(args=ros_args)
    node = MultiOdomPathVideoNode(args, panels)

    def request_stop(_signum, _frame):
        node.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        stop_fusion_processes(fusion_processes)


if __name__ == '__main__':
    main(sys.argv[1:])
