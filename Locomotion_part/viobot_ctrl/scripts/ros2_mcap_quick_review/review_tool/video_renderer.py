from __future__ import annotations
import bisect
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from .extractors import quaternion_to_yaw

@dataclass
class TimedFrame:
    t: float
    path: str

@dataclass
class TimedPoint:
    t: float
    x: float
    y: float
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

class VideoSink:
    def __init__(
        self,
        output: Path,
        width: int,
        height: int,
        fps: float,
        cfg: dict,
    ):
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps

        self.proc = None
        self.cv_writer = None

        # 不再优先使用 Synology 系统自带 ffmpeg。
        # 只使用 imageio-ffmpeg 提供的完整静态二进制。
        ffmpeg = None

        if str(cfg.get("video_encoder", "auto")).lower() != "opencv":
            try:
                import imageio_ffmpeg

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None

        if ffmpeg:
            cmd = [
                ffmpeg,
                "-y",
                "-nostdin",
                "-loglevel",
                "error",

                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",

                "-s",
                "{}x{}".format(width, height),
                "-r",
                str(fps),
                "-i",
                "-",

                "-an",
                "-c:v",
                "libx264",

                "-preset",
                str(cfg.get("ffmpeg_preset", "ultrafast")),

                "-crf",
                str(cfg.get("ffmpeg_crf", 28)),

                "-threads",
                str(cfg.get("ffmpeg_threads", 1)),

                "-pix_fmt",
                "yuv420p",

                "-movflags",
                "+faststart",

                str(output),
            ]

            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

        else:
            # imageio-ffmpeg 不可用时，退回 OpenCV。
            self.cv_writer = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            if not self.cv_writer.isOpened():
                raise RuntimeError(
                    "无法打开视频编码器。请安装 imageio-ffmpeg，"
                    "或检查 OpenCV 是否支持 mp4v。"
                )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                "Frame size does not match writer: expected {}x{}, got {}x{}".format(
                    self.width,
                    self.height,
                    frame.shape[1],
                    frame.shape[0],
                )
            )

        if self.proc is not None:
            try:
                self.proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                message = ""

                if self.proc.stderr is not None:
                    try:
                        message = self.proc.stderr.read().decode(
                            "utf-8",
                            errors="replace",
                        )
                    except Exception:
                        pass

                raise RuntimeError(
                    "ffmpeg pipe failed: {}".format(message.strip() or exc)
                )
        else:
            self.cv_writer.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.close()

            code = self.proc.wait()

            message = ""
            if self.proc.stderr is not None:
                try:
                    message = self.proc.stderr.read().decode(
                        "utf-8",
                        errors="replace",
                    ).strip()
                except Exception:
                    pass

                self.proc.stderr.close()

            if code != 0:
                raise RuntimeError(
                    "ffmpeg exited with code {}{}".format(
                        code,
                        ": " + message if message else "",
                    )
                )

        elif self.cv_writer is not None:
            self.cv_writer.release()

def _nearest_index(times: list[float], t: float) -> int | None:
    if not times:
        return None
    pos = bisect.bisect_left(times, t)
    candidates = ([pos] if pos < len(times) else []) + ([pos - 1] if pos > 0 else [])
    return min(candidates, key=lambda idx: abs(times[idx] - t))

def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas

def _put_text(image, text, xy, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def _map_points(points: np.ndarray, width: int, height: int, margin: int = 36) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)
    min_xy, max_xy = points.min(axis=0), points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min(max(1, width - 2 * margin) / span[0], max(1, height - 2 * margin) / span[1])
    center = (min_xy + max_xy) / 2.0
    mapped = (points - center) * scale
    mapped[:, 0] += width / 2.0
    mapped[:, 1] = height / 2.0 - mapped[:, 1]
    return np.rint(mapped).astype(np.int32)

def _draw_trajectory_panel(
    width, height, title, points, t, max_gap, draw_heading,
    heading_yaw_offset_deg=0.0,
):
    panel = np.full((height, width, 3), 20, dtype=np.uint8)
    _put_text(panel, title, (16, 28), 0.65, (255, 255, 255), 2)
    if not points:
        _put_text(panel, "NO TOPIC / NO VALID DATA", (25, height // 2), 0.65, (0, 0, 255), 2)
        return panel, None
    times = [p.t for p in points]
    coords = np.asarray([[p.x, p.y] for p in points], dtype=np.float64)
    # Keep the trajectory clear of both the title and the sample timestamp.
    mapped = _map_points(coords, width, height, margin=62)
    if len(mapped) >= 2:
        cv2.polylines(panel, [mapped.reshape(-1, 1, 2)], False, (90, 90, 90), 1)
    past_end = bisect.bisect_right(times, t)
    if past_end >= 2:
        cv2.polylines(panel, [mapped[:past_end].reshape(-1, 1, 2)], False, (0, 220, 255), 2)
    idx = _nearest_index(times, t)
    delta = times[idx] - t
    current = tuple(int(v) for v in mapped[idx])
    valid = abs(delta) <= max_gap
    if valid:
        cv2.circle(panel, current, 7, (0, 255, 0), -1)
        if draw_heading:
            p = points[idx]
            # A ROS quaternion describes the message's body +X axis.  Some VIO
            # devices (including Stereo3 here) move forward along body +Y, so
            # the displayed vehicle heading needs a configurable yaw offset.
            yaw = quaternion_to_yaw(p.qx, p.qy, p.qz, p.qw)
            yaw += math.radians(float(heading_yaw_offset_deg))
            end = (int(current[0] + 28 * math.cos(yaw)), int(current[1] - 28 * math.sin(yaw)))
            cv2.arrowedLine(panel, current, end, (255, 180, 0), 2, tipLength=0.25)
    else:
        _put_text(panel, "MISSING / STALE", (25, height // 2), 0.75, (0, 0, 255), 2)
    _put_text(
        panel, f"sample dt: {delta:+.3f} s", (16, 52), 0.5,
        (180, 180, 180) if valid else (0, 0, 255), 1,
    )
    return panel, delta

def render_review_video(output_path, frames, gps, vio, duration_sec, cfg, metadata_lines):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(cfg.get("output_width", 1280)), int(cfg.get("output_height", 720))
    fps, speed = float(cfg.get("output_fps", 10)), float(cfg.get("playback_speed", 2.0))
    if width < 960 or height < 540 or fps <= 0 or speed <= 0:
        raise ValueError("Invalid output size/fps/playback speed")
    camera_w, right_w = int(width * 0.63), width - int(width * 0.63)
    gps_h, vio_h = height // 2, height - height // 2
    frame_times = [item.t for item in frames]
    camera_gap = float(cfg.get("camera_max_gap_sec", 0.25))
    gps_gap, vio_gap = float(cfg.get("gps_max_gap_sec", 1.0)), float(cfg.get("vio_max_gap_sec", 0.2))
    vio_heading_offset = float(cfg.get("vio_heading_yaw_offset_deg", 0.0))
    duration_sec = max(float(duration_sec), 0.1)
    output_duration = duration_sec / speed
    output_count = max(1, int(math.ceil(output_duration * fps)))
    sink = VideoSink(output_path, width, height, fps, cfg)
    cached_path, cached_image = "", None
    try:
        for frame_idx in range(output_count):
            data_t = min(duration_sec, frame_idx / fps * speed)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            camera_panel = np.zeros((height, camera_w, 3), dtype=np.uint8)
            camera_delta = None
            idx = _nearest_index(frame_times, data_t)
            if idx is not None:
                item = frames[idx]
                camera_delta = item.t - data_t
                if abs(camera_delta) <= camera_gap:
                    if item.path != cached_path:
                        cached_image, cached_path = cv2.imread(item.path, cv2.IMREAD_COLOR), item.path
                    if cached_image is not None:
                        camera_panel = _fit_image(cached_image, camera_w, height)
            if idx is None:
                _put_text(camera_panel, "CAMERA TOPIC MISSING", (40, height // 2), 0.9, (0, 0, 255), 2)
            elif camera_delta is None or abs(camera_delta) > camera_gap:
                _put_text(camera_panel, "CAMERA DATA MISSING / STALE", (40, height // 2), 0.85, (0, 0, 255), 2)
            overlay = camera_panel.copy()
            # Browser video controls cover the bottom of the frame while the
            # pointer is over the player.  Keep all review metadata above them.
            cv2.rectangle(overlay, (0, height - 170), (camera_w, height), (0, 0, 0), -1)
            camera_panel = cv2.addWeighted(overlay, 0.68, camera_panel, 0.32, 0)
            _put_text(
                camera_panel,
                f"DATA TIME {data_t:7.2f} / {duration_sec:.2f} s   PLAYBACK {speed:g}x",
                (18, height - 140), 0.62, (255, 255, 255), 2,
            )
            dt_text = "N/A" if camera_delta is None else f"{camera_delta:+.3f} s"
            dt_color = (0, 0, 255) if camera_delta is None or abs(camera_delta) > camera_gap else (180, 255, 180)
            _put_text(camera_panel, f"Camera sample dt: {dt_text}", (18, height - 112), 0.55, dt_color, 1)
            for line_idx, line in enumerate(metadata_lines[:2]):
                _put_text(
                    camera_panel, line[:92],
                    (18, height - 82 + line_idx * 23),
                    0.45, (210, 210, 210), 1,
                )
            gps_panel, _ = _draw_trajectory_panel(right_w, gps_h, "FUSED GPS TRAJECTORY", gps, data_t, gps_gap, False)
            vio_panel, _ = _draw_trajectory_panel(
                right_w, vio_h, "VIO / ODOMETRY TRAJECTORY", vio,
                data_t, vio_gap, True, vio_heading_offset,
            )
            canvas[:, :camera_w], canvas[:gps_h, camera_w:], canvas[gps_h:, camera_w:] = camera_panel, gps_panel, vio_panel
            cv2.line(canvas, (camera_w, 0), (camera_w, height), (80, 80, 80), 1)
            cv2.line(canvas, (camera_w, gps_h), (width, gps_h), (80, 80, 80), 1)
            sink.write(canvas)
    finally:
        sink.close()
    return {
        "video_frame_count": output_count, "video_duration_sec": output_duration,
        "output_fps": fps, "playback_speed": speed,
        "output_width": width, "output_height": height,
    }
