#!/usr/bin/env python3
"""FrodoBot System2 数据集生成器

基于 FrodoBot 数据集生成训练 System2 所需的数据，包括：
- FPV 图像
- BEV 地图图像
- 导航指令
- 视觉提示（未来轨迹投影）

数据集格式：
data/frodobot_system2/
├── train.jsonl
├── val.jsonl
└── images/
    ├── ride_xxx_frame_xxx_fpv.jpg
    ├── ride_xxx_frame_xxx_bev.jpg
    └── ride_xxx_frame_xxx_prompt.jpg
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加 RouteServer 项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from mcp_server.map_providers import create_map_client
from mcp_server.route_context.bev_map_provider import BEVMapProviderSession
from mcp_server.route_context.build_bev import build_local_bev_context
from mcp_server.route_context.common import bearing_deg, normalize_route_polyline


# ==================== 数据结构定义 ====================
@dataclass
class GPSPoint:
    lat: float
    lng: float
    timestamp_ms: int


@dataclass
class VideoFrame:
    frame_id: int
    timestamp_sec: float
    gps: GPSPoint | None = None
    heading_deg: float | None = None
    future_trajectory: list[GPSPoint] | None = None


@dataclass
class RideDataset:
    ride_id: str
    gps_points: list[GPSPoint]
    video_frames: list[VideoFrame]
    recordings_dir: Path
    output_dir: Path


# ==================== CSV 数据解析 ====================
def parse_gps_csv(csv_path: Path) -> list[GPSPoint]:
    """解析 GPS CSV 文件"""
    points: list[GPSPoint] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["latitude"])
                lng = float(row["longitude"])
                ts_ms = int(float(row["timestamp"]))
                points.append(GPSPoint(lat=lat, lng=lng, timestamp_ms=ts_ms))
            except (ValueError, KeyError) as e:
                print(f"[WARN] 跳过无效 GPS 行: {row}, 错误: {e}")
    return sorted(points, key=lambda p: p.timestamp_ms)


def parse_video_timestamps_csv(csv_path: Path) -> list[VideoFrame]:
    """解析视频时间戳 CSV 文件"""
    frames: list[VideoFrame] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(row["frame_id"])
                ts_sec = float(row["timestamp"])
                frames.append(VideoFrame(frame_id=frame_id, timestamp_sec=ts_sec))
            except (ValueError, KeyError) as e:
                print(f"[WARN] 跳过无效视频帧: {row}, 错误: {e}")
    return sorted(frames, key=lambda f: f.timestamp_sec)


# ==================== GPS 插值与时间同步 ====================
def interpolate_gps(timestamp_ms: int, gps_points: list[GPSPoint]) -> GPSPoint | None:
    """为给定时间戳插值 GPS 坐标"""
    if not gps_points:
        return None

    # 精确匹配
    for p in gps_points:
        if p.timestamp_ms == timestamp_ms:
            return p

    # 边界处理
    if timestamp_ms < gps_points[0].timestamp_ms:
        return gps_points[0]
    if timestamp_ms > gps_points[-1].timestamp_ms:
        return gps_points[-1]

    # 二分查找
    left, right = 0, len(gps_points) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if gps_points[mid].timestamp_ms < timestamp_ms:
            left = mid
        else:
            right = mid

    p1, p2 = gps_points[left], gps_points[right]
    if p1.timestamp_ms == p2.timestamp_ms:
        return p1

    # 线性插值
    t1, t2 = p1.timestamp_ms, p2.timestamp_ms
    alpha = (timestamp_ms - t1) / (t2 - t1)
    lat = p1.lat + alpha * (p2.lat - p1.lat)
    lng = p1.lng + alpha * (p2.lng - p1.lng)

    return GPSPoint(lat=lat, lng=lng, timestamp_ms=timestamp_ms)


def estimate_heading(gps_points: list[GPSPoint], point_index: int, lookahead: int = 3) -> float | None:
    """从 GPS 轨迹估计航向（度）"""
    if not gps_points or point_index < 0 or point_index >= len(gps_points):
        return None

    start_idx = point_index
    end_idx = min(point_index + lookahead, len(gps_points) - 1)

    # 找到有足够距离的点
    start_point = gps_points[start_idx]
    for i in range(end_idx, start_idx, -1):
        end_point = gps_points[i]
        # 检查距离是否足够（约 1 米）
        dist_m = haversine_m(start_point.lat, start_point.lng, end_point.lat, end_point.lng)
        if dist_m > 1.0:
            return bearing_deg(start_point.lat, start_point.lng, end_point.lat, end_point.lng)

    # 如果没有足够的向前点，尝试向后看
    back_idx = max(0, point_index - lookahead)
    for i in range(back_idx, point_index):
        end_point = gps_points[i]
        dist_m = haversine_m(start_point.lat, start_point.lng, end_point.lat, end_point.lng)
        if dist_m > 1.0:
            # 反向航向
            h = bearing_deg(end_point.lat, end_point.lng, start_point.lat, start_point.lng)
            return h

    return None


def extract_future_trajectory(
    gps_points: list[GPSPoint],
    from_timestamp_ms: int,
    duration_sec: float = 5.0,
    max_points: int = 20,
) -> list[GPSPoint]:
    """提取未来轨迹"""
    end_timestamp_ms = from_timestamp_ms + int(duration_sec * 1000)
    trajectory: list[GPSPoint] = []

    for p in gps_points:
        if p.timestamp_ms < from_timestamp_ms:
            continue
        if p.timestamp_ms > end_timestamp_ms:
            break
        trajectory.append(p)
        if len(trajectory) >= max_points:
            break

    return trajectory


def sync_video_frames_with_gps(
    video_frames: list[VideoFrame],
    gps_points: list[GPSPoint],
) -> list[VideoFrame]:
    """同步视频帧与 GPS"""
    synced: list[VideoFrame] = []
    for i, frame in enumerate(video_frames):
        ts_ms = int(frame.timestamp_sec * 1000)
        gps = interpolate_gps(ts_ms, gps_points)
        if gps is None:
            continue

        # 找到对应 GPS 点的索引用于航向估计
        gps_idx = 0
        for j, p in enumerate(gps_points):
            if p.timestamp_ms >= ts_ms:
                gps_idx = max(0, j - 1)
                break

        heading = estimate_heading(gps_points, gps_idx)
        future_traj = extract_future_trajectory(gps_points, ts_ms)

        new_frame = VideoFrame(
            frame_id=frame.frame_id,
            timestamp_sec=frame.timestamp_sec,
            gps=gps,
            heading_deg=heading,
            future_trajectory=future_traj,
        )
        synced.append(new_frame)
    return synced


# ==================== 工具函数 ====================
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两个 GPS 点之间的距离（米）"""
    radius_m = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


# ==================== 视频解码 ====================
def find_video_files(recordings_dir: Path) -> list[Path]:
    """查找视频文件"""
    m3u8_files = list(recordings_dir.glob("*_video.m3u8"))
    if not m3u8_files:
        ts_files = sorted(recordings_dir.glob("*_video_*.ts"))
        return ts_files
    return m3u8_files


def read_m3u8_playlist(m3u8_path: Path) -> list[Path]:
    """读取 m3u8 播放列表"""
    ts_files: list[Path] = []
    base_dir = m3u8_path.parent
    with open(m3u8_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ts_path = base_dir / line
                if ts_path.exists():
                    ts_files.append(ts_path)
    return ts_files


def extract_video_frame(
    recordings_dir: Path,
    frame_id: int,
    target_timestamp_sec: float | None = None,
) -> np.ndarray | None:
    """从视频中提取指定帧"""
    video_files = find_video_files(recordings_dir)
    if not video_files:
        return None

    # 简单策略：尝试第一个视频，或者所有 TS 文件拼接
    ts_files: list[Path] = []
    for vf in video_files:
        if vf.suffix == ".m3u8":
            ts_files.extend(read_m3u8_playlist(vf))
        elif vf.suffix == ".ts":
            ts_files.append(vf)

    if not ts_files:
        return None

    # 尝试逐个打开视频文件
    for ts_path in ts_files:
        cap = cv2.VideoCapture(str(ts_path))
        if not cap.isOpened():
            continue

        # 尝试按时间或按帧号查找
        if target_timestamp_sec is not None:
            # 基于时间查找（如果视频有时间戳元数据）
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                frame_idx = int(target_timestamp_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        cap.release()

        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return None


# ==================== BEV 渲染（复用 RouteServer 代码） ====================
async def render_bev_map(
    gps: GPSPoint,
    heading_deg: float,
    route_polyline: list[dict[str, Any]] | None = None,
    output_size: int = 640,
    zoom: int = 19,
    tile_url_template: str | None = None,
) -> Image.Image | None:
    """渲染 BEV 地图（基于 OpenStreetMap 瓦片）"""
    import math
    from PIL import Image

    # 如果没有提供 route_polyline，使用当前点附近的简化路径
    if route_polyline is None:
        route_polyline = [
            {"lat": gps.lat, "lng": gps.lng},
        ]

    # 构建一个简单的 local_bev_context 用于渲染
    origin_gps = {"lat": gps.lat, "lng": gps.lng}

    # 创建一个模拟的 route_context
    route_context = {
        "pose_match": {"matched_point": {"lat": gps.lat, "lng": gps.lng}},
        "geometry": {
            "local_segment_polyline": [{"lat": gps.lat, "lng": gps.lng}],
            "connector_polyline": [],
        },
        "progress": {"next_waypoint_index": 0},
        "local_semantics": {
            "corridor_type": "front",
            "turn_type": "go_straight",
            "turn_semantics": "go_straight",
        },
    }

    local_bev_context = {
        "_origin_gps": origin_gps,
        "gps": origin_gps,
        "current_pose_bev": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
        "matched_point_bev": {"x_m": 0.0, "y_m": 0.0},
        "route_polyline_local": [],
        "local_segment_bev": [],
        "connector_bev": [],
        "next_waypoint_bev": None,
        "next_turn_point_bev": None,
        "bev_bbox": {"x_min_m": -15, "x_max_m": 15, "y_min_m": -10, "y_max_m": 30},
        "bev_rotation_deg": heading_deg if heading_deg is not None else 0.0,
        "corridor_type": "front",
        "turn_type": "go_straight",
        "turn_semantics": "go_straight",
        "corridor_sector_deg": {"center_deg": 0.0, "span_deg": 60.0},
        "turn_sector_deg": {"center_deg": 0.0, "span_deg": 35.0},
        "lookahead_m": 20.0,
    }

    # 使用 RouteServer 的渲染逻辑
    tile_size = 256
    mosaic_tiles = 5

    if mosaic_tiles % 2 == 0:
        mosaic_tiles += 1

    # 处理 heading 旋转
    convention = os.getenv("BEV_MAP_HEADING_CONVENTION", "compass").strip().lower() or "compass"
    offset_deg = float(os.getenv("BEV_MAP_HEADING_OFFSET_DEG", "0") or 0.0)

    if convention == "compass":
        # 0 = north, 90 = east -> image up = travel direction
        heading_rotation_deg = float(heading_deg if heading_deg is not None else 0.0) + offset_deg
    else:
        heading_rotation_deg = 0.0

    # Web Mercator 瓦片坐标
    def latlon_to_tile_xy(lat: float, lng: float, zoom: int) -> tuple[float, float]:
        n = 2.0**zoom
        x = (lng + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        return x, y

    center_x, center_y = latlon_to_tile_xy(gps.lat, gps.lng, zoom)
    center_px = center_x * tile_size
    center_py = center_y * tile_size
    center_tile_x = int(math.floor(center_x))
    center_tile_y = int(math.floor(center_y))
    half_tiles = mosaic_tiles // 2

    # 创建镶嵌图
    mosaic = Image.new("RGB", (mosaic_tiles * tile_size, mosaic_tiles * tile_size), (242, 239, 233))

    # 获取瓦片
    template = tile_url_template or os.getenv("BEV_MAP_TILE_URL_TEMPLATE", "https://tile.openstreetmap.org/{z}/{x}/{y}.png")
    user_agent = os.getenv("BEV_MAP_TILE_USER_AGENT", "UrbanNav/1.0")

    # 缓存目录
    cache_dir = Path(os.getenv("BEV_MAP_TILE_CACHE_DIR", "/tmp/urban_nav_osm_tiles")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_tile(x: int, y: int, z: int) -> Image.Image:
        cache_key = f"{z}_{x}_{y}.png"
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                pass

        url = template.format(z=z, x=x, y=y)
        headers = {"User-Agent": user_agent}

        try:
            timeout_s = float(os.getenv("BEV_MAP_TILE_TIMEOUT_S", "8") or 8)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=timeout_s) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        try:
                            img.save(cache_path, "PNG")
                        except Exception:
                            pass
                        return img
        except Exception as e:
            print(f"[WARN] 瓦片获取失败 {z}/{x}/{y}: {e}")

        return Image.new("RGB", (tile_size, tile_size), (242, 239, 233))

    # 并行获取瓦片
    tasks = []
    tile_coords = []
    for dx in range(-half_tiles, half_tiles + 1):
        for dy in range(-half_tiles, half_tiles + 1):
            tx = center_tile_x + dx
            ty = center_tile_y + dy
            tile_coords.append((tx, ty, dx, dy))
            tasks.append(fetch_tile(tx, ty, zoom))

    tiles = await asyncio.gather(*tasks, return_exceptions=True)

    for (tx, ty, dx, dy), tile in zip(tile_coords, tiles):
        if isinstance(tile, Exception):
            tile = Image.new("RGB", (tile_size, tile_size), (242, 239, 233))
        paste_x = (dx + half_tiles) * tile_size
        paste_y = (dy + half_tiles) * tile_size
        mosaic.paste(tile.resize((tile_size, tile_size)), (paste_x, paste_y))

    # 计算裁剪区域
    local_center_px = center_px - (center_tile_x - half_tiles) * tile_size
    local_center_py = center_py - (center_tile_y - half_tiles) * tile_size
    crop_side = min(mosaic.size) * 0.70

    crop_box = (
        int(round(local_center_px - crop_side / 2.0)),
        int(round(local_center_py - crop_side / 2.0)),
        int(round(local_center_px + crop_side / 2.0)),
        int(round(local_center_py + crop_side / 2.0)),
    )

    # 旋转镶嵌图
    rotated_mosaic = mosaic.rotate(
        float(heading_rotation_deg),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        center=(float(local_center_px), float(local_center_py)),
        fillcolor=(242, 239, 233),
    )

    ego_aligned = rotated_mosaic.crop(crop_box)
    image = ego_aligned.resize((output_size, output_size), Image.Resampling.LANCZOS).convert("RGB")

    # 绘制自我标记
    draw = ImageDraw.Draw(image)
    center = output_size // 2

    # 外圈
    draw.ellipse(
        (center - 12, center - 12, center + 12, center + 12),
        outline=(255, 140, 0),
        width=3,
    )

    # 向前的标记
    tick_length = 20
    draw.line(
        (center, center - 8, center, center - 8 - tick_length),
        fill=(255, 255, 255),
        width=4,
    )

    return image


# ==================== 视觉提示生成 ====================
def project_trajectory_to_fpv(
    trajectory: list[GPSPoint],
    current_gps: GPSPoint,
    heading_deg: float,
    fpv_size: tuple[int, int] = (1280, 720),
) -> list[tuple[float, float]]:
    """将 GPS 轨迹投影到 FPV 图像上（简化针孔相机模型）"""
    projected: list[tuple[float, float]] = []

    # 相机参数（启发式值，可调整）
    h_fov_deg = 90.0  # 水平视场角
    v_fov_deg = 60.0  # 垂直视场角
    camera_height_m = 1.2  # 相机高度
    pitch_deg = 10.0  # 向下倾斜

    w, h = fpv_size

    for point in trajectory:
        # 转换到以自我为中心的局部坐标（东北天坐标系）
        origin_lat, origin_lng = current_gps.lat, current_gps.lng
        mean_lat = math.radians((point.lat + origin_lat) / 2.0)

        east_m = (point.lng - origin_lng) * 111320.0 * max(0.1, math.cos(mean_lat))
        north_m = (point.lat - origin_lat) * 110540.0

        # 按 heading 旋转到车辆坐标系
        theta = math.radians(heading_deg if heading_deg is not None else 0.0)
        x_m = east_m * math.cos(theta) - north_m * math.sin(theta)  # 右
        y_m = east_m * math.sin(theta) + north_m * math.cos(theta)  # 前

        # 只投影前方的点
        if y_m <= 0.1:
            continue

        # 针孔投影
        # 应用向下倾斜
        pitch_rad = math.radians(pitch_deg)
        # 简单的透视投影
        k_h = w / (2.0 * math.tan(math.radians(h_fov_deg) / 2.0))
        k_v = h / (2.0 * math.tan(math.radians(v_fov_deg) / 2.0))

        # 投影到图像平面
        # x_m = 右, y_m = 前, z_m = 下 (相机高度)
        z_m = camera_height_m

        # 调整 for pitch
        y_relative = y_m * math.cos(pitch_rad) - z_m * math.sin(pitch_rad)
        z_relative = y_m * math.sin(pitch_rad) + z_m * math.cos(pitch_rad)

        if y_relative <= 0.1:
            continue

        u = w / 2.0 + (x_m / y_relative) * k_h
        v = h / 2.0 + (z_relative / y_relative) * k_v

        if 0 <= u < w and 0 <= v < h:
            projected.append((u, v))

    return projected


def render_visual_prompt(
    fpv_image: np.ndarray,
    trajectory: list[GPSPoint],
    current_gps: GPSPoint,
    heading_deg: float,
) -> np.ndarray:
    """在 FPV 图像上渲染视觉提示"""
    img = fpv_image.copy()
    h, w = img.shape[:2]

    # 投影轨迹
    projected = project_trajectory_to_fpv(trajectory, current_gps, heading_deg, (w, h))

    if len(projected) < 2:
        return img

    # 绘制路径
    overlay = img.copy()

    # 路径中心线
    pts = np.array(projected, dtype=np.int32)
    cv2.polylines(overlay, [pts], isClosed=False, color=(0, 165, 255), thickness=6, lineType=cv2.LINE_AA)

    # 路径点
    for u, v in projected:
        cv2.circle(overlay, (int(u), int(v)), radius=8, color=(0, 165, 255), thickness=-1, lineType=cv2.LINE_AA)

    # 混合
    return cv2.addWeighted(overlay, 0.7, img, 0.3, 0.0)


# ==================== 数据集生成 ====================
def generate_instruction() -> str:
    """生成导航指令（模板化）"""
    instructions = [
        "Follow the path to the destination.",
        "Navigate along this route.",
        "Proceed forward following the trajectory.",
        "Go to the destination using the path shown.",
        "Drive along the indicated route.",
    ]
    import random
    return random.choice(instructions)


async def process_ride(
    ride_dir: Path,
    output_dir: Path,
    sample_rate: int = 10,  # 每 N 帧采样一次
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    """处理单个 ride 目录"""
    ride_id = ride_dir.name
    print(f"[INFO] 处理 {ride_id}...")

    # 查找 CSV 文件
    gps_csv = None
    front_timestamps_csv = None

    for f in ride_dir.iterdir():
        if f.name.startswith("gps_data") and f.name.endswith(".csv"):
            gps_csv = f
        elif f.name.startswith("front_camera_timestamps") and f.name.endswith(".csv"):
            front_timestamps_csv = f

    if not gps_csv or not front_timestamps_csv:
        print(f"[WARN] 跳过 {ride_id}: 缺少必要的 CSV 文件")
        return []

    recordings_dir = ride_dir / "recordings"
    if not recordings_dir.exists():
        print(f"[WARN] 跳过 {ride_id}: 缺少 recordings 目录")
        return []

    # 解析数据
    gps_points = parse_gps_csv(gps_csv)
    video_frames = parse_video_timestamps_csv(front_timestamps_csv)
    print(f"[INFO] {ride_id}: {len(gps_points)} GPS 点, {len(video_frames)} 视频帧")

    if not gps_points or not video_frames:
        return []

    # 时间同步
    synced_frames = sync_video_frames_with_gps(video_frames, gps_points)
    print(f"[INFO] {ride_id}: 成功同步 {len(synced_frames)} 帧")

    # 采样
    if sample_rate > 1:
        synced_frames = synced_frames[::sample_rate]
    if max_frames is not None:
        synced_frames = synced_frames[:max_frames]

    # 确保图像输出目录存在
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []

    for i, frame in enumerate(synced_frames):
        if frame.gps is None:
            continue

        print(f"[INFO] {ride_id}: 处理帧 {i+1}/{len(synced_frames)} (frame_id={frame.frame_id})")

        # 提取 FPV 图像
        fpv_img = extract_video_frame(
            recordings_dir,
            frame.frame_id,
            frame.timestamp_sec,
        )

        if fpv_img is None:
            print(f"[WARN] {ride_id}: 无法提取帧 {frame.frame_id}，尝试使用占位图")
            # 创建占位图
            fpv_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(fpv_img, "FPV Placeholder", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        h, w = fpv_img.shape[:2]

        # 渲染 BEV 地图
        bev_img = await render_bev_map(
            frame.gps,
            frame.heading_deg if frame.heading_deg is not None else 0.0,
        )

        # 渲染视觉提示
        prompt_img = None
        if frame.future_trajectory and len(frame.future_trajectory) > 2:
            prompt_img = render_visual_prompt(
                fpv_img,
                frame.future_trajectory,
                frame.gps,
                frame.heading_deg if frame.heading_deg is not None else 0.0,
            )
        else:
            prompt_img = fpv_img.copy()

        # 保存图像
        base_name = f"{ride_id}_frame_{frame.frame_id:06d}"

        fpv_path = images_dir / f"{base_name}_fpv.jpg"
        Image.fromarray(fpv_img).save(fpv_path, quality=95)

        bev_path = images_dir / f"{base_name}_bev.jpg"
        bev_img.save(bev_path, quality=95)

        prompt_path = images_dir / f"{base_name}_prompt.jpg"
        Image.fromarray(prompt_img).save(prompt_path, quality=95)

        # 生成样本数据
        samples.append({
            "fpv_image": str(fpv_path.relative_to(output_dir)),
            "bev_image": str(bev_path.relative_to(output_dir)),
            "instruction": generate_instruction(),
            "visual_prompt_image": str(prompt_path.relative_to(output_dir)),
            "visual_prompt_text": "Follow the path indicated by the orange line.",
            "metadata": {
                "ride_id": ride_id,
                "frame_id": frame.frame_id,
                "current_gps": {
                    "lat": frame.gps.lat,
                    "lng": frame.gps.lng,
                },
                "heading_deg": frame.heading_deg,
                "timestamp_sec": frame.timestamp_sec,
            },
        })

    return samples


def save_jsonl(samples: list[dict[str, Any]], output_path: Path):
    """保存为 JSONL 格式"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")
    print(f"[INFO] 已保存 {len(samples)} 样本到 {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="FrodoBot System2 数据集生成器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/ubuntu/WorkSpace/dataset/output_rides_0"),
        help="FrodoBot 数据集根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/frodobot_system2"),
        help="输出目录",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=10,
        help="采样率（每 N 帧采样一次）",
    )
    parser.add_argument(
        "--max-frames-per-ride",
        type=int,
        default=None,
        help="每个 ride 最多处理帧数",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="训练集比例",
    )
    parser.add_argument(
        "--bev-tile-url",
        type=str,
        default=None,
        help="BEV 瓦片 URL 模板，例如: https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    )
    parser.add_argument(
        "--ride-ids",
        type=str,
        nargs="+",
        default=None,
        help="只处理指定的 ride ID（可选）",
    )
    args = parser.parse_args()

    # 设置环境变量
    if args.bev_tile_url:
        os.environ["BEV_MAP_TILE_URL_TEMPLATE"] = args.bev_tile_url

    # 发现 ride 目录
    dataset_dir = args.dataset_dir
    ride_dirs: list[Path] = []

    for entry in dataset_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("ride_"):
            if args.ride_ids and entry.name not in args.ride_ids:
                continue
            ride_dirs.append(entry)

    ride_dirs = sorted(ride_dirs)
    print(f"[INFO] 发现 {len(ride_dirs)} 个 rides")

    # 处理所有 rides
    all_samples: list[dict[str, Any]] = []

    for ride_dir in ride_dirs:
        samples = await process_ride(
            ride_dir,
            args.output_dir,
            sample_rate=args.sample_rate,
            max_frames=args.max_frames_per_ride,
        )
        all_samples.extend(samples)

    if not all_samples:
        print("[ERROR] 没有生成任何样本")
        return 1

    # 分割训练/验证集
    import random
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * args.train_split)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # 保存
    output_dir = args.output_dir
    save_jsonl(train_samples, output_dir / "train.jsonl")
    save_jsonl(val_samples, output_dir / "val.jsonl")

    print(f"\n[DONE]")
    print(f"总样本数: {len(all_samples)}")
    print(f"训练集: {len(train_samples)}")
    print(f"验证集: {len(val_samples)}")
    print(f"输出目录: {output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
