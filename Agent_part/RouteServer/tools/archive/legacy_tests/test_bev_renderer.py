#!/usr/bin/env python3
"""简单的 BEV 渲染测试工具

用于单独测试 BEV 地图渲染功能，支持：
- 输入 GPS 坐标
- 选择地图提供商（OSM/高德等）
- 输出 BEV 图像
"""
from __future__ import annotations

import argparse
import asyncio
import io
import math
import os
import sys
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image, ImageDraw

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


async def render_bev_map_simple(
    lat: float,
    lng: float,
    heading_deg: float = 0.0,
    output_size: int = 640,
    zoom: int = 19,
    tile_url_template: str | None = None,
) -> Image.Image:
    """简单的 BEV 地图渲染"""
    tile_size = 256
    mosaic_tiles = 5

    if mosaic_tiles % 2 == 0:
        mosaic_tiles += 1

    # 处理 heading 旋转
    convention = os.getenv("BEV_MAP_HEADING_CONVENTION", "compass").strip().lower() or "compass"
    offset_deg = float(os.getenv("BEV_MAP_HEADING_OFFSET_DEG", "0") or 0.0)

    if convention == "compass":
        # 0 = north, 90 = east -> image up = travel direction
        heading_rotation_deg = float(heading_deg) + offset_deg
    else:
        heading_rotation_deg = 0.0

    # Web Mercator 瓦片坐标
    def latlon_to_tile_xy(lat: float, lng: float, zoom: int) -> tuple[float, float]:
        n = 2.0**zoom
        x = (lng + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        return x, y

    center_x, center_y = latlon_to_tile_xy(lat, lng, zoom)
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


async def test_from_gps_csv(gps_csv_path: Path, output_dir: Path, sample_rate: int = 10):
    """从 GPS CSV 文件测试 BEV 渲染"""
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)

    points: list[tuple[float, float, float]] = []  # lat, lng, ts
    with open(gps_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["latitude"])
                lng = float(row["longitude"])
                ts = float(row["timestamp"])
                points.append((lat, lng, ts))
            except Exception as e:
                print(f"[WARN] 跳过行: {e}")

    if not points:
        print("[ERROR] 没有有效的 GPS 点")
        return

    print(f"[INFO] 加载了 {len(points)} 个 GPS 点")

    # 采样
    points = points[::sample_rate]

    for i, (lat, lng, ts) in enumerate(points):
        # 计算航向
        heading = 0.0
        if i < len(points) - 1:
            next_lat, next_lng, _ = points[i + 1]
            # 简单的航向计算
            def bearing_deg(lat1, lon1, lat2, lon2):
                y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
                x = (
                    math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
                    - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
                )
                return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

            heading = bearing_deg(lat, lng, next_lat, next_lng)

        print(f"[INFO] 渲染 {i+1}/{len(points)}: lat={lat:.6f}, lng={lng:.6f}, heading={heading:.1f}°")

        bev_img = await render_bev_map_simple(
            lat=lat,
            lng=lng,
            heading_deg=heading,
        )

        output_path = output_dir / f"bev_{i:04d}.jpg"
        bev_img.save(output_path, quality=95)

    print(f"[DONE] 输出到: {output_dir}")


async def main():
    parser = argparse.ArgumentParser(
        description="BEV 渲染测试工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=35.7167,
        help="纬度",
    )
    parser.add_argument(
        "--lng",
        type=float,
        default=139.8543,
        help="经度",
    )
    parser.add_argument(
        "--heading",
        type=float,
        default=90.0,
        help="航向（度），0=北，90=东",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bev_output.jpg"),
        help="输出图像路径",
    )
    parser.add_argument(
        "--tile-url",
        type=str,
        default=None,
        help="瓦片 URL 模板",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="缩放级别",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="输出图像尺寸",
    )

    # GPS CSV 模式
    parser.add_argument(
        "--gps-csv",
        type=Path,
        default=None,
        help="GPS CSV 文件路径（批量模式）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bev_outputs"),
        help="批量模式输出目录",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=10,
        help="批量模式采样率",
    )

    args = parser.parse_args()

    if args.gps_csv:
        await test_from_gps_csv(
            args.gps_csv,
            args.output_dir,
            args.sample_rate,
        )
        return 0

    # 单张模式
    if args.tile_url:
        os.environ["BEV_MAP_TILE_URL_TEMPLATE"] = args.tile_url

    print(f"[INFO] 渲染 BEV: lat={args.lat:.6f}, lng={args.lng:.6f}, heading={args.heading:.1f}°")

    bev_img = await render_bev_map_simple(
        lat=args.lat,
        lng=args.lng,
        heading_deg=args.heading,
        output_size=args.size,
        zoom=args.zoom,
        tile_url_template=args.tile_url,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bev_img.save(args.output, quality=95)
    print(f"[DONE] 已保存到: {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
