#!/usr/bin/env python3
"""Test FrodoBot BEV generation using existing RouteServer code."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Add the src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Now import from the project
from PIL import Image, ImageDraw

from mcp_server.services.visual_prompt_service import (
    _osm_tile_xy_float,
    _fetch_osm_tile_image,
    _bev_map_heading_rotation_deg,
    _draw_bev_navigation_path,
    _lat_lng_from_gps,
)


def load_gps_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load GPS data from FrodoBot CSV."""
    points: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append({
                    "lat": float(row["latitude"]),
                    "lng": float(row["longitude"]),
                    "timestamp_ms": float(row["timestamp"]),
                })
            except Exception as e:
                pass
    return points


def estimate_heading(points: list[dict[str, Any]], index: int, lookahead: int = 5) -> float:
    """Estimate heading from GPS points."""
    if not points or index < 0:
        return 0.0
    start = points[index]
    for i in range(index + 1, min(index + lookahead + 1, len(points))):
        end = points[i]
        # Check distance
        lat1, lon1 = start["lat"], start["lng"]
        lat2, lon2 = end["lat"], end["lng"]
        dist_m = haversine_m(lat1, lon1, lat2, lon2)
        if dist_m > 2.0:
            return bearing_deg(lat1, lon1, lat2, lon2)
    return 0.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters."""
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


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing in degrees (0 = north, 90 = east)."""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
    )
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def render_bev_map(
    lat: float,
    lng: float,
    heading_deg: float = 0.0,
    navigation_path_gps: list[tuple[float, float]] | None = None,
    output_size: int = 640,
    zoom: int = 19,
) -> Image.Image:
    """Render BEV map using existing project code."""
    tile_size = 256
    mosaic_tiles = 5

    if mosaic_tiles % 2 == 0:
        mosaic_tiles += 1

    # Heading rotation
    heading_rotation_deg, _ = _bev_map_heading_rotation_deg(heading_deg)

    # Tile coordinates
    center_x, center_y = _osm_tile_xy_float(lat=lat, lng=lng, zoom=zoom)
    center_px = center_x * tile_size
    center_py = center_y * tile_size
    center_tile_x = int(math.floor(center_x))
    center_tile_y = int(math.floor(center_y))
    half_tiles = mosaic_tiles // 2

    # Create mosaic
    mosaic = Image.new("RGB", (mosaic_tiles * tile_size, mosaic_tiles * tile_size), (242, 239, 233))

    # Fetch tiles
    tile_errors: list[str] = []
    fetched_tiles = 0
    for dx in range(-half_tiles, half_tiles + 1):
        for dy in range(-half_tiles, half_tiles + 1):
            tx = center_tile_x + dx
            ty = center_tile_y + dy
            try:
                tile = _fetch_osm_tile_image(x=tx, y=ty, zoom=zoom)
                fetched_tiles += 1
            except Exception as exc:
                tile_errors.append(f"{zoom}/{tx}/{ty}:{exc}")
                tile = Image.new("RGB", (tile_size, tile_size), (242, 239, 233))
            paste_x = (dx + half_tiles) * tile_size
            paste_y = (dy + half_tiles) * tile_size
            mosaic.paste(tile.resize((tile_size, tile_size)), (paste_x, paste_y))

    # Calculate crop
    local_center_px = center_px - (center_tile_x - half_tiles) * tile_size
    local_center_py = center_py - (center_tile_y - half_tiles) * tile_size
    crop_side = min(mosaic.size) * 0.70

    crop_box = (
        int(round(local_center_px - crop_side / 2.0)),
        int(round(local_center_py - crop_side / 2.0)),
        int(round(local_center_px + crop_side / 2.0)),
        int(round(local_center_py + crop_side / 2.0)),
    )

    # Rotate and crop
    rotated_mosaic = mosaic.rotate(
        float(heading_rotation_deg),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        center=(float(local_center_px), float(local_center_py)),
        fillcolor=(242, 239, 233),
    )
    ego_aligned = rotated_mosaic.crop(crop_box)
    image = ego_aligned.resize((output_size, output_size), Image.Resampling.LANCZOS).convert("RGB")

    # Draw navigation path if available
    if navigation_path_gps:
        _draw_bev_navigation_path(
            image=image,
            path_gps=navigation_path_gps,
            path_source="frodobot_trajectory",
            current_gps=(float(lat), float(lng)),
            zoom=int(zoom),
            tile_size=int(tile_size),
            mosaic_origin_tile_x=int(center_tile_x - half_tiles),
            mosaic_origin_tile_y=int(center_tile_y - half_tiles),
            rotation_center_px=(float(local_center_px), float(local_center_py)),
            heading_rotation_deg=float(heading_rotation_deg),
            crop_box=crop_box,
        )
    else:
        # Just draw ego marker
        draw = ImageDraw.Draw(image, "RGBA")
        center = output_size // 2
        marker_r = max(int(round(output_size * 0.018)), 7)
        draw.ellipse(
            [center - marker_r, center - marker_r, center + marker_r, center + marker_r],
            fill=(249, 115, 22, 250),
            outline=(255, 255, 255, 255),
            width=3,
        )
        heading_len = marker_r * 2.2
        draw.line([(center, center), (center, center - heading_len)], fill=(255, 255, 255, 255), width=3)

    return image


def main():
    parser = argparse.ArgumentParser(description="Test FrodoBot BEV rendering")
    parser.add_argument("--lat", type=float, default=35.716705, help="Latitude")
    parser.add_argument("--lng", type=float, default=139.854324, help="Longitude")
    parser.add_argument("--heading", type=float, default=90.0, help="Heading in degrees (0 = north, 90 = east)")
    parser.add_argument("--output", type=Path, default=Path("/tmp/bev_test_output.jpg"), help="Output image path")
    parser.add_argument("--size", type=int, default=640, help="Output image size")
    parser.add_argument("--zoom", type=int, default=19, help="OSM zoom level")

    # CSV batch mode
    parser.add_argument("--gps-csv", type=Path, default=None, help="Path to GPS CSV (batch mode)")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/bev_outputs"), help="Output directory for batch mode")
    parser.add_argument("--sample-rate", type=int, default=10, help="Sample rate for batch mode")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames for batch mode")

    args = parser.parse_args()

    if args.gps_csv and args.gps_csv.exists():
        print(f"Loading GPS data from {args.gps_csv}")
        points = load_gps_from_csv(args.gps_csv)
        if not points:
            print("No GPS points found")
            return 1

        print(f"Loaded {len(points)} GPS points")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        sampled = points[::args.sample_rate][:args.max_frames]
        print(f"Sampled {len(sampled)} frames")

        for i, point in enumerate(sampled):
            heading = estimate_heading(points, points.index(point) if point in points else i)
            print(f"Rendering {i+1}/{len(sampled)}: lat={point['lat']:.6f}, lng={point['lng']:.6f}, heading={heading:.1f}")

            # Build navigation path
            nav_path = []
            path_start = points.index(point) if point in points else i
            for j in range(path_start, min(path_start + 20, len(points))):
                nav_path.append((points[j]["lat"], points[j]["lng"]))

            try:
                img = render_bev_map(
                    lat=point["lat"],
                    lng=point["lng"],
                    heading_deg=heading,
                    navigation_path_gps=nav_path if nav_path else None,
                    output_size=args.size,
                    zoom=args.zoom,
                )
                out_path = args.output_dir / f"bev_frame_{i:04d}.jpg"
                img.save(out_path, quality=95)
            except Exception as e:
                print(f"Error rendering frame {i}: {e}")

        print(f"\nOutputs saved to {args.output_dir}")
        return 0

    # Single frame mode
    print(f"Rendering BEV: lat={args.lat:.6f}, lng={args.lng:.6f}, heading={args.heading:.1f}")
    try:
        img = render_bev_map(
            lat=args.lat,
            lng=args.lng,
            heading_deg=args.heading,
            output_size=args.size,
            zoom=args.zoom,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        img.save(args.output, quality=95)
        print(f"Saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
