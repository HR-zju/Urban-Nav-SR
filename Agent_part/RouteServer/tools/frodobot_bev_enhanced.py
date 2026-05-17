#!/usr/bin/env python3
"""FrodoBot BEV Dataset Generator - Enhanced Version.

Supports multiple map providers:
  - osm: OpenStreetMap (default)
  - carto: Carto Voyager
  - osm-toner: OSM Toner style
  - amap (needs API key): AMap
  - custom: Custom URL template

Features:
  - Batch processing with progress bar
  - Multiple map providers
  - Resume from interruption
  - Configurable output options
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add the src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from PIL import Image, ImageDraw, ImageFont

from mcp_server.services.visual_prompt_service import (
    _osm_tile_xy_float,
    _bev_map_heading_rotation_deg,
    _draw_bev_navigation_path,
    _lat_lng_from_gps,
)


# ==================== Map Provider Configurations ====================
MAP_PROVIDERS = {
    "osm": {
        "name": "OpenStreetMap",
        "url_template": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
    },
    "carto": {
        "name": "Carto Voyager",
        "url_template": "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "attribution": "© Carto, © OpenStreetMap",
    },
    "carto-dark": {
        "name": "Carto Dark Matter",
        "url_template": "https://a.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png",
        "attribution": "© Carto, © OpenStreetMap",
    },
    "osm-toner": {
        "name": "OSM Toner",
        "url_template": "https://a.tile.stamen.com/toner/{z}/{x}/{y}.png",
        "attribution": "© Stamen, © OpenStreetMap",
    },
    "osm-terrain": {
        "name": "OSM Terrain",
        "url_template": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap, © OpenTopoMap",
    },
}


def _fetch_tile_image(
    *,
    x: int,
    y: int,
    zoom: int,
    provider_config: dict[str, Any],
    cache_dir: Path,
) -> Image.Image:
    """Fetch tile with custom provider support."""
    n = 2**int(zoom)
    x_wrapped = int(x) % n
    y_clamped = max(0, min(n - 1, int(y)))

    template = provider_config["url_template"]
    cache_key = f"{provider_config.get('name', 'custom')}_{zoom}_{x_wrapped}_{y_clamped}"
    cache_path = cache_dir / f"{cache_key}.png"

    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = template.format(z=int(zoom), x=x_wrapped, y=y_clamped)

    try:
        from urllib.request import Request, urlopen

        request = Request(
            url,
            headers={
                "User-Agent": os.getenv(
                    "BEV_MAP_TILE_USER_AGENT",
                    "FrodoBot-BEV-Generator/1.0 (+https://github.com/your-org/urban-nav)",
                )
            },
        )
        timeout_s = float(os.getenv("BEV_MAP_TILE_TIMEOUT_S", "15") or 15)

        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read()

        cache_path.write_bytes(raw)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        print(f"  [WARN] Tile fetch failed {zoom}/{x}/{y}: {exc}")
        return Image.new("RGB", (256, 256), (242, 239, 233))


# ==================== Data Structures ====================
@dataclass
class GPSPoint:
    lat: float
    lng: float
    timestamp_ms: float


@dataclass
class RideInfo:
    ride_id: str
    gps_csv: Path
    video_timestamps_csv: Path | None = None
    recordings_dir: Path | None = None


@dataclass
class ProcessOptions:
    map_provider: str = "osm"
    output_size: int = 640
    zoom: int = 19
    sample_rate: int = 10
    max_frames_per_ride: int | None = None
    cache_dir: Path = Path("/tmp/frodobot_bev_cache")
    draw_trajectory: bool = True
    add_attribution: bool = True


# ==================== GPS Processing ====================
def load_gps_from_csv(csv_path: Path) -> list[GPSPoint]:
    """Load GPS data from FrodoBot CSV."""
    points: list[GPSPoint] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append(GPSPoint(
                    lat=float(row["latitude"]),
                    lng=float(row["longitude"]),
                    timestamp_ms=float(row["timestamp"]),
                ))
            except Exception as e:
                continue
    return points


def estimate_heading(points: list[GPSPoint], index: int, lookahead: int = 10) -> float:
    """Estimate heading from GPS points."""
    if not points or index < 0 or index >= len(points):
        return 0.0
    start = points[index]
    for i in range(index + 1, min(index + lookahead + 1, len(points))):
        end = points[i]
        dist_m = haversine_m(start.lat, start.lng, end.lat, end.lng)
        if dist_m > 2.0:
            return bearing_deg(start.lat, start.lng, end.lat, end.lng)
    # If no forward points, try backward
    for i in range(max(0, index - lookahead), index):
        end = points[i]
        dist_m = haversine_m(start.lat, start.lng, end.lat, end.lng)
        if dist_m > 2.0:
            return bearing_deg(end.lat, end.lng, start.lat, start.lng)
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


def interpolate_gps_for_timestamp(
    timestamp_ms: float,
    gps_points: list[GPSPoint],
) -> GPSPoint | None:
    """Interpolate GPS for a given timestamp."""
    if not gps_points:
        return None
    if timestamp_ms <= gps_points[0].timestamp_ms:
        return gps_points[0]
    if timestamp_ms >= gps_points[-1].timestamp_ms:
        return gps_points[-1]

    # Binary search
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

    alpha = (timestamp_ms - p1.timestamp_ms) / (p2.timestamp_ms - p1.timestamp_ms)
    return GPSPoint(
        lat=p1.lat + alpha * (p2.lat - p1.lat),
        lng=p1.lng + alpha * (p2.lng - p1.lng),
        timestamp_ms=timestamp_ms,
    )


# ==================== BEV Rendering ====================
def render_bev_map(
    lat: float,
    lng: float,
    heading_deg: float = 0.0,
    navigation_path_gps: list[tuple[float, float]] | None = None,
    options: ProcessOptions | None = None,
    provider_config: dict[str, Any] | None = None,
) -> Image.Image:
    """Render BEV map with custom provider support."""
    if options is None:
        options = ProcessOptions()
    if provider_config is None:
        provider_config = MAP_PROVIDERS["osm"]

    tile_size = 256
    mosaic_tiles = 5
    output_size = options.output_size
    zoom = options.zoom

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
    for dx in range(-half_tiles, half_tiles + 1):
        for dy in range(-half_tiles, half_tiles + 1):
            tx = center_tile_x + dx
            ty = center_tile_y + dy
            tile = _fetch_tile_image(
                x=tx, y=ty, zoom=zoom,
                provider_config=provider_config,
                cache_dir=options.cache_dir,
            )
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

    # Draw navigation path
    if options.draw_trajectory and navigation_path_gps:
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

    # Add attribution
    if options.add_attribution and provider_config.get("attribution"):
        draw = ImageDraw.Draw(image, "RGBA")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except Exception:
            font = ImageFont.load_default()

        attribution = provider_config["attribution"]
        bbox = draw.textbbox((0, 0), attribution, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Draw semi-transparent background
        padding = 4
        bg_rect = [
            output_size - text_w - padding * 2 - 5,
            output_size - text_h - padding * 2 - 5,
            output_size - 5,
            output_size - 5,
        ]
        draw.rectangle(bg_rect, fill=(0, 0, 0, 128))

        # Draw text
        draw.text(
            (output_size - text_w - padding - 5, output_size - text_h - padding - 5),
            attribution,
            font=font,
            fill=(255, 255, 255, 200),
        )

    return image


# ==================== Ride Discovery ====================
def discover_rides(dataset_dir: Path) -> list[RideInfo]:
    """Discover all ride directories in dataset."""
    rides: list[RideInfo] = []

    for entry in dataset_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("ride_"):
            continue

        ride_id = entry.name
        gps_csv = entry / "gps_data.csv"
        if not gps_csv.exists():
            # Try gps_data_*.csv format
            for f in entry.glob("gps_data_*.csv"):
                gps_csv = f
                break
        if not gps_csv.exists():
            continue

        ride_info = RideInfo(ride_id=entry.name, gps_csv=gps_csv)

        recordings_dir = entry / "recordings"
        if recordings_dir.exists():
            ride_info.recordings_dir = recordings_dir

        rides.append(ride_info)

    return sorted(rides, key=lambda r: r.ride_id)


def load_video_timestamps(csv_path: Path) -> list[tuple[int, float]] | None:
    """Load video timestamps CSV."""
    timestamps = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame_id = int(row.get("frame_id", 0))
                    ts = float(row.get("timestamp", 0))
                    timestamps.append((frame_id, ts))
                except Exception:
                    continue
    except Exception:
        pass

    if not timestamps:
        return None

    return sorted(timestamps, key=lambda t: t[1])


# ==================== Progress Bar ====================
class ProgressBar:
    def __init__(self, total: int, prefix: str = "Processing", width: int = 40):
        self.total = total
        self.current = 0
        self.prefix = prefix
        self.width = width
        self.start_time = time.time()

    def update(self, current: int | None = None):
        if current is not None:
            self.current = current
        else:
            self.current += 1

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
        else:
            eta = 0

        filled = int(self.width * self.current // self.total)
        bar = "█" * filled + "░" * (self.width - filled)

        percent = self.current / self.total * 100
        sys.stdout.write(
            f"\r{self.prefix} [{bar}] {percent:.1f}% {self.current}/{self.total} "
            f"[{elapsed:.1f}s < {eta:.1f}s]"
        )
        sys.stdout.flush()

    def finish(self):
        print()


# ==================== Main Processing ====================
def process_single_gps(
    ride_info: RideInfo,
    output_dir: Path,
    options: ProcessOptions,
    provider_config: dict[str, Any],
    progress: ProgressBar | None = None,
) -> list[dict[str, Any]]:
    """Process a single ride using GPS only (no video)."""
    samples: list[dict[str, Any]] = []

    gps_points = load_gps_from_csv(ride_info.gps_csv)
    if not gps_points:
        print(f"[WARN] No GPS data for {ride_info.ride_id}")
        return samples

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Sample points
    sampled_indices = list(range(0, len(gps_points), options.sample_rate))
    if options.max_frames_per_ride:
        sampled_indices = sampled_indices[:options.max_frames_per_ride]

    for i, idx in enumerate(sampled_indices):
        point = gps_points[idx]
        heading = estimate_heading(gps_points, idx)

        # Build navigation path
        nav_path = []
        for j in range(idx, min(idx + 30, len(gps_points))):
            nav_path.append((gps_points[j].lat, gps_points[j].lng))

        try:
            bev_img = render_bev_map(
                lat=point.lat,
                lng=point.lng,
                heading_deg=heading,
                navigation_path_gps=nav_path,
                options=options,
                provider_config=provider_config,
            )

            base_name = f"{ride_info.ride_id}_gps_{i:06d}"
            bev_path = images_dir / f"{base_name}_bev.jpg"
            bev_img.save(bev_path, quality=95)

            samples.append({
                "ride_id": ride_info.ride_id,
                "source": "gps_only",
                "bev_image": str(bev_path.relative_to(output_dir)),
                "metadata": {
                    "lat": point.lat,
                    "lng": point.lng,
                    "timestamp_ms": point.timestamp_ms,
                    "heading_deg": heading,
                },
            })
        except Exception as e:
            print(f"\n[WARN] Error processing frame {i} in {ride_info.ride_id}: {e}")

        if progress:
            progress.update(progress.current + 1)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="FrodoBot BEV Dataset Generator - Enhanced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Available map providers:
  {providers}

Examples:
  # Use OSM (default)
  python %(prog)s --dataset-dir /path/to/dataset --output-dir ./output

  # Use Carto (prettier style)
  python %(prog)s --dataset-dir /path/to/dataset --map-provider carto

  # Custom URL template
  python %(prog)s --dataset-dir /path/to/dataset --custom-url "https://your-server/{{z}}/{{x}}/{{y}}.png"
""".format(providers=', '.join(MAP_PROVIDERS.keys())),
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/ubuntu/WorkSpace/dataset/output_rides_0"),
        help="FrodoBot dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/frodobot_system2"),
        help="Output directory",
    )
    parser.add_argument(
        "--ride-ids",
        type=str,
        nargs="+",
        default=None,
        help="Process only specific ride IDs",
    )
    parser.add_argument(
        "--map-provider",
        type=str,
        default="osm",
        choices=list(MAP_PROVIDERS.keys()) + ["custom"],
        help="Map provider to use (default: osm)",
    )
    parser.add_argument(
        "--custom-url",
        type=str,
        default=None,
        help="Custom tile URL template (use with --map-provider custom)",
    )
    parser.add_argument(
        "--amap-key",
        type=str,
        default=None,
        help="AMap API key (if using AMap)",
    )

    # Processing options
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=10,
        help="Sample every N-th frame (default: 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames per ride (default: unlimited)",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=640,
        help="BEV image size (default: 640)",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="OSM zoom level (default: 19)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/frodobot_bev_cache"),
        help="Tile cache directory",
    )
    parser.add_argument(
        "--no-attribution",
        action="store_true",
        help="Hide map attribution",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Don't draw trajectory overlay",
    )

    # Resume options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output",
    )

    args = parser.parse_args()

    # Get provider config
    if args.map_provider == "custom":
        if not args.custom_url:
            print("[ERROR] --custom-url is required when using --map-provider custom")
            return 1
        provider_config = {
            "name": "Custom",
            "url_template": args.custom_url,
            "attribution": "© Map data provider",
        }
    else:
        provider_config = MAP_PROVIDERS.get(args.map_provider, MAP_PROVIDERS["osm"])

    options = ProcessOptions(
        map_provider=args.map_provider,
        output_size=args.output_size,
        zoom=args.zoom,
        sample_rate=args.sample_rate,
        max_frames_per_ride=args.max_frames,
        cache_dir=args.cache_dir,
        draw_trajectory=not args.no_trajectory,
        add_attribution=not args.no_attribution,
    )

    # Discover rides
    rides = discover_rides(args.dataset_dir)
    if not rides:
        print(f"[ERROR] No rides found in {args.dataset_dir}")
        return 1

    if args.ride_ids:
        ride_id_set = set(args.ride_ids)
        rides = [r for r in rides if r.ride_id in ride_id_set]
        if not rides:
            print(f"[ERROR] None of the specified rides found: {args.ride_ids}")
            return 1

    print(f"[INFO] Found {len(rides)} rides")
    print(f"[INFO] Map provider: {provider_config['name']}")

    # Check for existing output
    manifest_path = args.output_dir / "manifest.json"
    completed_rides = set()
    if args.resume and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                completed_rides = set(manifest.get("completed_rides", []))
            print(f"[INFO] Resuming, already completed: {len(completed_rides)} rides")
        except Exception:
            pass

    rides_to_process = [r for r in rides if r.ride_id not in completed_rides]
    if not rides_to_process:
        print("[INFO] All rides already processed!")
        return 0

    # Estimate total frames
    total_frames = 0
    for ride in rides_to_process:
        gps_points = load_gps_from_csv(ride.gps_csv)
        num_samples = (len(gps_points) + args.sample_rate - 1) // args.sample_rate
        if args.max_frames:
            num_samples = min(num_samples, args.max_frames)
        total_frames += num_samples

    print(f"[INFO] Estimated frames to process: {total_frames}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    progress = ProgressBar(total_frames, prefix="Processing")

    for i, ride in enumerate(rides_to_process):
        print(f"\n[{i+1}/{len(rides_to_process)}] Processing {ride.ride_id}")
        samples = process_single_gps(ride, args.output_dir, options, provider_config, progress)
        all_samples.extend(samples)
        completed_rides.add(ride.ride_id)

        # Save intermediate manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            options_dict = vars(options)
            # Convert Path objects to strings
            for k, v in options_dict.items():
                if isinstance(v, Path):
                    options_dict[k] = str(v)
            json.dump({
                "completed_rides": list(completed_rides),
                "options": options_dict,
                "provider": provider_config,
            }, f, indent=2, ensure_ascii=False)

    progress.finish()

    # Save final samples
    samples_json = args.output_dir / "samples.json"
    with open(samples_json, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE]")
    print(f"Processed {len(all_samples)} frames from {len(rides_to_process)} rides")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Samples: {samples_json.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
