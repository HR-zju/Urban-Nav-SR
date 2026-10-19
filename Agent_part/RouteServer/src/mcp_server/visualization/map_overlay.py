"""Map tile projection and ego-aligned BEV rendering, independent of prompt generation."""
from __future__ import annotations
import io
import math
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from PIL import Image, ImageDraw
from mcp_server.visualization.images import _jpeg_data_url_under_limit


def _lat_lng_from_gps(gps: Any) -> tuple[float, float] | None:
    if not isinstance(gps, dict):
        return None
    try:
        lat = float(gps.get("lat", gps.get("latitude")))
        lng = float(gps.get("lng", gps.get("lon", gps.get("longitude"))))
    except Exception:
        return None
    if not (-85.0 <= lat <= 85.0 and -180.0 <= lng <= 180.0):
        return None
    return lat, lng


def _osm_tile_xy_float(*, lat: float, lng: float, zoom: int) -> tuple[float, float]:
    """Project WGS84 into Web-Mercator slippy tile coordinates."""
    lat_rad = math.radians(max(-85.05112878, min(85.05112878, float(lat))))
    n = float(2**int(zoom))
    x = (float(lng) + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _fetch_osm_tile_image(*, x: int, y: int, zoom: int) -> Image.Image:
    """Fetch one real OSM raster tile with a small on-disk cache."""
    n = 2**int(zoom)
    x_wrapped = int(x) % n
    y_clamped = max(0, min(n - 1, int(y)))
    cache_dir = Path(os.getenv("BEV_MAP_TILE_CACHE_DIR", "/tmp/urban_nav_osm_tiles")).expanduser()
    tile_path = cache_dir / str(int(zoom)) / str(x_wrapped) / f"{y_clamped}.png"
    if tile_path.exists():
        return Image.open(tile_path).convert("RGB")
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    template = os.getenv("BEV_MAP_TILE_URL_TEMPLATE", "https://tile.openstreetmap.org/{z}/{x}/{y}.png").strip()
    url = template.format(z=int(zoom), x=x_wrapped, y=y_clamped)
    timeout_s = float(os.getenv("BEV_MAP_TILE_TIMEOUT_S", "8") or 8)
    request = Request(
        url,
        headers={
            "User-Agent": os.getenv(
                "BEV_MAP_TILE_USER_AGENT",
                "UrbanNavRouteServer/0.1 (+https://openstreetmap.org)",
            )
        },
    )
    with urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
    tile_path.write_bytes(raw)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _bev_map_heading_rotation_deg(raw_heading_deg: float) -> tuple[float, dict[str, Any]]:
    """Resolve map screenshot rotation from raw heading convention and offset.

    Supported conventions:
        compass: 0 deg = north, 90 deg = east, clockwise-positive.
        enu_yaw: 0 deg = east, 90 deg = north, counter-clockwise-positive.

    PIL rotates images counter-clockwise for positive angles. For a north-up
    slippy map, rotating by the compass heading puts the current forward
    direction at the top of the BEV screenshot.
    """
    convention = os.getenv("BEV_MAP_HEADING_CONVENTION", "compass").strip().lower() or "compass"
    offset_deg = float(os.getenv("BEV_MAP_HEADING_OFFSET_DEG", "0") or 0.0)
    raw = float(raw_heading_deg)
    if convention in {"enu", "enu_yaw", "ros_enu", "ros"}:
        compass_heading = (90.0 - raw) % 360.0
        resolved_convention = "enu_yaw"
    else:
        compass_heading = raw % 360.0
        resolved_convention = "compass"
    rotation_deg = (compass_heading + offset_deg) % 360.0
    return rotation_deg, {
        "raw_heading_deg": round(raw % 360.0, 3),
        "heading_convention": resolved_convention,
        "heading_offset_deg": round(float(offset_deg), 3),
        "compass_heading_deg": round(float(compass_heading), 3),
        "heading_rotation_deg": round(float(rotation_deg), 3),
    }


def _gps_path_from_context(local_bev_context: dict[str, Any], key: str) -> list[tuple[float, float]]:
    raw_points = local_bev_context.get(key)
    if not isinstance(raw_points, list):
        return []
    points: list[tuple[float, float]] = []
    for item in raw_points:
        gps = _lat_lng_from_gps(item)
        if gps is not None:
            points.append(gps)
    return points


def _rotate_point_about_center(
    *,
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    angle_deg: float,
) -> tuple[float, float]:
    """Map overlay points through the same forward transform as PIL.rotate.

    PIL rotates raster content counter-clockwise for positive angles in screen
    coordinates.  Vector overlays must use the matching image-coordinate
    transform; otherwise route/start/end markers drift relative to the OSM
    background after ego-alignment.
    """
    theta = math.radians(float(angle_deg))
    dx = float(x) - float(center_x)
    dy = float(y) - float(center_y)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return (
        float(center_x) + dx * cos_t + dy * sin_t,
        float(center_y) - dx * sin_t + dy * cos_t,
    )


def _draw_bev_navigation_path(
    *,
    image: Image.Image,
    path_gps: list[tuple[float, float]],
    path_source: str,
    current_gps: tuple[float, float],
    zoom: int,
    tile_size: int,
    mosaic_origin_tile_x: int,
    mosaic_origin_tile_y: int,
    rotation_center_px: tuple[float, float],
    heading_rotation_deg: float,
    crop_box: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Draw the route/reference path and ego pose into the BEV screenshot.

    Offline datasets can provide the full recorded GNSS trajectory, while
    online navigation can pass the active planner path through the same
    ``navigation_path_gps`` contract. The geometry pipeline is:

        WGS84 -> Web-Mercator mosaic pixels
              -> heading-aligned rotation around ego
              -> ego-centered crop
              -> resized BEV image coordinates
    """
    if not path_gps:
        return {"path_points_drawn": 0, "source": "none"}

    crop_w = max(float(crop_box[2] - crop_box[0]), 1.0)
    crop_h = max(float(crop_box[3] - crop_box[1]), 1.0)
    scale_x = float(image.size[0]) / crop_w
    scale_y = float(image.size[1]) / crop_h

    def _gps_to_image_xy(lat: float, lng: float) -> tuple[float, float]:
        tx, ty = _osm_tile_xy_float(lat=float(lat), lng=float(lng), zoom=int(zoom))
        mosaic_x = tx * float(tile_size) - float(mosaic_origin_tile_x) * float(tile_size)
        mosaic_y = ty * float(tile_size) - float(mosaic_origin_tile_y) * float(tile_size)
        rotated_x, rotated_y = _rotate_point_about_center(
            x=mosaic_x,
            y=mosaic_y,
            center_x=float(rotation_center_px[0]),
            center_y=float(rotation_center_px[1]),
            angle_deg=float(heading_rotation_deg),
        )
        return (
            (rotated_x - float(crop_box[0])) * scale_x,
            (rotated_y - float(crop_box[1])) * scale_y,
        )

    projected = [_gps_to_image_xy(lat, lng) for lat, lng in path_gps]
    current_xy = _gps_to_image_xy(current_gps[0], current_gps[1])
    draw = ImageDraw.Draw(image, "RGBA")
    if len(projected) >= 2:
        draw.line(projected, fill=(37, 99, 235, 235), width=max(int(round(image.size[0] * 0.012)), 4), joint="curve")
        draw.line(projected, fill=(219, 234, 254, 220), width=max(int(round(image.size[0] * 0.004)), 2), joint="curve")
    marker_r = max(int(round(image.size[0] * 0.018)), 7)
    start_r = max(int(round(image.size[0] * 0.012)), 5)
    if projected:
        sx, sy = projected[0]
        ex, ey = projected[-1]
        draw.ellipse([sx - start_r, sy - start_r, sx + start_r, sy + start_r], fill=(34, 197, 94, 245), outline=(255, 255, 255, 245), width=2)
        draw.rectangle([ex - start_r, ey - start_r, ex + start_r, ey + start_r], fill=(239, 68, 68, 245), outline=(255, 255, 255, 245), width=2)
    cx, cy = current_xy
    draw.ellipse([cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r], fill=(249, 115, 22, 250), outline=(255, 255, 255, 255), width=3)
    heading_len = marker_r * 2.2
    draw.line([(cx, cy), (cx, cy - heading_len)], fill=(255, 255, 255, 255), width=3)
    return {
        "path_points_drawn": int(len(projected)),
        "source": str(path_source or "navigation_path_gps"),
        "current_marker": "orange_circle",
        "start_marker": "green_circle",
        "end_marker": "red_square",
        "route_style": "blue_polyline",
    }


def _build_bev_map_overlay_payload(
    *,
    local_bev_context: dict[str, Any],
    llm_max_bytes: int,
    llm_jpeg_quality: int,
) -> dict[str, Any] | None:
    """Build a real OpenStreetMap screenshot as an ego-aligned BEV input.

    Unlike the earlier semantic sketch, this path uses actual OSM raster tiles:

        GNSS -> Web-Mercator tile mosaic -> heading rotation -> center crop

    The resulting image is a true map screenshot. It is intentionally not a
    structured semantic prompt; LVLM receives it as an image only.
    """
    if not isinstance(local_bev_context, dict):
        return None
    gps = _lat_lng_from_gps(
        local_bev_context.get("_origin_gps")
        or local_bev_context.get("origin_gps")
        or local_bev_context.get("gps")
    )
    if gps is None:
        return None
    lat, lng = gps
    output_size = int(os.getenv("BEV_MAP_TILE_OUTPUT_SIZE_PX", "640") or 640)
    tile_size = 256
    zoom = int(os.getenv("BEV_MAP_TILE_ZOOM", "19") or 19)
    mosaic_tiles = max(3, min(7, int(os.getenv("BEV_MAP_TILE_MOSAIC_TILES", "5") or 5)))
    if mosaic_tiles % 2 == 0:
        mosaic_tiles += 1
    raw_heading_deg = float(local_bev_context.get("bev_rotation_deg", 0.0) or 0.0)
    heading_rotation_deg, heading_debug = _bev_map_heading_rotation_deg(raw_heading_deg)
    center_x, center_y = _osm_tile_xy_float(lat=lat, lng=lng, zoom=zoom)
    center_px = center_x * tile_size
    center_py = center_y * tile_size
    center_tile_x = int(math.floor(center_x))
    center_tile_y = int(math.floor(center_y))
    half_tiles = mosaic_tiles // 2
    mosaic = Image.new("RGB", (mosaic_tiles * tile_size, mosaic_tiles * tile_size), (242, 239, 233))
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

    local_center_px = center_px - (center_tile_x - half_tiles) * tile_size
    local_center_py = center_py - (center_tile_y - half_tiles) * tile_size
    crop_side = min(mosaic.size) * 0.70
    crop_box = (
        int(round(local_center_px - crop_side / 2.0)),
        int(round(local_center_py - crop_side / 2.0)),
        int(round(local_center_px + crop_side / 2.0)),
        int(round(local_center_py + crop_side / 2.0)),
    )
    # Rotate the full tile mosaic around ego before cropping. Cropping first would
    # rotate a bounded square and expose irregular edge artifacts in the final BEV.
    rotated_mosaic = mosaic.rotate(
        float(heading_rotation_deg),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        center=(float(local_center_px), float(local_center_py)),
        fillcolor=(242, 239, 233),
    )
    ego_aligned = rotated_mosaic.crop(
        (
            crop_box[0],
            crop_box[1],
            crop_box[2],
            crop_box[3],
        )
    )
    image = ego_aligned.resize((output_size, output_size), Image.Resampling.LANCZOS).convert("RGB")
    navigation_path_gps = (
        _gps_path_from_context(local_bev_context, "planner_path_gps")
        or _gps_path_from_context(local_bev_context, "navigation_path_gps")
    )
    navigation_path_source = (
        "planner_path_gps"
        if _gps_path_from_context(local_bev_context, "planner_path_gps")
        else str(local_bev_context.get("navigation_path_source") or "navigation_path_gps")
    )
    navigation_path_debug = _draw_bev_navigation_path(
        image=image,
        path_gps=navigation_path_gps,
        path_source=navigation_path_source,
        current_gps=(float(lat), float(lng)),
        zoom=int(zoom),
        tile_size=int(tile_size),
        mosaic_origin_tile_x=int(center_tile_x - half_tiles),
        mosaic_origin_tile_y=int(center_tile_y - half_tiles),
        rotation_center_px=(float(local_center_px), float(local_center_py)),
        heading_rotation_deg=float(heading_rotation_deg),
        crop_box=crop_box,
    )

    data_url, byte_count, used_quality = _jpeg_data_url_under_limit(
        image,
        max_bytes=llm_max_bytes,
        jpeg_quality=llm_jpeg_quality,
    )
    return {
        "bev_map_overlay_image_data_url": data_url,
        "image_role": "bev_map_overlay",
        "source": "openstreetmap_raster_tiles",
        "llm_image_size": [output_size, output_size],
        "tile_provider": os.getenv("BEV_MAP_TILE_URL_TEMPLATE", "https://tile.openstreetmap.org/{z}/{x}/{y}.png"),
        "tile_zoom": int(zoom),
        "tile_mosaic": [int(mosaic_tiles), int(mosaic_tiles)],
        "tile_center_float": [round(float(center_x), 6), round(float(center_y), 6)],
        "alignment_order": "rotate_mosaic_then_center_crop",
        "rotation_center_px": [round(float(local_center_px), 3), round(float(local_center_py), 3)],
        "crop_side_px": round(float(crop_side), 3),
        "navigation_path_overlay": navigation_path_debug,
        "fetched_tiles": int(fetched_tiles),
        "tile_errors": tile_errors[:6],
        **heading_debug,
        "encoded_bytes": int(byte_count),
        "jpeg_quality": int(used_quality),
        "legend": {
            "image_role": "bev_map_overlay",
            "source": "real_openstreetmap_raster_screenshot",
            "coordinate_frame": "ego_aligned_bev",
            "forward_axis": "image_up_after_heading_rotation",
            "ego_marker": "orange circle; short white tick points forward/up",
            "route_overlay": "blue polyline; green circle=start; red square=end",
            "usage": "Real OSM screenshot for cross-view topology only; place final prompt on the FPV image, not in BEV coordinates.",
        },
    }
