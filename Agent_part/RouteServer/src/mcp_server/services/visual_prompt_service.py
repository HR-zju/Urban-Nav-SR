"""End-to-end orchestration for first-person visual navigation prompts.

The service combines route intent, FPV imagery, semantic segmentation, SoM
regions, LVLM generation, deterministic mask constraints, optional LVLM
verification, and debug/QA artifact export.

Pipeline:

    route inputs -> RouteContext/LocalBEV
    FPV image    -> LVLM-safe resized image
    segmentation -> semantic summary + SoM + safe_corridor_prior
    A(generator) -> rendered prompt in LVLM image coordinates
    scale back   -> rendered prompt in source image coordinates
    V(mask)      -> deterministic validate/repair/reject
    V(verifier)  -> optional LVLM repair, followed by V(mask) again

Coordinate transform:

    u_llm = u_source * (W_llm / W_source)
    v_llm = v_source * (H_llm / H_source)

    u_source = u_llm * (W_source / W_llm)
    v_source = v_llm * (H_source / H_llm)

The LVLM sees only LVLM-image coordinates. All persisted overlays and hard
segmentation checks run in source-image coordinates.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import io
import json
import math
import os
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from PIL import Image, ImageOps

from mcp_server.brain import (
    build_direct_prompt_question,
    build_direct_prompt_system_prompt,
    build_direct_prompt_verifier_question,
    build_direct_prompt_verifier_system_prompt,
    coerce_direct_prompt_decision,
)
from mcp_server.brain.fpv_segmentation import (
    build_safe_corridor_prior,
    build_segmentation_teacher_bundle,
    CORRIDOR_WIDTH_RATIO_ENV,
    CENTERLINE_OFFSET_RATIO_ENV,
    EDGE_MARGIN_RATIO_ENV,
    centerline_offset_ratio_limit,
    corridor_width_ratio_limits,
    edge_margin_ratio_limit,
    load_fpv_segmentation_context,
    repair_rendered_prompt_with_safe_corridor,
    segmentation_summary_for_llm,
    validate_rendered_prompt_against_segmentation,
)
from mcp_server.services.fpv_segmentation_service import segment_fpv_frame
from mcp_server.llm import (
    arbitrate_local_pov_candidates_with_llm,
    arbitrate_pov_candidates_with_llm,
    local_image_path_to_data_url,
)
from mcp_server.qa import (
    build_pov_prompt_candidate_bundle,
    build_visual_prompt_qa_bundle,
    pov_candidate_bundle_to_dict,
)
from mcp_server.route_context import (
    build_route_state,
    local_bev_context_to_dict,
    route_context_to_dict,
    visual_prompt_decision_to_dict,
)
from mcp_server.route_context.bev_map_provider import BEVMapProviderSession
from mcp_server.route_context.schema import VisualPromptDecision
from mcp_server.som import (
    build_numbered_som_overlay_payload,
    build_som_context,
    rank_som_regions_for_route,
    som_summary_for_llm,
)


def _image_href_for_overlay(*, image_path: Optional[str], image_url: Optional[str]) -> str:
    if str(image_path or "").strip():
        return local_image_path_to_data_url(str(image_path))
    return str(image_url or "").strip()


def _load_prompt_image(*, image_path: Optional[str], image_url: Optional[str]) -> Image.Image:
    if str(image_path or "").strip():
        return ImageOps.exif_transpose(Image.open(Path(str(image_path)).expanduser())).convert("RGB")
    raw = str(image_url or "").strip()
    if raw.startswith("data:image/") and "," in raw:
        _, encoded = raw.split(",", 1)
        return ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(encoded)))).convert("RGB")
    raise RuntimeError("Visual prompt rendering currently supports image_path or data:image URL inputs")


def _prompt_image_size(*, image_path: Optional[str], image_url: Optional[str]) -> tuple[int, int]:
    image = _load_prompt_image(image_path=image_path, image_url=image_url)
    return int(image.size[0]), int(image.size[1])


def _llm_image_limits() -> tuple[int, int, int]:
    """Resolve image compression limits for vision-model input.

    The LVLM input is a resized/compressed view of the source image. All
    geometry-bearing structured context must be scaled into that image plane
    before it is sent to the model.
    """
    max_side_raw = (
        os.getenv("VISUAL_PROMPT_LLM_IMAGE_MAX_SIDE", "").strip()
        or os.getenv("LLM_VISION_IMAGE_MAX_SIDE", "").strip()
    )
    max_bytes_raw = (
        os.getenv("VISUAL_PROMPT_LLM_IMAGE_MAX_BYTES", "").strip()
        or os.getenv("LLM_VISION_IMAGE_MAX_BYTES", "").strip()
    )
    quality_raw = (
        os.getenv("VISUAL_PROMPT_LLM_IMAGE_JPEG_QUALITY", "").strip()
        or os.getenv("LLM_VISION_IMAGE_JPEG_QUALITY", "").strip()
    )
    try:
        max_side = max(512, int(max_side_raw)) if max_side_raw else 1600
    except ValueError:
        max_side = 1600
    try:
        max_bytes = max(256_000, int(max_bytes_raw)) if max_bytes_raw else 7_500_000
    except ValueError:
        max_bytes = 7_500_000
    try:
        jpeg_quality = min(max(int(quality_raw), 40), 95) if quality_raw else 85
    except ValueError:
        jpeg_quality = 85
    return int(max_side), int(max_bytes), int(jpeg_quality)


def _resize_image_max_side(image: Image.Image, *, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height, 1)
    scale = min(1.0, float(max(max_side, 1)) / float(longest))
    if scale >= 0.999:
        return image.copy()
    return image.resize(
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )


def _jpeg_data_url_under_limit(
    image: Image.Image,
    *,
    max_bytes: int,
    jpeg_quality: int,
) -> tuple[str, int, int]:
    candidate = image.convert("RGB")
    quality_schedule = [int(jpeg_quality), 78, 70, 62, 54, 46]
    last_raw = b""
    for _ in range(4):
        for quality in quality_schedule:
            buffer = io.BytesIO()
            candidate.save(buffer, format="JPEG", quality=max(40, min(quality, 95)), optimize=True)
            raw = buffer.getvalue()
            last_raw = raw
            if len(raw) <= int(max_bytes):
                encoded = base64.b64encode(raw).decode("ascii")
                return f"data:image/jpeg;base64,{encoded}", len(raw), int(quality)
        width, height = candidate.size
        if max(width, height) <= 640:
            break
        candidate = candidate.resize(
            (max(1, int(round(width * 0.82))), max(1, int(round(height * 0.82)))),
            Image.Resampling.LANCZOS,
        )
    encoded = base64.b64encode(last_raw).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", len(last_raw), int(quality_schedule[-1])


def _build_llm_image_payload(
    image: Image.Image,
    *,
    max_side: int,
    max_bytes: int,
    jpeg_quality: int,
) -> Dict[str, Any]:
    """Encode an image for the LVLM and return bidirectional scale factors."""
    source_width, source_height = image.size
    llm_image = _resize_image_max_side(image.convert("RGB"), max_side=max_side)
    data_url, byte_count, used_quality = _jpeg_data_url_under_limit(
        llm_image,
        max_bytes=max_bytes,
        jpeg_quality=jpeg_quality,
    )
    llm_width, llm_height = llm_image.size
    return {
        "image_data_url": data_url,
        "source_image_size": [int(source_width), int(source_height)],
        "llm_image_size": [int(llm_width), int(llm_height)],
        "source_to_llm_scale": {
            "x": round(float(llm_width) / float(max(source_width, 1)), 8),
            "y": round(float(llm_height) / float(max(source_height, 1)), 8),
        },
        "llm_to_source_scale": {
            "x": round(float(source_width) / float(max(llm_width, 1)), 8),
            "y": round(float(source_height) / float(max(llm_height, 1)), 8),
        },
        "encoded_bytes": int(byte_count),
        "jpeg_quality": int(used_quality),
        "max_side": int(max_side),
        "max_bytes": int(max_bytes),
    }


def _local_image_path_to_llm_data_url(
    image_path: str,
    *,
    max_side: int,
    max_bytes: int,
    jpeg_quality: int,
) -> str:
    image = ImageOps.exif_transpose(Image.open(Path(str(image_path)).expanduser())).convert("RGB")
    return str(
        _build_llm_image_payload(
            image,
            max_side=max_side,
            max_bytes=max_bytes,
            jpeg_quality=jpeg_quality,
        )["image_data_url"]
    )


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
            "usage": "Real OSM screenshot for cross-view topology only; place final prompt on the FPV image, not in BEV coordinates.",
        },
    }


def _scale_point_uv(value: Any, *, scale_x: float, scale_y: float) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [round(float(value[0]) * scale_x, 2), round(float(value[1]) * scale_y, 2)]
    except Exception:
        return None


def _scale_bbox_uv(value: Any, *, scale_x: float, scale_y: float) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [
            int(round(float(value[0]) * scale_x)),
            int(round(float(value[1]) * scale_y)),
            int(round(float(value[2]) * scale_x)),
            int(round(float(value[3]) * scale_y)),
        ]
    except Exception:
        return None


_POINT_LIST_KEYS = {"path_uv", "coverage_polygon_uv", "mask", "mask_polygon_uv"}
_ID_LIST_KEYS = {
    "class_ids",
    "forbidden_class_ids",
    "traversable_class_ids",
    "boundary_class_ids",
}
_U_COORD_KEYS = {
    "run_mid_u",
    "preferred_run_mid_u",
    "left_inset_u",
    "right_inset_u",
}
_PX_WIDTH_KEYS = {
    "margin_px",
    "edge_margin_px",
    "corridor_width_px",
    "max_width_px",
    "preferred_width_px",
    "span_px",
}


def _scale_context_to_llm(value: Any, *, scale_x: float, scale_y: float, key_hint: str | None = None) -> Any:
    """Project source-image geometry into the resized image actually sent to the LVLM.

    Segmentation, SoM, and route-derived constraints are computed in source-image
    pixels. The generator/verifier only sees the compressed LVLM image, so every
    pixel-space field in those structured inputs must be mapped consistently.
    """
    if isinstance(value, dict):
        scaled: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"center_uv", "point_uv", "target_point_uv"}:
                scaled_point = _scale_point_uv(item, scale_x=scale_x, scale_y=scale_y)
                scaled[key] = scaled_point if scaled_point is not None else item
            elif key in {"bbox_uv"}:
                scaled_bbox = _scale_bbox_uv(item, scale_x=scale_x, scale_y=scale_y)
                scaled[key] = scaled_bbox if scaled_bbox is not None else item
            elif key in _POINT_LIST_KEYS and isinstance(item, list):
                scaled[key] = [
                    point
                    for point in (
                        _scale_point_uv(point, scale_x=scale_x, scale_y=scale_y)
                        for point in item
                    )
                    if point is not None
                ]
            elif key in _ID_LIST_KEYS:
                scaled[key] = item
            elif key in {"shape_hw", "image_shape_hw", "mask_shape_hw"} and isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    scaled[key] = [int(round(float(item[0]) * scale_y)), int(round(float(item[1]) * scale_x))]
                except Exception:
                    scaled[key] = item
            elif key in {"u_min", "u_max"}:
                try:
                    scaled[key] = int(round(float(item) * scale_x))
                except Exception:
                    scaled[key] = item
            elif key in _U_COORD_KEYS:
                try:
                    scaled[key] = round(float(item) * scale_x, 2)
                except Exception:
                    scaled[key] = item
            elif key in {"v_min", "v_max", "row_v"}:
                try:
                    scaled[key] = int(round(float(item) * scale_y))
                except Exception:
                    scaled[key] = item
            elif key == "width_px":
                try:
                    scaled[key] = int(round(float(item) * scale_x))
                except Exception:
                    scaled[key] = item
            elif key in _PX_WIDTH_KEYS:
                try:
                    scaled[key] = round(float(item) * scale_x, 2)
                except Exception:
                    scaled[key] = item
            else:
                scaled[key] = _scale_context_to_llm(item, scale_x=scale_x, scale_y=scale_y, key_hint=key)
        if "coordinate_space" not in scaled:
            scaled["coordinate_space"] = "llm_image_uv"
        return scaled
    if isinstance(value, list):
        if key_hint in _ID_LIST_KEYS:
            return value
        if key_hint in _POINT_LIST_KEYS:
            return [
                point
                for point in (
                    _scale_point_uv(point, scale_x=scale_x, scale_y=scale_y)
                    for point in value
                )
                if point is not None
            ]
        return [_scale_context_to_llm(item, scale_x=scale_x, scale_y=scale_y) for item in value]
    return value


def _scale_rendered_prompt_geometry(
    rendered_prompt: Dict[str, Any] | None,
    *,
    scale_x: float,
    scale_y: float,
) -> Dict[str, Any] | None:
    """Scale all rendered-prompt geometry fields between image coordinate spaces."""
    if not isinstance(rendered_prompt, dict):
        return None
    scaled = deepcopy(rendered_prompt)
    avg_scale = (float(scale_x) + float(scale_y)) / 2.0
    for key in ("center_uv",):
        point = _scale_point_uv(scaled.get(key), scale_x=scale_x, scale_y=scale_y)
        if point is not None:
            scaled[key] = point
    for key in ("path_uv", "coverage_polygon_uv", "mask"):
        raw_points = scaled.get(key)
        if isinstance(raw_points, list):
            scaled_points = [
                _scale_point_uv(point, scale_x=scale_x, scale_y=scale_y)
                for point in raw_points
            ]
            scaled[key] = [point for point in scaled_points if point is not None]
    path_samples = scaled.get("path_samples")
    if isinstance(path_samples, list):
        scaled_samples: list[dict[str, Any]] = []
        for sample in path_samples:
            if not isinstance(sample, dict):
                continue
            scaled_sample = dict(sample)
            for point_key in ("center_uv", "left_edge_uv", "right_edge_uv"):
                point = _scale_point_uv(sample.get(point_key), scale_x=scale_x, scale_y=scale_y)
                if point is not None:
                    scaled_sample[point_key] = point
            scaled_samples.append(scaled_sample)
        scaled["path_samples"] = scaled_samples
    try:
        scaled["span_px"] = round(float(scaled.get("span_px") or 0.0) * avg_scale, 2)
    except Exception:
        pass
    return scaled


def _scale_decision_to_source_coordinates(
    decision: VisualPromptDecision | None,
    *,
    scale_x: float,
    scale_y: float,
    source_image_size: list[int],
    llm_image_size: list[int],
) -> VisualPromptDecision | None:
    """Map LVLM-rendered prompt geometry back into the original FPV image plane."""
    if decision is None:
        return None
    scaled_prompt = _scale_rendered_prompt_geometry(decision.rendered_prompt, scale_x=scale_x, scale_y=scale_y)
    debug = dict(decision.debug or {})
    debug["llm_coordinate_mapping"] = {
        "from": "llm_image_uv",
        "to": "image_uv",
        "scale_x": round(float(scale_x), 8),
        "scale_y": round(float(scale_y), 8),
        "source_image_size": source_image_size,
        "llm_image_size": llm_image_size,
    }
    return VisualPromptDecision(
        selected_candidate_id=decision.selected_candidate_id,
        selected_prompt_type=decision.selected_prompt_type,
        instruction=decision.instruction,
        confidence=decision.confidence,
        reason=decision.reason,
        reject_all=decision.reject_all,
        fallback_used=decision.fallback_used,
        arbiter_source=decision.arbiter_source,
        rendered_prompt=scaled_prompt,
        debug=debug,
    )


def _render_prompt_overlay_svg(
    *,
    image_href: str,
    image_width: int,
    image_height: int,
    rendered_prompt: Dict[str, Any] | None,
    title: str = "Visual Prompt Overlay",
) -> str:
    width = max(int(image_width), 32)
    height = max(int(image_height), 32)
    max_display_width = max(int(os.getenv("VISUAL_PROMPT_SVG_DISPLAY_WIDTH", "1200") or 1200), 320)
    display_width = min(width, max_display_width)
    display_height = max(int(round(display_width * height / max(width, 1))), 1)
    min_dim = min(width, height)
    title_font = max(min_dim * 0.018, 18.0)
    label_font = max(min_dim * 0.014, 14.0)
    line_stroke = max(min_dim * 0.009, 5.0)
    outline_stroke = max(min_dim * 0.005, 2.0)
    marker_radius = max(min_dim * 0.011, 6.0)
    bg = (
        f'<image href="{image_href}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="xMidYMid slice" />'
        if image_href
        else f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0f172a" />'
    )
    if not isinstance(rendered_prompt, dict):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_width}" height="{display_height}" viewBox="0 0 {width} {height}">'
            f"{bg}"
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="rgba(2,6,23,0.10)" />'
            f'<text x="{title_font:.1f}" y="{title_font * 1.7:.1f}" fill="#e2e8f0" font-size="{title_font:.1f}">{escape(title)}</text>'
            f'<text x="{width/2:.1f}" y="{height/2:.1f}" text-anchor="middle" fill="#fca5a5" font-size="{title_font:.1f}">No rendered prompt</text>'
            "</svg>"
        )

    prompt_type = str(rendered_prompt.get("type") or "arrow").strip().lower()
    center = rendered_prompt.get("center_uv") or [width * 0.5, height * 0.55]
    try:
        cx = float(center[0])
        cy = float(center[1])
    except Exception:
        cx = width * 0.5
        cy = height * 0.55
    span = max(float(rendered_prompt.get("span_px", 72.0) or 72.0), 16.0)
    direction = rendered_prompt.get("direction_vec") or [0.0, -1.0]
    try:
        dx = float(direction[0])
        dy = float(direction[1])
    except Exception:
        dx, dy = 0.0, -1.0
    coverage_polygon = rendered_prompt.get("coverage_polygon_uv")
    path_samples = rendered_prompt.get("path_samples")

    anchor_x = width * 0.5
    anchor_y = height * 0.94
    label = str(rendered_prompt.get("semantic_tag") or prompt_type)
    termination_semantics = str(rendered_prompt.get("termination_semantics") or "").strip()
    support_confidence = rendered_prompt.get("support_confidence")

    overlay_parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="rgba(2,6,23,0.06)" />',
        f'<rect x="{title_font * 0.6:.1f}" y="{title_font * 0.45:.1f}" width="{min(width * 0.72, title_font * 24):.1f}" height="{title_font * 1.65:.1f}" rx="{title_font * 0.35:.1f}" fill="rgba(2,6,23,0.58)" />',
        f'<text x="{title_font:.1f}" y="{title_font * 1.55:.1f}" fill="#e2e8f0" font-size="{title_font:.1f}" font-family="monospace">{escape(title)}</text>',
    ]

    if isinstance(coverage_polygon, list) and len(coverage_polygon) >= 3:
        points = " ".join(
            f"{float(point[0]):.1f},{float(point[1]):.1f}"
            for point in coverage_polygon
            if isinstance(point, (list, tuple)) and len(point) == 2
        )
        if points:
            overlay_parts.append(
                f'<polygon points="{points}" fill="rgba(34,197,94,0.12)" stroke="#22c55e" stroke-width="{outline_stroke:.1f}" />'
            )

    if isinstance(path_samples, list):
        left_points = []
        right_points = []
        center_points = []
        for sample in path_samples:
            if not isinstance(sample, dict):
                continue
            left = sample.get("left_edge_uv")
            right = sample.get("right_edge_uv")
            center_sample = sample.get("center_uv")
            if isinstance(left, (list, tuple)) and len(left) == 2:
                left_points.append([float(left[0]), float(left[1])])
            if isinstance(right, (list, tuple)) and len(right) == 2:
                right_points.append([float(right[0]), float(right[1])])
            if isinstance(center_sample, (list, tuple)) and len(center_sample) == 2:
                center_points.append([float(center_sample[0]), float(center_sample[1])])
        if len(left_points) >= 2:
            left_str = " ".join(f"{point[0]:.1f},{point[1]:.1f}" for point in left_points)
            overlay_parts.append(
                f'<polyline points="{left_str}" fill="none" stroke="#22c55e" stroke-width="{outline_stroke * 1.2:.1f}" stroke-linecap="round" />'
            )
        if len(right_points) >= 2:
            right_str = " ".join(f"{point[0]:.1f},{point[1]:.1f}" for point in right_points)
            overlay_parts.append(
                f'<polyline points="{right_str}" fill="none" stroke="#22c55e" stroke-width="{outline_stroke * 1.2:.1f}" stroke-linecap="round" />'
            )

        if center_points:
            cx = float(center_points[-1][0])
            cy = float(center_points[-1][1])

    path_uv = rendered_prompt.get("path_uv")
    valid_path = (
        isinstance(path_uv, list)
        and len(path_uv) >= 2
        and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in path_uv)
    )

    if prompt_type in {"arrival", "arrival_marker", "destination_marker"}:
        overlay_parts.extend(
            [
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{marker_radius * 2.4:.1f}" fill="rgba(34,197,94,0.24)" stroke="#22c55e" stroke-width="{outline_stroke * 1.8:.1f}" />',
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{marker_radius * 0.8:.1f}" fill="#22c55e" />',
                f'<path d="M {cx - marker_radius * 1.0:.1f} {cy + marker_radius * 0.05:.1f} L {cx - marker_radius * 0.28:.1f} {cy + marker_radius * 0.75:.1f} L {cx + marker_radius * 1.15:.1f} {cy - marker_radius * 0.9:.1f}" fill="none" stroke="#ecfdf5" stroke-width="{outline_stroke * 1.7:.1f}" stroke-linecap="round" stroke-linejoin="round" />',
                f'<text x="{cx:.1f}" y="{cy + marker_radius * 3.45:.1f}" text-anchor="middle" fill="#dcfce7" font-size="{title_font:.1f}" font-family="monospace">ARRIVED</text>',
            ]
        )
    elif prompt_type in {"arrow", "arrow_path", "path_corridor"} and valid_path:
        points = " ".join(f"{float(point[0]):.1f},{float(point[1]):.1f}" for point in path_uv)
        overlay_parts.append(
            f'<polyline points="{points}" fill="none" stroke="#f97316" stroke-width="{line_stroke:.1f}" stroke-linecap="round" stroke-linejoin="round" />'
        )
        end_x = float(path_uv[-1][0])
        end_y = float(path_uv[-1][1])
        prev_x = float(path_uv[-2][0])
        prev_y = float(path_uv[-2][1])
        vx = end_x - prev_x
        vy = end_y - prev_y
        norm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
        ux, uy = vx / norm, vy / norm
        px, py = -uy, ux
        ah = max(span * 0.16, marker_radius * 2.2)
        ax1 = end_x - ux * ah - px * ah * 0.55
        ay1 = end_y - uy * ah - py * ah * 0.55
        ax2 = end_x - ux * ah + px * ah * 0.55
        ay2 = end_y - uy * ah + py * ah * 0.55
        overlay_parts.append(
            f'<polygon points="{end_x:.1f},{end_y:.1f} {ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f}" fill="#f97316" />'
        )
    elif prompt_type == "arrow":
        overlay_parts.append(
            f'<line x1="{anchor_x:.1f}" y1="{anchor_y:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" stroke="#f97316" stroke-width="{line_stroke:.1f}" stroke-linecap="round" />'
        )
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        ah = max(span * 0.16, marker_radius * 2.2)
        ax1 = cx - ux * ah - px * ah * 0.55
        ay1 = cy - uy * ah - py * ah * 0.55
        ax2 = cx - ux * ah + px * ah * 0.55
        ay2 = cy - uy * ah + py * ah * 0.55
        overlay_parts.append(
            f'<polygon points="{cx:.1f},{cy:.1f} {ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f}" fill="#f97316" />'
        )
    elif prompt_type == "region":
        overlay_parts.append(
            f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{span * 0.55:.1f}" ry="{span * 0.32:.1f}" fill="rgba(168,85,247,0.18)" stroke="#a855f7" stroke-width="{outline_stroke * 1.5:.1f}" />'
        )
    else:
        left_x = cx - span * 0.6
        right_x = cx + span * 0.6
        top_y = cy - span * 0.52
        bottom_y = cy + span * 0.52
        overlay_parts.append(
            f'<polygon points="{anchor_x:.1f},{anchor_y:.1f} {left_x:.1f},{bottom_y:.1f} {cx:.1f},{top_y:.1f} {right_x:.1f},{bottom_y:.1f}" fill="rgba(56,189,248,0.22)" stroke="#38bdf8" stroke-width="{outline_stroke * 1.5:.1f}" />'
        )

    if prompt_type not in {"arrival", "arrival_marker", "destination_marker"}:
        termination_label = termination_semantics[:32] if termination_semantics else ""
        support_label = ""
        if support_confidence is not None:
            try:
                support_label = f"support={float(support_confidence):.2f}"
            except Exception:
                support_label = ""
        overlay_parts.extend(
            [
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{marker_radius:.1f}" fill="#e2e8f0" />',
                f'<rect x="{cx + marker_radius * 1.4:.1f}" y="{cy - label_font * 1.45:.1f}" width="{min(width - cx - marker_radius * 2.0, max(label_font * 11.0, 1.0)):.1f}" height="{label_font * 1.7:.1f}" rx="{label_font * 0.35:.1f}" fill="rgba(2,6,23,0.58)" />',
                f'<text x="{cx + marker_radius * 1.9:.1f}" y="{cy - label_font * 0.22:.1f}" fill="#e2e8f0" font-size="{label_font:.1f}" font-family="monospace">{escape(label[:42])}</text>',
            ]
        )
        if termination_label or support_label:
            meta_text = " | ".join(item for item in [termination_label, support_label] if item)
            overlay_parts.extend(
                [
                    f'<rect x="{title_font * 0.6:.1f}" y="{title_font * 2.3:.1f}" width="{min(width * 0.78, max(len(meta_text) * label_font * 0.72, title_font * 8.0)):.1f}" height="{label_font * 1.65:.1f}" rx="{label_font * 0.35:.1f}" fill="rgba(2,6,23,0.52)" />',
                    f'<text x="{title_font:.1f}" y="{title_font * 3.0:.1f}" fill="#bbf7d0" font-size="{label_font:.1f}" font-family="monospace">{escape(meta_text)}</text>',
                ]
            )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_width}" height="{display_height}" viewBox="0 0 {width} {height}">'
        f"{bg}{''.join(overlay_parts)}</svg>"
    )


def _svg_to_data_url(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg, safe=":/?&=,+-_.!~*'()#% ")


def _is_arrival_state(route_context: Dict[str, Any], *, threshold_m: float | None = None) -> bool:
    enable_arrival_marker = str(os.getenv("VISUAL_PROMPT_ENABLE_ARRIVAL_MARKER", "")).strip().lower()
    if enable_arrival_marker not in {"1", "true", "yes", "on"}:
        return False
    progress = route_context.get("progress", {}) or {}
    pose_match = route_context.get("pose_match", {}) or {}
    if bool(pose_match.get("off_route")):
        return False
    try:
        remaining = float(progress.get("remaining_distance_m"))
    except Exception:
        return False
    if threshold_m is None:
        threshold_m = float(os.getenv("VISUAL_PROMPT_ARRIVAL_THRESHOLD_M", "2.0") or 2.0)
    return remaining <= max(float(threshold_m), 0.0)


def _arrival_visual_prompt_decision(
    *,
    route_context: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> VisualPromptDecision:
    progress = route_context.get("progress", {}) or {}
    try:
        remaining = float(progress.get("remaining_distance_m"))
    except Exception:
        remaining = 0.0
    return VisualPromptDecision(
        selected_candidate_id=None,
        selected_prompt_type="arrival_marker",
        instruction="You have arrived at the destination.",
        confidence=0.95,
        reason=f"RouteContext reports remaining_distance_m={remaining:.2f}, within the arrival threshold.",
        reject_all=False,
        fallback_used=False,
        arbiter_source="route_arrival_state",
        rendered_prompt={
            "type": "arrival_marker",
            "center_uv": [float(image_width) * 0.5, float(image_height) * 0.68],
            "direction_vec": [0.0, 0.0],
            "span_px": max(min(float(image_width), float(image_height)) * 0.12, 48.0),
            "path_uv": None,
            "mask": None,
            "semantic_tag": "destination_arrived",
        },
        debug={
            "remaining_distance_m": remaining,
            "arrival_threshold_m": float(os.getenv("VISUAL_PROMPT_ARRIVAL_THRESHOLD_M", "2.0") or 2.0),
            "arrival_marker_enabled": True,
        },
    )


def _reject_visual_prompt_decision(
    *,
    reason: str,
    arbiter_source: str,
    route_context: Dict[str, Any],
) -> VisualPromptDecision:
    local = route_context.get("local_semantics", {}) or {}
    return VisualPromptDecision(
        selected_candidate_id=None,
        selected_prompt_type=None,
        instruction=str(local.get("short_action_summary") or "Visual prompt unavailable for this frame.").strip(),
        confidence=0.0,
        reason=str(reason).strip() or "Visual prompt generation failed.",
        reject_all=True,
        fallback_used=False,
        arbiter_source=arbiter_source,
        rendered_prompt=None,
        debug={},
    )


def _apply_segmentation_constraint(
    *,
    decision: VisualPromptDecision | None,
    route_context: Dict[str, Any],
    segmentation_context: Dict[str, Any] | None,
) -> tuple[VisualPromptDecision | None, Dict[str, Any] | None]:
    """Hard V-stage: validate, repair, or reject prompt geometry.

    Flow:
        1. Validate the rendered prompt against the source-image mask.
        2. If invalid, attempt deterministic safe-corridor repair.
           - Local width / edge-margin failures are repaired row by row.
           - Obstacle / forbidden-hit failures trigger a full replan from the
             obstacle-aware safe corridor prior.
        3. Re-run validation after repair.
        4. Accept only if the repaired prompt passes the same hard checks.

    This keeps the LVLM creative but makes the final geometry auditable.
    """
    if decision is None or not isinstance(decision.rendered_prompt, dict) or not isinstance(segmentation_context, dict):
        return decision, None
    validation = validate_rendered_prompt_against_segmentation(decision.rendered_prompt, segmentation_context)
    if validation is None:
        return decision, None
    decision.debug = dict(decision.debug or {})
    decision.debug["fpv_segmentation_validation"] = validation
    if validation.get("valid"):
        return decision, validation

    repaired_prompt, repair_debug = repair_rendered_prompt_with_safe_corridor(
        decision.rendered_prompt,
        segmentation_context,
        validation=validation,
    )
    if isinstance(repair_debug, dict):
        decision.debug["segmentation_safe_corridor_repair"] = repair_debug
    if isinstance(repaired_prompt, dict) and repair_debug and bool(repair_debug.get("applied")):
        repaired_validation = validate_rendered_prompt_against_segmentation(repaired_prompt, segmentation_context)
        decision.debug["fpv_segmentation_validation_before_repair"] = validation
        decision.debug["fpv_segmentation_validation_after_repair"] = repaired_validation
        if isinstance(repaired_validation, dict) and repaired_validation.get("valid"):
            decision.rendered_prompt = repaired_prompt
            decision.reason = (
                str(decision.reason or "").strip()
                + " Geometry was recentered and inset using the segmentation safe-corridor guardrail."
            ).strip()
            decision.confidence = min(max(float(decision.confidence or 0.0), 0.0), 0.86)
            decision.arbiter_source = f"{decision.arbiter_source}_segmentation_repaired"
            decision.debug["fpv_segmentation_validation"] = repaired_validation
            return decision, repaired_validation

    reason = (
        "FPV segmentation traversability constraint rejected the rendered prompt: "
        f"{validation.get('reason')}"
    )
    rejected = _reject_visual_prompt_decision(
        reason=reason,
        arbiter_source="fpv_segmentation_reject",
        route_context=route_context,
    )
    rejected.debug["fpv_segmentation_validation"] = validation
    rejected.debug["rejected_rendered_prompt"] = decision.rendered_prompt
    rejected.debug["rejected_instruction"] = decision.instruction
    rejected.debug["rejected_confidence"] = decision.confidence
    return rejected, validation


def _arbiter_extra_image_urls(
    *,
    fpv_segmentation: Dict[str, Any] | None,
    enable_sam_visualization_input: bool,
    llm_max_side: int,
    llm_max_bytes: int,
    llm_jpeg_quality: int,
) -> tuple[list[str], dict[str, Any]]:
    debug: dict[str, Any] = {
        "enabled": bool(enable_sam_visualization_input),
        "used": False,
        "source": None,
        "error": None,
    }
    if not enable_sam_visualization_input or not isinstance(fpv_segmentation, dict):
        return [], debug

    visualization_url = str(
        fpv_segmentation.get("visualization_url")
        or fpv_segmentation.get("visualization_image_url")
        or ""
    ).strip()
    if visualization_url:
        debug["used"] = True
        debug["source"] = "visualization_url"
        return [visualization_url], debug

    visualization_path = str(fpv_segmentation.get("visualization_path") or "").strip()
    if not visualization_path:
        debug["error"] = "visualization_path_missing"
        return [], debug

    try:
        overlay_data_url = _local_image_path_to_llm_data_url(
            visualization_path,
            max_side=llm_max_side,
            max_bytes=llm_max_bytes,
            jpeg_quality=llm_jpeg_quality,
        )
        debug["used"] = True
        debug["source"] = "visualization_path"
        debug["visualization_path"] = str(Path(visualization_path).expanduser())
        return [overlay_data_url], debug
    except Exception as exc:
        debug["error"] = str(exc)
        debug["visualization_path"] = str(Path(visualization_path).expanduser())
        return [], debug


def _segmentation_overlay_legend(fpv_segmentation_summary: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(fpv_segmentation_summary, dict):
        return None
    return {
        "summary_source": "from_same_mask_entity",
        "color_semantics": {
            "traversable": "green preferred walking support",
            "conditional_traversable": "yellow/orange usable connector support",
            "boundary": "teal",
            "forbidden": "red",
        },
        "traversable_class_names": list(fpv_segmentation_summary.get("traversable_class_names") or []),
        "conditional_traversable_class_names": list(fpv_segmentation_summary.get("conditional_traversable_class_names") or []),
        "allowed_traversable_class_names": list(fpv_segmentation_summary.get("allowed_traversable_class_names") or []),
        "boundary_class_names": list(fpv_segmentation_summary.get("boundary_class_names") or []),
        "forbidden_class_names": list(fpv_segmentation_summary.get("forbidden_class_names") or []),
    }


def _spatial_constraint_regions(regions: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(regions, list):
        return []
    compact: list[dict[str, Any]] = []
    for region in regions[: max(int(limit), 1)]:
        if not isinstance(region, dict):
            continue
        compact.append(
            {
                "label": region.get("label"),
                "region_id": region.get("region_id"),
                "region_role": region.get("region_role"),
                "dominant_class_name": region.get("dominant_class_name"),
                "center_uv": region.get("center_uv"),
                "bbox_uv": region.get("bbox_uv"),
                "side": region.get("side"),
                "route_intent_score": region.get("route_intent_score"),
                "area_ratio": region.get("area_ratio"),
            }
        )
    return compact


def _build_som_spatial_constraints(
    *,
    navigation_mode: str,
    image_width: int,
    image_height: int,
    fpv_segmentation_summary: Dict[str, Any] | None,
    som_summary: Dict[str, Any] | None,
    som_overlay_legend: Dict[str, Any] | None,
    safe_corridor_prior: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Build explicit image-plane guardrails for the LVLM from segmentation/SoM.

    These constraints are not candidate geometry. They define admissible regions,
    bottom-start rules, and corridor-width limits that the direct generator and
    verifier must satisfy while still reading the FPV image themselves.

    Important: safe_corridor_prior is a teacher guardrail. It is passed to the
    LVLM as concrete centered/inset geometry, but final acceptance still depends
    on the deterministic V(mask) validation stage.
    """
    if not isinstance(fpv_segmentation_summary, dict) and not isinstance(som_summary, dict):
        return None
    start_v_min_ratio = 0.82
    preferred_v_min_ratio = 0.88
    u_range_ratio = [0.40, 0.60]
    preferred_u_range_ratio = [0.45, 0.55]
    bottom_openings = []
    if isinstance(fpv_segmentation_summary, dict):
        bottom_openings = list(fpv_segmentation_summary.get("bottom_openings") or [])
    preferred_regions = _spatial_constraint_regions((som_summary or {}).get("regions"), limit=8)
    conditional_regions = _spatial_constraint_regions((som_summary or {}).get("conditional_regions"), limit=8)
    forbidden_regions = _spatial_constraint_regions((som_summary or {}).get("forbidden_regions"), limit=12)
    max_width_ratio, preferred_width_ratio = corridor_width_ratio_limits(navigation_mode)
    centerline_offset_ratio = centerline_offset_ratio_limit()
    edge_margin_ratio = edge_margin_ratio_limit()
    label_mapping = []
    if isinstance(som_overlay_legend, dict):
        label_mapping = [
            {
                "label": item.get("label"),
                "region_id": item.get("region_id"),
                "region_role": item.get("region_role"),
                "dominant_class_name": item.get("dominant_class_name"),
                "bbox_uv": item.get("bbox_uv"),
                "center_uv": item.get("center_uv"),
            }
            for item in (som_overlay_legend.get("label_mapping") or [])
            if isinstance(item, dict)
        ]
    return {
        "schema_version": 1,
        "coordinate_space": "llm_image_uv",
        "image_size": {"width": int(image_width), "height": int(image_height)},
        "start_constraint": {
            "rule": "first path sample must start from the bottom-center image band, close to the viewer's current position",
            "required_v_min_ratio": start_v_min_ratio,
            "preferred_v_min_ratio": preferred_v_min_ratio,
            "required_u_range_ratio": u_range_ratio,
            "preferred_u_range_ratio": preferred_u_range_ratio,
            "required_v_min_px": round(float(image_height) * start_v_min_ratio, 2),
            "preferred_v_min_px": round(float(image_height) * preferred_v_min_ratio, 2),
            "required_u_range_px": [
                round(float(image_width) * u_range_ratio[0], 2),
                round(float(image_width) * u_range_ratio[1], 2),
            ],
            "preferred_u_range_px": [
                round(float(image_width) * preferred_u_range_ratio[0], 2),
                round(float(image_width) * preferred_u_range_ratio[1], 2),
            ],
            "bottom_openings": bottom_openings,
            "if_no_bottom_opening": "draw a short prompt grounded in the bottom-center band only if visibly safe, otherwise reject_all=true",
        },
        "corridor_width_constraint": {
            "rule": "path_corridor width must be compact and obstacle-aware, not the full extent of every visible traversable pixel",
            "max_width_ratio": round(float(max_width_ratio), 4),
            "preferred_width_ratio": round(float(preferred_width_ratio), 4),
            "width_ratio_reference": "image_width",
            "config_env": CORRIDOR_WIDTH_RATIO_ENV,
            "repair_instruction": "If a proposed corridor is too wide, keep the centerline if safe but move left_edge_uv/right_edge_uv inward until both edges stay on preferred support and clear forbidden labels.",
        },
        "centerline_constraint": {
            "rule": "for each path sample, place center_uv near the middle of the preferred traversable horizontal run at the same image row, like the geometric fallback centerline",
            "max_offset_ratio": round(float(centerline_offset_ratio), 4),
            "offset_reference": "half_width_of_preferred_traversable_run",
            "config_env": CENTERLINE_OFFSET_RATIO_ENV,
            "repair_instruction": "If center_uv is close to either traversable edge, move it toward the midpoint of the same-row preferred traversable run unless visible safety requires a shorter prompt.",
        },
        "edge_margin_constraint": {
            "rule": "left_edge_uv and right_edge_uv must stay inside the allowed traversable run and keep clearance from the run boundary; do not trace the exact sidewalk/free-space edge",
            "min_margin_ratio": round(float(edge_margin_ratio), 4),
            "margin_reference": "image_width",
            "min_margin_px": round(float(image_width) * float(edge_margin_ratio), 2),
            "config_env": EDGE_MARGIN_RATIO_ENV,
            "repair_instruction": "Move corridor boundaries inward from grass, road, curb, vehicles, people, and segmentation edges; the prompt should describe a safe lane inside the walkable area, not the whole traversable mask.",
        },
        "preferred_regions": preferred_regions,
        "conditional_connector_regions": conditional_regions,
        "forbidden_regions": forbidden_regions,
        "label_mapping": label_mapping,
        "safe_corridor_prior": safe_corridor_prior,
        "hard_rules": [
            "Do not use these constraints as a precomputed path; directly regress the visual prompt from the images.",
            "Do not place center_uv, left_edge_uv, or right_edge_uv inside forbidden_regions or red SoM labels.",
            "Keep path_corridor width within corridor_width_constraint; do not draw a full-image free-space polygon as the navigation prompt.",
            "Keep center_uv close to centerline_constraint; avoid hugging either side of the traversable region.",
            "Keep left_edge_uv/right_edge_uv inside edge_margin_constraint; boundaries need clearance from traversable-region edges.",
            "Use safe_corridor_prior as a concrete teacher guardrail when present: follow its centered path_samples unless the image shows an obstacle or occlusion that requires shortening/rejection.",
            "Road, curb, and gutter labels are conditional connectors only; prefer sidewalk/crosswalk/plaza support when visible.",
            "If segmentation_validation lists forbidden hit points, move the corresponding center/boundary point away from those labels or reject.",
            "If no bottom-center visible support exists, reject_all=true instead of starting mid-image.",
        ],
    }


def _segmentation_reject_code(validation: Dict[str, Any] | None) -> str | None:
    if not isinstance(validation, dict):
        return None
    reason = str(validation.get("reason") or "")
    if "low_center_traversable_ratio" in reason:
        return "low_center_traversable_ratio"
    if "center_path_hits_forbidden_surface" in reason:
        return "center_hits_forbidden"
    if "boundary_hits_forbidden_surface" in reason:
        return "boundary_hits_forbidden"
    if "mode_disallowed" in reason:
        return "mode_disallowed_surface"
    if "polygon_hits_forbidden" in reason:
        return "polygon_hits_forbidden"
    if "low_polygon_allowed_ratio" in reason:
        return "low_polygon_allowed_ratio"
    if "low_polygon_traversable_ratio" in reason:
        return "low_polygon_traversable_ratio"
    if "polygon_curb_ratio_too_high" in reason:
        return "polygon_curb_ratio_too_high"
    if "centerline_offset_too_high" in reason:
        return "centerline_offset_too_high"
    if "edge_margin_too_low" in reason:
        return "edge_margin_too_low"
    if "no_center_path_samples" in reason:
        return "no_center_path_samples"
    if "start_point_not_in_lower_middle_band" in reason:
        return "start_point_not_in_lower_middle_band"
    return None


def _build_inputs_for_A(
    *,
    route_context: Dict[str, Any],
    image_width: int,
    image_height: int,
    navigation_mode: str,
    fpv_segmentation_summary: Dict[str, Any] | None,
    som_summary: Dict[str, Any] | None,
    som_candidate_hints: list[dict[str, Any]],
    som_overlay_legend: Dict[str, Any] | None,
    bev_map_overlay_legend: Dict[str, Any] | None,
    som_spatial_constraints: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Assemble the A-stage LVLM contract from route intent and visual priors.

    The A-stage question is intentionally JSON: it is easier to audit, replay,
    scale between coordinate spaces, and compare across regression runs.
    """
    teacher_alignment_report = None
    overlay_legend = _segmentation_overlay_legend(fpv_segmentation_summary)
    if isinstance(fpv_segmentation_summary, dict):
        candidate_alignment = fpv_segmentation_summary.get("alignment_report")
        if isinstance(candidate_alignment, dict):
            teacher_alignment_report = candidate_alignment
    return {
        "summary_source": "from_same_mask_entity" if isinstance(fpv_segmentation_summary, dict) else "not_available",
        "alignment_report": teacher_alignment_report,
        "overlay_legend": overlay_legend,
        "question": build_direct_prompt_question(
            route_context=route_context,
            prompt_proposals=[],
            visual_regions={},
            image_width=image_width,
            image_height=image_height,
            navigation_mode=navigation_mode,
            fpv_segmentation_summary=fpv_segmentation_summary,
            som_summary=som_summary,
            som_candidate_hints=som_candidate_hints,
            som_overlay_legend=som_overlay_legend,
            bev_map_overlay_legend=bev_map_overlay_legend,
            som_spatial_constraints=som_spatial_constraints,
            alignment_report=teacher_alignment_report,
            overlay_legend=overlay_legend,
            summary_source=(
                "from_same_mask_entity" if isinstance(fpv_segmentation_summary, dict) else "not_available"
            ),
        ),
        "system_prompt": build_direct_prompt_system_prompt(),
    }


async def _run_A_generator(
    *,
    image_path: Optional[str],
    image_url: Optional[str],
    question: str,
    system_prompt: str,
    llm_detail: str,
    extra_image_urls: list[str],
    route_context: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[Dict[str, Any] | None, VisualPromptDecision | None]:
    """Call the LVLM generator and coerce its JSON into a typed decision."""
    if str(image_path or "").strip():
        llm_raw = await arbitrate_local_pov_candidates_with_llm(
            image_path=str(image_path),
            question=question,
            detail=llm_detail,
            system_prompt=system_prompt,
            extra_image_urls=extra_image_urls,
        )
    else:
        llm_raw = await arbitrate_pov_candidates_with_llm(
            image_url=str(image_url),
            question=question,
            detail=llm_detail,
            system_prompt=system_prompt,
            extra_image_urls=extra_image_urls,
        )
    decision = coerce_direct_prompt_decision(
        raw=llm_raw,
        route_context=route_context,
        image_width=image_width,
        image_height=image_height,
    )
    return llm_raw, decision


def _run_V_constraint(
    *,
    decision: VisualPromptDecision | None,
    route_context: Dict[str, Any],
    segmentation_context: Dict[str, Any] | None,
    constraint_stage: str,
) -> tuple[VisualPromptDecision | None, Dict[str, Any] | None]:
    """Apply deterministic segmentation constraints and annotate debug metadata."""
    constrained_decision, validation = _apply_segmentation_constraint(
        decision=decision,
        route_context=route_context,
        segmentation_context=segmentation_context,
    )
    if isinstance(constrained_decision, VisualPromptDecision):
        constrained_decision.debug = dict(constrained_decision.debug or {})
        constrained_decision.debug["constraint_stage"] = str(constraint_stage)
        reject_code = _segmentation_reject_code(validation)
        if reject_code:
            constrained_decision.debug["segmentation_reject_code"] = reject_code
    return constrained_decision, validation


async def _run_V_verifier(
    *,
    image_path: Optional[str],
    image_url: Optional[str],
    route_context: Dict[str, Any],
    rendered_prompt: Dict[str, Any],
    image_width: int,
    image_height: int,
    navigation_mode: str,
    fpv_segmentation_summary: Dict[str, Any] | None,
    som_summary: Dict[str, Any] | None,
    som_candidate_hints: list[dict[str, Any]],
    som_overlay_legend: Dict[str, Any] | None,
    bev_map_overlay_legend: Dict[str, Any] | None,
    som_spatial_constraints: Dict[str, Any] | None,
    segmentation_validation: Dict[str, Any] | None,
    alignment_report: Dict[str, Any] | None,
    overlay_legend: Dict[str, Any] | None,
    summary_source: str,
    llm_detail: str,
    extra_image_urls: list[str],
) -> tuple[Dict[str, Any] | None, str, VisualPromptDecision | None]:
    """Call the optional LVLM verifier/refiner for a proposed rendered prompt."""
    verifier_question = build_direct_prompt_verifier_question(
        route_context=route_context,
        rendered_prompt=rendered_prompt,
        image_width=image_width,
        image_height=image_height,
        navigation_mode=navigation_mode,
        fpv_segmentation_summary=fpv_segmentation_summary,
        som_summary=som_summary,
        som_candidate_hints=som_candidate_hints,
        som_overlay_legend=som_overlay_legend,
        bev_map_overlay_legend=bev_map_overlay_legend,
        som_spatial_constraints=som_spatial_constraints,
        segmentation_validation=segmentation_validation,
        alignment_report=alignment_report,
        overlay_legend=overlay_legend,
        summary_source=summary_source,
    )
    verifier_system_prompt = build_direct_prompt_verifier_system_prompt()
    if str(image_path or "").strip():
        verifier_raw = await arbitrate_local_pov_candidates_with_llm(
            image_path=str(image_path),
            question=verifier_question,
            detail=llm_detail,
            system_prompt=verifier_system_prompt,
            extra_image_urls=extra_image_urls,
        )
    else:
        verifier_raw = await arbitrate_pov_candidates_with_llm(
            image_url=str(image_url),
            question=verifier_question,
            detail=llm_detail,
            system_prompt=verifier_system_prompt,
            extra_image_urls=extra_image_urls,
        )
    verified_decision = coerce_direct_prompt_decision(
        raw=verifier_raw,
        route_context=route_context,
        image_width=image_width,
        image_height=image_height,
    )
    return verifier_raw, verifier_question, verified_decision


async def run_zero_shot_visual_prompt_pipeline(
    *,
    gps: Dict[str, float],
    route_polyline: List[Dict[str, Any]] | List[List[float]],
    coarse_waypoints: List[Dict[str, Any]],
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    fpv_segmentation: Optional[Dict[str, Any]] = None,
    sam3_request: Optional[Dict[str, Any]] = None,
    heading_deg: Optional[float] = None,
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
    front_extent_m: float = 30.0,
    back_extent_m: float = 10.0,
    side_extent_m: float = 15.0,
    visual_top_k: int = 5,
    candidate_top_k: int = 5,
    llm_detail: str = "auto",
    navigation_mode: str = "foot",
    enable_verifier: bool = False,
    enable_segmentation_overlay_input: bool = False,
    enable_som_overlay_input: bool | None = None,
    enable_sam_visualization_input: bool = False,
    enable_bev_map_provider: bool = False,
    bev_map_provider_session: Any | None = None,
    enforce_alignment: bool = True,
    emit_qa_bundle: bool = False,
    emit_llm_input_images: bool = False,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the end-to-end FPV visual prompt pipeline.

    Flow:
    1. Build route context from GPS/route/heading.
    2. Normalize the FPV frame to an LVLM-safe image size.
    3. Load or generate FPV segmentation, then derive SoM region summaries.
    4. Ask A(generator) to directly regress one prompt in LVLM coordinates.
    5. Map the prompt back to source pixels and enforce V(segmentation) checks.
    6. Optionally ask V(verifier) to repair/refine, then enforce V again.
    7. Return the decision, overlays, debug context, and optional QA bundle.
    """
    del visual_top_k, candidate_top_k
    resolved_enable_som_overlay_input = (
        bool(enable_segmentation_overlay_input)
        if enable_som_overlay_input is None
        else bool(enable_som_overlay_input)
    )
    started_at = perf_counter()
    resolved_request_id = str(request_id or uuid4())
    telemetry: Dict[str, Any] = {
        "request_id": resolved_request_id,
        "elapsed_ms": 0,
        "error_code": None,
        "stages": {},
    }

    def _mark_stage(
        stage: str,
        stage_started_at: float,
        *,
        status: str,
        error_code: str | None = None,
    ) -> None:
        telemetry["stages"][stage] = {
            "status": str(status),
            "elapsed_ms": int(round((perf_counter() - stage_started_at) * 1000.0)),
            "error_code": str(error_code) if error_code else None,
        }
        if error_code and telemetry.get("error_code") is None:
            telemetry["error_code"] = str(error_code)

    if not str(image_path or "").strip() and not str(image_url or "").strip():
        raise RuntimeError("run_zero_shot_visual_prompt_agent requires image_path or image_url")

    route_stage_started_at = perf_counter()
    state = build_route_state(
        gps=gps,
        heading_deg=heading_deg,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        lookahead_m=lookahead_m,
        off_route_threshold_m=off_route_threshold_m,
        front_extent_m=front_extent_m,
        back_extent_m=back_extent_m,
        side_extent_m=side_extent_m,
    )
    _mark_stage("route_context", route_stage_started_at, status="ok")
    route_context_dict = route_context_to_dict(state.route_context)
    local_bev_dict = local_bev_context_to_dict(state.local_bev_context)
    local_bev_dict["_origin_gps"] = {"lat": float(gps["lat"]), "lng": float(gps["lng"])}
    bev_map_error: str | None = None
    if enable_bev_map_provider:
        bev_map_stage_started_at = perf_counter()
        owned_bev_map_session = False
        bev_map_session = bev_map_provider_session
        try:
            if bev_map_session is None:
                bev_map_session = BEVMapProviderSession.from_env()
                await bev_map_session.connect()
                owned_bev_map_session = True
            semantic_layers = await bev_map_session.build_semantic_layers(
                gps=gps,
                heading_deg=heading_deg,
                local_bev_context=local_bev_dict,
            )
            local_bev_dict = dict(local_bev_dict)
            local_bev_dict["semantic_bev_layers"] = {
                **dict(local_bev_dict.get("semantic_bev_layers") or {}),
                **semantic_layers,
            }
            _mark_stage("bev_map_provider", bev_map_stage_started_at, status="ok")
        except Exception as exc:
            bev_map_error = str(exc)
            _mark_stage("bev_map_provider", bev_map_stage_started_at, status="error", error_code="BEV_MAP_PROVIDER_FAILED")
        finally:
            if owned_bev_map_session and bev_map_session is not None:
                try:
                    await bev_map_session.disconnect()
                except Exception:
                    pass
    else:
        _mark_stage("bev_map_provider", route_stage_started_at, status="skipped")
    prompt_image = _load_prompt_image(image_path=image_path, image_url=image_url)
    image_width, image_height = int(prompt_image.size[0]), int(prompt_image.size[1])
    llm_max_side, llm_max_bytes, llm_jpeg_quality = _llm_image_limits()
    llm_image_payload = _build_llm_image_payload(
        prompt_image,
        max_side=llm_max_side,
        max_bytes=llm_max_bytes,
        jpeg_quality=llm_jpeg_quality,
    )
    llm_image_url = str(llm_image_payload["image_data_url"])
    llm_image_width, llm_image_height = [int(value) for value in llm_image_payload["llm_image_size"]]
    source_to_llm_scale = llm_image_payload["source_to_llm_scale"]
    llm_to_source_scale = llm_image_payload["llm_to_source_scale"]
    qa_bundle: Dict[str, Any] | None = None
    candidate_bundle = None
    segmentation_validation: Dict[str, Any] | None = None
    segmentation_context = None
    segmentation_summary = None
    llm_segmentation_summary = None
    som_context: Dict[str, Any] | None = None
    som_summary: Dict[str, Any] | None = None
    llm_som_summary: Dict[str, Any] | None = None
    som_candidate_hints: list[dict[str, Any]] = []
    llm_som_candidate_hints: list[dict[str, Any]] = []
    som_spatial_constraints: Dict[str, Any] | None = None
    llm_som_spatial_constraints: Dict[str, Any] | None = None
    safe_corridor_prior: Dict[str, Any] | None = None
    llm_safe_corridor_prior: Dict[str, Any] | None = None
    som_overlay_payload: Dict[str, Any] | None = None
    som_overlay_legend: Dict[str, Any] | None = None
    bev_map_overlay_payload: Dict[str, Any] | None = None
    bev_map_overlay_legend: Dict[str, Any] | None = None
    segmentation_error: str | None = None
    segmentation_error_code: str | None = None
    segmentation_teacher_bundle: Dict[str, Any] | None = None
    extra_image_urls: list[str] = []
    multi_image_debug: dict[str, Any] = {
        "enabled": bool(
            resolved_enable_som_overlay_input
            or enable_sam_visualization_input
            or enable_bev_map_provider
        ),
        "som_overlay_enabled": bool(resolved_enable_som_overlay_input),
        "sam_visualization_enabled": bool(enable_sam_visualization_input),
        "used": False,
    }
    llm_input_images: list[dict[str, Any]] = []
    sam3_result: Dict[str, Any] | None = None
    normalized_sam3: Dict[str, Any] | None = None
    segmentation_stage_started_at = perf_counter()
    if not isinstance(fpv_segmentation, dict) and isinstance(sam3_request, dict):
        try:
            sam3_payload = dict(sam3_request)
            sam3_payload.setdefault("request_id", resolved_request_id)
            sam3_result = await segment_fpv_frame(**sam3_payload)
            normalized_sam3 = {
                "status": sam3_result.get("status"),
                "provider": sam3_result.get("provider"),
                "schema_version": sam3_result.get("schema_version"),
                "request_id": sam3_result.get("request_id"),
                "timing_ms": sam3_result.get("timing_ms"),
                "artifacts": sam3_result.get("artifacts"),
                "error": sam3_result.get("error"),
            }
            generated_fpv_segmentation = sam3_result.get("fpv_segmentation")
            if not isinstance(generated_fpv_segmentation, dict):
                artifacts = sam3_result.get("artifacts")
                if isinstance(artifacts, dict):
                    seg_npy = str(artifacts.get("segmentation_npy_path") or "").strip()
                    class_map = str(artifacts.get("class_mapping_path") or "").strip()
                    vis_path = str(artifacts.get("visualization_path") or "").strip()
                    if seg_npy and class_map:
                        generated_fpv_segmentation = {
                            "segmentation_npy_path": seg_npy,
                            "class_mapping_path": class_map,
                        }
                        if vis_path:
                            generated_fpv_segmentation["visualization_path"] = vis_path
            if str(sam3_result.get("status") or "").lower() != "ok":
                error_payload = sam3_result.get("error") or {}
                segmentation_error_code = str(error_payload.get("code") or "SEGMENTATION_PAYLOAD_INVALID")
                segmentation_error = str(
                    error_payload.get("message")
                    or "segment_fpv_frame returned status=error"
                )
                generated_fpv_segmentation = None
            if isinstance(generated_fpv_segmentation, dict):
                fpv_segmentation = generated_fpv_segmentation
            else:
                if segmentation_error is None:
                    segmentation_error = "segment_fpv_frame returned an invalid fpv_segmentation payload"
                    segmentation_error_code = "SEGMENTATION_PAYLOAD_INVALID"
        except Exception as exc:
            segmentation_error = str(exc)
            segmentation_error_code = "SEGMENTATION_COMMAND_FAILED"
    if isinstance(fpv_segmentation, dict):
        try:
            segmentation_context = load_fpv_segmentation_context(
                fpv_segmentation,
                navigation_mode=navigation_mode,
                image_width=image_width,
                image_height=image_height,
            )
            segmentation_summary = segmentation_summary_for_llm(segmentation_context)
            if isinstance(segmentation_summary, dict):
                llm_segmentation_summary = _scale_context_to_llm(
                    segmentation_summary,
                    scale_x=float(source_to_llm_scale["x"]),
                    scale_y=float(source_to_llm_scale["y"]),
                )
        except Exception as exc:
            segmentation_error = str(exc)
            segmentation_error_code = "SEGMENTATION_PAYLOAD_INVALID"
            if not enforce_alignment:
                segmentation_error = None
                segmentation_error_code = None
    if isinstance(segmentation_context, dict):
        # Segmentation and SoM share the same source-image mask. Keep both the
        # source-space debug payloads and the LVLM-space copies used in prompts.
        segmentation_teacher_bundle = build_segmentation_teacher_bundle(segmentation_context)
        safe_corridor_prior = build_safe_corridor_prior(segmentation_context)
        if isinstance(safe_corridor_prior, dict):
            llm_safe_corridor_prior = _scale_context_to_llm(
                safe_corridor_prior,
                scale_x=float(source_to_llm_scale["x"]),
                scale_y=float(source_to_llm_scale["y"]),
            )
        som_context = build_som_context(segmentation_context)
        som_candidate_hints = rank_som_regions_for_route(
            som_context,
            route_context_dict,
            top_k=5,
        )
        som_summary = som_summary_for_llm(
            som_context,
            ranked_regions=som_candidate_hints,
            max_regions=8,
        )
        if isinstance(som_summary, dict):
            llm_som_summary = _scale_context_to_llm(
                som_summary,
                scale_x=float(source_to_llm_scale["x"]),
                scale_y=float(source_to_llm_scale["y"]),
            )
        llm_som_candidate_hints = _scale_context_to_llm(
            som_candidate_hints,
            scale_x=float(source_to_llm_scale["x"]),
            scale_y=float(source_to_llm_scale["y"]),
        )
        if resolved_enable_som_overlay_input:
            som_overlay_payload = build_numbered_som_overlay_payload(
                image=prompt_image,
                som_context=som_context,
                ranked_regions=som_candidate_hints,
                max_regions=8,
                llm_max_side=llm_max_side,
                include_collage=False,
                image_format="JPEG",
                jpeg_quality=llm_jpeg_quality,
            )
            if isinstance(som_overlay_payload, dict):
                som_overlay_legend = som_overlay_payload.get("som_overlay_legend")
        llm_som_spatial_constraints = _build_som_spatial_constraints(
            navigation_mode=navigation_mode,
            image_width=llm_image_width,
            image_height=llm_image_height,
            fpv_segmentation_summary=llm_segmentation_summary,
            som_summary=llm_som_summary,
            som_overlay_legend=som_overlay_legend,
            safe_corridor_prior=llm_safe_corridor_prior,
        )
        som_spatial_constraints = _build_som_spatial_constraints(
            navigation_mode=navigation_mode,
            image_width=image_width,
            image_height=image_height,
            fpv_segmentation_summary=segmentation_summary,
            som_summary=som_summary,
            som_overlay_legend=(
                {
                    **som_overlay_legend,
                    "label_mapping": som_overlay_legend.get("source_label_mapping"),
                }
                if isinstance(som_overlay_legend, dict)
                else None
            ),
            safe_corridor_prior=safe_corridor_prior,
        )
        _mark_stage("segmentation", segmentation_stage_started_at, status="ok")
    elif segmentation_error:
        _mark_stage(
            "segmentation",
            segmentation_stage_started_at,
            status="error",
            error_code=segmentation_error_code or "SEGMENTATION_PAYLOAD_INVALID",
        )
    else:
        _mark_stage("segmentation", segmentation_stage_started_at, status="skipped")

    segmentation_extra_image_urls, multi_image_debug = _arbiter_extra_image_urls(
        fpv_segmentation=fpv_segmentation,
        enable_sam_visualization_input=enable_sam_visualization_input,
        llm_max_side=llm_max_side,
        llm_max_bytes=llm_max_bytes,
        llm_jpeg_quality=llm_jpeg_quality,
    )
    multi_image_debug["som_overlay_enabled"] = bool(resolved_enable_som_overlay_input)
    multi_image_debug["sam_visualization_enabled"] = bool(enable_sam_visualization_input)
    multi_image_debug["enabled"] = bool(
        resolved_enable_som_overlay_input
        or enable_sam_visualization_input
        or enable_bev_map_provider
    )
    multi_image_debug["llm_primary_image"] = {
        key: value
        for key, value in llm_image_payload.items()
        if key != "image_data_url"
    }
    extra_image_urls = []
    extra_image_records: list[dict[str, Any]] = []
    som_overlay_url = ""
    if resolved_enable_som_overlay_input and isinstance(som_overlay_payload, dict):
        som_overlay_url = str(som_overlay_payload.get("som_overlay_image_data_url") or "").strip()
        if som_overlay_url:
            extra_image_urls.append(som_overlay_url)
            extra_image_records.append(
                {
                    "role": "numbered_som_overlay",
                    "image_role": "numbered_som_overlay",
                    "data_url": som_overlay_url,
                    "metadata": {
                        "som_overlay_legend": som_overlay_legend,
                        "llm_image_size": (som_overlay_payload or {}).get("llm_image_size") if isinstance(som_overlay_payload, dict) else None,
                        "source_image_size": (som_overlay_payload or {}).get("source_image_size") if isinstance(som_overlay_payload, dict) else None,
                    },
                }
            )
            multi_image_debug["used"] = True
            multi_image_debug["primary_extra_image_role"] = "numbered_som_overlay"
            multi_image_debug["som_overlay_used"] = True
            multi_image_debug["som_overlay_source"] = "generated_numbered_som_overlay"
            multi_image_debug["som_overlay_legend"] = som_overlay_legend

    if enable_bev_map_provider:
        bev_map_overlay_payload = _build_bev_map_overlay_payload(
            local_bev_context=local_bev_dict,
            llm_max_bytes=llm_max_bytes,
            llm_jpeg_quality=llm_jpeg_quality,
        )
        if isinstance(bev_map_overlay_payload, dict):
            bev_map_overlay_url = str(bev_map_overlay_payload.get("bev_map_overlay_image_data_url") or "").strip()
            bev_map_overlay_legend = (
                bev_map_overlay_payload.get("legend")
                if isinstance(bev_map_overlay_payload.get("legend"), dict)
                else None
            )
            if bev_map_overlay_url:
                extra_image_urls.append(bev_map_overlay_url)
                extra_image_records.append(
                    {
                        "role": "bev_map_overlay",
                        "image_role": "bev_map_overlay",
                        "data_url": bev_map_overlay_url,
                        "metadata": {
                            key: value
                            for key, value in bev_map_overlay_payload.items()
                            if key != "bev_map_overlay_image_data_url"
                        },
                    }
                )
                multi_image_debug["used"] = True
                multi_image_debug["bev_map_overlay_used"] = True
                multi_image_debug["bev_map_overlay_legend"] = bev_map_overlay_legend

    for index, url in enumerate(segmentation_extra_image_urls):
        role = "segmentation_visualization" if index == 0 else f"extra_image_{index}"
        extra_image_urls.append(url)
        extra_image_records.append(
            {
                "role": role,
                "image_role": role,
                "data_url": url,
                "metadata": {"source": multi_image_debug.get("source")},
            }
        )

    if emit_llm_input_images:
        llm_input_images.append(
            {
                "role": "primary_fpv",
                "image_role": "primary_fpv",
                "data_url": llm_image_url,
                "metadata": {
                    key: value
                    for key, value in llm_image_payload.items()
                    if key != "image_data_url"
                },
            }
        )
        llm_input_images.extend(extra_image_records)

    llm_raw: Dict[str, Any] | None = None
    verifier_raw: Dict[str, Any] | None = None
    llm_error: str | None = None
    verifier_error: str | None = None
    decision = None
    llm_decision = None
    generator_decision = None
    question = ""
    system_prompt = ""
    verifier_question = ""
    generator_stage_started_at = perf_counter()
    verifier_stage_started_at = perf_counter()
    av_input_stage_started_at = perf_counter()
    v_constraint_generator_started_at = perf_counter()
    v_constraint_verifier_started_at = perf_counter()
    if _is_arrival_state(route_context_dict):
        decision = _arrival_visual_prompt_decision(
            route_context=route_context_dict,
            image_width=image_width,
            image_height=image_height,
        )
        _mark_stage("A_build_inputs", av_input_stage_started_at, status="skipped")
        _mark_stage("A_generator", generator_stage_started_at, status="skipped")
        _mark_stage("V_constraint_generator", v_constraint_generator_started_at, status="skipped")
        _mark_stage("V_verifier", verifier_stage_started_at, status="skipped")
        _mark_stage("V_constraint_verifier", v_constraint_verifier_started_at, status="skipped")
        _mark_stage("generator_llm", generator_stage_started_at, status="skipped")
        _mark_stage("verifier_llm", verifier_stage_started_at, status="skipped")
    else:
        try:
            av_inputs = _build_inputs_for_A(
                route_context=route_context_dict,
                image_width=llm_image_width,
                image_height=llm_image_height,
                navigation_mode=navigation_mode,
                fpv_segmentation_summary=llm_segmentation_summary,
                som_summary=llm_som_summary,
                som_candidate_hints=llm_som_candidate_hints,
                som_overlay_legend=som_overlay_legend,
                bev_map_overlay_legend=bev_map_overlay_legend,
                som_spatial_constraints=llm_som_spatial_constraints,
            )
            question = str(av_inputs.get("question") or "")
            system_prompt = str(av_inputs.get("system_prompt") or "")
            _mark_stage("A_build_inputs", av_input_stage_started_at, status="ok")
            llm_raw, llm_decision = await _run_A_generator(
                image_path=None,
                image_url=llm_image_url,
                question=question,
                system_prompt=system_prompt,
                llm_detail=llm_detail,
                extra_image_urls=extra_image_urls,
                route_context=route_context_dict,
                image_width=llm_image_width,
                image_height=llm_image_height,
            )
            generator_source_decision = _scale_decision_to_source_coordinates(
                llm_decision,
                scale_x=float(llm_to_source_scale["x"]),
                scale_y=float(llm_to_source_scale["y"]),
                source_image_size=[image_width, image_height],
                llm_image_size=[llm_image_width, llm_image_height],
            )
            decision = generator_source_decision
            _mark_stage("A_generator", generator_stage_started_at, status="ok")
            _mark_stage("generator_llm", generator_stage_started_at, status="ok")
            decision, segmentation_validation = _run_V_constraint(
                decision=decision,
                route_context=route_context_dict,
                segmentation_context=segmentation_context,
                constraint_stage="generator",
            )
            _mark_stage("V_constraint_generator", v_constraint_generator_started_at, status="ok")

            verifier_rendered_prompt = (
                llm_decision.rendered_prompt
                if llm_decision is not None and isinstance(llm_decision.rendered_prompt, dict)
                else None
            )
            generator_was_segmentation_rejected = bool(
                decision is not None
                and decision.reject_all
                and str(decision.arbiter_source) == "fpv_segmentation_reject"
                and isinstance(verifier_rendered_prompt, dict)
            )
            should_run_verifier = bool(
                enable_verifier
                and isinstance(verifier_rendered_prompt, dict)
                and (
                    (decision is not None and not decision.reject_all and isinstance(decision.rendered_prompt, dict))
                    or generator_was_segmentation_rejected
                )
            )
            if should_run_verifier:
                try:
                    llm_segmentation_validation = (
                        _scale_context_to_llm(
                            segmentation_validation,
                            scale_x=float(source_to_llm_scale["x"]),
                            scale_y=float(source_to_llm_scale["y"]),
                        )
                        if isinstance(segmentation_validation, dict)
                        else None
                    )
                    verifier_raw, verifier_question, verified_decision = await _run_V_verifier(
                        image_path=None,
                        image_url=llm_image_url,
                        route_context=route_context_dict,
                        rendered_prompt=verifier_rendered_prompt,
                        image_width=llm_image_width,
                        image_height=llm_image_height,
                        navigation_mode=navigation_mode,
                        fpv_segmentation_summary=llm_segmentation_summary,
                        som_summary=llm_som_summary,
                        som_candidate_hints=llm_som_candidate_hints,
                        som_overlay_legend=som_overlay_legend,
                        bev_map_overlay_legend=bev_map_overlay_legend,
                        som_spatial_constraints=llm_som_spatial_constraints,
                        segmentation_validation=llm_segmentation_validation,
                        alignment_report=av_inputs.get("alignment_report") if isinstance(av_inputs.get("alignment_report"), dict) else None,
                        overlay_legend=av_inputs.get("overlay_legend") if isinstance(av_inputs.get("overlay_legend"), dict) else None,
                        summary_source=str(av_inputs.get("summary_source") or "from_same_mask_entity"),
                        llm_detail=llm_detail,
                        extra_image_urls=extra_image_urls,
                    )
                    generator_decision = generator_source_decision
                    constrained_generator_decision = decision
                    verified_decision = _scale_decision_to_source_coordinates(
                        verified_decision,
                        scale_x=float(llm_to_source_scale["x"]),
                        scale_y=float(llm_to_source_scale["y"]),
                        source_image_size=[image_width, image_height],
                        llm_image_size=[llm_image_width, llm_image_height],
                    )
                    _mark_stage("V_verifier", verifier_stage_started_at, status="ok")
                    _mark_stage("verifier_llm", verifier_stage_started_at, status="ok")
                    verified_decision, verifier_segmentation_validation = _run_V_constraint(
                        decision=verified_decision,
                        route_context=route_context_dict,
                        segmentation_context=segmentation_context,
                        constraint_stage="verifier",
                    )
                    _mark_stage("V_constraint_verifier", v_constraint_verifier_started_at, status="ok")
                    verifier_debug = {
                        "generator_raw": llm_raw,
                        "generator_decision": visual_prompt_decision_to_dict(generator_decision),
                        "generator_rejected_decision": visual_prompt_decision_to_dict(decision) if generator_was_segmentation_rejected and decision is not None else None,
                        "verifier_raw": verifier_raw,
                        "verifier_decision": visual_prompt_decision_to_dict(verified_decision) if verified_decision is not None else None,
                        "repair_from_segmentation_reject": generator_was_segmentation_rejected,
                        "verifier_result_overrides_generator": verified_decision is not None,
                        "av_alignment_report": av_inputs.get("alignment_report"),
                        "av_summary_source": av_inputs.get("summary_source"),
                        "som_summary": som_summary,
                        "som_candidate_hints": som_candidate_hints,
                        "som_overlay_legend": som_overlay_legend,
                        "som_spatial_constraints": som_spatial_constraints,
                    }
                    if verified_decision is not None:
                        segmentation_validation = verifier_segmentation_validation
                        verified_decision.arbiter_source = (
                            "llm_direct_repaired" if generator_was_segmentation_rejected else "llm_direct_verified"
                        )
                        verified_decision.debug = {
                            **dict(verified_decision.debug or {}),
                            **verifier_debug,
                        }
                        decision = verified_decision
                except Exception as exc:
                    verifier_error = str(exc)
                    _mark_stage("V_verifier", verifier_stage_started_at, status="error", error_code="LLM_VERIFIER_FAILED")
                    _mark_stage(
                        "V_constraint_verifier",
                        v_constraint_verifier_started_at,
                        status="skipped",
                    )
                    _mark_stage(
                        "verifier_llm",
                        verifier_stage_started_at,
                        status="error",
                        error_code="LLM_VERIFIER_FAILED",
                    )
            else:
                _mark_stage("V_verifier", verifier_stage_started_at, status="skipped")
                _mark_stage("V_constraint_verifier", v_constraint_verifier_started_at, status="skipped")
                _mark_stage("verifier_llm", verifier_stage_started_at, status="skipped")
        except Exception as exc:
            llm_error = str(exc)
            _mark_stage("A_build_inputs", av_input_stage_started_at, status="error", error_code="LLM_GENERATOR_FAILED")
            _mark_stage("A_generator", generator_stage_started_at, status="error", error_code="LLM_GENERATOR_FAILED")
            _mark_stage("V_constraint_generator", v_constraint_generator_started_at, status="skipped")
            _mark_stage("V_verifier", verifier_stage_started_at, status="skipped")
            _mark_stage("V_constraint_verifier", v_constraint_verifier_started_at, status="skipped")
            _mark_stage(
                "generator_llm",
                generator_stage_started_at,
                status="error",
                error_code="LLM_GENERATOR_FAILED",
            )
            _mark_stage("verifier_llm", verifier_stage_started_at, status="skipped")

    if (
        enable_verifier
        and decision is not None
        and not decision.reject_all
        and isinstance(decision.rendered_prompt, dict)
        and verifier_raw is None
        and not verifier_error
        and question
    ):
        verifier_error = "visual prompt verifier did not run"

    if decision is None:
        decision = _reject_visual_prompt_decision(
            reason=(
                llm_error
                or "The RouteContext-driven LVLM agent did not return a valid rendered prompt."
            ),
            arbiter_source="llm_route_context_reject",
            route_context=route_context_dict,
        )

    decision_dict = visual_prompt_decision_to_dict(decision)
    decision_dict.setdefault("debug", {})
    if llm_error:
        decision_dict["debug"]["llm_error"] = llm_error
    if verifier_error:
        decision_dict["debug"]["verifier_error"] = verifier_error
    if segmentation_error:
        decision_dict["debug"]["fpv_segmentation_error"] = segmentation_error
    if segmentation_error_code:
        decision_dict["debug"]["fpv_segmentation_error_code"] = segmentation_error_code
    if segmentation_summary:
        decision_dict["debug"]["fpv_segmentation_summary"] = segmentation_summary
    if bev_map_error:
        decision_dict["debug"]["bev_map_provider_error"] = bev_map_error
    if som_summary:
        decision_dict["debug"]["som_summary"] = som_summary
    if som_candidate_hints:
        decision_dict["debug"]["som_candidate_hints"] = som_candidate_hints
    if som_overlay_legend:
        decision_dict["debug"]["som_overlay_legend"] = som_overlay_legend
    if bev_map_overlay_legend:
        decision_dict["debug"]["bev_map_overlay_legend"] = bev_map_overlay_legend
    if som_spatial_constraints:
        decision_dict["debug"]["som_spatial_constraints"] = som_spatial_constraints
    if safe_corridor_prior:
        decision_dict["debug"]["safe_corridor_prior"] = safe_corridor_prior
    if segmentation_teacher_bundle:
        decision_dict["debug"]["segmentation_teacher_bundle"] = segmentation_teacher_bundle
    if sam3_result:
        decision_dict["debug"]["sam3_result"] = sam3_result
    if normalized_sam3:
        decision_dict["debug"]["normalized_sam3"] = normalized_sam3
    if not enable_verifier:
        decision_dict["debug"]["verifier_skipped"] = True
    decision_dict["debug"]["multi_image_arbiter"] = {
        **multi_image_debug,
        "extra_image_count": len(extra_image_urls),
    }
    # Export the exact LVLM prompt payload for audit/debug. This is offline/debug-only
    # and can be large, but it avoids guessing whether the right constraints were sent.
    try:
        question_json = json.loads(question) if question else None
    except Exception:
        question_json = None
    decision_dict["debug"]["arbiter_prompt"] = {
        "generator": {
            "system_prompt": system_prompt,
            "question": question,
            "question_json": question_json,
            "llm_image": {
                key: value
                for key, value in llm_image_payload.items()
                if key != "image_data_url"
            },
        },
        "verifier": {
            "question": verifier_question,
        },
    }

    qa_stage_started_at = perf_counter()
    if emit_qa_bundle:
        candidate_bundle = build_pov_prompt_candidate_bundle(
            route_context=route_context_dict,
            local_bev_context=local_bev_dict,
            image_width=image_width,
            image_height=image_height,
            segmentation_context=segmentation_context,
            rendered_prompt=decision.rendered_prompt,
        )
        qa_bundle = build_visual_prompt_qa_bundle(
            route_context=route_context_dict,
            local_bev_context=local_bev_dict,
            image_width=image_width,
            image_height=image_height,
            fpv_segmentation_summary=segmentation_summary,
            segmentation_validation=segmentation_validation,
            rendered_prompt=decision.rendered_prompt,
            arbiter_decision=decision_dict,
            candidate_bundle=candidate_bundle,
            som_summary=som_summary,
            safe_corridor_prior=safe_corridor_prior,
        )
        decision_dict["debug"]["qa_bundle_summary"] = {
            "schema_version": qa_bundle.get("schema_version"),
            "task_counts": qa_bundle.get("task_counts", {}),
            "family_counts": qa_bundle.get("family_counts", {}),
        }
        _mark_stage("qa_bundle", qa_stage_started_at, status="ok")
    else:
        _mark_stage("qa_bundle", qa_stage_started_at, status="skipped")

    render_stage_started_at = perf_counter()
    image_href = _image_href_for_overlay(image_path=image_path, image_url=image_url)
    overlay_svg = _render_prompt_overlay_svg(
        image_href=image_href,
        image_width=image_width,
        image_height=image_height,
        rendered_prompt=decision.rendered_prompt,
        title="Mode-Aware Visual Prompt",
    )

    generator_overlay_svg: str | None = None
    generator_overlay_svg_data_url: str | None = None
    if generator_decision is not None and isinstance(generator_decision.rendered_prompt, dict):
        generator_overlay_svg = _render_prompt_overlay_svg(
            image_href=image_href,
            image_width=image_width,
            image_height=image_height,
            rendered_prompt=generator_decision.rendered_prompt,
            title="Generator Visual Prompt (Pre-Verifier)",
        )
        generator_overlay_svg_data_url = _svg_to_data_url(generator_overlay_svg)
    _mark_stage("render", render_stage_started_at, status="ok")

    telemetry["elapsed_ms"] = int(round((perf_counter() - started_at) * 1000.0))
    decision_dict["debug"]["telemetry"] = telemetry

    return {
        "request_id": resolved_request_id,
        "route_context": route_context_dict,
        "local_bev_context": local_bev_dict,
        "arbiter_decision": decision_dict,
        "overlay_svg": overlay_svg,
        "overlay_svg_data_url": _svg_to_data_url(overlay_svg),
        "generator_overlay_svg": generator_overlay_svg,
        "generator_overlay_svg_data_url": generator_overlay_svg_data_url,
        "sam3": sam3_result,
        "fpv_segmentation_summary": segmentation_summary,
        "som_summary": som_summary,
        "som_candidate_hints": som_candidate_hints,
        "som_spatial_constraints": som_spatial_constraints,
        "safe_corridor_prior": safe_corridor_prior,
        "som_overlay_image_data_url": (
            som_overlay_payload.get("som_overlay_image_data_url") if isinstance(som_overlay_payload, dict) else None
        ),
        "som_collage_image_data_url": (
            som_overlay_payload.get("som_collage_image_data_url") if isinstance(som_overlay_payload, dict) else None
        ),
        "som_overlay_legend": som_overlay_legend,
        "bev_map_overlay_image_data_url": (
            bev_map_overlay_payload.get("bev_map_overlay_image_data_url")
            if isinstance(bev_map_overlay_payload, dict)
            else None
        ),
        "bev_map_overlay_legend": bev_map_overlay_legend,
        "llm_image": {
            key: value
            for key, value in llm_image_payload.items()
            if key != "image_data_url"
        },
        "llm_input_images": llm_input_images if emit_llm_input_images else None,
        "qa_bundle": qa_bundle,
        "pov_prompt_candidate_bundle": pov_candidate_bundle_to_dict(candidate_bundle) if emit_qa_bundle else None,
        "telemetry": telemetry,
    }
