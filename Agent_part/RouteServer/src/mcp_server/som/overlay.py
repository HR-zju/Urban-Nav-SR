"""Render numbered Segment-of-Mask overlays for LVLM grounding.

The overlay gives the LVLM compact visual anchors: preferred traversable
components, conditional connector components, and forbidden components are drawn
with distinct palettes and numeric labels. The source mask remains in image_uv;
the rendered overlay may be resized for the LVLM, so every label center and
bounding box is scaled into the overlay image coordinate space.

Coordinate scaling:

    u_llm = u_source * (W_llm / W_source)
    v_llm = v_source * (H_llm / H_source)

Alpha compositing:

    RGB_out = alpha * RGB_overlay + (1 - alpha) * RGB_image

The returned legend preserves both source-space and LVLM-space mappings so the
generator prompt, verifier prompt, and persisted debug artifacts can be audited
against the same numbered regions.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_PALETTE: tuple[tuple[int, int, int], ...] = (
    (34, 197, 94),
    (14, 165, 233),
    (249, 115, 22),
    (168, 85, 247),
    (236, 72, 153),
    (234, 179, 8),
    (20, 184, 166),
    (59, 130, 246),
)
_FORBIDDEN_PALETTE: tuple[tuple[int, int, int], ...] = (
    (239, 68, 68),
    (220, 38, 38),
    (185, 28, 28),
    (244, 63, 94),
)
_CONDITIONAL_PALETTE: tuple[tuple[int, int, int], ...] = (
    (245, 158, 11),
    (234, 179, 8),
    (251, 146, 60),
)


def _image_data_url(image: Image.Image, *, image_format: str = "JPEG", quality: int = 85) -> str:
    buffer = io.BytesIO()
    normalized_format = str(image_format or "JPEG").strip().upper()
    if normalized_format in {"JPG", "JPEG"}:
        image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), optimize=True)
        mime = "image/jpeg"
    else:
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _resize_for_llm(image: Image.Image, *, max_side: int) -> tuple[Image.Image, dict[str, Any]]:
    """Resize an overlay while returning exact source-to-LLM scale factors."""
    width, height = image.size
    safe_max_side = max(int(max_side or 0), 512)
    longest = max(width, height, 1)
    scale = min(1.0, float(safe_max_side) / float(longest))
    if scale < 0.999:
        resized = image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        resized = image.copy()
    llm_width, llm_height = resized.size
    return resized, {
        "source_image_size": [int(width), int(height)],
        "llm_image_size": [int(llm_width), int(llm_height)],
        "scale_x": round(float(llm_width) / float(max(width, 1)), 8),
        "scale_y": round(float(llm_height) / float(max(height, 1)), 8),
        "max_side": int(safe_max_side),
    }


def _scale_point(point: Any, *, scale_x: float, scale_y: float) -> list[float] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        return [round(float(point[0]) * scale_x, 2), round(float(point[1]) * scale_y, 2)]
    except Exception:
        return None


def _scale_bbox(bbox: Any, *, scale_x: float, scale_y: float) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        return [
            int(round(float(bbox[0]) * scale_x)),
            int(round(float(bbox[1]) * scale_y)),
            int(round(float(bbox[2]) * scale_x)),
            int(round(float(bbox[3]) * scale_y)),
        ]
    except Exception:
        return None


def _scale_label_mapping_item(item: dict[str, Any], *, scale_x: float, scale_y: float) -> dict[str, Any]:
    scaled = dict(item)
    center = _scale_point(item.get("center_uv"), scale_x=scale_x, scale_y=scale_y)
    bbox = _scale_bbox(item.get("bbox_uv"), scale_x=scale_x, scale_y=scale_y)
    if center is not None:
        scaled["center_uv"] = center
    if bbox is not None:
        scaled["bbox_uv"] = bbox
    return scaled


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    try:
        return draw.textbbox((0, 0), text, font=font)
    except Exception:  # pragma: no cover - Pillow compatibility fallback
        width, height = draw.textsize(text, font=font)
        return 0, 0, int(width), int(height)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    *,
    center: Any,
    label_text: str,
    width: int,
    height: int,
    role: str,
) -> None:
    try:
        cx = float(center[0])
        cy = float(center[1])
    except Exception:
        return
    label_font = _font(max(int(min(width, height) * 0.04), 18))
    radius = max(int(min(width, height) * 0.026), 15)
    bbox = _text_bbox(draw, label_text, label_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0 = int(round(cx - radius))
    y0 = int(round(cy - radius))
    x1 = int(round(cx + radius))
    y1 = int(round(cy + radius))
    if role == "forbidden":
        fill = (127, 29, 29, 235)
        outline = (254, 226, 226, 250)
    elif role == "conditional_traversable":
        fill = (146, 64, 14, 235)
        outline = (254, 243, 199, 250)
    else:
        fill = (15, 23, 42, 230)
        outline = (255, 255, 255, 245)
    draw.ellipse((x0, y0, x1, y1), fill=fill, outline=outline, width=3)
    draw.text(
        (int(round(cx - text_w / 2)), int(round(cy - text_h / 2 - bbox[1]))),
        label_text,
        fill=(255, 255, 255, 255),
        font=label_font,
    )


def _append_component_overlay(
    *,
    overlay_arr: np.ndarray,
    component_labels: np.ndarray,
    regions: list[dict[str, Any]],
    label_mapping: list[dict[str, Any]],
    start_label: int,
    palette: tuple[tuple[int, int, int], ...],
    role: str,
    alpha: int,
    max_regions: int,
) -> int:
    """Paint selected connected components and append their numeric legend rows."""
    next_label = int(start_label)
    for idx, region in enumerate(regions[: max(int(max_regions), 0)]):
        component_id = region.get("component_id")
        try:
            numeric_component_id = int(component_id)
        except Exception:
            continue
        region_mask = component_labels == numeric_component_id
        if not bool(region_mask.any()):
            continue
        color = palette[idx % len(palette)]
        overlay_arr[region_mask, 0] = color[0]
        overlay_arr[region_mask, 1] = color[1]
        overlay_arr[region_mask, 2] = color[2]
        overlay_arr[region_mask, 3] = int(alpha)
        label = int(region.get("label") or next_label)
        next_label = max(next_label, label + 1)
        label_mapping.append(
            {
                "label": label,
                "region_id": region.get("region_id"),
                "region_role": role,
                "center_uv": region.get("center_uv"),
                "bbox_uv": region.get("bbox_uv"),
                "side": region.get("side"),
                "route_intent_score": region.get("route_intent_score"),
                "dominant_class_name": region.get("dominant_class_name"),
            }
        )
    return next_label


def build_numbered_som_overlay_payload(
    *,
    image: Image.Image,
    som_context: dict[str, Any] | None,
    ranked_regions: list[dict[str, Any]] | None,
    max_regions: int = 8,
    llm_max_side: int = 1600,
    include_collage: bool = False,
    image_format: str = "JPEG",
    jpeg_quality: int = 85,
) -> dict[str, Any] | None:
    """Build the LVLM-ready numbered SoM overlay image and legend contract."""
    if not isinstance(som_context, dict):
        return None
    component_labels = som_context.get("_component_labels")
    conditional_component_labels = som_context.get("_conditional_component_labels")
    forbidden_component_labels = som_context.get("_forbidden_component_labels")
    if not isinstance(component_labels, np.ndarray) or component_labels.ndim != 2:
        return None
    regions = [region for region in list(ranked_regions or []) if isinstance(region, dict)]
    conditional_regions = [region for region in regions if str(region.get("region_role") or "") == "conditional_traversable"]
    regions = [region for region in regions if str(region.get("region_role") or "traversable") == "traversable"]
    forbidden_regions = [region for region in list(som_context.get("forbidden_regions") or []) if isinstance(region, dict)]
    if not regions and not forbidden_regions:
        return None

    base = image.convert("RGBA")
    width, height = base.size
    if component_labels.shape != (height, width):
        return None
    if isinstance(forbidden_component_labels, np.ndarray) and forbidden_component_labels.shape != (height, width):
        forbidden_component_labels = None
    if isinstance(conditional_component_labels, np.ndarray) and conditional_component_labels.shape != (height, width):
        conditional_component_labels = None

    overlay_arr = np.zeros((height, width, 4), dtype=np.uint8)
    label_mapping: list[dict[str, Any]] = []
    next_label = _append_component_overlay(
        overlay_arr=overlay_arr,
        component_labels=component_labels,
        regions=regions,
        label_mapping=label_mapping,
        start_label=1,
        palette=_PALETTE,
        role="traversable",
        alpha=92,
        max_regions=max_regions,
    )
    if isinstance(conditional_component_labels, np.ndarray):
        next_label = _append_component_overlay(
            overlay_arr=overlay_arr,
            component_labels=conditional_component_labels,
            regions=conditional_regions,
            label_mapping=label_mapping,
            start_label=next_label,
            palette=_CONDITIONAL_PALETTE,
            role="conditional_traversable",
            alpha=82,
            max_regions=max_regions,
        )
    if isinstance(forbidden_component_labels, np.ndarray):
        _append_component_overlay(
            overlay_arr=overlay_arr,
            component_labels=forbidden_component_labels,
            regions=forbidden_regions,
            label_mapping=label_mapping,
            start_label=next_label,
            palette=_FORBIDDEN_PALETTE,
            role="forbidden",
            alpha=118,
            max_regions=min(max_regions, 8),
        )
    if not label_mapping:
        return None

    overlay = Image.alpha_composite(base, Image.fromarray(overlay_arr, mode="RGBA"))
    draw = ImageDraw.Draw(overlay)
    for item in label_mapping:
        _draw_label(
            draw,
            center=item.get("center_uv"),
            label_text=str(int(item["label"])),
            width=width,
            height=height,
            role=str(item.get("region_role") or "traversable"),
        )

    llm_overlay, scale_info = _resize_for_llm(overlay.convert("RGB"), max_side=llm_max_side)
    scale_x = float(scale_info["scale_x"])
    scale_y = float(scale_info["scale_y"])
    llm_label_mapping = [
        _scale_label_mapping_item(item, scale_x=scale_x, scale_y=scale_y)
        for item in label_mapping
    ]

    collage_data_url = None
    if include_collage:
        llm_base, _ = _resize_for_llm(base.convert("RGB"), max_side=llm_max_side)
        llm_width, llm_height = llm_overlay.size
        gap = max(int(llm_width * 0.012), 8)
        collage = Image.new("RGB", (llm_width * 2 + gap, llm_height), (15, 23, 42))
        collage.paste(llm_base.convert("RGB"), (0, 0))
        collage.paste(llm_overlay.convert("RGB"), (llm_width + gap, 0))
        collage_data_url = _image_data_url(collage, image_format=image_format, quality=jpeg_quality)

    return {
        "som_overlay_image_data_url": _image_data_url(
            llm_overlay,
            image_format=image_format,
            quality=jpeg_quality,
        ),
        "som_collage_image_data_url": collage_data_url,
        "som_overlay_legend": {
            "image_role": "numbered_som_overlay",
            "coordinate_space": "llm_image_uv",
            "source_coordinate_space": "image_uv",
            "source_image_size": scale_info["source_image_size"],
            "llm_image_size": scale_info["llm_image_size"],
            "llm_image_scale": {
                "x": scale_info["scale_x"],
                "y": scale_info["scale_y"],
            },
            "label_mapping": llm_label_mapping,
            "source_label_mapping": label_mapping,
            "legend_semantics": {
                "traversable": "green/blue labels are selectable walking support regions",
                "conditional_traversable": "yellow/orange labels are usable connector surfaces with lower priority than sidewalk/crosswalk",
                "forbidden": "red labels are obstacle/non-traversable regions to avoid",
            },
        },
        "llm_image_scale": {
            "x": scale_info["scale_x"],
            "y": scale_info["scale_y"],
        },
        "source_image_size": scale_info["source_image_size"],
        "llm_image_size": scale_info["llm_image_size"],
    }
