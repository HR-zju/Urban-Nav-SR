#!/usr/bin/env python3
"""Render a 3-panel inspection video for the sidewalk visual-prompt pipeline.

The current RouteServer artifacts contain multiple visual inputs per frame:

    FPV SoM/SAM numbered overlay
    final visual-prompt overlay SVG
    OSM BEV map crop

This script composes those views into a stable 3x1 MP4 layout:

    [SoM/SAM] [visual prompt on FPV] [BEV map]

The center panel is always FPV with the visual prompt overlaid. It prefers the
pipeline's generated overlay SVG, then falls back to drawing the final
``rendered_prompt`` from JSON on top of the frame image.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class FrameItem:
    frame_name: str
    image_path: Path | None
    json_path: Path
    fpv_path: Path | None
    som_path: Path | None
    bev_path: Path | None
    overlay_path: Path | None
    sam_vis_path: Path | None


@dataclass(frozen=True)
class PanelSpec:
    title: str
    image: np.ndarray | None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _load_manifest_frames(run_dir: Path) -> list[FrameItem]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"manifest.json not found under: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames") or []
    if not isinstance(frames, list):
        raise RuntimeError("manifest.json frames must be a list")

    out: list[FrameItem] = []
    for item in frames:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ok":
            continue
        frame_name = str(item.get("frame_name") or "").strip()
        if not frame_name:
            continue
        image_path = _resolve_run_artifact(
            run_dir=run_dir,
            manifest_path=item.get("image_path"),
            relative_dir="../images",
            fallback_name=f"{frame_name}.jpg",
        )
        json_path = _resolve_run_artifact(
            run_dir=run_dir,
            manifest_path=item.get("json_path"),
            relative_dir="json",
            fallback_name=f"{frame_name}.json",
        )
        if json_path is None:
            raise RuntimeError(
                f"Missing per-frame json for {frame_name}. Checked current run_dir and manifest json_path."
            )
        overlay_path = _resolve_run_artifact(
            run_dir=run_dir,
            manifest_path=item.get("overlay_path"),
            relative_dir="overlay",
            fallback_name=f"{frame_name}.svg",
        )
        llm_input_images = item.get("llm_input_images") if isinstance(item.get("llm_input_images"), list) else []
        fpv_path = _find_llm_input_image(
            llm_input_images,
            ".primary_fpv.",
            run_dir=run_dir,
            fallback=image_path,
        )
        som_path = _find_llm_input_image(llm_input_images, ".numbered_som_overlay.", run_dir=run_dir)
        bev_path = _find_llm_input_image(llm_input_images, ".bev_map_overlay.", run_dir=run_dir)
        sam_vis = _find_optional_path(run_dir / "seg_vis" / f"{frame_name}_visualization.jpg")
        out.append(
            FrameItem(
                frame_name=frame_name,
                image_path=image_path,
                json_path=json_path,
                fpv_path=fpv_path,
                som_path=som_path,
                bev_path=bev_path,
                overlay_path=overlay_path,
                sam_vis_path=sam_vis,
            )
        )
    if not out:
        raise RuntimeError(f"No ok frames found in manifest: {manifest_path}")
    return out


def _find_optional_path(path: Path) -> Path | None:
    path = path.expanduser()
    return path if path.exists() else None


def _resolve_run_artifact(
    *,
    run_dir: Path,
    manifest_path: Any,
    relative_dir: str,
    fallback_name: str | None = None,
) -> Path | None:
    """Resolve artifacts after a run directory has been copied or moved.

    Older manifests store absolute paths from the original machine/tmp
    location. For video rendering we prefer the current --run-dir copy whenever
    the basename exists there, then fall back to the manifest path.
    """
    manifest_candidate = Path(str(manifest_path or "")).expanduser()
    candidate_name = fallback_name or manifest_candidate.name
    if candidate_name:
        local_candidate = (run_dir / relative_dir / candidate_name).resolve()
        if local_candidate.exists():
            return local_candidate
    if manifest_candidate.exists():
        return manifest_candidate
    return None


def _find_llm_input_image(
    paths: list[Any],
    suffix_token: str,
    *,
    run_dir: Path,
    fallback: Path | None = None,
) -> Path | None:
    for value in paths:
        candidate = Path(str(value)).expanduser()
        if suffix_token not in candidate.name:
            continue
        local_candidate = run_dir / "llm_inputs" / candidate.name
        if local_candidate.exists():
            return local_candidate
        if candidate.exists():
            return candidate
    return fallback if fallback is not None and fallback.exists() else None


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return img


def _read_optional_bgr(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return _read_bgr(path)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return int(default)


def _letterbox(img: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    """Resize an image into a fixed panel while preserving aspect ratio."""
    width, height = int(size[0]), int(size[1])
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (12, 12, 12)
    if img is None:
        cv2.putText(
            canvas,
            "missing",
            (max(width // 2 - 55, 8), height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (150, 150, 150),
            2,
            cv2.LINE_AA,
        )
        return canvas
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return canvas
    scale = min(float(width) / float(w), float(height) / float(h))
    resized_w = max(int(round(w * scale)), 1)
    resized_h = max(int(round(h * scale)), 1)
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x0 = (width - resized_w) // 2
    y0 = (height - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


def _parse_svg_viewbox(root: ET.Element) -> tuple[int, int]:
    raw = str(root.attrib.get("viewBox") or "").strip()
    parts = raw.replace(",", " ").split()
    if len(parts) == 4:
        width = max(int(round(_safe_float(parts[2], 1200.0))), 1)
        height = max(int(round(_safe_float(parts[3], 675.0))), 1)
        return width, height
    width = max(int(round(_safe_float(root.attrib.get("width"), 1200.0))), 1)
    height = max(int(round(_safe_float(root.attrib.get("height"), 675.0))), 1)
    return width, height


def _parse_color_rgba(value: str, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    value = str(value or "").strip()
    if value.startswith("#") and len(value) in {4, 7}:
        if len(value) == 4:
            r = int(value[1] * 2, 16)
            g = int(value[2] * 2, 16)
            b = int(value[3] * 2, 16)
        else:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
        return r, g, b, 255
    match = re.fullmatch(r"rgba\(([^,]+),([^,]+),([^,]+),([^)]+)\)", value)
    if match:
        r = int(round(_safe_float(match.group(1), 0.0)))
        g = int(round(_safe_float(match.group(2), 0.0)))
        b = int(round(_safe_float(match.group(3), 0.0)))
        alpha_value = _safe_float(match.group(4), 1.0)
        a = int(round(alpha_value * 255.0 if alpha_value <= 1.0 else alpha_value))
        return max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255)), max(0, min(a, 255))
    return default


def _parse_points(value: str) -> list[tuple[float, float]]:
    numbers = [
        _safe_float(match.group(0), 0.0)
        for match in re.finditer(r"-?\d+(?:\.\d+)?", str(value or ""))
    ]
    return [
        (float(numbers[index]), float(numbers[index + 1]))
        for index in range(0, len(numbers) - 1, 2)
    ]


def _stroke_width(element: ET.Element, default: float = 2.0) -> int:
    return max(int(round(_safe_float(element.attrib.get("stroke-width"), default))), 1)


def _resize_cover_pil(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = int(size[0]), int(size[1])
    scale = max(float(width) / float(img.width), float(height) / float(img.height))
    resized = img.resize((max(int(round(img.width * scale)), 1), max(int(round(img.height * scale)), 1)))
    left = max((resized.width - width) // 2, 0)
    top = max((resized.height - height) // 2, 0)
    return resized.crop((left, top, left + width, top + height))


def _load_font(font_size: float) -> ImageFont.ImageFont:
    size = max(int(round(float(font_size))), 8)
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_svg_layer(
    canvas: Image.Image,
    draw_fn: Any,
) -> None:
    """Draw one SVG primitive on a transparent layer, then alpha-composite it.

    Pillow's ImageDraw can overwrite destination alpha when drawing translucent
    RGBA directly on an RGBA image. RouteServer overlays use full-frame
    translucent rects, so direct drawing would make the FPV base almost
    invisible after RGB conversion. Drawing on a separate transparent layer
    preserves normal alpha compositing semantics.
    """
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer, "RGBA")
    draw_fn(layer_draw)
    canvas.alpha_composite(layer)


def _svg_overlay_to_bgr(path: Path | None) -> np.ndarray | None:
    """Rasterize the runner's simple overlay SVG into a BGR image.

    The saved RouteServer overlays are intentionally simple: a base64-embedded
    FPV image plus lightweight rect/text annotations. Supporting that subset is
    enough for the video writer and avoids a heavyweight SVG dependency.
    """
    if path is None or not path.exists():
        return None
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    canvas_size = _parse_svg_viewbox(root)
    canvas = Image.new("RGBA", canvas_size, (12, 12, 12, 255))

    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "image":
            href = element.attrib.get("href") or element.attrib.get("{http://www.w3.org/1999/xlink}href")
            if isinstance(href, str) and href.startswith("data:image/") and "base64," in href:
                raw = base64.b64decode(href.split("base64,", 1)[1])
                base = Image.open(io.BytesIO(raw)).convert("RGBA")
                canvas.paste(_resize_cover_pil(base, canvas_size), (0, 0))
        elif tag == "rect":
            x = _safe_float(element.attrib.get("x"), 0.0)
            y = _safe_float(element.attrib.get("y"), 0.0)
            width = _safe_float(element.attrib.get("width"), float(canvas_size[0]))
            height = _safe_float(element.attrib.get("height"), float(canvas_size[1]))
            fill = _parse_color_rgba(str(element.attrib.get("fill") or ""), (0, 0, 0, 0))
            _draw_svg_layer(
                canvas,
                lambda draw, rect=[x, y, x + width, y + height], color=fill: draw.rectangle(rect, fill=color),
            )
        elif tag == "polygon":
            points = _parse_points(str(element.attrib.get("points") or ""))
            if len(points) < 2:
                continue
            fill_value = str(element.attrib.get("fill") or "")
            stroke_value = str(element.attrib.get("stroke") or "")
            fill = None if fill_value == "none" else _parse_color_rgba(fill_value, (0, 0, 0, 0))
            stroke = None if not stroke_value or stroke_value == "none" else _parse_color_rgba(stroke_value, (255, 255, 255, 255))
            stroke_px = _stroke_width(element)
            def _draw_polygon(draw: ImageDraw.ImageDraw, pts=points, fill_color=fill, stroke_color=stroke, width_px=stroke_px) -> None:
                draw.polygon(pts, fill=fill_color)
                if stroke_color is not None:
                    draw.line(pts + [pts[0]], fill=stroke_color, width=width_px)
            _draw_svg_layer(canvas, _draw_polygon)
        elif tag == "polyline":
            points = _parse_points(str(element.attrib.get("points") or ""))
            if len(points) < 2:
                continue
            stroke = _parse_color_rgba(str(element.attrib.get("stroke") or ""), (255, 255, 255, 255))
            stroke_px = _stroke_width(element)
            _draw_svg_layer(
                canvas,
                lambda draw, pts=points, color=stroke, width_px=stroke_px: draw.line(
                    pts,
                    fill=color,
                    width=width_px,
                    joint="curve",
                ),
            )
        elif tag == "circle":
            cx = _safe_float(element.attrib.get("cx"), 0.0)
            cy = _safe_float(element.attrib.get("cy"), 0.0)
            r = max(_safe_float(element.attrib.get("r"), 1.0), 0.0)
            fill = _parse_color_rgba(str(element.attrib.get("fill") or ""), (255, 255, 255, 255))
            _draw_svg_layer(
                canvas,
                lambda draw, box=[cx - r, cy - r, cx + r, cy + r], color=fill: draw.ellipse(box, fill=color),
            )
        elif tag == "text":
            text = "".join(element.itertext())
            if not text:
                continue
            x = _safe_float(element.attrib.get("x"), 0.0)
            y = _safe_float(element.attrib.get("y"), 0.0)
            font = _load_font(_safe_float(element.attrib.get("font-size"), 18.0))
            fill = _parse_color_rgba(str(element.attrib.get("fill") or ""), (235, 235, 235, 255))
            anchor = "mm" if str(element.attrib.get("text-anchor") or "") == "middle" else "la"
            _draw_svg_layer(
                canvas,
                lambda draw, xy=(x, y), value=text, font_obj=font, color=fill, text_anchor=anchor: draw.text(
                    xy,
                    value,
                    font=font_obj,
                    fill=color,
                    anchor=text_anchor,
                ),
            )

    rgb = np.array(canvas.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_panel_title(panel: np.ndarray, title: str) -> np.ndarray:
    out = panel.copy()
    title_h = 34
    cv2.rectangle(out, (0, 0), (out.shape[1], title_h), (20, 20, 20), thickness=-1)
    cv2.putText(
        out,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return out


def _draw_polyline(img: np.ndarray, pts: list[tuple[int, int]], color_bgr: tuple[int, int, int], thickness: int) -> None:
    if len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [arr], isClosed=False, color=color_bgr, thickness=int(thickness), lineType=cv2.LINE_AA)


def _draw_prompt_overlay_on_fpv(rgb_bgr: np.ndarray, prompt: dict[str, Any] | None, alpha: float = 0.55) -> np.ndarray:
    """Fallback renderer for final path_corridor prompts on top of FPV RGB."""
    if not isinstance(prompt, dict) or str(prompt.get("type") or "") != "path_corridor":
        return rgb_bgr
    overlay = rgb_bgr.copy()
    samples = prompt.get("path_samples") if isinstance(prompt.get("path_samples"), list) else []
    centers: list[tuple[int, int]] = []
    lefts: list[tuple[int, int]] = []
    rights: list[tuple[int, int]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        center = sample.get("center_uv")
        left = sample.get("left_edge_uv")
        right = sample.get("right_edge_uv")
        if isinstance(center, list) and len(center) >= 2:
            centers.append((_safe_int(center[0]), _safe_int(center[1])))
        if isinstance(left, list) and len(left) >= 2:
            lefts.append((_safe_int(left[0]), _safe_int(left[1])))
        if isinstance(right, list) and len(right) >= 2:
            rights.append((_safe_int(right[0]), _safe_int(right[1])))
    if len(lefts) == len(rights) and len(lefts) >= 2:
        poly = np.array(lefts + list(reversed(rights)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [poly], color=(60, 180, 80))
    _draw_polyline(overlay, lefts, color_bgr=(80, 200, 80), thickness=5)
    _draw_polyline(overlay, rights, color_bgr=(80, 200, 80), thickness=5)
    _draw_polyline(overlay, centers, color_bgr=(0, 120, 255), thickness=8)
    for point in centers:
        cv2.circle(overlay, point, radius=8, color=(0, 120, 255), thickness=-1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, float(alpha), rgb_bgr, float(1.0 - alpha), 0.0)


def _compose_frame(
    *,
    frame_name: str,
    status_text: str,
    som_bgr: np.ndarray | None,
    bev_bgr: np.ndarray | None,
    prompt_bgr: np.ndarray | None,
    tile_size: tuple[int, int],
) -> np.ndarray:
    panels = [
        PanelSpec("FPV SoM / SAM", som_bgr),
        PanelSpec("Visual Prompt on FPV", prompt_bgr),
        PanelSpec("BEV OSM Map", bev_bgr),
    ]
    rendered_panels = [
        _draw_panel_title(_letterbox(panel.image, tile_size), panel.title)
        for panel in panels
    ]
    combined = np.hstack(rendered_panels)

    bar_h = 38
    bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)
    cv2.putText(
        bar,
        f"frame={frame_name}  {status_text}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, combined])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose SoM/SAM, final visual-prompt overlay, and BEV panels into a 3x1 mp4 video."
    )
    parser.add_argument("--run-dir", required=True, help="Output directory from run_*_visual_prompt_pipeline (contains manifest.json)")
    parser.add_argument("--output-video", required=True, help="Output mp4 path")
    parser.add_argument("--fps", type=float, default=5.0, help="Video FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames")
    parser.add_argument("--tile-width", type=int, default=640, help="Width of each 3x1 panel tile")
    parser.add_argument("--tile-height", type=int, default=360, help="Height of each 3x1 panel tile")
    return parser.parse_args()


def _frame_status_text(payload: dict[str, Any]) -> str:
    arb = payload.get("arbiter_decision") if isinstance(payload.get("arbiter_decision"), dict) else {}
    reject_all = arb.get("reject_all")
    source = str(arb.get("source") or "unknown")
    reason = str(arb.get("reason") or "").strip()
    status = "reject_all=true" if reject_all is True else "reject_all=false" if reject_all is False else "reject_all=unknown"
    if reason:
        return f"{status}  source={source}  reason={reason[:120]}"
    return f"{status}  source={source}"


def _render_video_frame(item: FrameItem, *, tile_size: tuple[int, int]) -> np.ndarray:
    payload = json.loads(item.json_path.read_text(encoding="utf-8"))
    som = _read_optional_bgr(item.som_path)
    if som is None:
        som = _read_optional_bgr(item.sam_vis_path)
    bev = _read_optional_bgr(item.bev_path)
    prompt_overlay = _svg_overlay_to_bgr(item.overlay_path)
    if prompt_overlay is None:
        fpv = _read_optional_bgr(item.fpv_path)
        if fpv is None:
            fpv = _read_optional_bgr(item.image_path)
        arbiter = payload.get("arbiter_decision") if isinstance(payload.get("arbiter_decision"), dict) else {}
        rendered_prompt = arbiter.get("rendered_prompt") if isinstance(arbiter.get("rendered_prompt"), dict) else None
        prompt_overlay = _draw_prompt_overlay_on_fpv(fpv, rendered_prompt) if fpv is not None else None
    return _compose_frame(
        frame_name=item.frame_name,
        status_text=_frame_status_text(payload),
        som_bgr=som,
        bev_bgr=bev,
        prompt_bgr=prompt_overlay,
        tile_size=tile_size,
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    frames = _load_manifest_frames(run_dir)
    if args.max_frames is not None:
        frames = frames[: max(int(args.max_frames), 0)]
    if not frames:
        raise RuntimeError("No frames selected for video rendering")

    tile_size = (max(int(args.tile_width), 64), max(int(args.tile_height), 64))
    first_frame = _render_video_frame(frames[0], tile_size=tile_size)

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, float(args.fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_video}")

    try:
        writer.write(first_frame)
        for item in frames[1:]:
            frame = _render_video_frame(item, tile_size=tile_size)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    print(f"[OK] Wrote video: {output_video}")


if __name__ == "__main__":
    main()
