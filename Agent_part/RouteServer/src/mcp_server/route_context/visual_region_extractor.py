from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .schema import VisualRegion, VisualRegionBundle


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _load_image(
    *,
    image_path: str | None = None,
    image_url: str | None = None,
) -> Image.Image:
    if image_path:
        return Image.open(Path(image_path).expanduser()).convert("RGB")

    raw = str(image_url or "").strip()
    if raw.startswith("data:image/") and "," in raw:
        _, encoded = raw.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    raise ValueError("VisualRegionExtractor currently supports image_path or data:image URL inputs")


def _resize_for_analysis(image: Image.Image, max_width: int = 512) -> tuple[Image.Image, float]:
    width, height = image.size
    if width <= max_width:
        return image, 1.0
    scale = max_width / max(float(width), 1.0)
    resized = image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.BILINEAR)
    return resized, scale


def _compute_navigability(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    grad = np.sqrt(gx * gx + gy * gy)
    grad = grad / max(float(np.percentile(grad, 99) or 1.0), 1e-6)
    grad = np.clip(grad, 0.0, 1.0)

    h, w = gray.shape
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]

    bottom_bias = np.clip((ys - 0.25) / 0.75, 0.0, 1.0)
    center_bias = 1.0 - np.clip(np.abs(xs), 0.0, 1.0)
    sky_penalty = np.clip((b - np.maximum(r, g)) * 2.5, 0.0, 1.0) * (1.0 - bottom_bias)

    score = (
        0.42 * gray
        + 0.28 * (1.0 - grad)
        + 0.20 * bottom_bias
        + 0.10 * center_bias
        - 0.30 * sky_penalty
    )
    return np.clip(score, 0.0, 1.0)


def _heatmap_grid(score: np.ndarray, rows: int = 18, cols: int = 32) -> list[list[float]]:
    h, w = score.shape
    values: list[list[float]] = []
    for row in range(rows):
        y0 = int(round(row * h / rows))
        y1 = int(round((row + 1) * h / rows))
        line: list[float] = []
        for col in range(cols):
            x0 = int(round(col * w / cols))
            x1 = int(round((col + 1) * w / cols))
            cell = score[y0:y1 or None, x0:x1 or None]
            line.append(round(float(cell.mean()) if cell.size else 0.0, 4))
        values.append(line)
    return values


def _make_region(
    *,
    region_id: str,
    region_type: str,
    bbox_uv: list[float],
    polygon_uv: list[list[float]],
    score: float,
    source: str,
    debug: dict[str, Any],
) -> VisualRegion:
    return VisualRegion(
        region_id=region_id,
        region_type=region_type,
        frame="image_plane",
        bbox_uv=[round(float(v), 2) for v in bbox_uv],
        polygon_uv=[[round(float(x), 2), round(float(y), 2)] for x, y in polygon_uv],
        score=round(_clamp(score, 0.0, 1.0), 4),
        mask=None,
        source=source,
        debug=debug,
    )


def extract_visual_regions(
    *,
    image_path: str | None = None,
    image_url: str | None = None,
    top_k: int = 5,
) -> VisualRegionBundle:
    image = _load_image(image_path=image_path, image_url=image_url)
    resized, scale = _resize_for_analysis(image)
    rgb = np.asarray(resized.convert("RGB"))
    score_map = _compute_navigability(rgb)

    height, width = resized.size[1], resized.size[0]
    candidate_specs = [
        ("left_front", 0.06, 0.43, 0.36, 0.92),
        ("front", 0.28, 0.72, 0.28, 0.90),
        ("right_front", 0.57, 0.94, 0.36, 0.92),
        ("center_far", 0.32, 0.68, 0.18, 0.58),
        ("left_far", 0.08, 0.38, 0.20, 0.60),
        ("right_far", 0.62, 0.92, 0.20, 0.60),
    ]

    regions: list[VisualRegion] = []
    for region_id, x0_r, x1_r, y0_r, y1_r in candidate_specs:
        x0 = max(0, min(width - 1, int(round(x0_r * width))))
        x1 = max(x0 + 1, min(width, int(round(x1_r * width))))
        y0 = max(0, min(height - 1, int(round(y0_r * height))))
        y1 = max(y0 + 1, min(height, int(round(y1_r * height))))
        region = score_map[y0:y1, x0:x1]
        mean_score = float(region.mean()) if region.size else 0.0
        polygon = [[x0, y1], [x0 + (x1 - x0) * 0.18, y0], [x1 - (x1 - x0) * 0.18, y0], [x1, y1]]
        regions.append(
            _make_region(
                region_id=f"vr_{region_id}",
                region_type="navigable_region",
                bbox_uv=[x0 / scale, y0 / scale, x1 / scale, y1 / scale],
                polygon_uv=[[px / scale, py / scale] for px, py in polygon],
                score=mean_score,
                source="heuristic_navigability",
                debug={
                    "analysis_scale": round(scale, 4),
                    "region_id": region_id,
                    "mean_score": round(mean_score, 4),
                },
            )
        )

    regions.sort(key=lambda item: item.score, reverse=True)
    return VisualRegionBundle(
        image_width=image.size[0],
        image_height=image.size[1],
        navigability_heatmap=_heatmap_grid(score_map),
        candidate_regions=regions[: max(1, int(top_k))],
    )
