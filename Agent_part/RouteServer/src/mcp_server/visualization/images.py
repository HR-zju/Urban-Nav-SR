"""Shared image encoding for map and diagnostic rendering."""
from __future__ import annotations
import base64
import io
from typing import Optional
from pathlib import Path
import os
import mimetypes
try:
    from PIL import Image, ImageOps
except ImportError:
    Image = ImageOps = None


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


def _image_href_for_overlay(*, image_path: Optional[str], image_url: Optional[str]) -> str:
    if str(image_path or "").strip():
        return local_image_path_to_data_url(str(image_path))
    return str(image_url or "").strip()



def local_image_path_to_data_url(image_path: str) -> str:
    """Encode a local image using the existing size and JPEG transport limits.

    Args:
        image_path: Local image filename, with optional user-home expansion.
    Returns:
        Image data URL, retaining the original bytes when optimization fails.
    Raises:
        FileNotFoundError: The image does not exist.
        ValueError: The path does not identify a regular file.
        OSError: Image bytes cannot be read.
    """
    path = Path(str(image_path or "").strip()).expanduser()
    if not str(path):
        raise ValueError("image_path is required")
    if not path.exists():
        raise FileNotFoundError(f"image_path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"image_path is not a file: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    raw_bytes = path.read_bytes()

    max_side_raw = os.getenv("LLM_VISION_IMAGE_MAX_SIDE", "").strip()
    max_bytes_raw = os.getenv("LLM_VISION_IMAGE_MAX_BYTES", "").strip()
    jpeg_quality_raw = os.getenv("LLM_VISION_IMAGE_JPEG_QUALITY", "").strip()
    try:
        max_side = max(512, int(max_side_raw)) if max_side_raw else 1600
    except ValueError:
        max_side = 1600
    try:
        max_bytes = max(256_000, int(max_bytes_raw)) if max_bytes_raw else 2_500_000
    except ValueError:
        max_bytes = 2_500_000
    try:
        jpeg_quality = min(max(int(jpeg_quality_raw), 40), 95) if jpeg_quality_raw else 85
    except ValueError:
        jpeg_quality = 85

    should_optimize = len(raw_bytes) > max_bytes
    optimized_bytes = raw_bytes
    optimized_mime = mime_type

    if Image is not None:
        try:
            image = Image.open(path)
            if ImageOps is not None:
                image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            width, height = image.size
            longest = max(width, height, 1)
            resize_scale = min(1.0, float(max_side) / float(longest))
            if resize_scale < 0.999:
                image = image.resize(
                    (
                        max(1, int(round(width * resize_scale))),
                        max(1, int(round(height * resize_scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
                should_optimize = True
            if should_optimize:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
                optimized_bytes = buffer.getvalue()
                optimized_mime = "image/jpeg"
        except Exception:
            optimized_bytes = raw_bytes
            optimized_mime = mime_type

    encoded = base64.b64encode(optimized_bytes).decode("ascii")
    return f"data:{optimized_mime};base64,{encoded}"
