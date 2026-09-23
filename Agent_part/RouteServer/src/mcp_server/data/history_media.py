"""Materialize deduplicated observation frames using source video ordinals."""

from pathlib import Path
from typing import Iterable

import cv2


class MediaDecodeError(OSError):
    """A source video cannot be opened or decoded into requested frames."""


def extract_history_frames(
    video: Path, ordinals: Iterable[int], output: Path, *, jpeg_quality: int
) -> dict[int, Path]:
    """Decode a clip once and write only requested, unique frames at source size.

    Args:
        video: Local source video; its playback FPS does not define source time.
        ordinals: Zero-based frame ordinals from the observation plan.
        output: New clip-specific directory for extracted JPEGs.
        jpeg_quality: Explicit OpenCV JPEG quality, from 0 through 100.
    Returns:
        Sorted mapping from video ordinal to the written JPEG path.
    Raises:
        ValueError: If ordinals or JPEG quality are invalid.
        FileExistsError: If output already exists; completed shards are reused by
            the caller using their completion receipt, rather than overwritten.
        MediaDecodeError: If video opening or decoding fails. Partial output
            remains available for diagnosis and has no completion receipt.
        OSError: If an output JPEG cannot be written.
    """
    requested = sorted(set(ordinals))
    if requested and requested[0] < 0:
        raise ValueError("video ordinals must be nonnegative")
    if not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 0 and 100")
    if not requested:
        return {}

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise MediaDecodeError(f"{video}: unable to open video")
        output.mkdir(parents=True, exist_ok=False)
        paths = {}
        next_requested = 0
        # 顺序解码保证 ordinal 身份；容器 PTS/FPS 不能替代采集时间，也不用随机 seek。
        for ordinal in range(requested[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                raise MediaDecodeError(f"{video}: decode failed at ordinal {ordinal}")
            if ordinal == requested[next_requested]:
                path = output / f"{ordinal:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise OSError(f"{video}: JPEG write failed at ordinal {ordinal}: {path}")
                paths[ordinal] = path
                next_requested += 1
        return paths
    finally:
        capture.release()
