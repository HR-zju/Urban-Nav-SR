"""Verify one-pass ordinal selection and visible truncated-video failures."""

import cv2
import numpy as np
import pytest

from mcp_server.data.history_media import extract_history_frames


def test_deduplicated_media_preserves_ordinals_and_reports_truncation(tmp_path):
    """A real local video yields stable unique JPEGs and explicit decode errors."""
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    try:
        for ordinal in range(6):
            writer.write(np.full((32, 32, 3), ordinal * 40, dtype=np.uint8))
    finally:
        writer.release()
    paths = extract_history_frames(video, [5, 1, 1, 3], tmp_path / "frames", jpeg_quality=95)
    assert list(paths) == [1, 3, 5]
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 3
    for ordinal, path in paths.items():
        assert path.name == f"{ordinal:06d}.jpg"
        assert abs(float(cv2.imread(str(path)).mean()) - ordinal * 40) < 3
    with pytest.raises(OSError, match="decode failed at ordinal 6"):
        extract_history_frames(video, [7], tmp_path / "truncated", jpeg_quality=95)
