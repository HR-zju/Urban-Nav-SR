"""Check source-image ordinal preservation using real FFmpeg encoding."""

import importlib.util
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "tools/dataset_processing/process_realword_mcap.py"
SPEC = importlib.util.spec_from_file_location("mcap_video_encoding", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_numbered_images(root, count):
    root.mkdir()
    for index in range(count):
        image = np.zeros((48, 48, 3), dtype=np.uint8)
        for bit in range(9):
            row, col = divmod(bit, 3)
            image[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = (
                240 if index & (1 << bit) else 16
            )
        assert cv2.imwrite(str(root / f"{index + 1:06d}.jpg"), image)


def _read_numbers(video):
    capture = cv2.VideoCapture(str(video))
    numbers = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        value = 0
        for bit in range(9):
            row, col = divmod(bit, 3)
            if image[row * 16 + 8, col * 16 + 8].mean() > 128:
                value |= 1 << bit
        numbers.append(value)
    capture.release()
    return numbers


def test_encoding_preserves_all_439_ordinals(tmp_path):
    """Each source timestamp row must retain its image identity after encoding."""
    images = tmp_path / "images"
    _write_numbered_images(images, 439)
    output = tmp_path / "front.mp4"
    MODULE.encode_frame_sequence(images, output, 439, 20.0)
    assert _read_numbers(output) == list(range(439))


def test_legacy_concat_mapping(tmp_path):
    """Reproduce the old 20 Hz encoder to identify where its extra frame occurs."""
    images = tmp_path / "images"
    _write_numbered_images(images, 439)
    paths = sorted(images.glob("*.jpg"))
    listing = tmp_path / "images_list.txt"
    listing.write_text("".join(f"file '{p}'\nduration 0.05\n" for p in paths)
                       + f"file '{paths[-1]}'\n")
    output = tmp_path / "legacy.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c:v", "libx264", "-vf", "fps=20.0", "-pix_fmt", "yuv420p", str(output)],
                   check=True, capture_output=True)
    assert _read_numbers(output) == [*range(439), 438]


def test_source_count_mismatch_is_visible(tmp_path):
    """Fail before encoding when a source row has no corresponding JPEG."""
    images = tmp_path / "images"
    _write_numbered_images(images, 2)
    with pytest.raises(ValueError, match="2 images for 3 source rows"):
        MODULE.encode_frame_sequence(images, tmp_path / "front.mp4", 3, 20.0)
