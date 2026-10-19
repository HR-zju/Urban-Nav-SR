"""Protect the rendering boundary while retiring legacy policy execution."""
from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

from PIL import Image

from mcp_server.visualization.images import local_image_path_to_data_url


def test_rendering_imports_do_not_load_policy_services() -> None:
    """A fresh rendering process must not initialize LLM or legacy brain modules."""
    subprocess.run(
        [sys.executable, '-c', """
import sys
from mcp_server.visualization import images, map_overlay, prompt_overlay
for name in sys.modules:
    assert not name.startswith('mcp_server.brain'), name
    assert name not in {'mcp_server.llm', 'mcp_server.services.visual_prompt_service'}, name
"""],
        check=True,
    )


def test_local_image_encoding_preserves_small_original(tmp_path: Path, monkeypatch) -> None:
    """Moving the codec must preserve the existing small-image byte contract."""
    for key in ('LLM_VISION_IMAGE_MAX_SIDE', 'LLM_VISION_IMAGE_MAX_BYTES', 'LLM_VISION_IMAGE_JPEG_QUALITY'):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / 'frame.png'
    Image.new('RGB', (32, 32), 'red').save(path)
    header, payload = local_image_path_to_data_url(str(path)).split(',', 1)
    assert header == 'data:image/png;base64'
    assert base64.b64decode(payload) == path.read_bytes()
