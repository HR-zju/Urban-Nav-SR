from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


from mcp_server.visualization.frame_io import _write_data_url_image


class SidewalkPipelineArtifactsTest(unittest.TestCase):
    def test_llm_input_image_writer_preserves_role_suffix(self) -> None:
        data_url = "data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg").decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "000001.primary_fpv"
            output_path = _write_data_url_image(data_url, base)

            self.assertEqual(output_path, Path(tmp) / "000001.primary_fpv.jpg")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
