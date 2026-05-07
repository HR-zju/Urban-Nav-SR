from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_sidewalk_images_visual_prompt_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("run_sidewalk_images_visual_prompt_pipeline", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_write_data_url_image = _MODULE._write_data_url_image


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
