from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "brain" / "fpv_segmentation.py"
_SPEC = importlib.util.spec_from_file_location("fpv_segmentation", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

validate_fpv_segmentation_alignment = _MODULE.validate_fpv_segmentation_alignment


class FpvSegmentationAlignmentTest(unittest.TestCase):
    def test_alignment_valid_when_mask_matches_image_shape(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy_path = root / "000010_segmentation.npy"
            mapping_path = root / "class_mapping.json"
            np.save(npy_path, np.zeros((10, 20), dtype=np.int32))
            mapping_path.write_text('{"class_to_idx": {"sidewalk": 1}}', encoding="utf-8")
            payload = {
                "segmentation_npy_path": str(npy_path),
                "class_mapping_path": str(mapping_path),
                "alignment_contract": {
                    "frame_identity": {
                        "sequence_name": "ride_a",
                        "camera_view": "camera_front",
                        "frame_name": "000010",
                    }
                },
            }
            report = validate_fpv_segmentation_alignment(
                payload,
                image_width=20,
                image_height=10,
                sequence_name="ride_a",
                camera_view="camera_front",
                frame_name="000010",
            )
            self.assertTrue(bool(report.get("valid")))
            self.assertEqual(report.get("reasons"), [])

    def test_alignment_invalid_on_shape_mismatch(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy_path = root / "000011_segmentation.npy"
            mapping_path = root / "class_mapping.json"
            np.save(npy_path, np.zeros((8, 12), dtype=np.int32))
            mapping_path.write_text('{"class_to_idx": {"sidewalk": 1}}', encoding="utf-8")
            payload = {
                "segmentation_npy_path": str(npy_path),
                "class_mapping_path": str(mapping_path),
            }
            report = validate_fpv_segmentation_alignment(
                payload,
                image_width=20,
                image_height=10,
            )
            self.assertFalse(bool(report.get("valid")))
            self.assertIn("mask_image_size_mismatch", list(report.get("reasons") or []))


if __name__ == "__main__":
    unittest.main()
