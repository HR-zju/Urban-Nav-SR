from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image


def _load_module(name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CONTEXT = _load_module("som_context", "src/mcp_server/som/context.py")
_OVERLAY = _load_module("som_overlay", "src/mcp_server/som/overlay.py")


class SomOverlayTest(unittest.TestCase):
    def test_builds_numbered_overlay_payload_with_legend_mapping(self) -> None:
        mask = np.zeros((20, 24), dtype=np.int32)
        mask[10:20, 2:9] = 1
        mask[9:19, 14:23] = 1
        segmentation_context = {
            "mask": mask,
            "id_to_name": {0: "road", 1: "sidewalk"},
            "traversable_ids": {1},
            "forbidden_ids": {0},
        }
        som_context = _CONTEXT.build_som_context(segmentation_context, min_region_area_ratio=0.01)
        ranked = _CONTEXT.rank_som_regions_for_route(
            som_context,
            {"local_semantics": {"turn_type": "turn_right"}},
            top_k=2,
        )

        image = Image.new("RGB", (24, 20), (80, 80, 80))
        payload = _OVERLAY.build_numbered_som_overlay_payload(
            image=image,
            som_context=som_context,
            ranked_regions=ranked,
            max_regions=2,
        )

        self.assertIsInstance(payload, dict)
        self.assertTrue(payload["som_overlay_image_data_url"].startswith("data:image/jpeg;base64,"))
        self.assertIsNone(payload["som_collage_image_data_url"])
        legend = payload["som_overlay_legend"]
        self.assertEqual(legend["image_role"], "numbered_som_overlay")
        self.assertEqual(legend["coordinate_space"], "llm_image_uv")
        self.assertEqual(legend["source_coordinate_space"], "image_uv")
        self.assertEqual(legend["llm_image_size"], [24, 20])
        self.assertEqual(legend["label_mapping"][0]["label"], 1)
        self.assertEqual(legend["label_mapping"][0]["region_id"], ranked[0]["region_id"])
        self.assertEqual(legend["label_mapping"][0]["side"], "right")
        self.assertEqual(legend["source_label_mapping"][0]["center_uv"], ranked[0]["center_uv"])


if __name__ == "__main__":
    unittest.main()
