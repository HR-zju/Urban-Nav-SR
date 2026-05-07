from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "som" / "context.py"
_SPEC = importlib.util.spec_from_file_location("som_context", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_som_context = _MODULE.build_som_context
rank_som_regions_for_route = _MODULE.rank_som_regions_for_route
som_summary_for_llm = _MODULE.som_summary_for_llm


class SomContextTest(unittest.TestCase):
    def test_builds_and_ranks_traversable_regions(self) -> None:
        mask = np.zeros((10, 12), dtype=np.int32)
        mask[6:10, 1:5] = 1
        mask[5:9, 7:11] = 1
        mask[1:3, 5:8] = 2
        segmentation_context = {
            "mask": mask,
            "id_to_name": {0: "road", 1: "sidewalk", 2: "tree"},
            "traversable_ids": {1},
            "forbidden_ids": {0, 2},
        }
        som_context = build_som_context(segmentation_context, min_region_area_ratio=0.01)

        self.assertIsInstance(som_context, dict)
        self.assertEqual(som_context["region_count"], 2)
        self.assertEqual(som_context["regions"][0]["region_id"], "som_1")
        self.assertIn(som_context["regions"][0]["side"], {"left", "right", "center"})

        ranked = rank_som_regions_for_route(
            som_context,
            {"local_semantics": {"turn_type": "turn_right", "instruction_hint": "turn right"}},
            top_k=2,
        )
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["side"], "right")
        self.assertEqual(ranked[0]["label"], 1)
        self.assertIn("route_intent_score", ranked[0])

        summary = som_summary_for_llm(som_context, ranked_regions=ranked)
        self.assertEqual(summary["primary_region_id"], ranked[0]["region_id"])
        self.assertEqual(summary["regions"][0]["label"], 1)
        self.assertEqual(summary["regions"][0]["bbox_uv"], ranked[0]["bbox_uv"])
        self.assertGreaterEqual(summary["forbidden_region_count"], 1)
        self.assertIn("forbidden_regions", summary)
        self.assertGreaterEqual(summary["forbidden_regions"][0]["label"], len(summary["regions"]) + 1)

    def test_builds_conditional_traversable_regions(self) -> None:
        mask = np.zeros((12, 12), dtype=np.int32)
        mask[7:12, 0:12] = 0
        mask[4:9, 4:8] = 1
        mask[1:3, 5:8] = 2
        segmentation_context = {
            "mask": mask,
            "id_to_name": {0: "road", 1: "sidewalk", 2: "tree"},
            "traversable_ids": {1},
            "conditional_traversable_ids": {0},
            "allowed_traversable_ids": {0, 1},
            "forbidden_ids": {2},
        }

        som_context = build_som_context(segmentation_context, min_region_area_ratio=0.01)
        ranked = rank_som_regions_for_route(som_context, {"local_semantics": {"turn_type": "go_straight"}}, top_k=4)
        summary = som_summary_for_llm(som_context, ranked_regions=ranked)

        self.assertIsInstance(som_context, dict)
        self.assertGreaterEqual(som_context["conditional_region_count"], 1)
        self.assertTrue(any(region["region_role"] == "conditional_traversable" for region in ranked))
        self.assertIn("conditional_regions", summary)
        self.assertGreaterEqual(len(summary["conditional_regions"]), 1)


if __name__ == "__main__":
    unittest.main()
