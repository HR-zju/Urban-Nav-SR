"""Tests for UrbanNav instruction-set compilation."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys


ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROUTE_SERVER_ROOT / "tools" / "dataset_processing" / "build_urbannav_instruction_set.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_urbannav_instruction_set", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_instruction_set_builder_compiles_full_distribution(tmp_path: Path) -> None:
    """The full benchmark plan should not collapse to fixed nearby templates."""

    builder = _load_builder()
    artifacts = builder.build_instruction_set(
        taxonomy_path=ROUTE_SERVER_ROOT / "config" / "yelp_demand_taxonomy.json",
        distribution_path=ROUTE_SERVER_ROOT / "config" / "urbannav_10k_distribution_plan.json",
        provider_map_path=ROUTE_SERVER_ROOT / "config" / "urbannav_target_type_provider_map.json",
        regions_config_path=ROUTE_SERVER_ROOT / "config" / "urbannav_global_gt_collection_full_en.json",
        output_root=tmp_path,
        language="en",
        instruction_mode="case_variant",
        seed=20260629,
        include_semantic_analysis=False,
        allow_missing_target_types=False,
        overwrite=True,
    )

    manifest = artifacts["manifest"]
    assert manifest["instruction_rows"] == 10_000
    assert manifest["category_distribution"] == {
        "implicit_need_navigation": 3500,
        "multi_stop_route": 1500,
        "poi_navigation": 4000,
        "route_update_add_via": 1000,
    }
    assert manifest["single_stop_collection_cases"] == 7500
    assert manifest["complex_collection_cases"] == 2500
    assert manifest["template_count"] > 100
    assert manifest["unique_instruction_count"] > 9000

    rows = [json.loads(line) for line in (tmp_path / "instruction_set.jsonl").read_text(encoding="utf-8").splitlines()]
    unique_instructions = {row["instruction"] for row in rows}
    nearby_rows = [row for row in rows if "nearby" in row["instruction"].lower()]
    assert len(unique_instructions) > 9000
    assert 0 < len(nearby_rows) < len(rows)
    assert any(row["category"] == "multi_stop_route" for row in rows)
    assert any(row["category"] == "route_update_add_via" for row in rows)

    collection_config = json.loads((tmp_path / "collection_config_full_single_stop.json").read_text(encoding="utf-8"))
    assert sum(task["planned_count"] for task in collection_config["tasks"]) == 7500
    assert all(task["category"] in {"poi_navigation", "implicit_need_navigation"} for task in collection_config["tasks"])
