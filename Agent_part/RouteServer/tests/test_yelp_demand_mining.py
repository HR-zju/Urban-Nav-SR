"""Tests for Yelp-supported UrbanNav demand mining."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROUTE_SERVER_ROOT / "tools" / "dataset_processing" / "mine_yelp_urbannav_demands.py"


def _load_miner():
    spec = importlib.util.spec_from_file_location("mine_yelp_urbannav_demands", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_yelp_demand_miner_promotes_provider_mapped_demands(tmp_path: Path) -> None:
    """Mining should turn evidence-backed rules into merged taxonomy specs."""

    miner = _load_miner()
    yelp_root = tmp_path / "yelp"
    yelp_root.mkdir()
    _write_jsonl(
        yelp_root / "yelp_academic_dataset_business.json",
        [
            {
                "business_id": "b1",
                "name": "City Laundry",
                "city": "A",
                "categories": "Laundry Services, Local Services",
                "attributes": {},
            },
            {
                "business_id": "b2",
                "name": "Clean Clothes",
                "city": "B",
                "categories": "Dry Cleaning & Laundry, Local Services",
                "attributes": {},
            },
        ],
    )
    _write_jsonl(
        yelp_root / "yelp_academic_dataset_review.json",
        [
            {"business_id": "b1", "text": "Good laundromat where I can do laundry quickly."},
            {"business_id": "b2", "text": "Useful dry clean option when I need to wash clothes."},
        ],
    )
    _write_jsonl(
        yelp_root / "yelp_academic_dataset_tip.json",
        [
            {"business_id": "b1", "text": "Easy place to do laundry."},
            {"business_id": "b2", "text": "Dry clean pickup is fast."},
        ],
    )

    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "thresholds": {
                    "main_min_businesses": 1,
                    "main_min_cities": 1,
                    "main_min_text_evidence": 1,
                    "analysis_min_businesses": 1,
                    "analysis_min_cities": 1,
                    "analysis_min_text_evidence": 1,
                },
                "target_type_additions": {
                    "laundromat": {
                        "query_terms": ["laundromat"],
                        "osm_tags": ["shop=laundry"],
                        "tomtom_category_set": ["9362"],
                    }
                },
                "candidate_rules": [
                    {
                        "demand_group": "do_laundry_need",
                        "category": "implicit_need_navigation",
                        "subcategory": "do_laundry",
                        "target_types": ["laundromat"],
                        "semantic_tags": ["laundry"],
                        "templates_en": ["I need somewhere nearby to do laundry."],
                        "templates_zh": ["我需要找附近可以洗衣服的地方。"],
                        "yelp_categories_any": ["Laundry Services", "Dry Cleaning & Laundry"],
                        "attribute_any": [],
                        "text_pattern": "\\b(laundry|laundromat|dry clean|wash clothes)\\b",
                        "review_by_default": False,
                        "quota_weight": 50,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps({"version": "base", "specs": []}), encoding="utf-8")
    provider_map_path = tmp_path / "provider_map.json"
    provider_map_path.write_text(json.dumps({"version": "base", "target_types": {}}), encoding="utf-8")

    result = miner.mine_yelp_demands(
        yelp_root=yelp_root,
        rules_path=rules_path,
        taxonomy_path=taxonomy_path,
        provider_map_path=provider_map_path,
        distribution_path=tmp_path / "distribution.json",
        regions_config_path=tmp_path / "regions.json",
        output_root=tmp_path / "mining",
        taxonomy_output=tmp_path / "taxonomy_v2.json",
        provider_map_output=tmp_path / "provider_map_v2.json",
        instruction_output_root=tmp_path / "instructions",
        business_limit=None,
        review_limit=None,
        tip_limit=None,
        build_instruction_set=False,
        build_review_html=False,
        overwrite=True,
    )

    assert result["accepted_new_specs"] == 1
    taxonomy_v2 = json.loads((tmp_path / "taxonomy_v2.json").read_text(encoding="utf-8"))
    provider_map_v2 = json.loads((tmp_path / "provider_map_v2.json").read_text(encoding="utf-8"))
    assert taxonomy_v2["specs"][0]["demand_group"] == "do_laundry_need"
    assert "laundromat" in provider_map_v2["target_types"]
    report = json.loads((tmp_path / "mining" / "demand_mining_report.json").read_text(encoding="utf-8"))
    assert report["decision_distribution"]["accept_main"] == 1
