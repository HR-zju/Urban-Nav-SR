#!/usr/bin/env python3
"""Analyze Yelp Open Dataset signals for UrbanNav instruction-seed mining."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEMAND_PATTERNS = {
    "quiet_work_wifi": r"\b(quiet|study|work|laptop|wifi|wi[- ]?fi|outlet|plug|remote work|cowork)\b",
    "romantic_date": r"\b(date night|romantic|anniversary|intimate|cozy|ambien[ct]e|candle|special occasion)\b",
    "family_kids": r"\b(kids?|children|family|families|stroller|play area|kid[- ]friendly)\b",
    "parking": r"\b(parking|parked|garage|valet|lot|street parking)\b",
    "open_late_24h": r"\b(open late|late night|24 hours?|24/7|midnight|after hours)\b",
    "outdoor_seating": r"\b(outdoor|patio|terrace|sidewalk seating|outside seating)\b",
    "group_friendly": r"\b(group|large party|big party|groups|friends|team|crowd)\b",
    "quick_takeout": r"\b(quick|fast|takeout|take out|grab and go|to go|delivery)\b",
    "healthy_vegan": r"\b(healthy|vegan|vegetarian|gluten[- ]free|salad|organic)\b",
    "cheap_budget": r"\b(cheap|affordable|inexpensive|budget|reasonable price|good value)\b",
    "pet_dog_friendly": r"\b(dog friendly|dogs allowed|pet friendly|bring my dog|dog patio)\b",
    "breakfast_brunch": r"\b(breakfast|brunch|morning coffee|bagel|pancake)\b",
    "coffee_drink": r"\b(coffee|espresso|latte|tea|boba|drink|thirsty)\b",
    "dessert_snack": r"\b(dessert|ice cream|cake|pastry|snack|sweet)\b",
}


INSTRUCTION_SEED_TEMPLATES = {
    "quiet_work_wifi": {
        "category": "semantic_preference",
        "subcategory": "quiet_work_place",
        "target_types": ["cafe", "library", "coworking_space"],
        "semantic_tags": ["quiet", "wifi", "work_friendly", "power_outlet"],
        "templates": ["I want to find a quiet place with Wi-Fi where I can work."],
    },
    "romantic_date": {
        "category": "semantic_preference",
        "subcategory": "romantic_restaurant",
        "target_types": ["restaurant"],
        "semantic_tags": ["romantic", "date_night", "intimate"],
        "templates": ["I want to find a romantic restaurant for a date."],
    },
    "family_kids": {
        "category": "semantic_preference",
        "subcategory": "family_friendly_place",
        "target_types": ["restaurant", "attraction", "park"],
        "semantic_tags": ["family_friendly", "kids_friendly"],
        "templates": ["I need a place suitable for bringing kids."],
    },
    "parking": {
        "category": "semantic_preference",
        "subcategory": "place_with_parking",
        "target_types": ["restaurant", "shopping_mall", "store"],
        "semantic_tags": ["parking", "garage", "valet"],
        "templates": ["Find a nearby place where parking is convenient."],
    },
    "open_late_24h": {
        "category": "semantic_preference",
        "subcategory": "open_late_place",
        "target_types": ["restaurant", "convenience_store", "pharmacy"],
        "semantic_tags": ["open_late", "open_24h"],
        "templates": ["I need a place that is still open late."],
    },
    "outdoor_seating": {
        "category": "semantic_preference",
        "subcategory": "outdoor_seating_restaurant",
        "target_types": ["restaurant", "cafe"],
        "semantic_tags": ["outdoor_seating", "patio"],
        "templates": ["Find a nearby restaurant or cafe with outdoor seating."],
    },
    "group_friendly": {
        "category": "semantic_preference",
        "subcategory": "group_friendly_restaurant",
        "target_types": ["restaurant", "bar"],
        "semantic_tags": ["good_for_groups", "large_party"],
        "templates": ["I need a place suitable for a group meal."],
    },
    "quick_takeout": {
        "category": "implicit_need_navigation",
        "subcategory": "quick_takeout",
        "target_types": ["fast_food", "restaurant", "takeout"],
        "semantic_tags": ["quick_service", "takeout"],
        "templates": ["I need somewhere quick to grab food to go."],
    },
    "healthy_vegan": {
        "category": "semantic_preference",
        "subcategory": "healthy_or_vegan_food",
        "target_types": ["restaurant", "salad", "vegetarian_restaurant"],
        "semantic_tags": ["healthy", "vegan", "vegetarian"],
        "templates": ["Find a nearby healthy or vegan-friendly place to eat."],
    },
    "cheap_budget": {
        "category": "semantic_preference",
        "subcategory": "budget_friendly_place",
        "target_types": ["restaurant", "store"],
        "semantic_tags": ["affordable", "budget_friendly"],
        "templates": ["Find an affordable nearby place."],
    },
    "pet_dog_friendly": {
        "category": "semantic_preference",
        "subcategory": "dog_friendly_place",
        "target_types": ["restaurant", "cafe", "hotel"],
        "semantic_tags": ["dog_friendly", "pet_friendly"],
        "templates": ["Find a nearby place where I can bring my dog."],
    },
    "breakfast_brunch": {
        "category": "implicit_need_navigation",
        "subcategory": "breakfast_or_brunch",
        "target_types": ["restaurant", "cafe", "breakfast"],
        "semantic_tags": ["breakfast", "brunch"],
        "templates": ["I want to find somewhere for breakfast or brunch."],
    },
    "coffee_drink": {
        "category": "implicit_need_navigation",
        "subcategory": "coffee_or_drink",
        "target_types": ["cafe", "coffee_shop", "beverage_shop"],
        "semantic_tags": ["coffee", "drink"],
        "templates": ["I want to get coffee or something to drink nearby."],
    },
    "dessert_snack": {
        "category": "implicit_need_navigation",
        "subcategory": "dessert_or_snack",
        "target_types": ["bakery", "dessert_shop", "ice_cream"],
        "semantic_tags": ["dessert", "snack"],
        "templates": ["I want to find a place for dessert or a snack."],
    },
}


def read_jsonl(path: Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if limit is not None and index > limit:
                break
            yield json.loads(line)


def scan_text(
    path: Path,
    *,
    business_meta: dict[str, dict[str, Any]],
    patterns: dict[str, re.Pattern[str]],
    limit: int | None,
) -> dict[str, Any]:
    total = 0
    group_counts: Counter[str] = Counter()
    category_by_group: dict[str, Counter[str]] = defaultdict(Counter)
    city_by_group: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in read_jsonl(path, limit=limit):
        total += 1
        text = row.get("text") or ""
        business = business_meta.get(row.get("business_id"), {})
        for group, pattern in patterns.items():
            if not pattern.search(text):
                continue
            group_counts[group] += 1
            for category in business.get("categories", [])[:12]:
                category_by_group[group][category] += 1
            city_by_group[group][business.get("city") or ""] += 1
            if len(examples[group]) < 5:
                examples[group].append(
                    {
                        "business": business.get("name"),
                        "city": business.get("city"),
                        "stars": row.get("stars"),
                        "text_excerpt": text[:260].replace("\n", " "),
                    }
                )

    return {
        "rows_scanned": total,
        "group_counts": dict(group_counts.most_common()),
        "top_categories_by_group": {
            group: dict(counter.most_common(20)) for group, counter in category_by_group.items()
        },
        "top_cities_by_group": {
            group: dict(counter.most_common(20)) for group, counter in city_by_group.items()
        },
        "examples_by_group": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Yelp Open Dataset for UrbanNav instruction-seed mining.")
    parser.add_argument("--yelp-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--review-limit", type=int, default=1_000_000)
    parser.add_argument("--tip-limit", type=int, default=None)
    args = parser.parse_args()

    business_path = args.yelp_root / "yelp_academic_dataset_business.json"
    review_path = args.yelp_root / "yelp_academic_dataset_review.json"
    tip_path = args.yelp_root / "yelp_academic_dataset_tip.json"

    business_meta: dict[str, dict[str, Any]] = {}
    city_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    attr_counts: Counter[str] = Counter()
    attr_values: dict[str, Counter[str]] = defaultdict(Counter)
    open_counts: Counter[str] = Counter()

    for business in read_jsonl(business_path):
        categories = [item.strip() for item in (business.get("categories") or "").split(",") if item.strip()]
        attributes = business.get("attributes") or {}
        business_meta[business["business_id"]] = {
            "name": business.get("name"),
            "city": business.get("city"),
            "state": business.get("state"),
            "stars": business.get("stars"),
            "review_count": business.get("review_count"),
            "categories": categories,
            "attributes": attributes,
            "hours": business.get("hours") or {},
            "latitude": business.get("latitude"),
            "longitude": business.get("longitude"),
        }
        city_counts[business.get("city") or ""] += 1
        state_counts[business.get("state") or ""] += 1
        open_counts[str(business.get("is_open"))] += 1
        for category in categories:
            category_counts[category] += 1
        for key, value in attributes.items():
            attr_counts[key] += 1
            attr_values[key][str(value)] += 1

    patterns = {key: re.compile(value, re.I) for key, value in DEMAND_PATTERNS.items()}
    review_stats = scan_text(review_path, business_meta=business_meta, patterns=patterns, limit=args.review_limit)
    tip_stats = scan_text(tip_path, business_meta=business_meta, patterns=patterns, limit=args.tip_limit)

    selected_attrs = [
        "WiFi",
        "BusinessParking",
        "Ambience",
        "GoodForKids",
        "OutdoorSeating",
        "RestaurantsGoodForGroups",
        "NoiseLevel",
        "RestaurantsPriceRange2",
        "Open24Hours",
        "DogsAllowed",
        "HappyHour",
    ]

    summary = {
        "business_total": len(business_meta),
        "review_scanned": review_stats["rows_scanned"],
        "tip_scanned": tip_stats["rows_scanned"],
        "top_states": dict(state_counts.most_common(30)),
        "top_cities": dict(city_counts.most_common(50)),
        "open_counts": dict(open_counts),
        "top_categories": dict(category_counts.most_common(100)),
        "top_attributes": dict(attr_counts.most_common(100)),
        "selected_attribute_values": {
            key: dict(attr_values[key].most_common(20)) for key in selected_attrs
        },
        "review_demand_stats": review_stats,
        "tip_demand_stats": tip_stats,
        "instruction_seed_templates": INSTRUCTION_SEED_TEMPLATES,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("business_total", "review_scanned", "tip_scanned")}, indent=2))
    print("review_groups", summary["review_demand_stats"]["group_counts"])
    print("tip_groups", summary["tip_demand_stats"]["group_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
