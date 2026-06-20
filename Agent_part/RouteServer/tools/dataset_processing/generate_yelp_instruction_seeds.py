#!/usr/bin/env python3
"""Generate Yelp-guided UrbanNav instruction seeds.

This script creates instruction-only rows. It does not create destination or
route ground truth; downstream builders must rebuild those with the active map
provider and RouteServer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROUTESERVER_SRC = Path(__file__).resolve().parents[2] / "src"
if str(ROUTESERVER_SRC) not in sys.path:
    sys.path.insert(0, str(ROUTESERVER_SRC))

try:
    from mcp_server.llm import (
        LLMSettings,
        _close_async_client,
        _create_async_client,
        _extract_text_from_chat_response,
        _extract_text_from_responses_response,
        _parse_json_object_from_text,
    )
except Exception:  # pragma: no cover - optional dependency for offline/template mode.
    LLMSettings = None
    _close_async_client = None
    _create_async_client = None
    _extract_text_from_chat_response = None
    _extract_text_from_responses_response = None
    _parse_json_object_from_text = None


GT_BUILD_PLAN = {
    "origin": "sample_from_target_urban_nav_region",
    "gt_destination": "rebuild_with_poi_provider",
    "gt_route": "rebuild_with_routeserver_or_graph_shortest_path",
    "ranking_policy": "nearest_satisfying_poi_by_route_distance",
}


@dataclass(frozen=True)
class DemandSpec:
    demand_group: str
    category: str
    category_zh: str
    subcategory: str
    target_types: list[str]
    semantic_tags: list[str]
    templates_en: list[str]
    templates_zh: list[str]
    yelp_categories_any: tuple[str, ...] = ()
    attribute_any: tuple[str, ...] = ()
    text_pattern: str | None = None
    ordered_target_types: list[list[str]] | None = None
    review_by_default: bool = False


DEFAULT_YELP_TAXONOMY = Path(__file__).resolve().parents[2] / "config" / "yelp_demand_taxonomy.json"


EXTRA_TEMPLATES: dict[str, dict[str, list[str]]] = {}
MULTI_STOP_EXTRA: dict[str, dict[str, list[str]]] = {}
ROUTE_UPDATE_EXTRA: dict[str, dict[str, list[str]]] = {}
QUOTA_WEIGHTS: dict[str, int] = {}


def _as_str_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    return [str(item) for item in value if str(item).strip()]


def _as_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    return tuple(_as_str_list(value, field=field))


def load_yelp_demand_taxonomy(path: Path) -> list[DemandSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    specs_raw = data.get("specs")
    if not isinstance(specs_raw, list) or not specs_raw:
        raise ValueError(f"Yelp taxonomy config has no specs: {path}")

    global EXTRA_TEMPLATES, MULTI_STOP_EXTRA, ROUTE_UPDATE_EXTRA, QUOTA_WEIGHTS
    EXTRA_TEMPLATES = data.get("extra_templates") or {}
    MULTI_STOP_EXTRA = data.get("multi_stop_extra") or {}
    ROUTE_UPDATE_EXTRA = data.get("route_update_extra") or {}
    QUOTA_WEIGHTS = {}

    specs: list[DemandSpec] = []
    seen_groups: set[str] = set()
    required = {"demand_group", "category", "category_zh", "subcategory"}
    for index, item in enumerate(specs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"specs[{index}] must be an object")
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"specs[{index}] missing required fields: {missing}")
        demand_group = str(item["demand_group"]).strip()
        if not demand_group:
            raise ValueError(f"specs[{index}] has empty demand_group")
        if demand_group in seen_groups:
            raise ValueError(f"duplicate demand_group in Yelp taxonomy: {demand_group}")
        seen_groups.add(demand_group)
        text_pattern = item.get("text_pattern")
        if text_pattern:
            re.compile(str(text_pattern))
        specs.append(
            DemandSpec(
                demand_group=demand_group,
                category=str(item["category"]).strip(),
                category_zh=str(item["category_zh"]).strip(),
                subcategory=str(item["subcategory"]).strip(),
                target_types=_as_str_list(item.get("target_types"), field=f"{demand_group}.target_types"),
                semantic_tags=_as_str_list(item.get("semantic_tags"), field=f"{demand_group}.semantic_tags"),
                templates_en=_as_str_list(item.get("templates_en"), field=f"{demand_group}.templates_en"),
                templates_zh=_as_str_list(item.get("templates_zh"), field=f"{demand_group}.templates_zh"),
                yelp_categories_any=_as_tuple(item.get("yelp_categories_any"), field=f"{demand_group}.yelp_categories_any"),
                attribute_any=_as_tuple(item.get("attribute_any"), field=f"{demand_group}.attribute_any"),
                text_pattern=str(text_pattern) if text_pattern else None,
                ordered_target_types=item.get("ordered_target_types"),
                review_by_default=bool(item.get("review_by_default", False)),
            )
        )
        QUOTA_WEIGHTS[demand_group] = int(item.get("quota_weight") or 1)
    return specs


def templates_for_spec(spec: DemandSpec, external: dict[str, list[str]] | None = None) -> list[tuple[str, str]]:
    extras = EXTRA_TEMPLATES.get(spec.demand_group, {"zh": [], "en": []})
    if spec.category == "multi_stop_route":
        extra2 = MULTI_STOP_EXTRA.get(spec.demand_group, {"zh": [], "en": []})
        extras = {"zh": [*extras.get("zh", []), *extra2.get("zh", [])], "en": [*extras.get("en", []), *extra2.get("en", [])]}
    if spec.category == "route_update_add_via":
        extra2 = ROUTE_UPDATE_EXTRA.get(spec.demand_group, {"zh": [], "en": []})
        extras = {"zh": [*extras.get("zh", []), *extra2.get("zh", [])], "en": [*extras.get("en", []), *extra2.get("en", [])]}
    raw = [(text, "base") for text in spec.templates_zh + spec.templates_en]
    raw.extend((text, "manual_expanded") for text in extras.get("zh", []) + extras.get("en", []))
    for text in (external or {}).get(spec.demand_group, []):
        raw.append((text, "llm_external"))
    for text in (external or {}).get(spec.subcategory, []):
        raw.append((text, "llm_external"))

    dedup: dict[str, tuple[str, str]] = {}
    for text, source in raw:
        stripped = " ".join(str(text).split())
        if stripped:
            dedup.setdefault(normalize_instruction(stripped), (stripped, source))
    return list(dedup.values())


def load_external_paraphrases(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(path):
        instruction = str(row.get("instruction") or row.get("text") or "").strip()
        if not instruction:
            continue
        for key in (row.get("demand_group"), row.get("subcategory")):
            if key:
                out[str(key)].append(instruction)
    return dict(out)


def read_jsonl(path: Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if limit is not None and index > limit:
                break
            yield json.loads(line)


def load_api_keys(path: Path | None) -> list[str]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8")
    keys: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for part in line.split(","):
            key = part.strip()
            if key:
                keys.append(key)
    seen: set[str] = set()
    unique: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def normalize_instruction(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def instruction_language(text: str) -> str:
    return "zh" if is_chinese(text) else "en"


def instruction_variant(template: str, *, variant_index: int) -> str:
    return template


def variant_count(template: str) -> int:
    return 1


def core_instruction_key(text: str) -> str:
    return normalize_instruction(text)


def validate_instruction_text(
    text: str,
    *,
    spec: DemandSpec,
    business_names: set[str] | None = None,
    location_names: set[str] | None = None,
) -> tuple[bool, str]:
    stripped = " ".join(str(text or "").split())
    if len(stripped) < 6:
        return False, "too_short"
    if len(stripped) > 220:
        return False, "too_long"
    lowered = stripped.lower()
    blocked = ("yelp", "review", "rating", "stars", "business_id")
    if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", lowered) for token in blocked):
        return False, "metadata_leak"
    generic_name_tokens = {
        "bar",
        "burger",
        "burgers",
        "cafe",
        "coffee",
        "cup",
        "deli",
        "donut",
        "donuts",
        "food",
        "fresh",
        "good",
        "grill",
        "hotel",
        "juice",
        "kitchen",
        "pizza",
        "restaurant",
        "restaurants",
        "shop",
        "tea",
    }
    for name in business_names or set():
        normalized_name = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(name or "").strip().lower()).strip()
        if not normalized_name:
            continue
        name_tokens = normalized_name.split()
        if len(name_tokens) == 1 and (len(normalized_name) < 8 or normalized_name in generic_name_tokens):
            continue
        if len(name_tokens) > 1 and all(token in generic_name_tokens for token in name_tokens):
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(normalized_name)}(?!\w)")
        if pattern.search(lowered):
            return False, "business_name_leak"
    if spec.category == "route_update_add_via":
        update_tokens = (
            "add",
            "insert",
            "detour",
            "along my way",
            "on the way",
            "before continuing",
            "before the destination",
            "final destination",
            "current route",
            "route",
            "stop at",
            "加",
            "插入",
            "顺路",
            "路上",
            "中途",
            "路线",
            "绕",
            "目的地前",
            "继续去",
        )
        if not any(token in lowered or token in stripped for token in update_tokens):
            return False, "missing_route_update_intent"
    return True, "ok"


def map_retrievability_decision(instruction: str, spec: DemandSpec) -> tuple[str, str]:
    if spec.category == "semantic_preference":
        return "review", "semantic_preference_requires_attribute_or_human_review"
    return "ok", "map_retrievable"


def has_true_attr(attributes: dict[str, Any], keys: tuple[str, ...]) -> bool:
    if not keys:
        return False
    for key in keys:
        value = attributes.get(key)
        text = str(value).lower()
        if key == "BusinessParking" and any(token in text for token in ("'garage': true", "'lot': true", "'street': true", "'valet': true")):
            return True
        if key == "NoiseLevel" and "quiet" in text:
            return True
        if key == "WiFi" and ("free" in text or "paid" in text):
            return True
        if key == "RestaurantsPriceRange2" and text.strip("'\"") in {"1", "2"}:
            return True
        if text in {"true", "1", "yes"}:
            return True
    return False


def category_match(categories: list[str], wanted: tuple[str, ...]) -> bool:
    if not wanted:
        return False
    have = {item.lower() for item in categories}
    return any(item.lower() in have for item in wanted)


def build_business_pool(yelp_root: Path, *, limit: int | None) -> dict[str, dict[str, Any]]:
    business_path = yelp_root / "yelp_academic_dataset_business.json"
    pool: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(business_path, limit=limit):
        categories = [item.strip() for item in (row.get("categories") or "").split(",") if item.strip()]
        pool[row["business_id"]] = {
            "business_id": row["business_id"],
            "name": row.get("name"),
            "city": row.get("city"),
            "state": row.get("state"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "stars": row.get("stars"),
            "review_count": row.get("review_count") or 0,
            "is_open": row.get("is_open"),
            "categories": categories,
            "attributes": row.get("attributes") or {},
            "hours": row.get("hours") or {},
        }
    return pool


def scan_text_signals(
    yelp_root: Path,
    *,
    specs: list[DemandSpec],
    business_pool: dict[str, dict[str, Any]],
    review_limit: int | None,
    tip_limit: int | None,
) -> dict[str, Counter[str]]:
    patterns = {spec.demand_group: re.compile(spec.text_pattern, re.I) for spec in specs if spec.text_pattern}
    signals: dict[str, Counter[str]] = defaultdict(Counter)

    def scan_file(path: Path, limit: int | None, field: str) -> None:
        for row in read_jsonl(path, limit=limit):
            business_id = row.get("business_id")
            if business_id not in business_pool:
                continue
            text = row.get("text") or ""
            for group, pattern in patterns.items():
                if pattern.search(text):
                    signals[business_id][f"{field}:{group}"] += 1

    scan_file(yelp_root / "yelp_academic_dataset_review.json", review_limit, "review")
    scan_file(yelp_root / "yelp_academic_dataset_tip.json", tip_limit, "tip")
    return signals


def compact_text_snippet(text: str, pattern: re.Pattern[str], *, max_chars: int = 260) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    match = pattern.search(normalized)
    if not match:
        return normalized[:max_chars]
    start = max(0, match.start() - max_chars // 3)
    end = min(len(normalized), match.end() + max_chars * 2 // 3)
    snippet = normalized[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(normalized):
        snippet += "..."
    return snippet[: max_chars + 6]


def scan_text_examples(
    yelp_root: Path,
    *,
    specs: list[DemandSpec],
    business_pool: dict[str, dict[str, Any]],
    review_limit: int | None,
    tip_limit: int | None,
    max_examples_per_group: int,
) -> dict[str, list[dict[str, Any]]]:
    patterns = {spec.demand_group: re.compile(spec.text_pattern, re.I) for spec in specs if spec.text_pattern}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()

    def scan_file(path: Path, limit: int | None, field: str) -> None:
        if all(len(items) >= max_examples_per_group for items in examples.values()) and len(examples) >= len(patterns):
            return
        for row in read_jsonl(path, limit=limit):
            business_id = row.get("business_id")
            business = business_pool.get(business_id)
            if not business:
                continue
            text = row.get("text") or ""
            for group, pattern in patterns.items():
                if len(examples[group]) >= max_examples_per_group:
                    continue
                if not pattern.search(text):
                    continue
                key = (group, str(business_id), compact_text_snippet(text, pattern, max_chars=120))
                if key in seen:
                    continue
                seen.add(key)
                examples[group].append(
                    {
                        "business_id": business_id,
                        "business_name": business.get("name"),
                        "city": business.get("city"),
                        "state": business.get("state"),
                        "categories": business.get("categories", [])[:8],
                        "stars": business.get("stars"),
                        "review_count": business.get("review_count"),
                        "source": field,
                        "text_snippet": compact_text_snippet(text, pattern),
                    }
                )

    scan_file(yelp_root / "yelp_academic_dataset_review.json", review_limit, "review")
    scan_file(yelp_root / "yelp_academic_dataset_tip.json", tip_limit, "tip")
    return dict(examples)


def build_yelp_demand_evidence(
    *,
    specs: list[DemandSpec],
    candidate_cache: dict[str, list[dict[str, Any]]],
    text_examples: dict[str, list[dict[str, Any]]],
    evidence_per_subcategory: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        examples_by_business: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example in text_examples.get(spec.demand_group, []):
            examples_by_business[str(example.get("business_id") or "")].append(example)
        for rank, business in enumerate(candidate_cache.get(spec.demand_group, [])[:evidence_per_subcategory]):
            business_id = str(business.get("business_id") or "")
            snippets = examples_by_business.get(business_id) or []
            evidence = business.get("evidence", {}) or {}
            signal_count = int(evidence.get("review_signal_count", 0) or 0) + int(evidence.get("tip_signal_count", 0) or 0)
            score = 0.35
            if evidence.get("category_hit"):
                score += 0.2
            if evidence.get("attribute_hit"):
                score += 0.25
            score += min(0.2, signal_count / 20.0)
            rows.append(
                {
                    "evidence_id": f"yelp_evidence_{spec.demand_group}_{rank:05d}",
                    "dataset": "yelp_demand_evidence",
                    "category": spec.category,
                    "category_zh": spec.category_zh,
                    "subcategory": spec.subcategory,
                    "demand_group": spec.demand_group,
                    "target_types": spec.target_types or None,
                    "ordered_target_types": spec.ordered_target_types,
                    "semantic_tags": spec.semantic_tags or None,
                    "business_id": business_id,
                    "business_name": business.get("name"),
                    "city": business.get("city"),
                    "state": business.get("state"),
                    "business_categories": business.get("categories", [])[:12],
                    "attributes_present": [
                        key for key in spec.attribute_any if key in (business.get("attributes") or {})
                    ],
                    "evidence": evidence,
                    "evidence_score": round(min(score, 0.99), 3),
                    "snippets": [
                        {
                            "source": item.get("source"),
                            "text_snippet": item.get("text_snippet"),
                        }
                        for item in snippets[:4]
                    ],
                }
            )
    return rows


def evidence_by_group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("demand_group") or "")].append(row)
    return dict(grouped)


def candidate_businesses(
    spec: DemandSpec,
    *,
    business_pool: dict[str, dict[str, Any]],
    text_signals: dict[str, Counter[str]],
    min_review_count: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for business in business_pool.values():
        if business["is_open"] != 1:
            continue
        if int(business.get("review_count") or 0) < min_review_count:
            continue
        attr_hit = has_true_attr(business.get("attributes") or {}, spec.attribute_any)
        category_hit = category_match(business.get("categories") or [], spec.yelp_categories_any)
        signal_counter = text_signals.get(business["business_id"], Counter())
        review_hits = sum(v for k, v in signal_counter.items() if k.endswith(spec.demand_group) and k.startswith("review:"))
        tip_hits = sum(v for k, v in signal_counter.items() if k.endswith(spec.demand_group) and k.startswith("tip:"))
        text_hit = review_hits + tip_hits > 0
        if category_hit or attr_hit or text_hit:
            candidates.append(
                {
                    **business,
                    "evidence": {
                        "category_hit": category_hit,
                        "attribute_hit": attr_hit,
                        "review_signal_count": review_hits,
                        "tip_signal_count": tip_hits,
                    },
                }
            )
    candidates.sort(
        key=lambda item: (
            -int(item["evidence"]["review_signal_count"] + item["evidence"]["tip_signal_count"]),
            -int(item["evidence"]["attribute_hit"]),
            -int(item["evidence"]["category_hit"]),
            -float(item.get("stars") or 0),
            -int(item.get("review_count") or 0),
            str(item.get("business_id") or ""),
        )
    )
    return candidates


def quota_for_total(target_total: int, weights: dict[str, int]) -> dict[str, int]:
    if not weights:
        raise ValueError("No quota weights loaded from Yelp taxonomy config.")
    scale = target_total / sum(weights.values())
    quotas = {key: max(1, round(value * scale)) for key, value in weights.items()}
    delta = target_total - sum(quotas.values())
    keys = list(quotas)
    idx = 0
    while delta:
        key = keys[idx % len(keys)]
        if delta > 0:
            quotas[key] += 1
            delta -= 1
        elif quotas[key] > 1:
            quotas[key] -= 1
            delta += 1
        idx += 1
    return quotas


def make_seed(
    *,
    spec: DemandSpec,
    business: dict[str, Any] | None,
    instruction: str,
    index: int,
    status: str,
    reason: str,
    confidence: float,
    all_templates: list[str],
    base_template: str | None = None,
    template_source: str | None = None,
) -> dict[str, Any]:
    business_id = business.get("business_id") if business else None
    evidence = business.get("evidence", {}) if business else {}
    return {
        "id": f"yelp_instruction_{spec.category}_{spec.subcategory}_{index:06d}",
        "dataset": "yelp_instruction_seed",
        "conversion_status": status,
        "conversion_reason": reason,
        "conversion_confidence": round(confidence, 3),
        "category": spec.category,
        "category_zh": spec.category_zh,
        "subcategory": spec.subcategory,
        "instruction": instruction,
        "language": instruction_language(instruction),
        "instruction_templates": all_templates,
        "base_template": base_template,
        "template_source": template_source,
        "target_types": spec.target_types or None,
        "ordered_target_types": spec.ordered_target_types,
        "brand_aliases": None,
        "semantic_tags": spec.semantic_tags or None,
        "source": {
            "name": "yelp_open_dataset",
            "format": "Yelp",
            "demand_group": spec.demand_group,
            "business_id": business_id,
            "business_name": business.get("name") if business else None,
            "city": business.get("city") if business else None,
            "state": business.get("state") if business else None,
            "taxonomy": "cityseeker_refactored_v2",
            "gt_policy": "rebuild_with_poi_provider_and_routeserver",
        },
        "yelp_evidence": {
            "business_attribute": list(spec.attribute_any),
            "category_hit": evidence.get("category_hit", False),
            "attribute_hit": evidence.get("attribute_hit", False),
            "review_signal_count": evidence.get("review_signal_count", 0),
            "tip_signal_count": evidence.get("tip_signal_count", 0),
            "rating": business.get("stars") if business else None,
            "review_count": business.get("review_count") if business else None,
            "business_categories": business.get("categories", [])[:12] if business else [],
        },
        "gt_build_plan": dict(GT_BUILD_PLAN),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class YelpParaphraseGenerator:
    def __init__(self, *, instructions_per_spec: int, english_ratio: float) -> None:
        if LLMSettings is None:
            self.settings = None
        else:
            self.settings = LLMSettings.from_env()
        self.instructions_per_spec = instructions_per_spec
        self.english_ratio = english_ratio

    def is_runtime_available(self, api_keys: list[str]) -> bool:
        return bool(self.settings and (api_keys or self.settings.api_key) and _create_async_client is not None)

    def build_messages(
        self,
        spec: DemandSpec,
        *,
        candidates: list[dict[str, Any]],
        text_examples: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]] | None = None,
        batch_index: int = 0,
    ) -> list[dict[str, str]]:
        candidate_limit = int(os.getenv("LLM_YELP_CANDIDATE_EXAMPLES", "4") or 4)
        evidence_limit = int(os.getenv("LLM_YELP_EVIDENCE_EXAMPLES", "4") or 4)
        snippet_limit = int(os.getenv("LLM_YELP_SNIPPETS_PER_EVIDENCE", "1") or 1)
        text_example_limit = int(os.getenv("LLM_YELP_TEXT_EXAMPLES", "4") or 4)
        candidate_examples = [
            {
                "name": item.get("name"),
                "city": item.get("city"),
                "state": item.get("state"),
                "categories": item.get("categories", [])[:8],
                "attributes_present": [key for key in spec.attribute_any if key in (item.get("attributes") or {})],
                "evidence": item.get("evidence", {}),
                "stars": item.get("stars"),
                "review_count": item.get("review_count"),
            }
            for item in candidates[:candidate_limit]
        ]
        compact_evidence = [
            {
                "evidence_id": item.get("evidence_id"),
                "business_categories": item.get("business_categories", [])[:8],
                "attributes_present": item.get("attributes_present", []),
                "evidence": item.get("evidence", {}),
                "evidence_score": item.get("evidence_score"),
                "snippets": item.get("snippets", [])[:snippet_limit],
            }
            for item in (evidence_rows or [])[:evidence_limit]
        ]
        payload = {
            "task": "generate_urbannav_user_need_instructions_from_yelp_evidence",
            "category": spec.category,
            "category_zh": spec.category_zh,
            "subcategory": spec.subcategory,
            "demand_group": spec.demand_group,
            "target_types": spec.target_types,
            "ordered_target_types": spec.ordered_target_types,
            "semantic_tags": spec.semantic_tags,
            "seed_templates": spec.templates_zh + spec.templates_en,
            "yelp_category_filters": list(spec.yelp_categories_any),
            "yelp_attribute_filters": list(spec.attribute_any),
            "candidate_business_examples": candidate_examples,
            "review_tip_text_examples": text_examples[:text_example_limit],
            "demand_evidence_clusters": compact_evidence,
            "output_count": self.instructions_per_spec,
            "paraphrase_batch_index": batch_index,
            "batch_instruction": (
                "Generate a fresh batch with different sentence openings, intent wording, and level of specificity. "
                "Do not rely on generic nearby/closest suffixes for diversity."
            ),
            "language_policy": {
                "english_ratio_min": self.english_ratio,
                "english_count_min": math.ceil(self.instructions_per_spec * self.english_ratio),
                "chinese_count_max": self.instructions_per_spec - math.ceil(self.instructions_per_spec * self.english_ratio),
            },
        }
        system = (
            "You create diverse natural user navigation instructions for an UrbanNav benchmark. "
            "Use Yelp evidence only to enrich how people express needs; do not invent destination ground truth. "
            "Keep every instruction mapped to the provided category/subcategory and target types. "
            "Return exactly one compact JSON object, no markdown. "
            "Schema: {\"instructions\":[{\"text\":\"instruction\", \"language\":\"zh|en\", "
            "\"style\":\"short label\", \"evidence_used\":\"short phrase\"}]}. "
            "Generate genuinely different phrasings, including colloquial Chinese and natural English. "
            "At least the requested minimum number of instructions must be English. "
            "Avoid business names unless the subcategory explicitly requires a brand. "
            "Do not include coordinates, Yelp, or review wording copied verbatim. "
            "You may mention high rating or open-now/open-late needs when appropriate, because map APIs can weakly support them. "
            "Do not generate needs based on queue length, seat availability, crowding, cleanliness, service quality, authenticity, scenery, "
            "being famous/trendy/instagrammable, or being the best tasting option; those cannot be reliably retrieved from map APIs."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]

    async def _call_chat_completions(
        self,
        client: Any,
        spec: DemandSpec,
        *,
        candidates: list[dict[str, Any]],
        text_examples: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]] | None = None,
        batch_index: int = 0,
    ) -> str:
        assert self.settings is not None
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": max(self.settings.temperature, 0.7),
            "messages": self.build_messages(
                spec,
                candidates=candidates,
                text_examples=text_examples,
                evidence_rows=evidence_rows,
                batch_index=batch_index,
            ),
            "max_tokens": int(os.getenv("LLM_YELP_PARAPHRASE_MAX_TOKENS", "1200") or 1200),
        }
        if self.settings.response_format and self.settings.provider != "ark":
            payload["response_format"] = {"type": "json_object"}
        response = await client.chat.completions.create(**payload)
        return _extract_text_from_chat_response(response)  # type: ignore[misc]

    async def _call_responses_api(
        self,
        client: Any,
        spec: DemandSpec,
        *,
        candidates: list[dict[str, Any]],
        text_examples: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]] | None = None,
        batch_index: int = 0,
    ) -> str:
        assert self.settings is not None
        prompt = "\n\n".join(
            message["content"]
            for message in self.build_messages(
                spec,
                candidates=candidates,
                text_examples=text_examples,
                evidence_rows=evidence_rows,
                batch_index=batch_index,
            )
        )
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": int(os.getenv("LLM_YELP_PARAPHRASE_MAX_TOKENS", "1200") or 1200),
        }
        response = await client.responses.create(**payload)
        return _extract_text_from_responses_response(response)  # type: ignore[misc]

    async def generate_for_spec(
        self,
        client: Any,
        spec: DemandSpec,
        *,
        candidates: list[dict[str, Any]],
        text_examples: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]] | None = None,
        batch_index: int = 0,
    ) -> list[dict[str, Any]]:
        assert self.settings is not None
        errors: list[str] = []
        styles = [self.settings.api_style]
        if self.settings.allow_api_style_fallback:
            fallback = "responses" if self.settings.api_style == "chat_completions" else "chat_completions"
            if fallback not in styles:
                styles.append(fallback)
        for style in styles:
            try:
                content = (
                    await self._call_responses_api(
                        client,
                        spec,
                        candidates=candidates,
                        text_examples=text_examples,
                        evidence_rows=evidence_rows,
                        batch_index=batch_index,
                    )
                    if style == "responses"
                    else await self._call_chat_completions(
                        client,
                        spec,
                        candidates=candidates,
                        text_examples=text_examples,
                        evidence_rows=evidence_rows,
                        batch_index=batch_index,
                    )
                )
                parsed = _parse_json_object_from_text(content)  # type: ignore[misc]
                instructions = parsed.get("instructions") or []
                rows: list[dict[str, Any]] = []
                for item in instructions:
                    if isinstance(item, str):
                        text = item
                        payload = {}
                    elif isinstance(item, dict):
                        text = str(item.get("text") or item.get("instruction") or "").strip()
                        payload = item
                    else:
                        continue
                    if not text:
                        continue
                    ok, reject_reason = validate_instruction_text(
                        text,
                        spec=spec,
                        business_names={str(item.get("business_name") or "") for item in evidence_rows or []},
                    )
                    if not ok:
                        rows.append(
                            {
                                "demand_group": spec.demand_group,
                                "category": spec.category,
                                "subcategory": spec.subcategory,
                                "instruction": text,
                            "template_source": "llm_rejected",
                            "batch_index": batch_index,
                                "language": payload.get("language") or instruction_language(text),
                                "reject_reason": reject_reason,
                            }
                        )
                        continue
                    retrievability, retrievability_reason = map_retrievability_decision(text, spec)
                    if retrievability == "reject":
                        rows.append(
                            {
                                "demand_group": spec.demand_group,
                                "category": spec.category,
                                "subcategory": spec.subcategory,
                                "instruction": text,
                                "template_source": "llm_rejected",
                                "batch_index": batch_index,
                                "language": payload.get("language") or instruction_language(text),
                                "reject_reason": retrievability_reason,
                            }
                        )
                        continue
                    rows.append(
                        {
                            "demand_group": spec.demand_group,
                            "category": spec.category,
                            "subcategory": spec.subcategory,
                            "instruction": text,
                            "template_source": "llm_generated",
                            "batch_index": batch_index,
                            "language": payload.get("language") or instruction_language(text),
                            "style": payload.get("style"),
                            "evidence_used": payload.get("evidence_used"),
                            "map_retrievability": retrievability,
                            "map_retrievability_reason": retrievability_reason,
                        }
                    )
                return rows
            except Exception as exc:
                errors.append(f"{style}: {exc}")
        return [
            {
                "demand_group": spec.demand_group,
                "category": spec.category,
                "subcategory": spec.subcategory,
                "instruction": "",
                "template_source": "llm_error",
                "batch_index": batch_index,
                "error": " | ".join(errors),
            }
        ]


def balanced_limited_specs(specs: list[DemandSpec], limit: int | None) -> list[DemandSpec]:
    if limit is None or limit >= len(specs):
        return specs
    by_category: dict[str, list[DemandSpec]] = defaultdict(list)
    for spec in specs:
        by_category[spec.category].append(spec)
    category_order = ["implicit_need_navigation", "semantic_preference", "multi_stop_route", "route_update_add_via"]
    selected: list[DemandSpec] = []
    while len(selected) < limit:
        added = False
        for category in category_order:
            bucket = by_category.get(category) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


async def generate_llm_paraphrases(
    *,
    specs: list[DemandSpec],
    candidate_cache: dict[str, list[dict[str, Any]]],
    text_examples: dict[str, list[dict[str, Any]]],
    evidence_cache: dict[str, list[dict[str, Any]]],
    api_keys: list[str],
    llm_concurrency: int,
    llm_limit: int | None,
    llm_demand_groups: set[str] | None,
    instructions_per_spec: int,
    batches_per_spec: int,
    english_ratio: float,
    progress_interval: int,
) -> list[dict[str, Any]]:
    generator = YelpParaphraseGenerator(instructions_per_spec=instructions_per_spec, english_ratio=english_ratio)
    if not generator.is_runtime_available(api_keys):
        raise RuntimeError("LLM paraphrase generation is not configured. Set ARK_API_KEY or LLM_API_KEY, or pass --api-keys-file.")
    assert generator.settings is not None

    jobs = [spec for spec in specs if candidate_cache.get(spec.demand_group)]
    if llm_demand_groups:
        jobs = [spec for spec in jobs if spec.demand_group in llm_demand_groups or spec.subcategory in llm_demand_groups]
    jobs = balanced_limited_specs(jobs, llm_limit)
    batch_jobs = [(spec, batch_index) for spec in jobs for batch_index in range(max(1, batches_per_spec))]

    clients: list[Any] = []
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    try:
        key_pool = api_keys or [generator.settings.api_key]
        for key in key_pool:
            settings = replace(generator.settings, api_key=key)
            clients.append(_create_async_client(settings))  # type: ignore[misc]
        clients = [client for client in clients if client is not None]
        if not clients:
            raise RuntimeError("No usable LLM clients were created.")

        worker_count = max(1, min(llm_concurrency, len(batch_jobs)))
        queue: asyncio.Queue[tuple[DemandSpec, int] | None] = asyncio.Queue()
        for job in batch_jobs:
            queue.put_nowait(job)
        for _ in range(worker_count):
            queue.put_nowait(None)

        completed = 0
        total_jobs = len(batch_jobs)
        progress_every = max(1, progress_interval)
        print(
            f"[yelp-paraphrase] LLM jobs={total_jobs}, specs={len(jobs)}, batches={max(1, batches_per_spec)}, "
            f"workers={worker_count}, api_keys={len(clients)}",
            file=sys.stderr,
            flush=True,
        )

        async def _worker(worker_id: int) -> None:
            nonlocal completed
            client = clients[worker_id % len(clients)]
            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    return
                spec, batch_index = job
                rows = await generator.generate_for_spec(
                    client,
                    spec,
                    candidates=candidate_cache.get(spec.demand_group, []),
                    text_examples=text_examples.get(spec.demand_group, []),
                    evidence_rows=evidence_cache.get(spec.demand_group, []),
                    batch_index=batch_index,
                )
                rows_by_group.setdefault(spec.demand_group, []).extend(rows)
                completed += 1
                if completed == 1 or completed == total_jobs or completed % progress_every == 0:
                    print(f"[yelp-paraphrase] progress {completed}/{total_jobs}", file=sys.stderr, flush=True)
                queue.task_done()

        workers = [asyncio.create_task(_worker(worker_id)) for worker_id in range(worker_count)]
        await queue.join()
        await asyncio.gather(*workers)
    finally:
        if _close_async_client is not None:
            for client in clients:
                try:
                    await asyncio.wait_for(_close_async_client(client), timeout=5.0)
                except Exception:
                    pass

    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.extend(rows_by_group.get(spec.demand_group, []))
    return rows


def merge_paraphrase_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        instruction = str(row.get("instruction") or row.get("text") or "").strip()
        if not instruction:
            continue
        if row.get("template_source") in {"llm_error", "llm_rejected"}:
            continue
        for key in (row.get("demand_group"), row.get("subcategory")):
            if not key:
                continue
            dedupe_key = (str(key), normalize_instruction(instruction))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged[str(key)].append(instruction)
    return dict(merged)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Yelp-guided UrbanNav instruction seeds.")
    parser.add_argument("--yelp-root", required=True, type=Path)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_YELP_TAXONOMY, help="Yelp demand taxonomy JSON config.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-total", type=int, default=400)
    parser.add_argument(
        "--generation-mode",
        choices=["template", "hybrid", "llm_evidence"],
        default="hybrid",
        help=(
            "template: manual/external templates only; hybrid: templates plus optional LLM evidence paraphrases; "
            "llm_evidence: require LLM evidence paraphrase generation."
        ),
    )
    parser.add_argument("--business-limit", type=int)
    parser.add_argument("--review-limit", type=int, default=500_000)
    parser.add_argument("--tip-limit", type=int, default=None)
    parser.add_argument("--min-review-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--english-ratio", type=float, default=0.6, help="Target minimum English instruction ratio.")
    parser.add_argument(
        "--evidence-per-subcategory",
        type=int,
        default=80,
        help="Number of top Yelp demand evidence rows to keep per subcategory.",
    )
    parser.add_argument(
        "--llm-paraphrase-jsonl",
        type=Path,
        help=(
            "Optional JSONL with LLM/LVLM paraphrases. Each row should contain "
            "`instruction` or `text`, plus `demand_group` and/or `subcategory`."
        ),
    )
    parser.add_argument(
        "--llm-paraphrase-mode",
        choices=["none", "generate"],
        default="none",
        help="none: use manual/external templates only; generate: call the configured LLM to generate Yelp-grounded paraphrases.",
    )
    parser.add_argument("--api-keys-file", type=Path, help="Optional API key file, one key per line or comma-separated.")
    parser.add_argument("--llm-concurrency", type=int, default=4, help="Maximum concurrent LLM paraphrase requests.")
    parser.add_argument("--llm-limit", type=int, help="Optional maximum number of subcategories to send to LLM for smoke tests.")
    parser.add_argument(
        "--llm-demand-groups",
        help="Optional comma-separated demand_group/subcategory allowlist for LLM paraphrase generation.",
    )
    parser.add_argument("--llm-paraphrases-per-subcategory", type=int, default=16)
    parser.add_argument(
        "--llm-paraphrase-batches-per-subcategory",
        type=int,
        default=1,
        help="Number of independent LLM paraphrase batches to request for each subcategory. Default: 1.",
    )
    parser.add_argument(
        "--llm-paraphrase-output-jsonl",
        type=Path,
        help="Where to save generated LLM paraphrases. Defaults to output-dir/llm_paraphrases.generated.jsonl.",
    )
    parser.add_argument(
        "--llm-text-example-limit",
        type=int,
        default=40,
        help="Maximum Yelp review/tip snippets kept per demand group for LLM prompting.",
    )
    parser.add_argument("--progress-interval", type=int, default=5, help="Print one LLM progress line after this many jobs.")
    parser.add_argument(
        "--max-template-ratio",
        type=float,
        default=0.15,
        help="Maximum target share for one base template inside a subcategory. Default: 0.15.",
    )
    parser.add_argument(
        "--max-core-repeat-per-subcategory",
        type=int,
        default=12,
        help=(
            "Maximum times the same normalized instruction may appear inside one subcategory. "
            "Use 0 to disable. Default: 12."
        ),
    )
    parser.add_argument(
        "--preserve-output-order",
        action="store_true",
        help="Keep taxonomy generation order in output JSONL files. By default rows are shuffled for review readability.",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=30,
        help="Maximum generation attempts per subcategory quota unit before reporting underfilled. Default: 30.",
    )
    args = parser.parse_args()
    if not 0 < args.max_template_ratio <= 1:
        parser.error("--max-template-ratio must be in (0, 1].")
    if args.max_core_repeat_per_subcategory < 0:
        parser.error("--max-core-repeat-per-subcategory must be >= 0.")
    if args.max_attempts_multiplier < 1:
        parser.error("--max-attempts-multiplier must be >= 1.")
    if args.llm_paraphrase_batches_per_subcategory < 1:
        parser.error("--llm-paraphrase-batches-per-subcategory must be >= 1.")
    if not 0 <= args.english_ratio <= 1:
        parser.error("--english-ratio must be in [0, 1].")
    if args.generation_mode == "llm_evidence":
        args.llm_paraphrase_mode = "generate"
    llm_demand_groups = {
        item.strip()
        for item in (args.llm_demand_groups or "").split(",")
        if item.strip()
    } or None

    rng = random.Random(args.seed)
    specs = load_yelp_demand_taxonomy(args.taxonomy)
    api_keys = load_api_keys(args.api_keys_file)
    if api_keys and not os.getenv("LLM_PROVIDER", "").strip() and api_keys[0].startswith("ark-"):
        os.environ["LLM_PROVIDER"] = "ark"
    if api_keys:
        print(f"[yelp-seed] loaded {len(api_keys)} API keys from {args.api_keys_file}", file=sys.stderr, flush=True)

    business_pool = build_business_pool(args.yelp_root, limit=args.business_limit)
    text_signals = scan_text_signals(
        args.yelp_root,
        specs=specs,
        business_pool=business_pool,
        review_limit=args.review_limit,
        tip_limit=args.tip_limit,
    )
    quotas = quota_for_total(args.target_total, QUOTA_WEIGHTS)
    external_paraphrases = load_external_paraphrases(args.llm_paraphrase_jsonl)
    llm_rows: list[dict[str, Any]] = []

    candidate_cache: dict[str, list[dict[str, Any]]] = {}
    candidate_counts: dict[str, int] = {}
    for spec in specs:
        candidates = candidate_businesses(
            spec,
            business_pool=business_pool,
            text_signals=text_signals,
            min_review_count=args.min_review_count,
        )
        candidate_cache[spec.demand_group] = candidates
        candidate_counts[spec.demand_group] = len(candidates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_examples: dict[str, list[dict[str, Any]]] = {}
    yelp_evidence_rows: list[dict[str, Any]] = []
    evidence_cache: dict[str, list[dict[str, Any]]] = {}
    if args.generation_mode in {"hybrid", "llm_evidence"} or args.llm_paraphrase_mode == "generate":
        text_examples = scan_text_examples(
            args.yelp_root,
            specs=specs,
            business_pool=business_pool,
            review_limit=args.review_limit,
            tip_limit=args.tip_limit,
            max_examples_per_group=args.llm_text_example_limit,
        )
        yelp_evidence_rows = build_yelp_demand_evidence(
            specs=specs,
            candidate_cache=candidate_cache,
            text_examples=text_examples,
            evidence_per_subcategory=args.evidence_per_subcategory,
        )
        evidence_cache = evidence_by_group(yelp_evidence_rows)
        write_jsonl(args.output_dir / "yelp_demand_evidence.jsonl", yelp_evidence_rows)

    if args.llm_paraphrase_mode == "generate":
        llm_rows = asyncio.run(
            generate_llm_paraphrases(
                specs=specs,
                candidate_cache=candidate_cache,
                text_examples=text_examples,
                evidence_cache=evidence_cache,
                api_keys=api_keys,
                llm_concurrency=args.llm_concurrency,
                llm_limit=args.llm_limit,
                llm_demand_groups=llm_demand_groups,
                instructions_per_spec=args.llm_paraphrases_per_subcategory,
                batches_per_spec=args.llm_paraphrase_batches_per_subcategory,
                english_ratio=args.english_ratio,
                progress_interval=args.progress_interval,
            )
        )
        if args.generation_mode == "llm_evidence" and not any(str(row.get("instruction") or "").strip() and row.get("template_source") not in {"llm_error", "llm_rejected"} for row in llm_rows):
            raise RuntimeError("llm_evidence mode produced no valid LLM instructions.")
        llm_output = args.llm_paraphrase_output_jsonl or (args.output_dir / "llm_paraphrases.generated.jsonl")
        write_jsonl(llm_output, llm_rows)
        generated_paraphrases = merge_paraphrase_rows(llm_rows)
        for key, values in generated_paraphrases.items():
            external_paraphrases.setdefault(key, []).extend(values)

    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_instructions: set[str] = set()
    global_index = 0
    template_counts_by_subcategory: dict[str, Counter[str]] = {}
    template_source_counts_by_subcategory: dict[str, Counter[str]] = {}
    templates_per_demand_group: dict[str, int] = {}
    effective_template_caps: dict[str, int] = {}
    core_repeat_counts_by_subcategory: dict[str, Counter[str]] = {}
    language_counts_by_subcategory: dict[str, Counter[str]] = {}
    underfilled_subcategories: dict[str, dict[str, int]] = {}
    map_retrievability_counts: Counter[str] = Counter()

    for spec in specs:
        candidates = candidate_cache[spec.demand_group]
        quota = quotas[spec.demand_group]
        templates_with_source = templates_for_spec(spec, external_paraphrases)
        rng.shuffle(templates_with_source)
        all_templates = [text for text, _source in templates_with_source]
        templates_per_demand_group[spec.demand_group] = len(all_templates)
        requested_template_cap = max(1, math.floor(quota * args.max_template_ratio))
        minimum_template_cap = max(1, math.ceil(quota / max(1, len(all_templates))))
        max_unique_per_template = min((variant_count(text) for text in all_templates), default=1)
        effective_template_cap = min(max(requested_template_cap, minimum_template_cap), max_unique_per_template)
        effective_template_caps[spec.demand_group] = effective_template_cap
        template_usage: Counter[str] = Counter()
        template_source_usage: Counter[str] = Counter()
        language_usage: Counter[str] = Counter()
        core_usage: Counter[str] = Counter()
        template_counts_by_subcategory[spec.subcategory] = template_usage
        template_source_counts_by_subcategory[spec.subcategory] = template_source_usage
        core_repeat_counts_by_subcategory[spec.subcategory] = core_usage
        language_counts_by_subcategory[spec.subcategory] = language_usage
        target_en = math.ceil(quota * args.english_ratio)
        if not candidates:
            for _ in range(quota):
                global_index += 1
                desired_lang = "en" if language_usage["en"] < target_en else "zh"
                pool = [item for item in templates_with_source if instruction_language(item[0]) == desired_lang] or templates_with_source
                template, template_source = rng.choice(pool)
                language_usage[instruction_language(template)] += 1
                rejected.append(
                    make_seed(
                        spec=spec,
                        business=None,
                        instruction=template,
                        index=global_index,
                        status="rejected",
                        reason="no_yelp_candidate_with_minimum_evidence",
                        confidence=0.0,
                        all_templates=all_templates,
                        base_template=template,
                        template_source=template_source,
                    )
                )
            continue

        business_cursor = 0
        attempts = 0
        produced = 0
        template_cursor = 0
        max_attempts = max(quota * args.max_attempts_multiplier, len(templates_with_source) * 20)
        while produced < quota and attempts < max_attempts:
            attempts += 1
            business = candidates[business_cursor % len(candidates)]
            business_cursor += 1
            template = ""
            template_source = ""
            template_key = ""
            desired_lang = "en" if language_usage["en"] < target_en else "zh"
            for _ in range(len(templates_with_source)):
                candidate_template, candidate_source = templates_with_source[template_cursor % len(templates_with_source)]
                template_cursor += 1
                candidate_key = normalize_instruction(candidate_template)
                candidate_lang = instruction_language(candidate_template)
                if candidate_lang != desired_lang and attempts < max(quota * 40, len(templates_with_source) * 8):
                    continue
                if template_usage[candidate_key] < effective_template_cap:
                    template = candidate_template
                    template_source = candidate_source
                    template_key = candidate_key
                    break
            if not template:
                break

            variant_index = template_usage[template_key] % variant_count(template)
            instruction = instruction_variant(template, variant_index=variant_index)
            norm = normalize_instruction(instruction)
            retry_variant = 0
            while norm in used_instructions and retry_variant < variant_count(template):
                retry_variant += 1
                instruction = instruction_variant(template, variant_index=variant_index + retry_variant)
                norm = normalize_instruction(instruction)
            if norm in used_instructions:
                continue
            ok, reject_reason = validate_instruction_text(
                instruction,
                spec=spec,
                business_names={str(business.get("name") or "")},
            )
            if not ok:
                global_index += 1
                rejected.append(
                    make_seed(
                        spec=spec,
                        business=business,
                        instruction=instruction,
                        index=global_index,
                        status="rejected",
                        reason=f"instruction_validation_failed:{reject_reason}",
                        confidence=0.0,
                        all_templates=all_templates,
                        base_template=template,
                        template_source=template_source,
                    )
                )
                continue
            retrievability, retrievability_reason = map_retrievability_decision(instruction, spec)
            map_retrievability_counts[retrievability_reason] += 1
            if retrievability == "reject":
                global_index += 1
                rejected.append(
                    make_seed(
                        spec=spec,
                        business=business,
                        instruction=instruction,
                        index=global_index,
                        status="rejected",
                        reason=retrievability_reason,
                        confidence=0.0,
                        all_templates=all_templates,
                        base_template=template,
                        template_source=template_source,
                    )
                )
                continue
            core_key = core_instruction_key(instruction)
            if args.max_core_repeat_per_subcategory and core_usage[core_key] >= args.max_core_repeat_per_subcategory:
                continue
            used_instructions.add(norm)
            core_usage[core_key] += 1
            template_usage[template_key] += 1
            template_source_usage[template_source] += 1
            language_usage[instruction_language(instruction)] += 1
            global_index += 1
            evidence = business.get("evidence", {})
            evidence_count = int(evidence.get("review_signal_count", 0)) + int(evidence.get("tip_signal_count", 0))
            confidence = 0.72
            if evidence.get("attribute_hit"):
                confidence += 0.08
            if evidence.get("category_hit"):
                confidence += 0.06
            if evidence_count >= 2:
                confidence += 0.08
            status = "needs_review" if spec.review_by_default or evidence_count == 0 or retrievability == "review" else "accepted"
            if retrievability == "review":
                reason = retrievability_reason
            else:
                reason = "semantic_preference_or_weak_evidence_review" if status == "needs_review" else "yelp_evidence_and_template"
            row = make_seed(
                spec=spec,
                business=business,
                instruction=instruction,
                index=global_index,
                status=status,
                reason=reason,
                confidence=min(confidence, 0.95),
                all_templates=all_templates,
                base_template=template,
                template_source=template_source,
            )
            if status == "accepted":
                accepted.append(row)
            else:
                review.append(row)
            produced += 1
        if produced < quota:
            underfilled_subcategories[spec.subcategory] = {"produced": produced, "quota": quota}

    if not args.preserve_output_order:
        rng.shuffle(accepted)
        rng.shuffle(review)
        rng.shuffle(rejected)

    write_jsonl(args.output_dir / "task_instructions.accepted.jsonl", accepted)
    write_jsonl(args.output_dir / "task_instructions.review.jsonl", review)
    write_jsonl(args.output_dir / "task_instructions.reject.jsonl", rejected)

    all_rows = accepted + review + rejected
    summary = {
        "dataset": "yelp_instruction_seed",
        "generation_mode": args.generation_mode,
        "taxonomy": str(args.taxonomy),
        "target_total": args.target_total,
        "generated_total": len(all_rows),
        "accepted": len(accepted),
        "review": len(review),
        "rejected": len(rejected),
        "business_total_loaded": len(business_pool),
        "review_limit": args.review_limit,
        "tip_limit": args.tip_limit,
        "llm_paraphrase_jsonl": str(args.llm_paraphrase_jsonl) if args.llm_paraphrase_jsonl else None,
        "llm_paraphrase_mode": args.llm_paraphrase_mode,
        "llm_generated_rows": len(llm_rows),
        "llm_generated_valid_instructions": sum(1 for row in llm_rows if row.get("template_source") == "llm_generated"),
        "llm_concurrency": args.llm_concurrency if args.llm_paraphrase_mode == "generate" else None,
        "llm_paraphrase_batches_per_subcategory": (
            args.llm_paraphrase_batches_per_subcategory if args.llm_paraphrase_mode == "generate" else None
        ),
        "llm_demand_groups": sorted(llm_demand_groups) if llm_demand_groups else None,
        "english_ratio_target": args.english_ratio,
        "yelp_demand_evidence_rows": len(yelp_evidence_rows),
        "api_keys_loaded": len(api_keys),
        "max_template_ratio_requested": args.max_template_ratio,
        "max_core_repeat_per_subcategory": args.max_core_repeat_per_subcategory,
        "effective_template_caps": effective_template_caps,
        "quotas": quotas,
        "candidate_counts": candidate_counts,
        "templates_per_demand_group": templates_per_demand_group,
        "template_source_counts_by_subcategory": {
            key: dict(counter) for key, counter in template_source_counts_by_subcategory.items()
        },
        "language_counts": dict(Counter(row.get("language") or instruction_language(row.get("instruction") or "") for row in all_rows)),
        "english_ratio_actual": (
            Counter(row.get("language") or instruction_language(row.get("instruction") or "") for row in all_rows).get("en", 0)
            / len(all_rows)
            if all_rows
            else 0.0
        ),
        "language_by_subcategory": {key: dict(counter) for key, counter in language_counts_by_subcategory.items()},
        "underfilled_subcategories": underfilled_subcategories,
        "map_retrievability_filter_counts": dict(map_retrievability_counts),
        "max_template_share_by_subcategory": {
            key: (max(counter.values()) / sum(counter.values()) if counter else 0.0)
            for key, counter in template_counts_by_subcategory.items()
        },
        "max_core_repeat_by_subcategory": {
            key: (max(counter.values()) if counter else 0)
            for key, counter in core_repeat_counts_by_subcategory.items()
        },
        "unique_core_instructions_by_subcategory": {
            key: len(counter)
            for key, counter in core_repeat_counts_by_subcategory.items()
        },
        "by_category": dict(Counter(row.get("category") or "none" for row in all_rows)),
        "by_subcategory": dict(Counter(row.get("subcategory") or "none" for row in all_rows)),
        "by_status": dict(Counter(row.get("conversion_status") or "none" for row in all_rows)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
