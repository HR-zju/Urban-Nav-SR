#!/usr/bin/env python3
"""Convert CitySeeker rows into UrbanNav instruction seeds or task GT rows.

The preferred path is LVLM/LLM-assisted classification using the RouteServer
OpenAI-compatible configuration (``LLM_PROVIDER``, ``ARK_API_KEY`` or
``LLM_API_KEY``, ``LLM_MODEL``). Deterministic rules are still kept as priors and
offline fallback so the conversion remains reproducible when no model endpoint
is configured.

For UrbanNav dataset construction, prefer ``--instruction-only``. It transfers
CitySeeker demand instructions and taxonomy labels only; destination and route
GT should then be rebuilt with the target POI provider and RouteServer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_TAXONOMY = Path(__file__).resolve().parents[2] / "config" / "task_taxonomy_cityseeker.json"
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
except Exception:  # pragma: no cover - optional dependency path for offline rule mode.
    LLMSettings = None
    _close_async_client = None
    _create_async_client = None
    _extract_text_from_chat_response = None
    _extract_text_from_responses_response = None
    _parse_json_object_from_text = None


@dataclass(frozen=True)
class RuleResult:
    category: str | None
    category_zh: str | None
    subcategory: str | None
    confidence: float
    status: str
    reason: str
    raw_llm: dict[str, Any] | None = None


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "records", "cases"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    raise ValueError(f"Unsupported input JSON root: {type(data)!r}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def normalize_key(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("&", " and ")
    value = value.replace("/", " ")
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def normalize_phrase(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def first_template(item: dict[str, Any], *, prefer_cjk: bool) -> str | None:
    templates = item.get("templates") or []
    if not isinstance(templates, list):
        return None
    for template in templates:
        has_cjk = bool(re.search(r"[\u3400-\u9fff]", str(template)))
        if has_cjk == prefer_cjk:
            return str(template)
    return str(templates[0]) if templates else None


def build_taxonomy_indexes(taxonomy: dict[str, Any]) -> dict[str, Any]:
    categories = taxonomy.get("categories") or []
    by_category = {}
    by_subcategory = {}
    by_source_and_subcategory = {}
    by_source_category = defaultdict(list)

    for category in categories:
        category_id = str(category.get("category") or "").strip()
        if not category_id:
            continue
        by_category[category_id] = category
        for item in category.get("items") or []:
            subcategory = str(item.get("subcategory") or "").strip()
            if not subcategory:
                continue
            item_payload = {
                "category": category,
                "item": item,
            }
            by_subcategory[normalize_key(subcategory)] = item_payload
            source = str(item.get("source_cityseeker_category") or category.get("cityseeker_category") or "").strip()
            if source:
                by_source_and_subcategory[(source, normalize_key(subcategory))] = item_payload
                by_source_category[source].append(item_payload)

    return {
        "by_category": by_category,
        "by_subcategory": by_subcategory,
        "by_source_and_subcategory": by_source_and_subcategory,
        "by_source_category": by_source_category,
    }


def explicit_folder_alias(folder: str) -> str:
    aliases = {
        "healthcare facility": "healthcare_facility",
        "primary and secondary school": "school",
        "shopping mall": "shopping_mall",
        "convenience store": "convenience_store",
        "coffee shop": "coffee_shop",
        "parking lot": "parking_lot",
        "bus station": "bus_station",
        "subway station": "subway_station",
        "KFC or McDonald's": "kfc_or_mcdonalds",
        "KFC or McDonald's or Five Guys": "kfc_or_mcdonalds",
        "Bank of China": "bank_of_china",
        "Bank of China or Hang Seng Bank": "bank_of_china",
        "Apple or Huawei": "apple_or_huawei",
        "Apple": "apple_or_huawei",
        "FamilyMart, Lawson, Bianlifeng, or 7-Eleven": "7_eleven_or_familymart",
        "7-Eleven": "7_eleven_or_familymart",
        "7-Eleven or Duane Reade": "drugstore_chain",
        "Starbucks": "starbucks",
        "Adidas or Nike": "sportswear_store",
        "Adidas or Nike or Lululemon": "sportswear_store",
        "Foot Locker or Lululemon or New Balance": "sportswear_store",
        "ParknShop百佳 or Wellcome惠康 supermarket": "supermarket_chain",
        "Tesco or Sainsbury": "supermarket_chain",
        "Watsons or Mannings": "drugstore_chain",
        "Zara or H&M or Chanel": "fashion_store",
        "Best Buy or P.C. Richard & Son": "electronics_store_chain",
        "Lloyds Bank or Barclays Bank": "bank_chain",
        "Chase Bank": "bank_chain",
    }
    return aliases.get(folder, normalize_key(folder))


def implicit_folder_alias(folder: str) -> str:
    aliases = {
        "I am hungry": "hungry",
        "I am thirsty": "thirsty",
        "I want to bring my child to play": "child_play",
        "My phone balance is low and I need to recharge it": "phone_battery_or_recharge",
        "I need to go to the airport": "",
        "I am not feeling well": "not_feeling_well",
        "I want to exercise": "gym",
        "I want to find a place to work with Wi-Fi": "work_with_wifi",
        "I want to buy fruits and vegetables": "buy_fruits_vegetables",
        "I want to rest and read": "rest_and_read",
        "coffee shop": "coffee_shop",
        "parking lot": "parking_lot",
        "ATM": "atm",
        "restroom": "restroom",
        "cinema": "cinema",
        "gym": "gym",
    }
    return aliases.get(folder, normalize_key(folder))


def semantic_folder_alias(folder: str) -> str:
    aliases = {
        "restaurant with a comfortable environment": "comfortable_restaurant",
        "group or family-friendly restaurant": "family_friendly_restaurant",
        "restaurant with roadside parking": "restaurant_with_parking",
        "upscale restaurant": "upscale_restaurant",
        "restaurant with outdoor seating": "outdoor_seating_restaurant",
        "romantic restaurant": "romantic_restaurant",
    }
    return aliases.get(folder, normalize_key(folder))


def classify_rule(record: dict[str, Any], indexes: dict[str, Any]) -> RuleResult:
    cityseeker_category = str(record.get("category") or "").strip()
    folder = str(record.get("folder") or "").strip()
    question = str(record.get("question") or "").strip()

    by_subcategory = indexes["by_subcategory"]

    if cityseeker_category in {"question5", "question8"}:
        return RuleResult(None, None, None, 1.0, "rejected", f"{cityseeker_category}_removed_from_taxonomy")

    if cityseeker_category in {"question1", "question4"}:
        subcategory = explicit_folder_alias(folder)
        found = by_subcategory.get(normalize_key(subcategory))
        if found and found["category"].get("category") == "poi_navigation":
            return RuleResult("poi_navigation", "POI/品牌导航", found["item"]["subcategory"], 0.95, "accepted", "rule_exact_or_alias")
        return RuleResult("poi_navigation", "POI/品牌导航", subcategory or None, 0.45, "needs_review", "poi_subcategory_not_in_taxonomy")

    if cityseeker_category in {"question6", "question7"}:
        subcategory = implicit_folder_alias(folder)
        if not subcategory:
            return RuleResult(None, None, None, 0.9, "rejected", f"{folder}_not_in_current_scope")
        found = by_subcategory.get(normalize_key(subcategory))
        if found and found["category"].get("category") == "implicit_need_navigation":
            return RuleResult("implicit_need_navigation", "隐式需求导航", found["item"]["subcategory"], 0.9, "accepted", "rule_exact_or_alias")
        return RuleResult("implicit_need_navigation", "隐式需求导航", subcategory or None, 0.45, "needs_review", "implicit_subcategory_not_in_taxonomy")

    if cityseeker_category == "question2":
        parts = [normalize_key(part) for part in normalize_phrase(folder).split(" and ") if part.strip()]
        subcategory = "_then_".join(parts) if len(parts) >= 2 else normalize_key(folder)
        found = by_subcategory.get(normalize_key(subcategory))
        resolved = found["item"]["subcategory"] if found and found["category"].get("category") == "multi_stop_route" else subcategory
        return RuleResult(
            "multi_stop_route",
            "多目标导航",
            resolved,
            0.65,
            "needs_review",
            "cityseeker_question2_has_no_explicit_intermediate_stop_gt",
        )

    if cityseeker_category == "question9":
        subcategory = semantic_folder_alias(folder)
        found = by_subcategory.get(normalize_key(subcategory))
        resolved = found["item"]["subcategory"] if found and found["category"].get("category") == "semantic_preference" else subcategory
        return RuleResult(
            "semantic_preference",
            "语义偏好导航",
            resolved,
            0.7,
            "needs_review",
            "semantic_preference_requires_attribute_or_human_review",
        )

    return RuleResult(None, None, None, 0.0, "needs_review", f"unknown_cityseeker_category:{cityseeker_category}; question={question[:80]}")


def point_from_route_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "lat": item.get("latitude"),
        "lng": item.get("longitude"),
        "node_id": item.get("filename"),
        "filename": item.get("filename"),
    }


def convert_record(
    record: dict[str, Any],
    *,
    classification: RuleResult,
    taxonomy_indexes: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    route = record.get("complete_route") or []
    route_points = [point_from_route_item(item) for item in route if isinstance(item, dict)]
    origin = route_points[0] if route_points else {"node_id": record.get("from")}
    destination = route_points[-1] if route_points else {"node_id": record.get("to")}
    destination.update(
        {
            "name": record.get("folder"),
            "type": classification.subcategory or record.get("folder"),
            "provider_poi_id": record.get("to"),
        }
    )

    category_payload = taxonomy_indexes["by_category"].get(classification.category or "") or {}
    item_payload = taxonomy_indexes["by_subcategory"].get(normalize_key(classification.subcategory or "")) or {}
    item = item_payload.get("item") or {}

    case_id = "cityseeker_{city}_{qidx}_{idx}".format(
        city=normalize_key(record.get("city")),
        qidx=record.get("question_idx", "na"),
        idx=record.get("idx", "na"),
    )

    converted = {
        "id": case_id,
        "dataset": "cityseeker_converted",
        "conversion_status": classification.status,
        "conversion_reason": classification.reason,
        "conversion_confidence": classification.confidence,
        "city": record.get("city"),
        "scene_id": record.get("city"),
        "category": classification.category,
        "category_zh": classification.category_zh or category_payload.get("category_zh"),
        "subcategory": classification.subcategory,
        "source_cityseeker_category": record.get("category"),
        "source_cityseeker_folder": record.get("folder"),
        "instruction": record.get("question"),
        "instruction_templates": item.get("templates") or [record.get("question")],
        "origin": origin,
        "gt_destination": destination,
        "gt_route": route_points,
        "gt_waypoints": [
            {
                "coordinate": {"latitude": point.get("lat"), "longitude": point.get("lng")},
                "node_id": point.get("node_id"),
                "action": "unknown",
                "distance_m": None,
            }
            for point in route_points
        ],
        "total_weight": record.get("total_weight"),
        "total_steps": record.get("total_steps"),
        "difficulty": record.get("difficulty"),
        "source": {
            "name": source_name,
            "format": "CitySeeker",
            "question_idx": record.get("question_idx"),
            "idx": record.get("idx"),
            "from": record.get("from"),
            "to": record.get("to"),
            "taxonomy": "cityseeker_refactored_v2",
        },
        "raw_cityseeker": record,
    }

    if classification.category == "multi_stop_route":
        converted["gt_destination_sequence"] = []
        converted["gt_route_legs"] = []
        converted["multi_stop_gt_status"] = "missing_intermediate_stop_gt_in_cityseeker_json"

    return converted


def convert_instruction_seed(
    record: dict[str, Any],
    *,
    classification: RuleResult,
    taxonomy_indexes: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    category_payload = taxonomy_indexes["by_category"].get(classification.category or "") or {}
    item_payload = taxonomy_indexes["by_subcategory"].get(normalize_key(classification.subcategory or "")) or {}
    item = item_payload.get("item") or {}

    case_id = "cityseeker_instruction_{city}_{qidx}_{idx}".format(
        city=normalize_key(record.get("city")),
        qidx=record.get("question_idx", "na"),
        idx=record.get("idx", "na"),
    )

    seed = {
        "id": case_id,
        "dataset": "cityseeker_instruction_seed",
        "conversion_status": classification.status,
        "conversion_reason": classification.reason,
        "conversion_confidence": classification.confidence,
        "category": classification.category,
        "category_zh": classification.category_zh or category_payload.get("category_zh"),
        "subcategory": classification.subcategory,
        "instruction": record.get("question"),
        "instruction_templates": item.get("templates") or [record.get("question")],
        "target_types": item.get("target_types"),
        "ordered_target_types": item.get("ordered_target_types"),
        "brand_aliases": item.get("brand_aliases"),
        "semantic_tags": item.get("semantic_tags"),
        "source_cityseeker_category": record.get("category"),
        "source_cityseeker_folder": record.get("folder"),
        "source": {
            "name": source_name,
            "format": "CitySeeker",
            "city": record.get("city"),
            "question_idx": record.get("question_idx"),
            "idx": record.get("idx"),
            "taxonomy": "cityseeker_refactored_v2",
            "gt_policy": "do_not_reuse_cityseeker_to_or_complete_route",
        },
        "gt_build_plan": {
            "origin": "sample_from_target_urban_nav_region",
            "gt_destination": "rebuild_with_poi_provider",
            "gt_route": "rebuild_with_routeserver_or_graph_shortest_path",
            "ranking_policy": "nearest_satisfying_poi_by_route_distance",
        },
    }
    if classification.raw_llm is not None:
        seed["llm_classification"] = classification.raw_llm
    return seed


def llm_review_prompt(row: dict[str, Any], allowed_categories: list[dict[str, Any]]) -> dict[str, Any]:
    compact_categories = [
        {
            "category": item.get("category"),
            "category_zh": item.get("category_zh"),
            "description": item.get("description"),
            "subcategories": [sub.get("subcategory") for sub in item.get("items", [])],
        }
        for item in allowed_categories
    ]
    return {
        "id": row["id"],
        "task": "classify_or_reject_cityseeker_instruction",
        "instruction": row.get("instruction"),
        "source_cityseeker_category": row.get("source_cityseeker_category"),
        "source_cityseeker_folder": row.get("source_cityseeker_folder"),
        "rule_prediction": {
            "category": row.get("category"),
            "subcategory": row.get("subcategory"),
            "status": row.get("conversion_status"),
            "reason": row.get("conversion_reason"),
        },
        "allowed_categories": compact_categories,
        "expected_json_schema": {
            "status": "accepted|needs_review|rejected",
            "category": "one allowed category or null",
            "subcategory": "one allowed subcategory or new proposed subcategory",
            "reason": "short reason",
        },
    }


def compact_taxonomy_for_prompt(taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for category in taxonomy.get("categories") or []:
        compact.append(
            {
                "category": category.get("category"),
                "category_zh": category.get("category_zh"),
                "cityseeker_category": category.get("cityseeker_category"),
                "description": category.get("description"),
                "subcategories": [
                    {
                        "subcategory": item.get("subcategory"),
                        "target_types": item.get("target_types"),
                        "ordered_target_types": item.get("ordered_target_types"),
                        "brand_aliases": item.get("brand_aliases"),
                        "semantic_tags": item.get("semantic_tags"),
                        "templates": item.get("templates"),
                    }
                    for item in category.get("items", [])
                ],
            }
        )
    return compact


def category_zh_for(category_id: str | None, indexes: dict[str, Any]) -> str | None:
    if not category_id:
        return None
    category = indexes["by_category"].get(category_id) or {}
    return category.get("category_zh")


def normalize_llm_decision(
    parsed: dict[str, Any],
    *,
    rule_result: RuleResult,
    indexes: dict[str, Any],
) -> RuleResult:
    status = str(parsed.get("status") or "").strip().lower()
    if status not in {"accepted", "needs_review", "rejected"}:
        status = "needs_review"

    category = parsed.get("category")
    category = str(category).strip() if category is not None else None
    if category in {"", "null", "none"}:
        category = None

    subcategory = parsed.get("subcategory")
    subcategory = str(subcategory).strip() if subcategory is not None else None
    if subcategory in {"", "null", "none"}:
        subcategory = None

    reason = str(parsed.get("reason") or "llm_classification").strip()
    confidence_raw = parsed.get("confidence", rule_result.confidence)
    try:
        confidence = max(0.0, min(float(confidence_raw), 1.0))
    except Exception:
        confidence = rule_result.confidence

    if category and category not in indexes["by_category"]:
        return RuleResult(
            None,
            None,
            subcategory,
            min(confidence, 0.4),
            "needs_review",
            f"llm_unknown_category:{category}; {reason}",
            raw_llm=parsed,
        )

    if status == "accepted":
        found = indexes["by_subcategory"].get(normalize_key(subcategory or ""))
        if not category or not subcategory or not found or found["category"].get("category") != category:
            return RuleResult(
                category,
                category_zh_for(category, indexes),
                subcategory,
                min(confidence, 0.5),
                "needs_review",
                f"llm_accepted_invalid_or_unknown_subcategory:{subcategory}; {reason}",
                raw_llm=parsed,
            )

    return RuleResult(
        category,
        category_zh_for(category, indexes),
        subcategory,
        confidence,
        status,
        f"llm:{reason}",
        raw_llm=parsed,
    )


class CitySeekerTaskClassifier:
    """LLM classifier for CitySeeker -> UrbanNav task taxonomy conversion."""

    def __init__(self, taxonomy: dict[str, Any], indexes: dict[str, Any]) -> None:
        self.taxonomy = taxonomy
        self.indexes = indexes
        self.compact_taxonomy = compact_taxonomy_for_prompt(taxonomy)
        self.settings = LLMSettings.from_env() if LLMSettings is not None else None
        if self.settings is not None:
            timeout_s = float(os.getenv("LLM_CITYSEEKER_TIMEOUT_S", "20") or 20)
            max_retries = max(0, int(os.getenv("LLM_CITYSEEKER_MAX_RETRIES", "1") or 1))
            self.settings = replace(self.settings, timeout_s=timeout_s, max_retries=max_retries)

    def is_available(self) -> bool:
        return bool(
            self.settings is not None
            and _create_async_client is not None
            and _parse_json_object_from_text is not None
            and self.settings.api_key
        )

    def is_runtime_available(self, api_keys: list[str] | None = None) -> bool:
        return bool(
            self.settings is not None
            and _create_async_client is not None
            and _parse_json_object_from_text is not None
            and (self.settings.api_key or api_keys)
        )

    def build_messages(self, record: dict[str, Any], rule_result: RuleResult) -> list[dict[str, str]]:
        payload = {
            "instruction": record.get("question"),
            "cityseeker_category": record.get("category"),
            "cityseeker_folder": record.get("folder"),
            "city": record.get("city"),
            "route_fields_available": {
                "from": bool(record.get("from")),
                "to": bool(record.get("to")),
                "complete_route": bool(record.get("complete_route")),
                "total_weight": bool(record.get("total_weight")),
                "difficulty": bool(record.get("difficulty")),
            },
            "rule_prior": {
                "status": rule_result.status,
                "category": rule_result.category,
                "subcategory": rule_result.subcategory,
                "reason": rule_result.reason,
                "confidence": rule_result.confidence,
            },
            "allowed_taxonomy": self.compact_taxonomy,
        }
        system = (
            "You classify CitySeeker navigation dataset rows into the UrbanNav task taxonomy. "
            "Use the instruction semantics first, and use CitySeeker category/folder only as weak metadata. "
            "Return exactly one compact JSON object, no markdown. "
            "Schema: {\"status\":\"accepted|needs_review|rejected\", "
            "\"category\":\"one allowed category or null\", "
            "\"subcategory\":\"one allowed subcategory or null\", "
            "\"confidence\":0.0-1.0, \"reason\":\"short reason\"}. "
            "Accept only if the row fits one allowed category/subcategory and CitySeeker provides enough GT for that task. "
            "For multi_stop_route, if the row has only final to/complete_route but no explicit intermediate stop GT, "
            "use needs_review. For semantic_preference, use needs_review unless the provided metadata is sufficient to "
            "verify the subjective attribute. Reject transport-only, accessibility-only, or out-of-scope rows."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]

    async def _call_chat_completions(self, client: Any, record: dict[str, Any], rule_result: RuleResult) -> str:
        assert self.settings is not None
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": self.build_messages(record, rule_result),
            "max_tokens": int(os.getenv("LLM_CITYSEEKER_CLASSIFIER_MAX_TOKENS", "384") or 384),
        }
        if self.settings.response_format and self.settings.provider != "ark":
            payload["response_format"] = {"type": "json_object"}
        response = await client.chat.completions.create(**payload)
        return _extract_text_from_chat_response(response)  # type: ignore[misc]

    async def _call_responses_api(self, client: Any, record: dict[str, Any], rule_result: RuleResult) -> str:
        assert self.settings is not None
        prompt = "\n\n".join(message["content"] for message in self.build_messages(record, rule_result))
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": int(os.getenv("LLM_CITYSEEKER_CLASSIFIER_MAX_TOKENS", "384") or 384),
        }
        response = await client.responses.create(**payload)
        return _extract_text_from_responses_response(response)  # type: ignore[misc]

    async def classify(self, client: Any, record: dict[str, Any], rule_result: RuleResult) -> RuleResult:
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
                    await self._call_responses_api(client, record, rule_result)
                    if style == "responses"
                    else await self._call_chat_completions(client, record, rule_result)
                )
                parsed = _parse_json_object_from_text(content)  # type: ignore[misc]
                return normalize_llm_decision(parsed, rule_result=rule_result, indexes=self.indexes)
            except Exception as exc:
                errors.append(f"{style}: {exc}")
        return RuleResult(
            rule_result.category,
            rule_result.category_zh,
            rule_result.subcategory,
            min(rule_result.confidence, 0.4),
            "needs_review",
            "llm_failed:" + " | ".join(errors),
            raw_llm={"errors": errors},
        )


async def convert_records(
    records: list[dict[str, Any]],
    *,
    taxonomy: dict[str, Any],
    indexes: dict[str, Any],
    source_name: str,
    classification_mode: str,
    llm_limit: int | None,
    instruction_only: bool,
    api_keys: list[str],
    llm_concurrency: int,
    progress_interval: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rows_by_index: list[dict[str, Any] | None] = [None] * len(records)

    classifier = CitySeekerTaskClassifier(taxonomy, indexes)
    use_llm = classification_mode in {"llm", "hybrid", "llm_review_only"}
    if use_llm and not classifier.is_runtime_available(api_keys):
        if classification_mode == "llm":
            raise RuntimeError("LLM classifier is not configured. Set ARK_API_KEY or LLM_API_KEY, or use --classification-mode rule.")
        print("[convert] LLM classifier unavailable; falling back to rule mode.", file=sys.stderr)
        use_llm = False

    def _should_call_llm(rule_result: RuleResult) -> bool:
        if not use_llm:
            return False
        if classification_mode == "llm":
            return True
        if classification_mode == "hybrid":
            return rule_result.status != "accepted"
        if classification_mode == "llm_review_only":
            return rule_result.status == "needs_review"
        return False

    def _row_from_classification(record: dict[str, Any], classification: RuleResult) -> dict[str, Any]:
        if instruction_only:
            row = convert_instruction_seed(
                record,
                classification=classification,
                taxonomy_indexes=indexes,
                source_name=source_name,
            )
        else:
            row = convert_record(record, classification=classification, taxonomy_indexes=indexes, source_name=source_name)
        if classification.raw_llm is not None and "llm_classification" not in row:
            row["llm_classification"] = classification.raw_llm
        return row

    rule_results: list[RuleResult] = []
    llm_jobs: list[tuple[int, dict[str, Any], RuleResult]] = []
    for index, record in enumerate(records):
        rule_result = classify_rule(record, indexes)
        rule_results.append(rule_result)
        if _should_call_llm(rule_result) and (llm_limit is None or len(llm_jobs) < llm_limit):
            llm_jobs.append((index, record, rule_result))
        else:
            rows_by_index[index] = _row_from_classification(record, rule_result)

    clients: list[Any] = []
    try:
        if llm_jobs and use_llm:
            assert classifier.settings is not None
            key_pool = api_keys or [classifier.settings.api_key]
            for key in key_pool:
                settings = replace(classifier.settings, api_key=key)
                clients.append(_create_async_client(settings))  # type: ignore[misc]
            clients = [client for client in clients if client is not None]
            if not clients:
                raise RuntimeError("No usable LLM clients were created.")

            worker_count = max(1, min(llm_concurrency, len(llm_jobs)))
            queue: asyncio.Queue[tuple[int, dict[str, Any], RuleResult] | None] = asyncio.Queue()
            for job in llm_jobs:
                queue.put_nowait(job)
            for _ in range(worker_count):
                queue.put_nowait(None)

            completed = 0
            total_jobs = len(llm_jobs)
            progress_every = max(1, progress_interval)
            print(
                f"[convert] LLM jobs={total_jobs}, workers={worker_count}, api_keys={len(clients)}, mode={classification_mode}",
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
                    index, record, rule_result = job
                    classification = await classifier.classify(client, record, rule_result)
                    rows_by_index[index] = _row_from_classification(record, classification)
                    completed += 1
                    if completed == 1 or completed == total_jobs or completed % progress_every == 0:
                        print(
                            f"[convert] LLM progress {completed}/{total_jobs}",
                            file=sys.stderr,
                            flush=True,
                        )
                    queue.task_done()

            workers = [asyncio.create_task(_worker(worker_id)) for worker_id in range(worker_count)]
            await queue.join()
            await asyncio.gather(*workers)

        missing = [index for index, row in enumerate(rows_by_index) if row is None]
        if missing:
            raise RuntimeError(f"Internal error: {len(missing)} rows were not converted; first missing index={missing[0]}")

        for row in rows_by_index:
            assert row is not None
            if row["conversion_status"] == "accepted":
                accepted.append(row)
            elif row["conversion_status"] == "rejected":
                rejected.append(row)
            else:
                review.append(row)
    finally:
        if _close_async_client is not None:
            for client in clients:
                try:
                    await asyncio.wait_for(_close_async_client(client), timeout=5.0)
                except Exception:
                    pass

    return accepted, review, rejected


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Convert CitySeeker GT JSON to UrbanNav task GT JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="CitySeeker JSON/JSONL file.")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY, help="UrbanNav taxonomy JSON.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Converted accepted rows JSONL.")
    parser.add_argument("--review-jsonl", required=True, type=Path, help="Rows needing review JSONL.")
    parser.add_argument("--reject-jsonl", required=True, type=Path, help="Rejected rows JSONL.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional conversion summary JSON.")
    parser.add_argument("--llm-review-jsonl", type=Path, default=None, help="Optional LLM handoff prompts for review rows.")
    parser.add_argument("--include-review-in-output", action="store_true", help="Also include needs_review rows in output-jsonl.")
    parser.add_argument("--source-name", default="CitySeeker", help="Source dataset name.")
    parser.add_argument(
        "--classification-mode",
        choices=["rule", "hybrid", "llm", "llm_review_only"],
        default="hybrid",
        help=(
            "rule: deterministic only; hybrid: LLM only for rule review/reject rows; "
            "llm: LLM classifies every row; llm_review_only: LLM only for rule needs_review rows."
        ),
    )
    parser.add_argument("--llm-limit", type=int, default=None, help="Optional maximum number of LLM classifier calls for smoke tests.")
    parser.add_argument("--api-keys-file", type=Path, default=None, help="Optional file containing one or more API keys, one per line or comma-separated.")
    parser.add_argument("--llm-concurrency", type=int, default=4, help="Maximum concurrent LLM classifier requests.")
    parser.add_argument("--progress-interval", type=int, default=25, help="Print one progress line after this many completed LLM jobs.")
    parser.add_argument(
        "--instruction-only",
        action="store_true",
        help="Output instruction seeds only. Do not transfer CitySeeker to/complete_route as UrbanNav GT.",
    )
    args = parser.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    indexes = build_taxonomy_indexes(taxonomy)
    records = load_records(args.input)
    api_keys = load_api_keys(args.api_keys_file)
    if api_keys and not os.getenv("LLM_PROVIDER", "").strip() and api_keys[0].startswith("ark-"):
        os.environ["LLM_PROVIDER"] = "ark"
    if api_keys:
        print(f"[convert] loaded {len(api_keys)} API keys from {args.api_keys_file}", file=sys.stderr, flush=True)

    accepted, review, rejected = await convert_records(
        records,
        taxonomy=taxonomy,
        indexes=indexes,
        source_name=args.source_name,
        classification_mode=args.classification_mode,
        llm_limit=args.llm_limit,
        instruction_only=args.instruction_only,
        api_keys=api_keys,
        llm_concurrency=args.llm_concurrency,
        progress_interval=args.progress_interval,
    )

    output_rows = accepted + review if args.include_review_in_output else accepted
    write_jsonl(args.output_jsonl, output_rows)
    write_jsonl(args.review_jsonl, review)
    write_jsonl(args.reject_jsonl, rejected)

    if args.llm_review_jsonl:
        categories = taxonomy.get("categories") or []
        write_jsonl(args.llm_review_jsonl, [llm_review_prompt(row, categories) for row in review])

    summary = {
        "input": str(args.input),
        "taxonomy": str(args.taxonomy),
        "classification_mode": args.classification_mode,
        "instruction_only": args.instruction_only,
        "llm_concurrency": args.llm_concurrency,
        "api_keys_loaded": len(api_keys),
        "total": len(records),
        "output_rows": len(output_rows),
        "accepted": len(accepted),
        "needs_review": len(review),
        "rejected": len(rejected),
        "by_status": dict(Counter(row["conversion_status"] for row in accepted + review + rejected)),
        "by_category": dict(Counter(row.get("category") or "none" for row in accepted + review + rejected)),
        "by_source_cityseeker_category": dict(Counter(str(row.get("source_cityseeker_category")) for row in accepted + review + rejected)),
        "review_reasons": dict(Counter(row.get("conversion_reason") for row in review)),
        "reject_reasons": dict(Counter(row.get("conversion_reason") for row in rejected)),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
