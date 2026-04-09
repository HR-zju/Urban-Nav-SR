"""Destination recommendation orchestration."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
import json
import math
import os
import re
from time import perf_counter

from mcp_server.config import (
    default_intent_rules,
    default_scoring_weights,
    split_search_keywords,
    load_json_env,
)
from mcp_server.llm import (
    IntentParser,
    extract_explicit_destination_query,
    infer_destination_kind,
    rerank_candidates_with_llm,
    translate_instruction_for_cn_map,
    translate_result_bundle_to_english,
)
from mcp_server.map_providers import MapProviderClient


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def extract_key_waypoints(route_data: dict[str, Any], max_points: int = 12) -> list[dict[str, Any]]:
    coarse_waypoints = route_data.get("coarse_waypoints", [])
    if coarse_waypoints:
        extracted = []
        for item in coarse_waypoints:
            coordinate = item.get("coordinate", {}) or {}
            longitude = coordinate.get("longitude")
            latitude = coordinate.get("latitude")
            if longitude is None or latitude is None:
                location_text = item.get("location", "")
                try:
                    lon_str, lat_str = str(location_text).split(",", 1)
                    longitude = float(lon_str)
                    latitude = float(lat_str)
                except Exception:
                    continue
            extracted.append(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "instruction": item.get("instruction", ""),
                    "road": item.get("road", ""),
                    "action": item.get("action", ""),
                }
            )
        if len(extracted) <= max_points:
            return extracted
        stride = max(1, len(extracted) // max_points)
        return extracted[::stride][:max_points]

    waypoints: list[dict[str, Any]] = []
    routes = route_data.get("routes", [])
    if not routes:
        return waypoints

    route = routes[0]
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            location = maneuver.get("location")
            if location and isinstance(location, list) and len(location) == 2:
                waypoints.append(
                    {
                        "longitude": location[0],
                        "latitude": location[1],
                        "instruction": maneuver.get("instruction", ""),
                        "road": step.get("name", ""),
                    }
                )

    if len(waypoints) <= max_points:
        return waypoints

    stride = max(1, len(waypoints) // max_points)
    return waypoints[::stride][:max_points]


def extract_waypoint_sequence(route_data: dict[str, Any], max_points: int = 12) -> list[dict[str, float]]:
    semantic_points = extract_key_waypoints(route_data, max_points=max_points)
    return [
        {
            "longitude": float(item["longitude"]),
            "latitude": float(item["latitude"]),
        }
        for item in semantic_points
        if item.get("longitude") is not None and item.get("latitude") is not None
    ]


def keyword_match_score(text_blob: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    matches = sum(1 for keyword in keywords if keyword.lower() in text_blob)
    return matches / max(len(keywords), 1)


def clamp_score(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def apply_hard_constraints(candidate: dict[str, Any], hard_constraints: dict[str, Any]) -> tuple[bool, str]:
    text_blob = candidate["text_blob"]
    duration_min = candidate["summary"]["duration_min"]
    distance_m = candidate["summary"]["distance_m"]

    max_duration = hard_constraints.get("max_route_duration_min")
    if isinstance(max_duration, (int, float)) and duration_min > float(max_duration):
        return False, "duration_exceeded"

    max_distance = hard_constraints.get("max_route_distance_m")
    if isinstance(max_distance, (int, float)) and distance_m > float(max_distance):
        return False, "distance_exceeded"

    # Default: rely on LLM-generated search_keywords + map provider retrieval,
    # and do not apply language-sensitive text hard filters.
    # Set ENABLE_TEXT_HARD_CONSTRAINTS=true to enable legacy keyword constraints.
    text_filter_enabled = os.getenv("ENABLE_TEXT_HARD_CONSTRAINTS", "false").strip().lower() in {"1", "true", "yes"}
    if text_filter_enabled:
        for keyword in hard_constraints.get("must_include_keywords", []) or []:
            if keyword.lower() not in text_blob:
                return False, "missing_required_keyword"

        for keyword in hard_constraints.get("exclude_keywords", []) or []:
            if keyword.lower() in text_blob:
                return False, "excluded_keyword_hit"

    return True, "ok"


def _instruction_has_explicit_time_limit(user_instruction: str) -> bool:
    text = str(user_instruction or "").lower()
    patterns = [
        r"\bwithin\s+\d+\s*(?:min|mins|minute|minutes|hour|hours)\b",
        r"\bunder\s+\d+\s*(?:min|mins|minute|minutes|hour|hours)\b",
        r"\bno more than\s+\d+\s*(?:min|mins|minute|minutes|hour|hours)\b",
        r"\d+\s*(?:分钟|小时)内",
        r"\d+\s*(?:分钟|小时)\s*以内",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _instruction_has_explicit_distance_limit(user_instruction: str) -> bool:
    text = str(user_instruction or "").lower()
    patterns = [
        r"\bwithin\s+\d+\s*(?:m|meter|meters|km|kilometer|kilometers)\b",
        r"\bunder\s+\d+\s*(?:m|meter|meters|km|kilometer|kilometers)\b",
        r"\bno more than\s+\d+\s*(?:m|meter|meters|km|kilometer|kilometers)\b",
        r"\d+\s*(?:米|公里)内",
        r"\d+\s*(?:米|公里)\s*以内",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _instruction_has_explicit_include(user_instruction: str) -> bool:
    text = str(user_instruction or "").lower()
    patterns = [
        r"\bwith\b",
        r"\binclude\b",
        r"\bmust have\b",
        r"\bneed(s)?\b",
        r"\brequire(s|d)?\b",
        r"\b需要\b",
        r"\b必须\b",
        r"\b要有\b",
        r"\b带\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _instruction_has_explicit_exclude(user_instruction: str) -> bool:
    text = str(user_instruction or "").lower()
    patterns = [
        r"\bavoid\b",
        r"\bwithout\b",
        r"\bexclude\b",
        r"\bnot\b",
        r"\bno\b",
        r"\b不要\b",
        r"\b避免\b",
        r"\b排除\b",
        r"\b别\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _operationalize_hard_constraints(
    user_instruction: str,
    *,
    parsed: dict[str, Any],
    hard_constraints: dict[str, Any],
) -> dict[str, Any]:
    effective = {
        "max_route_duration_min": hard_constraints.get("max_route_duration_min"),
        "max_route_distance_m": hard_constraints.get("max_route_distance_m"),
        "must_include_keywords": list(hard_constraints.get("must_include_keywords", []) or []),
        "exclude_keywords": list(hard_constraints.get("exclude_keywords", []) or []),
    }

    destination_kind = str(parsed.get("destination_kind", "")).strip().lower()
    if destination_kind == "generic_poi":
        if not _instruction_has_explicit_time_limit(user_instruction):
            effective["max_route_duration_min"] = None
        if not _instruction_has_explicit_distance_limit(user_instruction):
            effective["max_route_distance_m"] = None
        if not _instruction_has_explicit_include(user_instruction):
            effective["must_include_keywords"] = []
        if not _instruction_has_explicit_exclude(user_instruction):
            effective["exclude_keywords"] = []

    return effective


def soft_rank(candidates: list[dict[str, Any]], optimize_for: str) -> list[dict[str, Any]]:
    if not candidates:
        return []

    weights = load_json_env("RANKING_WEIGHTS_JSON", default_scoring_weights())
    time_weight = float(weights.get("time", 0.5))
    distance_weight = float(weights.get("distance", 0.3))
    proximity_weight = float(weights.get("proximity", 0.1))
    keyword_weight = float(weights.get("keyword", 0.1))

    durations = [item["summary"]["duration_s"] for item in candidates]
    distances = [item["summary"]["distance_m"] for item in candidates]
    direct_distances = [item["direct_distance_m"] for item in candidates]
    min_duration, max_duration = min(durations), max(durations)
    min_distance, max_distance = min(distances), max(distances)
    min_direct, max_direct = min(direct_distances), max(direct_distances)

    def inv_norm(value: float, v_min: float, v_max: float) -> float:
        if abs(v_max - v_min) < 1e-9:
            return 1.0
        return 1.0 - ((value - v_min) / (v_max - v_min))

    for item in candidates:
        duration_score = inv_norm(item["summary"]["duration_s"], min_duration, max_duration)
        distance_score = inv_norm(item["summary"]["distance_m"], min_distance, max_distance)
        proximity_score = inv_norm(item["direct_distance_m"], min_direct, max_direct)
        keyword_score = item["keyword_score"]
        semantic_affinity = clamp_score(item.get("semantic_affinity", keyword_score))
        semantic_weight = clamp_score(item.get("semantic_weight", 0.0), 0.0, 0.6)

        if optimize_for == "time":
            time_component = 0.55 * duration_score
            distance_component = 0.15 * distance_score
        elif optimize_for == "distance":
            time_component = 0.15 * duration_score
            distance_component = 0.55 * distance_score
        else:
            time_component = time_weight * duration_score
            distance_component = distance_weight * distance_score

        base_score = (
            time_component
            + distance_component
            + proximity_weight * proximity_score
            + keyword_weight * keyword_score
        )
        item["score"] = round(
            (1.0 - semantic_weight) * base_score + semantic_weight * semantic_affinity,
            4,
        )
        item["score_breakdown"] = {
            "duration": round(duration_score, 4),
            "distance": round(distance_score, 4),
            "proximity": round(proximity_score, 4),
            "keyword": round(keyword_score, 4),
            "semantic": round(semantic_affinity, 4),
        }

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


async def _ctx_info(ctx: Any, message: str) -> None:
    if ctx and hasattr(ctx, "info"):
        await ctx.info(message)


async def _ctx_warning(ctx: Any, message: str) -> None:
    if ctx and hasattr(ctx, "warning"):
        await ctx.warning(message)


async def _ctx_report_progress(ctx: Any, current: int, total: int) -> None:
    if ctx and hasattr(ctx, "report_progress"):
        await ctx.report_progress(current, total)


def _compact_candidate_tags(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": candidate.get("type", ""),
        "address": candidate.get("address", ""),
        "provider": candidate.get("provider", ""),
        "provider_poi_id": candidate.get("provider_poi_id", ""),
    }


def _candidate_text_blob(candidate: dict[str, Any]) -> str:
    parts = [
        str(candidate.get("name", "")),
        str(candidate.get("type", "")),
        str(candidate.get("address", "")),
    ]
    raw = candidate.get("raw")
    if isinstance(raw, dict):
        parts.append(json.dumps(raw, ensure_ascii=False))
    return " ".join(part for part in parts if part).lower()


def _candidate_structured_text(candidate: dict[str, Any]) -> str:
    parts = [
        str(candidate.get("name", "")),
        str(candidate.get("type", "")),
    ]
    raw = candidate.get("raw")
    if isinstance(raw, dict):
        parts.append(str(raw.get("type", "")))
    return " ".join(part for part in parts if part).lower()


def _estimated_route_distance_m(direct_distance_m: float, mode: str) -> float:
    detour_factors = {
        "foot": float(os.getenv("ESTIMATED_ROUTE_DETOUR_FACTOR_FOOT", "1.25") or 1.25),
        "bike": float(os.getenv("ESTIMATED_ROUTE_DETOUR_FACTOR_BIKE", "1.2") or 1.2),
        "car": float(os.getenv("ESTIMATED_ROUTE_DETOUR_FACTOR_CAR", "1.35") or 1.35),
    }
    factor = detour_factors.get(mode, 1.25)
    return max(float(direct_distance_m or 0.0) * factor, float(direct_distance_m or 0.0))


def _estimated_route_duration_s(route_distance_m: float, mode: str) -> float:
    speed_mps = {
        "foot": float(os.getenv("ESTIMATED_SPEED_MPS_FOOT", "1.3") or 1.3),
        "bike": float(os.getenv("ESTIMATED_SPEED_MPS_BIKE", "4.2") or 4.2),
        "car": float(os.getenv("ESTIMATED_SPEED_MPS_CAR", "8.5") or 8.5),
    }
    speed = max(0.5, speed_mps.get(mode, 1.3))
    return float(route_distance_m or 0.0) / speed


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def _looks_english_instruction(text: str) -> bool:
    value = str(text or "").strip()
    if not value or _contains_cjk(value):
        return False
    return len(re.findall(r"[A-Za-z]", value)) >= 4


async def _run_limited(coroutines: list[Any], concurrency: int) -> list[Any]:
    if not coroutines:
        return []

    limit = max(1, int(concurrency))
    semaphore = asyncio.Semaphore(limit)

    async def _runner(coroutine: Any) -> Any:
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*[_runner(coroutine) for coroutine in coroutines], return_exceptions=True)


def _normalize_text_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _is_identifier_like_name(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return bool(re.fullmatch(r"[0-9][0-9\-_:.]*", text))


def _humanize_type_label(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    primary = text.split(";", 1)[0].replace("_", " ").replace("-", " ").strip()
    if not primary:
        return ""
    primary = re.sub(r"\s+", " ", primary)
    return primary.title()


def _display_name_from_tags(
    *,
    tags: dict[str, Any],
    fallback_type: str = "",
) -> str:
    raw_name = str(tags.get("name") or "").strip()
    if raw_name and not _is_identifier_like_name(raw_name):
        return raw_name

    for key in ("tourism", "leisure", "natural", "historic", "type"):
        label = _humanize_type_label(str(tags.get(key) or ""))
        if label:
            return f"Unnamed {label}".strip()

    label = _humanize_type_label(fallback_type)
    if label:
        return f"Unnamed {label}".strip()
    return "Unnamed Place"


def _apply_english_translations(
    result: dict[str, Any],
    translated_bundle: dict[str, Any],
) -> dict[str, Any]:
    translated_recommendations = translated_bundle.get("recommendations")
    if isinstance(translated_recommendations, list):
        translated_map = {
            str(item.get("id", "")): item
            for item in translated_recommendations
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        for recommendation in result.get("recommendations", []) or []:
            translated = translated_map.get(str(recommendation.get("id", "")))
            if not translated:
                continue
            if isinstance(translated.get("name"), str) and translated.get("name"):
                recommendation["name"] = translated["name"]
            translated_tags = translated.get("tags")
            if isinstance(translated_tags, dict) and isinstance(recommendation.get("tags"), dict):
                for field in ("type", "address"):
                    if isinstance(translated_tags.get(field), str):
                        recommendation["tags"][field] = translated_tags[field]

    translated_chosen = translated_bundle.get("chosen_destination")
    if isinstance(translated_chosen, dict) and isinstance(result.get("chosen_destination"), dict):
        if isinstance(translated_chosen.get("name"), str) and translated_chosen.get("name"):
            result["chosen_destination"]["name"] = translated_chosen["name"]
        translated_tags = translated_chosen.get("tags")
        if isinstance(translated_tags, dict) and isinstance(result["chosen_destination"].get("tags"), dict):
            for field in ("type", "address"):
                if isinstance(translated_tags.get(field), str):
                    result["chosen_destination"]["tags"][field] = translated_tags[field]

    translated_waypoints = translated_bundle.get("key_waypoints")
    if isinstance(translated_waypoints, list):
        for original, translated in zip(result.get("key_waypoints", []) or [], translated_waypoints):
            if not isinstance(translated, dict):
                continue
            for field in ("instruction", "road"):
                if isinstance(translated.get(field), str):
                    original[field] = translated[field]

    translated_rerank = translated_bundle.get("semantic_rerank")
    if isinstance(translated_rerank, dict) and isinstance(result.get("semantic_rerank"), dict):
        if isinstance(translated_rerank.get("summary"), str):
            result["semantic_rerank"]["summary"] = translated_rerank["summary"]

    return result


def _split_camel_case(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)


def _tokenize_semantic_terms(value: str) -> list[str]:
    text = _normalize_text_key(_split_camel_case(str(value or "").replace("_", " ")))
    if not text:
        return []

    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text):
        normalized = token.strip()
        if not normalized:
            continue
        if re.fullmatch(r"[a-z0-9]+", normalized):
            if len(normalized) >= 3:
                tokens.append(normalized)
            continue
        if len(normalized) >= 2:
            tokens.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def _strip_place_descriptors(value: str) -> str:
    text = _normalize_text_key(_split_camel_case(value))
    if not text:
        return ""

    english_patterns = [
        r"\bprovince\b",
        r"\bcity\b",
        r"\bdistrict\b",
        r"\bcounty\b",
        r"\bsubdistrict\b",
        r"\bstreet\b",
        r"\broad\b",
        r"\bavenue\b",
        r"\barea\b",
        r"\bscenic\s+area\b",
        r"\bscenic\s+spot\b",
        r"\btourist\s+attraction\b",
    ]
    chinese_patterns = [
        "\u7701",
        "\u5e02",
        "\u533a",
        "\u53bf",
        "\u8857\u9053",
        "\u8def",
        "\u8857",
        "\u5927\u8857",
        "\u5df7",
        "\u666f\u533a",
        "\u98ce\u666f\u540d\u80dc\u533a",
        "\u98ce\u666f\u540d\u80dc",
        "\u65c5\u6e38\u666f\u70b9",
        "\u98ce\u666f\u540d\u80dc\u533a",
    ]

    for pattern in english_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    for pattern in chinese_patterns:
        text = text.replace(pattern, " ")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _collect_query_terms(user_instruction: str, parsed: dict[str, Any]) -> list[str]:
    destination_kind = _normalize_text_key(parsed.get("destination_kind", ""))
    explicit_query = extract_explicit_destination_query(user_instruction)

    if destination_kind == "regional_scenic_area":
        values = [
            explicit_query,
            parsed.get("search_keywords", ""),
        ]
    else:
        values = [
            explicit_query,
            parsed.get("search_keywords", ""),
            " ".join(split_search_keywords(parsed.get("search_keywords", ""))),
            " ".join(str(item) for item in (parsed.get("keywords") or []) if str(item).strip()),
            parsed.get("destination_type", ""),
        ]

    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        candidates = [text, _strip_place_descriptors(text)]
        for candidate in candidates:
            for token in _tokenize_semantic_terms(candidate):
                if token not in seen:
                    seen.add(token)
                    terms.append(token)
    return terms


def _candidate_affinity_terms(candidate: dict[str, Any]) -> list[str]:
    tags = candidate.get("tags", {}) or {}
    raw_candidate = candidate.get("raw_candidate", {}) or {}
    values = [
        candidate.get("name", ""),
        tags.get("type", ""),
        raw_candidate.get("type", ""),
        _strip_place_descriptors(str(candidate.get("name", ""))),
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in _tokenize_semantic_terms(str(value or "")):
            if token not in seen:
                seen.add(token)
                terms.append(token)
    return terms


def _is_generic_scenic_container(candidate: dict[str, Any]) -> bool:
    text_blob = " ".join(
        [
            str(candidate.get("name", "")),
            str((candidate.get("tags", {}) or {}).get("type", "")),
            str((candidate.get("raw_candidate", {}) or {}).get("type", "")),
        ]
    ).lower()
    generic_tokens = (
        "park",
        "garden",
        "forest",
        "botanical",
        "scenic area",
        "\u516c\u56ed",
        "\u82b1\u56ed",
        "\u690d\u7269\u56ed",
        "\u68ee\u6797\u516c\u56ed",
        "\u666f\u533a",
    )
    subfeature_tokens = (
        "pier",
        "bridge",
        "causeway",
        "tower",
        "island",
        "viewpoint",
        "entrance",
        "visitor center",
        "\u7801\u5934",
        "\u6865",
        "\u5824",
        "\u5854",
        "\u5c9b",
        "\u89c2\u666f",
        "\u5165\u53e3",
        "\u6e38\u5ba2\u4e2d\u5fc3",
    )
    return any(token in text_blob for token in generic_tokens) and not any(
        token in text_blob for token in subfeature_tokens
    )


def _query_affinity_score(
    candidate: dict[str, Any],
    *,
    query_terms: list[str],
    anchor_terms: list[str] | None = None,
) -> float:
    candidate_terms = set(_candidate_affinity_terms(candidate))
    if not candidate_terms:
        return 0.0

    score = 0.0
    direct_matches = [term for term in query_terms if term in candidate_terms]
    if direct_matches:
        score += min(0.75, 0.35 + 0.18 * len(direct_matches))

    anchor_matches = [term for term in (anchor_terms or []) if term in candidate_terms]
    if anchor_matches:
        score += min(0.45, 0.2 + 0.12 * len(anchor_matches))

    candidate_kind = str(candidate.get("candidate_kind", ""))
    if candidate_kind in {"administrative_area", "road_label", "institution", "commercial_place"}:
        score -= 0.35
    if _is_generic_scenic_container(candidate) and not direct_matches:
        score -= 0.2

    return clamp_score(score)


def _regional_anchor_terms(candidates: list[dict[str, Any]], query_terms: list[str]) -> list[str]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("query_affinity", 0.0) or 0.0),
            _candidate_priority(str(item.get("candidate_kind", ""))),
            float(item.get("direct_distance_m", 0.0) or 0.0),
        ),
    )

    terms: list[str] = []
    seen: set[str] = set()
    for candidate in ranked[:4]:
        if str(candidate.get("candidate_kind", "")) in {"administrative_area", "road_label", "institution"}:
            continue
        name = str(candidate.get("name", ""))
        for value in (name, _strip_place_descriptors(name)):
            for token in _tokenize_semantic_terms(value):
                if token in query_terms or len(token) >= 2:
                    if token not in seen:
                        seen.add(token)
                        terms.append(token)
    return terms


def _regional_expansion_acceptance(
    candidate: dict[str, Any],
    *,
    query_affinity: float,
    anchor_distance_m: float,
) -> bool:
    candidate_kind = str(candidate.get("candidate_kind", ""))
    if candidate_kind not in {"scenic_destination", "named_landmark", "specific_poi", "support_facility", "minor_feature"}:
        return False
    if not candidate.get("is_concrete_destination"):
        return False
    if query_affinity >= 0.45:
        return True
    if _is_generic_scenic_container(candidate):
        return query_affinity >= 0.35
    if candidate_kind == "support_facility":
        return query_affinity >= 0.3 and anchor_distance_m <= 700
    if candidate_kind == "minor_feature":
        return query_affinity >= 0.35 and anchor_distance_m <= 800
    return query_affinity >= 0.2 or anchor_distance_m <= 900


def _resolve_destination_kind(user_instruction: str, parsed: dict[str, Any]) -> str:
    raw_kind = _normalize_text_key(parsed.get("destination_kind", ""))
    if raw_kind:
        return raw_kind
    return infer_destination_kind(
        user_instruction=user_instruction,
        destination_type=str(parsed.get("destination_type", "")),
        search_keywords=str(parsed.get("search_keywords", "")),
    )


def _is_explicit_destination_request(user_instruction: str, parsed: dict[str, Any]) -> bool:
    destination_kind = _resolve_destination_kind(user_instruction, parsed)
    if destination_kind in {
        "named_landmark",
        "regional_scenic_area",
        "administrative_area",
        "road_or_route_label",
        "specific_poi",
    }:
        return True

    explicit_query = extract_explicit_destination_query(user_instruction)
    if explicit_query:
        return True

    destination_type = _normalize_text_key(parsed.get("destination_type", ""))
    if destination_type in {
        "named_place",
        "place_name",
        "destination",
        "tourist attraction",
        "tourist_attraction",
        "scenic attraction",
        "scenic_attraction",
        "landmark",
    }:
        return True

    keywords = split_search_keywords(parsed.get("search_keywords"))
    return len(keywords) == 1 and len(keywords[0].split()) >= 2


def _candidate_priority(candidate_kind: str) -> int:
    priorities = {
        "scenic_destination": 0,
        "named_landmark": 1,
        "specific_poi": 2,
        "regional_area": 3,
        "generic_poi": 4,
        "support_facility": 5,
        "minor_feature": 6,
        "commercial_place": 7,
        "lodging": 8,
        "institution": 9,
        "road_label": 10,
        "administrative_area": 11,
        "unknown": 12,
    }
    return priorities.get(candidate_kind, priorities["unknown"])


def _classify_candidate_kind(candidate: dict[str, Any], *, destination_kind: str) -> str:
    tags = candidate.get("tags", {}) or {}
    raw_candidate = candidate.get("raw_candidate", {}) or {}
    text_blob = " ".join(
        [
            str(candidate.get("name", "")),
            str(tags.get("type", "")),
            str(raw_candidate.get("type", "")),
        ]
    ).lower()
    name_blob = _normalize_text_key(candidate.get("name", ""))

    scenic_region_tokens = (
        "scenic area",
        "scenic spot",
        "tourist attraction",
        "natural scenery",
        "lake",
        "park",
        "viewpoint",
        "pier",
        "visitor center",
        "\u98ce\u666f",
        "\u666f\u70b9",
        "\u666f\u533a",
        "\u540d\u80dc",
        "\u6e56\u6cca",
        "\u516c\u56ed",
        "\u7801\u5934",
        "\u6e38\u5ba2\u4e2d\u5fc3",
    )
    scenic_subfeature_tokens = (
        "bridge",
        "causeway",
        "island",
        "tower",
        "temple",
        "shore",
        "boat",
        "\u6865",
        "\u5824",
        "\u5c9b",
        "\u5854",
        "\u5bfa",
        "\u4ead",
        "\u8239",
    )
    admin_tokens = (
        "district",
        "county",
        "province",
        "city",
        "subdistrict",
        "administrative",
        "\u533a\u53bf\u7ea7\u5730\u540d",
        "\u884c\u653f",
        "\u533a",
        "\u53bf",
        "\u7701",
        "\u5e02",
        "\u8857\u9053",
    )
    road_tokens = (
        "road",
        "street",
        "avenue",
        "route",
        "\u9053\u8def\u540d",
        "\u4ea4\u901a\u5730\u540d",
        "\u8def",
        "\u8857",
        "\u5927\u8857",
        "\u5df7",
    )
    institution_tokens = (
        "media",
        "bureau",
        "government",
        "police",
        "center",
        "office",
        "\u878d\u5a92\u4f53",
        "\u5206\u5c40",
        "\u5c40",
        "\u673a\u6784",
        "\u4e2d\u5fc3",
    )
    lodging_tokens = (
        "hotel",
        "lodging",
        "hostel",
        "motel",
        "resort",
        "inn",
        "homestay",
        "\u9152\u5e97",
        "\u5bbe\u9986",
        "\u6c11\u5bbf",
        "typecode_100",
    )
    commercial_tokens = (
        "\u5546\u52a1\u5199\u5b57\u697c",
        "\u5546\u52a1\u4f4f\u5b85",
        "\u697c\u5b87",
        "\u5546\u573a",
        "office building",
        "commercial building",
    )
    food_tokens = (
        "restaurant",
        "cafe",
        "coffee",
        "tea",
        "bar",
        "food",
        "\u9910\u996e",
        "\u5496\u5561",
        "\u8336",
        "\u9152\u5427",
    )
    support_tokens = (
        "visitor center",
        "service center",
        "ticket office",
        "information center",
        "info center",
        "\u6e38\u5ba2\u670d\u52a1\u4e2d\u5fc3",
        "\u6e38\u5ba2\u4e2d\u5fc3",
        "\u670d\u52a1\u4e2d\u5fc3",
        "\u552e\u7968\u5904",
        "\u4fe1\u606f\u54a8\u8be2",
    )
    minor_feature_tokens = (
        "ruins",
        "statue",
        "pavilion",
        "path",
        "corridor",
        "memorial",
        "archway",
        "stele",
        "plaque",
        "\u9057\u5740",
        "\u50cf",
        "\u4ead",
        "\u9601",
        "\u5f84",
        "\u7891",
        "\u5eca",
        "\u574a",
    )

    if destination_kind == "regional_scenic_area" and any(token in name_blob for token in scenic_subfeature_tokens):
        return "scenic_destination"
    if any(token in text_blob for token in lodging_tokens):
        return "lodging"
    if any(token in text_blob for token in food_tokens):
        return "generic_poi"
    if any(token in text_blob for token in commercial_tokens):
        return "commercial_place"
    if destination_kind == "regional_scenic_area" and any(token in text_blob for token in support_tokens):
        return "support_facility"
    if destination_kind == "regional_scenic_area" and any(token in text_blob for token in minor_feature_tokens):
        return "minor_feature"
    if any(token in text_blob for token in scenic_region_tokens):
        if destination_kind == "regional_scenic_area" and ("lake" in text_blob or "\u6e56" in text_blob or "\u666f\u533a" in text_blob):
            return "regional_area"
        return "scenic_destination"
    if any(token in text_blob for token in admin_tokens):
        return "administrative_area"
    if destination_kind != "regional_scenic_area" and any(token in text_blob for token in road_tokens):
        return "road_label"
    if any(token in text_blob for token in institution_tokens):
        return "institution"
    if destination_kind == "named_landmark":
        return "named_landmark"
    return "generic_poi"


def _is_concrete_destination(candidate_kind: str) -> bool:
    return candidate_kind not in {"administrative_area", "road_label", "regional_area", "institution"}


def _regional_diversify_candidates(
    candidates: list[dict[str, Any]],
    *,
    desired_count: int,
    min_spacing_m: float,
    max_per_anchor: int = 2,
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates

    selected: list[dict[str, Any]] = []
    anchor_counts: dict[str, int] = {}

    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("query_affinity", 0.0) or 0.0),
            _candidate_priority(str(item.get("candidate_kind", ""))),
            float(item.get("direct_distance_m", 0.0) or 0.0),
        ),
    )

    for candidate in ordered:
        anchor_name = str(candidate.get("anchor_name", "")).strip().lower()
        if anchor_name and anchor_counts.get(anchor_name, 0) >= max_per_anchor:
            continue

        far_enough = True
        for picked in selected:
            spacing = haversine_m(
                float(candidate["latitude"]),
                float(candidate["longitude"]),
                float(picked["latitude"]),
                float(picked["longitude"]),
            )
            if spacing < min_spacing_m:
                far_enough = False
                break

        if far_enough or len(selected) < 2:
            selected.append(candidate)
            if anchor_name:
                anchor_counts[anchor_name] = anchor_counts.get(anchor_name, 0) + 1

        if len(selected) >= desired_count:
            break

    return selected or ordered[:desired_count]


def _annotate_candidate(candidate: dict[str, Any], *, destination_kind: str) -> dict[str, Any]:
    annotated = dict(candidate)
    candidate_kind = _classify_candidate_kind(annotated, destination_kind=destination_kind)
    annotated["candidate_kind"] = candidate_kind
    annotated["is_concrete_destination"] = _is_concrete_destination(candidate_kind)
    annotated["query_affinity"] = float(annotated.get("query_affinity", 0.0) or 0.0)
    return annotated


def _named_place_queries(
    search_keywords: str,
    *,
    user_instruction: str = "",
    locality_hints: list[str] | None = None,
) -> list[str]:
    parts = split_search_keywords(search_keywords) or [str(search_keywords or "").strip()]
    explicit_query = extract_explicit_destination_query(user_instruction)
    if explicit_query:
        parts.insert(0, explicit_query)
    queries: list[str] = []
    seen: set[str] = set()
    locality_hints = [str(item).strip() for item in (locality_hints or []) if str(item).strip()]

    for part in parts:
        base = str(part or "").strip()
        if not base:
            continue
        variants = [base]

        camel_spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base).strip()
        if camel_spaced and camel_spaced != base:
            variants.append(camel_spaced)

        underscored = base.replace("_", " ").strip()
        if underscored and underscored != base:
            variants.append(underscored)

        for variant in variants:
            expanded_variants = [variant]
            expanded_variants.extend(f"{variant} {hint}".strip() for hint in locality_hints)
            for expanded in expanded_variants:
                key = expanded.lower()
                if key and key not in seen:
                    seen.add(key)
                    queries.append(expanded)

    return queries[:8]


def _extract_locality_hints(reverse_geocode_result: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    if not isinstance(reverse_geocode_result, dict):
        return hints

    address = reverse_geocode_result.get("address")
    if isinstance(address, dict):
        for field in ("city", "district", "township", "county", "state", "province"):
            value = str(address.get(field) or "").strip()
            if value:
                hints.append(value)

    display_name = str(reverse_geocode_result.get("display_name") or "").strip()
    if display_name:
        chunks = re.split(r"[,/]", display_name)
        for chunk in chunks[:3]:
            value = str(chunk).strip()
            if value:
                hints.append(value)

    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        key = _normalize_text_key(hint)
        if key and key not in seen:
            seen.add(key)
            deduped.append(hint)
    return deduped[:4]


def _normalize_named_place_candidates(
    geocode_results: list[dict[str, Any]],
    current_latitude: float,
    current_longitude: float,
    max_direct_distance_m: float | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in geocode_results:
        try:
            lat = float(item.get("lat"))
            lon = float(item.get("lon"))
        except Exception:
            continue

        source = str(item.get("source", "geocode")).strip() or "geocode"
        display_name = str(item.get("display_name", "Named Place")).strip() or "Named Place"
        candidate_id = f"named:{source}:{round(lat, 6)}:{round(lon, 6)}"
        if candidate_id in seen:
            continue

        direct_distance = haversine_m(current_latitude, current_longitude, lat, lon)
        if isinstance(max_direct_distance_m, (int, float)) and direct_distance > float(max_direct_distance_m):
            continue

        seen.add(candidate_id)
        raw_item = item.get("raw", item)
        raw_type = ""
        raw_address = display_name
        if isinstance(raw_item, dict):
            raw_type = str(raw_item.get("type") or raw_item.get("tag") or "").strip()
            raw_address = str(raw_item.get("address") or raw_item.get("formatted_address") or display_name).strip() or display_name
        raw_candidate = {
            "name": display_name,
            "type": raw_type or "named_place",
            "address": raw_address,
            "provider": source,
            "raw": raw_item,
        }
        normalized.append(
            {
                "id": candidate_id,
                "name": display_name,
                "latitude": lat,
                "longitude": lon,
                "location": f"{lon},{lat}",
                "tags": {
                    "type": raw_type or "named_place",
                    "address": raw_address,
                    "provider": source,
                    "provider_poi_id": candidate_id,
                },
                "raw_candidate": raw_candidate,
                "direct_distance_m": round(direct_distance, 1),
            }
        )

    normalized.sort(key=lambda item: item["direct_distance_m"])
    return normalized


def _normalize_semantic_poi_candidates(
    candidates_raw: list[dict[str, Any]],
    *,
    destination_kind: str,
) -> list[dict[str, Any]]:
    normalized_candidates: list[dict[str, Any]] = []
    for item in candidates_raw:
        coordinates = item.get("coordinates", {}) or {}
        lat = coordinates.get("latitude")
        lon = coordinates.get("longitude")
        if lat is None or lon is None:
            continue

        name = item.get("name") or "Unnamed"
        normalized_candidates.append(
            _annotate_candidate(
                {
                    "id": item.get("provider_poi_id") or item.get("location") or f"{lat}-{lon}-{name}",
                    "name": name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "location": item.get("location") or f"{lon},{lat}",
                    "tags": _compact_candidate_tags(item),
                    "raw_candidate": item,
                    "direct_distance_m": float(item.get("distance_m", 0) or 0),
                },
                destination_kind=destination_kind,
            )
        )
    return normalized_candidates


def _normalize_nearby_poi_candidates(
    pois: list[dict[str, Any]],
    *,
    current_latitude: float,
    current_longitude: float,
    destination_kind: str,
    source_prefix: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in pois:
        tags = item.get("tags", {}) or {}
        lat = item.get("lat")
        lon = item.get("lon")
        if lat is None or lon is None:
            continue

        poi_type = str(tags.get("type") or "").strip()
        name = _display_name_from_tags(tags=tags, fallback_type=poi_type)
        provider_poi_id = str(item.get("id") or f"{source_prefix}:{lon}:{lat}:{name}")
        address = str(tags.get("address") or "").strip()
        normalized.append(
            _annotate_candidate(
                {
                    "id": provider_poi_id,
                    "name": name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "location": f"{float(lon)},{float(lat)}",
                    "tags": {
                        "type": poi_type,
                        "address": address,
                        "provider": str(tags.get("source") or source_prefix),
                        "provider_poi_id": provider_poi_id,
                    },
                    "raw_candidate": {
                        "name": name,
                        "type": poi_type,
                        "address": address,
                        "provider": str(tags.get("source") or source_prefix),
                        "raw": item,
                    },
                    "direct_distance_m": round(
                        haversine_m(current_latitude, current_longitude, float(lat), float(lon)),
                        1,
                    ),
                },
                destination_kind=destination_kind,
            )
        )
    return normalized


def _bbox_from_center(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    lat_delta = radius_m / 111000
    lon_divisor = max(0.2, math.cos(math.radians(lat)))
    lon_delta = radius_m / (111000 * lon_divisor)
    return (lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta)


def _normalize_feature_candidates(
    features: list[dict[str, Any]],
    *,
    current_latitude: float,
    current_longitude: float,
    destination_kind: str,
    source_prefix: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in features:
        center = item.get("center", {}) or {}
        lat = item.get("lat", center.get("lat"))
        lon = item.get("lon", center.get("lon"))
        if lat is None or lon is None:
            continue

        tags = item.get("tags", {}) or {}
        name = str(tags.get("name") or item.get("id") or "Unnamed").strip() or "Unnamed"
        provider_poi_id = str(item.get("id") or f"{source_prefix}:{lon}:{lat}:{name}")

        poi_type_parts = []
        for key in ("tourism", "leisure", "natural", "historic"):
            value = str(tags.get(key) or "").strip()
            if value:
                poi_type_parts.append(value)
        poi_type = ";".join(poi_type_parts) or "unknown"
        name = _display_name_from_tags(tags=tags, fallback_type=poi_type)
        address = str(tags.get("addr:full") or tags.get("address") or "").strip()

        normalized.append(
            _annotate_candidate(
                {
                    "id": provider_poi_id,
                    "name": name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "location": f"{float(lon)},{float(lat)}",
                    "tags": {
                        "type": poi_type,
                        "address": address,
                        "provider": source_prefix,
                        "provider_poi_id": provider_poi_id,
                    },
                    "raw_candidate": {
                        "name": name,
                        "type": poi_type,
                        "address": address,
                        "provider": source_prefix,
                        "raw": item,
                    },
                    "direct_distance_m": round(
                        haversine_m(current_latitude, current_longitude, float(lat), float(lon)),
                        1,
                    ),
                },
                destination_kind=destination_kind,
            )
        )
    return normalized


async def _expand_regional_scenic_area_candidates(
    *,
    base_candidates: list[dict[str, Any]],
    map_client: MapProviderClient,
    provider: str,
    current_latitude: float,
    current_longitude: float,
    radius: float,
    query_terms: list[str],
    ctx: Any = None,
) -> list[dict[str, Any]]:
    sorted_base = sorted(
        base_candidates,
        key=lambda item: (
            _candidate_priority(str(item.get("candidate_kind", ""))),
            -float(item.get("query_affinity", 0.0) or 0.0),
            float(item.get("direct_distance_m", 0.0) or 0.0),
        ),
    )
    anchor_terms = _regional_anchor_terms(sorted_base, query_terms)
    anchors = [
        candidate
        for candidate in sorted_base
        if candidate.get("candidate_kind") in {"regional_area", "scenic_destination", "named_landmark"}
        and float(candidate.get("query_affinity", 0.0) or 0.0) >= 0.15
    ][:3]
    if not anchors:
        anchors = [
            candidate
            for candidate in sorted_base
            if candidate.get("candidate_kind") in {"regional_area", "scenic_destination", "named_landmark", "specific_poi"}
        ][:2]

    expanded: list[dict[str, Any]] = []
    for candidate in sorted_base:
        affinity = _query_affinity_score(candidate, query_terms=query_terms, anchor_terms=anchor_terms)
        candidate["query_affinity"] = max(float(candidate.get("query_affinity", 0.0) or 0.0), affinity)
        if candidate.get("candidate_kind") in {"scenic_destination", "named_landmark", "specific_poi"} and candidate.get(
            "is_concrete_destination"
        ):
            if candidate["query_affinity"] >= 0.15 or not _is_generic_scenic_container(candidate):
                expanded.append(candidate)
        elif candidate.get("candidate_kind") == "regional_area" and candidate["query_affinity"] >= 0.25:
            expanded.append(candidate)

    search_radius = max(900, min(int(radius * 0.2), 2200))

    expand_concurrency = int(os.getenv("REGIONAL_EXPANSION_CONCURRENCY", "3") or 3)

    async def _fetch_anchor_nearby(anchor: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        nearby = await map_client.get_nearby_pois(
            lat=float(anchor["latitude"]),
            lon=float(anchor["longitude"]),
            radius=search_radius,
            categories=["tourism", "leisure"],
        )
        return anchor, nearby

    nearby_results = await _run_limited(
        [_fetch_anchor_nearby(anchor) for anchor in anchors],
        concurrency=expand_concurrency,
    )

    for result in nearby_results:
        if isinstance(result, Exception):
            continue

        anchor, nearby = result
        normalized_nearby = _normalize_nearby_poi_candidates(
            nearby,
            current_latitude=current_latitude,
            current_longitude=current_longitude,
            destination_kind="regional_scenic_area",
            source_prefix="regional_expansion",
        )
        for candidate in normalized_nearby:
            anchor_distance_m = haversine_m(
                float(anchor["latitude"]),
                float(anchor["longitude"]),
                float(candidate["latitude"]),
                float(candidate["longitude"]),
            )
            direct_affinity = _query_affinity_score(candidate, query_terms=query_terms, anchor_terms=anchor_terms)
            inherited_affinity = clamp_score(max(direct_affinity, float(anchor.get("query_affinity", 0.0) or 0.0) * 0.55))
            if anchor_distance_m <= 900:
                inherited_affinity = clamp_score(max(inherited_affinity, float(anchor.get("query_affinity", 0.0) or 0.0) * 0.75))
            candidate["query_affinity"] = inherited_affinity
            candidate["anchor_name"] = anchor.get("name", "")
            candidate["anchor_distance_m"] = round(anchor_distance_m, 1)
            if _regional_expansion_acceptance(
                candidate,
                query_affinity=float(candidate["query_affinity"]),
                anchor_distance_m=anchor_distance_m,
            ):
                expanded.append(candidate)

    concrete_count = sum(1 for candidate in expanded if candidate.get("is_concrete_destination"))
    if provider == "osm" and concrete_count < 3 and anchors:
        feature_queries = []
        for anchor in anchors[:2]:
            bbox = _bbox_from_center(
                float(anchor["latitude"]),
                float(anchor["longitude"]),
                max(search_radius * 1.25, 1400),
            )
            feature_queries.extend(
                [
                    (anchor, bbox, "tourism", ["attraction", "viewpoint", "museum", "gallery", "theme_park"]),
                    (anchor, bbox, "leisure", ["park", "garden", "nature_reserve"]),
                ]
            )

        async def _fetch_anchor_features(
            anchor: dict[str, Any],
            bbox: tuple[float, float, float, float],
            category: str,
            subcategories: list[str],
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            features = await map_client.search_features_by_category(
                bbox=bbox,
                category=category,
                subcategories=subcategories,
            )
            return anchor, features

        feature_results = await _run_limited(
            [
                _fetch_anchor_features(anchor, bbox, category, subcategories)
                for anchor, bbox, category, subcategories in feature_queries
            ],
            concurrency=expand_concurrency,
        )

        for result in feature_results:
            if isinstance(result, Exception):
                await _ctx_warning(ctx, f"OSM regional feature expansion failed: {result}")
                continue

            anchor, features = result
            normalized_features = _normalize_feature_candidates(
                features,
                current_latitude=current_latitude,
                current_longitude=current_longitude,
                destination_kind="regional_scenic_area",
                source_prefix="osm_feature_expansion",
            )
            for candidate in normalized_features:
                anchor_distance_m = haversine_m(
                    float(anchor["latitude"]),
                    float(anchor["longitude"]),
                    float(candidate["latitude"]),
                    float(candidate["longitude"]),
                )
                direct_affinity = _query_affinity_score(candidate, query_terms=query_terms, anchor_terms=anchor_terms)
                inherited_affinity = clamp_score(
                    max(
                        direct_affinity,
                        float(anchor.get("query_affinity", 0.0) or 0.0) * 0.65,
                    )
                )
                candidate["query_affinity"] = max(float(candidate.get("query_affinity", 0.0) or 0.0), inherited_affinity)
                candidate["anchor_name"] = anchor.get("name", "")
                candidate["anchor_distance_m"] = round(anchor_distance_m, 1)
                if _regional_expansion_acceptance(
                    candidate,
                    query_affinity=float(candidate["query_affinity"]),
                    anchor_distance_m=anchor_distance_m,
                ):
                    expanded.append(candidate)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in expanded:
        if candidate["id"] in seen:
            continue
        seen.add(candidate["id"])
        deduped.append(candidate)

    deduped.sort(
        key=lambda item: (
            _candidate_priority(str(item.get("candidate_kind", ""))),
            -float(item.get("query_affinity", 0.0) or 0.0),
            float(item.get("direct_distance_m", 0) or 0),
        )
    )
    return _regional_diversify_candidates(
        deduped,
        desired_count=max(12, min(len(deduped), 18)),
        min_spacing_m=float(os.getenv("REGIONAL_CANDIDATE_MIN_SPACING_M", "650") or 650),
    )


def _candidate_type_text(candidate: dict[str, Any]) -> str:
    tags = candidate.get("tags", {}) or {}
    raw_candidate = candidate.get("raw_candidate", {}) or {}
    return " ".join(
        [
            str(tags.get("type", "")),
            str(raw_candidate.get("type", "")),
            str(raw_candidate.get("typecode", "")),
        ]
    ).lower()


def _semantic_candidate_bias(
    candidate: dict[str, Any],
    *,
    parsed: dict[str, Any],
    explicit_destination: bool,
) -> float:
    destination_type = _normalize_text_key(parsed.get("destination_type", ""))
    destination_kind = _normalize_text_key(parsed.get("destination_kind", ""))
    candidate_kind = _normalize_text_key(candidate.get("candidate_kind", ""))
    type_blob = _candidate_type_text(candidate)
    name_blob = _normalize_text_key(candidate.get("name", ""))

    score = 0.0
    scenic_tokens = ("scenic", "attraction", "tourism", "park", "lake", "viewpoint", "110", "风景", "景点", "公园", "湖")
    lodging_tokens = ("hotel", "lodging", "hostel", "motel", "resort", "inn", "homestay", "100", "酒店", "宾馆", "民宿")
    restricted_tokens = ("not open", "closed", "private", "\u4e0d\u5bf9\u5916\u5f00\u653e", "\u6682\u505c\u8425\u4e1a")

    if explicit_destination or "tourist" in destination_type or "scenic" in destination_type or "landmark" in destination_type:
        if any(token in type_blob or token in name_blob for token in scenic_tokens):
            score += 0.35
        if any(token in type_blob or token in name_blob for token in lodging_tokens):
            score -= 0.45
        if destination_kind == "regional_scenic_area":
            if candidate_kind == "scenic_destination":
                score += 0.45
            elif candidate_kind == "named_landmark":
                score += 0.35
            elif candidate_kind == "support_facility":
                score -= 0.3
            elif candidate_kind == "minor_feature":
                score -= 0.25
            elif candidate_kind == "regional_area":
                score -= 0.1
            elif candidate_kind in {"administrative_area", "road_label", "institution", "commercial_place"}:
                score -= 0.6
        elif candidate_kind in {"administrative_area", "road_label"}:
            score -= 0.35

    if any(token in name_blob for token in restricted_tokens):
        score -= 0.4
    if destination_kind == "regional_scenic_area":
        query_affinity = float(candidate.get("query_affinity", 0.0) or 0.0)
        score += 0.55 * query_affinity
        if _is_generic_scenic_container(candidate) and query_affinity < 0.25:
            score -= 0.25

    return score


class DestinationRecommendationEngine:
    def __init__(
        self,
        map_client: MapProviderClient,
        provider: str,
        intent_parser: IntentParser | None = None,
    ) -> None:
        self.map_client = map_client
        self.provider = provider
        self.intent_parser = intent_parser or IntentParser()

    async def recommend(
        self,
        user_instruction: str,
        current_latitude: float,
        current_longitude: float,
        max_candidates: int = 5,
        search_radius: float = 3000,
        mode: str | None = None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        overall_started_at = perf_counter()
        eval_limit = int(os.getenv("ROUTE_EVAL_LIMIT", "25"))
        if self.provider == "amap":
            eval_limit = min(eval_limit, int(os.getenv("ROUTE_EVAL_LIMIT_AMAP", "8")))
        default_opt_target = os.getenv("DEFAULT_OPTIMIZE_FOR", "distance").strip() or "distance"

        original_instruction = user_instruction
        working_instruction = user_instruction
        translated_instruction = None
        english_localization_active = False
        if self.provider in {"amap", "baidu"} and _looks_english_instruction(user_instruction):
            localization_started_at = perf_counter()
            translated_instruction = await translate_instruction_for_cn_map(
                user_instruction=user_instruction,
                provider=self.provider,
            )
            if translated_instruction:
                working_instruction = translated_instruction
                english_localization_active = True
                await _ctx_info(
                    ctx,
                    f"Localized English instruction for {self.provider} in {perf_counter() - localization_started_at:.2f}s: {translated_instruction}",
                )
            else:
                await _ctx_warning(
                    ctx,
                    f"English instruction localization for {self.provider} failed after {perf_counter() - localization_started_at:.2f}s; continuing with original instruction.",
                )

        parse_started_at = perf_counter()
        parsed = await self.intent_parser.parse(working_instruction)
        await _ctx_info(ctx, f"Intent parse completed in {perf_counter() - parse_started_at:.2f}s.")
        intent_source = str(parsed.get("intent_source") or "")
        if intent_source.startswith("heuristic_llm_"):
            error_detail = str(parsed.get("llm_error") or "").strip()
            warning = f"LLM parse fallback activated: {intent_source}"
            if error_detail:
                warning = f"{warning}; detail={error_detail}"
            await _ctx_warning(ctx, warning)

        search_keywords = parsed.get("search_keywords", "")
        keywords = parsed.get("keywords") or split_search_keywords(search_keywords)
        hard_constraints = _operationalize_hard_constraints(
            working_instruction,
            parsed=parsed,
            hard_constraints=parsed.get("hard_constraints") or default_intent_rules()["default_hard_constraints"],
        )

        parsed_mode = str(parsed.get("mode") or "car").strip().lower()
        user_mode = str(mode or "").strip().lower()
        chosen_mode = user_mode if user_mode in {"car", "foot", "bike"} else parsed_mode
        if chosen_mode not in {"car", "foot", "bike"}:
            chosen_mode = "car"

        parsed_radius = parsed.get("radius_m")
        if not isinstance(parsed_radius, (int, float)) or parsed_radius <= 0:
            parsed_radius = 3000
        user_radius = search_radius if isinstance(search_radius, (int, float)) and search_radius > 0 else None
        radius = float(user_radius if user_radius is not None else parsed_radius)

        optimize_for = parsed.get("optimize_for", default_opt_target)

        parsed_top_k = parsed.get("max_candidates")
        if not isinstance(parsed_top_k, (int, float)) or parsed_top_k < 1:
            parsed_top_k = 5
        user_top_k = max_candidates if isinstance(max_candidates, (int, float)) and max_candidates >= 1 else None
        top_k = int(user_top_k if user_top_k is not None else parsed_top_k)
        top_k = max(1, top_k)

        semantic_keywords = search_keywords or "|".join(keywords)
        origin_location = f"{current_longitude},{current_latitude}"
        poi_limit = max(top_k, min(eval_limit, 25))
        destination_type = str(parsed.get("destination_type", "poi")).strip().lower()
        destination_kind = _resolve_destination_kind(working_instruction, parsed)
        query_terms = _collect_query_terms(working_instruction, parsed)
        parsed["destination_kind"] = destination_kind
        explicit_destination = _is_explicit_destination_request(working_instruction, parsed)
        if translated_instruction:
            parsed["provider_instruction_zh"] = translated_instruction
            parsed["original_user_instruction"] = original_instruction
        rerank_meta: dict[str, Any] | None = None

        await _ctx_info(
            ctx,
            f"Provider={self.provider}, mode={chosen_mode}, radius={radius}m, search_keywords={semantic_keywords}, destination_kind={destination_kind}",
        )

        normalized_candidates: list[dict[str, Any]] = []
        retrieval_started_at = perf_counter()
        if explicit_destination:
            locality_hints: list[str] = []
            try:
                locality_hints = _extract_locality_hints(
                    await self.map_client.reverse_geocode(current_latitude, current_longitude)
                )
            except Exception:
                locality_hints = []

            queries = _named_place_queries(
                semantic_keywords,
                user_instruction=working_instruction,
                locality_hints=locality_hints,
            )
            await _ctx_info(ctx, f"Named-place retrieval via geocode, queries={queries}")
            geocode_concurrency = int(os.getenv("GEOCODE_CONCURRENCY", "4") or 4)
            geocode_results = await _run_limited(
                [self.map_client.geocode(query) for query in queries],
                concurrency=geocode_concurrency,
            )
            geocode_hits: list[dict[str, Any]] = []
            for result in geocode_results:
                if isinstance(result, Exception):
                    continue
                geocode_hits.extend(result)
            normalized_candidates = _normalize_named_place_candidates(
                geocode_hits,
                current_latitude=current_latitude,
                current_longitude=current_longitude,
                max_direct_distance_m=max(float(radius) * 1.5, 5000.0),
            )
            normalized_candidates = [
                _annotate_candidate(candidate, destination_kind=destination_kind)
                for candidate in normalized_candidates
            ]
            for candidate in normalized_candidates:
                candidate["query_affinity"] = _query_affinity_score(candidate, query_terms=query_terms)
            if destination_kind == "regional_scenic_area":
                grounded_candidates = await _expand_regional_scenic_area_candidates(
                    base_candidates=normalized_candidates,
                    map_client=self.map_client,
                    provider=self.provider,
                    current_latitude=current_latitude,
                    current_longitude=current_longitude,
                    radius=radius,
                    query_terms=query_terms,
                    ctx=ctx,
                )
                if grounded_candidates:
                    normalized_candidates = grounded_candidates
                elif normalized_candidates and not any(
                    candidate.get("is_concrete_destination")
                    and candidate.get("candidate_kind") in {"scenic_destination", "named_landmark", "specific_poi"}
                    for candidate in normalized_candidates
                ):
                    await _ctx_info(
                        ctx,
                        "Named-place grounding produced only non-concrete regional candidates; fallback to semantic POI search.",
                    )
                    normalized_candidates = []
            if not normalized_candidates:
                await _ctx_info(ctx, "Named-place geocode returned no candidates; fallback to semantic POI search.")

        if not normalized_candidates:
            candidates_raw = await self.map_client.search_semantic_pois(
                location=origin_location,
                keywords=semantic_keywords,
                radius=int(radius),
                sortrule="distance",
                limit=poi_limit,
            )
            normalized_candidates = _normalize_semantic_poi_candidates(
                candidates_raw,
                destination_kind=destination_kind,
            )
            for candidate in normalized_candidates:
                candidate["query_affinity"] = _query_affinity_score(candidate, query_terms=query_terms)
        await _ctx_info(
            ctx,
            f"Candidate retrieval completed in {perf_counter() - retrieval_started_at:.2f}s with {len(normalized_candidates)} raw candidates.",
        )

        deduped = []
        seen: set[str] = set()
        for candidate in normalized_candidates:
            if candidate["id"] in seen:
                continue
            seen.add(candidate["id"])
            candidate["query_affinity"] = max(
                float(candidate.get("query_affinity", 0.0) or 0.0),
                _query_affinity_score(candidate, query_terms=query_terms),
            )
            deduped.append(candidate)

        if destination_kind == "regional_scenic_area":
            concrete_candidates = [
                candidate
                for candidate in deduped
                if candidate.get("is_concrete_destination")
                and candidate.get("candidate_kind") in {"scenic_destination", "named_landmark", "specific_poi"}
                and (
                    float(candidate.get("query_affinity", 0.0) or 0.0) >= 0.2
                    or float(candidate.get("anchor_distance_m", 999999.0) or 999999.0) <= 900
                )
            ]
            if concrete_candidates:
                deduped = concrete_candidates
            else:
                deduped = [
                    candidate
                    for candidate in deduped
                    if candidate.get("candidate_kind") not in {"administrative_area", "road_label", "institution", "commercial_place"}
                ]

        if deduped and explicit_destination:
            rerank_started_at = perf_counter()
            rerank_input = [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "type": (item.get("tags", {}) or {}).get("type", ""),
                    "address": (item.get("tags", {}) or {}).get("address", ""),
                    "distance_m": item.get("direct_distance_m"),
                    "candidate_kind": item.get("candidate_kind", ""),
                    "is_concrete_destination": item.get("is_concrete_destination", False),
                    "query_affinity": round(float(item.get("query_affinity", 0.0) or 0.0), 4),
                    "anchor_name": str(item.get("anchor_name", "")),
                    "anchor_distance_m": item.get("anchor_distance_m"),
                }
                for item in deduped[: min(len(deduped), 10)]
            ]
            rerank_meta = await rerank_candidates_with_llm(
                user_instruction=working_instruction,
                parsed_intent=parsed,
                candidates=rerank_input,
            )
            await _ctx_info(
                ctx,
                f"Candidate rerank completed in {perf_counter() - rerank_started_at:.2f}s for {len(rerank_input)} candidates.",
            )
            if rerank_meta and rerank_meta.get("ranked_candidate_ids"):
                order_map = {
                    candidate_id: index
                    for index, candidate_id in enumerate(rerank_meta["ranked_candidate_ids"])
                }
                deduped.sort(key=lambda item: order_map.get(item["id"], len(order_map) + item["direct_distance_m"]))
            else:
                deduped.sort(
                    key=lambda item: (
                        _candidate_priority(str(item.get("candidate_kind", ""))),
                        float(item.get("direct_distance_m", 0) or 0),
                    )
                )
        else:
            deduped.sort(
                key=lambda item: (
                    _candidate_priority(str(item.get("candidate_kind", ""))),
                    float(item.get("direct_distance_m", 0) or 0),
                )
            )

        stage1_filtered = []
        stage1_rejected = []
        route_failures = []
        lightweight_scoring_started_at = perf_counter()
        progress_total = max(1, len(deduped))

        for index, candidate in enumerate(deduped):
            await _ctx_report_progress(ctx, index, progress_total)
            direct_distance_m = round(
                candidate.get("direct_distance_m")
                or haversine_m(
                    current_latitude,
                    current_longitude,
                    candidate["latitude"],
                    candidate["longitude"],
                ),
                1,
            )
            estimated_distance_m = round(_estimated_route_distance_m(direct_distance_m, chosen_mode), 1)
            estimated_duration_s = round(_estimated_route_duration_s(estimated_distance_m, chosen_mode), 1)

            if destination_kind == "regional_scenic_area":
                text_blob = _candidate_structured_text(candidate["raw_candidate"])
            else:
                text_blob = _candidate_text_blob(candidate["raw_candidate"])

            staged = {
                "candidate": candidate,
                "route": None,
                "text_blob": text_blob,
                "summary": {
                    "distance_m": estimated_distance_m,
                    "duration_s": estimated_duration_s,
                    "duration_min": round(estimated_duration_s / 60, 1),
                },
                "summary_source": "estimated",
                "direct_distance_m": direct_distance_m,
                "keyword_score": clamp_score(
                    keyword_match_score(text_blob, keywords)
                    + _semantic_candidate_bias(
                        candidate,
                        parsed=parsed,
                        explicit_destination=explicit_destination,
                    )
                ),
                "semantic_affinity": clamp_score(
                    max(
                        float(candidate.get("query_affinity", 0.0) or 0.0),
                        keyword_match_score(text_blob, keywords),
                    )
                ),
                "semantic_weight": 0.45 if destination_kind == "regional_scenic_area" else 0.0,
            }

            passed, reason = apply_hard_constraints(staged, hard_constraints)
            if passed:
                stage1_filtered.append(staged)
            else:
                stage1_rejected.append(
                    {
                        "id": candidate["id"],
                        "name": candidate["name"],
                        "reason": reason,
                    }
                )

        await _ctx_info(
            ctx,
            f"Lightweight candidate scoring completed in {perf_counter() - lightweight_scoring_started_at:.2f}s for {len(deduped)} candidates.",
        )

        scored = soft_rank(stage1_filtered, optimize_for)
        picked = scored[:top_k]
        await _ctx_report_progress(ctx, 1, 1)

        recommendations = []
        for item in picked:
            candidate = item["candidate"]
            recommendations.append(
                {
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "coordinates": {
                        "latitude": candidate["latitude"],
                        "longitude": candidate["longitude"],
                    },
                    "score": item["score"],
                    "score_breakdown": item.get("score_breakdown", {}),
                    "route_summary": item["summary"],
                    "tags": candidate["tags"],
                    "candidate_kind": candidate.get("candidate_kind", "unknown"),
                    "is_concrete_destination": candidate.get("is_concrete_destination", False),
                    "query_affinity": round(float(candidate.get("query_affinity", 0.0) or 0.0), 4),
                    "route_summary_source": item.get("summary_source", "estimated"),
                    "location": candidate["location"],
                }
            )

        if not picked:
            return {
                "provider": self.provider,
                "input": {
                    "user_instruction": original_instruction,
                    "provider_instruction": translated_instruction or original_instruction,
                    "current_latitude": current_latitude,
                    "current_longitude": current_longitude,
                },
                "parsed_intent": parsed,
                "recommendations": [],
                "hard_constraint_rejections": stage1_rejected,
                "route_failures": route_failures,
                "semantic_rerank": rerank_meta,
                "message": "No candidates matched constraints.",
            }

        chosen_route_data = None
        chosen_route_summary = recommendations[0]["route_summary"].copy()
        chosen_route_started_at = perf_counter()
        try:
            chosen_route_data = await self.map_client.get_coarse_waypoints(
                origin=origin_location,
                destination=recommendations[0]["location"],
                mode=chosen_mode,
            )
            actual_summary = (chosen_route_data.get("route_summary", {}) or {})
            actual_distance_m = float(actual_summary.get("distance_m", 0) or 0)
            actual_duration_s = float(actual_summary.get("duration_s", 0) or 0)
            if actual_distance_m > 0 or actual_duration_s > 0:
                chosen_route_summary = {
                    "distance_m": round(actual_distance_m, 1),
                    "duration_s": round(actual_duration_s, 1),
                    "duration_min": round(actual_duration_s / 60, 1),
                }
                recommendations[0]["route_summary"] = chosen_route_summary
                recommendations[0]["route_summary_source"] = "actual"
            else:
                route_failures.append(
                    {
                        "id": recommendations[0]["id"],
                        "name": recommendations[0]["name"],
                        "reason": "empty_route_summary",
                    }
                )
                chosen_route_data = None
        except Exception as exc:
            route_failures.append(
                {
                    "id": recommendations[0]["id"],
                    "name": recommendations[0]["name"],
                    "reason": "route_exception",
                    "detail": str(exc),
                }
            )
            chosen_route_data = None

        await _ctx_info(
            ctx,
            f"Chosen-destination route computation completed in {perf_counter() - chosen_route_started_at:.2f}s.",
        )

        result = {
            # Rich semantic waypoints for explainability and debugging.
            "provider": self.provider,
            "input": {
                "user_instruction": original_instruction,
                "provider_instruction": translated_instruction or original_instruction,
                "current_latitude": current_latitude,
                "current_longitude": current_longitude,
            },
            "parsed_intent": parsed,
            "recommendations": recommendations,
            "chosen_destination": recommendations[0],
            "key_waypoints": extract_key_waypoints(chosen_route_data or {}),
            "waypoint_sequence": extract_waypoint_sequence(chosen_route_data or {}),
            "route_polyline": [
                {
                    "longitude": float(point[0]),
                    "latitude": float(point[1]),
                }
                for point in ((chosen_route_data or {}).get("route_polyline") or [])
                if isinstance(point, (list, tuple)) and len(point) == 2
            ],
            "route_summary": chosen_route_summary,
            "meta": {
                "candidate_count": len(deduped),
                "evaluated_count": len(stage1_filtered) + len(stage1_rejected),
                "hard_filter_pass_count": len(stage1_filtered),
                "hard_filter_reject_count": len(stage1_rejected),
                "selection_mode": chosen_mode,
                "selection_radius_m": radius,
                "selection_top_k": top_k,
                "optimization_target": optimize_for,
                "destination_kind": destination_kind,
                "candidate_routing_mode": "estimated",
                "chosen_destination_routing_mode": "actual" if chosen_route_data else "estimated_fallback",
            },
            "hard_constraint_rejections": stage1_rejected,
            "route_failures": route_failures,
            "semantic_rerank": rerank_meta,
            "timestamp": datetime.now().isoformat(),
        }

        if english_localization_active:
            result_localization_started_at = perf_counter()
            translated_bundle = await translate_result_bundle_to_english(
                result_bundle={
                    "recommendations": [
                        {
                            "id": item["id"],
                            "name": item["name"],
                            "tags": {
                                "type": (item.get("tags") or {}).get("type", ""),
                                "address": (item.get("tags") or {}).get("address", ""),
                            },
                        }
                        for item in result["recommendations"]
                    ],
                    "chosen_destination": {
                        "id": result["chosen_destination"]["id"],
                        "name": result["chosen_destination"]["name"],
                        "tags": {
                            "type": (result["chosen_destination"].get("tags") or {}).get("type", ""),
                            "address": (result["chosen_destination"].get("tags") or {}).get("address", ""),
                        },
                    },
                    "semantic_rerank": result.get("semantic_rerank") or {},
                }
            )
            if translated_bundle:
                result = _apply_english_translations(result, translated_bundle)
                await _ctx_info(
                    ctx,
                    f"English result localization completed in {perf_counter() - result_localization_started_at:.2f}s.",
                )
            else:
                await _ctx_warning(
                    ctx,
                    f"English result localization failed after {perf_counter() - result_localization_started_at:.2f}s; returning provider-language text.",
                )

        await _ctx_info(ctx, f"Recommendation pipeline completed in {perf_counter() - overall_started_at:.2f}s.")

        return result


__all__ = [
    "DestinationRecommendationEngine",
    "apply_hard_constraints",
    "clamp_score",
    "extract_key_waypoints",
    "extract_waypoint_sequence",
    "haversine_m",
    "keyword_match_score",
    "soft_rank",
]
