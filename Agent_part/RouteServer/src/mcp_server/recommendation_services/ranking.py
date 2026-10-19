"""Ranking responsibilities for destination recommendation."""
from __future__ import annotations

from typing import Any
import os
import re
from mcp_server.config import default_scoring_weights, load_json_env

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
