"""Policy responsibilities for destination recommendation."""
from __future__ import annotations

from typing import Any
import json
import os
import re
from mcp_server.config import split_search_keywords
from mcp_server.llm import extract_explicit_destination_query, infer_destination_kind
from mcp_server.recommendation_services.ranking import clamp_score
from mcp_server.recommendation_services.geometry import haversine_m

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


def _candidate_unavailable_reason(candidate: dict[str, Any]) -> str:
    tags = candidate.get("tags", {}) or {}
    raw_candidate = candidate.get("raw_candidate", {}) or {}
    raw = raw_candidate.get("raw")
    text_parts = [
        str(candidate.get("name", "")),
        str(tags.get("type", "")),
        str(tags.get("address", "")),
        str(raw_candidate.get("type", "")),
        str(raw_candidate.get("address", "")),
        str(raw_candidate.get("business_status", "")),
        str(raw_candidate.get("status", "")),
    ]
    if isinstance(raw, dict):
        text_parts.extend(
            [
                str(raw.get("business_status", "")),
                str(raw.get("status", "")),
                str(raw.get("shopinfo", "")),
                str(raw.get("type", "")),
                str(raw.get("address", "")),
            ]
        )
    haystack = " ".join(part for part in text_parts if part).lower()
    if not haystack:
        return ""
    restricted_tokens = (
        "temporarily closed",
        "closed",
        "not open",
        "暂停营业",
        "歇业",
        "停止营业",
        "休业",
        "打烊",
        "不对外开放",
    )
    for token in restricted_tokens:
        if token.lower() in haystack:
            return token
    return ""


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


def _normalize_text_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _scope_limited_radius_m(radius_m: float, *, mode: str, resolution_scope: str) -> float:
    if str(resolution_scope or "").strip().lower() != "session_local_via":
        return float(radius_m)
    defaults = {
        "foot": float(os.getenv("LOCAL_VIA_MAX_RADIUS_M_FOOT", "1800") or 3800),
        "bike": float(os.getenv("LOCAL_VIA_MAX_RADIUS_M_BIKE", "3000") or 6000),
        "car": float(os.getenv("LOCAL_VIA_MAX_RADIUS_M_CAR", "5000") or 10000),
    }
    return min(float(radius_m), defaults.get(str(mode or "").strip().lower(), 1800.0))


def _confidence_value(value: Any, default: float = 0.0) -> float:
    try:
        return clamp_score(float(value), 0.0, 1.0)
    except Exception:
        return clamp_score(default, 0.0, 1.0)


def _parsed_confidence(parsed: dict[str, Any]) -> float:
    return _confidence_value(parsed.get("confidence"), default=0.5)


def _parsed_named_entity_confidence(
    *,
    user_instruction: str,
    parsed: dict[str, Any],
) -> float:
    explicit_query = extract_explicit_destination_query(user_instruction)
    default_confidence = 0.82 if explicit_query else 0.08
    return _confidence_value(parsed.get("named_entity_confidence"), default=default_confidence)


def _should_use_named_place_geocode(
    *,
    resolution_scope: str,
    explicit_destination: bool,
    destination_kind: str,
    named_entity_confidence: float,
) -> bool:
    if not explicit_destination:
        return False
    if destination_kind not in {
        "named_landmark",
        "specific_poi",
        "regional_scenic_area",
        "administrative_area",
        "road_or_route_label",
    }:
        return False
    scope_key = str(resolution_scope or "").strip().lower()
    threshold_env = (
        "RECOMMEND_LOCAL_VIA_NAMED_GEOCODE_CONFIDENCE_THRESHOLD"
        if scope_key == "session_local_via"
        else "RECOMMEND_GLOBAL_NAMED_GEOCODE_CONFIDENCE_THRESHOLD"
    )
    threshold_default = 0.8 if scope_key == "session_local_via" else 0.55
    threshold = float(os.getenv(threshold_env, str(threshold_default)) or threshold_default)
    return float(named_entity_confidence) >= threshold


def _should_enable_llm_rerank(
    *,
    explicit_destination: bool,
    resolution_scope: str,
    parsed_confidence: float,
    intent_source: str,
    candidate_count: int,
) -> tuple[bool, str]:
    if not explicit_destination:
        return False, "not_explicit_destination"
    if candidate_count < 2:
        return False, "insufficient_candidates"
    if str(intent_source or "").startswith("heuristic"):
        return True, "heuristic_parse_fallback"
    scope_key = str(resolution_scope or "").strip().lower()
    threshold_env = (
        "RECOMMEND_LOCAL_VIA_RERANK_CONFIDENCE_THRESHOLD"
        if scope_key == "session_local_via"
        else "RECOMMEND_GLOBAL_RERANK_CONFIDENCE_THRESHOLD"
    )
    threshold_default = 0.55 if scope_key == "session_local_via" else 0.62
    threshold = float(os.getenv(threshold_env, str(threshold_default)) or threshold_default)
    if float(parsed_confidence) < threshold:
        return True, "low_parse_confidence"
    return False, "high_parse_confidence"


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


def _session_local_via_bias(
    candidate: dict[str, Any],
    *,
    direct_distance_m: float,
    search_radius_m: float,
) -> float:
    candidate_kind = _normalize_text_key(candidate.get("candidate_kind", ""))
    bias = 0.0
    if candidate.get("is_concrete_destination"):
        bias += 0.08
    if candidate_kind in {"administrative_area", "road_label", "institution", "commercial_place"}:
        bias -= 0.18
    proximity_ratio = 1.0 - min(max(float(direct_distance_m), 0.0), max(float(search_radius_m), 1.0)) / max(float(search_radius_m), 1.0)
    bias += 0.16 * clamp_score(proximity_ratio)
    bias += 0.18 * clamp_score(float(candidate.get("query_affinity", 0.0) or 0.0))
    return bias
