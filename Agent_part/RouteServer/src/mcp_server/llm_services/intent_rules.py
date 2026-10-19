"""Intent rules responsibilities for language and vision services."""
from __future__ import annotations

from typing import Any
import re

def _clamp_probability(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except Exception:
        return max(0.0, min(float(default), 1.0))


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def _looks_english_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value or _contains_cjk(value):
        return False
    alpha_count = len(re.findall(r"[A-Za-z]", value))
    return alpha_count >= 4


def extract_explicit_destination_query(user_instruction: str) -> str:
    text = str(user_instruction or "").strip()
    if not text:
        return ""

    english_patterns = [
        r"\b(?:go\s+to|head\s+to|visit|reach)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9\s\-\']{1,80})",
        r"\b(?:want\s+to\s+go\s+to|would\s+like\s+to\s+go\s+to)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9\s\-\']{1,80})",
    ]
    for pattern in english_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?:;")
        candidate = re.sub(r"\b(?:nearby|around me|close by|please|now)$", "", candidate, flags=re.IGNORECASE).strip()
        if candidate:
            return candidate

    chinese_patterns = [
        r"(?:\u60f3\u53bb|\u524d\u5f80|\u53bb)([\u3400-\u9fffA-Za-z0-9\s]{1,40})",
    ]
    for pattern in chinese_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?:;\u3002\uff01\uff1f")
        if candidate:
            return candidate

    return ""


def infer_destination_kind(
    *,
    user_instruction: str,
    destination_type: str = "",
    search_keywords: str = "",
) -> str:
    explicit_query = extract_explicit_destination_query(user_instruction)
    query = explicit_query or str(search_keywords or "").strip()
    query_key = re.sub(r"\s+", " ", query).strip().lower()
    destination_key = re.sub(r"\s+", " ", str(destination_type or "").strip()).lower()

    admin_tokens = (
        "district",
        "county",
        "province",
        "city",
        "township",
        "subdistrict",
        "\u533a",
        "\u53bf",
        "\u7701",
        "\u5e02",
        "\u8857\u9053",
        "\u9547",
    )
    road_tokens = (
        "road",
        "street",
        "avenue",
        "lane",
        "route",
        "\u8def",
        "\u8857",
        "\u5927\u8857",
        "\u5df7",
        "\u9053",
    )
    scenic_region_tokens = (
        "lake",
        "park",
        "scenic",
        "scenic area",
        "mountain",
        "beach",
        "island",
        "campus",
        "old town",
        "resort",
        "\u6e56",
        "\u516c\u56ed",
        "\u666f\u533a",
        "\u98ce\u666f",
        "\u5c71",
        "\u6d77\u6ee9",
        "\u5c9b",
        "\u6821\u533a",
        "\u53e4\u9547",
    )
    generic_poi_tokens = (
        "cafe",
        "coffee",
        "restaurant",
        "parking",
        "charging",
        "pharmacy",
        "bookstore",
        "store",
        "shop",
        "hospital",
        "\u5496\u5561",
        "\u9910\u5385",
        "\u505c\u8f66",
        "\u5145\u7535",
        "\u836f\u5e97",
        "\u4e66\u5e97",
        "\u5546\u5e97",
    )

    if query_key and any(token in query_key for token in admin_tokens):
        return "administrative_area"
    if query_key and any(token in query_key for token in road_tokens):
        return "road_or_route_label"
    if explicit_query and query_key and any(token in query_key for token in scenic_region_tokens):
        return "regional_scenic_area"
    if destination_key in {"tourist attraction", "tourist_attraction", "scenic attraction", "scenic_attraction"}:
        return "regional_scenic_area" if explicit_query else "specific_poi"
    if explicit_query:
        return "named_landmark"
    if query_key and any(token in query_key for token in generic_poi_tokens):
        return "generic_poi"
    return "generic_poi"


def _heuristic_named_entity_confidence(
    *,
    user_instruction: str,
    destination_type: str,
    destination_kind: str,
    search_keywords: str,
) -> float:
    explicit_query = extract_explicit_destination_query(user_instruction)
    if not explicit_query:
        return 0.08

    score = 0.55
    destination_type_key = re.sub(r"\s+", " ", str(destination_type or "").strip()).lower()
    destination_kind_key = re.sub(r"\s+", " ", str(destination_kind or "").strip()).lower()
    query = str(search_keywords or explicit_query).strip()
    if destination_type_key in {"named_place", "named_landmark"}:
        score += 0.15
    if destination_kind_key in {"named_landmark", "specific_poi", "regional_scenic_area"}:
        score += 0.18
    if len(explicit_query) >= 4:
        score += 0.04
    if query and explicit_query and query.strip().lower() == explicit_query.strip().lower():
        score += 0.05
    return _clamp_probability(score, default=0.08)


def _heuristic_parse_confidence(
    *,
    user_instruction: str,
    destination_type: str,
    destination_kind: str,
) -> float:
    score = 0.62
    text = str(user_instruction or "").strip().lower()
    if extract_explicit_destination_query(user_instruction):
        score += 0.14
    if str(destination_kind or "").strip().lower() in {
        "generic_poi",
        "named_landmark",
        "regional_scenic_area",
        "specific_poi",
    }:
        score += 0.06
    if str(destination_type or "").strip().lower() in {"poi", "named_place", "named_landmark"}:
        score += 0.04
    if any(token in text for token in (" or ", "/", "或者", "还是", "maybe", "perhaps")):
        score -= 0.16
    return _clamp_probability(score, default=0.62)


def _cjk_heuristic_intent(user_instruction: str) -> dict[str, str] | None:
    """Map common Chinese implicit needs to provider-facing search keywords.

    Args:
        user_instruction: User instruction or decomposed sub-goal.

    Returns:
        Destination type and search keywords, or ``None`` when no rule matches.
    """

    text = str(user_instruction or "")
    rules: list[tuple[str, str, str]] = [
        (r"饮料|喝的|饮品|奶茶|水", "drink_retail", "饮品|咖啡|便利店|超市"),
        (r"咖啡|买杯咖啡|喝咖啡", "cafe", "咖啡|咖啡店"),
        (r"吃饭|吃点|餐|小吃|快餐|食堂|美食", "restaurant", "餐厅|小吃|快餐|食堂"),
        (r"银行|ATM|取钱|现金|自助银行", "bank", "银行|ATM|自助银行|中国银行"),
        (r"药|不舒服|看诊|诊所|医院", "medical", "药店|医药|诊所|医院"),
        (r"办公|看书|自习|安静|图书馆|书店", "workspace", "图书馆|书店|咖啡|共享办公"),
        (r"日用品|饮料|便利店|超市|小卖部|零食", "retail", "便利店|超市|生活超市|小卖部"),
        (r"休息|坐下|坐一会|歇", "rest_stop", "咖啡|公园|广场|茶馆"),
    ]
    for pattern, destination_type, keywords in rules:
        if re.search(pattern, text):
            return {"destination_type": destination_type, "search_keywords": keywords}
    return None


DESTINATION_KIND_VALUES = {
    "generic_poi",
    "named_landmark",
    "regional_scenic_area",
    "administrative_area",
    "road_or_route_label",
    "specific_poi",
}
