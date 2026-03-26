"""Configuration helpers for map providers and intent parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os


def load_json_env(env_name: str, default: Any) -> Any:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def load_json_file(file_path: str, default: Any) -> Any:
    if not file_path:
        return default

    resolved_path = Path(file_path)
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path

    if not resolved_path.exists():
        return default

    try:
        return json.loads(resolved_path.read_text(encoding="utf-8"))
    except Exception:
        return default


def default_search_categories() -> list[str]:
    raw = os.getenv("DEFAULT_SEARCH_CATEGORIES", "amenity,shop,tourism,leisure").strip()
    categories = [item.strip() for item in raw.split(",") if item.strip()]
    return categories or ["amenity", "shop", "tourism", "leisure"]


def default_search_keywords() -> str:
    return "convenience store|cafe|bookstore|tea shop"


def default_category_keywords() -> dict[str, str]:
    return {
        "amenity": "restaurant|cafe|hospital|school|bank|bus station",
        "shop": "mall|supermarket|convenience store",
        "tourism": "attraction|museum|park",
        "leisure": "park|fitness|entertainment",
        "parking": "parking",
        "charging": "charging station",
    }


def default_map_i18n_rules() -> dict[str, Any]:
    return {
        "poi_type_rules": [
            {"label": "cafe", "tokens": ["cafe", "coffee", "\u5496\u5561"]},
            {"label": "bookstore", "tokens": ["bookstore", "library", "\u4e66\u5e97", "\u56fe\u4e66\u9986"]},
            {"label": "tea_shop", "tokens": ["tea", "milk tea", "\u8336", "\u5976\u8336"]},
            {"label": "convenience_store", "tokens": ["convenience", "\u4fbf\u5229\u5e97"]},
            {"label": "supermarket", "tokens": ["supermarket", "\u8d85\u5e02"]},
            {"label": "parking", "tokens": ["parking", "\u505c\u8f66"]},
            {"label": "ev_charging_station", "tokens": ["charging", "ev", "\u5145\u7535"]},
            {"label": "food_and_drink", "tokens": ["restaurant", "food", "dessert", "\u9910\u996e", "\u9910\u5385", "\u5feb\u9910", "\u751c\u54c1"]},
            {"label": "shopping", "tokens": ["shopping", "mall", "\u8d2d\u7269", "\u5546\u573a"]},
            {"label": "transportation", "tokens": ["transport", "bus", "subway", "train", "airport", "\u4ea4\u901a", "\u516c\u4ea4", "\u5730\u94c1", "\u706b\u8f66", "\u673a\u573a"]},
            {"label": "lodging", "tokens": ["hotel", "lodging", "\u9152\u5e97", "\u5bbe\u9986", "\u4f4f\u5bbf"]},
            {"label": "healthcare", "tokens": ["hospital", "pharmacy", "medical", "\u533b\u9662", "\u836f\u5e97", "\u533b\u7597"]},
            {"label": "education_and_culture", "tokens": ["school", "museum", "education", "culture", "\u5b66\u6821", "\u535a\u7269\u9986", "\u6559\u80b2", "\u6587\u5316", "\u79d1\u6559"]},
            {"label": "leisure_and_scenic", "tokens": ["park", "scenic", "attraction", "\u516c\u56ed", "\u98ce\u666f", "\u666f\u70b9"]},
            {"label": "financial_services", "tokens": ["bank", "finance", "insurance", "\u94f6\u884c", "\u91d1\u878d", "\u4fdd\u9669"]},
        ],
        "nav_action_aliases": {
            "arrive": ["arrive", "destination", "\u5230\u8fbe", "\u76ee\u7684\u5730"],
            "turn_left": ["left", "turn left", "\u5de6\u8f6c"],
            "turn_right": ["right", "turn right", "\u53f3\u8f6c"],
            "go_straight": ["straight", "continue", "\u76f4\u884c"],
            "bear_left": ["bear left", "\u5411\u5de6\u524d\u65b9"],
            "bear_right": ["bear right", "\u5411\u53f3\u524d\u65b9"],
            "keep_left": ["keep left", "\u9760\u5de6"],
            "keep_right": ["keep right", "\u9760\u53f3"],
            "u_turn": ["u-turn", "uturn", "\u8c03\u5934"],
        },
        "address_near_tokens": ["near", "\u8fd1"],
    }


def default_intent_rules() -> dict[str, Any]:
    return {
        "mode_tokens": {
            "foot": ["walk", "walking", "on foot"],
            "bike": ["bike", "bicycle", "cycling"],
            "car": ["drive", "driving", "car"],
        },
        "destination_rules": [
            {"type": "parking", "tokens": ["parking"]},
            {"type": "charging", "tokens": ["ev", "charging"]},
            {"type": "education", "tokens": ["school", "university", "college"]},
            {"type": "cafe", "tokens": ["cafe", "coffee"]},
            {"type": "restaurant", "tokens": ["restaurant", "food", "eat"]},
        ],
        "destination_keywords": {
            "parking": ["parking"],
            "charging": ["charging station", "ev"],
            "education": ["school", "university"],
            "cafe": ["cafe", "coffee"],
            "restaurant": ["restaurant", "food"],
        },
        "destination_search_keywords": {
            "parking": "parking lot|parking garage",
            "charging": "charging station|charging point",
            "education": "school|university|college",
            "cafe": "cafe|bookstore",
            "restaurant": "restaurant|food court",
            "poi": default_search_keywords(),
        },
        "time_optimization_tokens": ["fastest", "quick", "shortest time"],
        "nearby_tokens": ["nearby", "around me", "close"],
        "default_radius_m": 3000,
        "nearby_radius_m": 2000,
        "default_max_candidates": 5,
        "default_hard_constraints": {
            "max_route_duration_min": None,
            "max_route_distance_m": None,
            "must_include_keywords": [],
            "exclude_keywords": [],
        },
    }


def default_scoring_weights() -> dict[str, float]:
    return {
        "time": 0.5,
        "distance": 0.3,
        "proximity": 0.1,
        "keyword": 0.1,
    }


def default_few_shot_examples() -> list[dict[str, Any]]:
    return [
        {
            "request": "Find a quiet place to work nearby with coffee and good access",
            "output": {
                "destination_type": "workspace",
                "search_keywords": "cafe|coworking space|bookstore",
                "mode": "car",
                "radius_m": 2500,
                "optimize_for": "balanced",
                "max_candidates": 5,
                "hard_constraints": {
                    "max_route_duration_min": 25,
                    "max_route_distance_m": None,
                    "must_include_keywords": ["coffee"],
                    "exclude_keywords": [],
                },
            },
        },
        {
            "request": "Recommend an EV charging stop near me and avoid paid facilities",
            "output": {
                "destination_type": "charging",
                "search_keywords": "charging station|charging point",
                "mode": "car",
                "radius_m": 5000,
                "optimize_for": "distance",
                "max_candidates": 5,
                "hard_constraints": {
                    "max_route_duration_min": None,
                    "max_route_distance_m": None,
                    "must_include_keywords": ["charging"],
                    "exclude_keywords": ["paid"],
                },
            },
        },
        {
            "request": "Suggest a park for a short walk within 15 minutes",
            "output": {
                "destination_type": "park",
                "search_keywords": "park|garden|green space",
                "mode": "foot",
                "radius_m": 1800,
                "optimize_for": "time",
                "max_candidates": 5,
                "hard_constraints": {
                    "max_route_duration_min": 15,
                    "max_route_distance_m": None,
                    "must_include_keywords": ["park"],
                    "exclude_keywords": [],
                },
            },
        },
        {
            "request": "Find a grocery option for a bike trip with minimal route distance",
            "output": {
                "destination_type": "grocery",
                "search_keywords": "supermarket|convenience store|grocery",
                "mode": "bike",
                "radius_m": 3500,
                "optimize_for": "distance",
                "max_candidates": 5,
                "hard_constraints": {
                    "max_route_duration_min": None,
                    "max_route_distance_m": 5000,
                    "must_include_keywords": ["market"],
                    "exclude_keywords": [],
                },
            },
        },
    ]


def as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def split_search_keywords(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split("|") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def join_search_keywords(value: Any, default: str | None = None) -> str:
    parts = split_search_keywords(value)
    if parts:
        return "|".join(parts)
    return default or default_search_keywords()


def normalize_hard_constraints(raw: Any) -> dict[str, Any]:
    defaults = default_intent_rules()["default_hard_constraints"]
    if not isinstance(raw, dict):
        return defaults

    max_duration = raw.get("max_route_duration_min")
    if not isinstance(max_duration, (int, float)):
        max_duration = None

    max_distance = raw.get("max_route_distance_m")
    if not isinstance(max_distance, (int, float)):
        max_distance = None

    return {
        "max_route_duration_min": max_duration,
        "max_route_distance_m": max_distance,
        "must_include_keywords": as_str_list(raw.get("must_include_keywords")),
        "exclude_keywords": as_str_list(raw.get("exclude_keywords")),
    }


def normalize_few_shot_output(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    default_rules = default_intent_rules()
    destination_type = str(raw.get("destination_type", "poi")).strip() or "poi"
    search_keywords = join_search_keywords(
        raw.get("search_keywords") or raw.get("keywords"),
        default=default_rules.get("destination_search_keywords", {}).get(destination_type, default_search_keywords()),
    )

    mode = str(raw.get("mode", "car")).strip().lower()
    if mode not in {"car", "foot", "bike"}:
        mode = "car"

    radius_m = raw.get("radius_m", default_rules.get("default_radius_m", 3000))
    if not isinstance(radius_m, (int, float)) or radius_m <= 0:
        radius_m = default_rules.get("default_radius_m", 3000)

    optimize_for = str(raw.get("optimize_for", "distance")).strip().lower()
    if optimize_for not in {"time", "distance", "balanced"}:
        optimize_for = "distance"

    max_candidates = raw.get("max_candidates", default_rules.get("default_max_candidates", 5))
    if not isinstance(max_candidates, (int, float)) or max_candidates < 1:
        max_candidates = default_rules.get("default_max_candidates", 5)

    return {
        "destination_type": destination_type,
        "search_keywords": search_keywords,
        "mode": mode,
        "radius_m": int(radius_m),
        "optimize_for": optimize_for,
        "max_candidates": int(max_candidates),
        "hard_constraints": normalize_hard_constraints(raw.get("hard_constraints")),
    }


def normalize_few_shot_examples(raw_examples: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_examples, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_examples:
        if not isinstance(item, dict):
            continue

        request = str(item.get("request", "")).strip()
        output = normalize_few_shot_output(item.get("output"))
        if request and output:
            normalized.append({"request": request, "output": output})

    max_examples = int(os.getenv("LLM_FEW_SHOT_MAX_EXAMPLES", "6"))
    if max_examples < 1:
        max_examples = 6
    return normalized[:max_examples]


def get_few_shot_examples() -> list[dict[str, Any]]:
    defaults = default_few_shot_examples()
    default_file = "config/llm_few_shot_examples.json"
    file_path = os.getenv("LLM_FEW_SHOT_EXAMPLES_FILE", default_file).strip() or default_file

    file_examples = normalize_few_shot_examples(load_json_file(file_path, []))
    if file_examples:
        return file_examples

    env_examples = normalize_few_shot_examples(load_json_env("LLM_FEW_SHOT_EXAMPLES_JSON", defaults))
    if env_examples:
        return env_examples

    return normalize_few_shot_examples(defaults)


__all__ = [
    "as_str_list",
    "default_category_keywords",
    "default_few_shot_examples",
    "default_intent_rules",
    "default_search_keywords",
    "default_scoring_weights",
    "default_search_categories",
    "get_few_shot_examples",
    "join_search_keywords",
    "load_json_env",
    "load_json_file",
    "normalize_few_shot_examples",
    "normalize_few_shot_output",
    "normalize_hard_constraints",
    "split_search_keywords",
]
