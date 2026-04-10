"""Intent parsing and prompt-engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import ast
import base64
import inspect
import json
import mimetypes
import os
import re
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from mcp_server.config import (
    default_intent_rules,
    default_search_keywords,
    default_search_categories,
    get_few_shot_examples,
    join_search_keywords,
    load_json_env,
    split_search_keywords,
)


JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
DESTINATION_KIND_VALUES = {
    "generic_poi",
    "named_landmark",
    "regional_scenic_area",
    "administrative_area",
    "road_or_route_label",
    "specific_poi",
}


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def _looks_english_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value or _contains_cjk(value):
        return False
    alpha_count = len(re.findall(r"[A-Za-z]", value))
    return alpha_count >= 4


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_style: str
    api_key: str
    model: str
    base_url: str
    temperature: float = 0.0
    timeout_s: float = 20.0
    response_format: bool = True

    @classmethod
    def from_env(cls) -> "LLMSettings":
        explicit_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        ark_api_key = os.getenv("ARK_API_KEY", "").strip()
        provider = explicit_provider or ("ark" if ark_api_key else "openai")
        default_model = "doubao-seed-2-0-pro-260215" if provider == "ark" else "gpt-4o-mini"
        default_base_url = (
            "https://ark.cn-beijing.volces.com/api/v3"
            if provider == "ark"
            else "https://api.openai.com/v1"
        )
        api_key = os.getenv("LLM_API_KEY", "").strip() or ark_api_key
        api_style = os.getenv("LLM_API_STYLE", "").strip().lower() or (
            "chat_completions" if provider == "ark" else "chat_completions"
        )
        if api_style not in {"responses", "chat_completions"}:
            api_style = "chat_completions" if provider == "ark" else "chat_completions"
        return cls(
            provider=provider,
            api_style=api_style,
            api_key=api_key,
            model=os.getenv("LLM_MODEL", default_model).strip(),
            base_url=os.getenv("LLM_API_BASE_URL", default_base_url).rstrip("/"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0") or 0),
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "20") or 20),
            response_format=os.getenv("LLM_RESPONSE_FORMAT_JSON", "true").strip().lower() not in {"0", "false", "no"},
        )


def _extract_text_from_chat_response(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            value = getattr(text, "value", None)
            if isinstance(value, str):
                parts.append(value)
        return "\n".join(parts)
    return ""


def _extract_text_from_responses_response(response: Any) -> str:
    output_text = getattr(response, "output_text", "")
    if isinstance(output_text, str) and output_text:
        return output_text

    parts = []
    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            text = getattr(content_item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            value = getattr(text, "value", None)
            if isinstance(value, str) and value:
                parts.append(value)

    if parts:
        return "\n".join(parts)

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return json.dumps(model_dump(), ensure_ascii=False)
    return ""


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _iter_balanced_json_objects(text: str, max_objects: int = 20) -> list[str]:
    objects: list[str] = []
    if not text:
        return objects

    depth = 0
    start_index: int | None = None
    in_string = False
    quote_char = ""
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote_char:
                in_string = False
                quote_char = ""
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue

        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = text[start_index : index + 1].strip()
                if candidate:
                    objects.append(candidate)
                    if len(objects) >= max_objects:
                        break
                start_index = None

    return objects


def _parse_candidate_json_object(candidate: str) -> dict[str, Any] | None:
    text = str(candidate or "").strip()
    if not text:
        return None

    parse_attempts = [text, _remove_trailing_commas(text)]
    for attempt in parse_attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    # Last-resort parser for Python-dict-like outputs from LLMs.
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        candidate = str(raw or "").strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    raw_text = str(text or "").strip()
    _push(raw_text)

    for match in JSON_FENCE_PATTERN.finditer(raw_text):
        _push(match.group(1))

    for candidate in _iter_balanced_json_objects(raw_text):
        _push(candidate)

    return candidates


def _parse_json_object_from_text(text: str) -> dict[str, Any]:
    candidates = _extract_json_candidates(text)
    for candidate in candidates:
        parsed = _parse_candidate_json_object(candidate)
        if parsed is not None:
            return parsed

    preview = str(text or "").strip().replace("\n", " ")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    raise ValueError(f"No valid JSON object parsed from LLM response: {preview}")


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


def _create_async_client(settings: LLMSettings) -> Any:
    if AsyncOpenAI is None:
        return None
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_s,
        max_retries=0,
    )


def _create_async_client_with_timeout(settings: LLMSettings, timeout_s: float) -> Any:
    if AsyncOpenAI is None:
        return None
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=timeout_s,
        max_retries=0,
    )


async def _close_async_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        maybe_awaitable = close()
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


class BaseLLMService:
    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()

    def is_available(self) -> bool:
        return bool(self.settings.api_key and AsyncOpenAI is not None)


class IntentParser(BaseLLMService):
    """Parse recommendation instructions with an LLM and a heuristic fallback."""

    async def _call_llm_with_fallback(self, client: Any, user_instruction: str) -> tuple[str, str]:
        preferred_style = self.settings.api_style
        fallback_style = "responses" if preferred_style == "chat_completions" else "chat_completions"
        styles = [preferred_style, fallback_style]

        errors: list[str] = []
        for style in styles:
            try:
                if style == "responses":
                    content = await self._call_responses_api(client, user_instruction)
                else:
                    content = await self._call_chat_completions(client, user_instruction)

                if str(content or "").strip():
                    return content, style
                errors.append(f"{style}: empty_content")
            except Exception as exc:
                errors.append(f"{style}: {exc}")

        raise RuntimeError(" | ".join(errors) if errors else "unknown_llm_call_error")

    def _extract_explicit_destination_query(self, user_instruction: str) -> str:
        return extract_explicit_destination_query(user_instruction)

    def heuristic_parse(self, user_instruction: str) -> dict[str, Any]:
        rules = load_json_env("INTENT_RULES_JSON", default_intent_rules())
        text = user_instruction.lower()

        mode = "car"
        for candidate_mode, tokens in rules.get("mode_tokens", {}).items():
            if any(token.lower() in text for token in tokens):
                mode = candidate_mode
                break

        destination_type = "poi"
        search_keywords = rules.get("destination_search_keywords", {}).get("poi", default_search_keywords())

        explicit_query = self._extract_explicit_destination_query(user_instruction)
        if explicit_query:
            destination_type = "named_place"
            search_keywords = explicit_query
        else:
            for destination_rule in rules.get("destination_rules", []):
                tokens = destination_rule.get("tokens", [])
                if any(token.lower() in text for token in tokens):
                    destination_type = destination_rule.get("type", "poi")
                    search_keywords = rules.get("destination_search_keywords", {}).get(destination_type, search_keywords)
                    break

        optimize_for = "distance"
        if any(token.lower() in text for token in rules.get("time_optimization_tokens", [])):
            optimize_for = "time"

        radius = rules.get("default_radius_m", 3000)
        if any(token.lower() in text for token in rules.get("nearby_tokens", [])):
            radius = rules.get("nearby_radius_m", 2000)

        return {
            "destination_type": destination_type,
            "destination_kind": infer_destination_kind(
                user_instruction=user_instruction,
                destination_type=destination_type,
                search_keywords=search_keywords,
            ),
            "search_keywords": search_keywords,
            "mode": mode,
            "radius_m": radius,
            "optimize_for": optimize_for,
            "max_candidates": rules.get("default_max_candidates", 5),
            "hard_constraints": rules.get("default_hard_constraints", {}),
            "intent_source": "heuristic",
        }

    def build_messages(self, user_instruction: str) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                "You are an intent parser for map recommendation and routing. "
                "Return JSON only. The JSON schema must contain exactly these top-level keys: "
                "destination_type, destination_kind, search_keywords, mode, radius_m, optimize_for, "
                "max_candidates, hard_constraints. "
                "Allowed destination_kind values: generic_poi, named_landmark, regional_scenic_area, "
                "administrative_area, road_or_route_label, specific_poi. "
                "Allowed mode values: car, foot, bike. "
                "Allowed optimize_for values: time, distance, balanced. "
                "radius_m and max_candidates must be integers. "
                "search_keywords must be a provider-facing keyword string, preferably pipe-delimited, "
                "for example 'bookstore|cafe' or 'convenience store|tea shop'. "
                "If the user gives an explicit destination name (for example 'West Lake' or 'Times Square'), "
                "set destination_type appropriately, use destination_kind=named_landmark or "
                "destination_kind=regional_scenic_area when the destination is a broad scenic region, and put that destination name directly into "
                "search_keywords instead of generic categories. "
                "For broad need-based requests such as thirsty, hungry, tired, need coffee, need groceries, or need a restroom, "
                "output broad retrievable provider-facing search_keywords that cover several suitable POI types. "
                "Use hard_constraints only for constraints that are explicitly stated by the user. "
                "Do not invent max route duration, max route distance, must_include_keywords, or exclude_keywords from vague preferences alone. "
                "hard_constraints must include keys: max_route_duration_min, max_route_distance_m, "
                "must_include_keywords, exclude_keywords. "
                "Do not add explanation or markdown."
                ),
            }
        ]

        for example in get_few_shot_examples():
            request = example.get("request", "")
            output = example.get("output", {})
            if not request or not output:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Parse this request into domestic-map recommendation constraints. "
                        "Use provider-facing search_keywords instead of category taxonomy. "
                        f"request: {request}"
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(output, ensure_ascii=True),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    "Parse this request into domestic-map recommendation constraints. "
                    "Prefer a concise provider-facing search_keywords string that can be sent directly to "
                    "AMap/Baidu nearby search APIs. Avoid category taxonomy. "
                    f"request: {user_instruction}"
                ),
            }
        )
        return messages

    def build_prompt(self, user_instruction: str) -> str:
        sections = [self.build_messages(user_instruction)[0]["content"], "", "Few-shot examples:"]

        for example in get_few_shot_examples():
            request = example.get("request", "")
            output = example.get("output", {})
            if not request or not output:
                continue
            sections.append(f"Request: {request}")
            sections.append(f"JSON: {json.dumps(output, ensure_ascii=True)}")

        sections.extend(
            [
                "",
                "Now parse the following request and return JSON only.",
                f"Request: {user_instruction}",
            ]
        )
        return "\n".join(sections)

    async def _call_chat_completions(self, client: Any, user_instruction: str) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": self.build_messages(user_instruction),
        }
        # ARK chat completions can reject JSON response_format on some endpoints/models.
        # Keep the prompt JSON-only and fall back to regex extraction instead.
        if self.settings.response_format and self.settings.provider != "ark":
            payload["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**payload)
        return _extract_text_from_chat_response(response)

    async def _call_responses_api(self, client: Any, user_instruction: str) -> str:
        response = await client.responses.create(
            model=self.settings.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self.build_prompt(user_instruction),
                        }
                    ],
                }
            ],
        )
        return _extract_text_from_responses_response(response)

    def _normalize_llm_output(self, user_instruction: str, parsed: dict[str, Any]) -> dict[str, Any]:
        heuristic_defaults = self.heuristic_parse(user_instruction)
        normalized = dict(heuristic_defaults)
        normalized.update(parsed)

        mode = str(normalized.get("mode", "car")).strip().lower()
        if mode not in {"car", "foot", "bike"}:
            normalized["mode"] = "car"
        else:
            normalized["mode"] = mode

        optimize_for = str(normalized.get("optimize_for", "distance")).strip().lower()
        if optimize_for not in {"time", "distance", "balanced"}:
            normalized["optimize_for"] = "distance"
        else:
            normalized["optimize_for"] = optimize_for

        destination_type = str(normalized.get("destination_type", "poi")).strip() or "poi"
        destination_kind = str(normalized.get("destination_kind", "")).strip().lower()
        if destination_kind not in DESTINATION_KIND_VALUES:
            destination_kind = infer_destination_kind(
                user_instruction=user_instruction,
                destination_type=destination_type,
                search_keywords=str(normalized.get("search_keywords") or normalized.get("keywords") or ""),
            )
        normalized["destination_kind"] = destination_kind
        normalized["search_keywords"] = join_search_keywords(
            normalized.get("search_keywords") or normalized.get("keywords"),
            default=default_intent_rules().get("destination_search_keywords", {}).get(
                destination_type,
                default_search_keywords(),
            ),
        )
        normalized["categories"] = default_search_categories()
        normalized["keywords"] = split_search_keywords(normalized["search_keywords"])

        radius = normalized.get("radius_m", 3000)
        if not isinstance(radius, (int, float)) or radius <= 0:
            normalized["radius_m"] = heuristic_defaults["radius_m"]
        else:
            normalized["radius_m"] = int(radius)

        max_candidates = normalized.get("max_candidates", 5)
        if not isinstance(max_candidates, (int, float)) or max_candidates < 1:
            normalized["max_candidates"] = heuristic_defaults["max_candidates"]
        else:
            normalized["max_candidates"] = int(max_candidates)

        hard_constraints = normalized.get("hard_constraints")
        if not isinstance(hard_constraints, dict):
            normalized["hard_constraints"] = default_intent_rules()["default_hard_constraints"]
        else:
            normalized["hard_constraints"] = {
                "max_route_duration_min": hard_constraints.get("max_route_duration_min")
                if isinstance(hard_constraints.get("max_route_duration_min"), (int, float))
                else None,
                "max_route_distance_m": hard_constraints.get("max_route_distance_m")
                if isinstance(hard_constraints.get("max_route_distance_m"), (int, float))
                else None,
                "must_include_keywords": [
                    str(item).strip()
                    for item in hard_constraints.get("must_include_keywords", [])
                    if str(item).strip()
                ],
                "exclude_keywords": [
                    str(item).strip()
                    for item in hard_constraints.get("exclude_keywords", [])
                    if str(item).strip()
                ],
            }

        return normalized

    async def parse(self, user_instruction: str) -> dict[str, Any]:
        if not self.is_available():
            parsed = self.heuristic_parse(user_instruction)
            parsed["intent_source"] = "heuristic_no_llm"
            return parsed

        client = _create_async_client(self.settings)
        try:
            try:
                content, api_style_used = await self._call_llm_with_fallback(client, user_instruction)
            except Exception as exc:
                parsed = self.heuristic_parse(user_instruction)
                parsed["intent_source"] = "heuristic_llm_api_error"
                parsed["llm_error"] = str(exc)
                return parsed

            try:
                parsed = _parse_json_object_from_text(content)
            except Exception as exc:
                fallback = self.heuristic_parse(user_instruction)
                fallback["intent_source"] = "heuristic_llm_parse_error"
                fallback["llm_error"] = str(exc)
                return fallback

            normalized = self._normalize_llm_output(user_instruction, parsed)
            normalized["intent_source"] = "llm"
            normalized["llm_api_style_used"] = api_style_used
            return normalized
        except Exception:
            parsed = self.heuristic_parse(user_instruction)
            parsed["intent_source"] = "heuristic_llm_error"
            return parsed
        finally:
            await _close_async_client(client)


class CandidateReranker(BaseLLMService):
    """Rerank map candidates by semantic relevance to the user request."""

    def build_messages(
        self,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        simplified_candidates = [
            {
                "id": str(item.get("id", "")),
                "name": str(item.get("name", "")),
                "type": str(item.get("type", "")),
                "address": str(item.get("address", "")),
                "distance_m": item.get("distance_m"),
                "candidate_kind": str(item.get("candidate_kind", "")),
                "is_concrete_destination": bool(item.get("is_concrete_destination", False)),
                "query_affinity": item.get("query_affinity"),
                "anchor_name": str(item.get("anchor_name", "")),
                "anchor_distance_m": item.get("anchor_distance_m"),
            }
            for item in candidates
        ]
        return [
            {
                "role": "system",
                "content": (
                    "You rank map destination candidates for an embodied navigation system. "
                    "Return JSON only with exactly two keys: ranked_candidate_ids and summary. "
                    "ranked_candidate_ids must be an array of candidate ids ordered best to worst. "
                    "Prefer destinations that truly satisfy the user's semantic intent. "
                    "If the user asks for a landmark, scenic area, lake, park, tourist attraction, or named place, "
                    "rank concrete navigable scenic destinations ahead of administrative regions, roads, "
                    "institutions, hotels, stores, residences, or branded places that only contain the same "
                    "words in their names or addresses. For regional scenic areas, prioritize the destination "
                    "itself or canonical sub-features directly attached to that area. Do not rank merely nearby "
                    "scenic places highly unless the candidate is clearly part of the requested area."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_instruction": user_instruction,
                        "parsed_intent": parsed_intent,
                        "candidates": simplified_candidates,
                    },
                    ensure_ascii=True,
                ),
            },
        ]

    async def _call_chat_completions(
        self,
        client: Any,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> str:
        response = await client.chat.completions.create(
            model=self.settings.model,
            temperature=self.settings.temperature,
            messages=self.build_messages(
                user_instruction=user_instruction,
                parsed_intent=parsed_intent,
                candidates=candidates,
            ),
        )
        return _extract_text_from_chat_response(response)

    async def _call_responses_api(
        self,
        client: Any,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> str:
        response = await client.responses.create(
            model=self.settings.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "instruction": (
                                        "Rerank these destination candidates by semantic relevance and "
                                        "return JSON only with ranked_candidate_ids and summary."
                                    ),
                                    "user_instruction": user_instruction,
                                    "parsed_intent": parsed_intent,
                                    "candidates": candidates,
                                },
                                ensure_ascii=True,
                            ),
                        }
                    ],
                }
            ],
        )
        return _extract_text_from_responses_response(response)

    async def _call_llm_with_fallback(
        self,
        client: Any,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str, str]:
        preferred_style = self.settings.api_style
        fallback_style = "responses" if preferred_style == "chat_completions" else "chat_completions"
        styles = [preferred_style, fallback_style]

        errors: list[str] = []
        for style in styles:
            try:
                if style == "responses":
                    content = await self._call_responses_api(
                        client,
                        user_instruction=user_instruction,
                        parsed_intent=parsed_intent,
                        candidates=candidates,
                    )
                else:
                    content = await self._call_chat_completions(
                        client,
                        user_instruction=user_instruction,
                        parsed_intent=parsed_intent,
                        candidates=candidates,
                    )
                if str(content or "").strip():
                    return content, style
                errors.append(f"{style}: empty_content")
            except Exception as exc:
                errors.append(f"{style}: {exc}")

        raise RuntimeError(" | ".join(errors) if errors else "unknown_llm_call_error")

    async def rerank(
        self,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not candidates or not self.is_available():
            return None

        client = _create_async_client(self.settings)
        try:
            content, api_style_used = await self._call_llm_with_fallback(
                client,
                user_instruction=user_instruction,
                parsed_intent=parsed_intent,
                candidates=candidates,
            )
            parsed = _parse_json_object_from_text(content)
        except Exception:
            return None
        finally:
            await _close_async_client(client)

        ranked_ids = [
            str(item).strip()
            for item in parsed.get("ranked_candidate_ids", [])
            if str(item).strip()
        ]
        if not ranked_ids:
            return None
        return {
            "ranked_candidate_ids": ranked_ids,
            "summary": str(parsed.get("summary", "")).strip(),
            "api_style": api_style_used,
        }


class VisionAnalyzer(BaseLLMService):
    """Analyze an image through the configured multimodal LLM endpoint."""

    def _vision_timeout_s(self) -> float:
        raw = os.getenv("LLM_VISION_TIMEOUT_S", "").strip()
        if raw:
            try:
                return max(5.0, float(raw))
            except ValueError:
                pass
        return max(60.0, self.settings.timeout_s)

    def build_messages(
        self,
        image_url: str,
        question: str,
        detail: str = "auto",
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        prompt = system_prompt or (
            "You are a visual assistant. Answer the user's question about the image directly and concisely. "
            "If the image is unclear or inaccessible, say so plainly."
        )
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": detail,
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]

    async def _call_chat_completions(
        self,
        client: Any,
        image_url: str,
        question: str,
        detail: str,
        system_prompt: str | None,
    ) -> str:
        response = await client.chat.completions.create(
            model=self.settings.model,
            temperature=self.settings.temperature,
            messages=self.build_messages(image_url, question, detail, system_prompt),
        )
        return _extract_text_from_chat_response(response)

    async def _call_responses_api(
        self,
        client: Any,
        image_url: str,
        question: str,
        detail: str,
        system_prompt: str | None,
    ) -> str:
        prompt = system_prompt or (
            "You are a visual assistant. Answer the user's question about the image directly and concisely."
        )
        response = await client.responses.create(
            model=self.settings.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url, "detail": detail},
                        {"type": "input_text", "text": question},
                    ],
                }
            ],
        )
        return _extract_text_from_responses_response(response)

    async def analyze(
        self,
        image_url: str,
        question: str = "What is in this image?",
        detail: str = "auto",
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        if not image_url.strip():
            raise ValueError("image_url is required")
        if detail not in {"auto", "low", "high"}:
            detail = "auto"

        if not self.is_available():
            raise RuntimeError("LLM image analysis is not configured. Set ARK_API_KEY or LLM_API_KEY.")

        client = _create_async_client_with_timeout(self.settings, self._vision_timeout_s())
        try:
            if self.settings.api_style == "responses":
                answer = await self._call_responses_api(client, image_url, question, detail, system_prompt)
            else:
                answer = await self._call_chat_completions(client, image_url, question, detail, system_prompt)

            return {
                "provider": self.settings.provider,
                "api_style": self.settings.api_style,
                "model": self.settings.model,
                "image_url": image_url,
                "question": question,
                "detail": detail,
                "answer": answer.strip(),
            }
        finally:
            await _close_async_client(client)


class MapLocalizationService(BaseLLMService):
    """Translate provider-facing map instructions and result text."""

    async def _call_chat_completions(
        self,
        client: Any,
        *,
        messages: list[dict[str, Any]],
    ) -> str:
        response = await client.chat.completions.create(
            model=self.settings.model,
            temperature=self.settings.temperature,
            messages=messages,
        )
        return _extract_text_from_chat_response(response)

    async def _call_responses_api(
        self,
        client: Any,
        *,
        prompt: str,
    ) -> str:
        response = await client.responses.create(
            model=self.settings.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )
        return _extract_text_from_responses_response(response)

    async def _call_llm_with_fallback(
        self,
        client: Any,
        *,
        messages: list[dict[str, Any]],
        prompt: str,
    ) -> str:
        preferred_style = self.settings.api_style
        fallback_style = "responses" if preferred_style == "chat_completions" else "chat_completions"
        styles = [preferred_style, fallback_style]

        errors: list[str] = []
        for style in styles:
            try:
                if style == "responses":
                    content = await self._call_responses_api(client, prompt=prompt)
                else:
                    content = await self._call_chat_completions(client, messages=messages)
                if str(content or "").strip():
                    return content
                errors.append(f"{style}: empty_content")
            except Exception as exc:
                errors.append(f"{style}: {exc}")

        raise RuntimeError(" | ".join(errors) if errors else "unknown_localization_error")

    async def translate_instruction_to_chinese(
        self,
        *,
        user_instruction: str,
        provider: str,
    ) -> str | None:
        text = str(user_instruction or "").strip()
        if not text or _contains_cjk(text) or not _looks_english_text(text) or not self.is_available():
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "Translate the user's map/navigation instruction into concise Simplified Chinese for mainland "
                    "China map providers. Preserve the original intent, place names, and constraints. "
                    "Return JSON only with one key: translated_instruction."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "provider": provider,
                        "instruction": text,
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        prompt = json.dumps(
            {
                "instruction": (
                    "Translate this map/navigation request into concise Simplified Chinese for mainland-China map retrieval. "
                    "Return JSON only with translated_instruction."
                ),
                "provider": provider,
                "user_instruction": text,
            },
            ensure_ascii=True,
        )

        client = _create_async_client(self.settings)
        try:
            content = await self._call_llm_with_fallback(client, messages=messages, prompt=prompt)
            parsed = _parse_json_object_from_text(content)
        except Exception:
            return None
        finally:
            await _close_async_client(client)

        translated = str(parsed.get("translated_instruction") or "").strip()
        if not translated:
            return None
        return translated

    async def translate_result_bundle_to_english(
        self,
        *,
        result_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.is_available():
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "Translate Chinese map retrieval results into natural English. "
                    "Preserve all ids, ordering, numeric values, and non-text fields. "
                    "Only translate text-bearing fields such as name, type, address, instruction, road, and summary. "
                    "Return JSON only with the same top-level structure."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(result_bundle, ensure_ascii=False),
            },
        ]
        prompt = json.dumps(
            {
                "instruction": (
                    "Translate the Chinese text fields in this map result payload into English. "
                    "Keep ids, numeric values, booleans, and structure unchanged. Return JSON only."
                ),
                "payload": result_bundle,
            },
            ensure_ascii=False,
        )

        client = _create_async_client(self.settings)
        try:
            content = await self._call_llm_with_fallback(client, messages=messages, prompt=prompt)
            parsed = _parse_json_object_from_text(content)
        except Exception:
            return None
        finally:
            await _close_async_client(client)

        if not isinstance(parsed, dict):
            return None
        return parsed


async def parse_instruction_with_llm(user_instruction: str) -> dict[str, Any]:
    return await IntentParser().parse(user_instruction)


async def rerank_candidates_with_llm(
    *,
    user_instruction: str,
    parsed_intent: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return await CandidateReranker().rerank(
        user_instruction=user_instruction,
        parsed_intent=parsed_intent,
        candidates=candidates,
    )


async def analyze_image_with_llm(
    image_url: str,
    question: str = "What is in this image?",
    detail: str = "auto",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    return await VisionAnalyzer().analyze(
        image_url=image_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )


def local_image_path_to_data_url(image_path: str) -> str:
    path = Path(str(image_path or "").strip()).expanduser()
    if not str(path):
        raise ValueError("image_path is required")
    if not path.exists():
        raise FileNotFoundError(f"image_path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"image_path is not a file: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


async def analyze_local_image_with_llm(
    image_path: str,
    question: str = "What is in this image?",
    detail: str = "auto",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    data_url = local_image_path_to_data_url(image_path)
    result = await VisionAnalyzer().analyze(
        image_url=data_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )
    result["image_path"] = str(Path(image_path).expanduser())
    result["image_source"] = "local_path"
    return result


async def arbitrate_pov_candidates_with_llm(
    *,
    image_url: str,
    question: str,
    detail: str = "auto",
    system_prompt: str | None = None,
) -> dict[str, Any] | None:
    result = await VisionAnalyzer().analyze(
        image_url=image_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )
    answer = str(result.get("answer", "")).strip()
    if not answer:
        return None
    parsed = _parse_json_object_from_text(answer)
    parsed["_arbiter_provider"] = result.get("provider")
    parsed["_arbiter_model"] = result.get("model")
    parsed["_arbiter_api_style"] = result.get("api_style")
    return parsed


async def arbitrate_local_pov_candidates_with_llm(
    *,
    image_path: str,
    question: str,
    detail: str = "auto",
    system_prompt: str | None = None,
) -> dict[str, Any] | None:
    data_url = local_image_path_to_data_url(image_path)
    parsed = await arbitrate_pov_candidates_with_llm(
        image_url=data_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )
    if parsed is not None:
        parsed["_arbiter_image_path"] = str(Path(image_path).expanduser())
    return parsed


async def translate_instruction_for_cn_map(
    *,
    user_instruction: str,
    provider: str,
) -> str | None:
    return await MapLocalizationService().translate_instruction_to_chinese(
        user_instruction=user_instruction,
        provider=provider,
    )


async def translate_result_bundle_to_english(
    *,
    result_bundle: dict[str, Any],
) -> dict[str, Any] | None:
    return await MapLocalizationService().translate_result_bundle_to_english(result_bundle=result_bundle)


__all__ = [
    "CandidateReranker",
    "IntentParser",
    "LLMSettings",
    "MapLocalizationService",
    "VisionAnalyzer",
    "analyze_image_with_llm",
    "analyze_local_image_with_llm",
    "arbitrate_local_pov_candidates_with_llm",
    "arbitrate_pov_candidates_with_llm",
    "extract_explicit_destination_query",
    "infer_destination_kind",
    "local_image_path_to_data_url",
    "parse_instruction_with_llm",
    "rerank_candidates_with_llm",
    "translate_instruction_for_cn_map",
    "translate_result_bundle_to_english",
]
