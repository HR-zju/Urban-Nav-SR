"""Intent parsing and prompt-engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import ast
import base64
import inspect
import json
import mimetypes
import os
import io
import re
import traceback
from time import perf_counter
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

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


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_style: str
    api_key: str
    model: str
    base_url: str
    temperature: float = 0.0
    timeout_s: float = 45.0
    max_retries: int = 5
    allow_api_style_fallback: bool = False
    response_format: bool = True
    reasoning_effort: str | None = None

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
        model = (
            os.getenv("LLM_MODEL", "").strip()
            or os.getenv("VLM_MODEL", "").strip()
            or os.getenv("ARK_MODEL", "").strip()
            or default_model
        )
        base_url = (
            os.getenv("LLM_API_BASE_URL", "").strip()
            or os.getenv("ARK_BASE_URL", "").strip()
            or os.getenv("ARK_API_BASE", "").strip()
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or default_base_url
        )
        default_timeout_s = 45.0 if provider == "ark" else 20.0
        default_max_retries = 5 if provider == "ark" else 2
        reasoning_effort = (
            os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
            or os.getenv("ARK_REASONING_EFFORT", "").strip().lower()
        )
        if not reasoning_effort and provider == "ark":
            reasoning_effort = "minimal"
        if reasoning_effort not in {"minimal", "low", "medium", "high"}:
            reasoning_effort = None
        fallback_env = os.getenv("LLM_API_STYLE_FALLBACK", "").strip().lower()
        if fallback_env:
            allow_api_style_fallback = fallback_env in {"1", "true", "yes"}
        else:
            allow_api_style_fallback = provider != "ark"
        return cls(
            provider=provider,
            api_style=api_style,
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0") or 0),
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", str(default_timeout_s)) or default_timeout_s),
            max_retries=max(0, int(os.getenv("LLM_MAX_RETRIES", str(default_max_retries)) or default_max_retries)),
            allow_api_style_fallback=allow_api_style_fallback,
            response_format=os.getenv("LLM_RESPONSE_FORMAT_JSON", "true").strip().lower() not in {"0", "false", "no"},
            reasoning_effort=reasoning_effort,
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


def _model_dump_if_available(value: Any) -> dict[str, Any] | None:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            payload = dump()
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    if isinstance(value, dict):
        return value
    return None


def _response_usage_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    payload = _model_dump_if_available(usage)
    if isinstance(payload, dict):
        return payload
    return None


def _vision_url_payload_stats(image_urls: list[str]) -> dict[str, Any]:
    """Summarize multimodal request payload size without storing image bytes."""
    total_url_chars = 0
    data_url_count = 0
    approx_image_bytes = 0
    per_image: list[dict[str, Any]] = []
    for index, url in enumerate(image_urls):
        raw = str(url or "")
        total_url_chars += len(raw)
        is_data_url = raw.startswith("data:image/") and "," in raw
        approx_bytes = 0
        if is_data_url:
            data_url_count += 1
            encoded = raw.split(",", 1)[1]
            approx_bytes = int(round(len(encoded) * 0.75))
            approx_image_bytes += approx_bytes
        per_image.append(
            {
                "index": int(index),
                "is_data_url": bool(is_data_url),
                "url_chars": len(raw),
                "approx_image_bytes": approx_bytes,
            }
        )
    return {
        "image_count": len(image_urls),
        "data_url_count": int(data_url_count),
        "image_url_chars": int(total_url_chars),
        "approx_image_bytes": int(approx_image_bytes),
        "images": per_image,
    }


def _chat_response_metadata(response: Any) -> dict[str, Any]:
    choices = getattr(response, "choices", None) or []
    finish_reason = None
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
    return {
        "response_id": getattr(response, "id", None),
        "created": getattr(response, "created", None),
        "finish_reason": finish_reason,
        "usage": _response_usage_dict(response),
    }


def _responses_metadata(response: Any) -> dict[str, Any]:
    return {
        "response_id": getattr(response, "id", None),
        "created": getattr(response, "created_at", None) or getattr(response, "created", None),
        "status": getattr(response, "status", None),
        "usage": _response_usage_dict(response),
    }


def _apply_reasoning_effort(payload: dict[str, Any], settings: LLMSettings) -> dict[str, Any]:
    """Request minimal reasoning when the provider supports it."""
    effort = str(settings.reasoning_effort or "").strip().lower()
    if settings.provider == "ark" and effort in {"minimal", "low", "medium", "high"}:
        payload["reasoning_effort"] = effort
    return payload


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


def _ascii_probe(value: Any) -> dict[str, Any]:
    """Return a compact encoding probe without storing large request text."""
    text = str(value or "")
    first_non_ascii: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if ord(char) > 127:
            first_non_ascii = {
                "index": int(index),
                "char_repr": repr(char),
                "codepoint": ord(char),
                "context": text[max(0, index - 20): index + 20],
            }
            break
    return {
        "chars": len(text),
        "ascii": first_non_ascii is None,
        "first_non_ascii": first_non_ascii,
    }


def _vision_request_debug_summary(
    *,
    api_style: str,
    settings: LLMSettings,
    image_urls: list[str],
    question: str,
    system_prompt: str | None,
    detail: str,
) -> dict[str, Any]:
    """Summarize a vision request for failure debugging without image payloads."""
    payload_stats = _vision_url_payload_stats(image_urls)
    return {
        "api_style": api_style,
        "provider": settings.provider,
        "model": settings.model,
        "base_url_host": settings.base_url.split("//")[-1].split("/")[0],
        "reasoning_effort": settings.reasoning_effort,
        "temperature": settings.temperature,
        "detail": detail,
        "timeout_s": None,
        "image_payload": payload_stats,
        "system_prompt_probe": _ascii_probe(system_prompt or ""),
        "question_probe": _ascii_probe(question),
        "extra_image_count": max(0, len(image_urls) - 1),
    }


def _vision_call_debug_enabled() -> bool:
    return _truthy_env("LLM_DEBUG_TRACEBACK") or _truthy_env("LLM_VISION_DEBUG_TRACEBACK")


def _require_ascii_header_value(name: str, value: str) -> None:
    """Fail fast when a header value cannot be encoded safely for httpx."""
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"{name} must be ASCII-safe for HTTP headers; found non-ASCII characters. "
            f"prefix={repr(str(value)[:12])}"
        ) from exc


def _extract_chat_stream_delta_text(chunk: Any) -> str:
    """Extract text from one chat-completions streaming chunk."""

    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            value = getattr(text, "value", None)
            if isinstance(value, str):
                parts.append(value)
        return "".join(parts)
    return ""


def _chat_stream_finish_reason(chunk: Any) -> str | None:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return None
    finish_reason = getattr(choices[0], "finish_reason", None)
    return str(finish_reason) if finish_reason else None


def _chat_stream_metadata(*, chunks: int, response_id: str | None, created: Any, finish_reason: str | None, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_id": response_id,
        "created": created,
        "finish_reason": finish_reason,
        "chunks": int(chunks),
        "usage": usage,
    }


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


def _create_async_client(settings: LLMSettings) -> Any:
    if AsyncOpenAI is None:
        return None
    _require_ascii_header_value("LLM_API_KEY", settings.api_key)
    _require_ascii_header_value("LLM_API_BASE_URL", settings.base_url)
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
    )


def _create_async_client_with_timeout(settings: LLMSettings, timeout_s: float) -> Any:
    if AsyncOpenAI is None:
        return None
    _require_ascii_header_value("LLM_API_KEY", settings.api_key)
    _require_ascii_header_value("LLM_API_BASE_URL", settings.base_url)
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=timeout_s,
        max_retries=settings.max_retries,
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

    def _api_style_sequence(self) -> list[str]:
        preferred_style = self.settings.api_style
        if not self.settings.allow_api_style_fallback:
            return [preferred_style]
        fallback_style = "responses" if preferred_style == "chat_completions" else "chat_completions"
        ordered: list[str] = []
        for style in (preferred_style, fallback_style):
            if style not in ordered:
                ordered.append(style)
        return ordered


class IntentParser(BaseLLMService):
    """Parse recommendation instructions with an LLM and a heuristic fallback."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        super().__init__(settings=settings)
        intent_model = os.getenv("LLM_INTENT_MODEL", "").strip()
        intent_timeout = os.getenv("LLM_INTENT_TIMEOUT_S", "").strip()
        intent_retries = os.getenv("LLM_INTENT_MAX_RETRIES", "").strip()
        updates: dict[str, Any] = {}
        if intent_model:
            updates["model"] = intent_model
        if intent_timeout:
            updates["timeout_s"] = float(intent_timeout)
        if intent_retries:
            updates["max_retries"] = max(0, int(intent_retries))
        if updates:
            self.settings = replace(self.settings, **updates)

    def _prompt_profile(self) -> str:
        profile = os.getenv("LLM_INTENT_PROFILE", os.getenv("INTENT_PARSE_PROFILE", "compact"))
        profile = str(profile or "compact").strip().lower()
        return profile if profile in {"compact", "full"} else "compact"

    async def _call_llm_with_fallback(self, client: Any, user_instruction: str) -> tuple[str, str]:
        errors: list[str] = []
        for style in self._api_style_sequence():
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

        destination_kind = infer_destination_kind(
            user_instruction=user_instruction,
            destination_type=destination_type,
            search_keywords=search_keywords,
        )

        return {
            "destination_type": destination_type,
            "destination_kind": destination_kind,
            "search_keywords": search_keywords,
            "mode": mode,
            "radius_m": radius,
            "optimize_for": optimize_for,
            "max_candidates": rules.get("default_max_candidates", 5),
            "hard_constraints": rules.get("default_hard_constraints", {}),
            "confidence": _heuristic_parse_confidence(
                user_instruction=user_instruction,
                destination_type=destination_type,
                destination_kind=destination_kind,
            ),
            "named_entity_confidence": _heuristic_named_entity_confidence(
                user_instruction=user_instruction,
                destination_type=destination_type,
                destination_kind=destination_kind,
                search_keywords=search_keywords,
            ),
            "intent_source": "heuristic",
        }

    def build_messages(self, user_instruction: str) -> list[dict[str, str]]:
        if self._prompt_profile() == "compact":
            return [
                {
                    "role": "system",
                    "content": (
                        "Parse a navigation/recommendation request. Return one JSON object only with keys: "
                        "destination_type, destination_kind, search_keywords, mode, radius_m, optimize_for, "
                        "max_candidates, hard_constraints, confidence, named_entity_confidence. "
                        "destination_kind must be one of generic_poi,named_landmark,regional_scenic_area,"
                        "administrative_area,road_or_route_label,specific_poi. mode is car/foot/bike. "
                        "search_keywords must be concise provider-facing POI/place search text, pipe-delimited "
                        "when multiple categories are useful. hard_constraints has max_route_duration_min, "
                        "max_route_distance_m, must_include_keywords, exclude_keywords. JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Request: {user_instruction}",
                },
            ]

        messages = [
            {
                "role": "system",
                "content": (
                "You are an intent parser for map recommendation and routing. "
                "Return JSON only. The JSON schema must contain exactly these top-level keys: "
                "destination_type, destination_kind, search_keywords, mode, radius_m, optimize_for, "
                "max_candidates, hard_constraints, confidence, named_entity_confidence. "
                "Allowed destination_kind values: generic_poi, named_landmark, regional_scenic_area, "
                "administrative_area, road_or_route_label, specific_poi. "
                "Allowed mode values: car, foot, bike. "
                "Allowed optimize_for values: time, distance, balanced. "
                "radius_m and max_candidates must be integers. "
                "confidence and named_entity_confidence must be numbers between 0 and 1. "
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
        if self._prompt_profile() == "compact":
            return "\n".join(
                [
                    self.build_messages(user_instruction)[0]["content"],
                    f"Request: {user_instruction}",
                ]
            )

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
        max_tokens_env = os.getenv("LLM_INTENT_MAX_TOKENS", "").strip()
        if max_tokens_env:
            payload["max_tokens"] = max(32, int(max_tokens_env))
        elif self._prompt_profile() == "compact":
            payload["max_tokens"] = 256
        # ARK chat completions can reject JSON response_format on some endpoints/models.
        # Keep the prompt JSON-only and fall back to regex extraction instead.
        if self.settings.response_format and self.settings.provider != "ark":
            payload["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**payload)
        return _extract_text_from_chat_response(response)

    async def _call_responses_api(self, client: Any, user_instruction: str) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "input": [
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
        }
        max_tokens_env = os.getenv("LLM_INTENT_MAX_TOKENS", "").strip()
        if max_tokens_env:
            payload["max_output_tokens"] = max(32, int(max_tokens_env))
        elif self._prompt_profile() == "compact":
            payload["max_output_tokens"] = 256
        response = await client.responses.create(**payload)
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
        normalized["confidence"] = _clamp_probability(
            normalized.get("confidence"),
            default=_heuristic_parse_confidence(
                user_instruction=user_instruction,
                destination_type=destination_type,
                destination_kind=destination_kind,
            ),
        )
        normalized["named_entity_confidence"] = _clamp_probability(
            normalized.get("named_entity_confidence"),
            default=_heuristic_named_entity_confidence(
                user_instruction=user_instruction,
                destination_type=destination_type,
                destination_kind=destination_kind,
                search_keywords=str(normalized.get("search_keywords") or normalized.get("keywords") or ""),
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


def _split_itinerary_segments(user_instruction: str) -> list[str]:
    text = str(user_instruction or "").strip()
    if not text:
        return []

    # Treat sequence markers as boundaries, but do not split on weak fillers such as "想".
    normalized = re.sub(r"[；;。]", " | ", text)
    normalized = re.sub(r"[，,]\s*(?:再|然后|之后|接着|最后)\s*", " | ", normalized)
    normalized = re.sub(r"(?:^|[，,\s])先\s*", " | ", normalized)
    normalized = re.sub(
        r"\s+(?:and then|then|after that|next|finally)\s+",
        " | ",
        normalized,
        flags=re.IGNORECASE,
    )
    segments = [
        re.sub(r"\s+", " ", segment).strip(" ,|，")
        for segment in normalized.split("|")
    ]
    stopwords = {"想", "我想", "want", "i want", "would like", "想要"}
    cleaned: list[str] = []
    for segment in segments:
        if not segment or segment.lower() in stopwords:
            continue
        if len(segment) <= 1:
            continue
        cleaned.append(segment)
    return cleaned


def _normalize_goal_label(raw_label: Any, fallback: str = "generic_stop") -> str:
    label = re.sub(r"[^0-9A-Za-z_\-\u3400-\u9fff]+", "_", str(raw_label or "").strip().lower())
    label = re.sub(r"_+", "_", label).strip("_")
    return label or fallback


class GoalDecomposer(BaseLLMService):
    """Decompose a multi-activity user instruction into an ordered goal list."""

    def _itinerary_timeout_s(self) -> float:
        raw = os.getenv("LLM_ITINERARY_TIMEOUT_S", "").strip()
        if raw:
            try:
                return max(float(raw), 10.0)
            except Exception:
                pass
        return max(float(self.settings.timeout_s), 45.0)

    def heuristic_decompose(self, user_instruction: str) -> dict[str, Any]:
        parser = IntentParser(settings=self.settings)
        global_intent = parser.heuristic_parse(user_instruction)
        segments = _split_itinerary_segments(user_instruction)
        if not segments:
            segments = [str(user_instruction or "").strip()]

        goals: list[dict[str, Any]] = []
        for index, segment in enumerate(segments[:4]):
            parsed = parser.heuristic_parse(segment)
            goal_type = _normalize_goal_label(parsed.get("destination_type"), fallback="generic_stop")
            goals.append(
                {
                    "goal_index": index,
                    "goal_type": goal_type,
                    "request_text": segment,
                    "destination_type": parsed.get("destination_type", "poi"),
                    "destination_kind": parsed.get("destination_kind", "generic_poi"),
                    "search_keywords": parsed.get("search_keywords", default_search_keywords()),
                    "mode": parsed.get("mode", global_intent.get("mode", "car")),
                    "radius_m": parsed.get("radius_m", global_intent.get("radius_m", 3000)),
                    "optimize_for": parsed.get("optimize_for", global_intent.get("optimize_for", "distance")),
                    "hard_constraints": parsed.get("hard_constraints", default_intent_rules()["default_hard_constraints"]),
                    "rationale": "heuristic fallback only preserves structure; semantic generalization is provided by the LLM path",
                }
            )

        return {
            "goals": goals,
            "global_mode": global_intent.get("mode", "car"),
            "global_radius_m": global_intent.get("radius_m", 3000),
            "is_multi_goal": len(goals) > 1,
            "decomposition_source": "heuristic",
        }

    def build_messages(self, user_instruction: str) -> list[dict[str, str]]:
        examples = [
            {
                "request": "我今天累了，想先休息，吃顿晚餐，再看个电影",
                "output": {
                    "goals": [
                        {
                            "goal_index": 0,
                            "goal_type": "rest_stop",
                            "request_text": "先找一个能坐下休息的地方",
                            "destination_type": "rest_stop",
                            "destination_kind": "generic_poi",
                            "search_keywords": "咖啡馆|茶馆|书店|公园休息区",
                            "mode": "foot",
                            "radius_m": 3000,
                            "optimize_for": "distance",
                            "hard_constraints": {
                                "max_route_duration_min": None,
                                "max_route_distance_m": None,
                                "must_include_keywords": [],
                                "exclude_keywords": [],
                            },
                            "rationale": "The user is tired, so the first stop should be a nearby place where they can sit and rest.",
                        },
                        {
                            "goal_index": 1,
                            "goal_type": "dinner",
                            "request_text": "吃顿晚餐",
                            "destination_type": "dinner_restaurant",
                            "destination_kind": "generic_poi",
                            "search_keywords": "餐厅|晚餐|美食|饭店",
                            "mode": "foot",
                            "radius_m": 3000,
                            "optimize_for": "balanced",
                            "hard_constraints": {
                                "max_route_duration_min": None,
                                "max_route_distance_m": None,
                                "must_include_keywords": [],
                                "exclude_keywords": [],
                            },
                            "rationale": "Dinner should retrieve food-service POIs, not generic rest stops.",
                        },
                        {
                            "goal_index": 2,
                            "goal_type": "movie",
                            "request_text": "看个电影",
                            "destination_type": "cinema",
                            "destination_kind": "generic_poi",
                            "search_keywords": "电影院|影院|影城",
                            "mode": "foot",
                            "radius_m": 5000,
                            "optimize_for": "distance",
                            "hard_constraints": {
                                "max_route_duration_min": None,
                                "max_route_distance_m": None,
                                "must_include_keywords": [],
                                "exclude_keywords": [],
                            },
                            "rationale": "The final activity is watching a movie, so the retrievable supply is cinemas.",
                        },
                    ],
                    "global_mode": "foot",
                    "notes": "ordered_multi_goal",
                },
            },
            {
                "request": "I'm thirsty and want a quick coffee nearby",
                "output": {
                    "goals": [
                        {
                            "goal_index": 0,
                            "goal_type": "quick_drink_or_coffee",
                            "request_text": "find a quick coffee or drink stop nearby",
                            "destination_type": "drink_stop",
                            "destination_kind": "generic_poi",
                            "search_keywords": "coffee|cafe|tea shop|juice bar|convenience store",
                            "mode": "foot",
                            "radius_m": 2000,
                            "optimize_for": "distance",
                            "hard_constraints": {
                                "max_route_duration_min": None,
                                "max_route_distance_m": None,
                                "must_include_keywords": [],
                                "exclude_keywords": [],
                            },
                            "rationale": "The need is thirst plus coffee, so nearby beverage-service POIs are appropriate.",
                        },
                    ],
                    "global_mode": "foot",
                    "notes": "single_goal_need",
                },
            },
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trip-goal decomposer for map recommendation. "
                    "Return JSON only. The top-level JSON keys must be exactly: goals, global_mode, notes. "
                    "goals must be an ordered list. Each goal object must contain exactly: "
                    "goal_index, goal_type, request_text, destination_type, destination_kind, "
                    "search_keywords, mode, radius_m, optimize_for, hard_constraints, rationale. "
                    "Use open-vocabulary reasoning: goal_type and destination_type are semantic labels, not a closed taxonomy. "
                    "search_keywords must be a concise provider-facing query string, preferably pipe-delimited, that a domestic map API can retrieve. "
                    "For implicit human needs, infer the real-world POI/service supply that would satisfy the need, then write searchable keywords. "
                    "Use hard_constraints only for constraints that are explicitly stated by the user. "
                    "Allowed destination_kind values: generic_poi, named_landmark, regional_scenic_area, administrative_area, road_or_route_label, specific_poi. "
                    "Allowed mode values: car, foot, bike. Allowed optimize_for values: time, distance, balanced. "
                    "If the request is single-goal, still return a one-item goals list. "
                    "Do not solve routing. Do not return destinations. Only decompose the user's instruction into ordered retrievable sub-goals."
                ),
            }
        ]
        for example in examples:
            messages.append({"role": "user", "content": f"request: {example['request']}"})
            messages.append({"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=True)})
        messages.append({"role": "user", "content": f"request: {user_instruction}"})
        return messages

    def build_prompt(self, user_instruction: str) -> str:
        sections = [self.build_messages(user_instruction)[0]["content"], "", "Examples:"]
        for message in self.build_messages(user_instruction)[1:-1]:
            if message["role"] == "user":
                sections.append(f"Request: {message['content'].replace('request: ', '')}")
            else:
                sections.append(f"JSON: {message['content']}")
        sections.extend(["", "Now decompose this request and return JSON only.", f"Request: {user_instruction}"])
        return "\n".join(sections)

    async def _call_chat_completions(self, client: Any, user_instruction: str) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": self.build_messages(user_instruction),
        }
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
                    "content": [{"type": "input_text", "text": self.build_prompt(user_instruction)}],
                }
            ],
        )
        return _extract_text_from_responses_response(response)

    async def _call_llm_with_fallback(self, client: Any, user_instruction: str) -> tuple[str, str]:
        errors: list[str] = []
        for style in self._api_style_sequence():
            try:
                content = (
                    await self._call_responses_api(client, user_instruction)
                    if style == "responses"
                    else await self._call_chat_completions(client, user_instruction)
                )
                if str(content or "").strip():
                    return content, style
                errors.append(f"{style}: empty_content")
            except Exception as exc:
                errors.append(f"{style}: {exc}")
        raise RuntimeError(" | ".join(errors) if errors else "unknown_llm_call_error")

    def _normalize_output(self, user_instruction: str, parsed: dict[str, Any]) -> dict[str, Any]:
        heuristic = self.heuristic_decompose(user_instruction)
        parser = IntentParser(settings=self.settings)
        raw_goals = parsed.get("goals")
        if not isinstance(raw_goals, list) or not raw_goals:
            return heuristic

        goals: list[dict[str, Any]] = []
        for index, raw_goal in enumerate(raw_goals[:4]):
            if not isinstance(raw_goal, dict):
                continue
            request_text = str(raw_goal.get("request_text") or "").strip()
            if not request_text:
                continue
            normalized_intent = parser._normalize_llm_output(  # type: ignore[attr-defined]
                request_text,
                raw_goal,
            )
            raw_goal_type = str(raw_goal.get("goal_type") or raw_goal.get("destination_type") or "").strip()
            goal_type = _normalize_goal_label(
                raw_goal_type,
                fallback=_normalize_goal_label(normalized_intent.get("destination_type"), fallback="generic_stop"),
            )
            goals.append(
                {
                    "goal_index": index,
                    "goal_type": goal_type,
                    "request_text": request_text,
                    "destination_type": normalized_intent.get("destination_type", "poi"),
                    "destination_kind": normalized_intent.get("destination_kind", "generic_poi"),
                    "search_keywords": normalized_intent.get("search_keywords", default_search_keywords()),
                    "mode": normalized_intent.get("mode", heuristic.get("global_mode", "car")),
                    "radius_m": normalized_intent.get("radius_m", heuristic.get("global_radius_m", 3000)),
                    "optimize_for": normalized_intent.get("optimize_for", "distance"),
                    "hard_constraints": normalized_intent.get(
                        "hard_constraints",
                        default_intent_rules()["default_hard_constraints"],
                    ),
                    "rationale": str(raw_goal.get("rationale") or "").strip(),
                }
            )

        if not goals:
            return heuristic

        global_mode = str(parsed.get("global_mode") or heuristic.get("global_mode") or "car").strip().lower()
        if global_mode not in {"car", "foot", "bike"}:
            global_mode = heuristic.get("global_mode", "car")

        radii = [
            int(goal.get("radius_m", heuristic.get("global_radius_m", 3000)) or 3000)
            for goal in goals
            if isinstance(goal.get("radius_m", 3000), (int, float))
        ]

        return {
            "goals": goals,
            "global_mode": global_mode,
            "global_radius_m": max(radii or [int(heuristic.get("global_radius_m", 3000) or 3000)]),
            "is_multi_goal": len(goals) > 1,
            "decomposition_source": "llm",
            "notes": str(parsed.get("notes") or "").strip(),
        }

    async def parse(self, user_instruction: str) -> dict[str, Any]:
        if not self.is_available():
            parsed = self.heuristic_decompose(user_instruction)
            parsed["decomposition_source"] = "heuristic_no_llm"
            return parsed

        client = _create_async_client_with_timeout(self.settings, self._itinerary_timeout_s())
        try:
            try:
                content, api_style_used = await self._call_llm_with_fallback(client, user_instruction)
            except Exception as exc:
                parsed = self.heuristic_decompose(user_instruction)
                parsed["decomposition_source"] = "heuristic_llm_api_error"
                parsed["llm_error"] = str(exc)
                return parsed

            try:
                parsed = _parse_json_object_from_text(content)
            except Exception as exc:
                fallback = self.heuristic_decompose(user_instruction)
                fallback["decomposition_source"] = "heuristic_llm_parse_error"
                fallback["llm_error"] = str(exc)
                return fallback

            normalized = self._normalize_output(user_instruction, parsed)
            normalized["llm_api_style_used"] = api_style_used
            return normalized
        except Exception:
            parsed = self.heuristic_decompose(user_instruction)
            parsed["decomposition_source"] = "heuristic_llm_error"
            return parsed
        finally:
            await _close_async_client(client)


class ItineraryCandidateJudge(BaseLLMService):
    """Select the best concrete POI for one itinerary goal from Top-K map candidates."""

    def _itinerary_timeout_s(self) -> float:
        raw = os.getenv("LLM_ITINERARY_TIMEOUT_S", "").strip()
        if raw:
            try:
                return max(float(raw), 10.0)
            except Exception:
                pass
        return max(float(self.settings.timeout_s), 45.0)

    def _simplify_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        tags = candidate.get("tags") or {}
        route_summary = candidate.get("route_summary") or {}
        return {
            "id": str(candidate.get("id") or ""),
            "name": str(candidate.get("name") or ""),
            "type": str(tags.get("type") or ""),
            "address": str(tags.get("address") or ""),
            "candidate_kind": str(candidate.get("candidate_kind") or ""),
            "is_concrete_destination": bool(candidate.get("is_concrete_destination", False)),
            "location": str(candidate.get("location") or ""),
            "score": candidate.get("score"),
            "distance_m": route_summary.get("distance_m"),
            "duration_min": route_summary.get("duration_min"),
        }

    def build_messages(
        self,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        payload = {
            "goal": {
                "goal_index": goal.get("goal_index"),
                "goal_type": goal.get("goal_type"),
                "request_text": goal.get("request_text"),
                "destination_type": goal.get("destination_type"),
                "destination_kind": goal.get("destination_kind"),
                "search_keywords": goal.get("search_keywords"),
                "rationale": goal.get("rationale"),
            },
            "previous_stops": [
                {
                    "name": str(item.get("name") or ""),
                    "location": str(item.get("location") or ""),
                    "goal_type": str(item.get("goal_type") or ""),
                }
                for item in previous_stops
            ],
            "candidates": [self._simplify_candidate(item) for item in candidates],
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are an itinerary candidate judge for map navigation. "
                    "Return JSON only with exactly: selected_candidate_id, confidence, reason, reject_all. "
                    "Use open-vocabulary reasoning over the goal, its rationale/search query, and each candidate's name/type/address. "
                    "Do not rely on a closed taxonomy. Select a candidate only if it plausibly satisfies the goal as a real-world service/place. "
                    "Reject candidates that appear closed, unavailable, irrelevant to the activity, or duplicate a previous stop unless the goal explicitly asks to stay there. "
                    "If no candidate satisfies the goal, set selected_candidate_id=null and reject_all=true. "
                    "Prefer concrete POIs over vague administrative areas. Consider route distance and duration only after semantic suitability."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    def build_prompt(
        self,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]],
    ) -> str:
        messages = self.build_messages(goal=goal, candidates=candidates, previous_stops=previous_stops)
        return f"{messages[0]['content']}\n\nInput JSON:\n{messages[1]['content']}"

    async def _call_chat_completions(
        self,
        client: Any,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]],
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": self.build_messages(goal=goal, candidates=candidates, previous_stops=previous_stops),
        }
        if self.settings.response_format and self.settings.provider != "ark":
            payload["response_format"] = {"type": "json_object"}
        response = await client.chat.completions.create(**payload)
        return _extract_text_from_chat_response(response)

    async def _call_responses_api(
        self,
        client: Any,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]],
    ) -> str:
        response = await client.responses.create(
            model=self.settings.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self.build_prompt(
                                goal=goal,
                                candidates=candidates,
                                previous_stops=previous_stops,
                            ),
                        }
                    ],
                }
            ],
        )
        return _extract_text_from_responses_response(response)

    def _normalize_decision(self, parsed: dict[str, Any], candidate_ids: set[str]) -> dict[str, Any]:
        selected = parsed.get("selected_candidate_id")
        if selected is not None:
            selected = str(selected).strip()
        if selected and selected not in candidate_ids:
            selected = None

        confidence = parsed.get("confidence", 0.0)
        try:
            confidence_value = max(0.0, min(float(confidence), 1.0))
        except Exception:
            confidence_value = 0.0

        reject_all = bool(parsed.get("reject_all", False)) or selected is None
        return {
            "selected_candidate_id": selected,
            "confidence": confidence_value,
            "reason": str(parsed.get("reason") or "").strip(),
            "reject_all": reject_all,
            "judge_source": "llm",
        }

    async def select_candidate(
        self,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        previous_stops = previous_stops or []
        candidate_ids = {str(item.get("id") or "").strip() for item in candidates if str(item.get("id") or "").strip()}
        if not candidates or not candidate_ids:
            return {
                "selected_candidate_id": None,
                "confidence": 0.0,
                "reason": "No candidates were provided.",
                "reject_all": True,
                "judge_source": "no_candidates",
            }
        if not self.is_available():
            return {
                "selected_candidate_id": None,
                "confidence": 0.0,
                "reason": "LLM candidate judge is not configured.",
                "reject_all": True,
                "judge_source": "unavailable",
            }

        client = _create_async_client_with_timeout(self.settings, self._itinerary_timeout_s())
        try:
            try:
                if self.settings.api_style == "responses":
                    content = await self._call_responses_api(
                        client,
                        goal=goal,
                        candidates=candidates,
                        previous_stops=previous_stops,
                    )
                    api_style_used = "responses"
                else:
                    content = await self._call_chat_completions(
                        client,
                        goal=goal,
                        candidates=candidates,
                        previous_stops=previous_stops,
                    )
                    api_style_used = "chat_completions"
            except Exception as exc:
                return {
                    "selected_candidate_id": None,
                    "confidence": 0.0,
                    "reason": f"LLM candidate judge failed: {exc}",
                    "reject_all": True,
                    "judge_source": "llm_api_error",
                }

            try:
                parsed = _parse_json_object_from_text(content)
            except Exception as exc:
                return {
                    "selected_candidate_id": None,
                    "confidence": 0.0,
                    "reason": f"LLM candidate judge returned non-JSON output: {exc}",
                    "reject_all": True,
                    "judge_source": "llm_parse_error",
                    "raw_text": content,
                }

            decision = self._normalize_decision(parsed, candidate_ids)
            decision["llm_api_style_used"] = api_style_used
            decision["raw"] = parsed
            return decision
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
        errors: list[str] = []
        for style in self._api_style_sequence():
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
        return max(90.0, self.settings.timeout_s)

    def _stream_timing_enabled(self) -> bool:
        return _truthy_env("LLM_VISION_STREAM_TIMING") or _truthy_env("LLM_STREAM_TIMING")

    def build_messages(
        self,
        image_url: str,
        question: str,
        detail: str = "auto",
        system_prompt: str | None = None,
        extra_image_urls: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        prompt = system_prompt or (
            "You are a visual assistant. Answer the user's question about the image directly and concisely. "
            "If the image is unclear or inaccessible, say so plainly."
        )
        image_urls = [str(image_url).strip()]
        for extra in list(extra_image_urls or []):
            value = str(extra or "").strip()
            if value:
                image_urls.append(value)
        content: list[dict[str, Any]] = []
        for url in image_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": detail,
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": question,
            }
        )
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": content,
            },
        ]

    async def _stream_chat_completions(self, client: Any, payload: dict[str, Any]) -> dict[str, Any]:
        """Run chat-completions streaming and expose first-token timing.

        Without provider-side Server-Timing headers, the best client-side split is:

            time_to_first_token ~= upload + provider queue + image encode + prefill
            stream_decode_ms    ~= output decoding + streamed network return
        """

        started_at = perf_counter()
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        if _truthy_env("LLM_STREAM_INCLUDE_USAGE", default=True):
            stream_payload["stream_options"] = {"include_usage": True}

        try:
            stream = await client.chat.completions.create(**stream_payload)
        except Exception:
            if "stream_options" not in stream_payload:
                raise
            # Some OpenAI-compatible providers support streaming but reject
            # stream_options. Keep first-token timing and lose usage details.
            stream_payload.pop("stream_options", None)
            stream = await client.chat.completions.create(**stream_payload)
        chunks = 0
        content_chunks = 0
        first_chunk_ms: int | None = None
        first_content_ms: int | None = None
        last_content_ms: int | None = None
        response_id: str | None = None
        created: Any = None
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        parts: list[str] = []

        async for chunk in stream:
            now_ms = int(round((perf_counter() - started_at) * 1000.0))
            chunks += 1
            if first_chunk_ms is None:
                first_chunk_ms = now_ms
            response_id = response_id or getattr(chunk, "id", None)
            created = created or getattr(chunk, "created", None)
            finish_reason = _chat_stream_finish_reason(chunk) or finish_reason
            chunk_usage = _response_usage_dict(chunk)
            if chunk_usage:
                usage = chunk_usage
            text = _extract_chat_stream_delta_text(chunk)
            if text:
                parts.append(text)
                content_chunks += 1
                if first_content_ms is None:
                    first_content_ms = now_ms
                last_content_ms = now_ms

        roundtrip_ms = int(round((perf_counter() - started_at) * 1000.0))
        if first_content_ms is None:
            first_content_ms = first_chunk_ms
        stream_decode_ms = max(0, roundtrip_ms - int(first_content_ms)) if first_content_ms is not None else None
        return {
            "answer": "".join(parts),
            "roundtrip_ms": roundtrip_ms,
            "stream_timing": {
                "enabled": True,
                "time_to_first_chunk_ms": first_chunk_ms,
                "time_to_first_token_ms": first_content_ms,
                "stream_decode_ms": stream_decode_ms,
                "last_content_chunk_ms": last_content_ms,
                "chunks": chunks,
                "content_chunks": content_chunks,
            },
            "response": _chat_stream_metadata(
                chunks=chunks,
                response_id=response_id,
                created=created,
                finish_reason=finish_reason,
                usage=usage,
            ),
        }

    async def _call_chat_completions(
        self,
        client: Any,
        image_url: str,
        question: str,
        detail: str,
        system_prompt: str | None,
        extra_image_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        messages = self.build_messages(
            image_url,
            question,
            detail,
            system_prompt,
            extra_image_urls=extra_image_urls,
        )
        image_urls = [str(image_url).strip(), *[str(item).strip() for item in list(extra_image_urls or []) if str(item or "").strip()]]
        started_at = perf_counter()
        payload = _apply_reasoning_effort(
            {
                "model": self.settings.model,
                "temperature": self.settings.temperature,
                "messages": messages,
            },
            self.settings,
        )
        try:
            if self._stream_timing_enabled():
                streamed = await self._stream_chat_completions(client, payload)
                roundtrip_ms = int(streamed.get("roundtrip_ms") or 0)
                answer = str(streamed.get("answer") or "")
                response_metadata = streamed.get("response") if isinstance(streamed.get("response"), dict) else {}
                stream_timing = streamed.get("stream_timing") if isinstance(streamed.get("stream_timing"), dict) else {}
            else:
                response = await client.chat.completions.create(**payload)
                roundtrip_ms = int(round((perf_counter() - started_at) * 1000.0))
                answer = _extract_text_from_chat_response(response)
                response_metadata = _chat_response_metadata(response)
                stream_timing = {"enabled": False}
        except Exception as exc:
            if _vision_call_debug_enabled():
                debug_summary = _vision_request_debug_summary(
                    api_style="chat_completions",
                    settings=self.settings,
                    image_urls=image_urls,
                    question=question,
                    system_prompt=system_prompt,
                    detail=detail,
                )
                debug_summary["timeout_s"] = self._vision_timeout_s()
                setattr(exc, "vision_debug", debug_summary)
                setattr(exc, "traceback", traceback.format_exc())
            raise
        request_json = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return {
            "answer": answer,
            "telemetry": {
                "api_style": "chat_completions",
                "provider": self.settings.provider,
                "model": self.settings.model,
                "roundtrip_ms": roundtrip_ms,
                "request_text_chars": len(str(system_prompt or "")) + len(str(question or "")),
                "request_payload_chars": len(request_json),
                "response_text_chars": len(answer),
                "temperature": self.settings.temperature,
                "reasoning_effort": self.settings.reasoning_effort,
                "timeout_s": self._vision_timeout_s(),
                "max_retries": self.settings.max_retries,
                "image_payload": _vision_url_payload_stats(image_urls),
                "response": response_metadata,
                "stream_timing": stream_timing,
                "server_timing_available": False,
            },
        }

    async def _call_responses_api(
        self,
        client: Any,
        image_url: str,
        question: str,
        detail: str,
        system_prompt: str | None,
        extra_image_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        prompt = system_prompt or (
            "You are a visual assistant. Answer the user's question about the image directly and concisely."
        )
        images = [str(image_url).strip()]
        for extra in list(extra_image_urls or []):
            value = str(extra or "").strip()
            if value:
                images.append(value)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for url in images:
            content.append({"type": "input_image", "image_url": url, "detail": detail})
        content.append({"type": "input_text", "text": question})
        request_input = [
            {
                "role": "user",
                "content": content,
            }
        ]
        started_at = perf_counter()
        payload = _apply_reasoning_effort(
            {
                "model": self.settings.model,
                "input": request_input,
            },
            self.settings,
        )
        try:
            response = await client.responses.create(**payload)
            roundtrip_ms = int(round((perf_counter() - started_at) * 1000.0))
            answer = _extract_text_from_responses_response(response)
        except Exception as exc:
            if _vision_call_debug_enabled():
                debug_summary = _vision_request_debug_summary(
                    api_style="responses",
                    settings=self.settings,
                    image_urls=images,
                    question=question,
                    system_prompt=prompt,
                    detail=detail,
                )
                debug_summary["timeout_s"] = self._vision_timeout_s()
                setattr(exc, "vision_debug", debug_summary)
                setattr(exc, "traceback", traceback.format_exc())
            raise
        request_json = json.dumps(request_input, ensure_ascii=False, separators=(",", ":"))
        return {
            "answer": answer,
            "telemetry": {
                "api_style": "responses",
                "provider": self.settings.provider,
                "model": self.settings.model,
                "roundtrip_ms": roundtrip_ms,
                "request_text_chars": len(str(prompt or "")) + len(str(question or "")),
                "request_payload_chars": len(request_json),
                "response_text_chars": len(answer),
                "temperature": self.settings.temperature,
                "reasoning_effort": self.settings.reasoning_effort,
                "timeout_s": self._vision_timeout_s(),
                "max_retries": self.settings.max_retries,
                "image_payload": _vision_url_payload_stats(images),
                "response": _responses_metadata(response),
                "server_timing_available": False,
            },
        }

    async def analyze(
        self,
        image_url: str,
        question: str = "What is in this image?",
        detail: str = "auto",
        system_prompt: str | None = None,
        extra_image_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        if not image_url.strip():
            raise ValueError("image_url is required")
        if detail not in {"auto", "low", "high"}:
            detail = "auto"
        resolved_extra_image_urls = [
            str(item).strip() for item in list(extra_image_urls or []) if str(item or "").strip()
        ]

        if not self.is_available():
            raise RuntimeError("LLM image analysis is not configured. Set ARK_API_KEY or LLM_API_KEY.")

        client = _create_async_client_with_timeout(self.settings, self._vision_timeout_s())
        try:
            try:
                if self.settings.api_style == "responses":
                    call = await self._call_responses_api(
                        client,
                        image_url,
                        question,
                        detail,
                        system_prompt,
                        extra_image_urls=resolved_extra_image_urls,
                    )
                else:
                    call = await self._call_chat_completions(
                        client,
                        image_url,
                        question,
                        detail,
                        system_prompt,
                        extra_image_urls=resolved_extra_image_urls,
                    )
            except Exception as exc:
                if _vision_call_debug_enabled() and not isinstance(getattr(exc, "vision_debug", None), dict):
                    image_urls = [str(image_url).strip(), *resolved_extra_image_urls]
                    debug_summary = _vision_request_debug_summary(
                        api_style=self.settings.api_style,
                        settings=self.settings,
                        image_urls=image_urls,
                        question=question,
                        system_prompt=system_prompt,
                        detail=detail,
                    )
                    debug_summary["timeout_s"] = self._vision_timeout_s()
                    setattr(exc, "vision_debug", debug_summary)
                    setattr(exc, "traceback", traceback.format_exc())
                raise
            answer = str(call.get("answer") or "")

            return {
                "provider": self.settings.provider,
                "api_style": self.settings.api_style,
                "model": self.settings.model,
                "image_url": image_url,
                "image_urls": [str(image_url).strip(), *resolved_extra_image_urls],
                "question": question,
                "detail": detail,
                "answer": answer.strip(),
                "llm_telemetry": call.get("telemetry") if isinstance(call, dict) else None,
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
        errors: list[str] = []
        for style in self._api_style_sequence():
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
    raw_bytes = path.read_bytes()

    max_side_raw = os.getenv("LLM_VISION_IMAGE_MAX_SIDE", "").strip()
    max_bytes_raw = os.getenv("LLM_VISION_IMAGE_MAX_BYTES", "").strip()
    jpeg_quality_raw = os.getenv("LLM_VISION_IMAGE_JPEG_QUALITY", "").strip()
    try:
        max_side = max(512, int(max_side_raw)) if max_side_raw else 1600
    except ValueError:
        max_side = 1600
    try:
        max_bytes = max(256_000, int(max_bytes_raw)) if max_bytes_raw else 2_500_000
    except ValueError:
        max_bytes = 2_500_000
    try:
        jpeg_quality = min(max(int(jpeg_quality_raw), 40), 95) if jpeg_quality_raw else 85
    except ValueError:
        jpeg_quality = 85

    should_optimize = len(raw_bytes) > max_bytes
    optimized_bytes = raw_bytes
    optimized_mime = mime_type

    if Image is not None:
        try:
            image = Image.open(path)
            if ImageOps is not None:
                image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            width, height = image.size
            longest = max(width, height, 1)
            resize_scale = min(1.0, float(max_side) / float(longest))
            if resize_scale < 0.999:
                image = image.resize(
                    (
                        max(1, int(round(width * resize_scale))),
                        max(1, int(round(height * resize_scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
                should_optimize = True
            if should_optimize:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
                optimized_bytes = buffer.getvalue()
                optimized_mime = "image/jpeg"
        except Exception:
            optimized_bytes = raw_bytes
            optimized_mime = mime_type

    encoded = base64.b64encode(optimized_bytes).decode("ascii")
    return f"data:{optimized_mime};base64,{encoded}"


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
    extra_image_urls: list[str] | None = None,
) -> dict[str, Any] | None:
    result = await VisionAnalyzer().analyze(
        image_url=image_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
        extra_image_urls=extra_image_urls,
    )
    answer = str(result.get("answer", "")).strip()
    if not answer:
        return None
    parsed = _parse_json_object_from_text(answer)
    parsed["_arbiter_provider"] = result.get("provider")
    parsed["_arbiter_model"] = result.get("model")
    parsed["_arbiter_api_style"] = result.get("api_style")
    if result.get("llm_telemetry") is not None:
        parsed["_arbiter_llm_telemetry"] = result.get("llm_telemetry")
    return parsed


async def arbitrate_local_pov_candidates_with_llm(
    *,
    image_path: str,
    question: str,
    detail: str = "auto",
    system_prompt: str | None = None,
    extra_image_urls: list[str] | None = None,
) -> dict[str, Any] | None:
    data_url = local_image_path_to_data_url(image_path)
    parsed = await arbitrate_pov_candidates_with_llm(
        image_url=data_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
        extra_image_urls=extra_image_urls,
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
    "GoalDecomposer",
    "IntentParser",
    "ItineraryCandidateJudge",
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
