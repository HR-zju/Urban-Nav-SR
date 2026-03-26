"""Intent parsing and prompt-engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import inspect
import json
import os
import re

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


JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")


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


def _create_async_client(settings: LLMSettings) -> Any:
    if AsyncOpenAI is None:
        return None
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_s,
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
            "search_keywords": search_keywords,
            "mode": mode,
            "radius_m": radius,
            "optimize_for": optimize_for,
            "max_candidates": rules.get("default_max_candidates", 5),
            "hard_constraints": rules.get("default_hard_constraints", {}),
        }

    def build_messages(self, user_instruction: str) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                "You are an intent parser for map recommendation and routing. "
                "Return JSON only. The JSON schema must contain exactly these top-level keys: "
                "destination_type, search_keywords, mode, radius_m, optimize_for, "
                "max_candidates, hard_constraints. "
                "Allowed mode values: car, foot, bike. "
                "Allowed optimize_for values: time, distance, balanced. "
                "radius_m and max_candidates must be integers. "
                "search_keywords must be a provider-facing keyword string, preferably pipe-delimited, "
                "for example 'bookstore|cafe' or 'convenience store|tea shop'. "
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
            return self.heuristic_parse(user_instruction)

        client = _create_async_client(self.settings)
        try:
            if self.settings.api_style == "responses":
                content = await self._call_responses_api(client, user_instruction)
            else:
                content = await self._call_chat_completions(client, user_instruction)

            match = JSON_OBJECT_PATTERN.search(content)
            if not match:
                return self.heuristic_parse(user_instruction)

            parsed = json.loads(match.group(0))
            if not isinstance(parsed, dict):
                return self.heuristic_parse(user_instruction)
            return self._normalize_llm_output(user_instruction, parsed)
        except Exception:
            return self.heuristic_parse(user_instruction)
        finally:
            await _close_async_client(client)


class VisionAnalyzer(BaseLLMService):
    """Analyze an image through the configured multimodal LLM endpoint."""

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

        client = _create_async_client(self.settings)
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


async def parse_instruction_with_llm(user_instruction: str) -> dict[str, Any]:
    return await IntentParser().parse(user_instruction)


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


__all__ = [
    "IntentParser",
    "LLMSettings",
    "VisionAnalyzer",
    "analyze_image_with_llm",
    "parse_instruction_with_llm",
]
