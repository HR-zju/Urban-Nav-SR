"""Localization responsibilities for language and vision services."""
from __future__ import annotations

from typing import Any
import json
from mcp_server.llm_services.base import BaseLLMService
from mcp_server.llm_services.transport import _apply_chat_template_kwargs, _close_async_client, _create_async_client, _extract_text_from_chat_response, _extract_text_from_responses_response
from mcp_server.llm_services.intent_rules import _contains_cjk, _looks_english_text
from mcp_server.llm_services.json_codec import _parse_json_object_from_text

class MapLocalizationService(BaseLLMService):
    """Translate provider-facing map instructions and result text."""

    async def _call_chat_completions(
        self,
        client: Any,
        *,
        messages: list[dict[str, Any]],
    ) -> str:
        payload = _apply_chat_template_kwargs(
            {
                "model": self.settings.model,
                "temperature": self.settings.temperature,
                "messages": messages,
            },
            self.settings,
        )
        response = await client.chat.completions.create(**payload)
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
