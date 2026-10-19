"""Transport responsibilities for language and vision services."""
from __future__ import annotations

from typing import Any
import inspect
import json
import os
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from mcp_server.llm_services.settings import LLMSettings

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


def _usage_cached_tokens(usage: dict[str, Any] | None) -> int | None:
    """Extract prompt-cache hit tokens when the provider reports them."""

    if not isinstance(usage, dict):
        return None
    candidates: list[Any] = [
        usage.get("cached_tokens"),
        usage.get("cache_read_input_tokens"),
        usage.get("cached_input_tokens"),
    ]
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        candidates.append(prompt_details.get("cached_tokens"))
        candidates.append(prompt_details.get("cached_input_tokens"))
    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict):
        candidates.append(input_details.get("cached_tokens"))
        candidates.append(input_details.get("cached_input_tokens"))
    for candidate in candidates:
        try:
            if candidate is None:
                continue
            return int(round(float(candidate)))
        except Exception:
            continue
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
    usage = _response_usage_dict(response)
    return {
        "response_id": getattr(response, "id", None),
        "created": getattr(response, "created", None),
        "finish_reason": finish_reason,
        "usage": usage,
        "cached_tokens": _usage_cached_tokens(usage),
    }


def _responses_metadata(response: Any) -> dict[str, Any]:
    usage = _response_usage_dict(response)
    return {
        "response_id": getattr(response, "id", None),
        "created": getattr(response, "created_at", None) or getattr(response, "created", None),
        "status": getattr(response, "status", None),
        "usage": usage,
        "cached_tokens": _usage_cached_tokens(usage),
    }


def _apply_reasoning_effort(payload: dict[str, Any], settings: LLMSettings) -> dict[str, Any]:
    """Request minimal reasoning when the provider supports it."""
    effort = str(settings.reasoning_effort or "").strip().lower()
    if settings.provider == "ark" and effort in {"minimal", "low", "medium", "high"}:
        payload["reasoning_effort"] = effort
    return payload


def _apply_chat_template_kwargs(payload: dict[str, Any], settings: LLMSettings) -> dict[str, Any]:
    """Attach provider-specific chat-template kwargs when explicitly configured."""
    if _truthy_env("LLM_SINGLE_USER_MESSAGE"):
        payload = _compact_chat_payload_to_single_user(payload)
    if settings.chat_template_kwargs:
        extra_body = dict(payload.get("extra_body") or {})
        extra_body["chat_template_kwargs"] = dict(settings.chat_template_kwargs)
        payload["extra_body"] = extra_body
    return payload


def _compact_chat_payload_to_single_user(payload: dict[str, Any]) -> dict[str, Any]:
    """Collapse a chat-completions payload to one user message.

    Some OpenAI-compatible local VLM servers, including Molmo2 in the current
    deployment, reject system messages and few-shot assistant turns with
    ``Conversation roles must alternate user/assistant/...``.  The UrbanNav
    prompts are still valid as plain text, so this adapter preserves all
    sections while presenting them as a single user turn.
    """

    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) <= 1:
        return payload

    sections: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user") or "user").strip().upper()
        content = _message_content_to_text(message.get("content", ""))
        if not content.strip():
            continue
        if role == "SYSTEM":
            label = "SYSTEM INSTRUCTIONS"
        elif role == "ASSISTANT":
            label = "EXAMPLE ASSISTANT OUTPUT"
        elif role == "USER":
            label = "USER INPUT"
        else:
            label = role or "MESSAGE"
        sections.append(f"[{label}]\n{content.strip()}")

    collapsed = dict(payload)
    collapsed["messages"] = [
        {
            "role": "user",
            "content": "\n\n".join(sections),
        }
    ]
    return collapsed


def _message_content_to_text(content: Any) -> str:
    """Return text content from a chat message payload."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item.get("content"), str):
                    parts.append(str(item["content"]))
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)
    return str(content or "")


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


def create_async_llm_client(settings: LLMSettings) -> Any:
    """Create an OpenAI-compatible async LLM client.

    Args:
        settings: Provider, model, key, base URL, timeout, and retry settings.

    Returns:
        Async client object, or ``None`` when the optional OpenAI SDK is not
        installed.
    """

    return _create_async_client(settings)


async def close_async_llm_client(client: Any) -> None:
    """Close an async LLM client if it exposes a close method.

    Args:
        client: Client returned by ``create_async_llm_client`` or a compatible
            test double.

    Returns:
        ``None``.
    """

    await _close_async_client(client)


def extract_chat_response_text(response: Any) -> str:
    """Extract assistant text from a chat-completions response.

    Args:
        response: Response object returned by an OpenAI-compatible
            ``chat.completions.create`` call.

    Returns:
        Assistant text, or an empty string if the response has no text content.
    """

    return _extract_text_from_chat_response(response)


def apply_llm_reasoning_effort(payload: dict[str, Any], settings: LLMSettings) -> dict[str, Any]:
    """Apply provider-specific reasoning-effort parameters.

    Args:
        payload: Chat-completions request payload.
        settings: LLM settings that define provider and reasoning effort.

    Returns:
        Payload with provider-specific fields added when supported.
    """

    return _apply_reasoning_effort(payload, settings)
