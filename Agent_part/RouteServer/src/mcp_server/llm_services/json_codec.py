"""Json codec responsibilities for language and vision services."""
from __future__ import annotations

from typing import Any
import ast
import json
import re

def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _normalize_json_escapes(text: str) -> str:
    """Normalize a narrow set of invalid LLM JSON escapes.

    LaTeX-oriented models sometimes emit ``\\_`` inside JSON keys or values.
    JSON does not define that escape, while removing the backslash preserves
    the literal underscore intended by the model. Some legacy checkpoints also
    prefix a field key with ``$`` and omit its opening quote; only that narrow
    object-key typo is repaired here.
    """

    normalized = str(text or "").replace(r"\_", "_")
    normalized = re.sub(r'"\$([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'"\1":', normalized)
    return re.sub(r'([,{]\s*)\$([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'\1"\2":', normalized)


def _close_unterminated_json_containers(text: str, *, max_closers: int = 8) -> str | None:
    """Append only missing object/array delimiters to partial JSON.

    The repair is intentionally conservative: input must begin with an object,
    strings must already be terminated, and existing delimiters must be
    correctly nested. Missing content, keys, values, or quotes are never
    synthesized.
    """

    candidate = str(text or "").strip()
    if not candidate.startswith("{"):
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"}:
            if not stack or stack.pop() != char:
                return None
    if in_string or escaped or not stack or len(stack) > max_closers:
        return None
    return candidate + "".join(reversed(stack))


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

    normalized = _normalize_json_escapes(text)
    repaired = _close_unterminated_json_containers(normalized)
    parse_attempts = [
        text,
        _remove_trailing_commas(text),
        normalized,
        _remove_trailing_commas(normalized),
    ]
    if repaired is not None:
        parse_attempts.extend([repaired, _remove_trailing_commas(repaired)])
    seen: set[str] = set()
    for attempt in parse_attempts:
        if attempt in seen:
            continue
        seen.add(attempt)
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    # Last-resort parser for Python-dict-like outputs from LLMs.
    try:
        parsed = ast.literal_eval(normalized)
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


def _extract_json_array_field(text: str, field_name: str) -> list[Any] | None:
    """Extract a complete top-level array field from partial JSON text.

    This is intentionally narrow. It only recovers a syntactically complete
    array value for a named field and does not synthesize missing model output.
    """

    pattern = re.compile(rf'"{re.escape(field_name)}"\s*:\s*(\[.*?\])', re.DOTALL)
    for match in pattern.finditer(str(text or "")):
        raw_array = _remove_trailing_commas(match.group(1))
        try:
            parsed = json.loads(raw_array)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def _json_only_contract(schema: dict[str, Any]) -> str:
    """Build a compact JSON-only instruction with an example schema.

    Args:
        schema: JSON-serializable example object.

    Returns:
        Prompt text that strongly discourages explanations or markdown.
    """

    return (
        "Output requirements: return exactly one minified JSON object. "
        "Do not output markdown, code fences, comments, analysis, natural-language explanations, "
        "or any text before or after the JSON. "
        "Write underscores literally; never escape them as \\_. "
        "Stop immediately after the final closing brace and never repeat tokens or fields. "
        "Use double quotes for all JSON strings. Use null for unknown numeric constraints. "
        "The JSON object must follow this shape: "
        f"{json.dumps(schema, ensure_ascii=True, separators=(',', ':'))}"
    )


JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
