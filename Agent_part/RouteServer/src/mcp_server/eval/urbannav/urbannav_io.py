"""JSON/JSONL IO helpers for UrbanNav benchmark cases and predictions.

This module intentionally only performs serialization and schema normalization.
It does not call map providers, LLMs, or replay caches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavPrediction


def iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from JSONL, JSON array, or wrapped JSON files.

    Args:
        path: Input path. Supported roots are JSONL rows, JSON arrays, a single
            JSON object, or a wrapper object containing one of ``data``,
            ``cases``, ``records``, ``predictions``, or ``results``.

    Returns:
        Iterator of raw JSON dictionaries.

    Raises:
        ValueError: If a JSONL row is not an object or the JSON root is not a
            supported shape.
        json.JSONDecodeError: If the file is not valid JSON/JSONL.
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return
    if source.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_no} must be a JSON object")
            yield payload
        return

    payload = json.loads(text)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("data", "cases", "records", "predictions", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload
        return
    raise ValueError(f"Unsupported JSON root in {source}: {type(payload).__name__}")


def load_urbannav_cases(path: str | Path) -> list[UrbanNavCase]:
    """Load all UrbanNav cases into memory.

    Args:
        path: JSON/JSONL file containing frozen GT cases.

    Returns:
        List of normalized ``UrbanNavCase`` objects.
    """

    return [UrbanNavCase.from_dict(record) for record in iter_json_records(path)]


def iter_urbannav_cases(path: str | Path) -> Iterator[UrbanNavCase]:
    """Stream UrbanNav cases from disk.

    Args:
        path: JSON/JSONL file containing frozen GT cases.

    Returns:
        Iterator of normalized ``UrbanNavCase`` objects.
    """

    for record in iter_json_records(path):
        yield UrbanNavCase.from_dict(record)


def write_urbannav_cases(cases: Iterable[UrbanNavCase], path: str | Path) -> None:
    """Write UrbanNav cases as JSONL.

    Args:
        cases: Iterable of normalized cases.
        path: Output JSONL path. Parent directories are created if needed.

    Returns:
        None.
    """

    _write_jsonl((case.to_dict() for case in cases), path)


def load_urbannav_predictions(path: str | Path) -> list[UrbanNavPrediction]:
    """Load all UrbanNav predictions into memory.

    Args:
        path: JSON/JSONL file containing saved predictions.

    Returns:
        List of normalized ``UrbanNavPrediction`` objects.
    """

    return [UrbanNavPrediction.from_dict(record) for record in iter_json_records(path)]


def iter_urbannav_predictions(path: str | Path) -> Iterator[UrbanNavPrediction]:
    """Stream UrbanNav predictions from disk.

    Args:
        path: JSON/JSONL file containing saved predictions.

    Returns:
        Iterator of normalized ``UrbanNavPrediction`` objects.
    """

    for record in iter_json_records(path):
        yield UrbanNavPrediction.from_dict(record)


def write_urbannav_predictions(predictions: Iterable[UrbanNavPrediction], path: str | Path) -> None:
    """Write UrbanNav predictions as JSONL.

    Args:
        predictions: Iterable of normalized predictions.
        path: Output JSONL path. Parent directories are created if needed.

    Returns:
        None.
    """

    _write_jsonl((prediction.to_dict() for prediction in predictions), path)


def _write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write raw dictionaries to JSONL.

    Args:
        records: Iterable of JSON-serializable dictionaries.
        path: Output JSONL path.

    Returns:
        None.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
