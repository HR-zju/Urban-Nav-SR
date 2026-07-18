"""Small deterministic JSONL IO helpers used by annotation pipelines."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from one JSONL file.

    Args:
        path: Input JSONL path.

    Yields:
        Parsed dictionary rows.

    Raises:
        ValueError: If a non-empty line is invalid JSON or not an object.
    """

    input_path = Path(path).expanduser()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {input_path}:{line_number} is not an object")
            yield value


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomically replace a JSONL file with serialized rows.

    Args:
        path: Output JSONL path.
        rows: JSON-compatible dictionary rows.

    Returns:
        Number of written rows.
    """

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return count


def append_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    """Append and flush one JSON object to an open text handle.

    Args:
        handle: Writable text file handle.
        row: JSON-compatible dictionary.

    Returns:
        ``None``.
    """

    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


__all__ = ["append_jsonl_row", "iter_jsonl", "write_jsonl_atomic"]
