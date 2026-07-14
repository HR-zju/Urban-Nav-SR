"""JSONL IO for deterministic frozen episode cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from mcp_server.eval.episode.schema import FrozenEpisodeCase


def load_frozen_episode_cases(path: str | Path) -> list[FrozenEpisodeCase]:
    """Load frozen episode cases from JSONL.

    Args:
        path: Source JSONL path.

    Returns:
        Validated cases.

    Raises:
        ValueError: If a row is not a JSON object.
    """

    cases: list[FrozenEpisodeCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"episode row {line_number} must be an object")
            cases.append(FrozenEpisodeCase.model_validate(payload))
    return cases


def write_frozen_episode_cases(path: str | Path, cases: Iterable[FrozenEpisodeCase]) -> int:
    """Write validated frozen episode cases as JSONL.

    Args:
        path: Destination JSONL path.
        cases: Cases to serialize.

    Returns:
        Number of written rows.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for case in cases:
            validated = case if isinstance(case, FrozenEpisodeCase) else FrozenEpisodeCase.model_validate(case)
            handle.write(json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
