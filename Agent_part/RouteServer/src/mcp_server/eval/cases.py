"""Read/write helpers for ``sidewalk_agent_stage2_case_v1`` JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from mcp_server.agent.schemas import AgentEvalCase


def iter_eval_cases(path: str | Path) -> Iterator[AgentEvalCase]:
    """Yield eval cases from JSONL or a JSON array file.

    输入:
        ``path`` 指向 JSONL，每行一个 final Stage2 ``AgentEvalCase``；也兼容 JSON 数组。

    输出:
        逐条 ``AgentEvalCase``，方便大文件流式评测。
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text[0] == "[":
        for payload in json.loads(text):
            yield AgentEvalCase.from_dict(payload)
        return
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}:{line_no} must be a JSON object")
        yield AgentEvalCase.from_dict(payload)


def load_eval_cases(path: str | Path) -> list[AgentEvalCase]:
    return list(iter_eval_cases(path))


def write_eval_cases(cases: Iterable[AgentEvalCase], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
