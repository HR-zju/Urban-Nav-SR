"""Read/write helpers for ``sidewalk_agent_stage2_case_v1`` JSONL.

正式入口使用 ``iter_stage2_cases`` / ``load_stage2_cases`` /
``write_stage2_cases``。旧的 ``*_eval_cases`` 名称保留为兼容别名。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from mcp_server.agent.schemas import AgentEvalCase, AgentStage2Case


def iter_stage2_cases(path: str | Path) -> Iterator[AgentStage2Case]:
    """Yield Stage2 cases from JSONL or a JSON array file.

    输入:
        ``path`` 指向 JSONL 每行一个 ``AgentStage2Case``；也兼容 JSON 数组。

    输出:
        逐条 ``AgentStage2Case``，方便大文件流式评测和构造 cold-start trace。
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text[0] == "[":
        for payload in json.loads(text):
            yield AgentStage2Case.from_dict(payload)
        return
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}:{line_no} must be a JSON object")
        yield AgentStage2Case.from_dict(payload)


def load_stage2_cases(path: str | Path) -> list[AgentStage2Case]:
    """Load all Stage2 cases into memory."""

    return list(iter_stage2_cases(path))


def write_stage2_cases(cases: Iterable[AgentStage2Case], path: str | Path) -> None:
    """Write Stage2 cases as JSONL.

    输入:
        ``cases`` 是 ``AgentStage2Case`` 序列；``path`` 是目标 JSONL。

    输出:
        无返回值；函数会创建父目录并逐行写入 ``case.to_dict()``。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def iter_eval_cases(path: str | Path) -> Iterator[AgentEvalCase]:
    """Compatibility alias for old eval-case callers."""

    yield from iter_stage2_cases(path)


def load_eval_cases(path: str | Path) -> list[AgentEvalCase]:
    """Compatibility alias for old eval-case callers."""

    return list(load_stage2_cases(path))


def write_eval_cases(cases: Iterable[AgentEvalCase], path: str | Path) -> None:
    """Compatibility alias for old eval-case callers."""

    write_stage2_cases(cases, path)
