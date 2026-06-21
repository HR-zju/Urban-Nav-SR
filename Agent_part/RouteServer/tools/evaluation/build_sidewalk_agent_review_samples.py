#!/usr/bin/env python3
"""Build small SideWalk Agentic Stage2 review traces from eval cases.

输入:
    ``--cases`` 指向 ``sidewalk_agent_stage2_case_v1`` JSONL。

输出:
    ``--traces-out``: ``sidewalk_agent_trace_v2`` JSONL。
    ``--ms-swift-out``: ms-swift agent-template JSONL，使用 tool_call/tool_response role。
    ``--metrics-out``: 汇总 tool/readout/final CoT 指标，便于人工 review 前快速筛查。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.eval.cases import load_eval_cases
from mcp_server.eval.exporters import write_ms_swift_jsonl, write_traces_jsonl
from mcp_server.eval.runners import RouteServerTeacherRunner, run_cases


def _metric_average(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 6)


async def _run(args: argparse.Namespace) -> None:
    cases = load_eval_cases(args.cases)
    runner = RouteServerTeacherRunner(
        use_llm=False,
        execute_tools=False,
        enable_visual_prompt=False,
    )
    traces = await run_cases(runner, cases)
    cases_by_id = {case.case_id: case for case in cases}
    write_traces_jsonl(traces, args.traces_out)
    write_ms_swift_jsonl(traces, args.ms_swift_out, cases_by_id=cases_by_id)

    metric_names = sorted({key for trace in traces for key in trace.metrics.keys()})
    summary = {
        "case_count": len(cases),
        "trace_count": len(traces),
        "runner": "routeserver_teacher_offline",
        "metrics": {
            name: _metric_average([trace.metrics.get(name) for trace in traces])
            for name in metric_names
        },
        "cases": [
            {
                "case_id": trace.case_id,
                "metrics": trace.metrics,
                "selected_tool_call": (
                    trace.metadata.get("teacher_internal", {}).get("selected_student_tool_call")
                    if isinstance(trace.metadata, dict)
                    else None
                ),
            }
            for trace in traces
        ],
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=str(ROOT / "examples" / "agent_eval" / "sidewalk_agent_eval_cases_review.jsonl"),
    )
    parser.add_argument(
        "--traces-out",
        default=str(ROOT / "examples" / "agent_eval" / "sidewalk_agent_teacher_traces_review.jsonl"),
    )
    parser.add_argument(
        "--ms-swift-out",
        default=str(ROOT / "examples" / "agent_eval" / "sidewalk_agent_ms_swift_review.jsonl"),
    )
    parser.add_argument(
        "--metrics-out",
        default=str(ROOT / "examples" / "agent_eval" / "sidewalk_agent_metrics_review.json"),
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    main()
