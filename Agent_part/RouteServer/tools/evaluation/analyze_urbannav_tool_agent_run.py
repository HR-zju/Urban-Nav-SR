#!/usr/bin/env python3
"""Summarize and review UrbanNav tool-agent prediction traces.

This script is intentionally post-hoc: it reads saved UrbanNavPrediction JSONL
and, when available, evaluator rows.  It does not call an LLM, map provider, or
repair predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.agent.schemas import ToolCall  # noqa: E402
from mcp_server.agent.tool_parser import parse_agent_action  # noqa: E402
from mcp_server.agent.tool_schema import normalize_map_tool_arguments, validate_tool_call  # noqa: E402
from mcp_server.eval.urbannav.urbannav_io import load_urbannav_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="UrbanNavPrediction JSONL.")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional task_gt.jsonl for category metadata.")
    parser.add_argument("--eval-rows", type=Path, default=None, help="Optional evaluator eval_rows.jsonl.")
    parser.add_argument("--output-json", required=True, type=Path, help="Tool-use summary JSON.")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional markdown summary.")
    parser.add_argument("--review-html", type=Path, default=None, help="Optional trace review HTML.")
    parser.add_argument("--max-review-cases", type=int, default=160, help="Maximum cases rendered in HTML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = _load_cases(args.dataset)
    eval_rows = _load_eval_rows(args.eval_rows)
    analyses = [
        _analyze_prediction(payload=payload, case=cases.get(str(payload.get("case_id") or payload.get("id") or "")), eval_row=eval_rows.get(str(payload.get("case_id") or payload.get("id") or "")))
        for payload in _iter_jsonl(args.predictions)
    ]
    summary = _summarize(analyses)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown(summary), encoding="utf-8")
    if args.review_html:
        args.review_html.parent.mkdir(parents=True, exist_ok=True)
        args.review_html.write_text(_html(summary=summary, analyses=analyses, limit=args.max_review_cases), encoding="utf-8")
    print(json.dumps({"summary": str(args.output_json), "review_html": str(args.review_html) if args.review_html else None}, ensure_ascii=False, indent=2))


def _load_cases(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return {case.case_id: case for case in load_urbannav_cases(path)}


def _load_eval_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {str(row.get("case_id") or ""): row for row in _iter_jsonl(path)}


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _analyze_prediction(*, payload: dict[str, Any], case: Any | None, eval_row: dict[str, Any] | None) -> dict[str, Any]:
    case_id = str(payload.get("case_id") or payload.get("id") or "")
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
    agent_trace = trace.get("agent_trace") if isinstance(trace.get("agent_trace"), dict) else {}
    turns = agent_trace.get("turns") if isinstance(agent_trace.get("turns"), list) else []
    case_payload = _case_payload_from_turns(turns)
    category = str(
        (case.category if case is not None else "")
        or payload.get("category")
        or trace.get("category")
        or case_payload.get("category")
        or ""
    )
    subcategory = str((case.subcategory if case is not None else "") or trace.get("subcategory") or case_payload.get("subcategory") or "")
    tool_policy = case_payload.get("tool_policy") if isinstance(case_payload.get("tool_policy"), dict) else {}
    tool_calls = [_tool_call_from_turn(turn) for turn in turns if turn.get("role") == "tool_call"]
    tool_calls = [call for call in tool_calls if call is not None]
    tool_responses = [_tool_response_from_turn(turn) for turn in turns if turn.get("role") == "tool_response"]
    assistant_texts = [str(turn.get("content") or "") for turn in turns if turn.get("role") == "assistant"]
    finish = trace.get("finish") if isinstance(trace.get("finish"), dict) else _finish_from_assistant_texts(assistant_texts)
    prediction_exception = trace.get("error") == "prediction_exception"
    exception_detail = str(trace.get("detail") or "") if prediction_exception else ""
    action_valid = _action_valid(exception_detail=exception_detail, tool_calls=tool_calls)
    sequence_success, sequence_reason = _required_sequence_success(
        category=category,
        tool_policy=tool_policy,
        tool_calls=tool_calls,
        finish=finish,
    )
    grounded_finish, grounded_reason = _grounded_finish(tool_responses=tool_responses, finish=finish, category=category)
    tool_execution_success = bool(tool_responses) and all(bool(response.get("success")) for response in tool_responses)
    failure_reasons = []
    if prediction_exception:
        failure_reasons.append(exception_detail or "prediction_exception")
    if not action_valid:
        failure_reasons.append("tool_action_invalid")
    if not sequence_success:
        failure_reasons.append("tool_sequence_failed:" + sequence_reason)
    if not grounded_finish:
        failure_reasons.append("grounded_finish_failed:" + grounded_reason)
    if not tool_execution_success:
        failure_reasons.append("tool_execution_failed_or_missing")
    return {
        "case_id": case_id,
        "instruction": str((case.instruction if case is not None else "") or case_payload.get("instruction") or ""),
        "category": category,
        "subcategory": subcategory,
        "city": str((case.city if case is not None else "") or trace.get("city") or case_payload.get("city") or ""),
        "prediction_exception": prediction_exception,
        "prediction_exception_detail": exception_detail or None,
        "tool_action_valid": action_valid,
        "required_tool_sequence_success": sequence_success,
        "required_tool_sequence_reason": sequence_reason,
        "grounded_finish": grounded_finish,
        "grounded_finish_reason": grounded_reason,
        "tool_execution_success": tool_execution_success,
        "tool_call_count": len(tool_calls),
        "tool_response_count": len(tool_responses),
        "tool_ops": [_op(call) for call in tool_calls],
        "finish": finish,
        "failure_reasons": failure_reasons,
        "eval": eval_row or {},
        "turns": turns,
        "tool_policy": tool_policy,
    }


def _case_payload_from_turns(turns: list[dict[str, Any]]) -> dict[str, Any]:
    for turn in turns:
        if turn.get("role") != "user":
            continue
        content = str(turn.get("content") or "")
        marker = "UrbanNav case JSON:\n"
        if marker not in content:
            continue
        raw = content.split(marker, 1)[1].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _tool_call_from_turn(turn: dict[str, Any]) -> ToolCall | None:
    content = turn.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return None
    if not isinstance(content, dict):
        return None
    try:
        call = ToolCall.from_dict(content)
    except Exception:
        return None
    if call.tool == "map_tool":
        return ToolCall(tool=call.tool, arguments=normalize_map_tool_arguments(call.arguments), call_id=call.call_id)
    return call


def _tool_response_from_turn(turn: dict[str, Any]) -> dict[str, Any]:
    content = turn.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return {"success": False, "error": {"code": "unparseable_tool_response"}}
    return dict(content) if isinstance(content, dict) else {"success": False, "error": {"code": "invalid_tool_response"}}


def _finish_from_assistant_texts(texts: list[str]) -> dict[str, Any]:
    for text in reversed(texts):
        action = parse_agent_action(text)
        if action.action_type == "finish" and isinstance(action.finish, dict):
            return dict(action.finish)
    return {}


def _action_valid(*, exception_detail: str, tool_calls: list[ToolCall]) -> bool:
    if exception_detail.startswith("invalid_agent_action") or exception_detail == "empty_model_response":
        return False
    return all(validate_tool_call(call)[0] for call in tool_calls)


def _required_sequence_success(
    *,
    category: str,
    tool_policy: dict[str, Any],
    tool_calls: list[ToolCall],
    finish: dict[str, Any],
) -> tuple[bool, str]:
    ops = [_op(call) for call in tool_calls]
    if not finish:
        return False, "missing_finish"
    if category == "multi_stop_route" or tool_policy.get("task_mode") == "ordered_multi_stop":
        goal_count = int(tool_policy.get("goal_count") or len(finish.get("destination_sequence_ids") or []) or 2)
        cursor = 0
        for stop_index in range(1, goal_count + 1):
            search_at = _find_op_with_stop(tool_calls, "search_destination", stop_index, start=cursor)
            if search_at < 0:
                return False, f"missing_search_stop_{stop_index}"
            plan_at = _find_op_with_stop(tool_calls, "plan_route", stop_index, start=search_at + 1)
            if plan_at < 0:
                return False, f"missing_plan_stop_{stop_index}"
            cursor = plan_at + 1
        return True, "ok"
    if str(category).startswith("route_update") or tool_policy.get("task_mode") == "route_update":
        if not ops or ops[0] != "get_route_session":
            return False, "missing_initial_get_route_session"
        recommended = str(tool_policy.get("recommended_update_op") or "").strip()
        if recommended:
            try:
                update_index = ops.index(recommended)
            except ValueError:
                return False, f"missing_recommended_op_{recommended}"
            if recommended in {"add_via", "replace_destination"} and "search_destination" not in ops[1:update_index]:
                return False, "missing_update_search_destination"
        return True, "ok"
    try:
        search_index = ops.index("search_destination")
    except ValueError:
        return False, "missing_search_destination"
    try:
        plan_index = ops.index("plan_route")
    except ValueError:
        return False, "missing_plan_route"
    if plan_index <= search_index:
        return False, "plan_before_search"
    return True, "ok"


def _find_op_with_stop(tool_calls: list[ToolCall], op: str, stop_index: int, *, start: int) -> int:
    for index in range(start, len(tool_calls)):
        call = tool_calls[index]
        if _op(call) != op:
            continue
        try:
            if int(call.arguments.get("stop_index")) == stop_index:
                return index
        except Exception:
            continue
    return -1


def _grounded_finish(*, tool_responses: list[dict[str, Any]], finish: dict[str, Any], category: str) -> tuple[bool, str]:
    if not finish:
        return False, "missing_finish"
    candidate_ids: set[str] = set()
    route_ids: set[str] = set()
    for response in tool_responses:
        result = response.get("result") if isinstance(response.get("result"), dict) else {}
        route_id = result.get("route_id")
        if route_id:
            route_ids.add(str(route_id))
        extension = result.get("extension") if isinstance(result.get("extension"), dict) else {}
        if extension.get("final_route_id"):
            route_ids.add(str(extension["final_route_id"]))
        for candidate in result.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("candidate_id"):
                candidate_ids.add(str(candidate["candidate_id"]))
            if candidate.get("alias"):
                candidate_ids.add(str(candidate["alias"]))
    finish_route_id = str(finish.get("route_id") or finish.get("chosen_route_id") or "").strip()
    if not finish_route_id or finish_route_id not in route_ids:
        return False, "route_id_not_returned"
    if category == "multi_stop_route":
        raw_ids = finish.get("destination_sequence_ids") or finish.get("chosen_destination_sequence_ids") or []
        if not isinstance(raw_ids, list) or not raw_ids:
            return False, "missing_destination_sequence_ids"
        missing = [str(item) for item in raw_ids if str(item) not in candidate_ids]
        return (not missing, "ok" if not missing else "sequence_id_not_returned")
    destination_id = str(
        finish.get("chosen_destination_id") or finish.get("destination_id") or finish.get("candidate_id") or ""
    ).strip()
    if not destination_id:
        return False, "missing_chosen_destination_id"
    if destination_id not in candidate_ids:
        return False, "destination_id_not_returned"
    return True, "ok"


def _op(call: ToolCall) -> str:
    return str(call.arguments.get("op") or call.op or "").strip()


def _summarize(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(analyses)
    category_groups: dict[str, list[dict[str, Any]]] = {}
    for item in analyses:
        category_groups.setdefault(str(item.get("category") or "unknown"), []).append(item)
    return {
        "schema_version": "urbannav_tool_agent_trace_summary_v1",
        "total_cases": total,
        "tool_action_valid_rate": _mean_bool(item["tool_action_valid"] for item in analyses),
        "required_tool_sequence_success": _mean_bool(item["required_tool_sequence_success"] for item in analyses),
        "grounded_finish_rate": _mean_bool(item["grounded_finish"] for item in analyses),
        "tool_execution_success_rate": _mean_bool(item["tool_execution_success"] for item in analyses),
        "prediction_exception_rate": _mean_bool(item["prediction_exception"] for item in analyses),
        "prediction_exception_count": sum(1 for item in analyses if item["prediction_exception"]),
        "mean_tool_call_count": sum(float(item["tool_call_count"]) for item in analyses) / total if total else 0.0,
        "failure_reason_counts": dict(Counter(reason for item in analyses for reason in item["failure_reasons"]).most_common()),
        "by_category": {category: _summarize_group(group) for category, group in sorted(category_groups.items())},
    }


def _summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(group),
        "tool_action_valid_rate": _mean_bool(item["tool_action_valid"] for item in group),
        "required_tool_sequence_success": _mean_bool(item["required_tool_sequence_success"] for item in group),
        "grounded_finish_rate": _mean_bool(item["grounded_finish"] for item in group),
        "prediction_exception_rate": _mean_bool(item["prediction_exception"] for item in group),
    }


def _mean_bool(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1.0 if bool(item) else 0.0 for item in items) / len(items)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# UrbanNav Tool-Agent Trace Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total cases | {summary['total_cases']} |",
        f"| Tool action valid rate | {summary['tool_action_valid_rate']:.4f} |",
        f"| Required tool sequence success | {summary['required_tool_sequence_success']:.4f} |",
        f"| Grounded finish rate | {summary['grounded_finish_rate']:.4f} |",
        f"| Tool execution success rate | {summary['tool_execution_success_rate']:.4f} |",
        f"| Prediction exception rate | {summary['prediction_exception_rate']:.4f} |",
        "",
        "## By Category",
        "",
        "| Category | Cases | Action Valid | Sequence Success | Grounded Finish | Exception |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, row in summary["by_category"].items():
        lines.append(
            f"| `{category}` | {row['cases']} | {row['tool_action_valid_rate']:.4f} | "
            f"{row['required_tool_sequence_success']:.4f} | {row['grounded_finish_rate']:.4f} | "
            f"{row['prediction_exception_rate']:.4f} |"
        )
    lines.extend(["", "## Failure Reasons", "", "```json", json.dumps(summary["failure_reason_counts"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def _html(*, summary: dict[str, Any], analyses: list[dict[str, Any]], limit: int) -> str:
    ordered = sorted(
        analyses,
        key=lambda item: (
            not bool(item["failure_reasons"]),
            str(item.get("category") or ""),
            str(item.get("case_id") or ""),
        ),
    )[: max(0, limit)]
    cards = "\n".join(_case_card(item) for item in ordered)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UrbanNav Tool-Agent Trace Review</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f8; color: #18212a; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 10px; }}
    .summary, .card {{ background: white; border: 1px solid #d9e0e7; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
    .metric {{ background: #eef3f7; border-radius: 6px; padding: 10px; }}
    .metric b {{ display: block; font-size: 12px; color: #52606d; }}
    .metric span {{ font-size: 20px; font-weight: 650; }}
    .bad {{ color: #b42318; }}
    .ok {{ color: #057a55; }}
    .tag {{ display: inline-block; background: #e7eef6; border-radius: 999px; padding: 3px 8px; margin: 2px; font-size: 12px; }}
    pre {{ white-space: pre-wrap; overflow-x: auto; background: #0f1720; color: #e8eef5; padding: 10px; border-radius: 6px; font-size: 12px; }}
    details {{ margin-top: 10px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border-bottom: 1px solid #e1e7ee; text-align: left; padding: 7px; }}
  </style>
</head>
<body>
<main>
  <h1>UrbanNav Tool-Agent Trace Review</h1>
  <section class="summary">
    <div class="grid">
      {_metric("Cases", summary["total_cases"])}
      {_metric("Action valid", summary["tool_action_valid_rate"])}
      {_metric("Sequence success", summary["required_tool_sequence_success"])}
      {_metric("Grounded finish", summary["grounded_finish_rate"])}
      {_metric("Tool exec success", summary["tool_execution_success_rate"])}
      {_metric("Prediction exception", summary["prediction_exception_rate"])}
    </div>
    <details open><summary>Failure reason counts</summary><pre>{_esc(json.dumps(summary["failure_reason_counts"], ensure_ascii=False, indent=2))}</pre></details>
  </section>
  {cards}
</main>
</body>
</html>
"""


def _metric(name: str, value: Any) -> str:
    text = f"{value:.4f}" if isinstance(value, float) else str(value)
    cls = "bad" if "exception" in name.lower() and isinstance(value, float) and value > 0 else "ok"
    return f'<div class="metric"><b>{_esc(name)}</b><span class="{cls}">{_esc(text)}</span></div>'


def _case_card(item: dict[str, Any]) -> str:
    failure = item["failure_reasons"]
    cls = "bad" if failure else "ok"
    eval_row = item.get("eval") or {}
    return f"""
  <article class="card">
    <h2>#{_esc(item['case_id'])}</h2>
    <p>{_esc(item.get('instruction') or '')}</p>
    <div>
      <span class="tag">{_esc(item.get('category') or '')}</span>
      <span class="tag">{_esc(item.get('subcategory') or '')}</span>
      <span class="tag">{_esc(item.get('city') or '')}</span>
      <span class="tag {cls}">{_esc('ok' if not failure else ', '.join(failure[:3]))}</span>
    </div>
    <table>
      <tr><th>action_valid</th><td>{item['tool_action_valid']}</td><th>sequence_success</th><td>{item['required_tool_sequence_success']} ({_esc(item['required_tool_sequence_reason'])})</td></tr>
      <tr><th>grounded_finish</th><td>{item['grounded_finish']} ({_esc(item['grounded_finish_reason'])})</td><th>tool_exec_success</th><td>{item['tool_execution_success']}</td></tr>
      <tr><th>ops</th><td colspan="3">{_esc(' -> '.join(item['tool_ops']))}</td></tr>
      <tr><th>eval</th><td colspan="3">{_esc(_compact_eval(eval_row))}</td></tr>
    </table>
    <details><summary>Tool policy</summary><pre>{_esc(json.dumps(item.get('tool_policy') or {}, ensure_ascii=False, indent=2))}</pre></details>
    <details><summary>Finish</summary><pre>{_esc(json.dumps(item.get('finish') or {}, ensure_ascii=False, indent=2))}</pre></details>
    <details><summary>Turns</summary><pre>{_esc(json.dumps(item.get('turns') or [], ensure_ascii=False, indent=2))}</pre></details>
  </article>
"""


def _compact_eval(row: dict[str, Any]) -> str:
    if not row:
        return ""
    keys = ["destination_success", "hit@5", "task_hit@5", "route_success", "spl", "task_spl", "prediction_exception"]
    return json.dumps({key: row.get(key) for key in keys if key in row}, ensure_ascii=False, sort_keys=True)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
