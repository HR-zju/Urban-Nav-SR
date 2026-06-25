"""Pipeline helpers for Phase1 archive to Agentic Stage2 data builds.

The build CLI needs deterministic split assignment, slice-level reports, and
review sampling.  Keeping those rules here prevents the command-line script
from becoming the protocol definition.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mcp_server.agent.schemas import AgentStage2Case, AgentTrace, ToolCall
from mcp_server.eval.cases import write_stage2_cases
from mcp_server.eval.exporters import write_traces_jsonl


STAGE2_SPLITS = ("train", "val", "heldout")


@dataclass(frozen=True)
class Stage2SplitConfig:
    """Configuration for deterministic Stage2 split assignment.

    Args:
        train_ratio: Fraction assigned to ``train``.
        val_ratio: Fraction assigned to ``val``.
        heldout_ratio: Fraction assigned to ``heldout``.
        seed: Stable string mixed into the hash key.

    The assignment computes:

    \\[
    b = H(seed, split\\_key) \\bmod 10000
    \\]

    and maps the bucket ``b`` to train/val/heldout intervals.  Hashing on
    stable ids keeps future rebuilds reproducible even when input row order
    changes.
    """

    train_ratio: float = 0.90
    val_ratio: float = 0.05
    heldout_ratio: float = 0.05
    seed: str = "sidewalk_agentic_stage2_v1"

    def normalized_thresholds(self) -> tuple[int, int]:
        """Return bucket thresholds for train and val.

        Returns:
            ``(train_threshold, val_threshold)`` over a 10000-bucket space.

        Raises:
            ValueError: If ratios are negative or sum to zero.
        """

        ratios = [float(self.train_ratio), float(self.val_ratio), float(self.heldout_ratio)]
        if any(value < 0 for value in ratios):
            raise ValueError(f"Stage2 split ratios must be non-negative: {ratios}")
        total = sum(ratios)
        if total <= 0:
            raise ValueError("Stage2 split ratios must sum to a positive value")
        train = int(round((ratios[0] / total) * 10000))
        val = int(round(((ratios[0] + ratios[1]) / total) * 10000))
        return min(train, 10000), min(max(val, train), 10000)


def assign_stage2_split(case: AgentStage2Case, config: Stage2SplitConfig) -> str:
    """Assign one case to a stable Stage2 split.

    Args:
        case: Stage2 case.
        config: Split ratio and seed.

    Returns:
        ``train``, ``val``, or ``heldout``.
    """

    train_threshold, val_threshold = config.normalized_thresholds()
    key = _split_key(case)
    digest = hashlib.sha1(f"{config.seed}:{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) % 10000
    if bucket < train_threshold:
        return "train"
    if bucket < val_threshold:
        return "val"
    return "heldout"


def apply_stage2_splits(cases: list[AgentStage2Case], config: Stage2SplitConfig) -> dict[str, int]:
    """Mutate case split fields with deterministic assignments.

    Args:
        cases: Stage2 cases to update.
        config: Split ratio and seed.

    Returns:
        Count dictionary keyed by split name.
    """

    counts = {split: 0 for split in STAGE2_SPLITS}
    for case in cases:
        split = assign_stage2_split(case, config)
        case.split = split
        case.metadata["stage2_split_seed"] = config.seed
        case.metadata["stage2_split_key"] = _split_key(case)
        counts[split] += 1
    return counts


def count_stage2_splits(cases: list[AgentStage2Case]) -> dict[str, int]:
    """Count existing case split assignments.

    Args:
        cases: Stage2 cases whose ``split`` fields are already populated.

    Returns:
        Count dictionary keyed by split name.  The canonical ``train`` /
        ``val`` / ``heldout`` keys are always present; unexpected split labels
        are preserved as additional keys so report output can expose drift.
    """

    counts = {split: 0 for split in STAGE2_SPLITS}
    for case in cases:
        split = str(case.split or "unknown")
        counts[split] = counts.get(split, 0) + 1
    return counts


def classify_trace_tool_policy(trace: AgentTrace) -> str:
    """Classify a Stage2 trace as ``tool``, ``no_tool``, or ``failed_tool``.

    Args:
        trace: Stage2 trace.

    Returns:
        Tool-policy slice label.
    """

    has_tool = any(turn.role == "tool_call" for turn in trace.turns)
    if not has_tool:
        return "no_tool"
    for turn in trace.turns:
        if turn.role != "tool_response":
            continue
        content = turn.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return "failed_tool"
        if isinstance(content, dict) and content.get("success") is False:
            return "failed_tool"
    return "tool"


def first_trace_tool_call(trace: AgentTrace) -> ToolCall | None:
    """Return the first normalized tool call from a trace, if present.

    Args:
        trace: Stage2 trace.

    Returns:
        Parsed ``ToolCall`` when a valid ``tool_call`` role exists; otherwise
        ``None``.
    """

    for turn in trace.turns:
        if turn.role != "tool_call":
            continue
        content = turn.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return None
        if isinstance(content, dict):
            return ToolCall.from_dict(content)
    return None


def summarize_stage2_slices(
    cases: list[AgentStage2Case],
    traces: list[AgentTrace],
    rows: list[dict[str, Any]],
    *,
    case_validations: list[dict[str, Any]],
    trace_validations: list[dict[str, Any]],
    row_validations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize Stage2 build outputs by split/tool/op slices.

    Args:
        cases: Cases in build order.
        traces: Traces in build order.
        rows: Exported ms-swift rows in build order.
        case_validations: Case validation records.
        trace_validations: Trace validation records.
        row_validations: Row validation records.

    Returns:
        Nested summary with aggregate counts and per-slice metric averages.
    """

    records = _records(cases, traces, rows, case_validations, trace_validations, row_validations)
    return {
        "by_split": _summarize_by(records, key="split"),
        "by_tool_policy": _summarize_by(records, key="tool_policy"),
        "by_tool_name": _summarize_by(records, key="tool_name"),
        "by_op": _summarize_by(records, key="op"),
        "by_validation": _summarize_by(records, key="validation"),
        "by_intent": _summarize_by(records, key="phase1_intent"),
        "records": records,
    }


def sample_stage2_review_rows(
    cases: list[AgentStage2Case],
    traces: list[AgentTrace],
    rows: list[dict[str, Any]],
    *,
    case_validations: list[dict[str, Any]],
    trace_validations: list[dict[str, Any]],
    row_validations: list[dict[str, Any]],
    per_slice: int = 5,
) -> list[dict[str, Any]]:
    """Sample small review rows from every split/tool-policy/op slice.

    Args:
        cases: Cases in build order.
        traces: Traces in build order.
        rows: Exported ms-swift rows in build order.
        case_validations: Case validation records.
        trace_validations: Trace validation records.
        row_validations: Row validation records.
        per_slice: Maximum rows per ``(split, tool_policy, op)`` slice.

    Returns:
        JSON-serializable review rows with previews and validation errors.
    """

    if per_slice <= 0:
        return []
    records = _records(cases, traces, rows, case_validations, trace_validations, row_validations)
    counts: dict[tuple[str, str, str], int] = {}
    samples: list[dict[str, Any]] = []
    for record, case, trace, row, case_val, trace_val, row_val in zip(
        records,
        cases,
        traces,
        rows,
        case_validations,
        trace_validations,
        row_validations,
    ):
        key = (record["split"], record["tool_policy"], record["op"])
        if counts.get(key, 0) >= per_slice:
            continue
        counts[key] = counts.get(key, 0) + 1
        final_text = _final_assistant_text(trace)
        phase1_response = str(case.ground_truth.get("phase1_response") or "")
        samples.append(
            {
                "case_id": case.case_id,
                "split": record["split"],
                "tool_policy": record["tool_policy"],
                "tool_name": record["tool_name"],
                "op": record["op"],
                "phase1_intent": record["phase1_intent"],
                "images": case.images,
                "metrics": trace.metrics,
                "validation_errors": {
                    "case": list(case_val.get("errors") or []),
                    "trace": list(trace_val.get("errors") or []),
                    "row": list(row_val.get("errors") or []),
                },
                "phase1_response_preview": _preview(phase1_response),
                "final_response_preview": _preview(final_text),
                "ms_swift_message_roles": [message.get("role") for message in row.get("messages") or []],
            }
        )
    return samples


def write_jsonl_records(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write generic JSONL records to disk.

    Args:
        records: JSON-serializable records.
        path: Destination JSONL path.

    Returns:
        ``None``.  Parent directories are created automatically.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_review_index_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write a compact CSV index for manual review samples.

    Args:
        records: Records returned by ``sample_stage2_review_rows``.
        path: Destination CSV path.

    Returns:
        ``None``.  The CSV is intended for quick manual filtering by split,
        tool policy, tool name, op, and validation errors.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "split",
        "tool_policy",
        "tool_name",
        "op",
        "phase1_intent",
        "images",
        "case_errors",
        "trace_errors",
        "row_errors",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            errors = record.get("validation_errors") or {}
            writer.writerow(
                {
                    "case_id": record.get("case_id"),
                    "split": record.get("split"),
                    "tool_policy": record.get("tool_policy"),
                    "tool_name": record.get("tool_name"),
                    "op": record.get("op"),
                    "phase1_intent": record.get("phase1_intent"),
                    "images": "|".join(str(item) for item in record.get("images") or []),
                    "case_errors": "|".join(str(item) for item in errors.get("case") or []),
                    "trace_errors": "|".join(str(item) for item in errors.get("trace") or []),
                    "row_errors": "|".join(str(item) for item in errors.get("row") or []),
                }
            )


def write_stage2_split_outputs(
    *,
    cases: list[AgentStage2Case],
    traces: list[AgentTrace],
    rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Write cases/traces/ms-swift rows grouped by assigned split.

    Args:
        cases: Cases with ``case.split`` already assigned.
        traces: Traces in the same order as cases.
        rows: ms-swift rows in the same order as cases.
        output_dir: Destination directory.

    Returns:
        Mapping from split to written file paths and record count.

    Raises:
        ValueError: If cases, traces, and rows do not have aligned lengths.
    """

    if not (len(cases) == len(traces) == len(rows)):
        raise ValueError(
            "cases, traces, and rows must have aligned lengths: "
            f"{len(cases)} != {len(traces)} != {len(rows)}"
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for split in STAGE2_SPLITS:
        split_cases = [case for case in cases if case.split == split]
        split_traces = [trace for case, trace in zip(cases, traces) if case.split == split]
        split_rows = [row for case, row in zip(cases, rows) if case.split == split]
        cases_path = root / f"sidewalk_agent_stage2_cases_{split}.jsonl"
        traces_path = root / f"sidewalk_agent_stage2_traces_{split}.jsonl"
        rows_path = root / f"sidewalk_agentic_stage2_ms_swift_{split}.jsonl"
        write_stage2_cases(split_cases, cases_path)
        write_traces_jsonl(split_traces, traces_path)
        write_ms_swift_rows(split_rows, rows_path)
        outputs[split] = {
            "cases": str(cases_path),
            "traces": str(traces_path),
            "ms_swift": str(rows_path),
            "count": len(split_cases),
        }
    return outputs


def write_ms_swift_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write already-exported ms-swift rows without regenerating them.

    Args:
        rows: JSON-serializable ms-swift Stage2 rows.
        path: Destination JSONL path.

    Returns:
        ``None``.  This helper is used for split outputs after rows have
        already been generated and validated once.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_key(case: AgentStage2Case) -> str:
    meta = case.metadata or {}
    phase1_meta = meta.get("phase1_sidewalk_meta") if isinstance(meta.get("phase1_sidewalk_meta"), dict) else {}
    for payload in (phase1_meta, case.observation, meta):
        for key in ("source_id", "phase1_id", "clip_id", "frame_name", "frame_idx"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if value not in (None, ""):
                return str(value)
    return case.case_id


def _records(
    cases: list[AgentStage2Case],
    traces: list[AgentTrace],
    rows: list[dict[str, Any]],
    case_validations: list[dict[str, Any]],
    trace_validations: list[dict[str, Any]],
    row_validations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case, trace, row, case_val, trace_val, row_val in zip(
        cases,
        traces,
        rows,
        case_validations,
        trace_validations,
        row_validations,
    ):
        call = first_trace_tool_call(trace)
        policy = classify_trace_tool_policy(trace)
        all_valid = bool(case_val.get("valid")) and bool(trace_val.get("valid")) and bool(row_val.get("valid"))
        records.append(
            {
                "case_id": case.case_id,
                "split": case.split or "unknown",
                "tool_policy": policy,
                "tool_name": call.tool if call is not None else "none",
                "op": call.op if call is not None and call.op else "none",
                "validation": "valid" if all_valid else "invalid",
                "phase1_intent": _phase1_intent(case),
                "metrics": dict(trace.metrics),
                "message_count": len(row.get("messages") or []),
            }
        )
    return records


def _summarize_by(records: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(key) or "unknown"), []).append(record)
    return {name: _summarize_records(items) for name, items in sorted(grouped.items())}


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted({key for record in records for key in (record.get("metrics") or {}).keys()})
    return {
        "count": len(records),
        "valid_count": sum(1 for record in records if record.get("validation") == "valid"),
        "invalid_count": sum(1 for record in records if record.get("validation") != "valid"),
        "metrics": {
            metric: _metric_average([(record.get("metrics") or {}).get(metric) for record in records])
            for metric in metric_names
        },
    }


def _metric_average(values: list[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 6)


def _phase1_intent(case: AgentStage2Case) -> str:
    for payload in (case.ground_truth, case.observation, case.metadata):
        if not isinstance(payload, dict):
            continue
        for key in ("intent", "intent_gt"):
            value = payload.get(key)
            if value not in (None, ""):
                return str(value)
    return "unknown"


def _final_assistant_text(trace: AgentTrace) -> str:
    for turn in reversed(trace.turns):
        if turn.role == "assistant" and isinstance(turn.content, str):
            return turn.content
    return ""


def _preview(text: str, limit: int = 500) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:limit]
