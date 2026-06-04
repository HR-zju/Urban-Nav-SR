#!/usr/bin/env python3
"""Downsample SideWalk Phase1 source JSONL by navigation intent.

The input is usually the full 2 FPS Phase1 source manifest.  This script can
keep rare intents such as turns at full rate while thinning common intents such
as go_straight and stop.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def parse_intent_stride(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --intent-stride value: {value!r}; expected intent=stride")
        intent, stride_text = value.split("=", 1)
        intent = intent.strip()
        stride = int(stride_text)
        if not intent:
            raise ValueError(f"Invalid empty intent in --intent-stride {value!r}")
        if stride < 1:
            raise ValueError(f"Stride must be >= 1 for {intent}")
        result[intent] = stride
    return result


def row_intent(row: dict[str, Any]) -> str:
    route_context = row.get("route_context")
    if isinstance(route_context, dict) and route_context.get("intent_gt"):
        return str(route_context["intent_gt"])
    sidewalk_meta = row.get("sidewalk_meta")
    if isinstance(sidewalk_meta, dict) and sidewalk_meta.get("intent_gt"):
        return str(sidewalk_meta["intent_gt"])
    policy = row.get("trajectory_policy")
    if isinstance(policy, dict) and policy.get("intent_gt"):
        return str(policy["intent_gt"])
    return str(row.get("intent_gt") or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample Phase1 source JSONL by intent")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument(
        "--intent-stride",
        action="append",
        default=[],
        help="Intent-specific stride, e.g. go_straight=4. Can be repeated.",
    )
    parser.add_argument("--default-stride", type=int, default=1)
    parser.add_argument(
        "--group-by",
        default="clip_id,intent",
        choices=["global_intent", "clip_id,intent"],
        help="Apply stride globally per intent or independently inside each clip and intent.",
    )
    parser.add_argument("--phase", type=int, default=0, help="Keep rows where group index %% stride == phase %% stride")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_jsonl)
    output_path = Path(args.output_jsonl)
    report_path = Path(args.report_json)
    if int(args.default_stride) < 1:
        raise ValueError("--default-stride must be >= 1")
    intent_strides = parse_intent_stride(args.intent_stride)

    input_counts: dict[str, int] = defaultdict(int)
    output_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[tuple[str, str], int] = defaultdict(int)
    input_rows = 0
    output_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for row in read_jsonl(source_path):
            input_rows += 1
            intent = row_intent(row)
            input_counts[intent] += 1
            stride = int(intent_strides.get(intent, args.default_stride))
            if args.group_by == "global_intent":
                group_key = ("", intent)
            else:
                group_key = (str(row.get("clip_id") or ""), intent)
            group_index = group_counts[group_key]
            group_counts[group_key] += 1
            keep = (group_index % stride) == (int(args.phase) % stride)
            if not keep:
                continue
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_counts[intent] += 1
            output_rows += 1

    report = {
        "schema_version": "sidewalk_phase1_downsample_by_intent_v1",
        "source_jsonl": str(source_path),
        "output_jsonl": str(output_path),
        "intent_strides": intent_strides,
        "default_stride": int(args.default_stride),
        "group_by": args.group_by,
        "phase": int(args.phase),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "keep_rate": (output_rows / input_rows) if input_rows else 0.0,
        "input_counts": dict(sorted(input_counts.items())),
        "output_counts": dict(sorted(output_counts.items())),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
