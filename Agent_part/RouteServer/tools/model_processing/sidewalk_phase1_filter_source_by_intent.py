#!/usr/bin/env python3
"""Filter SideWalk Phase1 source JSONL by intent.

The common use case is reducing the dominant ``go_straight`` class while
keeping turn/stop samples dense:

    --go-straight-keep-every 4

This keeps one out of every four go_straight samples and preserves all other
intents.  Sampling is deterministic and stable by input order.
"""

from __future__ import annotations

import argparse
import json
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


def row_intent(row: dict[str, Any]) -> str:
    for container_name in ("route_context", "sidewalk_meta"):
        container = row.get(container_name)
        if isinstance(container, dict) and container.get("intent_gt"):
            return str(container.get("intent_gt"))
    return str(row.get("intent_gt") or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample SideWalk Phase1 source rows by intent")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument(
        "--go-straight-keep-every",
        type=int,
        default=4,
        help="Keep one go_straight row every N go_straight rows. Use 1 to keep all.",
    )
    parser.add_argument(
        "--keep-intents",
        default="stop,turn_left,turn_right",
        help="Comma-separated intents to always keep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_jsonl)
    output_path = Path(args.output_jsonl)
    report_path = Path(args.report_json)
    go_keep_every = max(1, int(args.go_straight_keep_every))
    keep_intents = {item.strip() for item in str(args.keep_intents).split(",") if item.strip()}

    input_rows = 0
    output_rows = 0
    go_seen = 0
    intent_counts_in: dict[str, int] = {}
    intent_counts_out: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for row in read_jsonl(source_path):
            input_rows += 1
            intent = row_intent(row)
            intent_counts_in[intent] = int(intent_counts_in.get(intent, 0)) + 1

            keep = False
            if intent == "go_straight":
                go_seen += 1
                keep = (go_seen - 1) % go_keep_every == 0
            elif intent in keep_intents:
                keep = True
            else:
                keep = True

            if keep:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_rows += 1
                intent_counts_out[intent] = int(intent_counts_out.get(intent, 0)) + 1
            else:
                skipped_counts[intent] = int(skipped_counts.get(intent, 0)) + 1

    report = {
        "schema_version": "sidewalk_phase1_filter_source_by_intent_v1",
        "source_jsonl": str(source_path),
        "output_jsonl": str(output_path),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "go_straight_keep_every": go_keep_every,
        "keep_intents": sorted(keep_intents),
        "intent_counts_in": intent_counts_in,
        "intent_counts_out": intent_counts_out,
        "skipped_counts": skipped_counts,
        "keep_rate": (output_rows / input_rows) if input_rows else 0.0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
