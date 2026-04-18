from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _extract_balanced_json(text: str, start_index: int) -> tuple[str | None, int]:
    if start_index < 0 or start_index >= len(text) or text[start_index] != "{":
        return None, start_index

    depth = 0
    in_string = False
    escaped = False
    quote_char = '"'

    for index in range(start_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote_char:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue

        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1], index + 1

    return None, len(text)


def _extract_timing_records_from_text(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cursor = 0
    marker = "TIMING"
    while cursor < len(text):
        hit = text.find(marker, cursor)
        if hit < 0:
            break
        brace_index = text.find("{", hit)
        if brace_index < 0:
            break
        payload_text, next_cursor = _extract_balanced_json(text, brace_index)
        if payload_text is None:
            cursor = brace_index + 1
            continue
        try:
            payload = json.loads(payload_text)
        except Exception:
            cursor = next_cursor
            continue
        if isinstance(payload, dict):
            records.append(payload)
        cursor = next_cursor
    return records


def _parse_timing_record(line: str) -> dict[str, Any] | None:
    text = str(line or "").strip()
    if not text:
        return None

    # Case 2: raw JSON line only.
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
        except Exception:
            return None
        if isinstance(payload, dict) and "stage" in payload:
            return payload

    return None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 1:
        return values[-1]
    # Nearest-rank percentile.
    rank = int(math.ceil(p * len(values)))
    index = min(max(rank - 1, 0), len(values) - 1)
    return values[index]


VALID_GROUP_FIELDS = (
    "stage",
    "component",
    "tool",
    "stage_name",
    "status",
    "source",
    "cache_kind",
    "artifact",
    "fast_intent",
    "resolution_path",
    "parse_profile",
    "intent_override_used",
)


def _record_key(record: dict[str, Any], group_by_fields: list[str]) -> str:
    parts: list[str] = []
    for field in group_by_fields:
        value = record.get(field)
        text = str(value or f"unknown_{field}")
        parts.append(text if len(group_by_fields) == 1 else f"{field}={text}")
    return " | ".join(parts)


def _extract_elapsed_ms(record: dict[str, Any]) -> float | None:
    value = record.get("elapsed_ms")
    if isinstance(value, (int, float)):
        return float(value)
    # Compatibility fallback: allow elapsed_s if present.
    value_s = record.get("elapsed_s")
    if isinstance(value_s, (int, float)):
        return float(value_s) * 1000.0
    return None


def collect_stage_values(
    log_paths: list[str],
    *,
    stage_prefix: str = "",
    group_by_fields: list[str] | None = None,
) -> tuple[dict[str, list[float]], int, int]:
    stage_values: dict[str, list[float]] = {}
    parsed_lines = 0
    total_lines = 0
    effective_group_by = list(group_by_fields or ["stage"])

    for path_text in log_paths:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        file_text = path.read_text(encoding="utf-8", errors="ignore")
        for raw_line in file_text.splitlines():
            total_lines += 1

        records = _extract_timing_records_from_text(file_text)
        if not records:
            for raw_line in file_text.splitlines():
                record = _parse_timing_record(raw_line)
                if record is not None:
                    records.append(record)

        for record in records:
            parsed_lines += 1
            stage = str(record.get("stage") or "")
            if stage_prefix and not stage.startswith(stage_prefix):
                continue
            elapsed_ms = _extract_elapsed_ms(record)
            if elapsed_ms is None:
                continue
            key = _record_key(record, group_by_fields=effective_group_by)
            stage_values.setdefault(key, []).append(elapsed_ms)

    return stage_values, parsed_lines, total_lines


def summarize(stage_values: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, values in stage_values.items():
        sorted_values = sorted(float(v) for v in values)
        count = len(sorted_values)
        mean_ms = sum(sorted_values) / count
        row = {
            "stage": stage,
            "count": count,
            "mean_ms": round(mean_ms, 2),
            "p50_ms": round(_percentile(sorted_values, 0.50), 2),
            "p95_ms": round(_percentile(sorted_values, 0.95), 2),
            "max_ms": round(sorted_values[-1], 2),
        }
        rows.append(row)

    rows.sort(key=lambda item: item["stage"])
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No timing records found.")
        return

    stage_width = max(5, max(len(str(row["stage"])) for row in rows))
    header = (
        f"{'stage'.ljust(stage_width)}  "
        f"{'count'.rjust(6)}  "
        f"{'mean_ms'.rjust(10)}  "
        f"{'p50_ms'.rjust(10)}  "
        f"{'p95_ms'.rjust(10)}  "
        f"{'max_ms'.rjust(10)}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        mean_text = format(float(row["mean_ms"]), ".2f").rjust(10)
        p50_text = format(float(row["p50_ms"]), ".2f").rjust(10)
        p95_text = format(float(row["p95_ms"]), ".2f").rjust(10)
        max_text = format(float(row["max_ms"]), ".2f").rjust(10)
        print(
            f"{str(row['stage']).ljust(stage_width)}  "
            f"{str(row['count']).rjust(6)}  "
            f"{mean_text}  "
            f"{p50_text}  "
            f"{p95_text}  "
            f"{max_text}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate TIMING log records by stage and print count/mean/p50/p95/max (ms)."
    )
    parser.add_argument(
        "--log",
        dest="logs",
        action="append",
        required=True,
        help="Path to a log file. Repeat --log to pass multiple files.",
    )
    parser.add_argument(
        "--stage-prefix",
        default="",
        help="Only include stage names that start with this prefix.",
    )
    parser.add_argument(
        "--group-by-component",
        action="store_true",
        help="Compatibility flag: prepend component to the default stage grouping.",
    )
    parser.add_argument(
        "--group-by",
        dest="group_by",
        action="append",
        choices=VALID_GROUP_FIELDS,
        help="Group by one or more record fields. Repeat to build composite keys, e.g. --group-by stage --group-by tool.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of table output.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path for the summary (table text or JSON).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_by_fields = list(args.group_by or [])
    if args.group_by_component:
        if not group_by_fields:
            group_by_fields = ["component", "stage"]
        elif "component" not in group_by_fields:
            group_by_fields = ["component"] + group_by_fields
    if not group_by_fields:
        group_by_fields = ["stage"]

    stage_values, parsed_lines, total_lines = collect_stage_values(
        args.logs,
        stage_prefix=str(args.stage_prefix or ""),
        group_by_fields=group_by_fields,
    )
    rows = summarize(stage_values)

    metadata = {
        "input_files": args.logs,
        "total_lines": total_lines,
        "timing_records_parsed": parsed_lines,
        "stage_count": len(rows),
        "group_by": group_by_fields,
    }

    if args.json:
        payload = {
            "meta": metadata,
            "rows": rows,
        }
        text = json.dumps(payload, indent=2, ensure_ascii=True)
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
            print(f"Wrote JSON summary to {args.output}")
        else:
            print(text)
        return

    lines: list[str] = []
    lines.append(
        "Parsed "
        f"{metadata['timing_records_parsed']} timing records from "
        f"{metadata['total_lines']} lines across {len(args.logs)} file(s)."
    )
    if not rows:
        lines.append("No timing records found.")
        text = "\n".join(lines)
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
            print(f"Wrote table summary to {args.output}")
        else:
            print(text)
        return

    # Build table text via same renderer used for stdout.
    from io import StringIO
    import contextlib

    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_table(rows)
    table_text = buffer.getvalue().rstrip("\n")

    text = "\n".join(lines + ["", table_text])
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote table summary to {args.output}")
        return

    print(text)


if __name__ == "__main__":
    main()
