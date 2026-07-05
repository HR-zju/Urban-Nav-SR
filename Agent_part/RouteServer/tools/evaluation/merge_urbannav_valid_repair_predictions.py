#!/usr/bin/env python3
"""Merge UrbanNav repair predictions while keeping only valid chosen rows.

The script is intentionally conservative: it never fabricates predictions and
never converts exception rows into valid rows.  A prediction is kept only when it
has a case id, is not marked as ``prediction_exception``, has a
``chosen_destination``, and has a non-empty ``route_polyline``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _case_id_from_prediction(row: dict[str, Any]) -> str:
    if row.get("case_id"):
        return str(row["case_id"])
    trace = row.get("trace") if isinstance(row.get("trace"), dict) else {}
    if trace.get("case_id"):
        return str(trace["case_id"])
    return ""


def _is_prediction_exception(row: dict[str, Any]) -> bool:
    trace = row.get("trace") if isinstance(row.get("trace"), dict) else {}
    return (
        trace.get("error") == "prediction_exception"
        or bool(row.get("prediction_exception"))
        or bool(row.get("exception"))
    )


def _is_valid_chosen(row: dict[str, Any]) -> bool:
    if _is_prediction_exception(row):
        return False
    return bool(row.get("chosen_destination")) and bool(row.get("route_polyline"))


def _iter_repair_prediction_paths(repair_root: Path) -> list[Path]:
    shard_dir = repair_root / "shards"
    if shard_dir.exists():
        return sorted(shard_dir.glob("shard_*/raw_predictions.jsonl"))
    merged = repair_root / "raw_predictions_merged.jsonl"
    if merged.exists():
        return [merged]
    direct = repair_root / "raw_predictions.jsonl"
    if direct.exists():
        return [direct]
    return []


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--base-valid", required=True, type=Path)
    parser.add_argument("--repair-root", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=15000)
    parser.add_argument("--output-valid", required=True, type=Path)
    parser.add_argument("--remaining-cases", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    args = parser.parse_args()

    dataset_rows = _read_jsonl(args.dataset)[: int(args.max_cases)]
    dataset_by_id = {str(row.get("id") or ""): row for row in dataset_rows}
    dataset_order = [str(row.get("id") or "") for row in dataset_rows]

    valid_by_case: dict[str, dict[str, Any]] = {}
    base_rows = _read_jsonl(args.base_valid)
    base_valid_count = 0
    for row in base_rows:
        cid = _case_id_from_prediction(row)
        if cid and cid in dataset_by_id and _is_valid_chosen(row):
            valid_by_case[cid] = row
            base_valid_count += 1

    repair_rows = 0
    repair_valid_rows = 0
    repair_exception_rows = 0
    repair_no_chosen_rows = 0
    replaced_or_added = 0
    for path in _iter_repair_prediction_paths(args.repair_root):
        for row in _read_jsonl(path):
            repair_rows += 1
            cid = _case_id_from_prediction(row)
            if _is_prediction_exception(row):
                repair_exception_rows += 1
            elif not _is_valid_chosen(row):
                repair_no_chosen_rows += 1
            if cid and cid in dataset_by_id and _is_valid_chosen(row):
                valid_by_case[cid] = row
                repair_valid_rows += 1
                replaced_or_added += 1

    output_rows = [valid_by_case[cid] for cid in dataset_order if cid in valid_by_case]
    remaining_rows = [dataset_by_id[cid] for cid in dataset_order if cid not in valid_by_case]
    _write_jsonl(args.output_valid, output_rows)
    _write_jsonl(args.remaining_cases, remaining_rows)

    report = {
        "schema_version": "urbannav_valid_repair_merge_v1",
        "dataset_cases": len(dataset_rows),
        "base_valid_rows": base_valid_count,
        "repair_rows": repair_rows,
        "repair_valid_rows": repair_valid_rows,
        "repair_exception_rows": repair_exception_rows,
        "repair_no_chosen_rows": repair_no_chosen_rows,
        "valid_output_rows": len(output_rows),
        "remaining_cases": len(remaining_rows),
        "valid_coverage": len(output_rows) / max(len(dataset_rows), 1),
        "output_valid": str(args.output_valid),
        "remaining_cases_jsonl": str(args.remaining_cases),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
