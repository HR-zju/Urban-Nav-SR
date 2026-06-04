#!/usr/bin/env python3
"""Merge reused and newly generated SideWalk Phase1 CoT rows in source order."""

from __future__ import annotations

import argparse
import glob
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge CoT JSONL files by id following source JSONL order")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--cot-glob", action="append", required=True, help="Glob for CoT JSONL files. Can be repeated.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_jsonl)
    cot_paths: list[Path] = []
    for pattern in args.cot_glob:
        cot_paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not cot_paths:
        raise FileNotFoundError(f"No CoT files matched: {args.cot_glob}")

    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicate_rows = 0
    for path in cot_paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("id") or "")
            if not sample_id:
                continue
            if sample_id in rows_by_id:
                duplicate_rows += 1
                continue
            rows_by_id[sample_id] = row

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    written_rows = 0
    missing_ids: list[str] = []
    with output_path.open("w", encoding="utf-8") as out_f:
        for source_row in read_jsonl(source_path):
            source_rows += 1
            sample_id = str(source_row.get("id") or "")
            cot_row = rows_by_id.get(sample_id)
            if cot_row is None:
                missing_ids.append(sample_id)
                continue
            out_f.write(json.dumps(cot_row, ensure_ascii=False) + "\n")
            written_rows += 1

    report = {
        "schema_version": "sidewalk_phase1_cot_merge_by_id_v1",
        "source_jsonl": str(source_path),
        "cot_glob": args.cot_glob,
        "cot_files": [str(p) for p in cot_paths],
        "source_rows": source_rows,
        "available_cot_rows": len(rows_by_id),
        "duplicate_cot_rows": duplicate_rows,
        "written_rows": written_rows,
        "missing_rows": len(missing_ids),
        "output_jsonl": args.output_jsonl,
        "missing_ids_preview": missing_ids[:50],
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing_ids:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
