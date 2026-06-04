#!/usr/bin/env python3
"""Reuse completed SideWalk Phase1 CoT rows by sample id.

This supports changing sampling rates after a partial CoT run.  It writes:

* a prefilled CoT JSONL containing rows already available for the new source
* a missing source JSONL containing rows that still need teacher generation
"""

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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reuse existing CoT rows for a new source JSONL by id")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--existing-cot-glob", required=True)
    parser.add_argument("--prefill-jsonl", required=True)
    parser.add_argument("--missing-source-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_jsonl)
    existing_paths = [Path(p) for p in sorted(glob.glob(args.existing_cot_glob))]
    if not existing_paths:
        raise FileNotFoundError(f"No existing CoT files matched: {args.existing_cot_glob}")

    existing_by_id: dict[str, dict[str, Any]] = {}
    duplicate_existing = 0
    for path in existing_paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("id") or "")
            if not sample_id:
                continue
            if sample_id in existing_by_id:
                duplicate_existing += 1
                continue
            existing_by_id[sample_id] = row

    prefill_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    source_rows = 0
    for row in read_jsonl(source_path):
        source_rows += 1
        sample_id = str(row.get("id") or "")
        existing = existing_by_id.get(sample_id)
        if existing is not None:
            prefill_rows.append(existing)
        else:
            missing_rows.append(row)

    reused_rows = write_jsonl(Path(args.prefill_jsonl), prefill_rows)
    missing_count = write_jsonl(Path(args.missing_source_jsonl), missing_rows)
    report = {
        "schema_version": "sidewalk_phase1_cot_reuse_by_id_v1",
        "source_jsonl": str(source_path),
        "existing_cot_glob": args.existing_cot_glob,
        "existing_cot_files": [str(p) for p in existing_paths],
        "existing_unique_rows": len(existing_by_id),
        "duplicate_existing_rows": duplicate_existing,
        "source_rows": source_rows,
        "reused_rows": reused_rows,
        "missing_rows": missing_count,
        "reuse_rate": (reused_rows / source_rows) if source_rows else 0.0,
        "prefill_jsonl": args.prefill_jsonl,
        "missing_source_jsonl": args.missing_source_jsonl,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
