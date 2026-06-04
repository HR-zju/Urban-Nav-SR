#!/usr/bin/env python3
"""Build a balanced review subset from an archived SideWalk Phase1 dataset."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sidewalk_phase1_archive import render_sample_html, safe_name, write_review_csv  # noqa: E402


def read_review_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def choose_rows(
    rows: list[dict[str, Any]],
    *,
    per_clip: int,
    max_samples: int | None,
    priority_intents: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("clip_id") or "unknown_clip")].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    intent_rank = {intent: idx for idx, intent in enumerate(priority_intents)}

    for clip_id in sorted(grouped):
        clip_rows = list(grouped[clip_id])
        by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in clip_rows:
            by_intent[str(row.get("intent_gt") or "unknown")].append(row)
        for items in by_intent.values():
            rng.shuffle(items)

        chosen: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for intent in priority_intents:
            if len(chosen) >= per_clip:
                break
            item = next((row for row in by_intent.get(intent, []) if str(row.get("id")) not in seen_ids), None)
            if item is not None:
                chosen.append(item)
                seen_ids.add(str(item.get("id")))

        remainder = [row for row in clip_rows if str(row.get("id")) not in seen_ids]
        remainder.sort(
            key=lambda row: (
                intent_rank.get(str(row.get("intent_gt") or "unknown"), len(intent_rank)),
                str(row.get("frame_name") or row.get("frame_idx") or ""),
            )
        )
        for item in remainder:
            if len(chosen) >= per_clip:
                break
            chosen.append(item)
            seen_ids.add(str(item.get("id")))

        selected.extend(chosen)

    if max_samples is not None and len(selected) > max_samples:
        selected.sort(
            key=lambda row: (
                priority_intents.index(str(row.get("intent_gt")))
                if str(row.get("intent_gt")) in priority_intents
                else len(priority_intents),
                str(row.get("clip_id") or ""),
                str(row.get("frame_name") or row.get("frame_idx") or ""),
            )
        )
        selected = selected[: int(max_samples)]

    return selected


def write_balanced_review(
    *,
    archive_root: Path,
    split: str,
    selected_rows: list[dict[str, Any]],
    output_name: str,
) -> dict[str, Any]:
    review_root = archive_root / "review" / output_name
    sample_dir = review_root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = archive_root / "manifests"

    links: list[str] = []
    for row in selected_rows:
        name = safe_name(f"{row.get('clip_id')}_{row.get('intent_gt')}_{row.get('frame_name') or row.get('frame_idx')}")
        sample_path = sample_dir / f"{name}.html"
        sample_path.write_text(render_sample_html(archive_root, sample_path, row), encoding="utf-8")
        links.append(
            "<li>"
            f"[{html.escape(str(row.get('intent_gt') or 'unknown'))}] "
            f"<a href=\"samples/{html.escape(sample_path.name)}\">{html.escape(str(row.get('id') or name))}</a>"
            f" clip={html.escape(str(row.get('clip_id') or ''))}"
            "</li>"
        )

    index = "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'><title>SideWalk Balanced Review</title></head><body>",
            "<h1>SideWalk Balanced Review</h1>",
            f"<p>split={html.escape(split)} samples={len(selected_rows)}</p>",
            "<ul>",
            *links,
            "</ul></body></html>",
        ]
    )
    (review_root / "index.html").write_text(index, encoding="utf-8")

    csv_path = manifest_dir / f"{split}.{output_name}.review.csv"
    write_review_csv(csv_path, selected_rows)

    summary = {
        "split": split,
        "review_root": str(review_root),
        "review_index": str(review_root / "index.html"),
        "review_csv": str(csv_path),
        "sample_rows": len(selected_rows),
        "clip_count": len({row.get("clip_id") for row in selected_rows}),
        "intent_counts": dict(Counter(str(row.get("intent_gt") or "unknown") for row in selected_rows)),
    }
    summary_path = manifest_dir / f"{split}.{output_name}.review_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, help="Archive root containing manifests/{split}.review.csv")
    parser.add_argument("--split", required=True)
    parser.add_argument("--per-clip", type=int, default=3, help="Maximum review samples per clip")
    parser.add_argument("--max-samples", type=int, default=2000, help="Global cap. Use <=0 for no cap")
    parser.add_argument(
        "--priority-intents",
        default="stop,turn_left,turn_right,go_straight",
        help="Comma-separated intent priority within each clip",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-name", default="balanced")
    args = parser.parse_args()

    archive_root = Path(args.archive_root).expanduser().resolve()
    review_csv = archive_root / "manifests" / f"{args.split}.review.csv"
    rows = read_review_csv(review_csv)
    priority_intents = [item.strip() for item in str(args.priority_intents).split(",") if item.strip()]
    selected = choose_rows(
        rows,
        per_clip=max(1, int(args.per_clip)),
        max_samples=None if int(args.max_samples) <= 0 else int(args.max_samples),
        priority_intents=priority_intents,
        seed=int(args.seed),
    )
    summary = write_balanced_review(
        archive_root=archive_root,
        split=str(args.split),
        selected_rows=selected,
        output_name=str(args.output_name),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
