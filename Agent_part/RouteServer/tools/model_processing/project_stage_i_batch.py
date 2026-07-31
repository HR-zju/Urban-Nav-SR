#!/usr/bin/env python3
"""Project canonical Reasoning/VQA artifacts into a Stage I exchange pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_i_batch import project_stage_i_batch_v1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical parents, provenance registry, and outputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reasoning-view-jsonl",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument("--vqa-source-jsonl", type=Path)
    parser.add_argument("--vqa-annotation-jsonl", type=Path)
    parser.add_argument(
        "--instruction-provenance-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-view-jsonl", required=True, type=Path)
    parser.add_argument("--output-drop-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument(
        "--modality-profile-id",
        action="append",
        dest="modality_profile_ids",
    )
    parser.add_argument("--allow-fpv-only-decision", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run deterministic fail-closed projection."""

    args = parse_args(argv)
    report = project_stage_i_batch_v1(
        output_view_jsonl=args.output_view_jsonl,
        output_drop_jsonl=args.output_drop_jsonl,
        instruction_provenance_jsonl=(
            args.instruction_provenance_jsonl
        ),
        reasoning_view_jsonls=args.reasoning_view_jsonl,
        vqa_source_jsonl=args.vqa_source_jsonl,
        vqa_annotation_jsonl=args.vqa_annotation_jsonl,
        modality_profile_ids=args.modality_profile_ids,
        allow_fpv_only_decision=args.allow_fpv_only_decision,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
