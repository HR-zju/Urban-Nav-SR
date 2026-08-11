#!/usr/bin/env python3
"""Build a DF-OPD Student manifest with V2 FullCoT replay lineage."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.df_opd.export import build_df_opd_student_manifest_v1


def main() -> int:
    """Parse local paths, emit the combined Student manifest, and print its receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl", required=True)
    parser.add_argument("--messages-jsonl", required=True)
    parser.add_argument(
        "--stage-i-full-cot-jsonl",
        required=True,
        help="filtered Stage-I V2 full_cot views; one strict public join per DF-OPD sample",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument(
        "--cue-artifact-root",
        help="base directory for relative private TC cue paths; defaults to canonical JSONL parent",
    )
    args = parser.parse_args()
    report = build_df_opd_student_manifest_v1(
        canonical_jsonl=args.canonical_jsonl,
        messages_jsonl=args.messages_jsonl,
        stage_i_full_cot_jsonl=args.stage_i_full_cot_jsonl,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        cue_artifact_root=args.cue_artifact_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
