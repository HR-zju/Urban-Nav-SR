#!/usr/bin/env python3
"""Build a digest-bound DF-OPD physical training manifest offline."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.df_opd.export import build_df_opd_trainable_manifest_v1


def main() -> int:
    """Parse local artifact paths, build one manifest, and print its receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl", required=True)
    parser.add_argument("--messages-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument(
        "--teacher-condition",
        required=True,
        choices=("same_input_t0", "correct_privilege_tc"),
        help="selected trainable Teacher view; TS and TO are evaluation-only",
    )
    parser.add_argument(
        "--cue-artifact-root",
        help="base directory for relative private Teacher cue paths; defaults to canonical JSONL parent",
    )
    args = parser.parse_args()
    report = build_df_opd_trainable_manifest_v1(
        canonical_jsonl=args.canonical_jsonl,
        messages_jsonl=args.messages_jsonl,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        teacher_condition=args.teacher_condition,
        cue_artifact_root=args.cue_artifact_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
