"""Build private canonical DF-OPD data from frozen Stage-I/II source views."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.df_opd.source_compiler import build_df_opd_canonical_from_stage_views_v1


def main() -> None:
    """Parse paths and write one review-first canonical compilation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-i-full-cot-jsonl', required=True)
    parser.add_argument('--stage-ii-manifest-jsonl', required=True)
    parser.add_argument('--output-canonical-jsonl', required=True)
    parser.add_argument('--output-messages-jsonl', required=True)
    parser.add_argument('--cue-artifact-root', required=True)
    parser.add_argument('--report-json', required=True)
    parser.add_argument('--profile-id', default='df_opd_2s16_q16_v1')
    parser.add_argument('--max-rows', type=int, default=None)
    args = parser.parse_args()
    report = build_df_opd_canonical_from_stage_views_v1(
        stage_i_full_cot_jsonl=args.stage_i_full_cot_jsonl,
        stage_ii_manifest_jsonl=args.stage_ii_manifest_jsonl,
        output_canonical_jsonl=args.output_canonical_jsonl,
        output_messages_jsonl=args.output_messages_jsonl,
        cue_artifact_root=args.cue_artifact_root,
        report_json=args.report_json,
        profile_id=args.profile_id,
        max_rows=args.max_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
