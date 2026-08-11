#!/usr/bin/env python3
"""Evaluate one complete Student-only DF-OPD inference ledger offline."""

from __future__ import annotations

import argparse
import json

from mcp_server.eval.df_opd_student import evaluate_df_opd_student_predictions_v1


def main() -> int:
    """Parse local ledger paths, write a metrics-only report, and print it."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl", required=True)
    parser.add_argument(
        "--inference-jsonl",
        required=True,
        action="append",
        help="Student prediction or failure ledger; repeat to include both",
    )
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    report = evaluate_df_opd_student_predictions_v1(
        canonical_jsonl=args.canonical_jsonl,
        inference_jsonls=args.inference_jsonl,
        report_json=args.report_json,
        dataset_fingerprint=args.dataset_fingerprint,
        checkpoint_digest=args.checkpoint_digest,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

