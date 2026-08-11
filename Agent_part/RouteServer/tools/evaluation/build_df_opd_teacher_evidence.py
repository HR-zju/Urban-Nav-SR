#!/usr/bin/env python3
"""Join private four-condition DF-OPD Teacher ledgers into evidence."""

from __future__ import annotations

import argparse
import json

from mcp_server.eval.df_opd_teacher_scoring import build_df_opd_teacher_evidence_v1


def main() -> int:
    """Parse local paths, write score/review sidecars, and print a receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl", required=True)
    parser.add_argument(
        "--inference-jsonl",
        required=True,
        action="append",
        help="private Teacher prediction/failure ledger; repeat for all four conditions",
    )
    parser.add_argument("--score-jsonl", required=True)
    parser.add_argument("--review-prediction-jsonl", required=True)
    parser.add_argument("--receipt-json", required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    receipt = build_df_opd_teacher_evidence_v1(
        canonical_jsonl=args.canonical_jsonl,
        inference_jsonls=args.inference_jsonl,
        score_jsonl=args.score_jsonl,
        review_prediction_jsonl=args.review_prediction_jsonl,
        receipt_json=args.receipt_json,
        dataset_fingerprint=args.dataset_fingerprint,
        checkpoint_digest=args.checkpoint_digest,
        seed=args.seed,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

