#!/usr/bin/env python3
"""Evaluate one complete Student-only DF-OPD inference ledger offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from mcp_server.data.df_opd.student_review import build_df_opd_student_review_bundle_v1
from mcp_server.eval.df_opd_student import evaluate_df_opd_student_predictions_v1
from mcp_server.data.report_manifest import (
    build_conventional_report_manifest,
    write_report_manifest,
)


def validate_planned_visualization_manifest(
    path: str | Path,
    *,
    run_id: str,
    run_root: str | Path,
    required_outputs: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    """Validate the planned manifest bound to this evaluation.

    Args:
        path: JSON manifest written before evaluation started.
        run_id: Stable run identifier expected in the manifest.
        run_root: Run-specific visualization root expected in the manifest.
        required_outputs: Optional artifact-id to path mapping that must be
            declared in ``planned_outputs``.

    Returns:
        The decoded planned manifest.

    Raises:
        ValueError: If the manifest schema, phase, kind, or identity is wrong.
    """

    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Visualization manifest must be a JSON object: {manifest_path}")
    if manifest.get("schema_version") != "research_visualization_manifest_v1":
        raise ValueError(f"Unsupported visualization manifest schema: {manifest_path}")
    if manifest.get("phase") != "planned" or manifest.get("run_kind") != "evaluation":
        raise ValueError(f"Visualization manifest is not a planned evaluation: {manifest_path}")
    if manifest.get("run_id") != run_id:
        raise ValueError(f"Visualization manifest run_id does not match: {manifest_path}")
    if manifest.get("run_root") != str(Path(run_root).resolve()):
        raise ValueError(f"Visualization manifest run_root does not match: {manifest_path}")
    if required_outputs is not None:
        planned_outputs = manifest.get("planned_outputs")
        if not isinstance(planned_outputs, list):
            raise ValueError(f"Visualization manifest planned_outputs must be a list: {manifest_path}")
        planned: dict[object, object] = {}
        for item in planned_outputs:
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or not isinstance(item.get("path"), str):
                raise ValueError(f"Visualization manifest has an invalid planned output: {manifest_path}")
            planned[item["id"]] = item["path"]
        for artifact_id, expected_path in required_outputs.items():
            expected = str(Path(expected_path).expanduser().resolve())
            if planned.get(artifact_id) != expected:
                raise ValueError(
                    f"Visualization manifest output {artifact_id!r} does not match: {manifest_path}"
                )
    return manifest


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
    parser.add_argument("--report-json")
    parser.add_argument(
        "--experiment-root",
        help="Convention-based output directory; derives scorecard, report, receipt and visualization paths.",
    )
    parser.add_argument(
        "--build-report-manifest",
        action="store_true",
        help="Create report_manifest.json from standard media directories; never overwrite an existing file.",
    )
    parser.add_argument(
        "--default-report-section",
        help="Explicit fallback section used with --build-report-manifest.",
    )
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--review-html")
    parser.add_argument("--review-manifest-json")
    parser.add_argument("--review-selection-jsonl")
    parser.add_argument("--media-root")
    parser.add_argument("--visualization-manifest")
    parser.add_argument("--visualization-index")
    parser.add_argument("--run-id")
    parser.add_argument("--run-root")
    args = parser.parse_args()
    if args.experiment_root is not None:
        experiment_root = Path(args.experiment_root).expanduser().resolve()
        if args.run_id is None:
            parser.error("--experiment-root requires --run-id")
        args.report_json = args.report_json or str(experiment_root / "scorecard.json")
        args.review_html = args.review_html or str(experiment_root / "student_review.html")
        args.review_manifest_json = args.review_manifest_json or str(experiment_root / "receipt.json")
        args.visualization_index = args.visualization_index or str(
            experiment_root / "visualization_index.json"
        )
        args.visualization_manifest = args.visualization_manifest or str(
            experiment_root / "visualization_manifest.json"
        )
        args.run_root = args.run_root or str(experiment_root)
    if args.report_json is None:
        parser.error("--report-json or --experiment-root is required")
    if args.visualization_manifest is not None:
        if args.run_id is None or args.run_root is None:
            parser.error("--visualization-manifest requires --run-id and --run-root")
        try:
            required_outputs = None
            if args.experiment_root is not None:
                required_outputs = {
                    "scorecard": args.report_json,
                    "student_review_html": args.review_html,
                    "receipt": args.review_manifest_json,
                    "visualization_index": args.visualization_index,
                }
            validate_planned_visualization_manifest(
                args.visualization_manifest,
                run_id=args.run_id,
                run_root=args.run_root,
                required_outputs=required_outputs,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    if args.build_report_manifest:
        if args.experiment_root is None:
            parser.error("--build-report-manifest requires --experiment-root")
        manifest_path = Path(args.experiment_root).expanduser().resolve() / "report_manifest.json"
        if manifest_path.exists():
            parser.error(f"report manifest already exists; refusing to overwrite: {manifest_path}")
        manifest = build_conventional_report_manifest(
            args.experiment_root, default_section=args.default_report_section
        )
        write_report_manifest(manifest_path, manifest)
    review_values = (
        args.review_html,
        args.review_manifest_json,
        args.visualization_index,
        args.run_id,
        args.run_root,
    )
    if any(value is not None for value in review_values) and not all(value is not None for value in review_values):
        parser.error(
            "review output requires --review-html, --review-manifest-json, "
            "--visualization-index, --run-id and --run-root together"
        )
    report = evaluate_df_opd_student_predictions_v1(
        canonical_jsonl=args.canonical_jsonl,
        inference_jsonls=args.inference_jsonl,
        report_json=args.report_json,
        dataset_fingerprint=args.dataset_fingerprint,
        checkpoint_digest=args.checkpoint_digest,
        seed=args.seed,
    )
    if args.review_html is not None:
        build_df_opd_student_review_bundle_v1(
            canonical_jsonl=args.canonical_jsonl,
            prediction_jsonls=args.inference_jsonl,
            output_html=args.review_html,
            output_manifest_json=args.review_manifest_json,
            review_selection_jsonl=args.review_selection_jsonl,
            media_root=args.media_root,
            visualization_index_json=args.visualization_index,
            scorecard_json=args.report_json,
            run_id=args.run_id,
            run_root=args.run_root,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

