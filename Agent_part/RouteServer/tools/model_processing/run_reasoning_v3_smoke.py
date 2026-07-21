#!/usr/bin/env python3
"""Run a four-dataset Reasoning V3 source/teacher/export/review smoke."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_audit import audit_reasoning_v3_artifacts
from mcp_server.data.reasoning_v3_quality import (
    ReasoningV3QualityThresholds,
    evaluate_reasoning_v3_quality,
)
from mcp_server.data.reasoning_v3_source import build_reasoning_source_manifest_v3
from mcp_server.data.reasoning_v3_views import export_reasoning_training_views_v3
from mcp_server.teacher.reasoning_v3_batch import ReasoningV3BatchConfig, label_reasoning_source_manifest_v3
from tools.model_processing.build_reasoning_v3_review_html import build_reasoning_v3_review_html


def _dataset_arg(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("dataset must be NAME=JSONL")
    return name.strip(), Path(path).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse end-to-end smoke CLI arguments.

    Args:
        argv: Optional argument vector for tests or programmatic use.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_dataset_arg, required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--limit-per-dataset", type=int, default=16)
    parser.add_argument("--required-fpv-frames", type=int, default=5)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--skip-source-errors", action="store_true")
    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="low")
    parser.add_argument("--min-acceptance-fraction", type=float, default=0.75)
    parser.add_argument("--max-reject-fraction", type=float, default=0.20)
    parser.add_argument("--max-conflict-fraction", type=float, default=0.10)
    parser.add_argument("--max-unknown-fraction", type=float, default=0.10)
    parser.add_argument("--max-insufficient-fraction", type=float, default=0.10)
    parser.add_argument("--max-task-unknown-fraction", type=float, default=0.20)
    parser.add_argument("--max-unsupported-rationale-fraction", type=float, default=0.20)
    parser.add_argument("--min-projection-qc-coverage", type=float, default=0.95)
    parser.add_argument("--allow-partial-datasets", action="store_true")
    parser.add_argument("--review-per-stratum", type=int, default=3)
    parser.add_argument("--ms-swift-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run source, annotation, export, audit, and review smoke stages.

    Args:
        argv: Optional argument vector for tests or programmatic use.
    """

    args = parse_args(argv)
    if bool(args.api_base) != bool(args.model):
        raise SystemExit("--api-base and --model must be provided together")
    run_root = args.run_root.expanduser().resolve()
    for name in ("sources", "annotations", "views", "reports", "review", "ms_swift"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    source_paths_by_family: dict[str, list[Path]] = {}
    source_reports: list[dict] = []
    required_families: set[str] = set()
    occurrence: Counter[str] = Counter()
    for dataset_name, input_path in args.dataset:
        family = _dataset_family(dataset_name)
        required_families.add(family)
        occurrence[family] += 1
        suffix = "" if occurrence[family] == 1 else f"_{occurrence[family]}"
        output = run_root / "sources" / f"{family}{suffix}.jsonl"
        report_path = run_root / "reports" / f"source_{family}{suffix}.json"
        report = build_reasoning_source_manifest_v3(
            input_path,
            output,
            config=ReasoningSourceBuildConfig(
                dataset_source=family,
                max_fpv_frames=max(1, args.required_fpv_frames),
                media_root=args.media_root,
            ),
            limit=max(1, args.limit_per_dataset),
            skip_errors=args.skip_source_errors,
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _assert_fpv_count(output, expected=max(1, args.required_fpv_frames))
        source_paths_by_family.setdefault(family, []).append(output)
        source_reports.append(report)
    canonical_families = {"sidewalk", "foxglove", "frodobots", "egowalk"}
    if not args.allow_partial_datasets and required_families != canonical_families:
        raise SystemExit(
            "V3.3 smoke requires SideWalk, Foxglove, FrodoBots, and EgoWalk; "
            f"got {sorted(required_families)}"
        )

    # 每个 family 保留独立 source，保证四份 HTML 和 QC 报告使用同一批样本。
    family_source_paths: dict[str, Path] = {}
    for family, paths in sorted(source_paths_by_family.items()):
        family_source = run_root / "sources" / f"{family}_smoke_all.jsonl"
        write_jsonl_atomic(family_source, (row for path in paths for row in iter_jsonl(path)))
        family_source_paths[family] = family_source
    merged_source = run_root / "sources" / "reasoning_v3_smoke_all.jsonl"
    write_jsonl_atomic(
        merged_source,
        (row for path in family_source_paths.values() for row in iter_jsonl(path)),
    )
    source_summary = _source_summary(merged_source)
    source_summary.update(
        {
            "schema_version": "urbannav_reasoning_v3_3_source_smoke_report_v1",
            "datasets": [{"name": name, "path": str(path)} for name, path in args.dataset],
            "source_reports": source_reports,
        }
    )
    (run_root / "reports/source_smoke_summary.json").write_text(
        json.dumps(source_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(source_summary, ensure_ascii=False, indent=2), flush=True)
    if not args.api_base:
        return

    _configure_openai_endpoint(args.api_base, args.model)
    annotations = run_root / "annotations/reasoning_v3_smoke.jsonl"
    annotation_report = asyncio.run(
        label_reasoning_source_manifest_v3(
            merged_source,
            annotations,
            rejected_jsonl=run_root / "annotations/reasoning_v3_smoke.rejected.jsonl",
            report_json=run_root / "reports/annotation.json",
            config=ReasoningV3BatchConfig(
                max_retries=max(0, args.max_retries),
                image_detail=args.image_detail,
                concurrency=max(1, args.concurrency),
                retry_failed_on_resume=True,
                progress_interval=10,
            ),
        )
    )
    quality_thresholds = ReasoningV3QualityThresholds(
        min_acceptance_fraction=args.min_acceptance_fraction,
        max_reject_fraction=args.max_reject_fraction,
        max_conflict_fraction=args.max_conflict_fraction,
        max_unknown_fraction=args.max_unknown_fraction,
        max_insufficient_fraction=args.max_insufficient_fraction,
        max_task_unknown_fraction=args.max_task_unknown_fraction,
        max_unsupported_rationale_fraction=args.max_unsupported_rationale_fraction,
        min_projection_qc_coverage=args.min_projection_qc_coverage,
    )
    quality_report = evaluate_reasoning_v3_quality(
        merged_source,
        annotations,
        thresholds=quality_thresholds,
        required_dataset_families=required_families,
    )
    (run_root / "reports/quality.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    views = run_root / "views/reasoning_v3_smoke.jsonl"
    export_report = export_reasoning_training_views_v3(
        merged_source,
        annotations,
        views,
        skip_errors=True,
    )
    (run_root / "reports/views.json").write_text(
        json.dumps(export_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = audit_reasoning_v3_artifacts(
        merged_source,
        annotations,
        views,
        max_insufficient_fraction=args.max_insufficient_fraction,
        max_unsupported_rationale_fraction=args.max_unsupported_rationale_fraction,
        required_dataset_families=required_families,
    )
    (run_root / "reports/acceptance_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    review_reports: dict[str, dict] = {}
    for family, family_source in sorted(family_source_paths.items()):
        review = build_reasoning_v3_review_html(
            family_source,
            annotations,
            run_root / "review" / f"{family}_reasoning_v3_3_smoke_review.html",
            per_stratum=max(1, args.review_per_stratum),
        )
        review_reports[family] = review
        (run_root / "reports" / f"review_{family}.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    combined_review = build_reasoning_v3_review_html(
        merged_source,
        annotations,
        run_root / "review" / "reasoning_v3_3_smoke_review_all.html",
        per_stratum=max(1, args.review_per_stratum),
    )
    review_reports["all"] = combined_review
    (run_root / "reports/review_all.json").write_text(
        json.dumps(combined_review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.ms_swift_root is not None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.ms_swift_root.resolve()) + (
            f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "swift.sidewalk_nav.data.versioned_reasoning_manifest",
                "--input-jsonl",
                str(views),
                "--output-jsonl",
                str(run_root / "ms_swift/reasoning_v3_smoke.jsonl"),
                "--report-json",
                str(run_root / "reports/ms_swift_manifest.json"),
                "--expected-version",
                "v3",
            ],
            env=env,
            check=True,
        )
    final = {
        "schema_version": "urbannav_reasoning_v3_3_smoke_pipeline_report_v1",
        "run_root": str(run_root),
        "source_summary": source_summary,
        "annotation_report": annotation_report,
        "view_report": export_report,
        "quality_report": quality_report,
        "acceptance_audit": audit,
        "review_reports": review_reports,
        "passed": bool(quality_report["passed"] and audit["passed"]),
    }
    (run_root / "reports/final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    if not final["passed"]:
        raise SystemExit(2)


def _assert_fpv_count(path: Path, *, expected: int) -> None:
    invalid = []
    for row in iter_jsonl(path):
        roles = row["model_input"].get("image_roles") or []
        count = sum(str(role).startswith("fpv.") for role in roles)
        if count != expected:
            invalid.append((row["model_input"].get("source_group_id"), count))
            if len(invalid) >= 20:
                break
    if invalid:
        raise RuntimeError(f"V3 smoke requires exactly {expected} FPV frames: {invalid}")


def _source_summary(path: Path) -> dict:
    counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        model_input = row["model_input"]
        family = _dataset_family(model_input.get("dataset_source") or "")
        counts["rows"] += 1
        counts[f"dataset:{family}"] += 1
        system1 = bool(model_input.get("system1_enabled"))
        counts[f"dataset:{family}:system1_rows"] += int(system1)
        provenance = row.get("label_provenance", {}).get("demonstrated_direction", {})
        if provenance.get("strength") == "strong":
            counts[f"dataset:{family}:strong_direction_rows"] += 1
            if system1 and family in {"sidewalk", "foxglove", "egowalk"}:
                counts["trajectory_qc_strong_direction_rows"] += 1
        if system1 and family in {"sidewalk", "foxglove", "egowalk"}:
            counts["trajectory_qc_eligible_rows"] += 1
    eligible = counts["trajectory_qc_eligible_rows"]
    coverage = counts["trajectory_qc_strong_direction_rows"] / eligible if eligible else 1.0
    frodo_system1 = counts["dataset:frodobots:system1_rows"]
    if coverage != 1.0:
        raise RuntimeError(f"strong direction coverage is not 100%: {coverage:.6f}")
    if frodo_system1 != 0:
        raise RuntimeError(f"FrodoBots System1 rows must be zero, got {frodo_system1}")
    return {
        "counts": dict(sorted(counts.items())),
        "trajectory_qc_strong_direction_coverage": coverage,
        "frodobots_system1_rows": frodo_system1,
    }


def _configure_openai_endpoint(api_base: str, model: str) -> None:
    os.environ.update(
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": model,
            "LLM_API_BASE_URL": api_base,
            "LLM_API_STYLE": "chat_completions",
            "LLM_API_KEY": os.environ.get("LLM_API_KEY") or "dummy",
            "LLM_TEMPERATURE": "0",
            "LLM_TIMEOUT_S": "120",
            "LLM_VISION_TIMEOUT_S": "300",
            "LLM_MAX_RETRIES": "1",
            "LLM_CHAT_TEMPLATE_ENABLE_THINKING": "0",
            "NO_PROXY": "127.0.0.1,localhost," + os.environ.get("NO_PROXY", ""),
            "no_proxy": "127.0.0.1,localhost," + os.environ.get("no_proxy", ""),
        }
    )


def _dataset_family(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    for family in ("sidewalk", "foxglove", "frodobots", "egowalk"):
        if normalized.startswith(family):
            return family
    raise ValueError(f"unsupported V3 smoke dataset: {value}")


if __name__ == "__main__":
    main()
