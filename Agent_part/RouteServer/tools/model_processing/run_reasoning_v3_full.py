#!/usr/bin/env python3
"""Prepare, label, and finalize a sharded minimal Reasoning V3.4 run."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from mcp_server.contracts.reasoning_dataset_v3 import (
    ReasoningAnnotationV3,
    ReasoningSourceV3,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_audit import audit_reasoning_v3_artifacts
from mcp_server.data.reasoning_v3_quality import (
    ReasoningV3QualityThresholds,
    evaluate_reasoning_v3_quality,
)
from mcp_server.data.reasoning_v3_source import build_reasoning_source_manifest_v3
from mcp_server.data.reasoning_v3_views import (
    DEFAULT_REASONING_V3_TRAINING_PROFILE,
    REASONING_V3_TRAINING_PROFILES,
    ReasoningViewBuildConfigV3,
    export_reasoning_training_views_v3,
)
from mcp_server.teacher.reasoning_v3_batch import (
    ReasoningV3BatchConfig,
    label_reasoning_source_manifest_v3,
)
from tools.model_processing.build_reasoning_v3_review_html import build_reasoning_v3_review_html


CANONICAL_DATASET_FAMILIES = frozenset({"sidewalk", "foxglove", "frodobots", "egowalk"})
PREPARE_SCHEMA_VERSION = "urbannav_reasoning_v3_4_full_prepare_v1"
WORKER_SCHEMA_VERSION = "urbannav_reasoning_v3_4_full_worker_done_v1"
MERGE_SCHEMA_VERSION = "urbannav_reasoning_v3_4_annotation_merge_report_v1"
FINAL_SCHEMA_VERSION = "urbannav_reasoning_v3_4_full_pipeline_report_v1"
_PROVIDER_SECRET_ENV_NAMES = (
    "ARK_API_KEY",
    "ARK_API_KEYS",
    "LLM_API_KEY",
    "LLM_API_KEYS",
    "OPENAI_API_KEY",
    "OPENAI_API_KEYS",
)


def _dataset_arg(value: str) -> tuple[str, Path]:
    """Parse one ``NAME=JSONL`` dataset argument.

    Args:
        value: CLI value containing a stable dataset name and input path.

    Returns:
        Dataset name and expanded input path.

    Raises:
        argparse.ArgumentTypeError: If the value is malformed.
    """

    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("dataset must be NAME=JSONL")
    return name.strip(), Path(path).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the full-run orchestration CLI.

    Args:
        argv: Optional argument vector for tests or programmatic use.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "finalize"))
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--dataset", action="append", type=_dataset_arg, default=[])
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--required-fpv-frames", type=int, default=5)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--skip-source-errors", action="store_true")
    parser.add_argument("--provider", choices=("ark", "openai"))
    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument(
        "--api-keys-file",
        type=Path,
        help=(
            "Optional newline or dotenv-style provider-key file. Keys are used "
            "only in worker memory and are never written to reports."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="low")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--ms-swift-root", type=Path)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--review-per-stratum", type=int, default=4)
    parser.add_argument(
        "--min-valid-annotation-fraction",
        type=float,
        default=0.75,
        help="Minimum accepted-target plus observation-only fraction.",
    )
    parser.add_argument(
        "--max-failed-annotation-fraction",
        type=float,
        default=0.20,
        help="Maximum rejected, error, or missing-annotation fraction.",
    )
    parser.add_argument(
        "--training-profile",
        choices=REASONING_V3_TRAINING_PROFILES,
        default=DEFAULT_REASONING_V3_TRAINING_PROFILE,
        help=(
            "V3.4 export policy: audit_only disables every loss; "
            "strong_action_only enables only eligible direction/System1 sidecars."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one V3.4 full-run stage.

    Args:
        argv: Optional argument vector for tests or programmatic use.
    """

    args = parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    _ensure_run_directories(run_root)
    if args.action == "prepare":
        _prepare(args, run_root)
        return
    if args.action == "worker":
        asyncio.run(_run_worker(args, run_root))
        return
    _finalize(args, run_root)


def _prepare(args: argparse.Namespace, run_root: Path) -> None:
    """Freeze normalized sources and split them into deterministic shards."""

    if not args.dataset:
        raise SystemExit("prepare requires at least one --dataset NAME=JSONL")
    shard_count = max(1, int(args.shard_count))
    required_fpv_frames = max(1, int(args.required_fpv_frames))
    source_entries: list[dict[str, Any]] = []
    families: set[str] = set()
    used_names: set[str] = set()
    for dataset_name, input_path in args.dataset:
        safe_name = _safe_name(dataset_name)
        if safe_name in used_names:
            raise ValueError(f"duplicate dataset name: {dataset_name}")
        used_names.add(safe_name)
        family = _dataset_family(dataset_name)
        families.add(family)
        source_path = run_root / "sources" / f"{safe_name}.jsonl"
        report_path = run_root / "reports" / f"source_{safe_name}.json"
        report = build_reasoning_source_manifest_v3(
            input_path,
            source_path,
            config=ReasoningSourceBuildConfig(
                dataset_source=family,
                max_fpv_frames=required_fpv_frames,
                media_root=args.media_root,
            ),
            skip_errors=args.skip_source_errors,
        )
        _write_json(report_path, report)
        rows = _assert_source_contract(source_path, expected_fpv_frames=required_fpv_frames)
        source_entries.append(
            {
                "name": safe_name,
                "family": family,
                "input_jsonl": str(input_path.resolve()),
                "source_jsonl": str(source_path),
                "rows": rows,
            }
        )
    if families != CANONICAL_DATASET_FAMILIES:
        raise RuntimeError(
            "V3.4 full run requires SideWalk, Foxglove, FrodoBots, and EgoWalk; "
            f"got {sorted(families)}"
        )

    merged_source = run_root / "sources" / "reasoning_v3_4_all.jsonl"
    write_jsonl_atomic(
        merged_source,
        (
            row
            for entry in source_entries
            for row in iter_jsonl(Path(entry["source_jsonl"]))
        ),
    )
    total_rows = _assert_unique_source_ids(merged_source)
    shard_paths = [run_root / "shards" / f"shard_{index:02d}.jsonl" for index in range(shard_count)]
    shard_counts = _partition_round_robin(merged_source, shard_paths)

    report = {
        "schema_version": PREPARE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "required_fpv_frames": required_fpv_frames,
        "media_root": str(args.media_root.resolve()) if args.media_root is not None else None,
        "shard_count": shard_count,
        "total_rows": total_rows,
        "dataset_families": sorted(families),
        "datasets": source_entries,
        "merged_source_jsonl": str(merged_source),
        "shards": [
            {"index": index, "path": str(path), "rows": rows}
            for index, (path, rows) in enumerate(zip(shard_paths, shard_counts, strict=True))
        ],
    }
    _write_json(run_root / "reports" / "prepare.json", report)
    _write_json(run_root / ".prepared_v3_4.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


async def _run_worker(args: argparse.Namespace, run_root: Path) -> None:
    """Label one prepared shard through an Ark or OpenAI-compatible endpoint."""

    prepared = _load_prepare_report(run_root)
    shard_count = int(prepared["shard_count"])
    if args.shard_index is None or not 0 <= int(args.shard_index) < shard_count:
        raise SystemExit(f"worker requires --shard-index in [0, {shard_count - 1}]")
    provider, api_base, model = _resolve_worker_endpoint(args)
    api_keys = _read_api_keys(args.api_keys_file, provider=provider)
    _configure_provider_endpoint(provider=provider, api_base=api_base, model=model)
    if provider == "ark" and not api_keys and not _provider_key_is_configured(provider):
        raise SystemExit(
            "Ark worker requires --api-keys-file, ARK_API_KEY, or LLM_API_KEY"
        )
    shard_index = int(args.shard_index)
    shard = run_root / "shards" / f"shard_{shard_index:02d}.jsonl"
    annotation = run_root / "annotations" / f"shard_{shard_index:02d}.jsonl"
    report = await label_reasoning_source_manifest_v3(
        shard,
        annotation,
        rejected_jsonl=run_root / "annotations" / f"shard_{shard_index:02d}.rejected.jsonl",
        report_json=run_root / "reports" / f"annotation_shard_{shard_index:02d}.json",
        config=ReasoningV3BatchConfig(
            max_retries=max(0, int(args.max_retries)),
            image_detail=args.image_detail,
            concurrency=max(1, int(args.concurrency)),
            retry_failed_on_resume=True,
            progress_interval=max(0, int(args.progress_interval)),
        ),
        api_keys=api_keys,
    )
    if report.get("status") != "done":
        raise RuntimeError(f"V3.4 worker did not finish: {report}")
    done = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "shard_index": shard_index,
        "provider": provider,
        "model": model,
        "report": report,
    }
    _write_json(run_root / f".worker_v3_4_{shard_index:02d}_done.json", done)
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


def _finalize(args: argparse.Namespace, run_root: Path) -> None:
    """Compact latest annotations, audit artifacts, export views, and build reviews."""

    prepared = _load_prepare_report(run_root)
    shard_count = int(prepared["shard_count"])
    missing_workers: list[int] = []
    worker_summaries: list[dict[str, Any]] = []
    for index in range(shard_count):
        marker_path = run_root / f".worker_v3_4_{index:02d}_done.json"
        if not marker_path.is_file():
            missing_workers.append(index)
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        report = marker.get("report") or {}
        if (
            marker.get("schema_version") != WORKER_SCHEMA_VERSION
            or marker.get("shard_index") != index
            or report.get("status") != "done"
        ):
            raise RuntimeError(f"invalid V3.4 worker marker: {marker_path}")
        worker_summaries.append(
            {
                "shard_index": index,
                "provider": marker.get("provider"),
                "model": marker.get("model"),
                "input_rows": report.get("input_rows"),
                "processed_rows": report.get("processed_rows"),
                "processed_accepted_rows": report.get("accepted_rows"),
                "processed_observation_only_rows": report.get("observation_only_rows"),
                "processed_rejected_rows": report.get("rejected_rows"),
                "processed_error_rows": report.get("error_rows"),
            }
        )
    if missing_workers:
        raise RuntimeError(f"V3.4 workers are incomplete: {missing_workers}")
    merged_source = Path(prepared["merged_source_jsonl"])
    shard_annotations = [
        run_root / "annotations" / f"shard_{index:02d}.jsonl"
        for index in range(shard_count)
    ]
    annotation_history = (
        run_root / "annotations" / "reasoning_v3_4_all.audit.jsonl"
    )
    latest_annotations = (
        run_root / "annotations" / "reasoning_v3_4_all.latest.jsonl"
    )
    merge_report = _merge_latest_annotations(
        merged_source,
        shard_annotations,
        history_output=annotation_history,
        latest_output=latest_annotations,
    )
    _write_json(run_root / "reports" / "annotation_merge.json", merge_report)

    thresholds = ReasoningV3QualityThresholds(
        min_acceptance_fraction=args.min_valid_annotation_fraction,
        max_reject_fraction=args.max_failed_annotation_fraction,
    )
    required_families = set(prepared["dataset_families"])
    quality = evaluate_reasoning_v3_quality(
        merged_source,
        latest_annotations,
        thresholds=thresholds,
        required_dataset_families=required_families,
    )
    _write_json(run_root / "reports" / "quality.json", quality)

    views = (
        run_root
        / "views"
        / f"reasoning_v3_4_{args.training_profile}.jsonl"
    )
    view_report = export_reasoning_training_views_v3(
        merged_source,
        latest_annotations,
        views,
        config=ReasoningViewBuildConfigV3(training_profile=args.training_profile),
        skip_errors=False,
    )
    _write_json(run_root / "reports" / "views.json", view_report)
    audit = audit_reasoning_v3_artifacts(
        merged_source,
        latest_annotations,
        views,
        required_dataset_families=required_families,
    )
    _write_json(run_root / "reports" / "acceptance_audit.json", audit)

    review_reports: dict[str, dict[str, Any]] = {}
    family_sources = _build_family_sources(prepared, run_root)
    for family, source_path in sorted(family_sources.items()):
        review = build_reasoning_v3_review_html(
            source_path,
            latest_annotations,
            run_root
            / "review"
            / f"{family}_reasoning_v3_4_review.complete.html",
            per_stratum=max(1, int(args.review_per_stratum)),
        )
        review_reports[family] = review
        _write_json(run_root / "reports" / f"review_{family}.json", review)
    combined_review = build_reasoning_v3_review_html(
        merged_source,
        latest_annotations,
        run_root / "review" / "reasoning_v3_4_review_all.complete.html",
        per_stratum=max(1, int(args.review_per_stratum)),
    )
    review_reports["all"] = combined_review
    _write_json(run_root / "reports" / "review_all.json", combined_review)

    ms_swift_report: dict[str, Any] | None = None
    if args.ms_swift_root is not None:
        ms_swift_output = (
            run_root
            / "ms_swift"
            / f"reasoning_v3_4_{args.training_profile}.jsonl"
        )
        ms_swift_report_path = run_root / "reports" / "ms_swift_manifest.json"
        view_rows = _count_jsonl_rows(views)
        if view_rows:
            _convert_ms_swift(
                python_bin=args.python_bin,
                ms_swift_root=args.ms_swift_root,
                views=views,
                output=ms_swift_output,
                report=ms_swift_report_path,
                training_profile=args.training_profile,
            )
            ms_swift_report = json.loads(
                ms_swift_report_path.read_text(encoding="utf-8")
            )
        else:
            write_jsonl_atomic(ms_swift_output, ())
            ms_swift_report = {
                "schema_version": "ms_swift_reasoning_v3_4_empty_export_report_v1",
                "status": "skipped_no_training_views",
                "training_profile": args.training_profile,
                "input_rows": 0,
                "output_rows": 0,
                "output_jsonl": str(ms_swift_output),
            }
            _write_json(ms_swift_report_path, ms_swift_report)

    missing_images = sum(int(report.get("missing_images", 0)) for report in review_reports.values())
    final = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "prepare_report": prepared,
        "worker_summaries": worker_summaries,
        "annotation_merge_report": merge_report,
        "quality_report": quality,
        "view_report": view_report,
        "acceptance_audit": audit,
        "review_reports": review_reports,
        "ms_swift_report": ms_swift_report,
        "missing_review_images": missing_images,
        "validation_scope": (
            "schema_evidence_provenance_availability_only;"
            "no_semantic_accuracy_calibration_projection_fusion_tracking_or_geometry_claim"
        ),
        "human_review_status": "pending",
        "passed": bool(quality.get("passed") and audit.get("passed") and missing_images == 0),
    }
    _write_json(run_root / "reports" / "final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    if not final["passed"]:
        raise SystemExit(2)


def _ensure_run_directories(run_root: Path) -> None:
    """Create only persistent run subdirectories used by V3.4."""

    for name in (
        "sources",
        "shards",
        "annotations",
        "views",
        "reports",
        "review",
        "ms_swift",
        "logs",
    ):
        (run_root / name).mkdir(parents=True, exist_ok=True)


def _load_prepare_report(run_root: Path) -> dict[str, Any]:
    """Load the immutable prepare marker required by workers and finalize."""

    path = run_root / ".prepared_v3_4.json"
    if not path.is_file():
        raise RuntimeError(f"V3.4 run is not prepared: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != PREPARE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported V3.4 prepare marker: {report.get('schema_version')}")
    return report


def _assert_source_contract(path: Path, *, expected_fpv_frames: int) -> int:
    """Validate exact multiframe input and return the row count."""

    invalid: list[tuple[str, int]] = []
    rows = 0
    for row in iter_jsonl(path):
        source = ReasoningSourceV3.model_validate(row)
        rows += 1
        model_input = source.model_input
        roles = model_input.image_roles
        actual = sum(str(role).startswith("fpv.") for role in roles)
        if actual != expected_fpv_frames:
            invalid.append((model_input.source_group_id, actual))
            if len(invalid) >= 20:
                break
    if invalid:
        raise RuntimeError(
            f"strict V3.4 FPV window QC failed path={path} "
            f"expected={expected_fpv_frames} examples={invalid}"
        )
    if rows == 0:
        raise RuntimeError(f"empty V3.4 source: {path}")
    return rows


def _assert_unique_source_ids(path: Path) -> int:
    """Reject empty or duplicated source ids and return total rows."""

    seen: set[str] = set()
    duplicates: list[tuple[int, str]] = []
    for index, row in enumerate(iter_jsonl(path), start=1):
        source_id = str((row.get("model_input") or {}).get("source_group_id") or "")
        if not source_id or source_id in seen:
            duplicates.append((index, source_id))
            if len(duplicates) >= 20:
                break
        seen.add(source_id)
    if duplicates:
        raise RuntimeError(f"duplicate V3.4 source ids: {duplicates}")
    if not seen:
        raise RuntimeError(f"empty merged V3.4 source: {path}")
    return len(seen)


def _partition_round_robin(source: Path, shard_paths: Sequence[Path]) -> list[int]:
    """Split one JSONL into stable balanced shards without retaining rows in RAM.

    Args:
        source: Merged V3.4 source JSONL.
        shard_paths: Ordered destination shard paths.

    Returns:
        Row count for each shard.

    Raises:
        ValueError: If no shard path is supplied.
        RuntimeError: If any destination shard is empty.
    """

    if not shard_paths:
        raise ValueError("at least one V3.4 shard is required")
    temporary_paths = [
        path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}") for path in shard_paths
    ]
    counts = [0 for _ in shard_paths]
    handles = []
    try:
        for path in temporary_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(path.open("w", encoding="utf-8"))
        # 轮询分片避免某一服务只看到单一数据集，并保证重跑时结果稳定。
        with source.open("r", encoding="utf-8") as input_handle:
            for row_index, line in enumerate(input_handle):
                shard_index = row_index % len(handles)
                handles[shard_index].write(line)
                counts[shard_index] += 1
    finally:
        for handle in handles:
            handle.close()
    if any(count == 0 for count in counts):
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise RuntimeError(f"empty V3.4 shard after partition: {counts}")
    for temporary, destination in zip(temporary_paths, shard_paths, strict=True):
        os.replace(temporary, destination)
    return counts


def _merge_latest_annotations(
    source_jsonl: Path,
    shard_annotations: Sequence[Path],
    *,
    history_output: Path,
    latest_output: Path,
) -> dict[str, Any]:
    """Merge append-only shard histories and compact one latest row per source.

    The history file preserves every provider attempt for audit. Quality,
    training-view export, HTML review, and ms-swift conversion consume only the
    compact latest file, so retries never inflate completion counts.
    """

    source_ids: set[str] = set()
    for row in iter_jsonl(source_jsonl):
        source = ReasoningSourceV3.model_validate(row)
        source_id = source.model_input.source_group_id
        if source_id in source_ids:
            raise RuntimeError(f"duplicate V3.4 source id during merge: {source_id}")
        source_ids.add(source_id)

    history_rows = 0

    def history_records() -> Iterable[dict[str, Any]]:
        nonlocal history_rows
        for path in shard_annotations:
            if not path.is_file():
                raise RuntimeError(f"missing V3.4 shard annotation file: {path}")
            for row in iter_jsonl(path):
                annotation = ReasoningAnnotationV3.model_validate(row)
                if annotation.source_group_id not in source_ids:
                    raise RuntimeError(
                        "V3.4 annotation references an unknown source id: "
                        f"{annotation.source_group_id}"
                    )
                history_rows += 1
                yield annotation.model_dump(mode="json", exclude_none=True)

    write_jsonl_atomic(history_output, history_records())

    latest_rows = 0
    latest_status_counts: dict[str, int] = {}
    source_shards: dict[str, int] = {}

    def latest_records() -> Iterable[dict[str, Any]]:
        nonlocal latest_rows
        for shard_index, path in enumerate(shard_annotations):
            latest_in_shard: dict[str, ReasoningAnnotationV3] = {}
            for row in iter_jsonl(path):
                annotation = ReasoningAnnotationV3.model_validate(row)
                latest_in_shard[annotation.source_group_id] = annotation
            for source_id, annotation in latest_in_shard.items():
                prior_shard = source_shards.get(source_id)
                if prior_shard is not None and prior_shard != shard_index:
                    raise RuntimeError(
                        "V3.4 source id appears in multiple annotation shards: "
                        f"{source_id}"
                    )
                source_shards[source_id] = shard_index
                latest_rows += 1
                latest_status_counts[annotation.status] = (
                    latest_status_counts.get(annotation.status, 0) + 1
                )
                yield annotation.model_dump(mode="json", exclude_none=True)

    write_jsonl_atomic(latest_output, latest_records())
    annotated_ids = set(source_shards)
    return {
        "schema_version": MERGE_SCHEMA_VERSION,
        "source_jsonl": str(source_jsonl),
        "history_jsonl": str(history_output),
        "latest_jsonl": str(latest_output),
        "source_rows": len(source_ids),
        "history_rows": history_rows,
        "latest_rows": latest_rows,
        "latest_status_counts": dict(sorted(latest_status_counts.items())),
        "missing_latest_rows": len(source_ids - annotated_ids),
        "orphan_latest_rows": len(annotated_ids - source_ids),
    }


def _count_jsonl_rows(path: Path) -> int:
    """Count validated non-empty JSONL rows without loading the file."""

    return sum(1 for _ in iter_jsonl(path))


def _build_family_sources(prepared: dict[str, Any], run_root: Path) -> dict[str, Path]:
    """Materialize one source join file per canonical dataset family."""

    result: dict[str, Path] = {}
    for family in sorted(CANONICAL_DATASET_FAMILIES):
        paths = [
            Path(entry["source_jsonl"])
            for entry in prepared["datasets"]
            if entry["family"] == family
        ]
        output = run_root / "sources" / f"{family}_all.jsonl"
        write_jsonl_atomic(output, (row for path in paths for row in iter_jsonl(path)))
        result[family] = output
    return result


def _convert_ms_swift(
    *,
    python_bin: Path,
    ms_swift_root: Path,
    views: Path,
    output: Path,
    report: Path,
    training_profile: str,
) -> None:
    """Run the version-gated ms-swift V3.4 adapter in a subprocess."""

    env = os.environ.copy()
    for name in _PROVIDER_SECRET_ENV_NAMES:
        env.pop(name, None)
    root = str(ms_swift_root.expanduser().resolve())
    env["PYTHONPATH"] = root + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    subprocess.run(
        [
            str(python_bin),
            "-m",
            "swift.sidewalk_nav.data.versioned_reasoning_manifest",
            "--input-jsonl",
            str(views),
            "--output-jsonl",
            str(output),
            "--report-json",
            str(report),
            "--expected-version",
            "v3",
            "--v3-training-profile",
            training_profile,
        ],
        env=env,
        check=True,
    )


def _resolve_worker_endpoint(args: argparse.Namespace) -> tuple[str, str, str]:
    """Resolve provider, base URL, and model without reading or reporting keys."""

    explicit = str(args.provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if not explicit:
        candidate_base = str(args.api_base or os.environ.get("LLM_API_BASE_URL") or "")
        explicit = (
            "ark"
            if os.environ.get("ARK_API_KEY") or "volces.com" in candidate_base
            else "openai"
        )
    if explicit not in {"ark", "openai"}:
        raise SystemExit(f"unsupported worker provider: {explicit}")

    api_base = str(
        args.api_base
        or os.environ.get("LLM_API_BASE_URL")
        or (
            os.environ.get("ARK_BASE_URL")
            or os.environ.get("ARK_API_BASE")
            if explicit == "ark"
            else os.environ.get("OPENAI_BASE_URL")
        )
        or ""
    ).strip()
    model = str(
        args.model
        or os.environ.get("LLM_MODEL")
        or os.environ.get("VLM_MODEL")
        or (os.environ.get("ARK_MODEL") if explicit == "ark" else "")
        or ""
    ).strip()
    if not api_base or not model:
        raise SystemExit(
            "worker requires --api-base and --model, or equivalent LLM environment settings"
        )
    return explicit, api_base.rstrip("/"), model


def _read_api_keys(path: Path | None, *, provider: str) -> tuple[str, ...]:
    """Read raw-line or dotenv-style provider keys without logging their values."""

    if path is None:
        return ()
    key_path = path.expanduser()
    try:
        lines = key_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read API key file: {key_path}") from exc

    accepted_names = {"LLM_API_KEY", "LLM_API_KEYS"}
    if provider == "ark":
        accepted_names.update({"ARK_API_KEY", "ARK_API_KEYS"})
    else:
        accepted_names.update({"OPENAI_API_KEY", "OPENAI_API_KEYS"})

    values: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        name, separator, assigned = line.partition("=")
        if separator and name.strip() in accepted_names:
            candidates = assigned.split(",") if name.strip().endswith("_KEYS") else [assigned]
        elif separator and name.strip().replace("_", "").isalnum() and name.strip()[:1].isalpha():
            # Ignore unrelated dotenv assignments, such as map-provider keys.
            continue
        else:
            candidates = [line]
        for candidate in candidates:
            value = candidate.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1].strip()
            if value and value not in values:
                values.append(value)
    if not values:
        raise RuntimeError(
            f"API key file contains no {provider} or generic LLM provider keys: {key_path}"
        )
    return tuple(values)


def _provider_key_is_configured(provider: str) -> bool:
    """Return whether the environment already contains a usable provider key."""

    if os.environ.get("LLM_API_KEY", "").strip():
        return True
    if provider == "ark":
        return bool(os.environ.get("ARK_API_KEY", "").strip())
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _configure_provider_endpoint(*, provider: str, api_base: str, model: str) -> None:
    """Configure the shared vision backend without serializing credentials."""

    if provider == "openai" and not os.environ.get("LLM_API_KEY", "").strip():
        os.environ["LLM_API_KEY"] = (
            os.environ.get("OPENAI_API_KEY", "").strip() or "dummy"
        )
    os.environ.update(
        {
            "LLM_PROVIDER": provider,
            "LLM_MODEL": model,
            "LLM_API_BASE_URL": api_base,
            "LLM_API_STYLE": "chat_completions",
            "LLM_TEMPERATURE": "0",
            "LLM_TIMEOUT_S": "120",
            "LLM_VISION_TIMEOUT_S": "300",
            "LLM_MAX_RETRIES": "1",
            "LLM_CHAT_TEMPLATE_ENABLE_THINKING": "0",
            "LLM_RESPONSE_FORMAT_JSON": "true",
        }
    )


def _dataset_family(value: str) -> str:
    """Map a concrete manifest name to one canonical V3.4 dataset family."""

    normalized = str(value).strip().lower().replace("_", "-")
    for family in ("sidewalk", "foxglove", "frodobots", "egowalk"):
        if normalized.startswith(family):
            return family
    raise ValueError(f"unsupported V3.4 dataset family: {value}")


def _safe_name(value: str) -> str:
    """Return a filesystem-safe stable dataset name."""

    result = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    if not result or not any(character.isalnum() for character in result):
        raise ValueError(f"invalid dataset name: {value!r}")
    return result


def _write_json(path: Path, value: Any) -> None:
    """Write one JSON report atomically for process-level readers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
