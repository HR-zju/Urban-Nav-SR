#!/usr/bin/env python3
"""Prepare, label, and finalize a sharded Reasoning V3.3 Full-CoT run."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_audit import audit_reasoning_v3_artifacts
from mcp_server.data.reasoning_v3_quality import (
    ReasoningV3QualityThresholds,
    evaluate_reasoning_v3_quality,
)
from mcp_server.data.reasoning_v3_source import build_reasoning_source_manifest_v3
from mcp_server.data.reasoning_v3_views import export_reasoning_training_views_v3
from mcp_server.teacher.reasoning_v3_batch import (
    ReasoningV3BatchConfig,
    label_reasoning_source_manifest_v3,
)
from tools.model_processing.build_reasoning_v3_review_html import build_reasoning_v3_review_html


CANONICAL_DATASET_FAMILIES = frozenset({"sidewalk", "foxglove", "frodobots", "egowalk"})
PREPARE_SCHEMA_VERSION = "urbannav_reasoning_v3_3_full_prepare_v1"
FINAL_SCHEMA_VERSION = "urbannav_reasoning_v3_3_full_pipeline_report_v1"


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
    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="low")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--ms-swift-root", type=Path)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--review-per-stratum", type=int, default=4)
    parser.add_argument("--min-acceptance-fraction", type=float, default=0.75)
    parser.add_argument("--max-reject-fraction", type=float, default=0.20)
    parser.add_argument("--max-conflict-fraction", type=float, default=0.10)
    parser.add_argument("--max-unknown-fraction", type=float, default=0.10)
    parser.add_argument("--max-insufficient-fraction", type=float, default=0.10)
    parser.add_argument("--max-task-unknown-fraction", type=float, default=0.20)
    parser.add_argument("--max-unsupported-rationale-fraction", type=float, default=0.20)
    parser.add_argument("--min-projection-qc-coverage", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one V3.3 full-run stage.

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
            "V3.3 full run requires SideWalk, Foxglove, FrodoBots, and EgoWalk; "
            f"got {sorted(families)}"
        )

    merged_source = run_root / "sources" / "reasoning_v3_3_all.jsonl"
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
    _write_json(run_root / ".prepared_v3_3.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


async def _run_worker(args: argparse.Namespace, run_root: Path) -> None:
    """Label one prepared shard through one OpenAI-compatible LVLM endpoint."""

    prepared = _load_prepare_report(run_root)
    shard_count = int(prepared["shard_count"])
    if args.shard_index is None or not 0 <= int(args.shard_index) < shard_count:
        raise SystemExit(f"worker requires --shard-index in [0, {shard_count - 1}]")
    if not args.api_base or not args.model:
        raise SystemExit("worker requires --api-base and --model")
    shard_index = int(args.shard_index)
    shard = run_root / "shards" / f"shard_{shard_index:02d}.jsonl"
    _configure_openai_endpoint(args.api_base, args.model)
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
    )
    if report.get("status") != "done":
        raise RuntimeError(f"V3.3 worker did not finish: {report}")
    done = {
        "schema_version": "urbannav_reasoning_v3_3_worker_done_v1",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "shard_index": shard_index,
        "api_base": args.api_base,
        "model": args.model,
        "report": report,
    }
    _write_json(run_root / f".worker_{shard_index:02d}_done.json", done)
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


def _finalize(args: argparse.Namespace, run_root: Path) -> None:
    """Merge workers, apply quality gates, export Full-CoT, and build reviews."""

    prepared = _load_prepare_report(run_root)
    shard_count = int(prepared["shard_count"])
    missing_workers = [
        index
        for index in range(shard_count)
        if not (run_root / f".worker_{index:02d}_done.json").is_file()
    ]
    if missing_workers:
        raise RuntimeError(f"V3.3 workers are incomplete: {missing_workers}")
    merged_source = Path(prepared["merged_source_jsonl"])
    merged_annotations = run_root / "annotations" / "reasoning_v3_3_all.jsonl"
    write_jsonl_atomic(
        merged_annotations,
        (
            row
            for index in range(shard_count)
            for row in iter_jsonl(run_root / "annotations" / f"shard_{index:02d}.jsonl")
        ),
    )

    thresholds = ReasoningV3QualityThresholds(
        min_acceptance_fraction=args.min_acceptance_fraction,
        max_reject_fraction=args.max_reject_fraction,
        max_conflict_fraction=args.max_conflict_fraction,
        max_unknown_fraction=args.max_unknown_fraction,
        max_insufficient_fraction=args.max_insufficient_fraction,
        max_task_unknown_fraction=args.max_task_unknown_fraction,
        max_unsupported_rationale_fraction=args.max_unsupported_rationale_fraction,
        min_projection_qc_coverage=args.min_projection_qc_coverage,
    )
    required_families = set(prepared["dataset_families"])
    quality = evaluate_reasoning_v3_quality(
        merged_source,
        merged_annotations,
        thresholds=thresholds,
        required_dataset_families=required_families,
    )
    _write_json(run_root / "reports" / "quality.json", quality)

    views = run_root / "views" / "reasoning_v3_3_full_cot.jsonl"
    view_report = export_reasoning_training_views_v3(
        merged_source,
        merged_annotations,
        views,
        skip_errors=True,
    )
    _write_json(run_root / "reports" / "views.json", view_report)
    audit = audit_reasoning_v3_artifacts(
        merged_source,
        merged_annotations,
        views,
        max_insufficient_fraction=args.max_insufficient_fraction,
        max_unsupported_rationale_fraction=args.max_unsupported_rationale_fraction,
        required_dataset_families=required_families,
    )
    _write_json(run_root / "reports" / "acceptance_audit.json", audit)

    review_reports: dict[str, dict[str, Any]] = {}
    family_sources = _build_family_sources(prepared, run_root)
    for family, source_path in sorted(family_sources.items()):
        review = build_reasoning_v3_review_html(
            source_path,
            merged_annotations,
            run_root / "review" / f"{family}_reasoning_v3_3_review.html",
            per_stratum=max(1, int(args.review_per_stratum)),
        )
        review_reports[family] = review
        _write_json(run_root / "reports" / f"review_{family}.json", review)
    combined_review = build_reasoning_v3_review_html(
        merged_source,
        merged_annotations,
        run_root / "review" / "reasoning_v3_3_review_all.html",
        per_stratum=max(1, int(args.review_per_stratum)),
    )
    review_reports["all"] = combined_review
    _write_json(run_root / "reports" / "review_all.json", combined_review)

    ms_swift_report: dict[str, Any] | None = None
    if args.ms_swift_root is not None:
        ms_swift_output = run_root / "ms_swift" / "reasoning_v3_3_full_cot.jsonl"
        ms_swift_report_path = run_root / "reports" / "ms_swift_manifest.json"
        _convert_ms_swift(
            python_bin=args.python_bin,
            ms_swift_root=args.ms_swift_root,
            views=views,
            output=ms_swift_output,
            report=ms_swift_report_path,
        )
        ms_swift_report = json.loads(ms_swift_report_path.read_text(encoding="utf-8"))

    missing_images = sum(int(report.get("missing_images", 0)) for report in review_reports.values())
    final = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "prepare_report": prepared,
        "quality_report": quality,
        "view_report": view_report,
        "acceptance_audit": audit,
        "review_reports": review_reports,
        "ms_swift_report": ms_swift_report,
        "missing_review_images": missing_images,
        "passed": bool(quality.get("passed") and audit.get("passed") and missing_images == 0),
    }
    _write_json(run_root / "reports" / "final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    if not final["passed"]:
        raise SystemExit(2)


def _ensure_run_directories(run_root: Path) -> None:
    """Create only persistent run subdirectories used by V3.3."""

    for name in ("sources", "shards", "annotations", "views", "reports", "review", "ms_swift", "logs"):
        (run_root / name).mkdir(parents=True, exist_ok=True)


def _load_prepare_report(run_root: Path) -> dict[str, Any]:
    """Load the immutable prepare marker required by workers and finalize."""

    path = run_root / ".prepared_v3_3.json"
    if not path.is_file():
        raise RuntimeError(f"V3.3 run is not prepared: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != PREPARE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported V3.3 prepare marker: {report.get('schema_version')}")
    return report


def _assert_source_contract(path: Path, *, expected_fpv_frames: int) -> int:
    """Validate exact multiframe input and return the row count."""

    invalid: list[tuple[str, int]] = []
    rows = 0
    for row in iter_jsonl(path):
        rows += 1
        model_input = row.get("model_input") or {}
        roles = model_input.get("image_roles") or []
        actual = sum(str(role).startswith("fpv.") for role in roles)
        if actual != expected_fpv_frames:
            invalid.append((str(model_input.get("source_group_id") or ""), actual))
            if len(invalid) >= 20:
                break
    if invalid:
        raise RuntimeError(
            f"strict V3.3 FPV window QC failed path={path} "
            f"expected={expected_fpv_frames} examples={invalid}"
        )
    if rows == 0:
        raise RuntimeError(f"empty V3.3 source: {path}")
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
        raise RuntimeError(f"duplicate V3.3 source ids: {duplicates}")
    if not seen:
        raise RuntimeError(f"empty merged V3.3 source: {path}")
    return len(seen)


def _partition_round_robin(source: Path, shard_paths: Sequence[Path]) -> list[int]:
    """Split one JSONL into stable balanced shards without retaining rows in RAM.

    Args:
        source: Merged V3.3 source JSONL.
        shard_paths: Ordered destination shard paths.

    Returns:
        Row count for each shard.

    Raises:
        ValueError: If no shard path is supplied.
        RuntimeError: If any destination shard is empty.
    """

    if not shard_paths:
        raise ValueError("at least one V3.3 shard is required")
    temporary_paths = [
        path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}") for path in shard_paths
    ]
    counts = [0 for _ in shard_paths]
    handles = []
    try:
        for path in temporary_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(path.open("w", encoding="utf-8"))
        # 轮询分片避免某一服务只看到单一数据集，同时保证重跑时分片结果稳定。
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
        raise RuntimeError(f"empty V3.3 shard after partition: {counts}")
    for temporary, destination in zip(temporary_paths, shard_paths, strict=True):
        os.replace(temporary, destination)
    return counts


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
) -> None:
    """Run the version-gated ms-swift V3.3 adapter in a subprocess."""

    env = os.environ.copy()
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
        ],
        env=env,
        check=True,
    )


def _configure_openai_endpoint(api_base: str, model: str) -> None:
    """Configure the existing vision backend for one local Qwen service."""

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
    """Map a concrete manifest name to one canonical dataset family."""

    normalized = str(value).strip().lower().replace("_", "-")
    for family in ("sidewalk", "foxglove", "frodobots", "egowalk"):
        if normalized.startswith(family):
            return family
    raise ValueError(f"unsupported V3.3 dataset family: {value}")


def _safe_name(value: str) -> str:
    """Return a filesystem-safe stable dataset name."""

    result = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
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
