#!/usr/bin/env python3
"""Materialize a planned history corpus and compile its Stage II training view.

This is an orchestration entry point. Sampling, media decoding, direction
projection, and Stage II schema validation remain in their existing modules.
The command writes one package per planned clip, one direction-projected
observation manifest per package, and one direct-action manifest per package.

The output is intentionally fail-fast. A partial package has no corpus
completion receipt, and an exception is allowed to identify the first bad
source instead of converting it into a silent drop.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Iterable


_ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _ROUTE_SERVER_ROOT.parents[2]
sys.path.insert(0, str(_ROUTE_SERVER_ROOT / "src"))
sys.path.insert(0, str(_ROUTE_SERVER_ROOT / "tools"))
sys.path.insert(0, str(_WORKSPACE_ROOT / "UrbanNav-Contracts" / "src"))
sys.path.insert(0, str(_WORKSPACE_ROOT / "ms-swift"))

from mcp_server.data.history_corpus import materialize_history_corpus_parallel
from mcp_server.data.jsonl_io import iter_jsonl
from model_processing.project_reasoning_v4_ego_directions import (
    project_reasoning_v4_ego_direction_manifest,
)
from swift.sidewalk_nav.data.stage_ii.stage_ii_manifest_v2 import (  # type: ignore[import-not-found]
    build_history_action_manifest,
)


DEFAULT_DIRECTION_PROFILE = "ego_degree10_12way_terminal_centroid_v2"
DEFAULT_QUERY_COUNT = 16


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit full-generation contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path,
                        help="Completed history plan-batch directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="New output directory for packages and manifests")
    parser.add_argument("--dataset-source", required=True,
                        choices=("egowalk", "sidewalk", "foxglove"))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--history-max-lag-ns", type=int,
                        help="Override the lag frozen in the plan; omitted means use the plan value")
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--target-profile-id", default=DEFAULT_DIRECTION_PROFILE)
    parser.add_argument("--archive", action="store_true",
                        help="Create one tar archive next to each completed package")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent clip workers within one dataset")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse a completed packages directory and continue manifest compilation")
    return parser.parse_args(argv)


def _write_json(path: Path, value: Any) -> None:
    """Write a small receipt after its producer has completed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _aggregate_jsonl(paths: Iterable[Path], destination: Path) -> int:
    """Concatenate completed per-clip JSONL manifests without changing rows."""

    count = 0
    with destination.open("x", encoding="utf-8") as output:
        for path in paths:
            for row in iter_jsonl(path):
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
                count += 1
    return count


def _compile_package(
    item: dict[str, Any],
    *,
    package_root: Path,
    dataset_source: str,
    target_profile_id: str,
    query_count: int,
    plan_rows: dict[str, dict[str, Any]],
    explicit_lag_ns: int | None,
    resume: bool,
) -> dict[str, Any]:
    """Compile one independent package; every path belongs to this clip."""

    package = package_root / item["package_directory"]
    observations = package / "manifests/observations.jsonl"
    direction = package / "manifests/observations_direction12.jsonl"
    direction_report = package / "manifests/observations_direction12_report.json"
    query_tag = f"q{query_count}"
    stage_ii = package / f"manifests/stage_ii_direct_action_{query_tag}.jsonl"
    stage_ii_drops = package / f"manifests/stage_ii_direct_action_{query_tag}_drops.jsonl"
    stage_ii_report = package / f"manifests/stage_ii_direct_action_{query_tag}_report.json"
    planned_lag_ns = int(plan_rows[item["package_directory"]]["max_lag_ns"])
    if explicit_lag_ns is not None and explicit_lag_ns != planned_lag_ns:
        raise ValueError(
            f"{dataset_source}/{item['package_directory']}: explicit lag "
            f"{explicit_lag_ns} disagrees with plan lag {planned_lag_ns}"
        )
    lag_ns = planned_lag_ns if explicit_lag_ns is None else explicit_lag_ns

    if stage_ii_report.is_file():
        if not stage_ii.is_file() or not direction.is_file() or not direction_report.is_file():
            raise ValueError(f"{package}: compilation receipt exists but output is incomplete")
        direction_result = json.loads(direction_report.read_text(encoding="utf-8"))
        stage_ii_result = json.loads(stage_ii_report.read_text(encoding="utf-8"))
    else:
        if resume:
            for partial in (stage_ii, stage_ii_drops):
                if partial.is_file():
                    partial.rename(Path(str(partial) + ".partial"))
        direction_result = project_reasoning_v4_ego_direction_manifest(
            observations,
            direction,
            report_json=direction_report,
            dataset_source=dataset_source,
            target_profile_id=target_profile_id,
            history_package=True,
            direction_from_sidecar=True,
        )
        stage_ii_result = build_history_action_manifest(
            package,
            stage_ii,
            query_count=query_count,
            max_lag_ns=lag_ns,
            output_drop_jsonl=stage_ii_drops,
            observation_jsonl=direction,
        )
        _write_json(stage_ii_report, stage_ii_result)
    return {
        "package_directory": item["package_directory"],
        "clip_id": item["clip_id"],
        "direction_path": direction,
        "stage_ii_path": stage_ii,
        "direction_report": direction_report,
        "stage_ii_report": stage_ii_report,
        "direction_counts": direction_result["counts"],
        "stage_ii_counts": stage_ii_result["counts"],
        "lag_ns": lag_ns,
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    """Run materialization, direction projection, and Stage II compilation."""

    plan = args.plan.resolve()
    output = args.output.resolve()
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg quality must be between 0 and 100")
    if args.history_max_lag_ns is not None and args.history_max_lag_ns < 0:
        raise ValueError("history max lag must be nonnegative")
    if args.query_count not in (4, 8, 16):
        raise ValueError("query count must be one of 4, 8, or 16")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    resume_existing = args.resume and output.exists()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    dataset_receipt = output / "generation_receipt.json"
    if resume_existing and dataset_receipt.is_file():
        return json.loads(dataset_receipt.read_text(encoding="utf-8"))

    plan_summary = json.loads((plan / "summary.json").read_text(encoding="utf-8"))
    if plan_summary.get("status") != "planned":
        raise ValueError(f"{plan}: expected a completed plan-batch summary")
    if plan_summary.get("dataset_source") != args.dataset_source:
        raise ValueError("plan dataset does not match the generation dataset")

    package_root = output / "packages"
    plan_rows = {
        row["plan_directory"]: row
        for row in iter_jsonl(plan / "clips.jsonl")
        if row.get("status") != "excluded"
    }
    materialized = materialize_history_corpus_parallel(
        plan,
        package_root,
        jpeg_quality=args.jpeg_quality,
        max_workers=args.workers,
        archive=args.archive,
        resume=resume_existing,
    )

    package_items = sorted(
        (item for item in iter_jsonl(package_root / "clips.jsonl")
         if item.get("status") == "materialized"),
        key=lambda item: item["package_directory"],
    )
    direction_paths: list[Path] = []
    stage_ii_paths: list[Path] = []
    clip_reports: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    lag_budgets: set[int] = set()
    compiled = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _compile_package,
                item,
                package_root=package_root,
                dataset_source=args.dataset_source,
                target_profile_id=args.target_profile_id,
                query_count=args.query_count,
                plan_rows=plan_rows,
                explicit_lag_ns=args.history_max_lag_ns,
                resume=args.resume,
            ): item["package_directory"]
            for item in package_items
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                compiled[result["package_directory"]] = result
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    for item in package_items:
        result = compiled[item["package_directory"]]
        direction_paths.append(result["direction_path"])
        stage_ii_paths.append(result["stage_ii_path"])
        lag_budgets.add(result["lag_ns"])
        direction_counts = result["direction_counts"]
        stage_ii_counts = result["stage_ii_counts"]
        counts.update({"direction_input_rows": direction_counts.get("input_rows", 0),
                       "direction_output_rows": direction_counts.get("output_rows", 0),
                       "stage_ii_input_rows": stage_ii_counts.get("input_rows", 0),
                       "stage_ii_compiled_rows": stage_ii_counts.get("compiled_rows", 0),
                       "stage_ii_missing_sidecar": stage_ii_counts.get("missing_linked_sidecar", 0)})
        clip_reports.append({
            "package_directory": result["package_directory"],
            "clip_id": result["clip_id"],
            "direction_report": str(result["direction_report"]),
            "stage_ii_report": str(result["stage_ii_report"]),
            "direction_counts": direction_counts,
            "stage_ii_counts": stage_ii_counts,
        })
    for item in iter_jsonl(package_root / "clips.jsonl"):
        if item.get("status") != "materialized":
            counts[f"excluded:{item['reason']}"] += 1

    aggregate_direction = output / (
        f"manifests/{args.dataset_source}_observations_direction12.jsonl"
    )
    aggregate_stage_ii = output / (
        f"manifests/{args.dataset_source}_stage_ii_direct_action_q{args.query_count}.jsonl"
    )
    aggregate_direction.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        for partial in (aggregate_direction, aggregate_stage_ii):
            if partial.is_file():
                partial.rename(Path(str(partial) + ".partial"))
    direction_rows = _aggregate_jsonl(direction_paths, aggregate_direction)
    stage_ii_rows = _aggregate_jsonl(stage_ii_paths, aggregate_stage_ii)
    result = {
        "status": "generated",
        "dataset_source": args.dataset_source,
        "plan": str(plan),
        "output": str(output),
        "history_profile": "history_4s_2hz_9f_v1",
        "jpeg_quality": args.jpeg_quality,
        "history_max_lag_ns": sorted(lag_budgets),
        "query_count": args.query_count,
        "direction_profile_id": args.target_profile_id,
        "materialized": materialized,
        "counts": dict(counts),
        "aggregate_direction_rows": direction_rows,
        "aggregate_stage_ii_rows": stage_ii_rows,
        "aggregate_direction_manifest": str(aggregate_direction),
        "aggregate_stage_ii_manifest": str(aggregate_stage_ii),
        "clips": clip_reports,
        "semantic_label_status": "preserved_source_fields; FullCoT/VQA regeneration is separate",
    }
    _write_json(output / "generation_receipt.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the explicit full-generation command."""

    result = generate(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
