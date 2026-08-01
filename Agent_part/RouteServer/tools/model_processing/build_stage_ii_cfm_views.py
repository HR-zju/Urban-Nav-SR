#!/usr/bin/env python3
"""Build Stage II CFM exchange views from Stage I inputs and existing GT."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.stage_i_sft import parse_stage_i_training_view_v1
from mcp_server.contracts.stage_ii_cfm import StageIIContextCacheV1
from mcp_server.data.stage_ii_sources import (
    load_stage_ii_trajectory_source_registry_v1,
)
from mcp_server.data.stage_ii_trajectory import build_stage_ii_trajectory_target_v1
from mcp_server.data.stage_ii_views import (
    build_stage_ii_projection_drop_v1,
    project_stage_ii_cfm_view_v1,
)


REPORT_VERSION = "urbannav_stage_ii_cfm_view_build_report_v1"


def build_stage_ii_cfm_views(
    *,
    stage_i_jsonl: str | Path,
    trajectory_jsonl: str | Path,
    reasoning_cache_jsonl: str | Path,
    decision_cache_jsonl: str | Path,
    trajectory_source_registry_json: str | Path,
    output_view_jsonl: str | Path,
    output_qc_jsonl: str | Path,
    output_drop_jsonl: str | Path,
) -> dict[str, Any]:
    """Join exact identities and emit accepted views plus QC/drop audits.

    Args:
        stage_i_jsonl: Formal Stage I FullCoT exchange views.
        trajectory_jsonl: Existing timed dataset trajectory sidecars.
        reasoning_cache_jsonl: Accepted/rejected Pass-R cache artifact.
        decision_cache_jsonl: Accepted/rejected Pass-D cache artifact.
        trajectory_source_registry_json: Dataset-to-source allowlist.
        output_view_jsonl: Accepted Stage II exchange-view destination.
        output_qc_jsonl: Per-target trajectory QC destination.
        output_drop_jsonl: Fail-closed projection audit destination.

    Returns:
        Deterministic input/output counts and artifact paths.

    Raises:
        ValueError: If a global registry or unique-index invariant is invalid.
    """

    source_registry = load_stage_ii_trajectory_source_registry_v1(
        trajectory_source_registry_json
    )
    trajectories = _index_unique(_iter_jsonl(trajectory_jsonl), "stage_i_sample_id")
    reasoning = _index_cache(reasoning_cache_jsonl, "reasoning_grounding")
    decisions = _index_cache(decision_cache_jsonl, "decision")
    views: list[Mapping[str, Any]] = []
    qc_rows: list[Mapping[str, Any]] = []
    drops: list[Mapping[str, Any]] = []
    counts: Counter[str] = Counter()
    for raw_view in _iter_jsonl(stage_i_jsonl):
        counts["input_rows"] += 1
        typed = parse_stage_i_training_view_v1(raw_view)
        if typed.task_type != "full_cot":
            counts["skipped_non_full_cot"] += 1
            continue
        try:
            raw_target = trajectories[typed.sample_id]
            try:
                source_spec = source_registry.dataset(typed.dataset_source)
            except KeyError as exc:
                raise ValueError("trajectory_source_registry_missing")
            if source_spec.status != "enabled":
                raise ValueError("trajectory_dataset_disabled")
            source_format = str(raw_target.get("source_format") or "")
            yaw_source = str(raw_target.get("yaw_source") or "")
            if source_format != source_spec.source_format:
                raise ValueError("trajectory_source_format_registry_mismatch")
            if yaw_source != source_spec.yaw_source:
                raise ValueError("trajectory_yaw_source_registry_mismatch")
            build = build_stage_ii_trajectory_target_v1(
                sample_id=typed.sample_id,
                dataset_source=typed.dataset_source,
                raw_trajectory=raw_target["trajectory_ego_t"],
                action_mask=raw_target.get("action_mask"),
                source_format=source_format,
                trajectory_source=str(raw_target.get("trajectory_source") or ""),
                allowed_trajectory_sources=source_spec.allowed_trajectory_sources,
                yaw_source=yaw_source,
            )
            qc_rows.append(build.qc.model_dump(mode="json"))
            counts[f"trajectory_qc:{build.qc.status}"] += 1
            if build.target is None:
                raise ValueError(str(build.qc.reason_codes[0]))
            review_status = str(raw_target.get("review_status") or "")
            view = project_stage_ii_cfm_view_v1(
                typed,
                trajectory_target=build.target,
                reasoning_context=reasoning[(typed.sample_id, typed.modality_profile_id)],
                decision_context=decisions[(typed.sample_id, typed.modality_profile_id)],
                review_status=review_status,
            )
        except (KeyError, TypeError, ValueError) as exc:
            drop = build_stage_ii_projection_drop_v1(
                parent_sample_id=typed.sample_id,
                source_group_id=typed.source_group_id,
                split_group_id=typed.split_group_id,
                dataset_source=typed.dataset_source,
                modality_profile_id=typed.modality_profile_id,
                reason_code=_drop_reason_code(exc),
                detail=str(exc)[:1000] or type(exc).__name__,
            )
            drops.append(drop.model_dump(mode="json"))
            counts[f"drop:{drop.reason_code}"] += 1
            continue
        views.append(view.model_dump(mode="json", exclude_none=True))
        counts["output_views"] += 1
        counts[f"dataset:{view.dataset_source}"] += 1
        counts[f"valid_steps:{view.trajectory_target.valid_step_count}"] += 1
    _write_jsonl(output_view_jsonl, views)
    _write_jsonl(output_qc_jsonl, qc_rows)
    _write_jsonl(output_drop_jsonl, drops)
    return {
        "schema_version": REPORT_VERSION,
        "inputs": {
            "stage_i_jsonl": str(stage_i_jsonl),
            "trajectory_jsonl": str(trajectory_jsonl),
            "reasoning_cache_jsonl": str(reasoning_cache_jsonl),
            "decision_cache_jsonl": str(decision_cache_jsonl),
            "trajectory_source_registry_json": str(trajectory_source_registry_json),
        },
        "outputs": {
            "view_jsonl": str(output_view_jsonl),
            "qc_jsonl": str(output_qc_jsonl),
            "drop_jsonl": str(output_drop_jsonl),
        },
        "counts": dict(counts),
    }


def _index_unique(rows, key: str) -> dict[str, dict[str, Any]]:
    """Index JSON rows by one required unique identity."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity or identity in result:
            raise ValueError(f"missing or duplicate {key}: {identity!r}")
        result[identity] = row
    return result


def _index_cache(path: str | Path, kind: str):
    """Index accepted caches by exact sample and modality profile."""

    result = {}
    for row in _iter_jsonl(path):
        cache = StageIIContextCacheV1.model_validate(row)
        if cache.context_kind != kind or cache.parse_status != "accepted":
            continue
        key = (cache.sample_id, cache.modality_profile_id)
        if key in result:
            raise ValueError(f"duplicate accepted {kind} cache: {key}")
        result[key] = cache
    return result


def _drop_reason_code(error: Exception) -> str:
    """Map a bounded exception prefix to ReasonCode syntax."""

    import re

    prefix = str(error).split(":", 1)[0].strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", prefix).strip("_")
    return (value or "stage_ii_projection_failed")[:96]


def _iter_jsonl(path: str | Path):
    """Yield non-empty JSON objects."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: str | Path, rows) -> None:
    """Write deterministic compact JSONL."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-jsonl", required=True, type=Path)
    parser.add_argument("--trajectory-jsonl", required=True, type=Path)
    parser.add_argument("--reasoning-cache-jsonl", required=True, type=Path)
    parser.add_argument("--decision-cache-jsonl", required=True, type=Path)
    parser.add_argument("--trajectory-source-registry-json", required=True, type=Path)
    parser.add_argument("--output-view-jsonl", required=True, type=Path)
    parser.add_argument("--output-qc-jsonl", required=True, type=Path)
    parser.add_argument("--output-drop-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build exchange views and write the report."""

    args = parse_args(argv)
    report = build_stage_ii_cfm_views(
        stage_i_jsonl=args.stage_i_jsonl,
        trajectory_jsonl=args.trajectory_jsonl,
        reasoning_cache_jsonl=args.reasoning_cache_jsonl,
        decision_cache_jsonl=args.decision_cache_jsonl,
        trajectory_source_registry_json=args.trajectory_source_registry_json,
        output_view_jsonl=args.output_view_jsonl,
        output_qc_jsonl=args.output_qc_jsonl,
        output_drop_jsonl=args.output_drop_jsonl,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
