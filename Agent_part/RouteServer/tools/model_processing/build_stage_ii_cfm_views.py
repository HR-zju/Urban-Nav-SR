#!/usr/bin/env python3
"""Build Stage II CFM exchange views from Stage I inputs and existing GT."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV1,
    StageIFullCoTViewV2,
    parse_stage_i_training_view_v1,
    parse_stage_i_training_view_v2,
)
from mcp_server.contracts.stage_ii_cfm import (
    StageIIContextCacheV1,
    StageIITrajectorySidecarV2,
    StageIITrajectoryTargetV1,
    StageIITrajectoryTargetV2,
    STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION,
)
from mcp_server.data.stage_ii.sources import load_stage_ii_trajectory_source_registry_v1
from mcp_server.data.openloop.benchmark import build_stage_ii_openloop_reference_from_parent_v1
from mcp_server.data.stage_ii.trajectory import build_stage_ii_trajectory_target_v1
from mcp_server.data.stage_ii.views import build_stage_ii_projection_drop_v1, project_stage_ii_cfm_view_v1, project_stage_ii_cfm_view_v1_from_stage_i_v2, project_stage_ii_cfm_view_v2, project_stage_ii_direct_latent_view_from_stage_i_v2


REPORT_VERSION = "urbannav_stage_ii_cfm_view_build_report_v1"


def build_stage_ii_cfm_views(
    *,
    stage_i_jsonl: str | Path,
    trajectory_jsonl: str | Path,
    reasoning_cache_jsonl: str | Path | None,
    decision_cache_jsonl: str | Path | None,
    trajectory_source_registry_json: str | Path,
    output_view_jsonl: str | Path,
    output_qc_jsonl: str | Path,
    output_drop_jsonl: str | Path,
    output_reference_jsonl: str | Path | None = None,
    output_eligible_jsonl: str | Path | None = None,
    denominator_only: bool = False,
    direct_latent: bool = False,
    max_source_step_m: float | None = None,
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
        output_reference_jsonl: Optional full eligible hidden references.
        output_eligible_jsonl: Optional public eligible-case identities.
        denominator_only: Build target-eligible references and identities
            without reading or filtering on Pass-R/Pass-D context caches.
        direct_latent: Build current-V4 cacheless direct-latent training views.
            This route accepts only V2 FullCoT parents and rejects supplied
            Pass-R/Pass-D artifacts, so the exchange contract cannot silently
            regress into a predicted-context run.
        max_source_step_m: Optional deterministic adjacent-source-pose bound
            applied before trajectory resampling.  A bounded admission profile
            must pass this explicitly; ``None`` preserves the historical V1
            construction behavior.

    Returns:
        Deterministic input/output counts and artifact paths.

    Raises:
        ValueError: If a global registry or unique-index invariant is invalid.
    """

    if denominator_only and direct_latent:
        raise ValueError("denominator_only and direct_latent are mutually exclusive")
    if direct_latent and (
        reasoning_cache_jsonl is not None or decision_cache_jsonl is not None
    ):
        raise ValueError(
            "direct_latent must not receive Pass-R/Pass-D cache artifacts"
        )

    source_registry = load_stage_ii_trajectory_source_registry_v1(
        trajectory_source_registry_json
    )
    trajectories = _index_trajectory_inputs(trajectory_jsonl)
    if denominator_only or direct_latent:
        reasoning = {}
        decisions = {}
    else:
        if reasoning_cache_jsonl is None or decision_cache_jsonl is None:
            raise ValueError(
                "reasoning and decision caches are required unless denominator_only is enabled"
            )
        reasoning = _index_cache(reasoning_cache_jsonl, "reasoning_grounding")
        decisions = _index_cache(decision_cache_jsonl, "decision")
    views: list[Mapping[str, Any]] = []
    qc_rows: list[Mapping[str, Any]] = []
    drops: list[Mapping[str, Any]] = []
    references: list[Mapping[str, Any]] = []
    eligible_rows: list[Mapping[str, Any]] = []
    counts: Counter[str] = Counter()
    for raw_view in _iter_jsonl(stage_i_jsonl):
        counts["input_rows"] += 1
        typed = _parse_full_cot(raw_view)
        if typed is None:
            counts["skipped_non_full_cot"] += 1
            continue
        try:
            raw_target = _trajectory_input_for_view(trajectories, typed)
            try:
                source_spec = source_registry.dataset(typed.dataset_source)
            except KeyError as exc:
                raise ValueError("trajectory_source_registry_missing")
            if source_spec.status != "enabled":
                raise ValueError("trajectory_dataset_disabled")
            target, qc, review_status, is_v2_sidecar, resolved_split_group_id = _resolve_trajectory_input(
                typed,
                raw_target,
                source_spec=source_spec,
                max_source_step_m=max_source_step_m,
            )
            qc_rows.append(qc)
            counts[f"trajectory_qc:{qc['status']}"] += 1
            reference = build_stage_ii_openloop_reference_from_parent_v1(
                typed,
                target,
            )
            references.append(
                reference.model_dump(mode="json", exclude_none=True)
            )
            eligible_rows.append(
                {
                    "sample_id": reference.sample_id,
                    "parent_sample_id": reference.parent_sample_id,
                    "parent_stage_i_sample_id": typed.sample_id,
                    "source_group_id": typed.source_group_id,
                    "split_group_id": typed.split_group_id,
                    "dataset_source": typed.dataset_source,
                    "modality_profile_id": typed.modality_profile_id,
                }
            )
            counts["eligible_references"] += 1
            if denominator_only:
                counts["denominator_only_rows"] += 1
                continue
            if direct_latent:
                if is_v2_sidecar:
                    raise ValueError(
                        "direct_latent does not support Stage II V2 sidecars"
                    )
                if not isinstance(typed, StageIFullCoTViewV2):
                    raise ValueError("direct_latent_requires_stage_i_full_cot_v2")
                view = project_stage_ii_direct_latent_view_from_stage_i_v2(
                    typed,
                    trajectory_target=target,
                    review_status=review_status,
                )
                counts["direct_latent_rows"] += 1
            else:
                cache_key = (typed.sample_id, typed.modality_profile_id)
                if cache_key not in reasoning:
                    raise ValueError("reasoning_context_cache_missing")
                if cache_key not in decisions:
                    raise ValueError("decision_context_cache_missing")
                if is_v2_sidecar:
                    if not isinstance(typed, StageIFullCoTViewV2):
                        raise ValueError("Stage II V2 sidecar requires V2 FullCoT")
                    view = project_stage_ii_cfm_view_v2(
                        typed,
                        trajectory_target=target,
                        reasoning_context=reasoning[cache_key],
                        decision_context=decisions[cache_key],
                        review_status=review_status,
                        split_group_id=resolved_split_group_id,
                    )
                else:
                    projector = (
                        project_stage_ii_cfm_view_v1
                        if isinstance(typed, StageIFullCoTViewV1)
                        else project_stage_ii_cfm_view_v1_from_stage_i_v2
                    )
                    view = projector(
                        typed,
                        trajectory_target=target,
                        reasoning_context=reasoning[cache_key],
                        decision_context=decisions[cache_key],
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
    if output_reference_jsonl is not None:
        _write_jsonl(output_reference_jsonl, references)
    if output_eligible_jsonl is not None:
        _write_jsonl(output_eligible_jsonl, eligible_rows)
    return {
        "schema_version": REPORT_VERSION,
        "inputs": {
            "stage_i_jsonl": str(stage_i_jsonl),
            "trajectory_jsonl": str(trajectory_jsonl),
            "reasoning_cache_jsonl": (
                str(reasoning_cache_jsonl)
                if reasoning_cache_jsonl is not None
                else None
            ),
            "decision_cache_jsonl": (
                str(decision_cache_jsonl)
                if decision_cache_jsonl is not None
                else None
            ),
            "trajectory_source_registry_json": str(trajectory_source_registry_json),
            "direct_latent": direct_latent,
            "max_source_step_m": max_source_step_m,
        },
        "outputs": {
            "view_jsonl": str(output_view_jsonl),
            "qc_jsonl": str(output_qc_jsonl),
            "drop_jsonl": str(output_drop_jsonl),
            "reference_jsonl": (
                str(output_reference_jsonl)
                if output_reference_jsonl is not None
                else None
            ),
            "eligible_jsonl": (
                str(output_eligible_jsonl)
                if output_eligible_jsonl is not None
                else None
            ),
        },
        "mode": (
            "denominator_only"
            if denominator_only
            else "direct_latent"
            if direct_latent
            else "complete_view"
        ),
        "join_contract": {
            "split_group_key": (
                "dataset_qualified(stage_i.split_group_id) == "
                "dataset_qualified(trajectory.split_group_id)"
            ),
            "split_identity_adapter_id": "dataset_qualified_split_group_v1",
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


def _index_trajectory_inputs(path: str | Path) -> dict[str, dict[str, Any]]:
    """Index legacy targets or V2 sidecars without changing legacy keys."""

    result: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        if row.get("schema_version") == STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION:
            sidecar = StageIITrajectorySidecarV2.model_validate(row)
            key = sidecar.source_group_id
        else:
            key = str(row.get("stage_i_sample_id") or "")
            if not key:
                raise ValueError("trajectory input lacks stage_i_sample_id or V2 sidecar schema")
        if key in result:
            raise ValueError(f"duplicate trajectory input identity: {key}")
        result[key] = row
    return result


def _trajectory_input_for_view(
    trajectories: Mapping[str, Mapping[str, Any]],
    stage_i_view: StageIFullCoTViewV1 | StageIFullCoTViewV2,
) -> Mapping[str, Any]:
    """Resolve old exact identity first, then the V2 source-group identity."""

    exact = trajectories.get(stage_i_view.sample_id)
    by_source = trajectories.get(stage_i_view.source_group_id)
    if exact is not None and by_source is not None and exact != by_source:
        raise ValueError("conflicting trajectory input identities for Stage I parent")
    if exact is not None:
        return exact
    if by_source is not None:
        return by_source
    raise KeyError(stage_i_view.sample_id)


def _resolve_trajectory_input(
    stage_i_view: StageIFullCoTViewV1 | StageIFullCoTViewV2,
    raw_target: Mapping[str, Any],
    *,
    source_spec: Any,
    max_source_step_m: float | None,
) -> tuple[
    StageIITrajectoryTargetV1 | StageIITrajectoryTargetV2,
    dict[str, Any],
    str,
    bool,
    str,
]:
    """Validate either a legacy target or a canonical V2 sidecar."""

    if raw_target.get("schema_version") == STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION:
        sidecar = StageIITrajectorySidecarV2.model_validate(raw_target)
        if (
            sidecar.source_group_id != stage_i_view.source_group_id
            or sidecar.dataset_source != stage_i_view.dataset_source
        ):
            raise ValueError("Stage II V2 sidecar/source identity mismatch")
        if _dataset_qualified_split_group(
            dataset_source=sidecar.dataset_source,
            split_group_id=sidecar.split_group_id,
        ) != _dataset_qualified_split_group(
            dataset_source=stage_i_view.dataset_source,
            split_group_id=stage_i_view.split_group_id,
        ):
            raise ValueError("Stage II V2 sidecar/source identity mismatch")
        target = sidecar.trajectory_target
        # V2 sidecars carry their own timed-pose policy.  The registry fields
        # used by the legacy V1 target builder describe a different raw-row
        # contract and must not silently reinterpret this target.
        # Keep the Stage-I split in the emitted view so existing Pass-R/Pass-D
        # caches remain bound to the exact observation identity.  The explicit
        # dataset-qualified adapter is only a join comparison rule.
        return (
            target,
            sidecar.trajectory_qc.model_dump(mode="json"),
            "not_required",
            True,
            stage_i_view.split_group_id,
        )

    source_format = str(raw_target.get("source_format") or "")
    yaw_source = str(raw_target.get("yaw_source") or "")
    if source_format != source_spec.source_format:
        raise ValueError("trajectory_source_format_registry_mismatch")
    if yaw_source != source_spec.yaw_source:
        raise ValueError("trajectory_yaw_source_registry_mismatch")
    build = build_stage_ii_trajectory_target_v1(
        sample_id=stage_i_view.sample_id,
        dataset_source=stage_i_view.dataset_source,
        raw_trajectory=raw_target["trajectory_ego_t"],
        action_mask=raw_target.get("action_mask"),
        source_format=source_format,
        trajectory_source=str(raw_target.get("trajectory_source") or ""),
        allowed_trajectory_sources=source_spec.allowed_trajectory_sources,
        yaw_source=yaw_source,
        max_source_step_m=max_source_step_m,
    )
    if build.target is None:
        raise ValueError(str(build.qc.reason_codes[0]))
    review_status = str(raw_target.get("review_status") or "")
    if review_status not in {"not_required", "resolved"}:
        raise ValueError("trajectory_review_not_resolved")
    return (
        build.target,
        build.qc.model_dump(mode="json"),
        review_status,
        False,
        stage_i_view.split_group_id,
    )


def _dataset_qualified_split_group(
    *,
    dataset_source: str,
    split_group_id: str,
) -> str:
    """Apply the explicit V4-to-V2 split identity compatibility rule."""

    prefix = f"{dataset_source}:"
    return split_group_id if split_group_id.startswith(prefix) else prefix + split_group_id


def _parse_full_cot(
    payload: Mapping[str, Any],
) -> StageIFullCoTViewV1 | StageIFullCoTViewV2 | None:
    """Parse one V1/V2 Stage-I row and retain only formal FullCoT parents."""

    schema_version = str(payload.get("schema_version") or "")
    if schema_version == "urbannav_stage_i_training_view_v2":
        typed = parse_stage_i_training_view_v2(dict(payload))
        return typed if isinstance(typed, StageIFullCoTViewV2) else None
    typed = parse_stage_i_training_view_v1(dict(payload))
    return typed if isinstance(typed, StageIFullCoTViewV1) else None


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
    if value and not value[0].isalpha():
        value = f"stage_ii_{value}"
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
    parser.add_argument("--reasoning-cache-jsonl", type=Path)
    parser.add_argument("--decision-cache-jsonl", type=Path)
    parser.add_argument("--trajectory-source-registry-json", required=True, type=Path)
    parser.add_argument("--output-view-jsonl", required=True, type=Path)
    parser.add_argument("--output-qc-jsonl", required=True, type=Path)
    parser.add_argument("--output-drop-jsonl", required=True, type=Path)
    parser.add_argument("--output-reference-jsonl", type=Path)
    parser.add_argument("--output-eligible-jsonl", type=Path)
    parser.add_argument("--denominator-only", action="store_true")
    parser.add_argument("--direct-latent", action="store_true")
    parser.add_argument("--max-source-step-m", type=float)
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
        output_reference_jsonl=args.output_reference_jsonl,
        output_eligible_jsonl=args.output_eligible_jsonl,
        denominator_only=args.denominator_only,
        direct_latent=args.direct_latent,
        max_source_step_m=args.max_source_step_m,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
