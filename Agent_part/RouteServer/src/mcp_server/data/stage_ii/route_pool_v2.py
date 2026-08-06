"""Build the canonical Stage II V2 reasoning-route pool."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV2,
    StageIReasoningDropoutViewV2,
    parse_stage_i_training_view_v2,
)
from mcp_server.contracts.stage_ii_cfm import (
    StageIICFMTrainingViewV2,
    StageIIContextCacheV1,
    StageIIProjectionDropV2,
    StageIITrajectorySidecarV2,
)
from mcp_server.data.stage_i.views import (
    project_stage_i_reasoning_dropout_from_full_v2,
)
from mcp_server.data.stage_ii.views import (
    build_stage_ii_projection_drop_v2,
    project_stage_ii_cfm_view_with_audit_v2,
)


REPORT_VERSION = "urbannav_stage_ii_cfm_v2_route_pool_build_report_v1"


def build_stage_ii_cfm_route_pool_v2(
    *,
    stage_i_full_jsonl: str | Path,
    trajectory_sidecar_jsonl: str | Path,
    reasoning_cache_jsonl: str | Path,
    decision_cache_jsonl: str | Path,
    output_route_jsonl: str | Path,
    output_qc_jsonl: str | Path,
    output_drop_jsonl: str | Path,
) -> dict[str, Any]:
    """Join V2 artifacts and emit independently auditable reasoning routes.

    Each accepted FullCoT parent deterministically produces one signed
    reasoning-dropout candidate.  The builder never accepts an externally
    supplied empty or failed reasoning record as the no-CoT route.  A parent
    may therefore yield zero, one, or two canonical routes; the ms-swift
    training sampler separately requires both routes by default before making
    its mutually exclusive per-epoch choice.

    Args:
        stage_i_full_jsonl: Formal Stage I V2 FullCoT views.
        trajectory_sidecar_jsonl: Accepted four-second V2 trajectory sidecars.
        reasoning_cache_jsonl: Pass-R cache records, including rejected rows.
        decision_cache_jsonl: Pass-D cache records, including rejected rows.
        output_route_jsonl: Canonical Stage II V2 route-pool destination.
        output_qc_jsonl: Reused accepted trajectory QC destination.
        output_drop_jsonl: Per-route fail-closed audit destination.

    Returns:
        Deterministic counts, input paths, output paths, and cache-index facts.

    Raises:
        ValueError: If a formal input, unique identity, or accepted cache
            identity conflicts with another row.
    """

    sidecars = _index_trajectory_sidecars(trajectory_sidecar_jsonl)
    reasoning, reasoning_stats = _index_context_caches(
        reasoning_cache_jsonl,
        context_kind="reasoning_grounding",
    )
    decisions, decision_stats = _index_context_caches(
        decision_cache_jsonl,
        context_kind="decision",
    )
    output_route = Path(output_route_jsonl).expanduser()
    output_qc = Path(output_qc_jsonl).expanduser()
    output_drop = Path(output_drop_jsonl).expanduser()
    for destination in (output_route, output_qc, output_drop):
        destination.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    seen_full_ids: set[str] = set()
    emitted_qc_ids: set[str] = set()
    with (
        output_route.open("w", encoding="utf-8") as route_handle,
        output_qc.open("w", encoding="utf-8") as qc_handle,
        output_drop.open("w", encoding="utf-8") as drop_handle,
    ):
        for raw_view in _iter_jsonl(stage_i_full_jsonl):
            counts["input_rows"] += 1
            parsed = parse_stage_i_training_view_v2(raw_view)
            if not isinstance(parsed, StageIFullCoTViewV2):
                counts["skipped_non_full_cot"] += 1
                continue
            if parsed.sample_id in seen_full_ids:
                raise ValueError(f"duplicate Stage I V2 FullCoT sample_id: {parsed.sample_id}")
            seen_full_ids.add(parsed.sample_id)
            counts["input_full_cot"] += 1
            dropout = project_stage_i_reasoning_dropout_from_full_v2(parsed)
            sidecar, sidecar_error = _resolve_sidecar(parsed, sidecars)
            if sidecar is not None and sidecar.sample_id not in emitted_qc_ids:
                _write_json_line(
                    qc_handle,
                    sidecar.trajectory_qc.model_dump(mode="json", exclude_none=True),
                )
                emitted_qc_ids.add(sidecar.sample_id)
                counts["output_qc"] += 1

            cache_key = (parsed.sample_id, str(parsed.modality_profile_id))
            decision = decisions.get(cache_key)
            reasoning_cache = reasoning.get(cache_key)
            accepted_modes: set[str] = set()
            for route in (parsed, dropout):
                context_mode = _context_mode(route)
                counts[f"candidate:{context_mode}"] += 1
                result: StageIICFMTrainingViewV2 | StageIIProjectionDropV2
                if sidecar is None:
                    result = _drop_for_route(
                        route,
                        reason_code=sidecar_error or "trajectory_sidecar_missing",
                        detail=(
                            sidecar_error
                            or f"no V2 trajectory sidecar for source_group_id={parsed.source_group_id}"
                        ),
                    )
                elif decision is None:
                    result = _drop_for_route(
                        route,
                        reason_code="decision_context_cache_missing",
                        detail=(
                            "no Pass-D cache for parent/modality key="
                            f"{cache_key!r}"
                        ),
                    )
                else:
                    result = project_stage_ii_cfm_view_with_audit_v2(
                        route,
                        trajectory_target=sidecar.trajectory_target,
                        reasoning_context=(
                            reasoning_cache
                            if isinstance(route, StageIFullCoTViewV2)
                            else None
                        ),
                        decision_context=decision,
                        review_status=_review_status(parsed),
                    )
                if isinstance(result, StageIICFMTrainingViewV2):
                    _write_json_line(
                        route_handle,
                        result.model_dump(mode="json", exclude_none=True),
                    )
                    accepted_modes.add(result.context_mode)
                    counts["output_routes"] += 1
                    counts[f"output_route:{result.context_mode}"] += 1
                    counts[f"dataset:{result.dataset_source}"] += 1
                else:
                    _write_json_line(
                        drop_handle,
                        result.model_dump(mode="json", exclude_none=True),
                    )
                    counts["output_drops"] += 1
                    counts[f"drop_route:{result.requested_context_mode}"] += 1
                    counts[f"drop_reason:{result.reason_code}"] += 1
            _count_parent_route_state(counts, accepted_modes)

    return {
        "schema_version": REPORT_VERSION,
        "inputs": {
            "stage_i_full_jsonl": str(Path(stage_i_full_jsonl).expanduser()),
            "trajectory_sidecar_jsonl": str(Path(trajectory_sidecar_jsonl).expanduser()),
            "reasoning_cache_jsonl": str(Path(reasoning_cache_jsonl).expanduser()),
            "decision_cache_jsonl": str(Path(decision_cache_jsonl).expanduser()),
        },
        "outputs": {
            "route_jsonl": str(output_route),
            "qc_jsonl": str(output_qc),
            "drop_jsonl": str(output_drop),
        },
        "join_contract": {
            "trajectory_key": "stage_i.source_group_id == trajectory.sample_id",
            "split_group_key": (
                "dataset_qualified(stage_i.split_group_id) == "
                "dataset_qualified(trajectory.split_group_id)"
            ),
            "context_key": "(parent_full_cot_sample_id, modality_profile_id)",
            "dropout_source": "deterministic_projection_from_validated_full_cot_v2",
            "paired_training_default": True,
        },
        "indexed": {
            "trajectory_sidecars": len(sidecars),
            "reasoning_caches": reasoning_stats,
            "decision_caches": decision_stats,
        },
        "counts": dict(sorted(counts.items())),
    }


def _index_trajectory_sidecars(path: str | Path) -> dict[str, StageIITrajectorySidecarV2]:
    """Index accepted trajectory sidecars by their canonical source parent."""

    result: dict[str, StageIITrajectorySidecarV2] = {}
    for raw in _iter_jsonl(path):
        sidecar = StageIITrajectorySidecarV2.model_validate(raw)
        existing = result.get(sidecar.sample_id)
        if existing is not None:
            if existing.sidecar_digest == sidecar.sidecar_digest:
                continue
            raise ValueError(f"conflicting V2 trajectory sidecar: {sidecar.sample_id}")
        result[sidecar.sample_id] = sidecar
    return result


def _index_context_caches(
    path: str | Path,
    *,
    context_kind: str,
) -> tuple[dict[tuple[str, str], StageIIContextCacheV1], dict[str, int]]:
    """Select one deterministic cache per parent while rejecting ambiguity."""

    result: dict[tuple[str, str], StageIIContextCacheV1] = {}
    counts: Counter[str] = Counter()
    for raw in _iter_jsonl(path):
        cache = StageIIContextCacheV1.model_validate(raw)
        counts["input_rows"] += 1
        if cache.context_kind != context_kind:
            counts["skipped_other_kind"] += 1
            continue
        key = (cache.sample_id, str(cache.modality_profile_id))
        existing = result.get(key)
        if existing is None:
            result[key] = cache
            counts[f"selected:{cache.parse_status}"] += 1
            continue
        if existing.cache_digest == cache.cache_digest:
            counts["duplicate_equivalent"] += 1
            continue
        if existing.parse_status == "accepted" and cache.parse_status == "accepted":
            raise ValueError(f"conflicting accepted {context_kind} cache: {key!r}")
        if existing.parse_status == "rejected" and cache.parse_status == "accepted":
            result[key] = cache
            counts["accepted_replaced_rejected"] += 1
        elif existing.parse_status == "accepted":
            counts["rejected_ignored_after_accepted"] += 1
        elif cache.cache_digest < existing.cache_digest:
            result[key] = cache
            counts["rejected_deterministic_replacement"] += 1
        else:
            counts["rejected_deterministic_ignore"] += 1
    counts["selected_keys"] = len(result)
    counts["selected_accepted"] = sum(
        cache.parse_status == "accepted" for cache in result.values()
    )
    counts["selected_rejected"] = sum(
        cache.parse_status == "rejected" for cache in result.values()
    )
    return result, dict(sorted(counts.items()))


def _resolve_sidecar(
    full: StageIFullCoTViewV2,
    sidecars: Mapping[str, StageIITrajectorySidecarV2],
) -> tuple[StageIITrajectorySidecarV2 | None, str | None]:
    """Resolve and validate one trajectory sidecar against Stage I lineage."""

    sidecar = sidecars.get(full.source_group_id)
    if sidecar is None:
        return None, "trajectory_sidecar_missing"
    identities = (
        sidecar.source_group_id == full.source_group_id,
        sidecar.dataset_source == full.dataset_source,
        _dataset_qualified_split_group(
            dataset_source=sidecar.dataset_source,
            split_group_id=sidecar.split_group_id,
        )
        == _dataset_qualified_split_group(
            dataset_source=full.dataset_source,
            split_group_id=full.split_group_id,
        ),
    )
    if not all(identities):
        return None, "trajectory_sidecar_identity_mismatch"
    return sidecar, None


def _dataset_qualified_split_group(
    *,
    dataset_source: str,
    split_group_id: str,
) -> str:
    """Return one collision-safe split identity across contract generations.

    Historical Stage I formal views stored the native clip identity, while the
    four-second trajectory builder prefixes the same identity with its dataset
    source. Treat only that exact anchored prefix as equivalent. Arbitrary
    suffix matches remain invalid, so a different clip or dataset still fails
    closed.
    """

    prefix = f"{dataset_source}:"
    return split_group_id if split_group_id.startswith(prefix) else prefix + split_group_id


def _drop_for_route(
    route: StageIFullCoTViewV2 | StageIReasoningDropoutViewV2,
    *,
    reason_code: str,
    detail: str,
) -> StageIIProjectionDropV2:
    """Build a route-specific audit when projection cannot be attempted."""

    parent_id = (
        route.sample_id
        if isinstance(route, StageIFullCoTViewV2)
        else route.dropout_link.parent_full_cot_sample_id
    )
    return build_stage_ii_projection_drop_v2(
        parent_sample_id=parent_id,
        stage_i_route_sample_id=route.sample_id,
        stage_i_route_task_type=route.task_type,
        requested_context_mode=_context_mode(route),
        source_group_id=route.source_group_id,
        split_group_id=route.split_group_id,
        dataset_source=route.dataset_source,
        modality_profile_id=str(route.modality_profile_id),
        reason_code=reason_code,
        detail=detail,
    )


def _context_mode(
    route: StageIFullCoTViewV2 | StageIReasoningDropoutViewV2,
) -> str:
    """Map one signed Stage I route to its Stage II context mode."""

    if isinstance(route, StageIFullCoTViewV2):
        return "reasoning_plus_decision"
    return "decision_only"


def _review_status(full: StageIFullCoTViewV2) -> str:
    """Resolve the formal action-review state without weakening admission."""

    explicit = str(full.metadata.get("review_status") or "")
    if explicit in {"not_required", "resolved"}:
        return explicit
    if full.parent_audit_id is None:
        raise ValueError("Stage II V2 FullCoT parent lacks a formal action audit")
    return "not_required"


def _count_parent_route_state(counts: Counter[str], modes: set[str]) -> None:
    """Count exact per-parent route coverage for downstream admission."""

    if modes == {"reasoning_plus_decision", "decision_only"}:
        counts["parent_routes:paired"] += 1
    elif modes == {"reasoning_plus_decision"}:
        counts["parent_routes:reasoning_only"] += 1
    elif modes == {"decision_only"}:
        counts["parent_routes:decision_only"] += 1
    else:
        counts["parent_routes:none"] += 1


def _iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects from one JSONL artifact."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object")
            yield value


def _write_json_line(handle, payload: Mapping[str, Any]) -> None:
    """Write one compact deterministic JSON object."""

    handle.write(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")) + "\n")


__all__ = ["REPORT_VERSION", "build_stage_ii_cfm_route_pool_v2"]
