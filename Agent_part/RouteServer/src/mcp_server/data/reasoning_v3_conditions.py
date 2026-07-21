"""Derive and fuse model-visible navigation context for Reasoning V3.3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3
from mcp_server.contracts.reasoning_v3 import (
    NavigationContextSourceSpecV3,
    NavigationSourceGroundingV3,
)


def build_navigation_context_sources_v3(
    source: ReasoningSourceV3,
) -> list[NavigationContextSourceSpecV3]:
    """Build typed route-specific sources while retaining optional BEV context.

    BEV differs from a generic instruction: when a BEV image is supplied it is
    always represented as route-topology context. If no direction can be read,
    the teacher must emit ``context_only`` or ``insufficient_evidence`` rather
    than silently dropping the image.

    Args:
        source: Frozen source containing model-visible evidence and private
            deterministic source directions.

    Returns:
        Ordered typed source specifications, capped at twelve entries.
    """

    fixed = source.teacher_supervision_context.get("fixed_labels") or {}
    source_directions = {
        str(ref): str(direction)
        for ref, direction in (fixed.get("condition_source_directions") or {}).items()
        if str(direction) in {"left", "straight", "right", "stop"}
    }
    result: list[NavigationContextSourceSpecV3] = []
    for entry in source.model_input.evidence_catalog:
        source_type = {
            "instruction": "instruction",
            "goal": "goal",
            "goal_image": "goal",
            "waypoint": "waypoint",
            "route": "route",
            "tool": "tool",
            "tool_route": "tool",
            "bev": "bev",
        }.get(entry.modality)
        if source_type is None:
            continue
        role = {
            # Instruction semantics are LVLM-authored in V3.3. The source builder
            # exposes the source identity but does not classify its text.
            "instruction": "unknown",
            "bev": "route_topology",
            "goal": "goal_context",
            "waypoint": "waypoint_sequence",
            "route": "route_topology",
            "tool": "tool_route",
        }[source_type]
        deterministic = source_directions.get(entry.ref, "unknown")
        if source_type == "instruction" and deterministic != "unknown":
            role = "direction_command"
        reliability = "high" if deterministic != "unknown" else ("medium" if source_type == "route" else "low")
        result.append(
            NavigationContextSourceSpecV3(
                source_type=source_type,
                source_ref=entry.ref,
                role=role,
                deterministic_direction=deterministic,
                reliability=reliability,
            )
        )
    return result[:12]


def fuse_navigation_context_v3(
    groundings: Iterable[NavigationSourceGroundingV3],
) -> tuple[str, str | None, str, str]:
    r"""Fuse per-source context relations without borrowing demonstration GT.

    Let \(D\) be the set of known directives among resolved sources and \(C\)
    their non-null target corridors. Fusion is positive only when
    \(|D|=1\), \(|C|\leq1\), and no source reports conflict.

    Args:
        groundings: Independently validated source-to-FPV relations.

    Returns:
        ``(direction, target_corridor, consistency, execution_status)``.
    """

    items = list(groundings)
    if not items:
        return "unknown", None, "not_applicable", "not_applicable"
    if any(item.status == "conflict" for item in items):
        return "unknown", None, "conflict", "conflict"
    resolved = [item for item in items if item.status in {"aligned", "locally_blocked"}]
    directions = {item.directive for item in resolved if item.directive != "unknown"}
    targets = {item.target_corridor_ref for item in resolved if item.target_corridor_ref is not None}
    if len(directions) > 1 or len(targets) > 1:
        return "unknown", None, "conflict", "conflict"
    direction = next(iter(directions), "unknown")
    target = next(iter(targets), None)
    if resolved:
        consistency = "single_source" if len(items) == 1 else (
            "consistent" if len(resolved) == len(items) else "partial"
        )
        status = "locally_blocked" if any(item.status == "locally_blocked" for item in resolved) else "clear"
        return direction, target, consistency, status
    statuses = {item.status for item in items}
    if statuses <= {"context_only", "ignored"} and "context_only" in statuses:
        return "unknown", None, "unknown", "context_only"
    if "not_visible" in statuses and statuses <= {"not_visible", "ignored"}:
        return "unknown", None, "unknown", "not_visible"
    return "unknown", None, "unknown", "insufficient_evidence"


def condition_source_directions_from_row_v3(row: Mapping[str, Any]) -> dict[str, str]:
    """Read directions only from explicit structured route evidence.

    Args:
        row: Original heterogeneous manifest row.

    Returns:
        Mapping from canonical source reference to coarse direction.
    """

    result: dict[str, str] = {}
    if bool(row.get("independent_route_evidence")):
        route_context = row.get("route_context")
        containers = [row]
        if isinstance(route_context, Mapping):
            containers.insert(0, route_context)
        for container in containers:
            for key in ("desired_direction", "route_direction", "direction"):
                direction = _normalize_direction(container.get(key))
                if direction != "unknown":
                    result["route.active"] = direction
                    return result
    return result


def _normalize_direction(value: Any) -> str:
    """Normalize common direction labels used by source manifests."""

    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"left", "turn_left", "go_left"}:
        return "left"
    if normalized in {"right", "turn_right", "go_right"}:
        return "right"
    if normalized in {"straight", "go_straight", "forward"}:
        return "straight"
    if normalized in {"stop", "hold", "hold_position"}:
        return "stop"
    return "unknown"


__all__ = [
    "build_navigation_context_sources_v3",
    "condition_source_directions_from_row_v3",
    "fuse_navigation_context_v3",
]
