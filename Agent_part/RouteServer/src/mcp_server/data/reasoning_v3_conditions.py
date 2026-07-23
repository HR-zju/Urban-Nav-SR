"""Expose model-visible navigation source identities for Reasoning V3.4.

This module deliberately performs no direction derivation, route-geometry
interpretation, reliability scoring, or multi-source fusion.  The LVLM may
account for an input as used/ignored and describe a visible relation, while
the only strong direction label is consumed separately from a profile-stamped
canonical record.
"""

from __future__ import annotations

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3
from mcp_server.contracts.reasoning_v3 import NavigationContextSourceSpecV3


_SOURCE_TYPE_BY_MODALITY = {
    "instruction": "instruction",
    "bev": "bev",
    "goal": "goal",
    "goal_image": "goal",
    "route": "route",
    "tool": "tool",
    "tool_route": "tool",
}

def build_navigation_context_sources_v3(
    source: ReasoningSourceV3,
) -> list[NavigationContextSourceSpecV3]:
    """Return ordered identities for navigation evidence visible to the LVLM.

    Args:
        source: Frozen V3.4 source after the model-visible boundary filter.

    Returns:
        At most twelve source specifications.  Every field is copied from the
        visible evidence catalog or from the fixed modality-to-role registry;
        no teacher-private direction or geometry is read.
    """

    result: list[NavigationContextSourceSpecV3] = []
    seen_refs: set[str] = set()
    for entry in source.model_input.evidence_catalog:
        source_type = _SOURCE_TYPE_BY_MODALITY.get(entry.modality)
        if source_type is None or entry.ref in seen_refs:
            continue
        seen_refs.add(entry.ref)
        result.append(
            NavigationContextSourceSpecV3(
                source_type=source_type,
                source_ref=entry.ref,
            )
        )
        if len(result) == 12:
            break
    return result


__all__ = ["build_navigation_context_sources_v3"]
