"""Stable facade for RouteServer-owned offline data preparation.

New code should import from io, canonical, stage_i, stage_ii, vqa or openloop.
Top-level imports remain only for migration compatibility.
"""
from importlib import import_module
from typing import Any

from mcp_server.data.io.jsonl import append_jsonl_row, iter_jsonl, write_jsonl_atomic

# Legacy reasoning exports stay public for compatibility, but they are loaded
# only when an old caller requests them. Stage-II data tools must not import the
# full agent stack just by importing ``mcp_server.data``.
_LEGACY_EXPORTS = {
    "ReasoningSourceBuildConfig": (
        "mcp_server.data.legacy.reasoning_v2.source",
        "ReasoningSourceBuildConfig",
    ),
    "build_reasoning_source_manifest": (
        "mcp_server.data.legacy.reasoning_v2.source",
        "build_reasoning_source_manifest",
    ),
    "build_reasoning_source_v2": (
        "mcp_server.data.legacy.reasoning_v2.source",
        "build_reasoning_source_v2",
    ),
    "ReasoningViewBuildConfig": (
        "mcp_server.data.legacy.reasoning_v2.views",
        "ReasoningViewBuildConfig",
    ),
    "build_reasoning_training_views": (
        "mcp_server.data.legacy.reasoning_v2.views",
        "build_reasoning_training_views",
    ),
    "export_reasoning_training_views": (
        "mcp_server.data.legacy.reasoning_v2.views",
        "export_reasoning_training_views",
    ),
}


def __getattr__(name: str) -> Any:
    """Load legacy compatibility exports on demand."""

    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "ReasoningSourceBuildConfig",
    "ReasoningViewBuildConfig",
    "append_jsonl_row",
    "build_reasoning_source_manifest",
    "build_reasoning_source_v2",
    "build_reasoning_training_views",
    "export_reasoning_training_views",
    "iter_jsonl",
    "write_jsonl_atomic",
]
