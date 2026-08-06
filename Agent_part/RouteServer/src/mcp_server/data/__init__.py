"""Stable facade for RouteServer-owned offline data preparation.

New code should import from io, canonical, stage_i, stage_ii, vqa or openloop.
Top-level imports remain only for migration compatibility.
"""
from mcp_server.data.io.jsonl import append_jsonl_row, iter_jsonl, write_jsonl_atomic
from mcp_server.data.legacy.reasoning_v2.source import (
    ReasoningSourceBuildConfig,
    build_reasoning_source_manifest,
    build_reasoning_source_v2,
)
from mcp_server.data.legacy.reasoning_v2.views import (
    ReasoningViewBuildConfig,
    build_reasoning_training_views,
    export_reasoning_training_views,
)

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
