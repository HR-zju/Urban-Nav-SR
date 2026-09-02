"""Stage II trajectory, context-cache and CFM-view builders."""

from importlib import import_module
from typing import Any

# Keep the public compatibility exports while loading cached-query builders
# only when requested. The trajectory corpus must not import the agent stack.
_LAZY_EXPORTS = {
    "build_stage_ii_cached_traj_query_observation_v1": (
        "mcp_server.data.stage_ii.views",
        "build_stage_ii_cached_traj_query_observation_v1",
    ),
    "build_stage_ii_cached_traj_query_source_view_v2": (
        "mcp_server.data.stage_ii.views",
        "build_stage_ii_cached_traj_query_source_view_v2",
    ),
    "build_stage_ii_cached_traj_query_training_view_v1": (
        "mcp_server.data.stage_ii.views",
        "build_stage_ii_cached_traj_query_training_view_v1",
    ),
    "build_stage_ii_cached_traj_query_training_view_v2": (
        "mcp_server.data.stage_ii.views",
        "build_stage_ii_cached_traj_query_training_view_v2",
    ),
    "build_stage_ii_cached_traj_query_pairs_v1": (
        "mcp_server.data.stage_ii.cached_pairs",
        "build_stage_ii_cached_traj_query_pairs_v1",
    ),
    "build_stage_ii_cached_traj_query_pairs_v2": (
        "mcp_server.data.stage_ii.cached_pairs",
        "build_stage_ii_cached_traj_query_pairs_v2",
    ),
    "build_stage_ii_cached_traj_query_source_views_v2": (
        "mcp_server.data.stage_ii.cached_pairs",
        "build_stage_ii_cached_traj_query_source_views_v2",
    ),
}


def __getattr__(name: str) -> Any:
    """Load cached-query compatibility exports on demand."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "build_stage_ii_cached_traj_query_observation_v1",
    "build_stage_ii_cached_traj_query_source_view_v2",
    "build_stage_ii_cached_traj_query_training_view_v1",
    "build_stage_ii_cached_traj_query_training_view_v2",
    "build_stage_ii_cached_traj_query_pairs_v1",
    "build_stage_ii_cached_traj_query_pairs_v2",
    "build_stage_ii_cached_traj_query_source_views_v2",
]
