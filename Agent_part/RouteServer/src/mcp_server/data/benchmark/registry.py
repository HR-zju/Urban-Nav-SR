"""Explicit registry for RouteServer-owned benchmark dataset plugins."""

from __future__ import annotations

from mcp_server.data.benchmark.citywalker_dataset_plugin import CityWalkerDatasetPluginV1
from mcp_server.data.benchmark.language_waypoint_dataset_plugin import (
    LanguageWaypointDatasetPluginV1,
)
from mcp_server.data.benchmark.urbannav_dataset_plugin import UrbanNavDatasetPluginV1
from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkDatasetPluginV1


_DATASET_PLUGINS_V1: dict[str, BenchmarkDatasetPluginV1] = {
    CityWalkerDatasetPluginV1.plugin_id: CityWalkerDatasetPluginV1(),
    LanguageWaypointDatasetPluginV1.plugin_id: LanguageWaypointDatasetPluginV1(),
    UrbanNavDatasetPluginV1.plugin_id: UrbanNavDatasetPluginV1(),
}


def get_benchmark_dataset_plugin_v1(plugin_id: str) -> BenchmarkDatasetPluginV1:
    """Return a RouteServer-owned dataset plugin by its frozen identity.

    Args:
        plugin_id: Profile-owned dataset plugin identifier.

    Returns:
        Registered dataset plugin implementation.

    Raises:
        KeyError: If the requested plugin is unsupported by this release.
    """

    try:
        return _DATASET_PLUGINS_V1[plugin_id]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark dataset plugin: {plugin_id}") from exc


def registered_benchmark_dataset_plugin_ids_v1() -> tuple[str, ...]:
    """Return sorted dataset-plugin identities implemented by RouteServer.

    Returns:
        Immutable sorted plugin-id tuple.
    """

    return tuple(sorted(_DATASET_PLUGINS_V1))


__all__ = ["get_benchmark_dataset_plugin_v1", "registered_benchmark_dataset_plugin_ids_v1"]
