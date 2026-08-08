"""Explicit registry for the first model-independent benchmark metric plugins."""

from __future__ import annotations

from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkMetricPluginV1
from mcp_server.eval.openloop.benchmark_plugin_metrics import TrajectoryGeometryMetricPluginV1
from mcp_server.eval.openloop.citywalker_benchmark_metrics import CityWalkerMetricPluginV1
from mcp_server.eval.openloop.urbannav_benchmark_metrics import UrbanNavMetricPluginV1


_METRIC_PLUGINS_V1: dict[str, BenchmarkMetricPluginV1] = {
    TrajectoryGeometryMetricPluginV1.metric_suite_id: TrajectoryGeometryMetricPluginV1(),
    CityWalkerMetricPluginV1.metric_suite_id: CityWalkerMetricPluginV1(),
    UrbanNavMetricPluginV1.metric_suite_id: UrbanNavMetricPluginV1(),
}


def get_metric_plugin_v1(metric_suite_id: str) -> BenchmarkMetricPluginV1:
    """Return the metric plugin named by a frozen benchmark profile.

    Args:
        metric_suite_id: Stable profile-owned metric suite identity.

    Returns:
        Registered immutable metric plugin implementation.

    Raises:
        KeyError: If the requested suite is not registered in this release.
    """

    try:
        return _METRIC_PLUGINS_V1[metric_suite_id]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark metric suite: {metric_suite_id}") from exc


def registered_metric_suite_ids_v1() -> tuple[str, ...]:
    """Return the sorted metric-suite identities supported by this release.

    Returns:
        Tuple of registered suite ids.
    """

    return tuple(sorted(_METRIC_PLUGINS_V1))


__all__ = ["get_metric_plugin_v1", "registered_metric_suite_ids_v1"]
