"""Base protocols for frozen benchmark dataset and metric plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkPredictionV1,
    BenchmarkProfileV1,
    BenchmarkReferenceV1,
)


@dataclass(frozen=True)
class BenchmarkJoinedRowV1:
    """One full-reference benchmark row with optional successful prediction.

    Args:
        reference: Hidden reference governing the denominator.
        prediction: Successful model output, if any.
        failure: Explicit adapter failure, if any.
    """

    reference: BenchmarkReferenceV1
    prediction: BenchmarkPredictionV1 | None
    failure: dict[str, Any] | None


class BenchmarkDatasetPluginV1(ABC):
    """Own canonical construction and validation of one benchmark dataset."""

    plugin_id: str

    @abstractmethod
    def build_package(self, **kwargs: Any) -> dict[str, Any]:
        """Materialize one frozen input/reference package.

        Args:
            **kwargs: Dataset-plugin-specific frozen build arguments.

        Returns:
            JSON-compatible package build report.
        """


class BenchmarkMetricPluginV1(ABC):
    """Evaluate one capability-gated metric suite over joined benchmark rows."""

    metric_suite_id: str

    @abstractmethod
    def evaluate(
        self,
        rows: Sequence[BenchmarkJoinedRowV1],
        *,
        profile: BenchmarkProfileV1,
    ) -> dict[str, Any]:
        """Return a JSON-compatible suite report.

        Args:
            rows: Complete-reference joined rows.
            profile: Frozen profile that authorizes time/metric semantics.

        Returns:
            Metric suite result with unsupported capabilities marked ``N/A``.
        """


__all__ = [
    "BenchmarkDatasetPluginV1",
    "BenchmarkJoinedRowV1",
    "BenchmarkMetricPluginV1",
]
