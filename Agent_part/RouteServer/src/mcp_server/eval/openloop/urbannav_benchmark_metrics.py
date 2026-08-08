"""UrbanNav-specific diagnostic metrics for the generic benchmark contract."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from mcp_server.contracts.benchmark_plugin import BenchmarkProfileV1
from mcp_server.eval.openloop.benchmark_plugin_base import (
    BenchmarkJoinedRowV1,
    BenchmarkMetricPluginV1,
)


class UrbanNavMetricPluginV1(BenchmarkMetricPluginV1):
    """Evaluate optional direction/readout diagnostics carried by UrbanNav rows."""

    metric_suite_id = "urbannav_openloop_metrics_v1"

    def evaluate(
        self,
        rows: Sequence[BenchmarkJoinedRowV1],
        *,
        profile: BenchmarkProfileV1,
    ) -> dict[str, Any]:
        """Evaluate diagnostic direction agreement if both sides expose it.

        Args:
            rows: Full UrbanNav joined output ledger.
            profile: Frozen profile; retained for protocol parity.

        Returns:
            Direction scorecard or explicit ``N/A`` when the selected adapter
            does not emit a textual direction readout.
        """

        del profile
        labeled: list[tuple[str, str]] = []
        for row in rows:
            label = row.reference.subgroup_metadata.get("direction_label")
            predicted = row.prediction.metadata.get("direction_label") if row.prediction is not None else None
            if isinstance(label, str) and isinstance(predicted, str):
                labeled.append((label, predicted))
        if not labeled:
            return {
                "metric_id": self.metric_suite_id,
                "status": "N/A",
                "reason": "direction_readout_unavailable",
                "reference_count": len(rows),
            }
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        correct = 0
        for label, predicted in labeled:
            counts[label][1] += 1
            if label == predicted:
                correct += 1
                counts[label][0] += 1
        recall = {label: hit / total for label, (hit, total) in sorted(counts.items()) if total}
        return {
            "metric_id": self.metric_suite_id,
            "status": "ok",
            "reference_count": len(rows),
            "readout_count": len(labeled),
            "readout_coverage": len(labeled) / len(rows) if rows else None,
            "direction_accuracy": correct / len(labeled),
            "direction_macro_recall": sum(recall.values()) / len(recall) if recall else None,
            "direction_recall_by_label": recall,
        }


__all__ = ["UrbanNavMetricPluginV1"]
