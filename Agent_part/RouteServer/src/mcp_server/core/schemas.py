"""Shared schemas for RouteServer core operations.

The core layer intentionally uses plain dictionaries for public payloads because
the existing APP and MCP APIs already expose dict-shaped JSON.  The small
dataclasses here are for internal operation results and evaluator metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CoreToolResult:
    """Normalized result envelope for core operations used by Agent tools.

    Args:
        success: Whether the operation succeeded.
        result: JSON-serializable operation payload.
        error: Optional error payload with code/message/detail.
        artifacts: Optional artifact descriptors, such as generated image crops.

    Returns:
        ``to_dict()`` returns the stable JSON object consumed by tool executors.
    """

    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "result": dict(self.result),
            "artifacts": list(self.artifacts),
            "error": self.error,
        }
