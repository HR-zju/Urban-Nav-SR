"""Few-shot registry for the Reasoning V3.4 minimal teachers.

V3.4 intentionally starts without semantic demonstrations.  The previous
examples encoded removed instruction/objective ontologies, projected
corridors, heuristic actions, and self-reported confidence.  Reintroducing a
demonstration requires a reviewed, versioned example whose claims are all
supported by student-visible evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3


REASONING_V3_FEWSHOT_VERSION = "urbannav_reasoning_v3_4_minimal_fewshot_empty_v1"
FewShotStageV3 = Literal["semantic_observation", "decision_rationale"]


@dataclass(frozen=True)
class ReasoningFewShotV3:
    """Describe one reviewed teacher demonstration."""

    example_id: str
    stage: FewShotStageV3
    dataset_tags: tuple[str, ...]
    modality_tags: tuple[str, ...]
    input_summary: dict[str, Any]
    output: dict[str, Any]


FEWSHOT_REGISTRY_V3 = MappingProxyType({})


def select_reasoning_v3_few_shots(
    source: ReasoningSourceV3,
    *,
    stage: FewShotStageV3,
    limit: int = 3,
) -> list[ReasoningFewShotV3]:
    """Return no examples until a V3.4 example passes manual review."""

    del source, stage, limit
    return []


__all__ = [
    "FEWSHOT_REGISTRY_V3",
    "REASONING_V3_FEWSHOT_VERSION",
    "ReasoningFewShotV3",
    "select_reasoning_v3_few_shots",
]
