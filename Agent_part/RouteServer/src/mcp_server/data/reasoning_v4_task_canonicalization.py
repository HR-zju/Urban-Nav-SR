"""Deterministic instruction canonicalization for Reasoning V4.6.

Task taxonomy is intentionally separated from per-sample visual annotation.
Known instruction templates resolve through a versioned registry.  Unknown
normalized fingerprints may be resolved once through an external annotation
cache and then reused consistently across samples.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from mcp_server.contracts.reasoning_v4 import TaskGroundingV4


REASONING_V4_TASK_CANONICALIZATION_VERSION = (
    "reasoning_v4_task_canonicalization_v2"
)

_GENERIC_ROUTE_INSTRUCTIONS = (
    "follow the route",
    "follow the planned route",
    "follow the navigation route",
    "navigate along the route",
    "continue along the route",
    "navigate safely on the sidewalk and follow the visible route",
    "请沿路线前进",
    "请按照路线前进",
    "沿导航路线前进",
)


def normalize_task_instruction_v4(instruction: str) -> str:
    """Normalize instruction text for stable registry lookup.

    Unicode is normalized with NFKC, letters are case-folded, punctuation and
    symbols become spaces, and all whitespace runs collapse to one ASCII space.

    Args:
        instruction: Raw prompt-visible instruction.

    Returns:
        Stable normalized instruction.

    Raises:
        ValueError: If the normalized instruction is empty.
    """

    normalized = unicodedata.normalize("NFKC", str(instruction)).casefold()
    normalized = "".join(
        " "
        if unicodedata.category(character).startswith(("P", "S"))
        else character
        for character in normalized
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise ValueError("task instruction is empty after normalization")
    return normalized


def task_instruction_fingerprint_v4(instruction: str) -> str:
    """Return the canonical SHA-256 instruction fingerprint.

    Args:
        instruction: Raw prompt-visible instruction.

    Returns:
        Lowercase SHA-256 digest of the normalized instruction.
    """

    normalized = normalize_task_instruction_v4(instruction)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _generic_route_task() -> TaskGroundingV4:
    """Build the frozen common route-following taxonomy."""

    return TaskGroundingV4(
        objective="follow_route",
        requirements=["route_adherence"],
        ambiguities=[],
    )


_TASK_REGISTRY = MappingProxyType(
    {
        task_instruction_fingerprint_v4(instruction): (
            _generic_route_task()
        )
        for instruction in _GENERIC_ROUTE_INSTRUCTIONS
    }
)


@dataclass(frozen=True)
class TaskCanonicalizationResultV4:
    """Describe one deterministic task-canonicalization result.

    Args:
        fingerprint: Normalized instruction fingerprint.
        normalized_instruction: Normalized instruction text.
        task_grounding: Canonical task taxonomy.
        source: Frozen registry or caller-provided annotation cache.
        registry_version: Exact canonicalization version.
    """

    fingerprint: str
    normalized_instruction: str
    task_grounding: TaskGroundingV4
    source: str
    registry_version: str = REASONING_V4_TASK_CANONICALIZATION_VERSION


def canonicalize_task_instruction_v4(
    instruction: str | None,
    *,
    annotation_cache: Mapping[str, TaskGroundingV4] | None = None,
) -> TaskCanonicalizationResultV4 | None:
    """Resolve a known or cached instruction taxonomy.

    Args:
        instruction: Prompt-visible instruction, if present.
        annotation_cache: Optional fingerprint-to-taxonomy cache for previously
            annotated unknown instructions.

    Returns:
        Deterministic result when a registry/cache entry exists, otherwise
        ``None`` so the caller can annotate the fingerprint once.
    """

    if instruction is None:
        return None
    normalized = normalize_task_instruction_v4(instruction)
    fingerprint = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()
    task = _TASK_REGISTRY.get(fingerprint)
    source = "versioned_registry"
    if task is None and annotation_cache is not None:
        task = annotation_cache.get(fingerprint)
        source = "annotation_cache"
    if task is None:
        return None
    return TaskCanonicalizationResultV4(
        fingerprint=fingerprint,
        normalized_instruction=normalized,
        task_grounding=task,
        source=source,
    )


def registered_task_fingerprints_v4() -> tuple[str, ...]:
    """Return sorted built-in registry fingerprints.

    Returns:
        Immutable sorted fingerprint tuple.
    """

    return tuple(sorted(_TASK_REGISTRY))


__all__ = [
    "REASONING_V4_TASK_CANONICALIZATION_VERSION",
    "TaskCanonicalizationResultV4",
    "canonicalize_task_instruction_v4",
    "normalize_task_instruction_v4",
    "registered_task_fingerprints_v4",
    "task_instruction_fingerprint_v4",
]
