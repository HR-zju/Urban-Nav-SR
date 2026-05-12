"""Reusable async batch execution primitives.

Batch tools often need bounded concurrency for slow remote calls while keeping
manifest output deterministic. This module centralizes the semaphore/task
bookkeeping so orchestration code can stay focused on domain work.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class OrderedTaskResult(Generic[R]):
    """Result envelope for one bounded-concurrency batch item."""

    index: int
    item: object
    ok: bool
    value: R | None = None
    error: str | None = None


async def run_ordered_bounded(
    items: Sequence[T],
    worker: Callable[[int, T], Awaitable[R]],
    *,
    concurrency: int,
) -> list[OrderedTaskResult[R]]:
    """Run async workers with bounded concurrency and stable result ordering.

    The returned list has the same order and length as ``items`` regardless of
    completion order. Worker exceptions are captured in ``OrderedTaskResult`` so
    callers can decide how to report partial failures.
    """

    limit = max(1, int(concurrency))
    semaphore = asyncio.Semaphore(limit)
    results: list[OrderedTaskResult[R] | None] = [None] * len(items)

    async def _run_one(index: int, item: T) -> None:
        async with semaphore:
            try:
                value = await worker(index, item)
                results[index] = OrderedTaskResult(index=index, item=item, ok=True, value=value)
            except Exception as exc:
                results[index] = OrderedTaskResult(index=index, item=item, ok=False, error=str(exc))

    tasks = [asyncio.create_task(_run_one(index, item)) for index, item in enumerate(items)]
    if tasks:
        await asyncio.gather(*tasks)
    return [
        result
        if result is not None
        else OrderedTaskResult(index=index, item=items[index], ok=False, error="task_did_not_complete")
        for index, result in enumerate(results)
    ]
