from __future__ import annotations

import asyncio
import unittest

from mcp_server.utils.async_batch import run_ordered_bounded


class AsyncBatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_ordered_bounded_preserves_input_order(self) -> None:
        async def worker(_index: int, value: int) -> str:
            await asyncio.sleep(0.01 * (3 - value))
            return f"item-{value}"

        results = await run_ordered_bounded([0, 1, 2, 3], worker, concurrency=4)

        self.assertTrue(all(result.ok for result in results))
        self.assertEqual([result.value for result in results], ["item-0", "item-1", "item-2", "item-3"])

    async def test_run_ordered_bounded_captures_worker_errors(self) -> None:
        async def worker(_index: int, value: int) -> int:
            if value == 2:
                raise RuntimeError("boom")
            return value * 10

        results = await run_ordered_bounded([1, 2, 3], worker, concurrency=2)

        self.assertEqual([result.ok for result in results], [True, False, True])
        self.assertEqual(results[0].value, 10)
        self.assertIn("boom", str(results[1].error))
        self.assertEqual(results[2].value, 30)


if __name__ == "__main__":
    unittest.main()
