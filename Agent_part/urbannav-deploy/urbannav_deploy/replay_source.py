"""Paced recorded observations for the existing deployment application."""
from __future__ import annotations

import json
from pathlib import Path
import time


class ReplaySource:
    """Deliver recorded rows on their source clock with latest-only admission.

    Args:
        requests: Records containing request_id, offset_sec and a service row.
        source_start_sec: Acquisition clock origin corresponding to offset zero.
        end_offset_sec: End of the recorded interval, including any idle tail.
        record: Shared event callback.
        monotonic: Local clock, injectable for deterministic CPU verification.

    Images in rows must already be accessible to the model service. This adapter
    preserves recorded data; it does not claim camera/GUI capture provenance.
    """
    def __init__(self, requests, source_start_sec, end_offset_sec, record,
                 monotonic=time.monotonic):
        self.requests = requests
        self.source_start = source_start_sec
        self.end_offset = end_offset_sec
        self.record, self.monotonic = record, monotonic
        self.index = 0
        self.started = None
        offsets = [row['offset_sec'] for row in requests]
        if not offsets or offsets[0] < 0 or offsets[-1] > end_offset_sec:
            raise ValueError('requests must lie inside the recorded interval')
        if any(a >= b for a, b in zip(offsets, offsets[1:])):
            raise ValueError('request offsets must strictly increase')

    def start(self):
        """Anchor source time to the local clock after model readiness."""
        self.started = self.monotonic()

    def now(self):
        """Return acquisition-clock seconds: source origin + elapsed local time."""
        return self.source_start + self.monotonic() - self.started

    def poll(self):
        """Return the latest due (request_id, row), or None before its deadline.

        Missed delivery slots are counted rather than burst-submitted. Acquisition
        timestamps inside the chosen row remain unchanged.
        """
        elapsed = self.monotonic() - self.started
        first = self.index
        while self.index < len(self.requests) and self.requests[self.index]['offset_sec'] <= elapsed:
            self.index += 1
        if first == self.index:
            return None
        selected = self.requests[self.index - 1]
        # 保留真实采集时钟；追赶投递不产生新帧，也不把旧观测改成当前时间。
        self.record({'event': 'replay_delivery', 'request_id': selected['request_id'],
                     'skipped_slots': self.index - first - 1,
                     'delivery_lag_sec': elapsed - selected['offset_sec'],
                     'observation_timestamp_sec': selected['row']['observation_timestamp_sec']})
        return selected['request_id'], selected['row']

    def exhausted(self):
        """Return whether the original recorded time interval has ended."""
        return self.monotonic() - self.started >= self.end_offset

    def close(self):
        """Release no resources: rows are in memory and media belongs to the caller."""


def create_source(*, options, record):
    """Load prepared service rows for a configured deployment application.

    Args:
        options: requests_jsonl, source_start_sec and end_offset_sec.
        record: Shared deployment event callback.
    Returns:
        Paced replay source with no model or hardware dependencies.
    Raises:
        ValueError: JSON or scheduled interval is invalid.
        OSError: The prepared request file cannot be read.
    """
    with Path(options['requests_jsonl']).open() as stream:
        requests = [json.loads(line) for line in stream if line.strip()]
    return ReplaySource(requests, options['source_start_sec'], options['end_offset_sec'], record)
