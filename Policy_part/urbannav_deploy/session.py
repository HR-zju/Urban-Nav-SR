"""Bounded asynchronous model session with an independent control boundary.

Model transport, request construction, acquisition and vehicle tracking remain
replaceable. This coordinator never computes wheel commands or assumes odometry.
"""
from __future__ import annotations
from collections import deque
import threading
import time
from typing import Protocol

from .transport import InferenceClient


class ControlAdapter(Protocol):
    """Vehicle integration owns odometry, frame transforms and speed limits."""
    def track(self, prediction: dict, source_now_sec: float) -> None:
        """Track a valid plan using current odometry, or stop if unavailable."""
        ...

    def safe_stop(self, reason: str) -> None:
        """Revoke autonomous output and request a zero-speed command."""
        ...


class DeploymentSession:
    """Own one latest-only pending request and an independent control watchdog.

    Args:
        client: Pluggable model transport with finite operation timeouts.
        controller: Vehicle adapter; recording adapters can provide shadow mode.
        max_plan_age_sec: Maximum source age allowed for a completed plan.
        record: Nonblocking telemetry callback owned by the application.

    The application invokes tick on its control clock, separate from submit.
    Ownership is explicitly supplied on every tick and defaults to disabled.
    """
    def __init__(self, client: InferenceClient, controller: ControlAdapter,
                 max_plan_age_sec: float, record=lambda event: None):
        if not 0 < max_plan_age_sec < float('inf'):
            raise ValueError('max_plan_age_sec must be finite and positive')
        self.client, self.controller, self.record = client, controller, record
        self.max_age = max_plan_age_sec
        self._condition = threading.Condition()
        self._control_lock = threading.Lock()
        self._pending = deque(maxlen=1)
        self._latest = None
        self._failure = None
        self._closed = False
        self._last_stamp = -float('inf')
        self._worker = None

    def start(self):
        """Check readiness and start one worker; never create a second worker."""
        with self._condition:
            if self._worker is not None or self._closed:
                raise RuntimeError('session cannot be restarted')
            health = self.client.health()
            if not health['ready'] or health['busy']:
                raise RuntimeError('model unavailable')
            self._worker = threading.Thread(target=self._run, name='urbannav-inference', daemon=True)
            self._worker.start()
        self.controller.safe_stop('waiting_for_plan')

    def submit(self, request_id, row, frames=None):
        """Replace unstarted work with newer input; reject reordered observations.

        Args:
            request_id: Unique prediction identity.
            row: Current observation and optional retained query.
            frames: Optional compressed bytes mapped by content-addressed frame references.
        Returns:
            The id dropped from the one-slot queue, or None.
        Raises:
            RuntimeError: Session is closed, not started, or model failed.
            ValueError: Observation clock is nonfinite or not strictly newer.
        """
        import math
        stamp = float(row['observation_timestamp_sec'])
        with self._condition:
            if self._closed or self._worker is None or self._failure:
                raise RuntimeError('session is not accepting observations')
            if not math.isfinite(stamp) or stamp <= self._last_stamp:
                raise ValueError('observations must have increasing finite source timestamps')
            self._last_stamp = stamp
            dropped = self._pending[0][0] if self._pending else None
            self._pending.append((request_id, row, frames))
            self._condition.notify()
        if dropped:
            self.record({'event': 'dropped_pending', 'request_id': dropped})
        return dropped

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending)
                if self._closed:
                    return
                request_id, row, frames = self._pending.popleft()
            started = time.monotonic()
            try:
                result = (self.client.predict(request_id, row) if frames is None else
                          self.client.predict(request_id, row, frames=frames))
                if result['request_id'] != request_id or result['timestamp_sec'] != row['observation_timestamp_sec']:
                    raise ValueError('model response identity/clock mismatch')
                with self._condition:
                    if not self._closed:
                        self._latest = result
                self.record({'event': 'prediction', 'request_id': request_id,
                             'seconds': time.monotonic() - started})
            except Exception as exc:
                with self._condition:
                    self._failure = repr(exc)
                    self._pending.clear()
                self.record({'event': 'model_failure', 'request_id': request_id, 'error': repr(exc)})
                return

    def tick(self, source_now_sec, *, autonomy_owned=False):
        """Run a nonblocking control tick, stopping on ownership, clock or model faults.

        Args:
            source_now_sec: Current control time mapped into the acquisition clock.
            autonomy_owned: Explicit, current vehicle ownership decision.
        Returns:
            Gate state: tracking or the reason for requesting a stop.
        """
        with self._control_lock:
            return self._tick(source_now_sec, autonomy_owned)

    def _tick(self, source_now_sec, autonomy_owned):
        # 与 close 的 stop 串行化，禁止退出停车后再发出旧 tick 的 track。
        import math
        with self._condition:
            prediction, failure, closed = self._latest, self._failure, self._closed
        age = source_now_sec - prediction['timestamp_sec'] if prediction else None
        if closed:
            reason = 'session_closed'
        elif failure:
            reason = 'model_failure'
        elif not autonomy_owned:
            reason = 'ownership_not_held'
        elif prediction is None:
            reason = 'waiting_for_plan'
        elif not math.isfinite(source_now_sec) or age < 0:
            reason = 'invalid_clock'
        elif age > self.max_age:
            reason = 'stale_plan'
        else:
            try:
                self.controller.track(prediction, source_now_sec)
            except Exception:
                self.controller.safe_stop('controller_failure')
                raise
            return 'tracking'
        self.controller.safe_stop(reason)
        return reason

    def close(self, timeout=5.0):
        """Stop control immediately and join worker within a bounded timeout.

        Returns:
            True only when the worker is terminal. False means an outstanding
            inference must be inspected/retrieved; it must not be reissued.
        """
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()
        with self._control_lock:
            self.controller.safe_stop('session_exit')
        if self._worker is not None:
            self._worker.join(timeout)
        return self._worker is None or not self._worker.is_alive()
