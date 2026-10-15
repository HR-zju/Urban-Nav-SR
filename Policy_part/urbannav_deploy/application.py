"""Application lifecycle around independently replaceable deployment adapters."""
from __future__ import annotations

import math
import threading
import time
from typing import Protocol

from .session import DeploymentSession


class ObservationSource(Protocol):
    """Acquisition adapter with a nonblocking poll and a mapped source clock."""
    def start(self) -> None:
        """Start acquisition; failures must leave close safe to call."""
        ...

    def poll(self) -> tuple[str, dict] | tuple[str, dict, dict[str, bytes]] | None:
        """Return the newest (id, row[, frame_bytes]) request without waiting."""
        ...

    def now(self) -> float:
        """Return current time in the acquisition clock, including during outages."""
        ...

    def exhausted(self) -> bool:
        """Report terminal source exhaustion, not a temporary missing frame."""
        ...

    def close(self) -> None:
        """Stop acquisition and release source-owned resources."""
        ...


class ShadowController:
    """Record eligible plan candidates without constructing vehicle drivers.

    Args:
        record: Nonblocking telemetry callback.
    """
    def __init__(self, record):
        self.record = record

    def track(self, prediction, source_now_sec):
        """Record a plan identity and age; never compute or send wheel commands."""
        self.record({'event': 'shadow_candidate', 'request_id': prediction['request_id'],
                     'source_now_sec': source_now_sec,
                     'plan_age_sec': source_now_sec - prediction['timestamp_sec']})

    def safe_stop(self, reason):
        """Record a suppressed control tick or lifecycle stop."""
        self.record({'event': 'shadow_stop', 'reason': reason})


class DeploymentApplication:
    """Run bounded inference and independent control with explicit teardown.

    Args:
        source: Nonblocking acquisition adapter owning clock mapping and history.
        session: Existing inference/control session.
        control_hz: Control scheduler rate, distinct from model/source rates.
        ownership: Nonblocking current ownership provider, False by default.
        record: Nonblocking telemetry callback.
        close_timeout_sec: Maximum wait for an outstanding inference on exit.
        monotonic: Scheduler clock, injectable for deterministic CPU tests.
        wait: Interruptible scheduler wait; defaults to the stop event's wait.

    Source exhaustion cancels pending work and stops control; it does not drain
    old observations into the vehicle. Lossless model evaluation has a separate
    runner. An unfinished inference on exit is reported, never silently retried.
    """
    def __init__(self, source: ObservationSource, session: DeploymentSession, *, control_hz=20.,
                 ownership=lambda: False, record=lambda event: None, close_timeout_sec=5.,
                 monotonic=time.monotonic, wait=None):
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError('control_hz must be finite and positive')
        if not math.isfinite(close_timeout_sec) or close_timeout_sec < 0:
            raise ValueError('close timeout must be finite and nonnegative')
        self.source, self.session = source, session
        self.period = 1. / control_hz
        self.ownership, self.record = ownership, record
        self.close_timeout = close_timeout_sec
        self.monotonic = monotonic
        self.stop_event = threading.Event()
        self.wait = wait or self.stop_event.wait
        self._started = False

    def stop(self):
        """Request cooperative exit; safe for a signal handler or another thread."""
        self.stop_event.set()

    def run(self):
        """Start adapters once, run control ticks, and always stop before releasing source.

        Returns:
            Tick/deadline counts and whether the inference worker has terminated.
        Raises:
            RuntimeError: Application is reused, or an adapter/model reports failure.
            Exception: Original adapter failure after stopping and closing resources.
        """
        if self._started:
            raise RuntimeError('application is single-use')
        self._started = True
        ticks = skipped = 0
        terminal = False
        try:
            # 即使模型健康检查或采集start失败，也执行finally里的停车和源清理。
            self.session.start()
            self.source.start()
            deadline = self.monotonic()
            while not self.stop_event.is_set():
                request = self.source.poll()
                if request is not None:
                    self.session.submit(*request)
                state = self.session.tick(self.source.now(), autonomy_owned=bool(self.ownership()))
                self.record({'event': 'control_tick', 'state': state,
                             'schedule_lag_sec': max(0., self.monotonic() - deadline)})
                ticks += 1
                if state == 'model_failure':
                    raise RuntimeError('model worker failed; inspect original request without retry')
                if self.source.exhausted():
                    break
                deadline += self.period
                now = self.monotonic()
                if now > deadline:
                    missed = math.floor((now - deadline) / self.period) + 1
                    skipped += missed
                    deadline += missed * self.period
                self.wait(max(0., deadline - self.monotonic()))
        finally:
            try:
                terminal = self.session.close(timeout=self.close_timeout)
            finally:
                self.source.close()
                self.record({'event': 'application_closed', 'inference_terminal': terminal,
                             'control_ticks': ticks, 'skipped_control_slots': skipped})
        return {'control_ticks': ticks, 'skipped_control_slots': skipped,
                'inference_terminal': terminal}
