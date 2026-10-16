"""Configuration-driven deployment entry point; imports no robot SDK by default."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import queue
import signal
import sys
import threading
import time

from .application import DeploymentApplication, ShadowController
from .configuration import load_adapter, validate_config
from .session import DeploymentSession


class EventWriter:
    """Write bounded telemetry outside the control thread.

    Args:
        stream: Writable text stream owned by the caller.
        capacity: Maximum buffered events; overflow is counted, never blocks control.
    """
    def __init__(self, stream, capacity=2048):
        self.stream = stream
        self.events = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._write, name='urbannav-events', daemon=True)
        self.thread.start()

    def record(self, event):
        """Enqueue a JSON-compatible event without waiting for output IO.

        Args:
            event: Immutable event mapping; callers must not mutate after enqueue.
        """
        try:
            # 在事件产生端计时，后台落盘延迟不计入控制/轨迹完成间隔。
            self.events.put_nowait({**event, 'recorded_monotonic_sec': time.monotonic()})
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            while not self.stopping.is_set() or not self.events.empty():
                try:
                    event = self.events.get(timeout=.05)
                except queue.Empty:
                    continue
                self.stream.write(json.dumps(event, allow_nan=False) + '\n')
                self.stream.flush()
        except Exception as exc:
            self.error = repr(exc)

    def close(self, timeout=2.):
        """Drain for a bounded time and report loss or a blocked writer.

        Args:
            timeout: Maximum seconds waiting for the output thread.
        Returns:
            Writer terminal state, dropped event count and optional IO error.
        """
        self.stopping.set()
        self.thread.join(timeout)
        return {'terminal': not self.thread.is_alive(), 'dropped': self.dropped, 'error': self.error}


def run_config(config, record):
    """Construct configured adapters, handle signals and run one application.

    Args:
        config: Trusted deployment configuration; factory imports execute local code.
        record: Nonblocking telemetry callback.
    Returns:
        Application termination counters; a nonterminal worker is not successful shutdown.
    Raises:
        ValueError: Configuration is invalid.
        Exception: Adapter construction, operation or cleanup fails.

    Factories must be inert constructors. Acquisition begins only in source.start;
    ownership providers and controller calls must be nonblocking. Factory-owned
    resources require a close method. ExitStack closes them even after partial setup.
    """
    validate_config(config)
    with ExitStack() as stack:
        def construct(key):
            adapter = load_adapter(config[key], record)
            if callable(getattr(adapter, 'close', None)):
                stack.callback(adapter.close)
            return adapter

        client = construct('inference')
        source = load_adapter(config['source'], record)
        # The application owns source teardown once started; cover partial construction too.
        application_owns_source = False
        stack.callback(lambda: None if application_owns_source else source.close())
        if config.get('mode', 'shadow') == 'shadow':
            controller = ShadowController(record)
            ownership = lambda: True  # Only allows recording candidates; no driver is constructed.
        else:
            controller = construct('controller')
            stack.callback(controller.safe_stop, 'configuration_exit')
            ownership = construct('ownership')
            if not callable(ownership):
                raise TypeError('ownership adapter must be callable')
        session = DeploymentSession(client, controller, config.get('max_plan_age_sec', .75), record)
        app = DeploymentApplication(source, session, control_hz=config.get('control_hz', 20.),
                                    ownership=ownership, record=record,
                                    close_timeout_sec=config.get('close_timeout_sec', 5.))
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(signum, lambda _signum, _frame: app.stop())
            stack.callback(signal.signal, signum, previous)
        application_owns_source = True
        return app.run()


def main(argv=None):
    """Validate configuration or run the explicit deployment composition.

    Args:
        argv: Optional CLI argument list; defaults to process arguments.
    Returns:
        Zero on success; two for incomplete shutdown or telemetry loss.
    Raises:
        Exception: Invalid configuration or original adapter/runtime failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true', help='validate only; no imports or connections')
    args = parser.parse_args(argv)
    with open(args.config) as stream:
        config = validate_config(json.load(stream))
    if args.check:
        print('configuration valid; adapters were not imported or contacted')
        return 0
    writer = EventWriter(sys.stdout)
    try:
        result = run_config(config, writer.record)
    finally:
        telemetry = writer.close()
        print(json.dumps({'event': 'telemetry_closed', **telemetry}), file=sys.stderr)
    return 0 if result['inference_terminal'] and telemetry == {
        'terminal': True, 'dropped': 0, 'error': None} else 2
