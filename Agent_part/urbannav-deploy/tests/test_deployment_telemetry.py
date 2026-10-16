"""Trace acquisition/control clocks and model output without hardware or inference."""
import io
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbannav_deploy.application import DeploymentApplication, ShadowController
from urbannav_deploy.cli import EventWriter
from urbannav_deploy.session import DeploymentSession


def test_event_clock_is_enqueue_time_even_when_writer_is_blocked():
    entered, release = threading.Event(), threading.Event()

    class SlowStream(io.StringIO):
        def write(self, value):
            entered.set()
            if not release.wait(2):
                raise TimeoutError('fixture writer release')
            return super().write(value)

    stream = SlowStream()
    writer = EventWriter(stream)
    original = {'event': 'control_tick', 'source_now_sec': 12.5}
    before = time.monotonic()
    writer.record(original)
    after_enqueue = time.monotonic()
    try:
        assert entered.wait(1)
    finally:
        release.set()
        status = writer.close()
    assert status == {'terminal': True, 'dropped': 0, 'error': None}
    recorded = json.loads(stream.getvalue())
    assert before <= recorded['recorded_monotonic_sec'] <= after_enqueue
    assert recorded['source_now_sec'] == 12.5
    assert 'recorded_monotonic_sec' not in original


def test_shadow_trace_preserves_prediction_and_distinct_clock_domains():
    complete = threading.Event()
    stream = io.StringIO()
    writer = EventWriter(stream)
    prediction = {
        'request_id': 'shadow-one', 'timestamp_sec': 10.,
        'trajectory_ego_t': [[1., 0., 0.]] * 32,
        'metadata': {'query_source_timestamp_sec': 9.8,
                     'query_profile': {'reused_query': True, 'completed_perf_counter_sec': None},
                     'query_schedule': {'reason': 'cache_reused', 'query_age_sec': .2},
                     'trajectory_completed_perf_counter_sec': 1000000.},
    }

    class Client:
        def health(self):
            return {'ready': True, 'busy': False}

        def predict(self, request_id, row):
            return prediction

    class Source:
        submitted = False

        def start(self):
            pass

        def poll(self):
            if not self.submitted:
                self.submitted = True
                return 'shadow-one', {'observation_timestamp_sec': 10.}

        def now(self):
            return 10.1

        def exhausted(self):
            return complete.is_set()

        def close(self):
            pass

    def record(event):
        writer.record(event)
        if event['event'] == 'prediction':
            complete.set()

    session = DeploymentSession(Client(), ShadowController(record), 1., record)
    app = DeploymentApplication(Source(), session, ownership=lambda: True, record=record,
                                wait=lambda _: complete.wait(1))
    try:
        assert app.run()['inference_terminal']
    finally:
        status = writer.close()
    assert status == {'terminal': True, 'dropped': 0, 'error': None}
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    emitted = [e for e in events if e['event'] == 'prediction']
    assert len(emitted) == 1
    assert emitted[0]['prediction'] == prediction
    ticks = [e for e in events if e['event'] == 'control_tick']
    assert ticks
    assert all(e['source_now_sec'] == 10.1 and e['autonomy_owned'] for e in ticks)
    assert all('recorded_monotonic_sec' in e for e in events)
