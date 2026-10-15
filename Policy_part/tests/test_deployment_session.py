"""Meaningful queue, ownership and fault boundary tests without a model/vehicle."""
import sys
import threading
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbannav_deploy.session import DeploymentSession


class Client:
    def __init__(self):
        self.started, self.release, self.finished = threading.Event(), threading.Event(), threading.Event()
        self.calls = []

    def health(self):
        return {'ready': True, 'busy': False}

    def predict(self, request_id, row):
        self.calls.append(request_id)
        self.started.set()
        if not self.release.wait(2):
            raise TimeoutError('fixture release')
        return {'request_id': request_id, 'timestamp_sec': row['observation_timestamp_sec']}


class Controller:
    def __init__(self):
        self.events = []

    def safe_stop(self, reason):
        self.events.append(reason)

    def track(self, prediction, source_now_sec):
        self.events.append('track')


class SessionTests(unittest.TestCase):
    def test_memory_frame_bytes_reach_inference_worker(self):
        client, control, completed = Client(), Controller(), threading.Event()
        seen = []

        def predict(request_id, row, *, frames):
            seen.append(frames)
            return {'request_id': request_id, 'timestamp_sec': row['observation_timestamp_sec']}

        client.predict = predict
        session = DeploymentSession(client, control, 1, lambda event: completed.set()
                                    if event['event'] == 'prediction' else None)
        session.start()
        frames = {'frame:fixture': b'compressed image'}
        try:
            session.submit('memory', {'observation_timestamp_sec': 1.}, frames)
            self.assertTrue(completed.wait(1))
            self.assertEqual(seen, [frames])
        finally:
            self.assertTrue(session.close())

    def test_latest_queue_control_watchdog_and_shutdown(self):
        client, control, completed = Client(), Controller(), threading.Event()
        session = DeploymentSession(client, control, 1, lambda e: completed.set()
                                    if e['event'] == 'prediction' and e['request_id'] == 'c' else None)
        session.start()
        session.submit('a', {'observation_timestamp_sec': 1.0})
        self.assertTrue(client.started.wait(1))
        session.submit('b', {'observation_timestamp_sec': 2.0})
        self.assertEqual(session.submit('c', {'observation_timestamp_sec': 3.0}), 'b')
        self.assertEqual(session.tick(3.0), 'ownership_not_held')
        client.release.set()
        self.assertTrue(completed.wait(1))
        self.assertEqual(client.calls, ['a', 'c'])
        self.assertEqual(session.tick(3.5, autonomy_owned=True), 'tracking')
        self.assertEqual(session.tick(4.1, autonomy_owned=True), 'stale_plan')
        self.assertEqual(session.tick(2.0, autonomy_owned=True), 'invalid_clock')
        self.assertTrue(session.close())
        self.assertEqual(control.events[-1], 'session_exit')

    def test_inflight_close_reports_nonterminal_without_retry(self):
        client, control = Client(), Controller()
        session = DeploymentSession(client, control, 1)
        session.start()
        session.submit('a', {'observation_timestamp_sec': 1.0})
        self.assertTrue(client.started.wait(1))
        self.assertFalse(session.close(timeout=.001))
        self.assertEqual(session.tick(1.0, autonomy_owned=True), 'session_closed')
        client.release.set()
        self.assertTrue(session.close())
        self.assertEqual(client.calls, ['a'])

class ShutdownOrderingTests(unittest.TestCase):
    def test_close_never_finishes_stop_before_inflight_control_tick(self):
        client = Client()
        client.release.set()
        entered, release, predicted = threading.Event(), threading.Event(), threading.Event()

        class BlockingController(Controller):
            def track(self, prediction, source_now_sec):
                entered.set()
                release.wait(1)
                self.events.append('track')

        control = BlockingController()
        session = DeploymentSession(client, control, 1, lambda event: predicted.set()
                                    if event['event'] == 'prediction' else None)
        session.start()
        session.submit('a', {'observation_timestamp_sec': 1.0})
        self.assertTrue(predicted.wait(1))
        tick = threading.Thread(target=lambda: session.tick(1.1, autonomy_owned=True))
        tick.start()
        self.assertTrue(entered.wait(1))
        closed = threading.Event()
        closer = threading.Thread(target=lambda: (session.close(), closed.set()))
        closer.start()
        self.assertFalse(closed.wait(.01))
        release.set()
        tick.join(1)
        closer.join(1)
        self.assertTrue(closed.is_set())
        self.assertEqual(control.events[-2:], ['track', 'session_exit'])


if __name__ == '__main__':
    unittest.main()
