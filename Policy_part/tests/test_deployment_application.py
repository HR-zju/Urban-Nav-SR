"""Application boundaries use fake acquisition/transport, never GPU or hardware."""
import io
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbannav_deploy.application import DeploymentApplication, ShadowController
from urbannav_deploy.cli import EventWriter, main, run_config
from urbannav_deploy.session import DeploymentSession


class Source:
    def __init__(self):
        self.polls = 0
        self.closed = 0

    def start(self):
        pass

    def poll(self):
        self.polls += 1
        return ('one', {'observation_timestamp_sec': 0.}) if self.polls == 1 else None

    def now(self):
        return self.polls / 20

    def exhausted(self):
        return self.polls == 5

    def close(self):
        self.closed += 1


class Client:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = 0

    def health(self):
        return {'ready': True, 'busy': False}

    def predict(self, request_id, row):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(2):
            raise TimeoutError('fixture blocked')
        return {'request_id': request_id, 'timestamp_sec': row['observation_timestamp_sec']}


class ApplicationTests(unittest.TestCase):
    def test_control_continues_during_inflight_and_acquisition_gap(self):
        source, client, events = Source(), Client(), []
        session = DeploymentSession(client, ShadowController(events.append), .75, events.append)
        clock = [0.]

        def wait(delay):
            self.assertTrue(client.entered.wait(1))
            clock[0] += delay

        app = DeploymentApplication(source, session, record=events.append,
                                    monotonic=lambda: clock[0], wait=wait, close_timeout_sec=0)
        try:
            result = app.run()
            self.assertEqual(result, {'control_ticks': 5, 'skipped_control_slots': 0,
                                      'inference_terminal': False})
            self.assertEqual(source.closed, 1)
            self.assertEqual(client.calls, 1)
            self.assertEqual([e['state'] for e in events if e['event'] == 'control_tick'],
                             ['ownership_not_held'] * 5)
        finally:
            client.release.set()
            session.close()

    def test_failure_is_reported_without_ownership(self):
        source, client, failed, events = Source(), Client(), threading.Event(), []

        def predict(*args):
            raise ConnectionError('fixture disconnect')

        client.predict = predict
        def record(event):
            events.append(event)
            if event['event'] == 'model_failure':
                failed.set()

        session = DeploymentSession(client, ShadowController(record), .75, record)
        app = DeploymentApplication(source, session, wait=lambda _: failed.wait(1))
        with self.assertRaisesRegex(RuntimeError, 'model worker failed'):
            app.run()
        self.assertEqual(source.closed, 1)
        self.assertIn('model_failure', [e.get('reason') for e in events])

    def test_startup_failure_closes_source_and_stops(self):
        source, client, events = Source(), Client(), []
        client.health = lambda: {'ready': False, 'busy': False}
        session = DeploymentSession(client, ShadowController(events.append), .75)
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            DeploymentApplication(source, session).run()
        self.assertEqual(source.closed, 1)
        self.assertEqual(events[-1]['reason'], 'session_exit')

    def test_check_does_not_import_site_adapters(self):
        path = Path(__file__).resolve().parents[1] / 'configs/deployment.shadow.example.json'
        with patch('urbannav_deploy.configuration.importlib.import_module',
                   side_effect=AssertionError('must not import')), patch('sys.stdout', io.StringIO()):
            self.assertEqual(main(['--config', str(path), '--check']), 0)

    def test_partial_construction_closes_source(self):
        source, client = Source(), Client()
        config = {'mode': 'live', **{key: {'factory': 'fake:create'} for key in
                                   ('source', 'inference', 'controller', 'ownership')}}
        with patch('urbannav_deploy.cli.load_adapter',
                   side_effect=[client, source, RuntimeError('driver unavailable')]):
            with self.assertRaisesRegex(RuntimeError, 'driver unavailable'):
                run_config(config, lambda event: None)
        self.assertEqual(source.closed, 1)

    def test_late_control_tick_skips_slots_without_burst(self):
        source, client, events = Source(), Client(), []
        source.poll = lambda: None
        clock = [0.]
        source.exhausted = lambda: len([e for e in events if e['event'] == 'control_tick']) == 2

        def ownership():
            if clock[0] == 0.:
                clock[0] = .21
            return False

        session = DeploymentSession(client, ShadowController(events.append), .75)
        result = DeploymentApplication(
            source, session, ownership=ownership, record=events.append,
            monotonic=lambda: clock[0], wait=lambda delay: clock.__setitem__(0, clock[0] + delay)).run()
        self.assertEqual(result['control_ticks'], 2)
        self.assertEqual(result['skipped_control_slots'], 4)
        self.assertAlmostEqual(clock[0], .25)
        self.assertEqual(client.calls, 0)

    def test_shadow_composition_closes_source_once_without_predicting(self):
        source, client, events = Source(), Client(), []
        source.poll = lambda: None
        source.exhausted = lambda: True
        config = {key: {'factory': 'fake:create'} for key in ('source', 'inference')}
        with patch('urbannav_deploy.cli.load_adapter', side_effect=[client, source]) as loader:
            result = run_config(config, events.append)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(source.closed, 1)
        self.assertEqual(client.calls, 0)
        self.assertTrue(result['inference_terminal'])

    def test_telemetry_io_failure_is_visible(self):
        class Broken:
            def write(self, value):
                raise OSError('disk full')
        writer = EventWriter(Broken())
        writer.record({'event': 'fixture'})
        result = writer.close()
        self.assertTrue(result['terminal'])
        self.assertIn('disk full', result['error'])


if __name__ == '__main__':
    unittest.main()
