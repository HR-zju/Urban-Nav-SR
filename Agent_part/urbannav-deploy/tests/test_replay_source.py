"""Verify new replay delivery behavior without sleeping or running a model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbannav_deploy.replay_source import ReplaySource


def test_pacing_skip_and_original_source_clock():
    clock, events = [1000.], []
    requests = [{'request_id': str(i), 'offset_sec': i * .2,
                 'row': {'observation_timestamp_sec': 10 + i * .2}}
                for i in range(4)]
    source = ReplaySource(requests, 10., 1., events.append, lambda: clock[0])
    source.start()
    assert source.poll() == ('0', requests[0]['row'])
    assert source.poll() is None
    clock[0] = 1000.61
    assert source.poll() == ('3', requests[3]['row'])
    assert events[-1]['skipped_slots'] == 2
    assert requests[3]['row']['observation_timestamp_sec'] == 10.6
    assert abs(source.now() - 10.61) < 1e-9
    assert not source.exhausted()
    clock[0] = 1001.
    assert source.exhausted()
    assert source.poll() is None
    source.close()
