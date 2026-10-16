"""New offline metrics only; no model, network or hardware calls."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbannav_deploy.telemetry import clock_rate, summarize


def prediction(local, server, query, reused=False, pid=1):
    return {'event': 'prediction', 'recorded_monotonic_sec': local,
            'prediction': {'timestamp_sec': 10.2, 'metadata': {
                'pid': pid, 'model_instance_id': 99,
                'query_source_timestamp_sec': 10.,
                'query_profile': {'reused_query': reused,
                                  'completed_perf_counter_sec': query},
                'trajectory_completed_perf_counter_sec': server}}}


def test_independent_clocks_reuse_and_gates():
    events = [prediction(2., 1000., 999.), prediction(2.2, 1000.2, None, True),
              prediction(2.5, 1000.5, 999.5)]
    events += [{'event': 'control_tick', 'recorded_monotonic_sec': t,
                'state': state, 'schedule_lag_sec': .01}
               for t, state in [(1., 'stale_plan'), (1.05, 'ownership_not_held'),
                                (1.1, 'tracking')]]
    events += [{'event': 'model_failure'}, {'event': 'dropped_pending'}]
    report = summarize(events)
    assert report['local_monotonic']['control']['hz'] == pytest.approx(20)
    server = report['server_perf_counter'][0]
    assert server['query']['hz'] == 2
    assert server['trajectory']['hz'] == 4
    assert server['trajectory']['max_interval_sec'] == pytest.approx(.3)
    assert server['reused_queries'] == 1
    assert report['query_source_age']['max_sec'] == pytest.approx(.2)
    assert report['control_states']['stale_plan'] == 1
    assert report['event_counts']['model_failure'] == 1
    assert report['log_completeness'].startswith('unknown')


def test_restart_missing_identity_and_sparse_evidence():
    first = prediction(2, 1000, 999)
    restarted = prediction(3, 1, .5, pid=2)
    missing = prediction(4, 2, 1.5)
    del missing['prediction']['metadata']['pid']
    result = summarize([first, restarted, missing])
    assert len(result['server_perf_counter']) == 2
    assert all(x['trajectory']['hz'] is None for x in result['server_perf_counter'])
    assert result['missing_server_identity'] == 1
    assert clock_rate([])['hz'] is None


@pytest.mark.parametrize('stamps', [[1, 1], [2, 1], [float('nan')], [float('inf')]])
def test_invalid_clock_is_not_reported_as_rate(stamps):
    with pytest.raises(ValueError):
        clock_rate(stamps)


def test_cli_reads_jsonl_and_writes_summary(tmp_path):
    source, output = tmp_path / 'events.jsonl', tmp_path / 'summary.json'
    source.write_text(json.dumps(prediction(2, 1000, 999)) + '\n')
    subprocess.run([sys.executable, '-m', 'urbannav_deploy.telemetry', '--events',
                    str(source), '--output', str(output)],
                   cwd=Path(__file__).resolve().parents[1], check=True)
    assert json.loads(output.read_text())['event_counts'] == {'prediction': 1}
