"""Summarize one deployment event stream without mixing host clocks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def clock_rate(stamps):
    """Compute (N-1)/(last-first) in a single clock domain.

    Args:
        stamps: Ordered completion timestamps in seconds.
    Returns:
        Sample count, average Hz and maximum interval; no rate for N < 2.
    Raises:
        ValueError: Timestamps are nonfinite or not strictly increasing.
    """
    values = [float(value) for value in stamps]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('nonfinite event timestamp')
    gaps = [b - a for a, b in zip(values, values[1:])]
    if any(gap <= 0 for gap in gaps):
        raise ValueError('clock must increase within one stream/session')
    return {'samples': len(values),
            'hz': (len(values) - 1) / (values[-1] - values[0]) if gaps else None,
            'max_interval_sec': max(gaps) if gaps else None}


def summarize(events):
    """Report observed clocks, gates and failures, not physical acceptance.

    Input must describe one client session. Server clocks are grouped by process
    and model identity. Missing server identity/timestamps remain explicit.
    Event files alone cannot establish writer completion or absence of drops.

    Args:
        events: Parsed JSONL event dictionaries for one client session.
    Returns:
        Observed rates, query ages, gate counts and explicit evidence limits.
    Raises:
        ValueError: Invalid clocks, ages, lags or multiple session closures.
        KeyError: Required event fields are absent.
    """
    counts, states, instances = Counter(), Counter(), {}
    controls, deliveries, ages, lags = [], [], [], []
    missing_identity = missing_local_clock = 0
    closures = []
    for event in events:
        kind = event['event']
        counts[kind] += 1
        if kind in ('control_tick', 'prediction'):
            stamp = event.get('recorded_monotonic_sec')
            if stamp is None:
                missing_local_clock += 1
            else:
                (controls if kind == 'control_tick' else deliveries).append(stamp)
        if kind == 'application_closed':
            closures.append(event)
        if kind == 'control_tick':
            states[event['state']] += 1
            lags.append(float(event['schedule_lag_sec']))
        if kind != 'prediction':
            continue
        prediction = event['prediction']
        meta = prediction['metadata']
        query_stamp = meta.get('query_source_timestamp_sec')
        if query_stamp is not None:
            age = float(prediction['timestamp_sec']) - float(query_stamp)
            if not math.isfinite(age) or age < 0:
                raise ValueError('invalid query source age')
            ages.append(age)
        if meta.get('pid') is None or meta.get('model_instance_id') is None:
            missing_identity += 1
            continue
        key = (meta['pid'], meta['model_instance_id'])
        group = instances.setdefault(key, {'query': [], 'trajectory': [],
                                          'reused_queries': 0, 'responses': 0})
        group['responses'] += 1
        profile = meta.get('query_profile') or {}
        reused = profile.get('reused_query')
        group['reused_queries'] += int(reused is True)
        if reused is False and profile.get('completed_perf_counter_sec') is not None:
            group['query'].append(profile['completed_perf_counter_sec'])
        if meta.get('trajectory_completed_perf_counter_sec') is not None:
            group['trajectory'].append(meta['trajectory_completed_perf_counter_sec'])
    if len(closures) > 1:
        raise ValueError('expected one client session, found multiple application closures')
    if any(not math.isfinite(lag) or lag < 0 for lag in lags):
        raise ValueError('invalid control schedule lag')
    return {
        'event_counts': dict(counts), 'control_states': dict(states),
        'local_monotonic': {'control': clock_rate(controls),
                            'prediction_delivery': clock_rate(deliveries)},
        'server_perf_counter': [
            {'pid': key[0], 'model_instance_id': key[1],
             'responses': group['responses'], 'reused_queries': group['reused_queries'],
             'query': clock_rate(group['query']), 'trajectory': clock_rate(group['trajectory'])}
            for key, group in instances.items()],
        'query_source_age': {'samples': len(ages), 'max_sec': max(ages) if ages else None},
        'max_control_schedule_lag_sec': max(lags) if lags else None,
        'missing_server_identity': missing_identity,
        'missing_local_clock': missing_local_clock,
        'application_closed': closures[0] if closures else None,
        'log_completeness': 'unknown: inspect separate telemetry_closed writer status',
        'physical_acceptance': 'not established by event statistics',
    }


def main():
    """Read an event file and write its summary; propagate malformed input errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.events.open() as stream:
        result = summarize(json.loads(line) for line in stream if line.strip())
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
