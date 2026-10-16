"""Causal multi-rate clip planning with explicit source and delivery clocks.

This module has no model, ROS or CUDA dependency. Dataset replay and live adapters
must preserve acquisition timestamps; a delivery tick never rewrites source time.
"""
from bisect import bisect_right
from collections import deque


def predecessor(items, stamp_ns):
    """Select the newest acquired item at or before the requested source time.

    Args:
        items: Timestamp-sorted immutable frame records.
        stamp_ns: Causal upper bound in source nanoseconds.
    Returns:
        Matching record, or None before the first observation.
    """
    index = bisect_right([item['timestamp_ns'] for item in items], stamp_ns) - 1
    return items[index] if index >= 0 else None


def validate_stream(items):
    """Reject ambiguous identities, invalid clocks and noncausal ordering.

    Args:
        items: Nonempty source frame records with timestamp_ns and path.
    Raises:
        ValueError: Timestamps or immutable identities are not unique and ordered.
    """
    if not items or any(type(x['timestamp_ns']) is not int for x in items):
        raise ValueError('stream needs integer nanosecond source timestamps')
    if any(a['timestamp_ns'] >= b['timestamp_ns'] for a, b in zip(items, items[1:])):
        raise ValueError('source timestamps must strictly increase')
    if len({x['path'] for x in items}) != len(items):
        raise ValueError('source paths must identify immutable distinct frames')


def plan_clip(fpv, bev, *, start_ns, end_ns, fpv_period_ns=200_000_000,
              bev_period_ns=500_000_000, prediction_period_ns=500_000_000,
              max_fpv_lag_ns=100_000_000, max_bev_age_ns=1_100_000_000):
    """Plan every delivery and prediction slot, including history warmup/rejects.

    Five consecutive FPV delivery samples form a window. BEV is delivered on its
    own clock and may only condition a window whose latest frame is not older
    than the map acquisition. No future lookup, padding or interpolated frames.

    Args:
        fpv: Sorted immutable FPV frame records.
        bev: Sorted immutable map frame records.
        start_ns: Inclusive source-clock start of the full clip.
        end_ns: Inclusive last timestamp, without extrapolating the source tail.
        fpv_period_ns: Acquisition delivery cadence, default 200 ms.
        bev_period_ns: Map delivery cadence, default 500 ms.
        prediction_period_ns: Model request cadence, default 500 ms.
        max_fpv_lag_ns: Allowed predecessor lag before clearing history.
        max_bev_age_ns: Allowed map age relative to the current FPV.
    Returns:
        Every scheduled delivery/prediction, with explicit rejects and clocks.
    Raises:
        ValueError: Source identity, timestamp ordering or cadence is invalid.
    """
    validate_stream(fpv)
    validate_stream(bev)
    periods = (fpv_period_ns, bev_period_ns, prediction_period_ns)
    if start_ns > end_ns or any(type(x) is not int or x <= 0 for x in periods):
        raise ValueError('invalid interval or cadence')
    events = []
    for kind, period in zip(('fpv', 'bev', 'predict'), periods):
        events.extend((tick, kind) for tick in range(start_ns, end_ns + 1, period))
    order = {'fpv': 0, 'bev': 1, 'predict': 2}
    history, delivered_maps, output = deque(maxlen=5), [], []
    for tick, kind in sorted(events, key=lambda x: (x[0], order[x[1]])):
        event = {'kind': kind, 'delivery_ns': tick, 'offset_sec': (tick - start_ns) / 1e9}
        if kind == 'fpv':
            item = predecessor(fpv, tick)
            if item is None or tick - item['timestamp_ns'] > max_fpv_lag_ns:
                history.clear()
                event.update(status='rejected', reason='missing_or_stale_fpv')
            elif history and item['timestamp_ns'] <= history[-1]['timestamp_ns']:
                history.clear()
                event.update(status='rejected', reason='repeated_fpv')
            else:
                history.append(item)
                event.update(status='delivered', frame=item)
        elif kind == 'bev':
            item = predecessor(bev, tick)
            if item is None:
                event.update(status='rejected', reason='missing_bev')
            else:
                if not delivered_maps or item['timestamp_ns'] > delivered_maps[-1]['timestamp_ns']:
                    delivered_maps.append(item)
                event.update(status='delivered', frame=item)
        elif len(history) < 5:
            event.update(status='rejected', reason='history_warmup_or_gap')
        else:
            current = history[-1]['timestamp_ns']
            image = predecessor(delivered_maps, current)
            if image is None or current - image['timestamp_ns'] > max_bev_age_ns:
                event.update(status='rejected', reason='missing_or_stale_bev')
            else:
                event.update(status='eligible', fpv=list(history), bev=image,
                             observation_ns=current, bev_age_sec=(current-image['timestamp_ns'])/1e9)
        output.append(event)
    return output
