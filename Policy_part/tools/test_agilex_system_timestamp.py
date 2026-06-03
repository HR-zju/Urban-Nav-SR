#!/usr/bin/env python3
# Probe AgileX WebSocket /system_timestamp and print request/response timing clearly.

import argparse
import asyncio
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import websockets
except ModuleNotFoundError:
    websockets = None

DEFAULT_URI = 'ws://192.168.1.102:9090'
DEFAULT_TOPIC = '/system_timestamp'


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def stamp_to_float(stamp: dict[str, Any]) -> Optional[float]:
    sec = stamp.get('secs', stamp.get('sec'))
    nsec = stamp.get('nsecs', stamp.get('nanosec'))
    if sec is None or nsec is None:
        return None
    try:
        return float(sec) + float(nsec) * 1e-9
    except (TypeError, ValueError):
        return None


def format_epoch(epoch_sec: Optional[float]) -> str:
    if epoch_sec is None:
        return '<missing>'
    dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
    return dt.isoformat(timespec='microseconds').replace('+00:00', 'Z')


def print_timestamp_message(data: dict[str, Any], local_receive_epoch: float, latency_ms: float) -> None:
    msg = data.get('msg') if isinstance(data.get('msg'), dict) else {}
    stamp = msg.get('stamp') if isinstance(msg.get('stamp'), dict) else {}
    stamp_sec = stamp.get('secs', stamp.get('sec'))
    stamp_nsec = stamp.get('nsecs', stamp.get('nanosec'))
    stamp_epoch = stamp_to_float(stamp)
    offset_ms = None if stamp_epoch is None else (local_receive_epoch - stamp_epoch) * 1000.0

    print()
    print('========== SYSTEM TIMESTAMP ==========')
    topic_value = data.get('topic')
    op_value = data.get('op')
    frame_id_value = msg.get('frame_id', '')
    seq_value = msg.get('seq', '')
    print(f'topic                    : {topic_value}')
    print(f'op                       : {op_value}')
    print(f'frame_id                 : {frame_id_value!r}')
    print(f'seq                      : {seq_value}')
    print()
    print('stamp for ROS header')
    print(f'  secs                   : {stamp_sec}')
    print(f'  nsecs                  : {stamp_nsec}')
    if stamp_epoch is None:
        print('  epoch_sec              : <missing>')
    else:
        print(f'  epoch_sec              : {stamp_epoch:.9f}')
    print(f'  utc                    : {format_epoch(stamp_epoch)}')
    print()
    print('local receive timing')
    print(f'  local_epoch_sec        : {local_receive_epoch:.9f}')
    print(f'  local_utc              : {format_epoch(local_receive_epoch)}')
    print(f'  request_to_receive_ms  : {latency_ms:.3f}')
    if offset_ms is None:
        print('  local_minus_stamp_ms   : <missing>')
    else:
        print(f'  local_minus_stamp_ms   : {offset_ms:.3f}')
    print('======================================')


async def probe(args: argparse.Namespace) -> int:
    if websockets is None:
        print('Missing Python package websockets. Run in foundation_nav or install websockets.', file=sys.stderr)
        return 2

    request = {
        'op': 'subscribe',
        'topic': args.topic,
    }

    connect_kwargs: dict[str, Any] = {}
    if 'proxy' in inspect.signature(websockets.connect).parameters:
        connect_kwargs['proxy'] = None

    print('WebSocket URI:')
    print(f'  {args.uri}')
    print()
    print('Subscribe request JSON sent to AgileX:')
    print(pretty_json(request))

    received = 0
    async with websockets.connect(args.uri, ping_interval=None, **connect_kwargs) as websocket:
        request_sent_monotonic = time.monotonic()
        await websocket.send(json.dumps(request))

        while received < args.count:
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            except asyncio.TimeoutError:
                print()
                print(f'Timeout: did not receive publish data from {args.topic} within {args.timeout:.1f}s.')
                return 1

            local_receive_epoch = time.time()
            latency_ms = (time.monotonic() - request_sent_monotonic) * 1000.0

            print()
            print('Raw WebSocket response:')
            print(raw)

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                print('Skip: response is not JSON.')
                continue

            print()
            print('Parsed response JSON:')
            print(pretty_json(data))

            if data.get('op') != 'publish' or data.get('topic') != args.topic:
                print(f'Skip: not target publish message for {args.topic}.')
                continue

            received += 1
            print_timestamp_message(data, local_receive_epoch, latency_ms)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Subscribe AgileX /system_timestamp over WebSocket and print request/response details.'
    )
    parser.add_argument('--uri', default=DEFAULT_URI, help=f'AgileX WebSocket URI, default: {DEFAULT_URI}')
    parser.add_argument('--topic', default=DEFAULT_TOPIC, help=f'Timestamp topic, default: {DEFAULT_TOPIC}')
    parser.add_argument('--count', type=int, default=1, help='Number of timestamp messages to print, default: 1')
    parser.add_argument('--timeout', type=float, default=10.0, help='Receive timeout in seconds, default: 10')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(probe(args)))


if __name__ == '__main__':
    main()
