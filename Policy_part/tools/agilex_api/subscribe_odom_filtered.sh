#!/usr/bin/env bash
set -euo pipefail

URI="${1:-ws://192.168.1.102:9090}"
TOPIC="/odom"
TYPE="nav_msgs/Odometry"

/home/orin26/miniforge3/condabin/mamba run -n foundation_nav python -c '
import asyncio
import inspect
import json
import math
import sys

import websockets

uri = sys.argv[1]
topic = sys.argv[2]
msg_type = sys.argv[3]


def nested(data, *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def yaw_from_quaternion(q):
    x = as_float(q.get("x"))
    y = as_float(q.get("y"))
    z = as_float(q.get("z"))
    w = as_float(q.get("w"), 1.0)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def print_odom(data):
    msg = data.get("msg", {})
    stamp = nested(msg, "header", "stamp", default={}) or {}
    frame_id = nested(msg, "header", "frame_id", default="")
    child_frame_id = msg.get("child_frame_id", "")
    stamp_sec = stamp.get("secs", stamp.get("sec"))
    stamp_nsec = stamp.get("nsecs", stamp.get("nanosec"))

    position = nested(msg, "pose", "pose", "position", default={}) or {}
    orientation = nested(msg, "pose", "pose", "orientation", default={}) or {}
    linear = nested(msg, "twist", "twist", "linear", default={}) or {}
    angular = nested(msg, "twist", "twist", "angular", default={}) or {}

    x = as_float(position.get("x"))
    y = as_float(position.get("y"))
    z = as_float(position.get("z"))
    yaw = yaw_from_quaternion(orientation)
    vx = as_float(linear.get("x"))
    vy = as_float(linear.get("y"))
    vz = as_float(linear.get("z"))
    wz = as_float(angular.get("z"))
    topic_name = data.get("topic")
    qx = as_float(orientation.get("x"))
    qy = as_float(orientation.get("y"))
    qz = as_float(orientation.get("z"))
    qw = as_float(orientation.get("w"), 1.0)

    print()
    print("========== ODOM FILTERED ==========")
    print(f"topic          : {topic_name}")
    print(f"frame          : {frame_id} -> {child_frame_id}")
    print(f"stamp          : {stamp_sec}.{stamp_nsec}")
    print()
    print("pose")
    print(f"  position     : x={x:.6f} m, y={y:.6f} m, z={z:.6f} m")
    print(f"  yaw          : {yaw:.6f} rad ({math.degrees(yaw):.3f} deg)")
    print(f"  quaternion   : x={qx:.6f}, y={qy:.6f}, z={qz:.6f}, w={qw:.6f}")
    print()
    print("velocity")
    print(f"  linear       : x={vx:.6f} m/s, y={vy:.6f} m/s, z={vz:.6f} m/s")
    print(f"  angular_z    : {wz:.6f} rad/s ({math.degrees(wz):.3f} deg/s)")
    print("===================================")


async def main() -> None:
    request = {
        "op": "subscribe",
        "topic": topic,
        "type": msg_type,
    }

    connect_kwargs = {}
    if "proxy" in inspect.signature(websockets.connect).parameters:
        connect_kwargs["proxy"] = None

    async with websockets.connect(uri, ping_interval=None, **connect_kwargs) as websocket:
        await websocket.send(json.dumps(request))
        print("Sent WebSocket subscribe request:")
        print(json.dumps(request, ensure_ascii=False, indent=2))

        try:
            while True:
                raw = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(raw)
                if data.get("op") == "publish" and data.get("topic") == topic:
                    print_odom(data)
                    return
        except asyncio.TimeoutError:
            print(f"Timeout: did not receive a publish message from {topic} within 10 seconds.")
            sys.exit(1)


asyncio.run(main())
' "$URI" "$TOPIC" "$TYPE"
