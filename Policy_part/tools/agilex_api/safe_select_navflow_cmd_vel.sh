#!/usr/bin/env bash
set -euo pipefail

URI="${1:-ws://192.168.1.102:9090}"
TOPIC="/navflow_cmd_vel"
TYPE="geometry_msgs/Twist"

/home/zjy/miniforge3/condabin/mamba run -n foundation_nav python -c '
import asyncio
import inspect
import json
import sys

import websockets

uri = sys.argv[1]
topic = sys.argv[2]
msg_type = sys.argv[3]
eps = 1e-9


async def call_service(websocket, service, request_id, args=None):
    request = {
        "op": "call_service",
        "service": service,
        "id": request_id,
        "args": args or {},
    }
    await websocket.send(json.dumps(request))
    while True:
        raw = await asyncio.wait_for(websocket.recv(), timeout=10)
        data = json.loads(raw)
        if data.get("op") == "service_response" and data.get("id") == request_id:
            return data


def get_float(msg, *keys):
    value = msg
    for key in keys:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


async def main() -> None:
    connect_kwargs = {}
    if "proxy" in inspect.signature(websockets.connect).parameters:
        connect_kwargs["proxy"] = None

    async with websockets.connect(uri, ping_interval=None, **connect_kwargs) as websocket:
        print("ROS graph before switch")
        for graph_topic in (topic, "/mux_vel/selected", "/base_cmd_vel"):
            topic_type = await call_service(
                websocket, "/rosapi/topic_type", f"type:{graph_topic}", {"topic": graph_topic}
            )
            publishers = await call_service(
                websocket, "/rosapi/publishers", f"pubs:{graph_topic}", {"topic": graph_topic}
            )
            subscribers = await call_service(
                websocket, "/rosapi/subscribers", f"subs:{graph_topic}", {"topic": graph_topic}
            )
            print(graph_topic)
            print("  type:", topic_type.get("values"))
            print("  publishers:", publishers.get("values"))
            print("  subscribers:", subscribers.get("values"))

        mux_list = await call_service(websocket, "/mux_vel/list", "mux_list")
        print("mux list:")
        print(json.dumps(mux_list, ensure_ascii=False, indent=2))

        await websocket.send(json.dumps({"op": "subscribe", "topic": topic, "type": msg_type}))
        print(f"waiting for one {topic} message...")
        try:
            while True:
                raw = await asyncio.wait_for(websocket.recv(), timeout=8)
                data = json.loads(raw)
                if data.get("op") == "publish" and data.get("topic") == topic:
                    msg = data.get("msg", {})
                    values = {
                        "linear.x": get_float(msg, "linear", "x"),
                        "linear.y": get_float(msg, "linear", "y"),
                        "linear.z": get_float(msg, "linear", "z"),
                        "angular.x": get_float(msg, "angular", "x"),
                        "angular.y": get_float(msg, "angular", "y"),
                        "angular.z": get_float(msg, "angular", "z"),
                    }
                    print(f"received {topic}:")
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                    print("summary:", values)
                    if not all(abs(value) <= eps for value in values.values()):
                        print("NOT switching mux because speed is non-zero.")
                        sys.exit(2)
                    break
        except asyncio.TimeoutError:
            print(f"Timeout: did not receive {topic}. NOT switching mux.")
            sys.exit(1)

        add_response = await call_service(websocket, "/mux_vel/add", "add_navflow", {"topic": topic})
        print("mux add response:")
        print(json.dumps(add_response, ensure_ascii=False, indent=2))

        select_response = await call_service(
            websocket, "/mux_vel/select", "select_navflow", {"topic": topic}
        )
        print("mux select response:")
        print(json.dumps(select_response, ensure_ascii=False, indent=2))


asyncio.run(main())
' "$URI" "$TOPIC" "$TYPE"
