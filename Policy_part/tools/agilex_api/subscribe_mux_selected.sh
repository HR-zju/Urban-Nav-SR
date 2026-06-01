#!/usr/bin/env bash
set -euo pipefail

URI="${1:-ws://192.168.1.102:9090}"
TOPIC="/mux_vel/selected"
TYPE="std_msgs/String"

/home/zjy/miniforge3/condabin/mamba run -n foundation_nav python -c '
import asyncio
import inspect
import json
import sys

import websockets

uri = sys.argv[1]
topic = sys.argv[2]
msg_type = sys.argv[3]


async def main() -> None:
    request = {"op": "subscribe", "topic": topic, "type": msg_type}
    connect_kwargs = {}
    if "proxy" in inspect.signature(websockets.connect).parameters:
        connect_kwargs["proxy"] = None

    async with websockets.connect(uri, ping_interval=None, **connect_kwargs) as websocket:
        await websocket.send(json.dumps(request))
        print("Sent WebSocket subscribe request:")
        print(json.dumps(request, ensure_ascii=False, indent=2))

        while True:
            raw = await asyncio.wait_for(websocket.recv(), timeout=10)
            data = json.loads(raw)
            if data.get("op") == "publish" and data.get("topic") == topic:
                print("Received mux selected source:")
                print(json.dumps(data, ensure_ascii=False, indent=2))
                selected = data.get("msg", {}).get("data")
                if selected is not None:
                    print(f"selected: {selected}")
                return


asyncio.run(main())
' "$URI" "$TOPIC" "$TYPE"
