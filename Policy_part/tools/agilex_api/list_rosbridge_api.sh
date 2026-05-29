#!/usr/bin/env bash
set -euo pipefail

URI="${1:-ws://192.168.1.102:9090}"

/home/zjy/miniforge3/condabin/mamba run -n foundation_nav python -c '
import asyncio
import inspect
import json
import sys

import websockets

uri = sys.argv[1]


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


async def main() -> None:
    connect_kwargs = {}
    if "proxy" in inspect.signature(websockets.connect).parameters:
        connect_kwargs["proxy"] = None

    async with websockets.connect(uri, ping_interval=None, **connect_kwargs) as websocket:
        topics_response = await call_service(websocket, "/rosapi/topics", "topics")
        services_response = await call_service(websocket, "/rosapi/services", "services")

        topics = topics_response.get("values", {}).get("topics", [])
        types = topics_response.get("values", {}).get("types", [])
        services = services_response.get("values", {}).get("services", [])

        print(f"URI: {uri}")
        print(f"Topics: {len(topics)}")
        for topic, topic_type in zip(topics, types):
            print(f"{topic}  [{topic_type}]")

        print()
        print(f"Services: {len(services)}")
        for service in services:
            print(service)


asyncio.run(main())
' "$URI"
