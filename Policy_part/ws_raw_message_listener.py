import asyncio
from pathlib import Path

import websockets

HOST = "0.0.0.0"
PORT = 8765
SAVE_PATH = Path("/home/nuvo/SR/foundation-policy/Policy_part/topomaps/LLM_Policy/route_waypoints.json")


async def handler(websocket, path=None):
    print("client connected:", websocket.remote_address)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        async for message in websocket:
            print("\n收到原始消息:")
            print(message)

            # 每次收到新消息都覆盖写入目标文件
            SAVE_PATH.write_text(message, encoding="utf-8")
            print(f"原始消息已写入: {SAVE_PATH}")
    except websockets.exceptions.ConnectionClosed as e:
        print("client disconnected:", e.code, e.reason)


async def main():
    print(f"WebSocket server listening on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
