from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os


async def run_demo(image_url: str, question: str, detail: str) -> None:
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "analyze_image",
                {
                    "image_url": image_url,
                    "question": question,
                    "detail": detail,
                },
            )

            payload = {}
            for content_item in result.content:
                if content_item.type == "text":
                    payload = json.loads(content_item.text)
                    break

            print("=== Image Analysis Result ===")
            print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image understanding demo for osm-mcp-server")
    parser.add_argument("--image-url", required=True, help="Public image URL")
    parser.add_argument("--question", default="What is in this image?", help="Question for the model")
    parser.add_argument("--detail", default="auto", choices=["auto", "low", "high"], help="Vision detail level")

    args = parser.parse_args()
    asyncio.run(run_demo(args.image_url, args.question, args.detail))
