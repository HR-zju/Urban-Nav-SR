from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse
import asyncio
import json
import os


def _extract_json_payload(tool_name: str, result) -> dict:
    raw_text = ""
    for content_item in result.content:
        if content_item.type != "text":
            continue
        raw_text = content_item.text
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            raise RuntimeError(f"{tool_name} returned non-JSON payload: {raw_text}")
    raise RuntimeError(f"{tool_name} returned no text payload")


async def run_demo(image_url: str | None, image_path: str | None, question: str, detail: str) -> None:
    server_params = StdioServerParameters(
        command="osm-mcp-server",
        args=[],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if image_path:
                result = await session.call_tool(
                    "analyze_local_image",
                    {
                        "image_path": image_path,
                        "question": question,
                        "detail": detail,
                    },
                )
            else:
                result = await session.call_tool(
                    "analyze_image",
                    {
                        "image_url": image_url,
                        "question": question,
                        "detail": detail,
                    },
                )

            payload = _extract_json_payload(
                "analyze_local_image" if image_path else "analyze_image",
                result,
            )

            print("=== Image Analysis Result ===")
            print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image understanding demo for osm-mcp-server")
    parser.add_argument("--image-url", help="Public image URL")
    parser.add_argument("--image-path", help="Local image path")
    parser.add_argument("--question", default="What is in this image?", help="Question for the model")
    parser.add_argument("--detail", default="auto", choices=["auto", "low", "high"], help="Vision detail level")

    args = parser.parse_args()
    if not args.image_url and not args.image_path:
        parser.error("one of --image-url or --image-path is required")
    asyncio.run(run_demo(args.image_url, args.image_path, args.question, args.detail))
