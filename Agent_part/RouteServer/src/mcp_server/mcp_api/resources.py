"""MCP endpoints for resources."""
from __future__ import annotations

from typing import Tuple
import json
import aiohttp
from mcp_server.mcp_api.app import mcp

@mcp.resource("location://place/{query}")
async def get_place_resource(query: str) -> str:
    """
    Get information about a place by name.

    Args:
        query: Place name or address to look up

    Returns:
        JSON string with place information
    """
    async with aiohttp.ClientSession() as session:
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        async with session.get(
            nominatim_url,
            params={
                "q": query,
                "format": "json",
                "limit": 1
            },
            headers={"User-Agent": "LocationApp-MCP-Server/1.0"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return json.dumps(data)
            else:
                raise Exception(f"Failed to get place info for {query}: {response.status}")


@mcp.resource("location://map/{style}/{z}/{x}/{y}")
async def get_map_style(style: str, z: int, x: int, y: int) -> Tuple[bytes, str]:
    """
    Get a styled map tile at the specified coordinates.

    Args:
        style: Map style (standard, cycle, transport, etc.)
        z: Zoom level
        x: X coordinate
        y: Y coordinate

    Returns:
        Tuple of (tile image bytes, mime type)
    """
    # Map styles to their respective tile servers
    tile_servers = {
        "standard": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "cycle": "https://tile.thunderforest.com/cycle/{z}/{x}/{y}.png",
        "transport": "https://tile.thunderforest.com/transport/{z}/{x}/{y}.png",
        "landscape": "https://tile.thunderforest.com/landscape/{z}/{x}/{y}.png",
        "outdoor": "https://tile.thunderforest.com/outdoors/{z}/{x}/{y}.png"
    }


    if style not in tile_servers:
        style = "standard"

    tile_url = tile_servers[style].replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))

    async with aiohttp.ClientSession() as session:
        async with session.get(tile_url) as response:
            if response.status == 200:
                tile_data = await response.read()
                return tile_data, "image/png"
            else:
                raise Exception(f"Failed to get {style} tile at {z}/{x}/{y}: {response.status}")
