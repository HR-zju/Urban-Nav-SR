"""Shared MCP application and provider lifecycle."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator
from mcp.server.fastmcp import FastMCP
from mcp_server.map_providers import MapProviderClient, create_map_client

@dataclass
class AppContext:
    osm_client: MapProviderClient
    provider: str


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage OSM client lifecycle"""
    # FastMCP enters this context once on startup and closes it on shutdown.
    osm_client, provider = create_map_client()
    try:
        await osm_client.connect()
        yield AppContext(osm_client=osm_client, provider=provider)
    finally:
        await osm_client.disconnect()


mcp = FastMCP(
    "Location-Based App MCP Server",
    dependencies=["aiohttp", "geojson", "shapely", "haversine"],
    lifespan=app_lifespan
)
