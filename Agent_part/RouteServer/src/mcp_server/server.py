"""Default MCP entrypoint; register domain tools without legacy visual prompts."""
from mcp_server.mcp_api.app import mcp
from mcp_server.mcp_api import (
    city_services,
    map_tools,
    neighborhood,
    recommendations,
    resources,
    route_sessions,
    route_visualization,
    vision_tools,
)

if __name__ == "__main__":
    mcp.run()
