"""Student-visible tool specifications and executor capability registry."""

from mcp_server.tools.registry import DEFAULT_TOOL_REGISTRY, ToolOperationSpec, ToolSpecRegistry
from mcp_server.tools.specs import (
    IMAGE_TOOL_NAME,
    MAP_TOOL_NAME,
    USER_INTERACTION_TOOL_NAME,
    normalize_map_tool_arguments,
)

__all__ = [
    "DEFAULT_TOOL_REGISTRY",
    "IMAGE_TOOL_NAME",
    "MAP_TOOL_NAME",
    "ToolOperationSpec",
    "ToolSpecRegistry",
    "USER_INTERACTION_TOOL_NAME",
    "normalize_map_tool_arguments",
]
