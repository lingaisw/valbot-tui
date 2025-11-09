# Tool extensions directory with dynamic tool loading
#
# Tools are loaded dynamically from their source locations via a registry.
#
# Usage in agents:
#   from agent_plugins.tool_extensions import docs_search_tools
#   result = docs_search_tools.search_documents(...)

import os
import sys
import importlib.util
from typing import Optional, Any

# Tool registry: maps tool name -> tool location
# Populated by PluginManager.load_tools()
_TOOL_REGISTRY = {}


def register_tool(tool_name: str, tool_path: str):
    """Register a tool's location in the registry."""
    _TOOL_REGISTRY[tool_name] = tool_path


def import_tool(tool_name: str) -> Optional[Any]:
    """
    Dynamically import a tool module by name from the registry.

    Args:
        tool_name: Name of the tool (e.g., 'docs_search_tools')

    Returns:
        The imported module object, or None if not found
    """
    if tool_name not in _TOOL_REGISTRY:
        return None

    tool_path = _TOOL_REGISTRY[tool_name]

    if not os.path.exists(tool_path):
        return None

    # Load module from file path
    spec = importlib.util.spec_from_file_location(tool_name, tool_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[f'agent_plugins.tool_extensions.{tool_name}'] = module
    spec.loader.exec_module(module)
    return module


def get_registered_tools():
    """Get list of all registered tool names."""
    return list(_TOOL_REGISTRY.keys())


def get_tool_path(tool_name: str) -> Optional[str]:
    """Get the file path for a registered tool."""
    return _TOOL_REGISTRY.get(tool_name)


def __getattr__(name: str) -> Any:
    """
    Enable attribute-style access to tools.
    Allows: from agent_plugins.tool_extensions import docs_search_tools
    """
    module = import_tool(name)
    if module is not None:
        return module

    raise AttributeError(f"module 'agent_plugins.tool_extensions' has no attribute '{name}'")

