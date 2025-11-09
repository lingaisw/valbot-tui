"""
Extension Manager - Handles agent and tool extensions for Valbot.

This module provides:
- AgentAdder: Add agents from local files or repositories
- ToolAdder: Add tools from local files or repositories
- ToolDependencyHandler: Manage tool dependencies and availability
- RegistryParser: Parse registry.json files for agents and tools
"""

from extension_manager.agent_adder import AgentAdder
from extension_manager.tool_adder import ToolAdder
from extension_manager.tool_dependency_handler import ToolDependencyHandler
from extension_manager.registry_parser import RegistryParser

__all__ = [
    "AgentAdder",
    "ToolAdder",
    "ToolDependencyHandler",
    "RegistryParser",
]
