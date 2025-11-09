"""
Registry Parser - Common utility for parsing registry.json files.

Handles parsing of both agent and tool registries with support for:
- List format: [{"name": "...", "path": "..."}]
- Dict/map format: {"key": {"path": "..."}}
- Section format: {"agents": [...], "tools": [...]}
"""

from typing import Any, Dict, List, Optional


class RegistryParser:
    """Parse registry.json files for agents and tools."""

    @staticmethod
    def normalize_entry(obj: Any, fallback_name: Optional[str] = None, capture_required_tools: bool = False) -> Optional[Dict[str, Any]]:
        """
        Normalize a registry entry (agent or tool) into a standard dict.

        Args:
            obj: The entry object to normalize
            fallback_name: Name to use if 'name' field is missing
            capture_required_tools: Whether to capture 'required_tools' field (for agents)

        Returns:
            Normalized dict or None if invalid
        """
        if not isinstance(obj, dict):
            return None

        name = obj.get("name") or fallback_name
        path = obj.get("path") or obj.get("location")
        if not name or not path:
            return None

        result = {
            "name": name,
            "description": obj.get("description", ""),
            "path": path,
        }

        if obj.get("ref"):
            result["ref"] = obj["ref"]

        # Capture required_tools for agents
        if capture_required_tools and "required_tools" in obj:
            if isinstance(obj["required_tools"], list):
                result["required_tools"] = obj["required_tools"]

        return result

    @staticmethod
    def parse_section(data: Any, section_name: str, capture_required_tools: bool = False) -> List[Dict[str, Any]]:
        """
        Parse a section of registry.json (agents or tools).

        Supports multiple formats:
        - Direct list: [{"name": "x", "path": "y"}]
        - With section: {"agents": [{"name": "x", "path": "y"}]}
        - Map format: {"key": {"path": "y"}}

        Args:
            data: The registry data
            section_name: Name of section to parse ("agents" or "tools")
            capture_required_tools: Whether to capture required_tools field

        Returns:
            List of normalized entries
        """
        entries = []

        if isinstance(data, list):
            # Direct list format
            for item in data:
                entry = RegistryParser.normalize_entry(item, capture_required_tools=capture_required_tools)
                if entry:
                    entries.append(entry)

        elif isinstance(data, dict):
            # Check for explicit section
            if section_name in data and isinstance(data[section_name], list):
                for item in data[section_name]:
                    entry = RegistryParser.normalize_entry(item, capture_required_tools=capture_required_tools)
                    if entry:
                        entries.append(entry)
            else:
                # Map format: key -> entry dict
                for key, value in data.items():
                    entry = RegistryParser.normalize_entry(value, fallback_name=key, capture_required_tools=capture_required_tools)
                    if entry:
                        entries.append(entry)

        return entries

    @staticmethod
    def parse_agents(data: Any) -> List[Dict[str, Any]]:
        """Parse agents section from registry, including required_tools."""
        return RegistryParser.parse_section(data, "agents", capture_required_tools=True)

    @staticmethod
    def parse_tools(data: Any) -> List[Dict[str, Any]]:
        """Parse tools section from registry."""
        return RegistryParser.parse_section(data, "tools", capture_required_tools=False)
